import torch
import torch.optim as optim
import random
import numpy as np
import json
import os
import argparse
from typing import List, Dict, Any
from game import GameV2
from transformer import HeartsTransformer
from data_structure import Card, PassDirection

# Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPISODES = 1000
BATCH_SIZE = 1 # Update every game for now
HIDDEN_DIM = 128

class AIPlayer:
    def __init__(self, model: HeartsTransformer, device='cpu'):
        self.model = model
        self.device = device
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        self.passed_cards = []
        self.received_cards = []
        self.hand_before_pass = set()

    def reset(self):
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        self.passed_cards = []
        self.received_cards = []
        self.hand_before_pass = set()

    def pass_policy(self, player, info) -> List[Card]:
        self.hand_before_pass = set(player.hand)
        selected_cards = []
        
        # We need to select 3 cards. We do this autoregressively.
        # The model sees the hand, picks 1 card, then sees the hand minus that card, picks next, etc.
        current_hand = list(player.hand)
        
        for _ in range(3):
            # 1. Update Model State with current temporary hand
            # We pass current_hand as legal_actions because we can pass any card we hold
            temp_info = info.copy()
            temp_info['hand'] = current_hand
            temp_info['is_passing'] = True # Explicitly mark as passing phase
            
            self.model.update_from_info(
                temp_info, 
                legal_actions=current_hand,
                passed_cards=[], # Not passed yet
                received_cards=[] # Not received yet
            )
            
            # 2. Forward pass
            logits, value = self.model(x=None)
            logits = logits.squeeze() # (52)
            
            # 3. Mask illegal moves (cards not in current_hand)
            mask = torch.full((52,), float('-inf'), device=self.device)
            legal_indices = [c.to_id() for c in current_hand]
            mask[legal_indices] = 0
            masked_logits = logits + mask
            
            # 4. Sample action
            probs = torch.softmax(masked_logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            
            # 5. Save log prob and value for training
            self.saved_log_probs.append(dist.log_prob(action_idx))
            self.saved_values.append(value)
            
            # 6. Decode card
            suit_val = action_idx.item() // 13
            rank_val = (action_idx.item() % 13) + 1
            
            # Find the actual card object
            selected_card = next(c for c in current_hand if c.suit.value == suit_val and c.rank == rank_val)
            
            selected_cards.append(selected_card)
            current_hand.remove(selected_card)

        self.passed_cards = selected_cards
        return selected_cards

    def play_policy(self, player, info, legal_actions, order):
        # Deduce received cards if not already done and we passed cards
        if not self.received_cards and self.passed_cards:
             remaining = self.hand_before_pass - set(self.passed_cards)
             current_hand_set = set(player.hand)
             self.received_cards = list(current_hand_set - remaining)

        # Update Model State
        info['is_passing'] = False # Explicitly mark as playing phase
        self.model.update_from_info(
            info, 
            legal_actions, 
            passed_cards=self.passed_cards, 
            received_cards=self.received_cards
        )
        
        # Forward pass
        # Ensure we pass the device to assemble_input implicitly by not passing x
        # The model's assemble_input now checks self.parameters().device
        logits, value = self.model(x=None) # (1, 1, 52) -> (1, 52)
        logits = logits.squeeze(0).squeeze(0) # (52)
        
        # Mask illegal moves
        # Create a mask of -inf
        mask = torch.full((52,), float('-inf'), device=self.device)
        
        # Set legal actions to 0 (or keep logits)
        legal_indices = [c.to_id() for c in legal_actions]
        mask[legal_indices] = 0
        
        # Add mask to logits
        masked_logits = logits + mask
        
        # Softmax to get probabilities
        probs = torch.softmax(masked_logits, dim=0)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        
        # Save log prob and value
        self.saved_log_probs.append(dist.log_prob(action_idx))
        self.saved_values.append(value)
        
        # Convert index back to Card
        # 0..51 -> Suit * 13 + (Rank-1)
        suit_val = action_idx.item() // 13
        rank_val = (action_idx.item() % 13) + 1
        
        # Find the actual card object in legal_actions to ensure object identity if needed
        # (Though Card is a dataclass so equality works by value)
        selected_card = Card(suit=list(from_suit_int(suit_val))[0], rank=rank_val)
        
        return selected_card

def from_suit_int(val):
    # Helper to get Suit enum from int
    from data_structure import Suit
    for s in Suit:
        if s.value == val:
            yield s

def random_policy(player, info, legal_actions, order):
    return random.choice(legal_actions)

def random_pass_policy(player, info):
    return random.sample(player.hand, 3)

def get_card_strength(card):
    # Ace (1) is highest (14), then King (13), ..., 2 is lowest
    if card.rank == 1:
        return 14
    return card.rank

def min_policy(player, info, legal_actions, order):
    # Play the smallest card (weakest)
    return min(legal_actions, key=get_card_strength)

def max_policy(player, info, legal_actions, order):
    # Play the largest card (strongest)
    return max(legal_actions, key=get_card_strength)

def train():
    model_path = 'hearts_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = HeartsTransformer(d_model=HIDDEN_DIM).to(device)
    
    # Interactive Model Loading
    if os.path.exists(model_path):
        user_input = input(f"Found existing model '{model_path}'. Load it? (y/n): ").strip().lower()
        if user_input == 'y':
            print(f"Loading model from {model_path}...")
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("Starting training from scratch.")
    else:
        print("No existing model found. Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    game = GameV2()
    ai_player = AIPlayer(model, device)
    
    # Create opponent AI instances for Self-Play
    # They share the same model weights but maintain their own game state
    opponent_agents = [AIPlayer(model, device) for _ in range(3)]
    
    # We will train Player 0
    # Players 1, 2, 3 will evolve based on curriculum
    
    running_reward = 0
    running_score = 0
    
    # Logging Data
    log_data = {
        'episodes': [],
        'scores': [],
        'avg_scores': [],
        'policy_losses': [],
        'value_losses': [],
        'difficulty': []
    }
    
    for i_episode in range(EPISODES):
        ai_player.reset()
        for opp in opponent_agents:
            opp.reset()
        
        # Dynamic Difficulty Curriculum
        progress = i_episode / EPISODES
        difficulty = "Random"
        
        # Default: Random
        p1_policy = random_policy
        p2_policy = random_policy
        p3_policy = random_policy
        
        p1_pass = random_pass_policy
        p2_pass = random_pass_policy
        p3_pass = random_pass_policy
        
        if progress < 0.2:
            # Stage 1: Beginner (Random)
            difficulty = "Beginner (Random)"
            
        elif progress < 0.5:
            # Stage 2: Intermediate (Heuristics)
            # Opponents play conservatively (Min) or aggressively (Max)
            difficulty = "Intermediate (Heuristics)"
            p1_policy = min_policy
            p2_policy = min_policy
            p3_policy = max_policy # One chaotic player
            
        elif progress < 0.8:
            # Stage 3: Advanced (Mixed AI + Heuristics)
            # One opponent is a copy of the current AI (Self-Play)
            difficulty = "Advanced (1 AI + Heuristics)"
            p1_policy = opponent_agents[0].play_policy
            p1_pass = opponent_agents[0].pass_policy
            p2_policy = min_policy
            p3_policy = min_policy
            
        else:
            # Stage 4: Expert (Full Self-Play)
            # All opponents are current AI
            difficulty = "Expert (Self-Play)"
            p1_policy = opponent_agents[0].play_policy
            p1_pass = opponent_agents[0].pass_policy
            p2_policy = opponent_agents[1].play_policy
            p2_pass = opponent_agents[1].pass_policy
            p3_policy = opponent_agents[2].play_policy
            p3_pass = opponent_agents[2].pass_policy

        current_policies = [ai_player.play_policy, p1_policy, p2_policy, p3_policy]
        pass_policies = [ai_player.pass_policy, p1_pass, p2_pass, p3_pass]
        
        # Random pass direction
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        # Run Game
        scores, raw_scores, events, trick_history = game.run_game_training(current_policies, pass_policies, pass_direction=pass_dir)
        
        # Calculate Reward for Player 0
        player_final_score = scores[0]
        player_raw_score = raw_scores[0]
        shot_the_moon = (player_raw_score == 26)
        
        rewards = []
        
        # 1. Passing Rewards (if applicable)
        if pass_dir != PassDirection.KEEP:
             # Assign final game outcome proxy to passing
             # If STM, big bonus. If not, negative score.
             pass_reward = 50.0 if shot_the_moon else -float(player_final_score)
             for _ in range(3):
                 rewards.append(pass_reward)

        # 2. Play Rewards (Trick by Trick)
        for trick in trick_history:
            if shot_the_moon:
                r = 4.0 # Reward for every step leading to STM
            else:
                points_taken = trick.score if trick.winner == 0 else 0
                
                if points_taken > 0:
                    # We took points -> Penalty
                    r = -float(points_taken)
                else:
                    # We didn't take points -> Small Reward
                    r = 0.2 
                    
                    # Extra bonus if we specifically dodged points (someone else took them)
                    if trick.winner != 0 and trick.score > 0:
                        r += 1.0
            rewards.append(r)
            
        # Update Policy
        optimizer.zero_grad()
        policy_loss = []
        value_loss = []
        
        # Ensure alignment
        if len(rewards) != len(ai_player.saved_log_probs):
            min_len = min(len(rewards), len(ai_player.saved_log_probs))
            rewards = rewards[:min_len]
            ai_player.saved_log_probs = ai_player.saved_log_probs[:min_len]
            ai_player.saved_values = ai_player.saved_values[:min_len]

        # Calculate Returns (Cumulative Discounted Reward)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(device)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate Advantage and Losses
        for log_prob, value, R in zip(ai_player.saved_log_probs, ai_player.saved_values, returns):
            advantage = R - value.item()
            
            # Policy Loss: -log_prob * advantage
            policy_loss.append(-log_prob * advantage)
            
            # Value Loss: MSE(value, R)
            # We use smooth_l1_loss or mse_loss
            value_loss.append(torch.nn.functional.mse_loss(value.squeeze(), torch.tensor(R).to(device)))
            
        current_p_loss = 0.0
        current_v_loss = 0.0
        if policy_loss:
            p_loss = torch.stack(policy_loss).sum()
            v_loss = torch.stack(value_loss).sum()
            
            # Total Loss = Policy Loss + 0.5 * Value Loss
            loss = p_loss + 0.5 * v_loss
            
            loss.backward()
            optimizer.step()
            current_p_loss = p_loss.item()
            current_v_loss = v_loss.item()
            
        total_reward = sum(rewards)
        if i_episode == 0:
            running_score = player_final_score
            running_reward = total_reward
        else:
            running_score = 0.05 * player_final_score + 0.95 * running_score
            running_reward = 0.05 * total_reward + 0.95 * running_reward
        
        # Log Data
        log_data['episodes'].append(i_episode)
        log_data['scores'].append(player_final_score)
        log_data['avg_scores'].append(running_score)
        log_data['policy_losses'].append(current_p_loss)
        log_data['value_losses'].append(current_v_loss)
        log_data['difficulty'].append(difficulty)
        
        # Write to file every 10 episodes
        if i_episode % 10 == 0:
            try:
                with open("training_log.json", "w") as f:
                    json.dump(log_data, f)
            except Exception as e:
                print(f"Error writing log: {e}")
        
        if i_episode % 50 == 0:
            print(f"Episode {i_episode}\tAvg Score: {running_score:.2f}\tAvg Reward: {running_reward:.2f}\tPass: {pass_dir.name}\tDiff: {difficulty}")
            torch.save(model.state_dict(), model_path)

    print("Training Complete.")
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train()
