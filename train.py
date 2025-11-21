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
import copy
from collections import deque

# Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPISODES = 5000
BATCH_SIZE = 32 # Update every 32 games
HIDDEN_DIM = 128
PPO_EPOCHS = 4
CLIP_EPS = 0.2

class OpponentPool:
    def __init__(self, max_size=50):
        self.pool = deque(maxlen=max_size)
    
    def add(self, model_state_dict):
        # Store a deep copy of the state dict on CPU to save VRAM
        state_copy = {k: v.cpu().clone() for k, v in model_state_dict.items()}
        self.pool.append(state_copy)
    
    def sample(self):
        if not self.pool:
            return None
        return random.choice(self.pool)

class AIPlayer:
    def __init__(self, model: HeartsTransformer, device='cpu'):
        self.model = model
        self.device = device
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_states = [] # Store raw state snapshots
        self.saved_actions = [] # Store action indices
        self.saved_masks = [] # Store action masks
        self.rewards = []
        self.passed_cards = []
        self.received_cards = []
        self.hand_before_pass = set()

    def reset(self):
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_states = []
        self.saved_actions = []
        self.saved_masks = []
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
            
            # Snapshot state for PPO
            state_snapshot = self.model.get_raw_state()
            self.saved_states.append(state_snapshot)

            # 2. Forward pass
            logits, value = self.model(x=None)
            logits = logits.squeeze() # (52)
            
            # 3. Mask illegal moves (cards not in current_hand)
            mask = torch.full((52,), float('-inf'), device=self.device)
            legal_indices = [c.to_id() for c in current_hand]
            mask[legal_indices] = 0
            self.saved_masks.append(mask) # Save mask
            
            masked_logits = logits + mask
            
            # 4. Sample action
            probs = torch.softmax(masked_logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            
            # 5. Save log prob and value for training
            self.saved_log_probs.append(dist.log_prob(action_idx))
            self.saved_values.append(value)
            self.saved_actions.append(action_idx)
            
            # 6. Decode card
            suit_val = action_idx.item() // 13
            rank_val = (action_idx.item() % 13) + 1
            
            # Find the actual card object
            selected_card = next(c for c in current_hand if c.suit.value == suit_val and c.rank == rank_val)
            
            selected_cards.append(selected_card)
            current_hand.remove(selected_card)

        self.passed_cards = selected_cards
        return selected_cards

    def play_policy(self, player, info, legal_actions, order, override_policy=None):
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
        
        # Snapshot state for PPO
        state_snapshot = self.model.get_raw_state()
        self.saved_states.append(state_snapshot)
        
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
        self.saved_masks.append(mask) # Save mask
        
        # Add mask to logits
        masked_logits = logits + mask
        
        # Softmax to get probabilities
        probs = torch.softmax(masked_logits, dim=0)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        
        if override_policy:
            # Use the override policy to select a card
            selected_card = override_policy(player, info, legal_actions, order)
            # Find the index of this card to store it as the "action taken"
            # This is important if we want to do Behavior Cloning later, 
            # but for Value Pretraining we just need the trajectory to be valid.
            # We still store it so the PPO loop structure doesn't break (though we won't use it for policy update in pretrain)
            action_idx = torch.tensor(selected_card.to_id(), device=self.device)
        else:
            action_idx = dist.sample()
            # Convert index back to Card
            # 0..51 -> Suit * 13 + (Rank-1)
            suit_val = action_idx.item() // 13
            rank_val = (action_idx.item() % 13) + 1
            selected_card = Card(suit=list(from_suit_int(suit_val))[0], rank=rank_val)
        
        # Save log prob and value
        self.saved_log_probs.append(dist.log_prob(action_idx))
        self.saved_values.append(value)
        self.saved_actions.append(action_idx)
        
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

def pretrain_value_net(model, optimizer, device, episodes=200):
    print(f"Starting Value Network Pretraining for {episodes} episodes...")
    game = GameV2()
    ai_player = AIPlayer(model, device)
    
    # We will use a mix of Min and Max policies to generate diverse "competent" gameplay data
    # The AIPlayer will run in "Shadow Mode": predicting values but playing heuristic moves.
    
    running_loss = 0.0
    
    for i_episode in range(episodes):
        ai_player.reset()
        
        # Randomly choose a heuristic for the "Main Player" (Shadow Mode)
        # This ensures the ValueNet sees states resulting from different styles of play
        main_heuristic = random.choice([min_policy, max_policy, random_policy])
        
        # Opponents
        p1_policy = random.choice([min_policy, max_policy, random_policy])
        p2_policy = random.choice([min_policy, max_policy, random_policy])
        p3_policy = random.choice([min_policy, max_policy, random_policy])
        
        # Pass policies (random for now)
        pass_policies = [random_pass_policy] * 4
        
        # Wrapper to inject heuristic into AIPlayer
        def shadow_policy(player, info, legal_actions, order):
            return ai_player.play_policy(player, info, legal_actions, order, override_policy=main_heuristic)
            
        current_policies = [shadow_policy, p1_policy, p2_policy, p3_policy]
        
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        # Run Game
        scores, raw_scores, events, trick_history = game.run_game_training(current_policies, pass_policies, pass_direction=pass_dir)
        
        # Calculate Rewards (Same as training)
        player_final_score = scores[0]
        player_raw_score = raw_scores[0]
        shot_the_moon = (player_raw_score == 26)
        
        rewards = []
        if pass_dir != PassDirection.KEEP:
             # Scale rewards by 0.01
             pass_reward = 0.5 if shot_the_moon else -float(player_final_score) / 100.0
             for _ in range(3):
                 rewards.append(pass_reward)

        for trick in trick_history:
            if shot_the_moon:
                r = 0.04 
            else:
                points_taken = trick.score if trick.winner == 0 else 0
                if points_taken > 0:
                    r = -float(points_taken) / 100.0
                else:
                    r = 0.002 
                    if trick.winner != 0 and trick.score > 0:
                        r += 0.01
            rewards.append(r)
            
        # Update Value Net Only
        optimizer.zero_grad()
        
        # Alignment
        if len(rewards) != len(ai_player.saved_values):
            min_len = min(len(rewards), len(ai_player.saved_values))
            rewards = rewards[:min_len]
            ai_player.saved_values = ai_player.saved_values[:min_len]

        # Calculate Returns
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(device)
        # NOTE: We do NOT normalize returns for Value Pretraining to learn absolute scale
        
        values = torch.stack(ai_player.saved_values).squeeze()
        
        # MSE Loss
        loss = torch.nn.functional.mse_loss(values, returns)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i_episode + 1) % 50 == 0:
            print(f"Pretrain Episode {i_episode+1}/{episodes}\tAvg Value Loss: {running_loss/50:.4f}")
            running_loss = 0.0
            
    print("Value Network Pretraining Complete.")

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
    
    # Pretrain Value Network
    pretrain_value_net(model, optimizer, device, episodes=200)
    
    game = GameV2()
    ai_player = AIPlayer(model, device)
    
    # Opponent Pool Setup
    pool = OpponentPool()
    
    # Create separate opponent models for stable self-play
    opponent_models = [HeartsTransformer(d_model=HIDDEN_DIM).to(device) for _ in range(3)]
    opponent_agents = [AIPlayer(opp_model, device) for opp_model in opponent_models]
    
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
    
    # Batch buffers
    batch_log_probs = []
    batch_values = []
    batch_states = []
    batch_actions = []
    batch_masks = []
    batch_returns = []

    for i_episode in range(EPISODES):
        ai_player.reset()
        for opp in opponent_agents:
            opp.reset()
            
        # Update Pool periodically
        if i_episode % 50 == 0:
            pool.add(model.state_dict())
        
        # Dynamic Difficulty Curriculum
        progress = i_episode / EPISODES
        difficulty = "Random"
        
        # Default Policies
        p_policies = [random_policy] * 3
        pass_policies = [random_pass_policy] * 3
        
        if progress < 0.3:
            # Stage 1: Basic Heuristics (Deterministic)
            # Start with Min/Max as they are more "logical" and easier to learn basic mechanics from.
            difficulty = "Beginner (Heuristics)"
            p_policies = [min_policy, min_policy, max_policy] # 2 Conservative, 1 Aggressive
            
        elif progress < 0.6:
            # Stage 2: Introducing Chaos (Random)
            # Now that we know the rules, handle unpredictable random players which can be harder.
            difficulty = "Intermediate (Random + Heuristics)"
            p_policies = [random_policy, min_policy, max_policy]
            
        else:
            # Stage 3: Advanced (Pool + Self-Play)
            difficulty = "Advanced (Pool + Heuristics)"
            # Mix of Pool and Heuristics
            for i in range(3):
                if pool.pool and random.random() < 0.7: # 70% chance to play against past self
                    past_state = pool.sample()
                    opponent_models[i].load_state_dict(past_state)
                    p_policies[i] = opponent_agents[i].play_policy
                    pass_policies[i] = opponent_agents[i].pass_policy
                else:
                    # Fallback to heuristics/random mix
                    p_policies[i] = random.choice([min_policy, max_policy, random_policy])
                    pass_policies[i] = random_pass_policy

        current_policies = [ai_player.play_policy] + p_policies
        current_pass_policies = [ai_player.pass_policy] + pass_policies
        
        # Random pass direction
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        # Run Game
        scores, raw_scores, events, trick_history = game.run_game_training(current_policies, current_pass_policies, pass_direction=pass_dir)
        
        # Calculate Reward for Player 0
        player_final_score = scores[0]
        player_raw_score = raw_scores[0]
        shot_the_moon = (player_raw_score == 26)
        
        rewards = []
        
        # 1. Passing Rewards (if applicable)
        if pass_dir != PassDirection.KEEP:
             # Assign final game outcome proxy to passing
             # If STM, big bonus. If not, negative score.
             # Scale rewards by 0.01 to keep values in reasonable range (-0.26 to +0.26 approx)
             pass_reward = 0.5 if shot_the_moon else -float(player_final_score) / 100.0
             for _ in range(3):
                 rewards.append(pass_reward)

        # 2. Play Rewards (Trick by Trick)
        for trick in trick_history:
            if shot_the_moon:
                r = 0.04 # Reward for every step leading to STM
            else:
                points_taken = trick.score if trick.winner == 0 else 0
                
                if points_taken > 0:
                    # We took points -> Penalty
                    r = -float(points_taken) / 100.0
                else:
                    # We didn't take points -> Small Reward
                    r = 0.002 
                    
                    # Extra bonus if we specifically dodged points (someone else took them)
                    if trick.winner != 0 and trick.score > 0:
                        r += 0.01
            rewards.append(r)
            
        # Update Policy
        optimizer.zero_grad()
        
        # Ensure alignment
        if len(rewards) != len(ai_player.saved_log_probs):
            min_len = min(len(rewards), len(ai_player.saved_log_probs))
            rewards = rewards[:min_len]
            ai_player.saved_log_probs = ai_player.saved_log_probs[:min_len]
            ai_player.saved_values = ai_player.saved_values[:min_len]
            ai_player.saved_states = ai_player.saved_states[:min_len]
            ai_player.saved_actions = ai_player.saved_actions[:min_len]
            ai_player.saved_masks = ai_player.saved_masks[:min_len]

        # Calculate Returns (Cumulative Discounted Reward)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
            
        # Accumulate to batch
        batch_log_probs.extend(ai_player.saved_log_probs)
        batch_values.extend(ai_player.saved_values)
        batch_states.extend(ai_player.saved_states)
        batch_actions.extend(ai_player.saved_actions)
        batch_masks.extend(ai_player.saved_masks)
        batch_returns.extend(returns)

        current_p_loss = 0.0
        current_v_loss = 0.0

        # --- PPO Update (Batch) ---
        if (i_episode + 1) % BATCH_SIZE == 0:
            
            # Convert batch lists to tensors
            b_returns = torch.tensor(batch_returns).to(device)
            
            # Normalize returns for stability
            if len(b_returns) > 1:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-9)

            # Detach old log probs and values for comparison
            old_log_probs = torch.stack(batch_log_probs).detach()
            old_values = torch.stack(batch_values).detach().squeeze()
            
            # Calculate Advantages (using old values)
            advantages = b_returns - old_values
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                
            # PPO Epochs
            for _ in range(PPO_EPOCHS):
                # Batch processing
                logits, new_values = model(batch_raw_states=batch_states)
                logits = logits.squeeze() # (Batch, 52)
                new_values = new_values.squeeze() # (Batch)
                
                # Masks
                masks = torch.stack(batch_masks) # (Batch, 52)
                masked_logits = logits + masks
                
                probs = torch.softmax(masked_logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                
                # Actions
                b_actions = torch.stack(batch_actions)
                new_log_probs = dist.log_prob(b_actions)
                entropies = dist.entropy()
                
                # Ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Surrogate Loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = torch.nn.functional.mse_loss(new_values, b_returns)
                
                # Entropy Bonus
                entropy_loss = -0.01 * entropies.mean()
                
                loss = policy_loss + 0.5 * value_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                current_p_loss = policy_loss.item()
                current_v_loss = value_loss.item()
            
            # Clear batch buffers
            batch_log_probs = []
            batch_values = []
            batch_states = []
            batch_actions = []
            batch_masks = []
            batch_returns = []
            
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
