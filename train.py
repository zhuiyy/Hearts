import torch
import torch.optim as optim
import random
import numpy
import json
import os
from typing import List
from game import GameV2
from transformer import HeartsTransformer
from data_structure import Card, PassDirection
from collections import deque
import strategies
from strategies import ExpertPolicy

import gpu_selector

# Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPISODES = 5000
BATCH_SIZE = 32 # Update every 32 games
HIDDEN_DIM = 128
PPO_EPOCHS = 4
CLIP_EPS = 0.2
WEIGHT_DECAY = 1e-5 # L2 Regularization
DROPOUT = 0.1 # Dropout probability
MAX_GRAD_NORM = 0.5 # Gradient Clipping

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

def pretrain_supervised(model, optimizer, device, episodes=1000):
    print(f"Starting Supervised Pretraining (Policy + Value) for {episodes} episodes...")
    game = GameV2()
    ai_player = AIPlayer(model, device)
    
    # Adaptive Learning Rate Scheduler
    # Reduce LR if loss stops decreasing for 'patience' number of updates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    running_loss = 0.0
    running_p_loss = 0.0
    running_v_loss = 0.0
    
    # Batch buffers
    batch_states = []
    batch_actions = []
    batch_masks = []
    batch_returns = []
    
    for i_episode in range(episodes):
        ai_player.reset()
        
        # Use Expert Policy as the teacher
        main_heuristic = ExpertPolicy.play_policy
        
        # Opponents can be mixed
        p1_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        p2_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        p3_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        
        # Pass policies
        # Use Expert Pass Policy for the main player to learn passing too!
        # But our AIPlayer.pass_policy logic is autoregressive and complex to override directly with a simple function 
        # that returns 3 cards at once.
        # For now, let's focus on playing policy cloning. 
        # If we want to clone passing, we need to adapt the shadow_pass_policy to return cards one by one or 
        # force the AIPlayer to record the expert's choice.
        
        # Let's stick to random pass for opponents, but for the main player we want to learn expert passing?
        # The current AIPlayer structure records actions one by one. 
        # ExpertPolicy.pass_policy returns 3 cards.
        # We can make a wrapper to feed them one by one?
        # Or just ignore passing pretraining for now and focus on play.
        # Let's use random pass for now to keep it simple, or implement a shadow pass later.
        pass_policies = [strategies.random_pass_policy] * 4
        
        # Wrapper to inject heuristic into AIPlayer
        def shadow_policy(player, info, legal_actions, order):
            return ai_player.play_policy(player, info, legal_actions, order, override_policy=main_heuristic)
            
        current_policies = [shadow_policy, p1_policy, p2_policy, p3_policy]
        
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        # Run Game
        scores, raw_scores, events, trick_history = game.run_game_training(current_policies, pass_policies, pass_direction=pass_dir)
        
        # Calculate Rewards
        player_final_score = scores[0]
        player_raw_score = raw_scores[0]
        shot_the_moon = (player_raw_score == 26)
        
        rewards = []
        if pass_dir != PassDirection.KEEP:
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
            
        # Alignment
        if len(rewards) != len(ai_player.saved_values):
            min_len = min(len(rewards), len(ai_player.saved_values))
            rewards = rewards[:min_len]
            ai_player.saved_values = ai_player.saved_values[:min_len]
            ai_player.saved_states = ai_player.saved_states[:min_len]
            ai_player.saved_actions = ai_player.saved_actions[:min_len]
            ai_player.saved_masks = ai_player.saved_masks[:min_len]

        # Calculate Returns
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
            
        # Accumulate to batch
        batch_states.extend(ai_player.saved_states)
        batch_actions.extend(ai_player.saved_actions)
        batch_masks.extend(ai_player.saved_masks)
        batch_returns.extend(returns)
        
        # Update every BATCH_SIZE episodes
        if (i_episode + 1) % BATCH_SIZE == 0:
            b_returns = torch.tensor(batch_returns).to(device)
            b_actions = torch.stack(batch_actions)
            
            # Forward pass on batch
            logits, values = model(batch_raw_states=batch_states)
            logits = logits.squeeze() # (Batch, 52)
            values = values.squeeze() # (Batch)
            
            # 1. Value Loss (MSE)
            value_loss = torch.nn.functional.mse_loss(values, b_returns)
            
            # 2. Policy Loss (Cross Entropy / Behavior Cloning)
            # We want to maximize log_prob of the action taken by the heuristic
            # CrossEntropyLoss expects logits (unnormalized) and target indices
            
            # Apply mask to logits before CrossEntropy?
            # CrossEntropyLoss doesn't take a mask, but we can set illegal logits to -inf
            masks = torch.stack(batch_masks)
            masked_logits = logits + masks
            
            policy_loss = torch.nn.functional.cross_entropy(masked_logits, b_actions)
            
            loss = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()
            
            # Step Scheduler with current loss
            scheduler.step(loss)
            
            running_loss += loss.item()
            running_p_loss += policy_loss.item()
            running_v_loss += value_loss.item()
            
            # Clear batch
            batch_states = []
            batch_actions = []
            batch_masks = []
            batch_returns = []
        
        if (i_episode + 1) % 50 == 0:
            avg_loss = running_loss / (50 / BATCH_SIZE)
            avg_p = running_p_loss / (50 / BATCH_SIZE)
            avg_v = running_v_loss / (50 / BATCH_SIZE)
            print(f"Pretrain Episode {i_episode+1}/{episodes}\tLoss: {avg_loss:.4f} (P: {avg_p:.4f}, V: {avg_v:.4f})")
            running_loss = 0.0
            running_p_loss = 0.0
            running_v_loss = 0.0
            
    print("Supervised Pretraining Complete.")

def train():
    model_path = 'hearts_model.pth'
    
    device = gpu_selector.select_device()
    print(f"Training on {device}")
    
    model = HeartsTransformer(d_model=HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Add scheduler for PPO training as well to prevent oscillation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, verbose=True)
    
    start_episode = 0
    loaded_from_checkpoint = False
    
    # Interactive Model Loading
    if os.path.exists(model_path):
        user_input = input(f"Found existing model '{model_path}'. Load it? (y/n): ").strip().lower()
        if user_input == 'y':
            print(f"Loading model from {model_path}...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Check if it's a new format checkpoint (dict) or old format (state_dict)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'epoch' in checkpoint:
                        start_episode = checkpoint['epoch'] + 1
                        print(f"Resuming from episode {start_episode}")
                else:
                    # Old format fallback
                    model.load_state_dict(checkpoint)
                    print("Loaded legacy model format (no epoch info). Starting from episode 0.")
                
                print("Model loaded successfully.")
                loaded_from_checkpoint = True
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("Starting training from scratch.")
    else:
        print("No existing model found. Starting from scratch.")

    
    # Pretrain if not loaded
    if not loaded_from_checkpoint:
        # Increased pretraining episodes to allow convergence and LR scheduling to work
        pretrain_supervised(model, optimizer, device, episodes=5000)
    else:
        print("Skipping pretraining for loaded model.")
    
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

    # --- Curriculum Setup ---
    try:
        total_episodes_input = input("Enter total episodes for PPO training (default 10000): ").strip()
        TOTAL_EPISODES = int(total_episodes_input) if total_episodes_input else 10000
    except ValueError:
        TOTAL_EPISODES = 10000
    
    print(f"Total PPO Episodes: {TOTAL_EPISODES}")

    # Define Stages (Opponent Configurations)
    # Format: (Name, [Opponent1, Opponent2, Opponent3])
    # Note: Opponents are functions or strings 'pool'
    STAGES = [
        ("Stage 2: 3 Min Bots", [strategies.min_policy, strategies.min_policy, strategies.min_policy]),
        ("Stage 3: 2 Min + 1 Expert", [strategies.min_policy, strategies.min_policy, ExpertPolicy.play_policy]),
        ("Stage 4: 1 Min + 2 Expert", [strategies.min_policy, ExpertPolicy.play_policy, ExpertPolicy.play_policy]),
        ("Stage 5: 3 Expert", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, ExpertPolicy.play_policy]),
        ("Stage 6: 2 Expert + 1 Random", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, strategies.random_policy]),
        ("Stage 7: 2 Expert + 1 Pool", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, 'pool']),
        ("Stage 8: 1 Expert + 2 Pool", [ExpertPolicy.play_policy, 'pool', 'pool']),
    ]
    
    # Stage Weights (Relative duration)
    # S2(1), S3(2), S4(3), S5(4), S6(2), S7(4), S8(4) -> Total 20
    STAGE_WEIGHTS = [1, 2, 3, 4, 2, 4, 4]
    
    if loaded_from_checkpoint:
        # Resume from Stage 5 (Index 3)
        start_stage_idx = 3
        print("Resuming training: Starting from Stage 5 (3 Expert Bots)")
    else:
        # Start from Stage 2 (Index 0)
        start_stage_idx = 0
        print("Starting fresh training: Starting from Stage 2 (3 Min Bots)")
        
    # Calculate episodes per stage
    active_weights = STAGE_WEIGHTS[start_stage_idx:]
    total_weight = sum(active_weights)
    stage_episodes = [int(TOTAL_EPISODES * (w / total_weight)) for w in active_weights]
    
    # Adjust last stage to match total exactly
    stage_episodes[-1] += TOTAL_EPISODES - sum(stage_episodes)
    
    # Build Schedule: [(End Episode, Stage Index)]
    schedule = []
    current_end = start_episode
    for i, count in enumerate(stage_episodes):
        current_end += count
        schedule.append((current_end, start_stage_idx + i))
        print(f"  - {STAGES[start_stage_idx + i][0]}: {count} episodes (until ep {current_end})")

    current_stage_idx = start_stage_idx
    
    for i_episode in range(start_episode, start_episode + TOTAL_EPISODES):
        ai_player.reset()
        for opp in opponent_agents:
            opp.reset()
            
        # Update Pool periodically
        if i_episode % 50 == 0:
            pool.add(model.state_dict())
        
        # Determine Current Stage
        target_stage_idx = current_stage_idx
        for end_ep, stage_idx in schedule:
            if end_ep > i_episode:
                target_stage_idx = stage_idx
                break
        
        # Check for Stage Transition
        if target_stage_idx != current_stage_idx:
            print(f"\n*** Curriculum Stage Transition: {STAGES[current_stage_idx][0]} -> {STAGES[target_stage_idx][0]} ***")
            print(f"*** Resetting Learning Rate to {LEARNING_RATE} to adapt to new opponents ***")
            
            # Reset Learning Rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
                
            # Reset Scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, verbose=True)
            
            current_stage_idx = target_stage_idx
        
        stage_name, opponent_config = STAGES[current_stage_idx]
        difficulty = stage_name
        
        # Setup Opponents
        p_policies = []
        pass_policies = []
        
        for i, opp_type in enumerate(opponent_config):
            if opp_type == 'pool':
                # Play against pool
                if pool.pool:
                    past_state = pool.sample()
                    opponent_models[i].load_state_dict(past_state)
                    p_policies.append(opponent_agents[i].play_policy)
                    pass_policies.append(opponent_agents[i].pass_policy)
                else:
                    # Fallback if pool empty
                    p_policies.append(ExpertPolicy.play_policy)
                    pass_policies.append(strategies.random_pass_policy)
            else:
                # Static Policy
                p_policies.append(opp_type)
                # For static policies, use random pass or expert pass?
                # Let's use random pass for min/random, expert pass for expert
                if opp_type == ExpertPolicy.play_policy:
                    pass_policies.append(ExpertPolicy.pass_policy)
                else:
                    pass_policies.append(strategies.random_pass_policy)

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
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
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
            
            # Step the scheduler based on running score (we want to minimize score)
            scheduler.step(running_score)
            
            torch.save({
                'epoch': i_episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

    print("Training Complete.")
    torch.save({
        'epoch': start_episode + TOTAL_EPISODES - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)

if __name__ == "__main__":
    train()
