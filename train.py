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
# --- PPO Training ---
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPISODES = 5000 # Default total episodes if not specified
BATCH_SIZE = 32 # Update every 32 games
PPO_EPOCHS = 4
CLIP_EPS = 0.2
ENTROPY_COEF = 0.03 # Entropy regularization coefficient
VALUE_LOSS_COEF = 0.5 # Value loss coefficient
MAX_GRAD_NORM = 0.5 # Gradient Clipping

# --- Model Architecture ---
HIDDEN_DIM = 256 # Increased from 128
DROPOUT = 0.1 # Dropout probability
WEIGHT_DECAY = 1e-5 # L2 Regularization

# --- Supervised Pretraining ---
PRETRAIN_LR = 1e-4 # Lower LR for stability
PRETRAIN_EPISODES = 5000
PRETRAIN_BATCH_SIZE = 32 # Reduced from 128 to prevent OOM
LABEL_SMOOTHING = 0.0 # MUST BE 0.0 when using Masking (-inf), otherwise Loss becomes Inf

# --- DAgger (Dataset Aggregation) ---
DAGGER_BETA_START = 1.0 # Start with pure teacher forcing
DAGGER_BETA_DECAY = 0.9995 # Decay per episode
DAGGER_BETA_MIN = 0.3 # Minimum teacher forcing ratio

# --- Curriculum Learning ---
CURRICULUM_SCORE_THRESHOLD = 6.0 # Avg score to pass a stage
CURRICULUM_STABILITY_WINDOW = 500 # Episodes to maintain score
MAX_STAGE_EPISODES = 3000 # Max episodes per stage before forced transition
POOL_STAGE_DURATION = 2000 # Episodes for pool stages

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

    def play_policy(self, player, info, legal_actions, order, override_policy=None, teacher_policy=None, beta=1.0):
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
        
        # Determine Action to Play and Action to Save (Label)
        
        # Default: Student plays, Student is label
        student_action_idx = dist.sample()
        
        # Convert index back to Card
        suit_val = student_action_idx.item() // 13
        rank_val = (student_action_idx.item() % 13) + 1
        student_card = Card(suit=list(from_suit_int(suit_val))[0], rank=rank_val)
        
        action_to_play = student_card
        action_to_save = student_action_idx
        
        if override_policy:
            # Pure Override (e.g. for pure BC or debugging)
            selected_card = override_policy(player, info, legal_actions, order)
            action_to_play = selected_card
            action_to_save = torch.tensor(selected_card.to_id(), device=self.device)
            
        elif teacher_policy:
            # DAgger / Teacher Forcing Logic
            # 1. Get Teacher Action
            teacher_card = teacher_policy(player, info, legal_actions, order)
            teacher_action_idx = torch.tensor(teacher_card.to_id(), device=self.device)
            
            # 2. Set Label to Teacher Action (Always train to imitate teacher)
            action_to_save = teacher_action_idx
            
            # 3. Decide who plays based on Beta
            # Beta = 1.0 -> Teacher plays (Pure BC)
            # Beta = 0.0 -> Student plays (Pure DAgger)
            if random.random() < beta:
                action_to_play = teacher_card
            else:
                action_to_play = student_card

        # Save log prob and value
        # Note: We save log_prob of the SAVED action (Target), or the PLAYED action?
        # For PPO, we need log_prob of PLAYED action.
        # For Supervised, we don't use log_prob, we use CrossEntropy(logits, saved_action).
        # So let's save log_prob of PLAYED action to be safe/consistent, 
        # but saved_actions will contain the LABEL.
        # WAIT: If we use saved_actions for PPO later, it must match the trajectory!
        # But pretrain_supervised doesn't use PPO. It uses custom loop.
        # So for pretrain, saved_actions = LABEL is fine.
        # But we must ensure we don't run PPO update on this buffer if it's mixed.
        # pretrain_supervised has its own update loop, so it's fine.
        
        self.saved_log_probs.append(dist.log_prob(torch.tensor(action_to_play.to_id(), device=self.device)))
        self.saved_values.append(value)
        self.saved_actions.append(action_to_save) # <--- This is the Target for CrossEntropy
        
        return action_to_play

def from_suit_int(val):
    # Helper to get Suit enum from int
    from data_structure import Suit
    for s in Suit:
        if s.value == val:
            yield s

def pretrain_supervised(model, device, episodes=PRETRAIN_EPISODES):
    print(f"Starting Supervised Pretraining (Policy + Value) for {episodes} episodes...")
    game = GameV2()
    ai_player = AIPlayer(model, device)
    
    # Create a separate optimizer for pretraining
    # Use a smaller LR for Policy stability, but keep it separate
    # Reduced LR to 1e-4 to prevent gradient explosion/inf loss
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    
    # Adaptive Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    running_loss = 0.0
    running_p_loss = 0.0
    running_v_loss = 0.0
    running_acc = 0.0
    
    # Batch buffers
    batch_states = []
    batch_actions = []
    batch_masks = []
    batch_returns = []
    
    # DAgger Beta Schedule
    # Start with 1.0 (Pure BC) and decay to 0.5 (Mix)
    # We don't want to go to 0.0 because random exploration might be too chaotic for Value learning
    beta = DAGGER_BETA_START
    beta_decay = DAGGER_BETA_DECAY # Decay per episode
    min_beta = DAGGER_BETA_MIN
    
    for i_episode in range(episodes):
        ai_player.reset()
        
        # Update Beta
        beta = max(min_beta, beta * beta_decay)
        
        # Use Expert Policy as the teacher
        main_heuristic = ExpertPolicy.play_policy
        
        # Opponents can be mixed
        p1_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        p2_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        p3_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        
        # Pass policies
        pass_policies = [strategies.random_pass_policy] * 4
        
        # Wrapper to inject heuristic into AIPlayer using DAgger logic
        def shadow_policy(player, info, legal_actions, order):
            return ai_player.play_policy(
                player, info, legal_actions, order, 
                override_policy=None, 
                teacher_policy=main_heuristic,
                beta=beta
            )
            
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
        
        # Update every PRETRAIN_BATCH_SIZE episodes
        if (i_episode + 1) % PRETRAIN_BATCH_SIZE == 0:
            b_returns = torch.tensor(batch_returns).to(device)
            
            # Normalize returns for stability (Critical for Value Loss)
            if len(b_returns) > 1:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-9)
            
            b_actions = torch.stack(batch_actions)
            
            # Forward pass on batch
            logits, values = model(batch_raw_states=batch_states)
            logits = logits.squeeze() # (Batch, 52)
            values = values.squeeze() # (Batch)
            
            # 1. Value Loss (MSE)
            value_loss = torch.nn.functional.mse_loss(values, b_returns)
            
            # 2. Policy Loss (Cross Entropy with Label Smoothing)
            masks = torch.stack(batch_masks)
            masked_logits = logits + masks
            
            # Check for NaN/Inf in logits
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                # If we have infs that are NOT from the mask (mask uses -inf), we have a problem.
                # But mask uses -inf, so isinf will be true.
                # We need to check for positive inf or nan.
                if torch.isnan(masked_logits).any() or (masked_logits == float('inf')).any():
                     print("Warning: NaN or Pos Inf detected in logits!")
            
            # Use label smoothing to prevent overconfidence and reduce oscillation
            # Ensure b_actions are within valid range [0, 51]
            # And ensure the target logit is not masked out (should be impossible if data is correct)
            
            try:
                policy_loss = torch.nn.functional.cross_entropy(masked_logits, b_actions, label_smoothing=LABEL_SMOOTHING)
            except RuntimeError as e:
                print(f"Error in CrossEntropy: {e}")
                policy_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Calculate Accuracy
            with torch.no_grad():
                pred_actions = torch.argmax(masked_logits, dim=1)
                accuracy = (pred_actions == b_actions).float().mean()
            
            # Check for Inf Loss
            if torch.isinf(policy_loss) or torch.isnan(policy_loss):
                print("Warning: Policy Loss is Inf/NaN. Skipping update.")
                loss = torch.tensor(0.0, device=device, requires_grad=True) # Dummy loss
            else:
                loss = policy_loss + VALUE_LOSS_COEF * value_loss
            
            optimizer.zero_grad()
            
            if loss.item() != 0.0:
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                # Step Scheduler with current loss
                scheduler.step(loss)
            
            running_loss += loss.item() if not torch.isnan(loss) else 0
            running_p_loss += policy_loss.item() if not torch.isnan(policy_loss) else 0
            running_v_loss += value_loss.item() if not torch.isnan(value_loss) else 0
            running_acc += accuracy.item()
            
            # Clear batch
            batch_states = []
            batch_actions = []
            batch_masks = []
            batch_returns = []
        
        if (i_episode + 1) % 50 == 0:
            # Adjust logging frequency scaling
            steps = 50 / PRETRAIN_BATCH_SIZE
            if steps < 1: steps = 1 # Avoid division by zero if batch > 50
            
            # Only print if we actually updated (approximate)
            # Or just print the running averages
            # Since batch size is 128, we update every 128 episodes.
            # Printing every 50 might show stale data.
            pass

        if (i_episode + 1) % PRETRAIN_BATCH_SIZE == 0:
             avg_loss = running_loss 
             avg_p = running_p_loss 
             avg_v = running_v_loss 
             avg_acc = running_acc
             print(f"Pretrain Episode {i_episode+1}/{episodes}\tLoss: {avg_loss:.4f} (P: {avg_p:.4f}, V: {avg_v:.4f})\tAcc: {avg_acc:.2%}\tBeta: {beta:.2f}")
             running_loss = 0.0
             running_p_loss = 0.0
             running_v_loss = 0.0
             running_acc = 0.0
            
    print("Supervised Pretraining Complete.")

def train():
    model_path = 'hearts_model.pth'
    
    device = gpu_selector.select_device()
    print(f"Training on {device}")
    
    model = HeartsTransformer(d_model=HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Add scheduler for PPO training as well to prevent oscillation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
    
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
        pretrain_supervised(model, device, episodes=PRETRAIN_EPISODES)
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
    # MODIFIED: Added "Random Bots" as warm-up to prevent "Expert Shock"
    STAGES = [
        ("Stage 2: 3 Random Bots", [strategies.random_policy, strategies.random_policy, strategies.random_policy]),
        ("Stage 3: 2 Min + 1 Expert", [strategies.min_policy, strategies.min_policy, ExpertPolicy.play_policy]),
        ("Stage 4: 1 Min + 2 Expert", [strategies.min_policy, ExpertPolicy.play_policy, ExpertPolicy.play_policy]),
        ("Stage 5: 3 Expert", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, ExpertPolicy.play_policy]),
        ("Stage 6: 2 Expert + 1 Random", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, strategies.random_policy]),
        ("Stage 7: 2 Expert + 1 Pool", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, 'pool']),
        ("Stage 8: 1 Expert + 2 Pool", [ExpertPolicy.play_policy, 'pool', 'pool']),
        ("Stage 9: 3 Pool", ['pool', 'pool', 'pool']),
    ]
    
    # Stage Weights (Relative duration)
    STAGE_WEIGHTS = [1, 2, 3, 4, 2, 4, 4, 4]
    
    if loaded_from_checkpoint:
        # Resume from Stage 5 (Index 3) - 3 Experts
        start_stage_idx = 3
        print("Resuming training: Starting from Stage 5 (3 Expert Bots)")
    else:
        # Start from Stage 2 (Index 0)
        start_stage_idx = 0
        print("Starting fresh training: Starting from Stage 2 (3 Random Bots)")
        
    # Dynamic Curriculum State
    current_stage_idx = start_stage_idx
    stage_episode_count = 0
    consecutive_good_epochs = 0
    
    # Pool stages start index (Stage 7 is index 5)
    POOL_START_IDX = 5
    
    print(f"Starting Dynamic Curriculum Training...")
    print(f"Transition Condition (Non-Pool Stages): Avg Score < {CURRICULUM_SCORE_THRESHOLD} for {CURRICULUM_STABILITY_WINDOW} consecutive episodes OR Max {MAX_STAGE_EPISODES} episodes.")
    
    for i_episode in range(start_episode, start_episode + TOTAL_EPISODES):
        ai_player.reset()
        for opp in opponent_agents:
            opp.reset()
            
        # Update Pool periodically
        if i_episode % 50 == 0:
            pool.add(model.state_dict())
        
        # --- Dynamic Stage Logic ---
        stage_episode_count += 1
        
        # Check transition logic
        should_transition = False
        
        # Only apply dynamic logic for non-pool stages (Indices 0-4)
        if current_stage_idx < POOL_START_IDX:
            # Check performance condition
            if running_score < CURRICULUM_SCORE_THRESHOLD:
                consecutive_good_epochs += 1
            else:
                consecutive_good_epochs = 0
                
            if consecutive_good_epochs >= CURRICULUM_STABILITY_WINDOW:
                print(f"  -> Performance condition met: {consecutive_good_epochs} consecutive episodes with score < {CURRICULUM_SCORE_THRESHOLD}")
                should_transition = True
            elif stage_episode_count >= MAX_STAGE_EPISODES:
                print(f"  -> Max stage episodes reached ({MAX_STAGE_EPISODES})")
                should_transition = True
        else:
            # For Pool stages, use a fixed duration (e.g., 2000 episodes)
            if stage_episode_count >= POOL_STAGE_DURATION:
                should_transition = True

        # Perform Transition
        if should_transition and current_stage_idx < len(STAGES) - 1:
            target_stage_idx = current_stage_idx + 1
            
            print(f"\n*** Curriculum Stage Transition: {STAGES[current_stage_idx][0]} -> {STAGES[target_stage_idx][0]} ***")
            print(f"*** Resetting Learning Rate to {LEARNING_RATE} to adapt to new opponents ***")
            
            # Reset Learning Rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
                
            # Reset Scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
            
            current_stage_idx = target_stage_idx
            stage_episode_count = 0
            consecutive_good_epochs = 0
        
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
                
                # Entropy Bonus (Increased to 0.03 to encourage exploration against tough opponents)
                entropy_loss = -ENTROPY_COEF * entropies.mean()
                
                loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss
                
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
