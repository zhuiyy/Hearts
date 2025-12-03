import torch
import torch.optim as optim
import random
import json
import os
import time
from collections import deque
from game import GameV2
from transformer import HeartsTransformer
from data_structure import PassDirection
import strategies
from strategies import ExpertPolicy
import gpu_selector
from agent import AIPlayer, OpponentPool
import config

def train():
    model_path = config.MODEL_PATH
    device_selection = gpu_selector.select_device()
    
    use_data_parallel = False
    device_ids = []
    
    if isinstance(device_selection, str) and "cuda:" in device_selection and "," in device_selection:
        # Multiple GPUs selected
        use_data_parallel = True
        gpu_indices = [int(x) for x in device_selection.split(":")[1].split(",")]
        device_ids = gpu_indices
        device = torch.device(f"cuda:{gpu_indices[0]}") # Main device
        print(f"Training on Multiple GPUs: {device_ids}")
    else:
        device = device_selection
        print(f"Training on {device}")
    
    model = HeartsTransformer(d_model=config.HIDDEN_DIM, dropout=config.DROPOUT).to(device)
    
    if use_data_parallel:
        # We wrap the model for training updates, but keep 'model' as the base for inference
        parallel_model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        parallel_model = model

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
    
    start_episode = 0
    loaded_from_checkpoint = False
    loaded_running_score = None
    loaded_stage_idx = None
    
    # Check for existing RL checkpoint first
    if os.path.exists(model_path):
        user_input = input(f"Found existing RL model '{model_path}'. Load it? (y/n): ").strip().lower()
        if user_input == 'y':
            print(f"Loading model from {model_path}...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'epoch' in checkpoint:
                        start_episode = checkpoint['epoch'] + 1
                        print(f"Resuming from episode {start_episode}")
                    if 'running_score' in checkpoint:
                        loaded_running_score = checkpoint['running_score']
                    if 'stage_idx' in checkpoint:
                        loaded_stage_idx = checkpoint['stage_idx']
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded legacy model format. Starting from episode 0.")
                print("Model loaded successfully.")
                loaded_from_checkpoint = True
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("Starting training from scratch (or pretrained).")
    
    # If not loaded from RL checkpoint, check for Pretrained model
    if not loaded_from_checkpoint and os.path.exists(config.PRETRAINED_MODEL_PATH):
        user_input = input(f"Found PRETRAINED model '{config.PRETRAINED_MODEL_PATH}'. Load it as starting point? (y/n): ").strip().lower()
        if user_input == 'y':
            print(f"Loading pretrained model from {config.PRETRAINED_MODEL_PATH}...")
            try:
                checkpoint = torch.load(config.PRETRAINED_MODEL_PATH, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Filter out keys that don't match (e.g. if architecture changed slightly)
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and v.size() == model_dict[k].size()}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    
                    # We do NOT load optimizer state from pretraining, as we are starting fresh RL
                    print(f"Pretrained weights loaded ({len(pretrained_dict)}/{len(model_dict)} layers). Starting RL from scratch.")
                else:
                    print("Invalid pretrained model format.")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")

    log_data = {'episodes': [], 'scores': [], 'avg_scores': [], 'policy_losses': [], 'value_losses': [], 'difficulty': [], 'phase': []}
    
    game = GameV2()
    ai_player = AIPlayer(model, device)
    pool = OpponentPool()
    
    opponent_models = [HeartsTransformer(d_model=config.HIDDEN_DIM).to(device) for _ in range(3)]
    opponent_agents = [AIPlayer(opp_model, device) for opp_model in opponent_models]
    
    running_reward = 0
    running_score = 0
    
    batch_log_probs = []
    batch_values = []
    batch_states = []
    batch_actions = []
    batch_masks = []
    batch_returns = []

    try:
        total_episodes_input = input("Enter total episodes for PPO training (default 10000): ").strip()
        TOTAL_EPISODES = int(total_episodes_input) if total_episodes_input else 10000
    except ValueError:
        TOTAL_EPISODES = 10000
    
    print(f"Total PPO Episodes: {TOTAL_EPISODES}")

    STAGES = [
        # Removed "3 Random Bots" stage to prevent learning bad habits (e.g. "playing low always works")
        ("Stage 1: 1 Expert + 2 Random", [ExpertPolicy.play_policy, strategies.random_policy, strategies.random_policy]),
        ("Stage 2: 2 Expert + 1 Random", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, strategies.random_policy]),
        ("Stage 3: 3 Expert", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, ExpertPolicy.play_policy]),
        ("Stage 4: 2 Expert + 1 Pool", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, 'pool']),
        ("Stage 5: 1 Expert + 2 Pool", [ExpertPolicy.play_policy, 'pool', 'pool']),
        ("Stage 6: 3 Pool", ['pool', 'pool', 'pool']),
    ]
    
    if loaded_from_checkpoint:
        if loaded_stage_idx is not None:
            start_stage_idx = loaded_stage_idx
            print(f"Resuming training: Starting from Stage {start_stage_idx} (Loaded from checkpoint)")
        else:
            start_stage_idx = 2
            print("Resuming training: Starting from Stage 3 (3 Expert Bots) - Default for legacy checkpoint")
    else:
        start_stage_idx = 0
        print("Starting fresh training: Starting from Stage 1 (1 Expert + 2 Random)")
        
    current_stage_idx = start_stage_idx
    stage_episode_count = 0
    consecutive_good_epochs = 0
    POOL_START_IDX = 3
    
    best_running_score = float('inf') # Track best score
    if loaded_running_score is not None:
        best_running_score = loaded_running_score # Assume loaded model was the best at its time

    for i_episode in range(start_episode, start_episode + TOTAL_EPISODES):
        ai_player.reset()
        for opp in opponent_agents:
            opp.reset()
            
        if i_episode % 50 == 0:
            pool.add(model.state_dict())
        
        stage_episode_count += 1
        should_transition = False
        
        if current_stage_idx < POOL_START_IDX:
            if running_score < config.CURRICULUM_SCORE_THRESHOLD:
                consecutive_good_epochs += 1
            else:
                consecutive_good_epochs = 0
                
            if consecutive_good_epochs >= config.CURRICULUM_STABILITY_WINDOW:
                print(f"  -> Performance condition met: {consecutive_good_epochs} consecutive episodes with score < {config.CURRICULUM_SCORE_THRESHOLD}")
                should_transition = True
            elif stage_episode_count >= config.MAX_STAGE_EPISODES:
                print(f"  -> Max stage episodes reached ({config.MAX_STAGE_EPISODES})")
                should_transition = True
        else:
            if stage_episode_count >= config.POOL_STAGE_DURATION:
                should_transition = True

        if should_transition and current_stage_idx < len(STAGES) - 1:
            target_stage_idx = current_stage_idx + 1
            print(f"\n*** Curriculum Stage Transition: {STAGES[current_stage_idx][0]} -> {STAGES[target_stage_idx][0]} ***")
            print(f"*** Resetting Learning Rate to {config.LEARNING_RATE} ***")
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
            current_stage_idx = target_stage_idx
            stage_episode_count = 0
            consecutive_good_epochs = 0
        
        stage_name, opponent_config = STAGES[current_stage_idx]
        difficulty = stage_name
        
        p_policies = []
        pass_policies = []
        
        for i, opp_type in enumerate(opponent_config):
            if opp_type == 'pool':
                if pool.pool:
                    past_state = pool.sample()
                    opponent_models[i].load_state_dict(past_state)
                    p_policies.append(opponent_agents[i].play_policy)
                    pass_policies.append(opponent_agents[i].pass_policy)
                else:
                    p_policies.append(ExpertPolicy.play_policy)
                    pass_policies.append(strategies.random_pass_policy)
            else:
                p_policies.append(opp_type)
                if opp_type == ExpertPolicy.play_policy:
                    pass_policies.append(ExpertPolicy.pass_policy)
                else:
                    pass_policies.append(strategies.random_pass_policy)

        current_policies = [ai_player.play_policy] + p_policies
        current_pass_policies = [ai_player.pass_policy] + pass_policies
        
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        scores, raw_scores, events, trick_history = game.run_game_training(current_policies, current_pass_policies, pass_direction=pass_dir)
        
        player_final_score = scores[0]
        player_raw_score = raw_scores[0]
        
        # --- NEW REWARD FUNCTION: Relative Score ---
        # Goal: Maximize (Avg_Opponent_Score - My_Score)
        # This naturally handles Shooting the Moon (My=0, Opps=26 -> Huge Reward)
        # and normal play (My=0, Opps=Avg -> Positive Reward)
        
        opp_scores = scores[1:]
        avg_opp_score = sum(opp_scores) / 3.0
        
        # Scale factor: 10.0 roughly normalizes the range to [-2.6, +2.6]
        # STM: (26 - 0)/10 = +2.6
        # Eat Queen: (4.3 - 13)/10 = -0.87
        # Perfect Safe: (8.6 - 0)/10 = +0.86
        relative_reward = (avg_opp_score - player_final_score) / 10.0
        
        rewards = []
        
        # Distribute reward across all actions in the episode
        # We use a pure terminal reward to avoid noisy intermediate signals
        # But we can add small shaping rewards for tricks if needed. 
        # For now, let's try pure terminal reward + small trick penalty shaping.
        
        # Pass Phase Rewards (3 actions)
        if pass_dir != PassDirection.KEEP:
             for _ in range(3):
                 rewards.append(relative_reward * 0.1) # Small signal for passing

        # Trick Phase Rewards (13 actions)
        for trick in trick_history:
            # Intermediate shaping: slight penalty for taking points to encourage safety
            # But main signal comes from final relative_reward
            r = 0.0
            if trick.winner == 0:
                points = trick.score
                if points > 0:
                    r = -0.05 * (points / 13.0) # Small penalty for taking points
            
            # Add the terminal reward to the last action, or distribute it?
            # Standard PPO often uses terminal reward at the end.
            # Here we add the relative_reward to EVERY step (or just the last).
            # Let's add it to every step but scaled down, or just use it as the return.
            # Better approach: The return G_t will propagate the terminal reward back.
            # So we just give 0 intermediate reward and big terminal reward.
            
            rewards.append(r)
            
        # Add the big relative reward to the very last action
        rewards[-1] += relative_reward

        if len(rewards) != len(ai_player.saved_log_probs):
            min_len = min(len(rewards), len(ai_player.saved_log_probs))
            rewards = rewards[:min_len]
            ai_player.saved_log_probs = ai_player.saved_log_probs[:min_len]
            ai_player.saved_values = ai_player.saved_values[:min_len]
            ai_player.saved_states = ai_player.saved_states[:min_len]
            ai_player.saved_actions = ai_player.saved_actions[:min_len]
            ai_player.saved_masks = ai_player.saved_masks[:min_len]

        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + config.GAMMA * R
            returns.insert(0, R)
            
        batch_log_probs.extend(ai_player.saved_log_probs)
        batch_values.extend(ai_player.saved_values)
        batch_states.extend(ai_player.saved_states)
        batch_actions.extend(ai_player.saved_actions)
        batch_masks.extend(ai_player.saved_masks)
        batch_returns.extend(returns)

        current_p_loss = 0.0
        current_v_loss = 0.0

        if (i_episode + 1) % config.BATCH_SIZE == 0:
            b_returns = torch.tensor(batch_returns).to(device)
            if len(b_returns) > 1:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-9)

            old_log_probs = torch.stack(batch_log_probs).detach()
            old_values = torch.stack(batch_values).detach().squeeze()
            
            advantages = b_returns - old_values
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                
            for _ in range(config.PPO_EPOCHS):
                # Use parallel_model for forward pass (supports DataParallel)
                # We pass batch_raw_states as argument so DataParallel can split it
                logits, new_values = parallel_model(batch_raw_states=batch_states)
                logits = logits.squeeze()
                new_values = new_values.squeeze()
                
                masks = torch.stack(batch_masks)
                masked_logits = logits + masks
                
                probs = torch.softmax(masked_logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                
                b_actions = torch.stack(batch_actions)
                new_log_probs = dist.log_prob(b_actions)
                entropies = dist.entropy()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = torch.nn.functional.mse_loss(new_values, b_returns)
                
                progress = min(1.0, (i_episode - start_episode) / TOTAL_EPISODES)
                current_entropy_coef = config.ENTROPY_COEF_START - (config.ENTROPY_COEF_START - config.ENTROPY_COEF_END) * progress
                
                entropy_loss = -current_entropy_coef * entropies.mean()
                
                loss = policy_loss + config.VALUE_LOSS_COEF * value_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                
                # Throttle to reduce GPU load
                time.sleep(config.THROTTLE_TIME)
                
                current_p_loss = policy_loss.item()
                current_v_loss = value_loss.item()
            
            batch_log_probs = []
            batch_values = []
            batch_states = []
            batch_actions = []
            batch_masks = []
            batch_returns = []
            
        total_reward = sum(rewards)
        if i_episode == 0:
            running_score = 10.0 # Start with a conservative score to avoid initial fluctuation bias
            running_reward = total_reward
        elif i_episode == start_episode and loaded_running_score is not None:
             # Resume running score from checkpoint
             running_score = loaded_running_score
             running_reward = total_reward 
        elif i_episode == start_episode:
             # Resumed but no saved score
             running_score = 10.0
             running_reward = total_reward
        else:
            running_score = 0.05 * player_final_score + 0.95 * running_score
            running_reward = 0.05 * total_reward + 0.95 * running_reward
        
        log_data['episodes'].append(i_episode)
        log_data['scores'].append(player_final_score)
        log_data['avg_scores'].append(running_score)
        log_data['policy_losses'].append(current_p_loss)
        log_data['value_losses'].append(current_v_loss)
        log_data['difficulty'].append(difficulty)
        log_data['phase'].append('ppo')
        
        if i_episode % 10 == 0:
            try:
                with open(config.LOG_FILE, "w") as f:
                    json.dump(log_data, f)
            except Exception as e:
                print(f"Error writing log: {e}")
        
        if i_episode % 50 == 0:
            print(f"Episode {i_episode}\tAvg Score: {running_score:.2f}\tAvg Reward: {running_reward:.2f}\tPass: {pass_dir.name}\tDiff: {difficulty}")
            scheduler.step(running_score)
            torch.save({
                'epoch': i_episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'running_score': running_score,
                'stage_idx': current_stage_idx
            }, model_path)
            
            # Save Best Model
            # Only save best model after some warmup to let moving average stabilize
            if i_episode > start_episode + 50 and running_score < best_running_score:
                best_running_score = running_score
                torch.save({
                    'epoch': i_episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'score': best_running_score,
                    'running_score': running_score,
                    'stage_idx': current_stage_idx
                }, config.BEST_MODEL_PATH)
                print(f"  -> New Best Model Saved! Score: {best_running_score:.2f}")

    print("Training Complete.")
    torch.save({
        'epoch': start_episode + TOTAL_EPISODES - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'running_score': running_score,
        'stage_idx': current_stage_idx
    }, model_path)

if __name__ == "__main__":
    train()
