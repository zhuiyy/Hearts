import torch
import torch.optim as optim
import random
import json
import os
import time
from game import GameV2
from transformer import HeartsTransformer
from data_structure import PassDirection
import strategies
from strategies import ExpertPolicy
import gpu_selector
from agent import AIPlayer
import config

def pretrain_supervised(model, device, log_data, episodes=1000, parallel_model=None):
    print("Starting Supervised Pretraining...")
    if parallel_model is None:
        parallel_model = model
        
    optimizer = optim.Adam(model.parameters(), lr=config.PRETRAIN_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    game = GameV2()
    ai_player = AIPlayer(model, device)
    
    running_loss = 0.0
    running_p_loss = 0.0
    running_v_loss = 0.0
    running_acc = 0.0
    
    # For logging
    running_score = 0
    last_p_loss = 0.0
    last_v_loss = 0.0
    
    batch_states = []
    batch_actions = []
    batch_masks = []
    batch_returns = []
    
    # CTDE Batches
    batch_global_states = []
    batch_sq_labels = []
    batch_void_labels = []
    
    beta = config.DAGGER_BETA_START
    beta_decay = config.DAGGER_BETA_DECAY
    min_beta = config.DAGGER_BETA_MIN
    
    for i_episode in range(episodes):
        ai_player.reset()
        beta = max(min_beta, beta * beta_decay)
        
        main_heuristic = ExpertPolicy.play_policy
        
        # Opponents
        p1_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        p2_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        p3_policy = random.choice([strategies.min_policy, strategies.max_policy, strategies.random_policy, ExpertPolicy.play_policy])
        
        def shadow_pass_policy(player, info):
            return ai_player.pass_policy(
                player, info,
                teacher_policy=ExpertPolicy.pass_policy,
                beta=beta
            )
            
        pass_policies = [shadow_pass_policy, strategies.random_pass_policy, strategies.random_pass_policy, strategies.random_pass_policy]
        
        def shadow_policy(player, info, legal_actions, order):
            return ai_player.play_policy(
                player, info, legal_actions, order, 
                override_policy=None, 
                teacher_policy=main_heuristic,
                beta=beta
            )
            
        current_policies = [shadow_policy, p1_policy, p2_policy, p3_policy]
        
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        scores, raw_scores, events, trick_history = game.run_game_training(current_policies, pass_policies, pass_direction=pass_dir)
        
        # Rewards
        player_final_score = scores[0]
        player_raw_score = raw_scores[0]
        
        # --- Logging ---
        if i_episode == 0:
            running_score = player_final_score
        else:
            running_score = 0.05 * player_final_score + 0.95 * running_score
            
        log_data['episodes'].append(i_episode - episodes)
        log_data['scores'].append(player_final_score)
        log_data['avg_scores'].append(running_score)
        log_data['policy_losses'].append(last_p_loss)
        log_data['value_losses'].append(last_v_loss)
        log_data['difficulty'].append("Pretrain")
        log_data['phase'].append("pretrain")
        
        if i_episode % 10 == 0:
            try:
                with open(config.LOG_FILE, "w") as f:
                    json.dump(log_data, f)
            except Exception as e:
                print(f"Error writing log: {e}")

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
            
        if len(rewards) != len(ai_player.saved_values):
            min_len = min(len(rewards), len(ai_player.saved_values))
            rewards = rewards[:min_len]
            ai_player.saved_values = ai_player.saved_values[:min_len]
            ai_player.saved_states = ai_player.saved_states[:min_len]
            ai_player.saved_actions = ai_player.saved_actions[:min_len]
            ai_player.saved_masks = ai_player.saved_masks[:min_len]
            # CTDE Truncation
            ai_player.saved_global_states = ai_player.saved_global_states[:min_len]
            ai_player.saved_sq_labels = ai_player.saved_sq_labels[:min_len]
            ai_player.saved_void_labels = ai_player.saved_void_labels[:min_len]

        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + config.GAMMA * R
            returns.insert(0, R)
            
        batch_states.extend(ai_player.saved_states)
        batch_actions.extend(ai_player.saved_actions)
        batch_masks.extend(ai_player.saved_masks)
        batch_returns.extend(returns)
        
        # CTDE Extend
        batch_global_states.extend(ai_player.saved_global_states)
        batch_sq_labels.extend(ai_player.saved_sq_labels)
        batch_void_labels.extend(ai_player.saved_void_labels)
        
        if (i_episode + 1) % config.PRETRAIN_BATCH_SIZE == 0:
            b_returns = torch.tensor(batch_returns).to(device)
            if len(b_returns) > 1:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-9)
            
            b_actions = torch.stack(batch_actions)
            masks = torch.stack(batch_masks)
            
            # CTDE Stacking
            b_global_states = torch.stack(batch_global_states)
            b_sq_labels = torch.stack(batch_sq_labels)
            b_void_labels = torch.stack(batch_void_labels)
            
            for _ in range(config.PRETRAIN_EPOCHS):
                # Use parallel_model here too
                # x, padding_mask = model.assemble_batch(batch_states, device=device)
                
                logits, values, pred_sq, pred_void = parallel_model(
                    batch_raw_states=batch_states,
                    global_state=b_global_states
                )
                logits = logits.squeeze()
                values = values.squeeze()
                
                value_loss = torch.nn.functional.mse_loss(values, b_returns)
                
                # Aux Losses
                loss_sq = torch.nn.functional.cross_entropy(pred_sq, b_sq_labels)
                loss_void = torch.nn.functional.binary_cross_entropy_with_logits(pred_void, b_void_labels)
                
                masked_logits = logits + masks
                
                try:
                    policy_loss = torch.nn.functional.cross_entropy(masked_logits, b_actions, label_smoothing=config.LABEL_SMOOTHING)
                except RuntimeError as e:
                    print(f"Error in CrossEntropy: {e}")
                    policy_loss = torch.tensor(0.0, device=device, requires_grad=True)

                with torch.no_grad():
                    pred_actions = torch.argmax(masked_logits, dim=1)
                    accuracy = (pred_actions == b_actions).float().mean()
                
                loss = policy_loss + config.VALUE_LOSS_COEF * value_loss + 0.5 * loss_sq + 0.5 * loss_void
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step(loss)
                
                # Throttle to reduce GPU load
                time.sleep(config.THROTTLE_TIME)
                
                running_loss += loss.item()
                running_p_loss += policy_loss.item()
                running_v_loss += value_loss.item()
                running_acc += accuracy.item()
                
                last_p_loss = policy_loss.item()
                last_v_loss = value_loss.item()
            
            batch_states = []
            batch_actions = []
            batch_masks = []
            batch_returns = []
            
            # CTDE Clear
            batch_global_states = []
            batch_sq_labels = []
            batch_void_labels = []
        
        if (i_episode + 1) % config.PRETRAIN_BATCH_SIZE == 0:
             avg_loss = running_loss / config.PRETRAIN_EPOCHS
             avg_p = running_p_loss / config.PRETRAIN_EPOCHS
             avg_v = running_v_loss / config.PRETRAIN_EPOCHS
             avg_acc = running_acc / config.PRETRAIN_EPOCHS
             current_lr = optimizer.param_groups[0]['lr']
             print(f"Pretrain Episode {i_episode+1}/{episodes}\tLoss: {avg_loss:.4f} (P: {avg_p:.4f}, V: {avg_v:.4f})\tAcc: {avg_acc:.2%}\tBeta: {beta:.2f}\tLR: {current_lr:.2e}")
             running_loss = 0.0
             running_p_loss = 0.0
             running_v_loss = 0.0
             running_acc = 0.0
            
    print("Supervised Pretraining Complete.")
    
    # Save Pretrained Model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episodes': episodes
    }, config.PRETRAINED_MODEL_PATH)
    print(f"Pretrained model saved to {config.PRETRAINED_MODEL_PATH}")

def main():
    device_selection = gpu_selector.select_device()
    
    use_data_parallel = False
    device_ids = []
    
    if isinstance(device_selection, str) and "cuda:" in device_selection and "," in device_selection:
        use_data_parallel = True
        gpu_indices = [int(x) for x in device_selection.split(":")[1].split(",")]
        device_ids = gpu_indices
        device = torch.device(f"cuda:{gpu_indices[0]}")
        print(f"Pretraining on Multiple GPUs: {device_ids}")
    else:
        device = device_selection
        print(f"Pretraining on {device}")
    
    model = HeartsTransformer(d_model=config.HIDDEN_DIM, dropout=config.DROPOUT).to(device)
    
    if use_data_parallel:
        parallel_model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        parallel_model = model

    log_data = {'episodes': [], 'scores': [], 'avg_scores': [], 'policy_losses': [], 'value_losses': [], 'difficulty': [], 'phase': []}
    
    try:
        episodes_input = input(f"Enter pretrain episodes (default {config.PRETRAIN_EPISODES}): ").strip()
        episodes = int(episodes_input) if episodes_input else config.PRETRAIN_EPISODES
    except ValueError:
        episodes = config.PRETRAIN_EPISODES

    pretrain_supervised(model, device, log_data, episodes=episodes, parallel_model=parallel_model)

if __name__ == "__main__":
    main()
