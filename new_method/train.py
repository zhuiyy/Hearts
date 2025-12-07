import torch
import torch.optim as optim
import torch.nn.functional as F
from game import GameV2
from model import HeartsProNet
from agent import SotaAgent
from data_structure import PassDirection
import random
import json
import os
import config
from collections import deque

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = HeartsProNet(config.HIDDEN_DIM, config.LSTM_HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    
    # Load checkpoint if exists
    start_episode = 0
    best_score = float('inf')
    
    # Force load pretrained model if no checkpoint exists OR if user wants to restart
    # But logic here is: Checkpoint > Pretrained > Fresh
    
    if os.path.exists(config.MODEL_PATH):
        # If you want to force restart from pretrain, delete the checkpoint file manually
        # or change this logic.
        try:
            checkpoint = torch.load(config.MODEL_PATH)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint.get('episode', 0) + 1
                best_score = checkpoint.get('best_score', float('inf'))
                print(f"Resuming training from episode {start_episode} (Best Score: {best_score:.2f})")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded legacy model format (weights only). Starting fresh epoch.")
        except Exception as e:
            print(f"Could not load existing model: {e}, starting fresh.")
            
    elif os.path.exists(config.PRETRAINED_MODEL_PATH):
        try:
            print(f"Loading pretrained model from {config.PRETRAINED_MODEL_PATH}...")
            model.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH))
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
    else:
        print("No existing model found. Consider running 'pretrain.py' first for faster convergence.")

    agent = SotaAgent(model, device, use_pimc=False)
    game = GameV2()
    
    # Logging
    log_data = {'episodes': [], 'scores': [], 'rewards': []}
    # Try to load existing log if resuming
    if start_episode > 0 and os.path.exists(config.LOG_FILE):
        try:
            with open(config.LOG_FILE, 'r') as f:
                log_data = json.load(f)
        except:
            pass
            
    score_window = deque(maxlen=100)
    reward_window = deque(maxlen=100)
    
    for episode in range(start_episode, config.TOTAL_EPISODES):
        # agent.reset() # REMOVED: Do not clear buffer here! Only clear after update.
        
        # Track actions count to assign rewards correctly later
        n_actions_before = len(agent.saved_actions)

        # Wrapper for GameV2 to use SotaAgent
        def agent_policy_wrapper(player, info, legal, order):
            return agent.act(player, info, legal, order, training=True)
            
        def random_policy(player, info, legal, order):
            return random.choice(legal)
            
        # Curriculum: Start with 3 Random Bots
        # Later stages can replace random_policy with past versions of agent
        from strategies import ExpertPolicy
        
        def expert_policy_wrapper(player, info, legal, order):
            return ExpertPolicy.play_policy(player, info, legal, order)
            
        # policies = [agent_policy_wrapper, random_policy, random_policy, random_policy]
        # Use Expert Bots as opponents for better training signal
        policies = [agent_policy_wrapper, expert_policy_wrapper, expert_policy_wrapper, expert_policy_wrapper]
        
        pass_policies = [agent.pass_policy, ExpertPolicy.pass_policy, ExpertPolicy.pass_policy, ExpertPolicy.pass_policy]
        
        # Randomize pass direction
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        
        # Calculate Reward
        my_score = scores[0]
        opp_scores = scores[1:]
        opp_avg = sum(opp_scores) / 3.0
        
        # Reward Shaping
        # 1. Relative Score
        reward = (opp_avg - my_score) / 10.0
        
        # 2. Shooting the Moon Bonus
        if my_score == 0 and 26 in opp_scores:
            reward += 5.0 # Big bonus for STM
            
        # Assign reward to actions taken IN THIS EPISODE only
        n_actions_after = len(agent.saved_actions)
        steps_this_episode = n_actions_after - n_actions_before
        agent.rewards.extend([reward] * steps_this_episode)
        
        score_window.append(my_score)
        avg_score = sum(score_window) / len(score_window)
        
        reward_window.append(reward)
        avg_reward = sum(reward_window) / len(reward_window)
        
        # PPO Update
        if (episode + 1) % config.BATCH_SIZE == 0:
            loss_info = update_ppo(agent, optimizer)
            scheduler.step()
            print(f"Ep {episode+1} | Avg Score (100): {avg_score:.2f} | Avg Reward (100): {avg_reward:.2f} | Loss: {loss_info['loss']:.4f}")
            
            # Save Checkpoint
            if (episode + 1) % 100 == 0:
                checkpoint = {
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score
                }
                torch.save(checkpoint, config.MODEL_PATH)
                
            # Save Log to File
            try:
                with open(config.LOG_FILE, 'w') as f:
                    json.dump(log_data, f)
            except Exception as e:
                print(f"Error saving log: {e}")

        if episode % 10 == 0:
            log_data['episodes'].append(episode)
            log_data['scores'].append(my_score)
            log_data['rewards'].append(reward)
            
            # Save Log to File
            try:
                with open(config.LOG_FILE, 'w') as f:
                    json.dump(log_data, f)
            except Exception as e:
                print(f"Error saving log: {e}")
            
    # Final Save
    checkpoint = {
        'episode': config.TOTAL_EPISODES - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_score
    }
    torch.save(checkpoint, config.MODEL_PATH)
    print("Training Complete.")

def update_ppo(agent, optimizer):
    # 1. Stack Data from Buffer
    if not agent.saved_static_obs:
        return {'loss': 0.0}

    b_static_obs = torch.stack(agent.saved_static_obs)
    b_seq_cards = torch.stack(agent.saved_seq_cards)
    b_seq_players = torch.stack(agent.saved_seq_players)
    b_global_priv = torch.stack(agent.saved_global_priv)
    b_masks = torch.stack(agent.saved_masks)
    
    b_actions = torch.stack(agent.saved_actions)
    b_log_probs = torch.stack(agent.saved_log_probs).detach()
    b_qs_labels = torch.stack(agent.saved_qs_labels)
    
    # 2. Calculate Returns & Advantages
    # Using simple Monte Carlo returns (already in agent.rewards)
    b_rewards = torch.tensor(agent.rewards, dtype=torch.float32).to(agent.device)
    b_old_values = torch.stack(agent.saved_values).detach()
    
    # Normalize Returns
    b_returns = b_rewards
    if b_returns.std() > 1e-5:
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-5)
        
    # Advantages = Returns - Values
    advantages = b_returns - b_old_values
    if advantages.std() > 1e-5:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
    # 3. PPO Epochs
    total_loss = 0
    dataset_size = b_static_obs.size(0)
    indices = list(range(dataset_size))
    
    for _ in range(config.PPO_EPOCHS):
        random.shuffle(indices)
        
        # Full batch update for now (dataset_size is small, ~400 steps for batch 32)
        # Re-evaluate
        logits, values, qs_pred = agent.model(b_static_obs, b_seq_cards, b_seq_players, b_global_priv)
        values = values.squeeze()
        
        # Mask Logits
        masked_logits = logits + b_masks
        probs = torch.softmax(masked_logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        
        new_log_probs = dist.log_prob(b_actions)
        entropy = dist.entropy().mean()
        
        # Ratio
        ratio = torch.exp(new_log_probs - b_log_probs)
        
        # Surrogate Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss
        value_loss = F.mse_loss(values, b_returns)
        
        # Aux Loss (SQ Prediction)
        qs_loss = F.cross_entropy(qs_pred, b_qs_labels)
        
        # Total Loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + 0.5 * qs_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        
    # Clear Buffer
    agent.reset()
    
    return {'loss': total_loss / config.PPO_EPOCHS}

if __name__ == "__main__":
    train()
