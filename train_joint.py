"""
Joint Training: Passing Network + Playing Network with PPO

This script trains both networks together:
1. PassingNetwork decides which 3 cards to pass
2. HeartsLSTM decides which card to play each turn
3. Both receive reward based on final game score
4. PPO updates both networks

Usage:
    python train_joint.py
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from game import GameV2
from model import HeartsLSTM
from passing_model import PassingNetwork
from agent import SimpleFCNAgent
from passing_agent import PassingAgent
from data_structure import PassDirection
from strategies import ExpertPolicy
import random
import json
import os
from collections import deque
from tqdm import tqdm
import config


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    # rewards and values should be same length
    # We treat terminal state value as 0
    next_value = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]
    
    return advantages


def train_joint():
    """
    Joint training of PassingNetwork and HeartsLSTM using PPO.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Joint Training on {device}")
    
    # Initialize models
    play_model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
    pass_model = PassingNetwork(hidden_dim=256).to(device)
    
    # Optimizers
    play_optimizer = optim.Adam(play_model.parameters(), lr=config.LR)
    pass_optimizer = optim.Adam(pass_model.parameters(), lr=config.LR * 0.5)  # Lower LR for passing
    
    # Load pretrained weights (from joint pretraining)
    start_episode = 0
    
    if os.path.exists(config.PRETRAINED_MODEL_PATH):
        try:
            checkpoint = torch.load(config.PRETRAINED_MODEL_PATH)
            play_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained playing model (acc={checkpoint.get('accuracy', 'N/A'):.4f})")
        except Exception as e:
            print(f"Could not load playing model: {e}")
    else:
        print("WARNING: No pretrained playing model found. Run pretrain_joint.py first!")
    
    if os.path.exists(config.PASSING_PRETRAINED_PATH):
        try:
            checkpoint = torch.load(config.PASSING_PRETRAINED_PATH)
            pass_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained passing model (acc={checkpoint.get('accuracy', 'N/A'):.4f})")
        except Exception as e:
            print(f"Could not load passing model: {e}")
    else:
        print("WARNING: No pretrained passing model found. Run pretrain_joint.py first!")
    
    # Create agents
    play_agent = SimpleFCNAgent(play_model, device)
    pass_agent = PassingAgent(pass_model, device)
    
    # Link them together!
    play_agent.set_passing_agent(pass_agent)
    
    game = GameV2()
    
    # Logging
    log_data = {'episodes': [], 'scores': [], 'pass_losses': [], 'play_losses': []}
    score_window = deque(maxlen=100)
    
    # Training parameters
    batch_size = 128  # Collect this many games before update (increased from 32)
    total_episodes = config.TOTAL_EPISODES
    
    print(f"\nStarting joint training for {total_episodes} episodes...")
    print(f"Batch size: {batch_size} games per update")
    
    for episode in range(start_episode, total_episodes):
        # Reset agents for new batch
        if episode % batch_size == 0:
            play_agent.reset()
            pass_agent.reset()
        
        # Choose pass direction - EXCLUDE KEEP to always have passing data!
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS])
        
        # Define policies
        def agent_play_policy(player, info, legal, order):
            return play_agent.act(player, info, legal, order, training=True)
        
        def agent_pass_policy(player, info):
            return play_agent.pass_policy_training(player, info)
        
        # Opponents (curriculum learning)
        progress = episode / total_episodes
        if progress < 0.25:
            opp_policies = [ExpertPolicy.play_policy] * 3
            opp_pass = [ExpertPolicy.pass_policy] * 3
        elif progress < 0.5:
            # Mix
            if random.random() < 0.5:
                opp_policies = [ExpertPolicy.play_policy] * 3
            else:
                opp_policies = [lambda p,i,l,o: random.choice(l)] * 3
            opp_pass = [ExpertPolicy.pass_policy] * 3
        else:
            opp_policies = [ExpertPolicy.play_policy] * 3
            opp_pass = [ExpertPolicy.pass_policy] * 3
        
        policies = [agent_play_policy] + opp_policies
        pass_policies = [agent_pass_policy] + opp_pass
        
        # Reset LSTM episode memory
        play_agent.reset_episode_memory()
        
        # Run game
        scores, trick_rewards, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        
        my_score = scores[0]
        score_window.append(my_score)
        
        # Compute reward for passing (based on final score)
        # Passing happens once at start, so it gets full game reward
        if pass_dir != PassDirection.KEEP:
            pass_reward = -my_score / 26.0  # Normalize to roughly [-1, 0]
            pass_agent.add_reward(pass_reward)
        
        # Compute rewards for playing
        # Each action in the game gets the trick reward
        for r in trick_rewards[0]:
            play_agent.rewards.append(r / 26.0)  # Normalize
        
        # PPO Update every batch_size episodes
        if (episode + 1) % batch_size == 0 and episode > 0:
            # === Update Playing Network ===
            play_loss = update_play_network(play_agent, play_model, play_optimizer, device)
            
            # === Update Passing Network ===
            pass_loss = update_pass_network(pass_agent, pass_model, pass_optimizer, device)
            
            # Logging
            avg_score = sum(score_window) / len(score_window)
            
            if (episode + 1) % (batch_size * 2) == 0:
                print(f"Ep {episode+1} | Score: {avg_score:.2f} | "
                      f"PlayLoss: {play_loss:.4f} | PassLoss: {pass_loss:.4f}")
            
            log_data['episodes'].append(episode + 1)
            log_data['scores'].append(avg_score)
            log_data['play_losses'].append(play_loss)
            log_data['pass_losses'].append(pass_loss)
            
            # Reset for next batch
            play_agent.reset()
            pass_agent.reset()
        
        # Save periodically
        if (episode + 1) % 5000 == 0:
            save_models(play_model, pass_model, play_optimizer, pass_optimizer, episode, log_data)
    
    # Final save
    save_models(play_model, pass_model, play_optimizer, pass_optimizer, total_episodes, log_data)
    print("\nTraining complete!")


def update_play_network(agent, model, optimizer, device):
    """PPO update for playing network using REINFORCE with baseline."""
    if len(agent.saved_actions) == 0:
        return 0.0
    
    # Get stored data
    old_log_probs = torch.stack(agent.saved_log_probs)  # Keep gradients!
    old_values = torch.stack(agent.saved_values)
    rewards = torch.tensor(agent.rewards, device=device, dtype=torch.float32)
    
    # Pad rewards if needed
    n_actions = len(old_log_probs)
    if len(rewards) < n_actions:
        rewards = F.pad(rewards, (0, n_actions - len(rewards)))
    elif len(rewards) > n_actions:
        rewards = rewards[:n_actions]
    
    # Compute returns (simple: just use the rewards directly, or cumulative)
    # For Hearts, reward is given per trick, so we can use it directly
    returns = rewards
    
    # Compute advantages (reward - baseline)
    with torch.no_grad():
        advantages = returns - old_values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Policy gradient loss (REINFORCE with baseline)
    policy_loss = -(old_log_probs * advantages).mean()
    
    # Value loss
    value_loss = F.mse_loss(old_values, returns)
    
    # Total loss
    loss = policy_loss + 0.5 * value_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    return loss.item()


def update_pass_network(agent, model, optimizer, device):
    """PPO update for passing network."""
    data = agent.get_training_data()
    if data is None:
        print(f"[DEBUG] PassNetwork: data is None, saved_actions len = {len(agent.saved_actions)}")
        return 0.0
    
    hand_vecs = data['hand_vecs']
    pass_dir_vecs = data['pass_dir_vecs']
    hand_masks = data['hand_masks']
    actions = data['actions']
    old_log_probs = data['old_log_probs'].detach()  # [N, 3]
    old_values = data['old_values'].detach()        # [N, 3]
    rewards = data['rewards']                        # [N]
    
    print(f"[DEBUG] PassNetwork: actions={len(actions)}, rewards={len(rewards)}")
    
    # Safety check: skip if any NaN
    if torch.isnan(old_log_probs).any() or torch.isnan(rewards).any():
        print("[DEBUG] PassNetwork: NaN detected!")
        return 0.0
    
    # Safety check: need at least 1 sample
    if len(rewards) == 0:
        print("[DEBUG] PassNetwork: rewards is empty!")
        return 0.0
    
    # Each passing decision gets the full game reward (expanded to 3 steps)
    rewards_expanded = rewards.unsqueeze(1).expand(-1, 3).float().to(device)  # [N, 3]
    
    # Compute advantages (simplified: reward - baseline)
    advantages = rewards_expanded - old_values
    returns = rewards_expanded
    
    # Normalize
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_loss = 0
    for _ in range(config.PPO_EPOCHS):
        # Re-evaluate actions to get new log_probs with gradients
        new_log_probs, new_values, entropy = model.evaluate_actions(
            hand_vecs, pass_dir_vecs, hand_masks, actions
        )
        
        # PPO ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - config.CLIP_EPS, 1 + config.CLIP_EPS) * advantages.detach()
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns.detach())
        entropy_loss = -entropy.mean()
        
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # Update old_log_probs for next PPO epoch
        old_log_probs = new_log_probs.detach()
        
        total_loss += loss.item()
    
    return total_loss / config.PPO_EPOCHS


def save_models(play_model, pass_model, play_opt, pass_opt, episode, log_data):
    """Save both models."""
    torch.save({
        'episode': episode,
        'model_state_dict': play_model.state_dict(),
        'optimizer_state_dict': play_opt.state_dict(),
    }, config.MODEL_PATH)
    
    torch.save({
        'episode': episode,
        'model_state_dict': pass_model.state_dict(),
        'optimizer_state_dict': pass_opt.state_dict(),
    }, config.PASSING_MODEL_PATH)
    
    with open(config.LOG_FILE, 'w') as f:
        json.dump(log_data, f)
    
    print(f"  Saved models at episode {episode}")


def evaluate_joint():
    """Evaluate joint model performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    play_model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
    pass_model = PassingNetwork(hidden_dim=256).to(device)
    
    if os.path.exists(config.MODEL_PATH):
        checkpoint = torch.load(config.MODEL_PATH)
        play_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded playing model")
    
    if os.path.exists(config.PASSING_MODEL_PATH):
        checkpoint = torch.load(config.PASSING_MODEL_PATH)
        pass_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded passing model")
    
    play_model.eval()
    pass_model.eval()
    
    play_agent = SimpleFCNAgent(play_model, device)
    pass_agent = PassingAgent(pass_model, device)
    play_agent.set_passing_agent(pass_agent)
    
    game = GameV2()
    
    # Evaluate vs Expert
    print("\n===== vs Expert =====")
    scores_vs_expert = []
    for _ in tqdm(range(200)):
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, 
                                   PassDirection.ACROSS, PassDirection.KEEP])
        
        def agent_policy(p, i, l, o):
            return play_agent.act(p, i, l, o, training=False)
        
        policies = [agent_policy] + [ExpertPolicy.play_policy] * 3
        pass_policies = [play_agent.pass_policy] + [ExpertPolicy.pass_policy] * 3
        
        play_agent.reset_episode_memory()
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        scores_vs_expert.append(scores[0])
    
    print(f"vs Expert: Avg Score = {sum(scores_vs_expert)/len(scores_vs_expert):.2f}")
    
    # Evaluate vs Random
    print("\n===== vs Random =====")
    scores_vs_random = []
    for _ in tqdm(range(200)):
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, 
                                   PassDirection.ACROSS, PassDirection.KEEP])
        
        def agent_policy(p, i, l, o):
            return play_agent.act(p, i, l, o, training=False)
        def random_policy(p, i, l, o):
            return random.choice(l)
        def random_pass(p, i):
            return random.sample(list(p.hand), 3)
        
        policies = [agent_policy] + [random_policy] * 3
        pass_policies = [play_agent.pass_policy] + [random_pass] * 3
        
        play_agent.reset_episode_memory()
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        scores_vs_random.append(scores[0])
    
    print(f"vs Random: Avg Score = {sum(scores_vs_random)/len(scores_vs_random):.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        evaluate_joint()
    else:
        train_joint()
