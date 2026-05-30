import torch
import torch.optim as optim
import torch.nn.functional as F
from game import GameV2
from model import HeartsLSTM
from agent import SimpleFCNAgent
from data_structure import PassDirection
import random
import json
import os
import config
from collections import deque
from strategies import ExpertPolicy

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LSTM on {device}")
    
    # We use batch_first=True, so State is [Batch, Seq, Dim]
    model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    start_episode = 0
    best_score = float('inf')
    
    if os.path.exists(config.MODEL_PATH):
        try:
            checkpoint = torch.load(config.MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            best_score = checkpoint.get('best_score', float('inf'))
            print(f"Resuming from ep {start_episode}")
        except:
            print("Starting fresh.")

    agent = SimpleFCNAgent(model, device)
    game = GameV2()
    
    # Data Logging
    log_data = {'episodes': [], 'scores': [], 'rewards': [], 'losses': [], 'value_losses': [], 'entropies': []}
    if start_episode > 0 and os.path.exists(config.LOG_FILE):
        try:
            with open(config.LOG_FILE, 'r') as f:
                log_data = json.load(f)
        except:
            pass

    score_window = deque(maxlen=100)
    reward_window = deque(maxlen=100)
    episode_step_counts = []  # Track steps per episode for GAE
    
    for episode in range(start_episode, config.TOTAL_EPISODES):
        n_actions_before = len(agent.saved_actions)

        # Policy Wrappers
        def agent_policy_wrapper(player, info, legal, order):
            return agent.act(player, info, legal, order, training=True)
            
        def random_policy(player, info, legal, order):
            return random.choice(legal)
        
        def expert_policy_wrapper(player, info, legal, order):
            return ExpertPolicy.play_policy(player, info, legal, order)
        
        # --- 改进的对手池策略 (针对强Rule-based对手训练) ---
        # 关键：尽早暴露给Expert对手，因为目标是打败Expert
        # 但保留少量随机对手用于探索多样性
        rand_val = random.random()
        opponents = []
        
        if episode < 2000:
            # 初期热身：50% Random, 50% Expert (快速适应)
            if rand_val < 0.5:
                opponents = [random_policy, random_policy, random_policy]
            else:
                opponents = [expert_policy_wrapper, expert_policy_wrapper, expert_policy_wrapper]
        elif episode < 10000:
            # 过渡期：20% Random, 80% Expert
            if rand_val < 0.2:
                opponents = [random_policy, random_policy, random_policy]
            else:
                opponents = [expert_policy_wrapper, expert_policy_wrapper, expert_policy_wrapper]
        else:
            # 主训练期：5% Random (保持探索), 95% Expert
            if rand_val < 0.05:
                opponents = [random_policy, random_policy, random_policy]
            else:
                opponents = [expert_policy_wrapper, expert_policy_wrapper, expert_policy_wrapper]

        random.shuffle(opponents)
        policies = [agent_policy_wrapper] + opponents
        
        pass_policies = [agent.pass_policy, ExpertPolicy.pass_policy, ExpertPolicy.pass_policy, ExpertPolicy.pass_policy]
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        # Reset Agent's Short-term Memory for the new game
        agent.reset_episode_memory()
        
        scores, trick_rewards_history, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        
        # ========== 改进的Reward设计 (对抗强Rule-based对手) ==========
        # 目标: 比对手得分更低才是胜利，需要相对评估而非绝对分数
        
        my_score = scores[0]
        opp_scores = scores[1:]
        opp_avg = sum(opp_scores) / 3.0
        opp_min = min(opp_scores)
        opp_max = max(opp_scores)
        
        # 1. 相对优势奖励 (核心指标)
        # 正值=我比对手平均好, 负值=我比对手平均差
        relative_advantage = (opp_avg - my_score) / 13.0  # 缩放到[-2, 2]范围
        
        # 2. 排名奖励
        my_rank = sum(1 for s in opp_scores if s < my_score)  # 0=最好, 3=最差
        rank_reward = (1.5 - my_rank) * 0.3  # +0.45(第1) to -0.45(第4)
        
        # 3. 避免灾难惩罚 (吃SQ或大量红心)
        disaster_penalty = 0.0
        if my_score >= 13:  # 吃了SQ或很多红心
            disaster_penalty = -0.5
        if my_score >= 20:  # 差点被Shoot the Moon
            disaster_penalty = -1.0
        
        # 4. Shoot the Moon相关
        stm_bonus = 0.0
        if my_score == 0 and 26 in opp_scores:
            stm_bonus = 3.0  # 成功STM
        elif 26 in opp_scores and my_score > 0:
            stm_bonus = -2.0  # 被STM了
        elif my_score == 26:
            stm_bonus = -3.0  # 我STM失败
        
        terminal_reward = relative_advantage + rank_reward + disaster_penalty + stm_bonus
        
        # Assign Rewards to Actions
        n_actions_after = len(agent.saved_actions)
        steps_total = n_actions_after - n_actions_before
        
        my_trick_rewards = trick_rewards_history[0]
        rewards_to_assign = []
        num_tricks = len(my_trick_rewards)
        
        # 累积分数用于计算中间奖励
        cumulative_score = 0
        
        for i in range(num_tricks):
            # 1. 即时惩罚 (吃分)
            points_taken = -my_trick_rewards[i]  # 转为正数
            cumulative_score += points_taken
            
            # 更强的即时信号
            step_r = my_trick_rewards[i] / 6.5  # 放大信号 (原来是/13)
            
            # 2. SQ特别惩罚
            if points_taken >= 13:  # 吃了SQ
                step_r -= 0.5  # 额外惩罚
            
            # 3. 安全通过奖励 (没吃分)
            if points_taken == 0:
                step_r += 0.02  # 小奖励鼓励安全
            
            # 4. 终局奖励分散到最后3步 (更好的信用分配)
            if i >= num_tricks - 3:
                step_r += terminal_reward / 3.0
            
            rewards_to_assign.append(step_r)
        
        # Track steps for this episode (for GAE computation)
        episode_step_counts.append(steps_total)
            
        # Safety check: Match lengths
        if len(rewards_to_assign) != steps_total:
            if len(rewards_to_assign) < steps_total:
                rewards_to_assign.extend([0] * (steps_total - len(rewards_to_assign)))
            else:
                rewards_to_assign = rewards_to_assign[:steps_total]
            
        agent.rewards.extend(rewards_to_assign)
        
        score_window.append(my_score)
        reward_window.append(sum(rewards_to_assign)) # Log total reward for episode
        
        avg_score = sum(score_window) / len(score_window)
        avg_reward = sum(reward_window) / len(reward_window)
        
        if (episode + 1) % config.BATCH_SIZE == 0:
            loss_info = update_ppo(agent, optimizer, episode_step_counts)
            episode_step_counts = []  # Reset for next batch
            
            # Update Log
            log_data['episodes'].append(episode + 1)
            log_data['scores'].append(avg_score)
            log_data['rewards'].append(avg_reward)
            log_data['losses'].append(loss_info['loss'])
            log_data['value_losses'].append(loss_info['value_loss'])
            log_data['entropies'].append(loss_info['entropy'])
            
            # Save Log to file
            try:
                with open(config.LOG_FILE, 'w') as f:
                    json.dump(log_data, f)
            except Exception as e:
                print(f"Log save error: {e}")

            print(f"Ep {episode+1} | Score: {avg_score:.2f} | VLoss: {loss_info['value_loss']:.3f} | Entropy: {loss_info['entropy']:.3f}")
            
            if (episode + 1) % 100 == 0:
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score
                }, config.MODEL_PATH)

def compute_gae_returns(rewards, values, episode_ends, gamma=0.99, lam=0.95, device='cpu'):
    """
    Compute GAE (Generalized Advantage Estimation) and discounted returns.
    
    Args:
        rewards: [N] tensor of rewards
        values: [N] tensor of value estimates
        episode_ends: [M] list of indices where episodes end (exclusive, cumulative step counts)
        gamma: discount factor
        lam: GAE lambda
        device: torch device
    
    Returns:
        advantages: [N] tensor
        returns: [N] tensor (value targets)
    """
    N = len(rewards)
    advantages = torch.zeros(N, device=device)
    returns = torch.zeros(N, device=device)
    
    # Convert episode_ends to episode boundaries
    # episode_ends = [13, 26, 39, ...] means steps 0-12 are ep1, 13-25 are ep2, etc.
    
    ep_start = 0
    for ep_end in episode_ends:
        ep_end = min(ep_end, N)  # Safety
        
        if ep_start >= ep_end:
            continue
            
        # Process this episode
        gae = 0
        # Go backwards within episode
        for t in reversed(range(ep_start, ep_end)):
            is_last = (t == ep_end - 1)
            
            if is_last:
                next_value = 0  # Terminal state, no future value
            else:
                next_value = values[t + 1]
            
            # TD error: r + gamma * V(s') - V(s)
            delta = rewards[t] + gamma * next_value - values[t]
            
            # GAE: delta + gamma * lam * gae
            gae = delta + gamma * lam * (0 if is_last else gae)
            
            advantages[t] = gae
            returns[t] = gae + values[t]  # Return = Advantage + Value
        
        ep_start = ep_end
    
    return advantages, returns


def update_ppo(agent, optimizer, episode_step_counts):
    """
    PPO update with proper GAE advantage estimation.
    
    Args:
        agent: the agent with collected experiences
        optimizer: the optimizer
        episode_step_counts: list of step counts per episode (e.g., [13, 13, 13, ...])
    """
    if not agent.saved_state_seqs:
        return {'loss': 0.0}

    # Stack sequences: [DatasetSize, SeqLen, Dim]
    max_len = 14
    lengths = []
    padded_seqs = []
    
    for seq in agent.saved_state_seqs:
        if seq.device != agent.device:
             seq = seq.to(agent.device)

        L = seq.size(1)
        lengths.append(min(L, max_len))
        if L < max_len:
            pad_size = max_len - L
            padding = torch.zeros((1, pad_size, config.INPUT_DIM), device=agent.device)
            padded_seq = torch.cat([seq, padding], dim=1)
            padded_seqs.append(padded_seq)
        elif L > max_len:
            padded_seqs.append(seq[:, :max_len, :])
        else:
            padded_seqs.append(seq)
            
    b_states = torch.cat(padded_seqs, dim=0)
    b_lengths = torch.tensor(lengths, dtype=torch.long, device=agent.device)

    b_global_priv = torch.stack(agent.saved_global_priv)
    b_masks = torch.stack(agent.saved_masks)
    b_actions = torch.stack(agent.saved_actions)
    b_log_probs = torch.stack(agent.saved_log_probs).detach()
    b_qs_labels = torch.stack(agent.saved_qs_labels)
    
    b_rewards = torch.tensor(agent.rewards, dtype=torch.float32).to(agent.device)
    b_old_values = torch.stack(agent.saved_values).detach()
    
    # Build episode boundaries for GAE
    episode_ends = []
    cumsum = 0
    for count in episode_step_counts:
        cumsum += count
        episode_ends.append(cumsum)
    
    # Compute proper GAE advantages and returns
    raw_advantages, b_returns = compute_gae_returns(
        b_rewards, b_old_values, episode_ends, 
        gamma=config.GAMMA, lam=0.95, device=agent.device
    )
    
    # Normalize Advantage (across the whole batch)
    if raw_advantages.std() > 1e-5:
        advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-5)
    else:
        advantages = raw_advantages
    
    dataset_size = b_states.size(0)
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    
    # Reset agent memory for next rollout
    agent.reset() 
    
    for _ in range(config.PPO_EPOCHS):
        # Full batch update
        # We pass hidden=None so LSTM starts fresh, which is an approximation 
        # but acceptable since we provide full history sequence in b_states.
        logits, values, qs_pred, _ = agent.model(b_states, b_global_priv, lengths=b_lengths)
        values = values.squeeze()
        
        masked_logits = logits + b_masks
        probs = torch.softmax(masked_logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        
        new_log_probs = dist.log_prob(b_actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - b_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values, b_returns)
        qs_loss = F.cross_entropy(qs_pred, b_qs_labels)
        
        loss = policy_loss + 0.5 * value_loss - config.ENTROPY_COEF * entropy + 0.5 * qs_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        
    n_epochs = config.PPO_EPOCHS
    return {
        'loss': total_loss / n_epochs,
        'policy_loss': total_policy_loss / n_epochs,
        'value_loss': total_value_loss / n_epochs,
        'entropy': total_entropy / n_epochs
    }

if __name__ == "__main__":
    train()
