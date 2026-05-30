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
import sys
import time
from collections import deque
from tqdm import tqdm
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def set_seed(seed=config.SEED):
    """Seed Python and torch so training/eval runs are easier to compare."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_terminal_reward(scores, player_id=0):
    """Reward a player by final relative score, not just absolute points."""
    my_score = scores[player_id]
    opp_scores = [score for idx, score in enumerate(scores) if idx != player_id]
    opp_avg = sum(opp_scores) / len(opp_scores)

    relative_advantage = (opp_avg - my_score) / 13.0
    my_rank = sum(1 for score in opp_scores if score < my_score)
    rank_reward = (1.5 - my_rank) * 0.3

    disaster_penalty = 0.0
    if my_score >= 13:
        disaster_penalty = -0.5
    if my_score >= 20:
        disaster_penalty = -1.0

    stm_bonus = 0.0
    if my_score == 0 and 26 in opp_scores:
        stm_bonus = 3.0
    elif 26 in opp_scores and my_score > 0:
        stm_bonus = -2.0
    elif my_score == 26:
        stm_bonus = -3.0

    return relative_advantage + rank_reward + disaster_penalty + stm_bonus


def build_play_rewards(trick_rewards, terminal_reward, n_actions):
    """Blend trick-level penalties with a terminal reward at the episode end."""
    rewards = []
    for reward in trick_rewards:
        step_reward = reward / 6.5
        points_taken = -reward
        if points_taken >= 13:
            step_reward -= 0.5
        if points_taken == 0:
            step_reward += 0.02
        rewards.append(step_reward)

    if rewards:
        terminal_steps = min(3, len(rewards))
        for idx in range(len(rewards) - terminal_steps, len(rewards)):
            rewards[idx] += terminal_reward / terminal_steps

    if len(rewards) < n_actions:
        rewards.extend([0.0] * (n_actions - len(rewards)))
    return rewards[:n_actions]


def compute_gae_returns(rewards, values, episode_step_counts, gamma=None, lam=None, device="cpu"):
    """Compute GAE separately inside each game boundary."""
    gamma = config.GAMMA if gamma is None else gamma
    lam = config.GAE_LAMBDA if lam is None else lam
    advantages = torch.zeros_like(rewards, device=device)
    returns = torch.zeros_like(rewards, device=device)

    ep_start = 0
    for count in episode_step_counts:
        ep_end = min(ep_start + count, rewards.numel())
        if ep_start >= ep_end:
            continue

        gae = torch.tensor(0.0, device=device)
        for t in reversed(range(ep_start, ep_end)):
            next_value = torch.tensor(0.0, device=device) if t == ep_end - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        ep_start = ep_end

    return advantages, returns


def explained_variance(predictions, targets):
    target_var = torch.var(targets)
    if target_var < 1e-8:
        return 0.0
    return (1.0 - torch.var(targets - predictions) / target_var).item()


def empty_loss_info():
    return {
        'loss': 0.0,
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'bc_loss': 0.0,
        'entropy': 0.0,
        'clip_fraction': 0.0,
        'explained_variance': 0.0,
    }


def load_model_state(model, checkpoint_path, device, label):
    if not os.path.exists(checkpoint_path):
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded {label} model from {checkpoint_path}")
        return True
    except RuntimeError as exc:
        reason = str(exc).splitlines()[0]
        print(f"Skipping incompatible {label} checkpoint {checkpoint_path}: {reason}")
        return False


def eval_selection_score(eval_summary):
    """Lower is better: player-0 average score against Expert opponents."""
    if not eval_summary:
        return float('inf')
    return float(eval_summary.get('vs_expert_avg_score', float('inf')))


def load_checkpoint_eval_score(checkpoint_path):
    """Read a checkpoint's stored eval metric without loading model weights into a live model."""
    if not os.path.exists(checkpoint_path):
        return float('inf')
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as exc:
        print(f"Could not inspect checkpoint {checkpoint_path}: {exc}")
        return float('inf')
    return eval_selection_score(checkpoint.get('eval'))


def metric_slug(value):
    """Make a metric safe for filenames while preserving readable precision."""
    if value == float('inf'):
        return "inf"
    return f"{value:.3f}".replace(".", "p")


def build_eval_snapshot_paths(eval_summary, episode):
    """Create unique play/pass snapshot paths for every fixed-seed eval."""
    expert_score = metric_slug(eval_summary['vs_expert_avg_score'])
    random_score = metric_slug(eval_summary['vs_random_avg_score'])
    base_name = f"ppo_ep{episode:06d}_expert{expert_score}_random{random_score}"
    return (
        os.path.join(config.PPO_SNAPSHOT_DIR, f"{base_name}_play.pth"),
        os.path.join(config.PPO_SNAPSHOT_DIR, f"{base_name}_pass.pth"),
    )


def save_eval_snapshot(play_model, pass_model, play_opt, pass_opt, episode, log_data, eval_summary, ppo_mode):
    """Always preserve eval checkpoints so later training cannot erase useful candidates."""
    play_path, pass_path = build_eval_snapshot_paths(eval_summary, episode)
    save_models(
        play_model,
        pass_model,
        play_opt,
        pass_opt,
        episode,
        log_data,
        model_path=play_path,
        pass_model_path=pass_path,
        metadata={
            'checkpoint_type': 'eval_snapshot',
            'selection_metric': 'vs_expert_avg_score',
            'eval': eval_summary,
            'ppo_mode': ppo_mode,
        },
    )
    print(f"  Saved eval snapshot: {os.path.basename(play_path)}")


def save_best_if_improved(
    play_model,
    pass_model,
    play_opt,
    pass_opt,
    episode,
    log_data,
    eval_summary,
    best_eval_score,
    ppo_mode,
):
    """Only overwrite best checkpoint if it beats both in-memory and on-disk best."""
    current_score = eval_selection_score(eval_summary)
    disk_best_score = load_checkpoint_eval_score(config.BEST_MODEL_PATH)
    score_to_beat = min(best_eval_score, disk_best_score)

    if current_score >= score_to_beat:
        print(
            f"  Eval did not improve best: {current_score:.2f} "
            f"(best {score_to_beat:.2f})"
        )
        return score_to_beat

    save_models(
        play_model,
        pass_model,
        play_opt,
        pass_opt,
        episode,
        log_data,
        model_path=config.BEST_MODEL_PATH,
        pass_model_path=config.PASSING_BEST_MODEL_PATH,
        metadata={
            'checkpoint_type': 'best',
            'selection_metric': 'vs_expert_avg_score',
            'eval': eval_summary,
            'ppo_mode': ppo_mode,
        },
    )
    print(f"  New best PPO checkpoint: vs Expert Avg Score = {current_score:.2f}")
    return current_score


def summarize_score_history(score_history):
    """Summarize player-0 performance from full four-player score rows."""
    if not score_history:
        return {
            'avg_score': 0.0,
            'avg_rank': 0.0,
            'win_rate': 0.0,
            'top2_rate': 0.0,
        }

    my_scores = [scores[0] for scores in score_history]
    ranks = [1 + sum(1 for score in scores[1:] if score < scores[0]) for scores in score_history]
    wins = sum(1 for rank in ranks if rank == 1)
    top2 = sum(1 for rank in ranks if rank <= 2)

    return {
        'avg_score': sum(my_scores) / len(my_scores),
        'avg_rank': sum(ranks) / len(ranks),
        'win_rate': wins / len(ranks),
        'top2_rate': top2 / len(ranks),
    }


def print_eval_summary(label, summary):
    print(
        f"{label}: Avg Score = {summary['avg_score']:.2f} | "
        f"Avg Rank = {summary['avg_rank']:.2f} | "
        f"Win Rate = {summary['win_rate']:.1%} | "
        f"Top-2 = {summary['top2_rate']:.1%}"
    )


def train_joint():
    """
    Joint training of PassingNetwork and HeartsLSTM using PPO.
    """
    set_seed()
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
    log_data = {
        'episodes': [],
        'scores': [],
        'rewards': [],
        'play_losses': [],
        'play_policy_losses': [],
        'play_value_losses': [],
        'play_bc_losses': [],
        'play_entropies': [],
        'play_clip_fractions': [],
        'play_explained_variance': [],
        'pass_losses': [],
        'pass_policy_losses': [],
        'pass_value_losses': [],
        'pass_entropies': [],
        'pass_clip_fractions': [],
        'eval': [],
    }
    score_window = deque(maxlen=100)
    reward_window = deque(maxlen=100)
    episode_step_counts = []
    best_eval_score = load_checkpoint_eval_score(config.BEST_MODEL_PATH)
    next_eval_episode = config.PPO_EVAL_INTERVAL
    
    # Training parameters
    batch_size = config.BATCH_SIZE
    total_episodes = config.TOTAL_EPISODES
    
    train_start_time = time.time()
    max_minutes = getattr(config, "PPO_MAX_MINUTES", None)
    stop_reason = "reached total episodes"
    print(f"\nStarting joint training for {total_episodes} episodes...")
    print(f"Batch size: {batch_size} games per update")
    print(
        "PPO mode: "
        f"{'play+pass' if config.PPO_TRAIN_PASSING else 'play-only'} | "
        f"{'Expert' if config.PPO_USE_EXPERT_PASSING else 'learned'} passing for player 0 | "
        f"BC anchor coef={config.BC_ANCHOR_COEF}"
    )

    print("\nEvaluating pretrained starting point before PPO...")
    initial_eval = evaluate_for_training_checkpoint(play_model, pass_model, device)
    initial_eval['episode'] = start_episode
    log_data['eval'].append(initial_eval)
    save_eval_snapshot(
        play_model,
        pass_model,
        play_optimizer,
        pass_optimizer,
        start_episode,
        log_data,
        initial_eval,
        'pretrained_start',
    )
    best_eval_score = save_best_if_improved(
        play_model,
        pass_model,
        play_optimizer,
        pass_optimizer,
        start_episode,
        log_data,
        initial_eval,
        best_eval_score,
        'pretrained_start',
    )
    print(f"  Starting best checkpoint: vs Expert Avg Score = {best_eval_score:.2f}")
    
    for episode in range(start_episode, total_episodes):
        if max_minutes is not None and max_minutes > 0:
            elapsed_minutes = (time.time() - train_start_time) / 60.0
            if elapsed_minutes >= max_minutes:
                stop_reason = f"hit wall-clock limit ({max_minutes} min)"
                print(f"\nStopping early: {stop_reason}")
                break

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
            if config.PPO_USE_EXPERT_PASSING:
                return ExpertPolicy.pass_policy(player, info)
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
        
        n_actions_before = len(play_agent.saved_actions)

        # Run game
        scores, trick_rewards, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        
        my_score = scores[0]
        terminal_reward = compute_terminal_reward(scores, player_id=0)
        score_window.append(my_score)
        reward_window.append(terminal_reward)
        
        # Passing happens once at the start and optimizes the same terminal objective.
        if config.PPO_TRAIN_PASSING and not config.PPO_USE_EXPERT_PASSING and pass_dir != PassDirection.KEEP:
            pass_agent.add_reward(terminal_reward)
        
        n_actions_after = len(play_agent.saved_actions)
        steps_this_episode = n_actions_after - n_actions_before
        episode_step_counts.append(steps_this_episode)
        play_agent.rewards.extend(build_play_rewards(
            trick_rewards[0],
            terminal_reward,
            steps_this_episode
        ))

        if (episode + 1) % config.PPO_PROGRESS_INTERVAL == 0:
            avg_score = sum(score_window) / len(score_window)
            avg_reward = sum(reward_window) / len(reward_window)
            print(
                f"Ep {episode+1} rollout | Score: {avg_score:.2f} | "
                f"Reward: {avg_reward:.3f} | Steps: {steps_this_episode}"
            )
        
        # PPO Update every batch_size episodes
        if (episode + 1) % batch_size == 0 and episode > 0:
            # === Update Playing Network ===
            play_info = update_play_network(
                play_agent,
                play_model,
                play_optimizer,
                device,
                episode_step_counts
            )
            
            # === Update Passing Network ===
            if config.PPO_TRAIN_PASSING and not config.PPO_USE_EXPERT_PASSING:
                pass_info = update_pass_network(pass_agent, pass_model, pass_optimizer, device)
            else:
                pass_info = empty_loss_info()
            
            # Logging
            avg_score = sum(score_window) / len(score_window)
            avg_reward = sum(reward_window) / len(reward_window)
            
            if (episode + 1) % (batch_size * 2) == 0:
                print(f"Ep {episode+1} | Score: {avg_score:.2f} | "
                      f"Reward: {avg_reward:.3f} | "
                      f"PlayLoss: {play_info['loss']:.4f} | PassLoss: {pass_info['loss']:.4f}")
            
            log_data['episodes'].append(episode + 1)
            log_data['scores'].append(avg_score)
            log_data['rewards'].append(avg_reward)
            log_data['play_losses'].append(play_info['loss'])
            log_data['play_policy_losses'].append(play_info['policy_loss'])
            log_data['play_value_losses'].append(play_info['value_loss'])
            log_data['play_bc_losses'].append(play_info['bc_loss'])
            log_data['play_entropies'].append(play_info['entropy'])
            log_data['play_clip_fractions'].append(play_info['clip_fraction'])
            log_data['play_explained_variance'].append(play_info['explained_variance'])
            log_data['pass_losses'].append(pass_info['loss'])
            log_data['pass_policy_losses'].append(pass_info['policy_loss'])
            log_data['pass_value_losses'].append(pass_info['value_loss'])
            log_data['pass_entropies'].append(pass_info['entropy'])
            log_data['pass_clip_fractions'].append(pass_info['clip_fraction'])
            
            # Reset for next batch
            play_agent.reset()
            pass_agent.reset()
            episode_step_counts = []

            if episode + 1 >= next_eval_episode:
                eval_summary = evaluate_for_training_checkpoint(play_model, pass_model, device)
                eval_summary['episode'] = episode + 1
                log_data['eval'].append(eval_summary)
                while next_eval_episode <= episode + 1:
                    next_eval_episode += config.PPO_EVAL_INTERVAL

                ppo_mode = 'play_only_bc_anchor' if not config.PPO_TRAIN_PASSING else 'joint_ppo'
                save_eval_snapshot(
                    play_model,
                    pass_model,
                    play_optimizer,
                    pass_optimizer,
                    episode + 1,
                    log_data,
                    eval_summary,
                    ppo_mode,
                )
                best_eval_score = save_best_if_improved(
                    play_model,
                    pass_model,
                    play_optimizer,
                    pass_optimizer,
                    episode + 1,
                    log_data,
                    eval_summary,
                    best_eval_score,
                    ppo_mode,
                )
        
        # Save periodically
        if (episode + 1) % 5000 == 0:
            save_models(play_model, pass_model, play_optimizer, pass_optimizer, episode, log_data)
    
    # Final save
    final_episode = total_episodes
    if log_data['episodes']:
        final_episode = log_data['episodes'][-1]
    save_models(play_model, pass_model, play_optimizer, pass_optimizer, final_episode, log_data)
    print(f"\nTraining complete ({stop_reason}).")


def update_play_network(agent, model, optimizer, device, episode_step_counts):
    """PPO update for the playing network with fresh log-probs and GAE."""
    if len(agent.saved_actions) == 0:
        return empty_loss_info()

    n_actions = len(agent.saved_actions)
    max_len = max(seq.size(1) for seq in agent.saved_state_seqs)
    lengths = []
    padded_seqs = []
    for seq in agent.saved_state_seqs:
        seq = seq.to(device)
        seq_len = seq.size(1)
        lengths.append(seq_len)
        if seq_len < max_len:
            padding = torch.zeros((1, max_len - seq_len, config.INPUT_DIM), device=device)
            seq = torch.cat([seq, padding], dim=1)
        padded_seqs.append(seq)

    states = torch.cat(padded_seqs, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    global_priv = torch.stack(agent.saved_global_priv).to(device)
    masks = torch.stack(agent.saved_masks).to(device)
    actions = torch.stack(agent.saved_actions).to(device)
    expert_actions = torch.stack(agent.saved_expert_actions).to(device)
    old_log_probs = torch.stack(agent.saved_log_probs).detach().to(device)
    old_values = torch.stack(agent.saved_values).detach().to(device).view(-1)
    qs_labels = torch.stack(agent.saved_qs_labels).to(device)
    rewards = torch.tensor(agent.rewards, dtype=torch.float32, device=device)

    if rewards.numel() < n_actions:
        rewards = F.pad(rewards, (0, n_actions - rewards.numel()))
    elif rewards.numel() > n_actions:
        rewards = rewards[:n_actions]

    with torch.no_grad():
        advantages, returns = compute_gae_returns(
            rewards,
            old_values,
            episode_step_counts,
            device=device
        )
        if advantages.numel() > 1 and advantages.std() > 1e-5:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_bc_loss = 0.0
    total_entropy = 0.0
    total_clip_fraction = 0.0
    total_explained_variance = 0.0

    for _ in range(config.PPO_EPOCHS):
        logits, values, qs_pred, _ = model(states, global_priv, hidden=None, lengths=lengths)
        values = values.view(-1)
        masked_logits = logits + masks
        dist = torch.distributions.Categorical(logits=masked_logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        qs_loss = F.cross_entropy(qs_pred, qs_labels)
        bc_loss = F.cross_entropy(masked_logits, expert_actions)
        loss = (
            policy_loss
            + config.VALUE_COEF * value_loss
            - config.ENTROPY_COEF * entropy
            + config.AUX_COEF * qs_loss
            + config.BC_ANCHOR_COEF * bc_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        clip_fraction = ((ratio < 1.0 - config.CLIP_EPS) | (ratio > 1.0 + config.CLIP_EPS)).float().mean()
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_bc_loss += bc_loss.item()
        total_entropy += entropy.item()
        total_clip_fraction += clip_fraction.item()
        total_explained_variance += explained_variance(values.detach(), returns)

    n_epochs = config.PPO_EPOCHS
    return {
        'loss': total_loss / n_epochs,
        'policy_loss': total_policy_loss / n_epochs,
        'value_loss': total_value_loss / n_epochs,
        'bc_loss': total_bc_loss / n_epochs,
        'entropy': total_entropy / n_epochs,
        'clip_fraction': total_clip_fraction / n_epochs,
        'explained_variance': total_explained_variance / n_epochs,
    }


def update_pass_network(agent, model, optimizer, device):
    """PPO update for passing network."""
    data = agent.get_training_data()
    if data is None:
        return empty_loss_info()
    
    hand_vecs = data['hand_vecs'].to(device)
    pass_dir_vecs = data['pass_dir_vecs'].to(device)
    hand_masks = data['hand_masks'].to(device)
    actions = data['actions'].to(device)
    old_log_probs = data['old_log_probs'].detach().to(device)  # [N, 3]
    old_values = data['old_values'].detach().to(device)        # [N, 3]
    rewards = data['rewards'].to(device)                       # [N]
    
    # Safety check: skip if any NaN
    if torch.isnan(old_log_probs).any() or torch.isnan(rewards).any():
        return empty_loss_info()
    
    # Safety check: need at least 1 sample
    if len(rewards) == 0:
        return empty_loss_info()
    
    # Each passing decision gets the full game reward (expanded to 3 steps)
    rewards_expanded = rewards.unsqueeze(1).expand(-1, 3).float().to(device)  # [N, 3]
    
    # Compute advantages (simplified: reward - baseline)
    advantages = rewards_expanded - old_values
    returns = rewards_expanded
    
    # Normalize
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_clip_fraction = 0.0

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
        
        loss = policy_loss + config.VALUE_COEF * value_loss - config.ENTROPY_COEF * entropy.mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        clip_fraction = ((ratio < 1.0 - config.CLIP_EPS) | (ratio > 1.0 + config.CLIP_EPS)).float().mean()
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.mean().item()
        total_clip_fraction += clip_fraction.item()

    n_epochs = config.PPO_EPOCHS
    return {
        'loss': total_loss / n_epochs,
        'policy_loss': total_policy_loss / n_epochs,
        'value_loss': total_value_loss / n_epochs,
        'entropy': total_entropy / n_epochs,
        'clip_fraction': total_clip_fraction / n_epochs,
        'explained_variance': explained_variance(new_values.detach().view(-1), returns.view(-1)),
    }


def save_models(
    play_model,
    pass_model,
    play_opt,
    pass_opt,
    episode,
    log_data,
    model_path=None,
    pass_model_path=None,
    metadata=None,
):
    """Save both models."""
    model_path = model_path or config.MODEL_PATH
    pass_model_path = pass_model_path or config.PASSING_MODEL_PATH
    metadata = metadata or {}

    torch.save({
        'episode': episode,
        'model_state_dict': play_model.state_dict(),
        'optimizer_state_dict': play_opt.state_dict(),
        **metadata,
    }, model_path)
    
    torch.save({
        'episode': episode,
        'model_state_dict': pass_model.state_dict(),
        'optimizer_state_dict': pass_opt.state_dict(),
        **metadata,
    }, pass_model_path)
    
    with open(config.LOG_FILE, 'w') as f:
        json.dump(log_data, f)
    
    print(f"  Saved models at episode {episode}")


def evaluate_for_training_checkpoint(play_model, pass_model, device):
    """Run deterministic checkpoint eval without perturbing the training RNG stream."""
    python_rng_state = random.getstate()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = play_model.training
    pass_was_training = pass_model.training

    try:
        return evaluate_joint(
            play_model=play_model,
            pass_model=pass_model,
            device=device,
            num_games=config.EVAL_GAMES,
            seed=config.SEED,
            use_expert_passing_for_player0=config.PPO_USE_EXPERT_PASSING,
        )
    finally:
        random.setstate(python_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        play_model.train(was_training)
        pass_model.train(pass_was_training)


def evaluate_joint(
    play_model=None,
    pass_model=None,
    device=None,
    num_games=None,
    seed=None,
    use_expert_passing_for_player0=None,
):
    """Evaluate joint model performance against expert and random opponents."""
    if seed is not None:
        set_seed(seed)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_games = num_games or config.EVAL_GAMES

    if play_model is None:
        play_model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
        play_paths = [config.BEST_MODEL_PATH, config.MODEL_PATH, config.PRETRAINED_MODEL_PATH]
        if not any(load_model_state(play_model, path, device, 'playing') for path in play_paths):
            print("No compatible playing checkpoint found; evaluating current initialized model.")

    if pass_model is None:
        pass_model = PassingNetwork(hidden_dim=256).to(device)
        pass_paths = [config.PASSING_BEST_MODEL_PATH, config.PASSING_MODEL_PATH, config.PASSING_PRETRAINED_PATH]
        if not any(load_model_state(pass_model, path, device, 'passing') for path in pass_paths):
            print("No compatible passing checkpoint found; evaluating current initialized model.")
    
    play_model.eval()
    pass_model.eval()
    
    play_agent = SimpleFCNAgent(play_model, device)
    pass_agent = PassingAgent(pass_model, device)
    play_agent.set_passing_agent(pass_agent)
    use_expert_passing_for_player0 = (
        config.PPO_USE_EXPERT_PASSING
        if use_expert_passing_for_player0 is None
        else use_expert_passing_for_player0
    )
    
    game = GameV2()
    
    # Evaluate vs Expert
    print("\n===== vs Expert =====")
    scores_vs_expert = []
    for _ in tqdm(range(num_games)):
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, 
                                   PassDirection.ACROSS, PassDirection.KEEP])
        
        def agent_policy(p, i, l, o):
            return play_agent.act(p, i, l, o, training=False)
        
        policies = [agent_policy] + [ExpertPolicy.play_policy] * 3
        player0_pass_policy = ExpertPolicy.pass_policy if use_expert_passing_for_player0 else play_agent.pass_policy
        pass_policies = [player0_pass_policy] + [ExpertPolicy.pass_policy] * 3
        
        play_agent.reset_episode_memory()
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        scores_vs_expert.append(scores)
    
    expert_summary = summarize_score_history(scores_vs_expert)
    print_eval_summary("vs Expert", expert_summary)
    
    # Evaluate vs Random
    print("\n===== vs Random =====")
    scores_vs_random = []
    for _ in tqdm(range(num_games)):
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, 
                                   PassDirection.ACROSS, PassDirection.KEEP])
        
        def agent_policy(p, i, l, o):
            return play_agent.act(p, i, l, o, training=False)
        def random_policy(p, i, l, o):
            return random.choice(l)
        def random_pass(p, i):
            return random.sample(list(p.hand), 3)
        
        policies = [agent_policy] + [random_policy] * 3
        player0_pass_policy = ExpertPolicy.pass_policy if use_expert_passing_for_player0 else play_agent.pass_policy
        pass_policies = [player0_pass_policy] + [random_pass] * 3
        
        play_agent.reset_episode_memory()
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        scores_vs_random.append(scores)
    
    random_summary = summarize_score_history(scores_vs_random)
    print_eval_summary("vs Random", random_summary)
    return {
        'vs_expert_avg_score': expert_summary['avg_score'],
        'vs_expert_avg_rank': expert_summary['avg_rank'],
        'vs_expert_win_rate': expert_summary['win_rate'],
        'vs_expert_top2_rate': expert_summary['top2_rate'],
        'vs_random_avg_score': random_summary['avg_score'],
        'vs_random_avg_rank': random_summary['avg_rank'],
        'vs_random_win_rate': random_summary['win_rate'],
        'vs_random_top2_rate': random_summary['top2_rate'],
        'num_games': num_games,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        evaluate_joint(seed=config.SEED)
    else:
        train_joint()
