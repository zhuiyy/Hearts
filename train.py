import torch
import torch.optim as optim
import random
import numpy as np
import json
import os
from typing import List, Optional
from game import GameV2
from transformer import HeartsTransformer
from data_structure import Card, Suit, PassDirection
from collections import deque
import strategies
from strategies import ExpertPolicy
import gpu_selector

# Hyperparameters
# --- PPO Training ---
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPISODES = 5000 
BATCH_SIZE = 64 
PPO_EPOCHS = 10 
CLIP_EPS = 0.2
ENTROPY_COEF_START = 0.03 
ENTROPY_COEF_END = 0.005 
VALUE_LOSS_COEF = 0.5 
MAX_GRAD_NORM = 0.5 

# --- Model Architecture ---
HIDDEN_DIM = 256 
DROPOUT = 0.1 
WEIGHT_DECAY = 1e-5 

# --- Supervised Pretraining ---
PRETRAIN_LR = 1e-4 
PRETRAIN_EPISODES = 5000
PRETRAIN_BATCH_SIZE = 64 
PRETRAIN_EPOCHS = 10 
LABEL_SMOOTHING = 0.0 

# --- DAgger (Dataset Aggregation) ---
DAGGER_BETA_START = 1.0 
DAGGER_BETA_DECAY = 0.9998 
DAGGER_BETA_MIN = 0.3 

# --- Curriculum Learning ---
CURRICULUM_SCORE_THRESHOLD = 6.0 
CURRICULUM_STABILITY_WINDOW = 500 
MAX_STAGE_EPISODES = 3000 
POOL_STAGE_DURATION = 2000

class OpponentPool:
    def __init__(self, max_size=50):
        self.pool = deque(maxlen=max_size)
    
    def add(self, model_state_dict):
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
        self.saved_states = [] 
        self.saved_actions = [] 
        self.saved_masks = [] 
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

    def pass_policy(self, player, info, teacher_policy=None, beta=1.0) -> List[Card]:
        self.hand_before_pass = set(player.hand)
        selected_cards = []
        
        teacher_cards_set = set()
        if teacher_policy:
            teacher_cards_list = teacher_policy(player, info)
            teacher_cards_set = set(teacher_cards_list)
        
        current_hand = list(player.hand)
        
        for i in range(3):
            temp_info = info.copy()
            temp_info['hand'] = current_hand
            temp_info['is_passing'] = True 
            
            self.model.update_from_info(
                temp_info, 
                legal_actions=current_hand,
                passed_cards=[], 
                received_cards=[] 
            )
            
            state_snapshot = self.model.get_raw_state()
            self.saved_states.append(state_snapshot)

            logits, value = self.model(x=None)
            logits = logits.squeeze() 
            
            mask = torch.full((52,), float('-inf'), device=self.device)
            legal_indices = [c.to_id() for c in current_hand]
            mask[legal_indices] = 0
            self.saved_masks.append(mask) 
            
            masked_logits = logits + mask
            
            probs = torch.softmax(masked_logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            
            student_action_idx = dist.sample()
            
            action_to_play_idx = student_action_idx
            action_to_save_idx = student_action_idx
            
            if teacher_policy:
                available_teacher_cards = sorted(
                    [c for c in teacher_cards_set if c in current_hand],
                    key=lambda c: c.to_id()
                )
                
                if available_teacher_cards:
                    teacher_card = available_teacher_cards[0]
                    teacher_action_idx = torch.tensor(teacher_card.to_id(), device=self.device)
                    action_to_save_idx = teacher_action_idx
                    
                    if random.random() < beta:
                        action_to_play_idx = teacher_action_idx
            
            self.saved_log_probs.append(dist.log_prob(action_to_play_idx))
            self.saved_values.append(value)
            self.saved_actions.append(action_to_save_idx) 
            
            action_idx_val = action_to_play_idx.item()
            suit_val = action_idx_val // 13
            rank_val = (action_idx_val % 13) + 1
            
            selected_card = next(c for c in current_hand if c.suit.value == suit_val and c.rank == rank_val)
            
            selected_cards.append(selected_card)
            current_hand.remove(selected_card)

        self.passed_cards = selected_cards
        return selected_cards

    def play_policy(self, player, info, legal_actions, order, override_policy=None, teacher_policy=None, beta=1.0):
        if not self.received_cards and self.passed_cards:
             remaining = self.hand_before_pass - set(self.passed_cards)
             current_hand_set = set(player.hand)
             self.received_cards = list(current_hand_set - remaining)

        info['is_passing'] = False 
        self.model.update_from_info(
            info, 
            legal_actions, 
            passed_cards=self.passed_cards, 
            received_cards=self.received_cards
        )
        
        state_snapshot = self.model.get_raw_state()
        self.saved_states.append(state_snapshot)
        
        logits, value = self.model(x=None) 
        logits = logits.squeeze()
        if logits.dim() > 1: logits = logits.squeeze(0)
        
        mask = torch.full((52,), float('-inf'), device=self.device)
        
        legal_indices = [c.to_id() for c in legal_actions]
        mask[legal_indices] = 0
        self.saved_masks.append(mask) 
        
        masked_logits = logits + mask
        
        probs = torch.softmax(masked_logits, dim=0)
        
        dist = torch.distributions.Categorical(probs)
        
        student_action_idx = dist.sample()
        
        suit_val = student_action_idx.item() // 13
        rank_val = (student_action_idx.item() % 13) + 1
        student_card = Card(suit=Suit(suit_val), rank=rank_val)
        
        action_to_play = student_card
        action_to_save = student_action_idx
        
        if override_policy:
            selected_card = override_policy(player, info, legal_actions, order)
            action_to_play = selected_card
            action_to_save = torch.tensor(selected_card.to_id(), device=self.device)
            
        elif teacher_policy:
            teacher_card = teacher_policy(player, info, legal_actions, order)
            teacher_action_idx = torch.tensor(teacher_card.to_id(), device=self.device)
            
            action_to_save = teacher_action_idx
            
            if random.random() < beta:
                action_to_play = teacher_card
            else:
                action_to_play = student_card

        self.saved_log_probs.append(dist.log_prob(torch.tensor(action_to_play.to_id(), device=self.device)))
        self.saved_values.append(value)
        self.saved_actions.append(action_to_save) 
        
        return action_to_play

def pretrain_supervised(model, device, episodes=1000):
    print("Starting Supervised Pretraining...")
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    game = GameV2()
    ai_player = AIPlayer(model, device)
    
    running_loss = 0.0
    running_p_loss = 0.0
    running_v_loss = 0.0
    running_acc = 0.0
    
    batch_states = []
    batch_actions = []
    batch_masks = []
    batch_returns = []
    
    beta = DAGGER_BETA_START
    beta_decay = DAGGER_BETA_DECAY
    min_beta = DAGGER_BETA_MIN
    
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

        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
            
        batch_states.extend(ai_player.saved_states)
        batch_actions.extend(ai_player.saved_actions)
        batch_masks.extend(ai_player.saved_masks)
        batch_returns.extend(returns)
        
        if (i_episode + 1) % PRETRAIN_BATCH_SIZE == 0:
            b_returns = torch.tensor(batch_returns).to(device)
            if len(b_returns) > 1:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-9)
            
            b_actions = torch.stack(batch_actions)
            masks = torch.stack(batch_masks)
            
            for _ in range(PRETRAIN_EPOCHS):
                x, padding_mask = model.assemble_batch(batch_states, device=device)
                
                logits, values = model(x=x, padding_mask=padding_mask)
                logits = logits.squeeze()
                values = values.squeeze()
                
                value_loss = torch.nn.functional.mse_loss(values, b_returns)
                
                masked_logits = logits + masks
                
                try:
                    policy_loss = torch.nn.functional.cross_entropy(masked_logits, b_actions, label_smoothing=LABEL_SMOOTHING)
                except RuntimeError as e:
                    print(f"Error in CrossEntropy: {e}")
                    policy_loss = torch.tensor(0.0, device=device, requires_grad=True)

                with torch.no_grad():
                    pred_actions = torch.argmax(masked_logits, dim=1)
                    accuracy = (pred_actions == b_actions).float().mean()
                
                loss = policy_loss + VALUE_LOSS_COEF * value_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step(loss)
                
                running_loss += loss.item()
                running_p_loss += policy_loss.item()
                running_v_loss += value_loss.item()
                running_acc += accuracy.item()
            
            batch_states = []
            batch_actions = []
            batch_masks = []
            batch_returns = []
        
        if (i_episode + 1) % PRETRAIN_BATCH_SIZE == 0:
             avg_loss = running_loss / PRETRAIN_EPOCHS
             avg_p = running_p_loss / PRETRAIN_EPOCHS
             avg_v = running_v_loss / PRETRAIN_EPOCHS
             avg_acc = running_acc / PRETRAIN_EPOCHS
             current_lr = optimizer.param_groups[0]['lr']
             print(f"Pretrain Episode {i_episode+1}/{episodes}\tLoss: {avg_loss:.4f} (P: {avg_p:.4f}, V: {avg_v:.4f})\tAcc: {avg_acc:.2%}\tBeta: {beta:.2f}\tLR: {current_lr:.2e}")
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
    
    start_episode = 0
    loaded_from_checkpoint = False
    
    if os.path.exists(model_path):
        user_input = input(f"Found existing model '{model_path}'. Load it? (y/n): ").strip().lower()
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
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded legacy model format. Starting from episode 0.")
                print("Model loaded successfully.")
                loaded_from_checkpoint = True
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("Starting training from scratch.")
    else:
        print("No existing model found. Starting from scratch.")

    if not loaded_from_checkpoint:
        pretrain_supervised(model, device, episodes=PRETRAIN_EPISODES)
    else:
        print("Skipping pretraining for loaded model.")
    
    game = GameV2()
    ai_player = AIPlayer(model, device)
    pool = OpponentPool()
    
    opponent_models = [HeartsTransformer(d_model=HIDDEN_DIM).to(device) for _ in range(3)]
    opponent_agents = [AIPlayer(opp_model, device) for opp_model in opponent_models]
    
    running_reward = 0
    running_score = 0
    
    log_data = {'episodes': [], 'scores': [], 'avg_scores': [], 'policy_losses': [], 'value_losses': [], 'difficulty': []}
    
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
        ("Stage 2: 3 Random Bots", [strategies.random_policy, strategies.random_policy, strategies.random_policy]),
        ("Stage 3: 2 Min + 1 Expert", [strategies.min_policy, strategies.min_policy, ExpertPolicy.play_policy]),
        ("Stage 4: 1 Min + 2 Expert", [strategies.min_policy, ExpertPolicy.play_policy, ExpertPolicy.play_policy]),
        ("Stage 5: 3 Expert", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, ExpertPolicy.play_policy]),
        ("Stage 6: 2 Expert + 1 Random", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, strategies.random_policy]),
        ("Stage 7: 2 Expert + 1 Pool", [ExpertPolicy.play_policy, ExpertPolicy.play_policy, 'pool']),
        ("Stage 8: 1 Expert + 2 Pool", [ExpertPolicy.play_policy, 'pool', 'pool']),
        ("Stage 9: 3 Pool", ['pool', 'pool', 'pool']),
    ]
    
    if loaded_from_checkpoint:
        start_stage_idx = 3
        print("Resuming training: Starting from Stage 5 (3 Expert Bots)")
    else:
        start_stage_idx = 0
        print("Starting fresh training: Starting from Stage 2 (3 Random Bots)")
        
    current_stage_idx = start_stage_idx
    stage_episode_count = 0
    consecutive_good_epochs = 0
    POOL_START_IDX = 5
    
    for i_episode in range(start_episode, start_episode + TOTAL_EPISODES):
        ai_player.reset()
        for opp in opponent_agents:
            opp.reset()
            
        if i_episode % 50 == 0:
            pool.add(model.state_dict())
        
        stage_episode_count += 1
        should_transition = False
        
        if current_stage_idx < POOL_START_IDX:
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
            if stage_episode_count >= POOL_STAGE_DURATION:
                should_transition = True

        if should_transition and current_stage_idx < len(STAGES) - 1:
            target_stage_idx = current_stage_idx + 1
            print(f"\n*** Curriculum Stage Transition: {STAGES[current_stage_idx][0]} -> {STAGES[target_stage_idx][0]} ***")
            print(f"*** Resetting Learning Rate to {LEARNING_RATE} ***")
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
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
                    if points_taken >= 13:
                        r -= 0.5 
                else:
                    r = 0.002 
                    if trick.winner != 0 and trick.score > 0:
                        r += 0.01
            rewards.append(r)
            
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
            R = r + GAMMA * R
            returns.insert(0, R)
            
        batch_log_probs.extend(ai_player.saved_log_probs)
        batch_values.extend(ai_player.saved_values)
        batch_states.extend(ai_player.saved_states)
        batch_actions.extend(ai_player.saved_actions)
        batch_masks.extend(ai_player.saved_masks)
        batch_returns.extend(returns)

        current_p_loss = 0.0
        current_v_loss = 0.0

        if (i_episode + 1) % BATCH_SIZE == 0:
            b_returns = torch.tensor(batch_returns).to(device)
            if len(b_returns) > 1:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-9)

            old_log_probs = torch.stack(batch_log_probs).detach()
            old_values = torch.stack(batch_values).detach().squeeze()
            
            advantages = b_returns - old_values
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                
            for _ in range(PPO_EPOCHS):
                logits, new_values = model(batch_raw_states=batch_states)
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
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = torch.nn.functional.mse_loss(new_values, b_returns)
                
                progress = min(1.0, (i_episode - start_episode) / TOTAL_EPISODES)
                current_entropy_coef = ENTROPY_COEF_START - (ENTROPY_COEF_START - ENTROPY_COEF_END) * progress
                
                entropy_loss = -current_entropy_coef * entropies.mean()
                
                loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
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
            running_score = player_final_score
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
        
        if i_episode % 10 == 0:
            try:
                with open("training_log.json", "w") as f:
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
            }, model_path)

    print("Training Complete.")
    torch.save({
        'epoch': start_episode + TOTAL_EPISODES - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)

if __name__ == "__main__":
    train()
