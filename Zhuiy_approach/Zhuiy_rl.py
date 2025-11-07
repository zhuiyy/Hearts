'''
Author: Zhuiy
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable, Union
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.dirname(os.path.abspath(__file__)) + '/data'
from game import Game, Card, Suit
from Zhuiy_sample.Zhuiy_sample_policy import sample_policy


def data_save(data: List[float], name: str, append: bool=False) -> None:
    df = pd.DataFrame(data)
    path = data_path + '/' + name + '.csv'
    if append and os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

def info_to_tensor(info: dict) -> np.ndarray:
    hand_array = np.zeros(52, dtype=np.float32)
    table_array = np.zeros(52, dtype=np.float32)
    current_table_array = np.zeros(52, dtype=np.float32)
    
    if info['hand']:
        hand_indices = [(card.suit.value * 13 + card.rank - 1) for card in info['hand']]
        hand_array[hand_indices] = 1
    
    if info['table']:
        table_indices = [(card.suit.value * 13 + card.rank - 1) for card, _ in info['table']]
        table_array[table_indices] = 1
    
    if info['current_table']:
        current_table_indices = [(card.suit.value * 13 + card.rank - 1) for card, _ in info['current_table']]
        current_table_array[current_table_indices] = 1
    
    current_suit_array = np.zeros(4, dtype=np.float32)
    if info['current_suit'] is not None:
        current_suit_array[info['current_suit'].value] = 1
    
    points_array = np.array([info['points']], dtype=np.float32)
    hearts_broken_array = np.array([1.0 if info['hearts_broken'] else 0.0], dtype=np.float32)
    piggy_pulled_array = np.array([1.0 if info['piggy_pulled'] else 0.0], dtype=np.float32)
    
    state_array = np.concatenate([
        hand_array, points_array, table_array, 
        current_suit_array, current_table_array, 
        hearts_broken_array, piggy_pulled_array
    ])
    
    return state_array

def actions_to_mask(actions: List[Card]) -> np.ndarray:
    mask_array = np.zeros(52, dtype=np.bool_)
    if actions:
        action_indices = [(card.suit.value * 13 + card.rank - 1) for card in actions]
        mask_array[action_indices] = True
    return mask_array

def actions_to_tensor(actions: List[Card]) -> np.ndarray:
    action_array = np.zeros(52, dtype=np.float32)
    if actions:
        action_indices = [(card.suit.value * 13 + card.rank - 1) for card in actions]
        action_array[action_indices] = 1
    return action_array

def action_to_card(action: int) -> Card:
    suit = action // 13
    rank = action % 13 + 1
    return Card(Suit(suit), rank)

def card_to_action(card: Card) -> int:
    return card.suit * 13 + card.rank - 1

class Value_net(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, device: Union[str, torch.device]):
        super(Value_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.to(device)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(state)))

class Policy_net(nn.Module):
    def __init__(self, state_dim: int, hidden_dim1: int, hidden_dim2: int,  action_dim: int, device: Union[str, torch.device]):
        super(Policy_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        self.to(device)
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.fc3(F.relu(self.fc2(F.relu(self.fc1(state))))), dim=-1)
    
class RF:
    def __init__(self, state_dim: int, hidden_dim1: int, hidden_dim2: int, action_dim: int, lr: float, gamma: float, device: Union[str, torch.device]):
        self.training_policy_net = Policy_net(state_dim, hidden_dim1, hidden_dim2, action_dim, device)
        self.value_net = Value_net(state_dim, (hidden_dim1 + hidden_dim2) // 2, device)
        self.gamma = gamma
        self.device = device
        self.lr = lr
        self.total_episodes = 0
        self.transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'masks': []
            }
        self.log = []
        self.loss_log = []
        self.p_optimizer = torch.optim.Adam(self.training_policy_net.parameters(), lr=self.lr)
        # Reduce ValueNet LR significantly for high-variance environment
        self.v_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.action_log_probs = []
        self.entropy_terms = []

    # -------- Curriculum helpers --------
    @staticmethod
    def linear_curriculum(start_ratio: float, end_ratio: float, start_episode: int, end_episode: int):
        """
        Returns a function f(ep) -> ratio in [0,1].
        - ep < start_episode: ratio = start_ratio
        - ep > end_episode: ratio = end_ratio
        - else: linear interpolation between start_ratio and end_ratio
        Typical use: slowly increase probability of unmasked training episodes.
        """
        def schedule(ep: int) -> float:
            if ep <= start_episode:
                return max(0.0, min(1.0, start_ratio))
            if ep >= end_episode:
                return max(0.0, min(1.0, end_ratio))
            t = (ep - start_episode) / max(1, (end_episode - start_episode))
            val = start_ratio + t * (end_ratio - start_ratio)
            return max(0.0, min(1.0, val))
        return schedule

    def take_action_with_mask(self, state: Union[np.ndarray, torch.Tensor], mask: Optional[Union[np.ndarray, torch.Tensor]]=None) -> Tuple[int, torch.Tensor]:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(self.device)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
    
        probs = self.training_policy_net(state)
        if mask is not None:
            probs = probs.masked_fill(~mask, 0)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = mask.float() / (mask.float().sum() + 1e-8)
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        # Store entropy term for regularization
        self.entropy_terms.append(action_dist.entropy())
        
        return action.item(), log_prob
    
    def take_action_without_mask(self, state: Union[np.ndarray, torch.Tensor], mask: Optional[Union[np.ndarray, torch.Tensor]]=None) -> Tuple[int, torch.Tensor]:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(self.device)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
    
        probs = self.training_policy_net(state)
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        # Store entropy term for regularization
        self.entropy_terms.append(action_dist.entropy())
        
        return action.item(), log_prob


    def reward_design(self, ai_score_delta: List[int], ai_actions: List[int], ai_info: List[np.ndarray], shot: Optional[bool]) -> List[int]:
        num_rounds = len(ai_score_delta)
        if shot is None:
            reward = [0] * num_rounds
            reward[-1] = -100
            return reward
        
        if shot is False:
            reward = [0] * num_rounds
            for i in range(num_rounds):
                reward[i] -= ai_score_delta[i] * 3
                if ai_actions[i] == 50: # SPADES Queen
                    if ai_score_delta[i] > 0:
                        reward[i] -= 20
                    else:
                        reward[i] += 10
                if ai_actions[i] < 13: # HEARTS
                    if ai_score_delta[i] > 0:
                        reward[i] -= 5
                    else:
                        reward[i] += 10
        else: # shot the moon
            reward = [5] * num_rounds
            for i in range(num_rounds):
                reward[i] += ai_score_delta[i]
        return reward

    def update(self, ai_score_delta: List[int], ai_actions: List[int], ai_log_probs: List[torch.Tensor], ai_info: List[np.ndarray], shot: Optional[bool]) -> Tuple[float, float]:
        states_np = np.stack(ai_info)
        states = torch.from_numpy(states_np).float().to(self.device)

        rewards_np = np.array(self.reward_design(ai_score_delta, ai_actions, ai_info, shot), dtype=np.float32)
        rewards = torch.from_numpy(rewards_np).float().view(-1, 1).to(self.device)

        log_probs_full = torch.stack(ai_log_probs).view(-1, 1).to(self.device)
        entropy_full = torch.stack(self.entropy_terms).view(-1, 1).to(self.device) if len(self.entropy_terms) == len(ai_log_probs) else None

        if shot is None:
            # Only use the final timestep when an illegal move occurred
            states = states[-1:]
            rewards = rewards[-1:]
            log_probs_full = log_probs_full[-1:]
            if entropy_full is not None:
                entropy_full = entropy_full[-1:]

        # Value prediction for the selected timesteps
        value_predicted = self.value_net(states).float().view(-1, 1).to(self.device)

        # Compute returns with normalization
        returns = []
        G = torch.zeros(1, device=self.device)
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.cat(returns).view(-1, 1)
        
        # Normalize returns to stabilize value learning
        if shot is not None and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        # Advantages
        advantages = (returns - value_predicted.detach())
        if shot is not None and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # Policy update with stronger entropy regularization
        self.p_optimizer.zero_grad()
        ent_coef = 0.05  # Increased to encourage more exploration
        if entropy_full is not None:
            policy_loss = -(log_probs_full * advantages).mean() - ent_coef * entropy_full.mean()
        else:
            policy_loss = -(log_probs_full * advantages).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.training_policy_net.parameters(), 0.5)
        self.p_optimizer.step()

        # Value update with stronger gradient clipping
        self.v_optimizer.zero_grad()
        value_loss = F.mse_loss(value_predicted, returns)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.v_optimizer.step()

        return policy_loss.item(), value_loss.item()
    
    def save(self, path: str) -> None:
        torch.save({
            'policy_net_state_dict': self.training_policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'p_optimizer_state_dict': self.p_optimizer.state_dict(),
            'v_optimizer_state_dict': self.v_optimizer.state_dict(),
            'total_episodes': self.total_episodes
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.training_policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.p_optimizer.load_state_dict(checkpoint['p_optimizer_state_dict'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
        self.total_episodes = checkpoint.get('total_episodes', 0)
    
    def training_policy(self, player, player_info: dict, actions: List[Card], order: int) -> Card:
        a, b = self.take_action_with_mask(info_to_tensor(player_info), actions_to_mask(actions))
        self.action_log_probs.append(b)
        return action_to_card(a)

    def training_policy_unmasked(self, player, player_info: dict, actions: List[Card], order: int) -> Card:
        # Occasionally used for curriculum: allow sampling without mask
        a, b = self.take_action_without_mask(info_to_tensor(player_info))
        self.action_log_probs.append(b)
        return action_to_card(a)
    
    def policy(self, player, player_info: dict, actions: List[Card], order: int) -> Card:
        a, b = self.take_action_with_mask(info_to_tensor(player_info), actions_to_mask(actions))
        return action_to_card(a)

    def showcase_policy(self, player, player_info: dict, actions: List[Card], order: int) -> Card:
        # Unmasked sampling to check rule-following behavior
        chosen_action_idx, _ = self.take_action_without_mask(info_to_tensor(player_info))
        chosen_card = action_to_card(chosen_action_idx)

        legal_actions = actions
        if chosen_card not in legal_actions:
            print(f"--- [Showcase] AI player {player.player_id} attempted an illegal play: {chosen_card} ---")
            print(f"--- Legal actions: {legal_actions} ---")
        
        return chosen_card
    
    def train(self, game: Game, oppo_policy: List[Callable], episodes: int, append_log: bool=False, curriculum: Optional[Callable]=None) -> None:
        p_loss = []
        v_loss = []
        points = []
        
        for i in tqdm(range(episodes)):
            self.action_log_probs = []
            self.entropy_terms = []
            # Decide whether to use an unmasked episode for the learning agent, based on curriculum
            ep_index = self.total_episodes + i
            ratio = 0.0
            if callable(curriculum):
                try:
                    ratio = float(curriculum(ep_index))
                except TypeError:
                    # For backward compatibility: curriculum(ep_index, episodes)
                    ratio = float(curriculum(ep_index, episodes))
            ratio = max(0.0, min(1.0, ratio))
            use_unmasked = (random.random() < ratio)

            learner_policy = self.training_policy_unmasked if use_unmasked else self.training_policy

            score, shot, ai_score_delta, ai_actions, ai_masks, ai_info = game.fight([learner_policy] + oppo_policy, True, False, False)
            ai_actions = np.array([card_to_action(action) for action in ai_actions])
            ai_masks = np.array([actions_to_mask(mask) for mask in ai_masks])
            ai_info = np.array([info_to_tensor(info) for info in ai_info])

            p, v = self.update(ai_score_delta, ai_actions, self.action_log_probs, ai_info, shot)
            p_loss.append(p)
            v_loss.append(v)
            points.append(score[0])
        
        self.total_episodes += episodes
        self.save(data_path + '/model_checkpoint.pth')
        data_save(p_loss, 'p_loss', append=append_log)
        data_save(v_loss, 'v_loss', append=append_log)
        data_save(points, 'score', append=append_log)
        print(f'Training finished. total_episodes: {self.total_episodes}, mean_score: {sum(points) / episodes}, mean_p_loss: {sum(p_loss) / episodes}, mean_v_loss: {sum(v_loss) / episodes}')

    def evaluation(self, game: Game, oppo_policy: List[Callable], episodes: int) -> None:
        s = np.array([0, 0, 0, 0])
        print('Evaluating')
        for i in tqdm(range(episodes)):
            score, a, b, c, d, e = game.fight([self.policy] + oppo_policy, True, False, False)
            s += score
        print(s / episodes)

    def showcase_games(self, game: Game, oppo_policy: List[Callable], episodes: int=3) -> None:
        print("\n--- Show 3 games ---")
        for i in range(episodes):
            print(f"\n--- Game {i+1} ---")
            score, _, _, _, _, _ = game.fight([self.showcase_policy] + oppo_policy, training=True, verbose=True, human_0=False)
            print(f"--- Game {i+1} finished, final score: {score} ---")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Reduce learning rate for more stable training
    model = RF(163, 128, 64, 52, 5e-4, 0.98, device)
    Hearts = Game()

    while True:
        to_train = input("Train? (y/n): ")
        if to_train in ['y', 'n']:
            break
    
    while True:
        to_load = input("Load existing model? (y/n): ")
        if to_load in ['y', 'n']:
            break
    
    model_path = data_path + '/model_checkpoint.pth'

    if to_train == 'n':
        if to_load == 'y':
            if os.path.exists(model_path):
                model.load(model_path)
                print(f"Model loaded. Trained episodes: {model.total_episodes}.")
                model.showcase_games(Hearts, [sample_policy, sample_policy, sample_policy])
            else:
                print(f"Model file not found: {model_path}. Unable to evaluate.")
                exit(0)
        else:
            print('Kicking you off')
            time.sleep(2)
            exit(0)
    else: # to_train == 'y'
        append_log = False
        if to_load == 'y':
            if os.path.exists(model_path):
                model.load(model_path)
                print(f"Loaded model: {model_path}. Continue training. Trained episodes: {model.total_episodes}.")
                append_log = True
                model.train(Hearts, [sample_policy, sample_policy, sample_policy], 300, append_log=append_log)
                model.showcase_games(Hearts, [sample_policy, sample_policy, sample_policy])
            else:
                print(f"Model file not found: {model_path}. Training from scratch.")
                model.train(Hearts, [sample_policy, sample_policy, sample_policy], 300, append_log=False)
                model.showcase_games(Hearts, [sample_policy, sample_policy, sample_policy])
        else: # to_load == 'n'
            print("Training from scratch.")
            for log_file in ['p_loss.csv', 'v_loss.csv', 'score.csv']:
                if os.path.exists(data_path + '/' + log_file):
                    os.remove(data_path + '/' + log_file)
            model.train(Hearts, [sample_policy, sample_policy, sample_policy], 300, append_log=False)
            model.showcase_games(Hearts, [sample_policy, sample_policy, sample_policy])
