import torch
import random
import numpy as np
from collections import deque
from typing import List
from data_structure import Card, Suit
from model import HeartsProNet
from pimc import PIMCSelector

class SotaAgent:
    def __init__(self, model: HeartsProNet, device='cpu', use_pimc=False):
        self.model = model
        self.device = device
        self.use_pimc = use_pimc
        self.pimc = PIMCSelector(model, device)
        
        # Buffers for PPO
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_actions = []
        self.rewards = []
        
        # Re-evaluation buffers (Inputs needed to re-run forward pass)
        self.saved_static_obs = []
        self.saved_seq_cards = []
        self.saved_seq_players = []
        self.saved_global_priv = []
        self.saved_masks = []
        
        # Aux buffers
        self.saved_qs_labels = []
        self.saved_qs_preds = []
        
    def reset(self):
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_actions = []
        self.rewards = []
        
        self.saved_static_obs = []
        self.saved_seq_cards = []
        self.saved_seq_players = []
        self.saved_global_priv = []
        self.saved_masks = []
        
        self.saved_qs_labels = []
        self.saved_qs_preds = []

    def preprocess_obs(self, info):
        # Construct Static Features (124 dim)
        # Hand (52)
        hand_vec = torch.zeros(52)
        for c in info['hand']:
            hand_vec[c.to_id()] = 1.0
            
        # Table (52) - Cards currently on table
        table_vec = torch.zeros(52)
        for c, _ in info['current_table']:
            table_vec[c.to_id()] = 1.0
            
        # Voids (16) - Need to track this in Game or infer from history
        # For now, placeholder zeros to avoid data leakage (Ground truth void_label is for training targets only)
        void_vec = torch.zeros(16)
        
        # Scores (4)
        scores_vec = torch.tensor(info['scoreboard'], dtype=torch.float32) / 26.0
        
        static_obs = torch.cat([hand_vec, table_vec, void_vec, scores_vec])
        
        # Construct Dynamic Features (LSTM Sequence)
        # History is list of (Card, PlayerID)
        # In GameV2.get_player_info, 'trick_history' is a list of TrickRecord objects
        # But 'history' key in get_game_info is list of (Card, PlayerID)
        # Let's check what 'info' contains. It comes from get_player_info.
        
        # Fix: Use 'trick_history' from info to reconstruct the sequence of cards played
        # Or use 'table' which contains all cards played so far in order?
        # GameV2.get_player_info returns 'table' as list of (Card, PlayerID) which is exactly what we need.
        
        history = info.get('table', []) # List of (Card, PlayerID)
        
        seq_cards = []
        seq_players = []
        
        for c, pid in history:
            seq_cards.append(c.to_id())
            seq_players.append(pid)
            
        # Pad to length 52 (max game length)
        # We pad with 52 (Card) and 4 (Player) which are out of bounds for normal IDs
        # The embedding layers in model.py are size 53 and 5 respectively, so this is valid padding.
        target_len = 52
        current_len = len(seq_cards)
        
        if current_len < target_len:
            padding_len = target_len - current_len
            seq_cards.extend([52] * padding_len)
            seq_players.extend([4] * padding_len)
        else:
            seq_cards = seq_cards[:target_len]
            seq_players = seq_players[:target_len]
            
        return static_obs, torch.tensor(seq_cards), torch.tensor(seq_players)

    def act(self, player, info, legal_actions, order, training=True):
        static_obs, seq_cards, seq_players = self.preprocess_obs(info)
        
        # Add batch dim
        static_obs_b = static_obs.unsqueeze(0).to(self.device)
        seq_cards_b = seq_cards.unsqueeze(0).to(self.device)
        seq_players_b = seq_players.unsqueeze(0).to(self.device)
        
        # Global info for Critic (only if training)
        global_priv = None
        global_priv_b = None
        if training and 'global_state' in info:
            global_priv = info['global_state']
            global_priv_b = global_priv.unsqueeze(0).to(self.device)
            
        logits, value, qs_pred = self.model(static_obs_b, seq_cards_b, seq_players_b, global_priv_b)
        
        # Masking
        mask = torch.full((52,), float('-inf'), device=self.device)
        legal_indices = [c.to_id() for c in legal_actions]
        mask[legal_indices] = 0
        
        masked_logits = logits.squeeze() + mask
        probs = torch.softmax(masked_logits, dim=0)
        
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        
        # Save for PPO
        if training:
            self.saved_log_probs.append(dist.log_prob(action_idx))
            self.saved_values.append(value.squeeze())
            self.saved_actions.append(action_idx)
            self.saved_qs_preds.append(qs_pred.squeeze())
            self.saved_qs_labels.append(torch.tensor(info.get('sq_label', 4), device=self.device))
            
            # Save inputs for re-evaluation
            self.saved_static_obs.append(static_obs.to(self.device))
            self.saved_seq_cards.append(seq_cards.to(self.device))
            self.saved_seq_players.append(seq_players.to(self.device))
            if global_priv is not None:
                self.saved_global_priv.append(global_priv.to(self.device))
            else:
                # Should not happen in training if game.py is correct, but safe fallback
                self.saved_global_priv.append(torch.zeros(208).to(self.device))
            self.saved_masks.append(mask)
        
        # Convert to Card
        suit_val = action_idx.item() // 13
        rank_val = (action_idx.item() % 13) + 1
        selected_card = Card(Suit(suit_val), rank_val)
        
        # PIMC Override (Inference Only)
        if not training and self.use_pimc:
            # Get Top-3 from Policy
            top_k_indices = torch.topk(probs, k=min(3, len(legal_actions))).indices.tolist()
            top_k_cards = [Card(Suit(idx//13), (idx%13)+1) for idx in top_k_indices]
            
            # Run Search
            best_card = self.pimc.select_best_action(player, info, legal_actions, top_k_candidates=top_k_cards)
            return best_card
            
        return selected_card

    def pass_policy(self, player, info):
        # Simple heuristic or reuse model for passing
        # For now, random 3
        return random.sample(player.hand, 3)
