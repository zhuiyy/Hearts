import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple
from data_structure import Card, Suit, GameState, PassDirection

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class hand_matrix:
    def __init__(self):
        self.matrix = torch.zeros((13, 22))

    def update(self, hand: List[Card], legal_actions: List[Card], received_cards: List[Card]):
        self.matrix.zero_()
        sorted_hand = sorted(hand)
        
        for i, card in enumerate(sorted_hand):
            if i >= 13: break
            self.matrix[i, 0] = 1.0
            self.matrix[i, 1 + card.suit.value] = 1.0
            self.matrix[i, 5 + (card.rank - 1)] = 1.0
            self.matrix[i, 18] = card.value()
            if card in legal_actions:
                self.matrix[i, 19] = 1.0
            if card in received_cards:
                self.matrix[i, 21] = 1.0

class current_table_matrix:
    def __init__(self):
        self.matrix = torch.zeros((4, 24))

    def update(self, current_table, my_player_id, current_suit):
        self.matrix.zero_()
        best_rank = -1
        winning_idx = -1
        
        # Find winning card
        for idx, (card, pid) in enumerate(current_table):
            if card.suit == current_suit:
                r_val = (card.rank - 2) % 13
                if r_val > best_rank:
                    best_rank = r_val
                    winning_idx = idx

        for i, (card, pid) in enumerate(current_table):
            self.matrix[i, 0] = 1.0
            rel_pid = (pid - my_player_id) % 4
            self.matrix[i, 1 + rel_pid] = 1.0
            self.matrix[i, 5 + card.suit.value] = 1.0
            self.matrix[i, 9 + (card.rank - 1)] = 1.0
            self.matrix[i, 22] = card.value()
            if i == winning_idx:
                self.matrix[i, 23] = 1.0

class history_matrix:
    def __init__(self):
        self.matrix = torch.zeros((52, 25))

    def update(self, trick_history, my_player_id):
        self.matrix.zero_()
        row = 0
        for trick in trick_history:
            lead_suit = trick.cards[0][0].suit
            best_rank = -1
            winner_local_idx = -1
            
            # Find winner of the trick
            for idx, (card, pid) in enumerate(trick.cards):
                if card.suit == lead_suit:
                    r_val = (card.rank - 2) % 13
                    if r_val > best_rank:
                        best_rank = r_val
                        winner_local_idx = idx
            
            for idx, (card, pid) in enumerate(trick.cards):
                if row >= 52: break
                self.matrix[row, 0] = 1.0
                rel_pid = (pid - my_player_id) % 4
                self.matrix[row, 1 + rel_pid] = 1.0
                self.matrix[row, 5 + card.suit.value] = 1.0
                self.matrix[row, 9 + (card.rank - 1)] = 1.0
                self.matrix[row, 22] = card.value()
                if idx == 0:
                    self.matrix[row, 23] = 1.0
                if idx == winner_local_idx:
                    self.matrix[row, 24] = 1.0
                row += 1

class hidden_matrix:
    def __init__(self):
        self.matrix = torch.zeros((39, 23))

    def update(self, hand, table_history, passed_cards, pass_direction):
        self.matrix.zero_()
        known_cards = set(hand)
        for trick in table_history:
            for card, _ in trick.cards:
                known_cards.add(card)
        
        row = 0
        for suit in Suit:
            for rank in range(1, 14):
                card = Card(suit, rank)
                if card in known_cards:
                    continue
                
                if row >= 39: break
                
                self.matrix[row, 0] = 1.0
                self.matrix[row, 1 + suit.value] = 1.0
                self.matrix[row, 5 + (rank - 1)] = 1.0
                self.matrix[row, 18] = card.value()
                
                if card in passed_cards:
                    self.matrix[row, 19] = 1.0
                    if pass_direction == PassDirection.LEFT:
                        self.matrix[row, 20] = 1.0
                    elif pass_direction == PassDirection.ACROSS:
                        self.matrix[row, 21] = 1.0
                    elif pass_direction == PassDirection.RIGHT:
                        self.matrix[row, 22] = 1.0
                row += 1

class player_stats_matrix:
    def __init__(self):
        self.matrix = torch.zeros((4, 2))

    def update(self, players_stats, my_player_id):
        self.matrix.zero_()
        if not players_stats: return
        for i in range(4):
            target_pid = (my_player_id + i) % 4
            stats = players_stats[target_pid]
            self.matrix[i, 0] = min(stats['points'] / 100.0, 1.0)
            if stats['has_sq']:
                self.matrix[i, 1] = 1.0

class game_stats_matrix:
    def __init__(self):
        self.matrix = torch.zeros((1, 9))

    def update(self, game_state: GameState, rounds: int, pass_direction: PassDirection, is_passing: bool = False):
        self.matrix.zero_()
        self.matrix[0, 0] = rounds / 13.0
        if rounds == 0:
            self.matrix[0, 1] = 1.0
        if game_state.heart_broken:
            self.matrix[0, 2] = 1.0
        if game_state.piggy_pulled:
            self.matrix[0, 3] = 1.0
        
        if pass_direction == PassDirection.LEFT:
            self.matrix[0, 4] = 1.0
        elif pass_direction == PassDirection.RIGHT:
            self.matrix[0, 5] = 1.0
        elif pass_direction == PassDirection.ACROSS:
            self.matrix[0, 6] = 1.0
        elif pass_direction == PassDirection.KEEP:
            self.matrix[0, 7] = 1.0
            
        if is_passing:
            self.matrix[0, 8] = 1.0

class HeartsTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        
        self.matrix = [
            hand_matrix(),
            current_table_matrix(),
            history_matrix(),
            hidden_matrix(),
            player_stats_matrix(),
            game_stats_matrix()
        ]
        
        self.input_projections = nn.ModuleList([
            nn.Linear(22, d_model), # hand
            nn.Linear(24, d_model), # current_table
            nn.Linear(25, d_model), # history
            nn.Linear(23, d_model), # hidden
            nn.Linear(2, d_model),  # player_stats
            nn.Linear(9, d_model)   # game_stats
        ])
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.segment_embeddings = nn.Embedding(7, d_model)
        self.card_embedding = nn.Embedding(53, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # History Encoder
        history_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.history_encoder = nn.TransformerEncoder(history_layer, num_layers=4)
        
        # Main Encoder
        main_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.main_encoder = nn.TransformerEncoder(main_layer, num_layers=num_layers)
        
        self.output_head = nn.Linear(d_model, 52)
        
        # --- CTDE & Aux Heads ---
        # Global Encoder for Critic (4 players * 52 cards = 208)
        self.global_encoder = nn.Sequential(
            nn.Linear(208, 512),
            nn.ReLU(),
            nn.Linear(512, d_model),
            nn.ReLU()
        )
        
        # Value Head now takes [Actor_Features; Global_Features]
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Aux 1: Predict SQ Location (5 classes: 0-3 players, 4=played/unknown)
        self.aux_sq_head = nn.Linear(d_model, 5)
        
        # Aux 2: Predict Void Suits (4 players * 4 suits = 16 binary labels)
        self.aux_void_head = nn.Linear(d_model, 16)

    def update_from_info(self, info, legal_actions, passed_cards=[], received_cards=[]):
        # info dict contains: 'game_state', 'rounds', 'pass_direction', 'my_id'
        
        game_state = info['game_state']
        player_id = info['my_id']
        
        player = game_state.players[player_id]
        
        self.matrix[0].update(player.hand, legal_actions, received_cards)
        self.matrix[1].update(game_state.current_table, player_id, game_state.current_suit)
        
        # Trick history is needed.
        trick_history = info.get('trick_history', [])
        self.matrix[2].update(trick_history, player_id)
        
        self.matrix[3].update(player.hand, trick_history, passed_cards, info['pass_direction'])
        
        # Players stats
        players_stats = info.get('players_stats', [])
        self.matrix[4].update(players_stats, player_id)
        
        self.matrix[5].update(game_state, info['rounds'], info['pass_direction'], info.get('is_passing', False))

    def get_raw_state(self):
        return [m.matrix.clone().detach() for m in self.matrix]

    def assemble_batch(self, batch_raw_states, device=None, dtype=torch.float32):
        if device is None:
            device = self.cls_token.device
            
        num_matrices = len(self.matrix)
        batched_sources = [[] for _ in range(num_matrices)]
        
        for state in batch_raw_states:
            for i, m in enumerate(state):
                batched_sources[i].append(m)
                
        embeddings_list = []
        mask_list = []
        
        for i, matrix_list in enumerate(batched_sources):
            if matrix_list and matrix_list[0].device != device:
                 stacked = torch.stack(matrix_list).to(device=device, dtype=dtype)
            else:
                 stacked = torch.stack([m.to(device=device, dtype=dtype) for m in matrix_list])
            
            batch_size, rows, _ = stacked.shape
            
            if i < 4: 
                is_padding = (stacked[:, :, 0] == 0.0)
            else:
                is_padding = torch.zeros((batch_size, rows), dtype=torch.bool, device=device)
            
            mask_list.append(is_padding)
            
            proj = self.input_projections[i](stacked)
            
            seg_emb = self.segment_embeddings(torch.tensor(i, device=device))
            proj = proj + seg_emb
            
            batch_indices = torch.full((batch_size, rows), 52, dtype=torch.long, device=device)
            
            if i < 4:
                valid_mask = (stacked[:, :, 0] == 1.0)
                
                if i == 0 or i == 3: 
                    suits = stacked[:, :, 1:5].argmax(dim=2)
                    ranks = stacked[:, :, 5:18].argmax(dim=2)
                    ids = suits * 13 + ranks
                    batch_indices[valid_mask] = ids[valid_mask]
                    
                elif i == 1 or i == 2: 
                    suits = stacked[:, :, 5:9].argmax(dim=2)
                    ranks = stacked[:, :, 9:22].argmax(dim=2)
                    ids = suits * 13 + ranks
                    batch_indices[valid_mask] = ids[valid_mask]
            
            card_emb = self.card_embedding(batch_indices)
            proj = proj + card_emb
            
            embeddings_list.append(proj)
            
        final_embeddings = torch.cat(embeddings_list, dim=1)
        final_mask = torch.cat(mask_list, dim=1)
        
        return final_embeddings, final_mask

    def forward(self, x=None, padding_mask=None, batch_raw_states=None, global_state=None):
        if x is None:
            if batch_raw_states is None:
                # Single inference mode using internal state
                batch_raw_states = [self.get_raw_state()]
            x, padding_mask = self.assemble_batch(batch_raw_states)
            
        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Add Segment Embedding for CLS (Index 6)
        cls_seg_emb = self.segment_embeddings(torch.tensor(6, device=x.device))
        cls_tokens = cls_tokens + cls_seg_emb
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Update mask for CLS (False = not padded)
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
        
        x = self.pos_encoder(x)
        
        # Dual Transformer Logic
        # History is at index 2.
        # Sequence lengths:
        # CLS: 1
        # Hand: 13
        # Table: 4
        # History: 52
        # Hidden: 39
        # PStats: 4
        # GStats: 1
        
        # Indices:
        # CLS: 0
        # Hand: 1-13
        # Table: 14-17
        # History: 18-69 (52 items)
        # Hidden: 70-108
        # ...
        
        history_start = 1 + 13 + 4
        history_end = history_start + 52
        
        x_history = x[:, history_start:history_end, :]
        mask_history = padding_mask[:, history_start:history_end]
        
        # Handle all-masked history to prevent NaNs
        all_masked = mask_history.all(dim=1)
        
        if all_masked.any():
            mask_history_for_enc = mask_history.clone()
            mask_history_for_enc[all_masked, 0] = False
            x_history_encoded = self.history_encoder(x_history, src_key_padding_mask=mask_history_for_enc)
        else:
            x_history_encoded = self.history_encoder(x_history, src_key_padding_mask=mask_history)
        
        x_pre = x[:, :history_start, :]
        x_post = x[:, history_end:, :]
        
        mask_pre = padding_mask[:, :history_start]
        mask_post = padding_mask[:, history_end:]
        
        x_combined = torch.cat((x_pre, x_history_encoded, x_post), dim=1)
        mask_combined = torch.cat((mask_pre, mask_history, mask_post), dim=1)
            
        output = self.main_encoder(x_combined, src_key_padding_mask=mask_combined)
        
        pooled = output[:, 0, :] 
        
        logits = self.output_head(pooled)
        
        # --- CTDE Logic ---
        value = None
        if global_state is not None:
            # Training mode: Use real global state
            global_features = self.global_encoder(global_state)
            critic_input = torch.cat([pooled, global_features], dim=1)
            value = self.value_head(critic_input)

        # --- Aux Heads ---
        pred_sq = self.aux_sq_head(pooled)
        pred_void = self.aux_void_head(pooled)
        
        return logits, value, pred_sq, pred_void

