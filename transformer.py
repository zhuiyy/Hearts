import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from data_structure import Card, Suit, GameState, PassDirection

def card_to_tensor(card: Card) -> torch.Tensor:
    # Helper to create card features: Suit (4) + Rank (13) + Value (1)
    # Total 18 dims
    t = torch.zeros(18)
    t[card.suit.value] = 1.0
    t[4 + (card.rank - 1)] = 1.0
    t[17] = card.value()
    return t

class hand_matrix:
    def __init__(self):
        self.matrix = torch.zeros((13, 22))
        # make_sense *1
        # suit * 4
        # rank * 13
        # value * 1
        # is_available * 1
        # is_possible_to_win * 1
        # is_received * 1

    def info_init(self):
        self.matrix.zero_()

    def update(self, hand: List[Card], legal_actions: List[Card], received_cards: List[Card]):
        self.matrix.zero_()
        for i, card in enumerate(hand):
            if i >= 13: break
            # make_sense
            self.matrix[i, 0] = 1.0
            # suit (1-4) -> indices 1-4
            self.matrix[i, 1 + card.suit.value] = 1.0
            # rank (1-13) -> indices 5-17
            self.matrix[i, 5 + (card.rank - 1)] = 1.0
            # value -> index 18
            self.matrix[i, 18] = card.value()
            # is_available -> index 19
            if card in legal_actions:
                self.matrix[i, 19] = 1.0
            # is_possible_to_win -> index 20 (Placeholder logic)
            # is_received -> index 21
            if card in received_cards:
                self.matrix[i, 21] = 1.0

class current_table_matrix:
    def __init__(self):
        self.matrix = torch.zeros((4, 24))
        # make_sense *1
        # player_id * 4 (Relative)
        # suit * 4
        # rank * 13
        # value * 1
        # is_biggest * 1

    def info_init(self):
        self.matrix.zero_()

    def update(self, current_table: List[Tuple[Card, int]], my_player_id: int, current_suit: Optional[Suit]):
        self.matrix.zero_()
        if not current_table:
            return

        # Determine current winner (biggest)
        winning_idx = -1
        if current_suit is not None:
            best_rank = -1
            for idx, (card, pid) in enumerate(current_table):
                if card.suit == current_suit:
                    # Ace is 1, but in Hearts Ace is high. 
                    # Rank 1 (Ace) > 13 (King) > ... > 2
                    # We need a comparison value.
                    # (rank - 2) % 13: 2->0, ..., K->11, A->12. Correct.
                    r_val = (card.rank - 2) % 13
                    if r_val > best_rank:
                        best_rank = r_val
                        winning_idx = idx

        for i, (card, pid) in enumerate(current_table):
            # make_sense
            self.matrix[i, 0] = 1.0
            # player_id (Relative: 0=Me, 1=Left, 2=Across, 3=Right)
            rel_pid = (pid - my_player_id) % 4
            self.matrix[i, 1 + rel_pid] = 1.0
            # suit
            self.matrix[i, 5 + card.suit.value] = 1.0
            # rank
            self.matrix[i, 9 + (card.rank - 1)] = 1.0
            # value
            self.matrix[i, 22] = card.value()
            # is_biggest
            if i == winning_idx:
                self.matrix[i, 23] = 1.0
    
class history_matrix:
    def __init__(self):
        self.matrix = torch.zeros((52, 25))
        # make_sense *1
        # player_id * 4
        # suit * 4
        # rank * 13
        # value * 1
        # is_leading * 1
        # is_winning * 1

    def info_init(self):
        self.matrix.zero_()

    def update(self, table_history: List[Tuple[Card, int]], my_player_id: int):
        self.matrix.zero_()
        # Process history in chunks of 4 (tricks)
        for i in range(0, len(table_history), 4):
            trick = table_history[i : i+4]
            if not trick: continue
            
            # Determine lead suit and winner for this trick
            lead_card = trick[0][0]
            lead_suit = lead_card.suit
            
            best_rank = -1
            winner_local_idx = -1
            
            for idx, (card, pid) in enumerate(trick):
                if card.suit == lead_suit:
                    r_val = (card.rank - 2) % 13
                    if r_val > best_rank:
                        best_rank = r_val
                        winner_local_idx = idx
            
            # Fill matrix
            for idx, (card, pid) in enumerate(trick):
                row = i + idx
                if row >= 52: break
                
                self.matrix[row, 0] = 1.0
                # player_id (Relative)
                rel_pid = (pid - my_player_id) % 4
                self.matrix[row, 1 + rel_pid] = 1.0
                # suit
                self.matrix[row, 5 + card.suit.value] = 1.0
                # rank
                self.matrix[row, 9 + (card.rank - 1)] = 1.0
                # value
                self.matrix[row, 22] = card.value()
                # is_leading
                if idx == 0:
                    self.matrix[row, 23] = 1.0
                # is_winning
                if idx == winner_local_idx:
                    self.matrix[row, 24] = 1.0

class hidden_matrix:
    def __init__(self):
        self.matrix = torch.zeros((39, 23)) # Corrected size
        # make_sense * 1 (idx 0)
        # suit * 4 (idx 1-4)
        # rank * 13 (idx 5-17)
        # value * 1 (idx 18)
        # is_passed_by_me * 1 (idx 19)
        # owner_left * 1 (idx 20)
        # owner_across * 1 (idx 21)
        # owner_right * 1 (idx 22)

    def info_init(self):
        self.matrix.zero_()

    def update(self, hand: List[Card], table_history: List[Tuple[Card, int]], 
               passed_cards: List[Card], pass_direction: PassDirection):
        self.matrix.zero_()
        
        # Set of known cards (in hand or played)
        known_cards = set(hand)
        for c, _ in table_history:
            known_cards.add(c)
            
        # Iterate all 52 cards to find hidden ones
        row = 0
        for suit in Suit:
            for rank in range(1, 14):
                card = Card(suit, rank)
                if card in known_cards:
                    continue
                
                if row >= 39: break
                
                # make_sense
                self.matrix[row, 0] = 1.0
                # suit (1-4) -> indices 1-4
                self.matrix[row, 1 + suit.value] = 1.0
                # rank (1-13) -> indices 5-17
                self.matrix[row, 5 + (rank - 1)] = 1.0
                # value -> index 18
                # Wait, 5 + 12 = 17. So index 18 is correct for value.
                # But matrix size is (39, 14).
                # Ah, I reduced the size but kept the old indices!
                # Old indices were based on 21 or something?
                # Let's recount:
                # make_sense: 0
                # suit: 1, 2, 3, 4
                # rank: 5..17 (13 dims)
                # value: 18
                # is_passed_by_me: 19
                # owner_left: 20
                # owner_across: 21
                # owner_right: 22
                # Total needed: 23 dims.
                # But I initialized it with 14!
                # Why 14?
                # make_sense (1) + suit (4) + rank (13) + value (1) = 19 already.
                # I must have miscalculated or intended to use embeddings for rank/suit?
                # No, I am using one-hot.
                # So I need to increase the dimension of hidden_matrix.
                
                self.matrix[row, 18] = card.value()
                
                # Pass info
                if card in passed_cards:
                    self.matrix[row, 19] = 1.0 # is_passed_by_me
                    
                    # Owner logic
                    if pass_direction == PassDirection.LEFT:
                        self.matrix[row, 20] = 1.0 # Left
                    elif pass_direction == PassDirection.ACROSS:
                        self.matrix[row, 21] = 1.0 # Across
                    elif pass_direction == PassDirection.RIGHT:
                        self.matrix[row, 22] = 1.0 # Right
                        
                row += 1

class player_stats_matrix:
    def __init__(self):
        self.matrix = torch.zeros((4, 2))
        # points * 1
        # has_spade_queen * 1 (Did they take it?)

    def info_init(self):
        self.matrix.zero_()

    def update(self, players: List, my_player_id: int):
        self.matrix.zero_()
        for i in range(4):
            target_pid = (my_player_id + i) % 4
            player = players[target_pid]
            self.matrix[i, 0] = player.points
            has_sq = False
            for c in player.table:
                if c.suit == Suit.SPADES and c.rank == 12:
                    has_sq = True
                    break
            if has_sq:
                self.matrix[i, 1] = 1.0

    def update_from_dicts(self, players_stats: List[dict], my_player_id: int):
        self.matrix.zero_()
        if not players_stats: return
        for i in range(4):
            target_pid = (my_player_id + i) % 4
            stats = players_stats[target_pid]
            self.matrix[i, 0] = stats['points']
            if stats['has_sq']:
                self.matrix[i, 1] = 1.0

class game_stats_matrix:
    def __init__(self):
        self.matrix = torch.zeros((1, 9))
        # rounds_played * 1
        # is_first_round * 1
        # heart_broken * 1
        # piggy_pulled * 1
        # pass_dir_left * 1
        # pass_dir_right * 1
        # pass_dir_across * 1
        # pass_dir_keep * 1
        # is_passing_phase * 1  <-- New Feature

    def info_init(self):
        self.matrix.zero_()

    def update(self, game_state: GameState, rounds: int, pass_direction: PassDirection, is_passing: bool = False):
        self.update_values(rounds, game_state.heart_broken, game_state.piggy_pulled, pass_direction, is_passing)

    def update_values(self, rounds: int, heart_broken: bool, piggy_pulled: bool, pass_direction: PassDirection, is_passing: bool = False):
        self.matrix.zero_()
        self.matrix[0, 0] = rounds
        self.matrix[0, 1] = 1.0 if rounds == 1 else 0.0
        self.matrix[0, 2] = 1.0 if heart_broken else 0.0
        self.matrix[0, 3] = 1.0 if piggy_pulled else 0.0
        
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
    def __init__(self, d_model=64, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.matrix = [
            hand_matrix(),
            current_table_matrix(),
            history_matrix(),
            hidden_matrix(),
            player_stats_matrix(),
            game_stats_matrix()
        ]
        
        # Calculate total input dimension based on matrix shapes
        # hand: 13x22, table: 4x24, history: 52x25, hidden: 39x14, p_stats: 4x2, g_stats: 1x8
        # Total rows (seq_len) = 13 + 4 + 52 + 39 + 4 + 1 = 113
        # Max cols (feature_dim) = 25 (history)
        # We need to project each block to d_model
        
        self.input_projections = nn.ModuleList([
            nn.Linear(22, d_model), # hand
            nn.Linear(24, d_model), # current_table
            nn.Linear(25, d_model), # history
            nn.Linear(23, d_model), # hidden (Corrected from 14)
            nn.Linear(2, d_model),  # player_stats
            nn.Linear(9, d_model)   # game_stats (Updated to 9)
        ])
        
        # [CLS] Token: A learnable parameter to aggregate global information
        # Shape: (1, 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head - predicting card probabilities (52 cards)
        self.output_head = nn.Linear(d_model, 52)
        
        # Value head - predicting state value (scalar)
        self.value_head = nn.Linear(d_model, 1)

    def update_state(self, player_id: int, game_state: GameState, legal_actions: List[Card], 
                     rounds: int, pass_direction: PassDirection,
                     passed_cards: List[Card] = [], received_cards: List[Card] = [], is_passing: bool = False):
        """
        Update all internal matrices based on the current game state.
        """
        player = game_state.players[player_id]
        
        # 1. Hand Matrix
        self.matrix[0].update(player.hand, legal_actions, received_cards)
        
        # 2. Current Table Matrix
        self.matrix[1].update(game_state.current_table, player_id, game_state.current_suit)
        
        # 3. History Matrix
        self.matrix[2].update(game_state.table, player_id)
        
        # 4. Hidden Matrix
        self.matrix[3].update(player.hand, game_state.table, passed_cards, pass_direction)
        
        # 5. Player Stats Matrix
        self.matrix[4].update(game_state.players, player_id)
        
        # 6. Game Stats Matrix
        self.matrix[5].update(game_state, rounds, pass_direction, is_passing)

    def update_from_info(self, info: dict, legal_actions: List[Card], 
                         passed_cards: List[Card] = [], received_cards: List[Card] = []):
        """
        Update matrices using the info dictionary from GameV2.
        """
        player_id = info['player_id']
        
        # 1. Hand Matrix
        self.matrix[0].update(info['hand'], legal_actions, received_cards)
        
        # 2. Current Table Matrix
        self.matrix[1].update(info['current_table'], player_id, info['current_suit'])
        
        # 3. History Matrix
        self.matrix[2].update(info['table'], player_id)
        
        # 4. Hidden Matrix
        self.matrix[3].update(info['hand'], info['table'], passed_cards, info.get('pass_direction', PassDirection.KEEP))
        
        # 5. Player Stats Matrix
        # We need to adapt player_stats_matrix.update to take a list of dicts or objects
        # Currently it expects List[Player].
        # Let's modify player_stats_matrix.update to handle dicts too.
        self.matrix[4].update_from_dicts(info.get('players_stats', []), player_id)
        
        # 6. Game Stats Matrix
        # We need to adapt game_stats_matrix.update to take raw values
        self.matrix[5].update_values(info['rounds'], info['heart_broken'], info['piggy_pulled'], 
                                     info.get('pass_direction', PassDirection.KEEP),
                                     info.get('is_passing', False))

    def get_raw_state(self):
        """Return a snapshot of the current raw matrix data."""
        return [m.matrix.clone().detach() for m in self.matrix]

    def assemble_input(self, raw_state=None, device=None, dtype=torch.float32):
        """
        Assemble the cached matrices into a sequence of embeddings.
        Instead of a block diagonal sparse matrix (which is huge and mostly empty),
        we project each feature matrix to d_model and concatenate them along the sequence dimension.
        """
        # Ensure we use the device of the model if not specified
        if device is None:
            device = next(self.parameters()).device

        embeddings = []
        # Use provided raw_state or internal self.matrix
        sources = raw_state if raw_state is not None else [m.matrix for m in self.matrix]

        for i, tensor in enumerate(sources):
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=dtype, device=device)
            else:
                tensor = tensor.to(dtype=dtype, device=device)
            
            # Project to d_model
            proj = self.input_projections[i](tensor)
            embeddings.append(proj)
            
        # Concatenate along sequence dimension (dim 0)
        # Result: (Total_Seq_Len, d_model)
        return torch.cat(embeddings, dim=0)
    
    def assemble_batch(self, batch_raw_states, device=None, dtype=torch.float32):
        """
        Assemble a batch of raw states into a single tensor.
        batch_raw_states: List[List[Tensor]]
        Returns: (Batch, Seq_Len, d_model)
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Transpose list of lists: [[m1, m2...], [m1, m2...]] -> [[m1, m1...], [m2, m2...]]
        num_matrices = len(self.matrix)
        batched_sources = [[] for _ in range(num_matrices)]
        
        for state in batch_raw_states:
            for i, m in enumerate(state):
                batched_sources[i].append(m)
                
        embeddings = []
        for i, matrix_list in enumerate(batched_sources):
            # Optimization: Stack on CPU first, then move to GPU
            # This avoids creating thousands of small tensors on GPU and reduces fragmentation
            # Assuming m is on CPU (which it should be from get_raw_state)
            
            # Check first element to see if it's already on device
            if matrix_list and matrix_list[0].device != device:
                 stacked = torch.stack(matrix_list).to(device=device, dtype=dtype)
            else:
                 stacked = torch.stack([m.to(device=device, dtype=dtype) for m in matrix_list])
            
            # Project: (Batch, Rows, d_model)
            proj = self.input_projections[i](stacked)
            embeddings.append(proj)
            
        # Concatenate along sequence dimension (dim 1)
        return torch.cat(embeddings, dim=1)

    def forward(self, x=None, raw_state=None, batch_raw_states=None):
        # If x is not provided, assemble from internal state or provided raw_state
        if batch_raw_states is not None:
            x = self.assemble_batch(batch_raw_states)
            # x is (Batch, Seq_Len, Dim)
        elif x is None:
            x = self.assemble_input(raw_state=raw_state)
            # x is (Seq_Len, Dim) -> unsqueeze -> (1, Seq_Len, Dim)
            x = x.unsqueeze(0)
            
        # Prepend [CLS] token
        batch_size = x.size(0)
        # Expand cls_token to match batch size: (Batch, 1, Dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate: (Batch, 1 + Seq_Len, Dim)
        x = torch.cat((cls_tokens, x), dim=1)
            
        output = self.transformer_encoder(x)
        
        # Use [CLS] token output (index 0) as the pooled representation
        # This allows the model to learn how to aggregate info via attention
        # instead of a blind mean pooling.
        pooled = output[:, 0, :] 
        
        logits = self.output_head(pooled)
        value = self.value_head(pooled)
        
        return logits, value

