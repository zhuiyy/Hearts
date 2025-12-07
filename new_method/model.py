import torch
import torch.nn as nn
import torch.nn.functional as F

class HeartsProNet(nn.Module):
    def __init__(self, hidden_dim=512, lstm_hidden=128):
        super().__init__()
        
        # --- 1. Static Feature Encoder (Hand + Table + Voids) ---
        # Input: 
        # - Hand: 52 (Boolean)
        # - Table: 52 (Boolean, cards played in current trick)
        # - Voids: 16 (4 players * 4 suits, Boolean)
        # - Scores: 4 (Normalized scores)
        # Total Static Input: 52 + 52 + 16 + 4 = 124
        
        self.static_input_dim = 52 + 52 + 16 + 4
        
        self.card_encoder = nn.Sequential(
            nn.Linear(self.static_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- 2. Dynamic Feature Encoder (History Sequence) ---
        # Input: Sequence of played cards (Card ID + Player ID)
        # Embedding: Card (52) + Player (4) = 56 dim embedding? 
        # Or just one-hot 52 + one-hot 4.
        # Let's use an embedding layer for cards.
        
        self.card_embedding = nn.Embedding(53, 32) # 52 cards + 1 padding/start
        self.player_embedding = nn.Embedding(5, 8) # 4 players + 1 padding/unknown
        
        self.lstm_input_dim = 32 + 8
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=lstm_hidden, batch_first=True)
        
        # --- 3. Policy Head (Actor) ---
        # Input: Static Features + Dynamic Features
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 52) # Logits for 52 cards
        )
        
        # --- 4. Auxiliary Task Head (SQ Prediction) ---
        # Predict who has the Queen of Spades (0-3) + 1 for Played/Unknown
        self.qs_pred_head = nn.Linear(hidden_dim + lstm_hidden, 5)
        
        # --- 5. Value Head (Critic) - CTDE ---
        # Input: Local Features + Global Privileged Info
        # Global Info: Other 3 players' hands (3 * 52 = 156) + 4 * 52 = 208
        self.global_encoder = nn.Sequential(
            nn.Linear(208, 128),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(hidden_dim + lstm_hidden + 128, 1)

    def forward(self, 
                static_obs, 
                history_seq_cards, 
                history_seq_players, 
                global_priv_info=None):
        """
        static_obs: [Batch, 124]
        history_seq_cards: [Batch, SeqLen] (Indices 0-52)
        history_seq_players: [Batch, SeqLen] (Indices 0-4)
        global_priv_info: [Batch, 156] (Optional, for training only)
        """
        
        # 1. Static Features
        static_feat = self.card_encoder(static_obs)
        
        # 2. Dynamic Features (LSTM)
        card_emb = self.card_embedding(history_seq_cards)
        player_emb = self.player_embedding(history_seq_players)
        lstm_input = torch.cat([card_emb, player_emb], dim=2)
        
        # We only care about the final hidden state
        _, (h_n, _) = self.lstm(lstm_input)
        dynamic_feat = h_n[-1] # [Batch, lstm_hidden]
        
        # Combine
        features = torch.cat([static_feat, dynamic_feat], dim=1)
        
        # 3. Policy
        logits = self.policy_head(features)
        
        # 4. Aux Task
        qs_pred = self.qs_pred_head(features)
        
        value = None
        # 5. Value (Critic)
        if global_priv_info is not None:
            priv_feat = self.global_encoder(global_priv_info)
            critic_input = torch.cat([features, priv_feat], dim=1)
            value = self.value_head(critic_input)
            
        return logits, value, qs_pred
