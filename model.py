import torch
import torch.nn as nn
import torch.nn.functional as F

class HeartsLSTM(nn.Module):
    def __init__(self, input_dim=375, hidden_dim=512, lstm_layers=2):
        super().__init__()
        
        # 1. Feature Extractor (Shared) - Enhanced with residual connection
        self.fc_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. LSTM Core - 2 layers for better temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        # 3. Policy Head - Deeper for complex decision making
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 52)
        )
        
        # 4. Value Head (Critic) - Much deeper for accurate value estimation
        # Critical for PPO: better value estimates = better advantage estimates
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + 156, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 5. Aux Head - Predict who has SQ
        self.qs_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # 6. NEW: Opponent Modeling Head - Predict what opponents might play
        # This encourages learning opponent patterns
        self.opponent_pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 52 * 3)  # 预测3个对手的出牌概率
        )

    def forward(self, state_seq, global_priv_info=None, hidden=None, lengths=None):
        """
        state_seq: [Batch, SeqLen, InputDim]
        hidden: (h_0, c_0) for LSTM
        """
        # Embed features
        # [B, S, I] -> [B, S, H]
        batch_size, seq_len, _ = state_seq.size()
        
        x = self.fc_embed(state_seq)
        
        # LSTM
        # out: [B, S, H], hidden: ([L, B, H], [L, B, H])
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # Take the last real timestep. Batched training pads sequences, while
        # rollout/inference passes unpadded prefixes.
        if lengths is None:
            last_out = lstm_out[:, -1, :]
        else:
            lengths = lengths.to(state_seq.device).clamp(min=1, max=seq_len)
            batch_idx = torch.arange(batch_size, device=state_seq.device)
            last_out = lstm_out[batch_idx, lengths - 1, :]
        
        # Policy
        logits = self.policy_head(last_out)
        
        # Aux
        qs_pred = self.qs_head(last_out)
        
        # Value
        value = None
        if global_priv_info is not None:
            # global_priv_info is usually [Batch, 156] (for the current step)
            # We concat with the LSTM summary of history
            critic_input = torch.cat([last_out, global_priv_info], dim=1)
            value = self.value_head(critic_input)
            
        return logits, value, qs_pred, new_hidden
