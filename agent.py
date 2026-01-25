import torch
import random
from collections import deque
from data_structure import Card, Suit
from model import HeartsLSTM

class SimpleFCNAgent: # Keeping the name to minimize refactor, but it's now LSTM
    def __init__(self, model: HeartsLSTM, device='cpu'):
        self.model = model
        self.device = device
        
        # Buffers for PPO
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_actions = []
        self.rewards = []
        
        # Re-evaluation buffers (LSTM needs Sequence)
        # We will store: List of Tensors.
        # But wait, training LSTM with PPO is tricky.
        # Simpler approach: 
        # During rollout, we feed features one by one and keep hidden state.
        # During update, we might need to maximize sequence length re-computation or use truncated BPTT.
        # For simplicity in this project: We will store (HistorySequence) for each step.
        # Since seq length is max 52, we can just store the FULL sequence up to that point for every step.
        # It's memory heavy but fine for 10000 steps.
        
        self.saved_state_seqs = [] # List of [1, SeqLen, Dim]
        self.saved_global_priv = []
        self.saved_masks = []
        self.saved_qs_labels = []
        
        # LSTM Hidden State container for rollout
        self.current_hidden = None
        
        # Episode History Buffer (to build sequence)
        self.episode_obs_history = [] 

    def reset(self):
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_actions = []
        self.rewards = []
        self.saved_state_seqs = []
        self.saved_global_priv = []
        self.saved_masks = []
        self.saved_qs_labels = []
        
        self.episode_obs_history = []
        self.current_hidden = None

    def reset_episode_memory(self):
        """Clears the history for the current episode (game), but keeps PPO buffers."""
        self.episode_obs_history = []
        self.current_hidden = None

    def preprocess_obs(self, info):
        # Full Feature Engineering (241 dim)
        
        # 1. Hand (52)
        hand_vec = torch.zeros(52)
        my_hand_ids = set()
        for c in info['hand']:
            cid = c.to_id()
            hand_vec[cid] = 1.0
            my_hand_ids.add(cid)
            
        # 2. Table (Current Trick) (52)
        table_vec = torch.zeros(52)
        current_trick_points = 0
        has_sq = False
        has_hearts = False
        
        for c, _ in info['current_table']:
            table_vec[c.to_id()] = 1.0
            current_trick_points += c.value()
            if c.suit == Suit.SPADES and c.rank == 12: has_sq = True
            if c.suit == Suit.HEARTS: has_hearts = True
            
        # Analyze History: who played what?
        # info['full_history'] contains (Card, player_id) for all past tricks
        full_history = info.get('full_history', [])
        
        # 3. My Played History (52)
        # 4. Others Played History (52)
        history_me = torch.zeros(52)
        history_others = torch.zeros(52)
        
        # 9. Void Inference (Others) (16)
        # Track what logic implies about voids.
        # This is complex. Simplified logic:
        # If someone failed to follow suit, they are void.
        
        voids = torch.zeros((4, 4)) # [player, suit]
        
        # Danger Card Tracking
        sq_out = 0.0
        sk_out = 0.0
        sa_out = 0.0
        
        # Replay history to find voids
        # We need to know who led each trick to know "following suit" logic.
        # But 'full_history' is just a flat list. GameV2 stacks them in order.
        # We can reconstruct tricks. Each trick has 4 cards (except current).
        
        processed_cards = 0
        total_played = len(full_history)
        
        completed_tricks_count = total_played // 4
        
        for t in range(completed_tricks_count):
            trick = full_history[t*4 : (t+1)*4]
            lead_card, lead_pid = trick[0]
            lead_suit = lead_card.suit
            
            for c, pid in trick:
                # Mark history
                cid = c.to_id()
                if pid == info['player_id']:
                    history_me[cid] = 1.0
                else:
                    history_others[cid] = 1.0
                    
                # Track Danger
                if c.suit == Suit.SPADES:
                    if c.rank == 12: sq_out = 1.0
                    if c.rank == 13: sk_out = 1.0
                    if c.rank == 1: sa_out = 1.0
                
                # Void Logic
                if c.suit != lead_suit:
                    # Player failed to follow suit -> Void in lead_suit
                    if pid < 4: # Safety
                        voids[pid][lead_suit] = 1.0
        
        # Flatten Voids
        void_vec = voids.flatten()
        
        # 5. Scores (4)
        scores_vec = torch.tensor(info['scoreboard'], dtype=torch.float32) / 26.0
        
        # 6. Current Trick Info (6)
        # [Has Hearts?, Has SQ?, My Rank High?, Others Rank High?, Count, Is Lead?]
        trick_info = torch.zeros(6)
        if has_hearts: trick_info[0] = 1.0
        if has_sq: trick_info[1] = 1.0
        
        current_count = len(info['current_table'])
        trick_info[4] = current_count / 4.0
        trick_info[5] = 1.0 if current_count == 0 else 0.0 # Is Lead
        
        # Rank Info
        if current_count > 0:
            lead_suit_current = info['current_table'][0][0].suit
            
            # Max rank on table in lead suit
            max_rank = -1
            for c, _ in info['current_table']:
                if c.suit == lead_suit_current:
                    r = 14 if c.rank == 1 else c.rank
                    if r > max_rank: max_rank = r
            
            # Can I beat it? (Rough check)
            # Find my highest legal card in that suit
            my_highest = -1
            # We don't have legal_actions passed to preprocess_obs usually, 
            # but we can infer from hand.
            for c in info['hand']:
                if c.suit == lead_suit_current:
                    r = 14 if c.rank == 1 else c.rank
                    if r > my_highest: my_highest = r
            
            if my_highest > max_rank: trick_info[2] = 1.0 # I can win
            trick_info[3] = max_rank / 14.0 # Strength of current winner
            
        # 7. Danger Cards Status (3)
        # Also check current trick for danger cards
        for c, _ in info['current_table']:
            if c.suit == Suit.SPADES:
                if c.rank == 12: sq_out = 1.0
                if c.rank == 13: sk_out = 1.0
                if c.rank == 1: sa_out = 1.0
                
        danger_vec = torch.tensor([sq_out, sk_out, sa_out])
        
        # 8. Suit Counts (My Hand) (4)
        suit_counts = torch.zeros(4)
        for c in info['hand']:
            suit_counts[c.suit] += 1
        suit_counts = suit_counts / 13.0
        
        # 9. 换牌信息 (52 + 52 + 4 = 108维)
        # 9a. 我传出去的牌 (52维 one-hot)
        passed_vec = torch.zeros(52)
        passed_cards = info.get('passed_cards', [])
        for c in passed_cards:
            passed_vec[c.to_id()] = 1.0
            
        # 9b. 我收到的牌 (52维 one-hot)
        received_vec = torch.zeros(52)
        received_cards = info.get('received_cards', [])
        for c in received_cards:
            received_vec[c.to_id()] = 1.0
            
        # 9c. 传牌方向 (4维 one-hot: LEFT, RIGHT, ACROSS, KEEP)
        from data_structure import PassDirection
        pass_dir_vec = torch.zeros(4)
        pass_dir = info.get('pass_direction', PassDirection.KEEP)
        pass_dir_vec[pass_dir.value] = 1.0
        
        # ========== 新增对手建模特征 ==========
        
        # 10. Opponent Suit Counts Estimate (12维: 3 opponents * 4 suits)
        # 估计每个对手各花色剩余牌数
        # 初始每人13张，平均每花色3.25张，根据历史调整
        opponent_suit_counts = torch.zeros(3, 4)
        
        # 统计已打出的牌 (按花色和玩家)
        cards_played_by_suit = [0, 0, 0, 0]  # 各花色已打出总数
        my_id = info['player_id']
        
        for c, pid in full_history:
            cards_played_by_suit[c.suit] += 1
        
        # 当前trick也算
        for c, pid in info['current_table']:
            cards_played_by_suit[c.suit] += 1
        
        # 计算各花色剩余牌 (13 - 已打出)
        remaining_by_suit = [13 - cards_played_by_suit[s] for s in range(4)]
        
        # 估计对手持牌 (剩余牌 - 我手里的) / 3
        my_suit_counts_raw = [0, 0, 0, 0]
        for c in info['hand']:
            my_suit_counts_raw[c.suit] += 1
        
        opp_idx = 0
        for pid in range(4):
            if pid == my_id:
                continue
            for suit in range(4):
                # 如果对手已知void，则为0
                if voids[pid][suit] > 0:
                    opponent_suit_counts[opp_idx][suit] = 0
                else:
                    # 估计: (剩余 - 我的) / 非void对手数
                    non_void_opps = sum(1 for p in range(4) if p != my_id and voids[p][suit] == 0)
                    if non_void_opps > 0:
                        estimate = (remaining_by_suit[suit] - my_suit_counts_raw[suit]) / non_void_opps
                        opponent_suit_counts[opp_idx][suit] = max(0, estimate) / 13.0  # 归一化
            opp_idx += 1
        
        opponent_suit_vec = opponent_suit_counts.flatten()  # 12维
        
        # 11. SQ Location Probability (4维)
        # 追踪SQ在各玩家手中的概率
        sq_location = torch.zeros(4)
        sq_card_id = 11 * 4 + 0  # Q♠ = rank 12 (index 11), suit 0
        
        if sq_out > 0:
            # SQ已经打出，概率都是0
            pass
        elif hand_vec[sq_card_id] > 0:
            # 我手里有SQ
            sq_location[my_id] = 1.0
        else:
            # SQ在某个对手手里，根据void推断
            for pid in range(4):
                if pid == my_id:
                    continue
                if voids[pid][Suit.SPADES] > 0:
                    # 对手已void黑桃，不可能有SQ
                    sq_location[pid] = 0.0
                else:
                    sq_location[pid] = 1.0
            # 归一化
            total = sq_location.sum()
            if total > 0:
                sq_location = sq_location / total
        
        # 12. Current Trick Winner Prediction (4维)
        # 预测当前trick谁会赢
        trick_winner_pred = torch.zeros(4)
        if current_count > 0:
            lead_suit = info['current_table'][0][0].suit
            winning_pid = -1
            winning_rank = -1
            for c, pid in info['current_table']:
                if c.suit == lead_suit:
                    r = 14 if c.rank == 1 else c.rank
                    if r > winning_rank:
                        winning_rank = r
                        winning_pid = pid
            if winning_pid >= 0:
                trick_winner_pred[winning_pid] = 1.0
        
        # 13. Game Progress (1维)
        # 当前是第几轮 (1-13)
        tricks_completed = completed_tricks_count
        game_progress = torch.tensor([tricks_completed / 13.0])
        
        # 14. Remaining Cards Per Suit (4维)
        remaining_suit_vec = torch.tensor(remaining_by_suit, dtype=torch.float32) / 13.0
        
        # 15. Points at Risk (1维)
        # 当前trick桌上有多少分
        points_at_risk = torch.tensor([current_trick_points / 26.0])
        
        # Concatenate All (349 + 26 = 375维)
        state_vec = torch.cat([
            hand_vec,              # 52
            table_vec,             # 52
            history_me,            # 52
            history_others,        # 52
            scores_vec,            # 4
            trick_info,            # 6
            danger_vec,            # 3
            suit_counts,           # 4
            void_vec,              # 16
            passed_vec,            # 52
            received_vec,          # 52
            pass_dir_vec,          # 4
            opponent_suit_vec,     # 12 (新增)
            sq_location,           # 4  (新增)
            trick_winner_pred,     # 4  (新增)
            game_progress,         # 1  (新增)
            remaining_suit_vec,    # 4  (新增)
            points_at_risk         # 1  (新增)
        ])
        
        return state_vec
    
    def pass_policy(self, player, info):
        """
        Pass policy - can use either learned PassingNetwork or fallback to Expert.
        
        To use learned passing:
            agent.set_passing_agent(passing_agent)
        """
        # If we have a passing agent, use it
        if hasattr(self, 'passing_agent') and self.passing_agent is not None:
            return self.passing_agent.select_cards(player, info, deterministic=False)
        
        # Fallback to expert policy
        from strategies import ExpertPolicy
        return ExpertPolicy.pass_policy(player, info)
    
    def set_passing_agent(self, passing_agent):
        """Set the PassingAgent for learned passing."""
        self.passing_agent = passing_agent
    
    def pass_policy_training(self, player, info):
        """
        Pass policy for training - stores gradients for PPO.
        """
        if hasattr(self, 'passing_agent') and self.passing_agent is not None:
            return self.passing_agent.select_cards_training(player, info)
        
        from strategies import ExpertPolicy
        return ExpertPolicy.pass_policy(player, info)

    def act(self, player, info, legal_actions, order, training=True):
        state_vec = self.preprocess_obs(info)
        
        # Append to episode history
        self.episode_obs_history.append(state_vec)
        
        # Build sequence: [1, SeqLen, Dim]
        seq_tensor = torch.stack(self.episode_obs_history).unsqueeze(0).to(self.device)
        
        global_priv_b = None
        if training and 'global_state' in info:
            global_priv_b = info['global_state'].unsqueeze(0).to(self.device)
            
        # Forward pass with LSTM
        # We pass self.current_hidden. If input is full sequence, we might want to scan?
        # Actually for inference efficiency, we can just pass the NEWEST step and keep hidden.
        # But for training consistency (and PPO update simplicity), let's feed full sequence.
        # Optimization: Feed full seq and ignore returned hidden for next step (since we rebuild seq).
        # This is slower O(N^2) but safer implementation.
        logits, value, qs_pred, _ = self.model(seq_tensor, global_priv_b, hidden=None)
        
        # Masking
        mask = torch.full((52,), float('-inf'), device=self.device)
        legal_indices = [c.to_id() for c in legal_actions]
        
        # Safety check: must have at least one legal action
        if len(legal_indices) == 0:
            legal_indices = [0]  # Fallback
            
        mask[legal_indices] = 0
        
        masked_logits = logits.squeeze() + mask
        
        # Safety check for NaN/Inf
        if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).all():
            # Fallback to uniform over legal actions
            probs = torch.zeros(52, device=self.device)
            probs[legal_indices] = 1.0 / len(legal_indices)
        else:
            probs = torch.softmax(masked_logits, dim=0)
        
        # Epsilon Greedy Exploration
        curr_actions = legal_indices
        if training and random.random() < 0.05:
            action_idx = torch.tensor(random.choice(curr_actions), device=self.device)
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
        
        if training:
            dist = torch.distributions.Categorical(probs)
            self.saved_log_probs.append(dist.log_prob(action_idx))
            
            if value is not None:
                self.saved_values.append(value.squeeze())
            else:
                self.saved_values.append(torch.tensor(0.0).to(self.device))
                
            self.saved_actions.append(action_idx)
            # Store the current sequence snapshot!
            # We copy it to CPU to save GPU memory if needed, but keeping on device for speed
            self.saved_state_seqs.append(seq_tensor) 
            self.saved_masks.append(mask)
            self.saved_qs_labels.append(torch.tensor(info.get('sq_label', 4), device=self.device))
            
            if global_priv_b is not None:
                self.saved_global_priv.append(global_priv_b.squeeze())
            else:
                self.saved_global_priv.append(torch.zeros(156).to(self.device))
        
        # Decode action
        chosen_id = action_idx.item()
        # Find card object
        for c in legal_actions:
            if c.to_id() == chosen_id:
                return c
        return legal_actions[0] # Fallback
