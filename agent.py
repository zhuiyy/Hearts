import torch
import random
from collections import deque
from typing import List
from data_structure import Card, Suit
from transformer import HeartsTransformer

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
        
        # CTDE Data
        self.saved_global_states = []
        self.saved_sq_labels = []
        self.saved_void_labels = []

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
        
        # CTDE Data
        self.saved_global_states = []
        self.saved_sq_labels = []
        self.saved_void_labels = []

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
            
            # CTDE Collection (Pass Phase)
            # Even in pass phase, we can collect global state, though SQ/Void might be less relevant
            # But to keep batch alignment, we MUST collect something.
            # Let's collect real data if available, or zeros.
            global_state = info.get('global_state', None)
            sq_label = info.get('sq_label', 4)
            void_label = info.get('void_label', [0.0]*16)
            
            if global_state is not None:
                self.saved_global_states.append(global_state)
                # Forward pass with global state (for Value Head)
                logits, value, _, _ = self.model(x=None, global_state=global_state.unsqueeze(0))
            else:
                # Inference mode (no global state)
                # We append a dummy zero tensor to keep list length consistent, 
                # OR we handle it in train.py. 
                # Ideally, agent.py should be robust.
                self.saved_global_states.append(torch.zeros(208, device=self.device))
                logits, value, _, _ = self.model(x=None, global_state=None)

            self.saved_sq_labels.append(torch.tensor(sq_label, device=self.device))
            self.saved_void_labels.append(torch.tensor(void_label, device=self.device, dtype=torch.float32))

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
            
            self.saved_log_probs.append(dist.log_prob(action_to_play_idx).detach())
            if value is not None:
                self.saved_values.append(value.detach())
            else:
                self.saved_values.append(torch.tensor(0.0, device=self.device))
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
        
        # CTDE Collection (Play Phase)
        global_state = info.get('global_state', None)
        sq_label = info.get('sq_label', 4)
        void_label = info.get('void_label', [0.0]*16)
        
        if global_state is not None:
            self.saved_global_states.append(global_state)
            # Forward pass with global state
            logits, value, _, _ = self.model(x=None, global_state=global_state.unsqueeze(0))
        else:
            self.saved_global_states.append(torch.zeros(208, device=self.device))
            logits, value, _, _ = self.model(x=None, global_state=None)

        self.saved_sq_labels.append(torch.tensor(sq_label, device=self.device))
        self.saved_void_labels.append(torch.tensor(void_label, device=self.device, dtype=torch.float32))
        
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

        self.saved_log_probs.append(dist.log_prob(torch.tensor(action_to_play.to_id(), device=self.device)).detach())
        if value is not None:
            self.saved_values.append(value.detach())
        else:
            self.saved_values.append(torch.tensor(0.0, device=self.device))
        self.saved_actions.append(action_to_save) 
        
        return action_to_play
