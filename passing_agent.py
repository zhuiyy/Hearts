"""
Passing Agent - handles card passing decisions and integrates with game.
"""

import torch
from data_structure import Card, Suit, PassDirection


class PassingAgent:
    """Agent that uses PassingNetwork to select cards to pass."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # Buffers for PPO training
        self.saved_log_probs = []  # List of [3] tensors
        self.saved_values = []     # List of [3] tensors
        self.saved_actions = []    # List of [3] tensors
        self.saved_hand_vecs = []
        self.saved_pass_dir_vecs = []
        self.saved_hand_masks = []
        self.rewards = []          # One reward per episode (game)
    
    def reset(self):
        """Reset buffers for new training batch."""
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_actions = []
        self.saved_hand_vecs = []
        self.saved_pass_dir_vecs = []
        self.saved_hand_masks = []
        self.rewards = []
    
    def make_pass_features(self, hand, pass_direction):
        """
        Create input features for passing network.
        
        Args:
            hand: list of Card objects (13 cards)
            pass_direction: PassDirection enum
        
        Returns:
            hand_vec: [52] one-hot of hand cards
            pass_dir_vec: [4] one-hot of pass direction
            hand_mask: [52] mask for valid selections
        """
        # Hand vector
        hand_vec = torch.zeros(52, dtype=torch.float32, device=self.device)
        for card in hand:
            hand_vec[card.to_id()] = 1.0
        
        # Pass direction one-hot
        pass_dir_vec = torch.zeros(4, dtype=torch.float32, device=self.device)
        pass_dir_vec[pass_direction.value] = 1.0
        
        # Mask: -inf for cards not in hand
        hand_mask = torch.full((52,), float('-inf'), dtype=torch.float32, device=self.device)
        for card in hand:
            hand_mask[card.to_id()] = 0.0
        
        return hand_vec, pass_dir_vec, hand_mask
    
    def select_cards(self, player, info, deterministic=False):
        """
        Select 3 cards to pass. Compatible with game.py's pass_policy interface.
        
        Args:
            player: Player object with player.hand
            info: game info dict (contains 'pass_direction')
            deterministic: if True, always pick best; else sample
        
        Returns:
            cards_to_pass: list of 3 Card objects
        """
        pass_direction = info.get('pass_direction', PassDirection.KEEP)
        
        if pass_direction == PassDirection.KEEP:
            # No passing in KEEP round - return empty (game.py handles this)
            return []
        
        hand = list(player.hand)
        hand_vec, pass_dir_vec, hand_mask = self.make_pass_features(hand, pass_direction)
        
        # Add batch dimension
        hand_vec = hand_vec.unsqueeze(0)
        pass_dir_vec = pass_dir_vec.unsqueeze(0)
        hand_mask = hand_mask.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            selected, log_probs, values = self.model.select_three_cards(
                hand_vec, pass_dir_vec, hand_mask, deterministic=deterministic
            )
        
        # Convert indices to Card objects
        card_indices = selected[0].cpu().numpy()  # [3]
        cards_to_pass = [Card.from_id(int(idx)) for idx in card_indices]
        
        return cards_to_pass
    
    def select_cards_training(self, player, info):
        """
        Select cards during training (stores tensors for gradient computation).
        
        Args:
            player: Player object with player.hand
            info: game info dict
        
        Returns:
            cards_to_pass: list of 3 Card objects
        """
        pass_direction = info.get('pass_direction', PassDirection.KEEP)
        
        if pass_direction == PassDirection.KEEP:
            return []
        
        hand = list(player.hand)
        hand_vec, pass_dir_vec, hand_mask = self.make_pass_features(hand, pass_direction)
        
        hand_vec_b = hand_vec.unsqueeze(0)
        pass_dir_vec_b = pass_dir_vec.unsqueeze(0)
        hand_mask_b = hand_mask.unsqueeze(0)
        
        self.model.train()
        selected, log_probs, values = self.model.select_three_cards(
            hand_vec_b, pass_dir_vec_b, hand_mask_b, deterministic=False
        )
        
        # Store for PPO update
        self.saved_log_probs.append(log_probs[0])  # [3]
        self.saved_values.append(values[0])        # [3]
        self.saved_actions.append(selected[0])     # [3]
        self.saved_hand_vecs.append(hand_vec)
        self.saved_pass_dir_vecs.append(pass_dir_vec)
        self.saved_hand_masks.append(hand_mask)
        
        # Convert indices to Card objects
        card_indices = selected[0].detach().cpu().numpy()
        cards_to_pass = [Card.from_id(int(idx)) for idx in card_indices]
        
        return cards_to_pass
    
    def add_reward(self, reward):
        """Add reward after game ends."""
        self.rewards.append(reward)
    
    def get_training_data(self):
        """
        Get all stored data for PPO update.
        
        Returns:
            dict with batched tensors
        """
        if len(self.saved_actions) == 0:
            return None
        
        return {
            'hand_vecs': torch.stack(self.saved_hand_vecs),       # [N, 52]
            'pass_dir_vecs': torch.stack(self.saved_pass_dir_vecs), # [N, 4]
            'hand_masks': torch.stack(self.saved_hand_masks),     # [N, 52]
            'actions': torch.stack(self.saved_actions),           # [N, 3]
            'old_log_probs': torch.stack(self.saved_log_probs),   # [N, 3]
            'old_values': torch.stack(self.saved_values),         # [N, 3]
            'rewards': torch.tensor(self.rewards, device=self.device),  # [N]
        }


def add_from_id_to_card():
    """Add from_id class method to Card if it doesn't exist."""
    if not hasattr(Card, 'from_id'):
        @classmethod
        def from_id(cls, card_id):
            suit = Suit(card_id // 13)
            rank = (card_id % 13) + 1
            return cls(suit, rank)
        Card.from_id = from_id


# Auto-add the method when this module is imported
add_from_id_to_card()
