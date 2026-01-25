"""
Passing Network for Hearts AI

A simple MLP that learns to select 3 cards to pass.
Uses sequential selection: pick 1st card, then 2nd, then 3rd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PassingNetwork(nn.Module):
    """
    MLP network for selecting 3 cards to pass.
    
    Uses sequential selection to capture dependencies:
    - Step 1: Select 1st card from 13 hand cards
    - Step 2: Select 2nd card from remaining 12
    - Step 3: Select 3rd card from remaining 11
    
    This allows the network to learn things like:
    "If I already passed Q♠, I don't need to pass K♠"
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # Input features:
        # - 52 dim: hand cards (one-hot)
        # - 52 dim: already selected cards (one-hot, for steps 2&3)
        # - 4 dim: pass direction (LEFT/RIGHT/ACROSS/KEEP)
        # Total: 108 dim
        self.input_dim = 108
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output: score for each of 52 cards
        self.card_head = nn.Linear(hidden_dim, 52)
        
        # Value head for RL (estimates expected negative score)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, hand_vec, selected_vec, pass_dir_vec, hand_mask):
        """
        Forward pass for one selection step.
        
        Args:
            hand_vec: [B, 52] one-hot of hand cards
            selected_vec: [B, 52] one-hot of already selected cards (0 for step 1)
            pass_dir_vec: [B, 4] one-hot of pass direction
            hand_mask: [B, 52] mask where -inf for unavailable cards, 0 for available
        
        Returns:
            logits: [B, 52] selection logits (masked)
            value: [B, 1] state value estimate
        """
        # Concatenate inputs
        x = torch.cat([hand_vec, selected_vec, pass_dir_vec], dim=-1)  # [B, 108]
        
        # Encode
        features = self.encoder(x)  # [B, hidden_dim]
        
        # Card selection logits
        logits = self.card_head(features)  # [B, 52]
        
        # Apply mask (can't select cards not in hand or already selected)
        masked_logits = logits + hand_mask
        
        # Value estimate
        value = self.value_head(features)  # [B, 1]
        
        return masked_logits, value
    
    def select_three_cards(self, hand_vec, pass_dir_vec, hand_mask, deterministic=False):
        """
        Select 3 cards to pass using sequential selection.
        
        Args:
            hand_vec: [B, 52] one-hot of hand cards
            pass_dir_vec: [B, 4] one-hot of pass direction
            hand_mask: [B, 52] initial mask (-inf for cards not in hand)
            deterministic: if True, always pick highest prob; else sample
        
        Returns:
            selected_cards: [B, 3] indices of selected cards
            log_probs: [B, 3] log probabilities of each selection
            values: [B, 3] value estimates at each step
        """
        batch_size = hand_vec.size(0)
        device = hand_vec.device
        
        selected_cards = []
        log_probs = []
        values = []
        
        selected_vec = torch.zeros(batch_size, 52, device=device)
        current_mask = hand_mask.clone()
        
        for step in range(3):
            # Forward pass
            logits, value = self.forward(hand_vec, selected_vec, pass_dir_vec, current_mask)
            
            # Safety check for NaN/Inf
            if torch.isnan(logits).any() or torch.isinf(logits).all():
                # Fallback: select from available cards uniformly
                valid_mask = (current_mask == 0)
                probs = valid_mask.float()
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                # Sample or argmax
                probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                card_idx = probs.argmax(dim=-1)
            else:
                # Clamp to avoid NaN in Categorical
                probs = probs.clamp(min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                dist = torch.distributions.Categorical(probs)
                card_idx = dist.sample()
            
            # Compute log prob
            log_prob = F.log_softmax(logits.clamp(min=-100, max=100), dim=-1)
            selected_log_prob = log_prob.gather(1, card_idx.unsqueeze(1)).squeeze(1)
            
            # Store results
            selected_cards.append(card_idx)
            log_probs.append(selected_log_prob)
            values.append(value.squeeze(1))
            
            # Update selected_vec and mask for next step
            selected_vec = selected_vec.scatter(1, card_idx.unsqueeze(1), 1.0)
            current_mask = current_mask.scatter(1, card_idx.unsqueeze(1), float('-inf'))
        
        selected_cards = torch.stack(selected_cards, dim=1)  # [B, 3]
        log_probs = torch.stack(log_probs, dim=1)  # [B, 3]
        values = torch.stack(values, dim=1)  # [B, 3]
        
        return selected_cards, log_probs, values
    
    def evaluate_actions(self, hand_vec, pass_dir_vec, hand_mask, actions):
        """
        Evaluate log probabilities of given actions (for PPO).
        
        Args:
            hand_vec: [B, 52]
            pass_dir_vec: [B, 4]
            hand_mask: [B, 52]
            actions: [B, 3] the 3 cards that were selected
        
        Returns:
            log_probs: [B, 3]
            values: [B, 3]
            entropy: [B] average entropy across 3 steps
        """
        batch_size = hand_vec.size(0)
        device = hand_vec.device
        
        log_probs = []
        values = []
        entropies = []
        
        selected_vec = torch.zeros(batch_size, 52, device=device)
        current_mask = hand_mask.clone()
        
        for step in range(3):
            logits, value = self.forward(hand_vec, selected_vec, pass_dir_vec, current_mask)
            
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            
            # Get log prob of the action taken
            action = actions[:, step]
            selected_log_prob = log_prob.gather(1, action.unsqueeze(1)).squeeze(1)
            
            # Entropy
            entropy = -(probs * log_prob).sum(dim=-1)
            
            log_probs.append(selected_log_prob)
            values.append(value.squeeze(1))
            entropies.append(entropy)
            
            # Update for next step (use actual action taken)
            selected_vec = selected_vec.scatter(1, action.unsqueeze(1), 1.0)
            current_mask = current_mask.scatter(1, action.unsqueeze(1), float('-inf'))
        
        log_probs = torch.stack(log_probs, dim=1)  # [B, 3]
        values = torch.stack(values, dim=1)  # [B, 3]
        entropy = torch.stack(entropies, dim=1).mean(dim=1)  # [B]
        
        return log_probs, values, entropy
