"""
Pre-train the Passing Network using DAgger.

Usage:
    python pretrain_passing.py          # Train
    python pretrain_passing.py eval     # Evaluate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
import sys
from tqdm import tqdm

from passing_model import PassingNetwork
from passing_agent import PassingAgent
from data_structure import Card, Suit, PassDirection
from strategies import ExpertPolicy
import config


class PassingDataset(Dataset):
    """Dataset of (hand, pass_direction, expert_selection) tuples."""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        hand_vec, pass_dir_vec, hand_mask, expert_cards = self.data[idx]
        return hand_vec, pass_dir_vec, hand_mask, expert_cards


def deal_random_hand():
    """Deal a random 13-card hand."""
    deck = [Card(suit, rank) for suit in Suit for rank in range(1, 14)]
    random.shuffle(deck)
    return deck[:13]


class DummyPlayer:
    """Dummy player object for ExpertPolicy compatibility."""
    def __init__(self, hand):
        self.hand = hand


def collect_expert_pass_data(num_samples=50000, use_dagger=False, agent=None, beta=0.5):
    """
    Collect expert passing decisions.
    
    Args:
        num_samples: number of hands to collect
        use_dagger: if True, sometimes let agent make decisions (for distribution)
        agent: PassingAgent (required if use_dagger=True)
        beta: probability of using expert (for DAgger)
    
    Returns:
        list of (hand_vec, pass_dir_vec, hand_mask, expert_cards)
    """
    print(f"Collecting {num_samples} expert passing decisions...")
    if use_dagger:
        print(f"  Using DAgger with beta={beta}")
    
    data = []
    
    for _ in tqdm(range(num_samples)):
        # Random hand and direction (skip KEEP since no passing)
        hand = deal_random_hand()
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS])
        
        # Create dummy player for ExpertPolicy
        player = DummyPlayer(hand)
        info = {'pass_direction': pass_dir}
        
        # Get expert selection (always the label)
        expert_cards = ExpertPolicy.pass_policy(player, info)
        expert_ids = torch.tensor([c.to_id() for c in expert_cards], dtype=torch.long)
        
        # Build features
        hand_vec = torch.zeros(52, dtype=torch.float32)
        for card in hand:
            hand_vec[card.to_id()] = 1.0
        
        pass_dir_vec = torch.zeros(4, dtype=torch.float32)
        pass_dir_vec[pass_dir.value] = 1.0
        
        hand_mask = torch.full((52,), float('-inf'), dtype=torch.float32)
        for card in hand:
            hand_mask[card.to_id()] = 0.0
        
        data.append((hand_vec, pass_dir_vec, hand_mask, expert_ids))
    
    return data


def compute_sequential_loss(model, hand_vec, pass_dir_vec, hand_mask, expert_cards, device):
    """
    Compute loss for sequential card selection.
    
    The expert_cards are [B, 3] indices. We compute cross-entropy
    for each step, updating the mask appropriately (teacher forcing).
    """
    batch_size = hand_vec.size(0)
    
    selected_vec = torch.zeros(batch_size, 52, device=device)
    current_mask = hand_mask.clone()
    
    total_loss = 0
    correct = 0
    
    for step in range(3):
        # Forward
        logits, _ = model(hand_vec, selected_vec, pass_dir_vec, current_mask)
        
        # Target for this step
        target = expert_cards[:, step]
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(logits, target)
        total_loss = total_loss + loss
        
        # Accuracy
        pred = logits.argmax(dim=-1)
        correct += (pred == target).sum().item()
        
        # Update selected_vec and mask for next step (teacher forcing)
        selected_vec = selected_vec.scatter(1, target.unsqueeze(1), 1.0)
        current_mask = current_mask.scatter(1, target.unsqueeze(1), float('-inf'))
    
    return total_loss / 3, correct / (3 * batch_size)


def evaluate_passing(model, device, num_samples=1000):
    """Evaluate passing accuracy."""
    model.eval()
    
    correct_cards = 0
    total_cards = 0
    exact_match = 0
    total_hands = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            hand = deal_random_hand()
            pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS])
            
            player = DummyPlayer(hand)
            info = {'pass_direction': pass_dir}
            expert_cards = ExpertPolicy.pass_policy(player, info)
            expert_set = set(c.to_id() for c in expert_cards)
            
            # Model prediction
            hand_vec = torch.zeros(1, 52, device=device)
            for card in hand:
                hand_vec[0, card.to_id()] = 1.0
            
            pass_dir_vec = torch.zeros(1, 4, device=device)
            pass_dir_vec[0, pass_dir.value] = 1.0
            
            hand_mask = torch.full((1, 52), float('-inf'), device=device)
            for card in hand:
                hand_mask[0, card.to_id()] = 0.0
            
            selected, _, _ = model.select_three_cards(
                hand_vec, pass_dir_vec, hand_mask, deterministic=True
            )
            pred_set = set(selected[0].cpu().numpy().tolist())
            
            # Count correct
            correct = len(expert_set & pred_set)
            correct_cards += correct
            total_cards += 3
            
            if expert_set == pred_set:
                exact_match += 1
            total_hands += 1
    
    card_acc = correct_cards / total_cards
    exact_acc = exact_match / total_hands
    
    print(f"Card Accuracy: {card_acc:.4f} ({correct_cards}/{total_cards})")
    print(f"Exact Match:   {exact_acc:.4f} ({exact_match}/{total_hands})")
    
    return card_acc, exact_acc


def pretrain_passing(num_samples=50000, epochs=20, batch_size=256, lr=1e-3, dagger_rounds=3):
    """
    Pre-train PassingNetwork using behavior cloning + DAgger.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pre-training PassingNetwork on {device}")
    
    # Initialize model
    model = PassingNetwork(hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    agent = PassingAgent(model, device)
    
    best_acc = 0
    all_data = []
    
    for dagger_iter in range(dagger_rounds + 1):
        if dagger_iter == 0:
            print(f"\n{'='*50}")
            print(f"Round 0: Pure Behavior Cloning")
            print(f"{'='*50}")
            new_data = collect_expert_pass_data(num_samples, use_dagger=False)
        else:
            beta = max(0.3, 1.0 - dagger_iter * 0.2)
            print(f"\n{'='*50}")
            print(f"Round {dagger_iter}: DAgger with beta={beta}")
            print(f"{'='*50}")
            new_data = collect_expert_pass_data(
                num_samples // 2,
                use_dagger=True,
                agent=agent,
                beta=beta
            )
        
        all_data.extend(new_data)
        print(f"Total dataset size: {len(all_data)}")
        
        dataset = PassingDataset(all_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        round_epochs = epochs if dagger_iter == 0 else max(5, epochs // 2)
        
        for epoch in range(round_epochs):
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{round_epochs}")
            for hand_vec, pass_dir_vec, hand_mask, expert_cards in pbar:
                hand_vec = hand_vec.to(device)
                pass_dir_vec = pass_dir_vec.to(device)
                hand_mask = hand_mask.to(device)
                expert_cards = expert_cards.to(device)
                
                # Compute sequential loss
                loss, acc = compute_sequential_loss(
                    model, hand_vec, pass_dir_vec, hand_mask, expert_cards, device
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                batch_size_actual = hand_vec.size(0)
                total_loss += loss.item() * batch_size_actual
                total_correct += acc * batch_size_actual
                total_samples += batch_size_actual
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{total_correct/total_samples:.3f}'
                })
            
            epoch_loss = total_loss / total_samples
            epoch_acc = total_correct / total_samples
            
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch,
                    'dagger_round': dagger_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': epoch_acc,
                }, config.PASSING_PRETRAINED_PATH)
                print(f"  -> Saved best model (acc={epoch_acc:.4f})")
        
        # Evaluate after each round
        print(f"\n--- Evaluation after Round {dagger_iter} ---")
        evaluate_passing(model, device, num_samples=2000)
    
    print(f"\n{'='*50}")
    print(f"Pre-training Complete! Best Accuracy: {best_acc:.4f}")
    print(f"Model saved to: {config.PASSING_PRETRAINED_PATH}")
    print(f"{'='*50}")
    
    return model


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        # Evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PassingNetwork(hidden_dim=256).to(device)
        
        if os.path.exists(config.PASSING_PRETRAINED_PATH):
            checkpoint = torch.load(config.PASSING_PRETRAINED_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {config.PASSING_PRETRAINED_PATH}")
            print(f"Training accuracy was: {checkpoint.get('accuracy', 'N/A')}")
        else:
            print("No pretrained model found!")
            return
        
        print("\n===== Evaluation Results =====")
        evaluate_passing(model, device, num_samples=5000)
        
    else:
        # Training mode
        pretrain_passing(
            num_samples=50000,
            epochs=15,
            batch_size=256,
            lr=1e-3,
            dagger_rounds=3
        )


if __name__ == "__main__":
    main()
