"""
Joint Pre-training: PassingNetwork + HeartsLSTM using DAgger

Both networks train together from the start, so they learn to work with each other.
The playing network sees the actual cards passed/received (from either AI or Expert),
not just Expert's passing decisions.

Usage:
    python pretrain_joint.py          # Train both networks together
    python pretrain_joint.py eval     # Evaluate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
import sys
from tqdm import tqdm
from collections import deque

from game import GameV2
from model import HeartsLSTM
from passing_model import PassingNetwork
from agent import SimpleFCNAgent
from passing_agent import PassingAgent
from data_structure import Card, Suit, PassDirection
from strategies import ExpertPolicy
import config


class JointDataset(Dataset):
    """Dataset for joint training."""
    
    def __init__(self, pass_data, play_data):
        """
        pass_data: list of (hand_vec, pass_dir_vec, hand_mask, expert_cards)
        play_data: list of (state_seq, mask, action_id, global_priv)
        """
        self.pass_data = pass_data
        self.play_data = play_data
    
    def __len__(self):
        return max(len(self.pass_data), len(self.play_data))
    
    def get_pass_batch(self, indices):
        if len(self.pass_data) == 0:
            return None
        batch = [self.pass_data[i % len(self.pass_data)] for i in indices]
        hand_vecs = torch.stack([x[0] for x in batch])
        pass_dir_vecs = torch.stack([x[1] for x in batch])
        hand_masks = torch.stack([x[2] for x in batch])
        expert_cards = torch.stack([x[3] for x in batch])
        return hand_vecs, pass_dir_vecs, hand_masks, expert_cards
    
    def get_play_batch(self, indices):
        if len(self.play_data) == 0:
            return None
        batch = [self.play_data[i % len(self.play_data)] for i in indices]
        return batch


def pad_sequences(batch):
    """Collate function to pad sequences to the same length."""
    state_seqs, masks, action_ids, global_privs = zip(*batch)
    
    max_len = max(s.size(0) for s in state_seqs)
    max_len = max(max_len, 1)
    
    padded_seqs = []
    for seq in state_seqs:
        L = seq.size(0)
        if L < max_len:
            padding = torch.zeros(max_len - L, config.INPUT_DIM)
            padded = torch.cat([seq, padding], dim=0)
        else:
            padded = seq[:max_len]
        padded_seqs.append(padded)
    
    batch_seqs = torch.stack(padded_seqs)
    batch_masks = torch.stack(masks)
    batch_actions = torch.tensor(action_ids, dtype=torch.long)
    batch_global = torch.stack(global_privs)
    
    return batch_seqs, batch_masks, batch_actions, batch_global


def collect_joint_data(num_games, play_model, pass_model, play_agent, pass_agent, device, beta=0.5):
    """
    Collect data for both networks by playing actual games.
    
    Key insight: The playing network's training data includes the ACTUAL
    passed/received cards (from either AI or Expert decisions).
    
    Args:
        num_games: number of games to play
        play_model, pass_model: the neural networks
        play_agent, pass_agent: the agents
        beta: probability of using Expert (1.0 = pure BC, 0.0 = pure AI)
    
    Returns:
        pass_data: list of passing training samples
        play_data: list of playing training samples
    """
    print(f"Collecting {num_games} games (beta={beta})...")
    
    game = GameV2()
    pass_data = []
    play_data = []
    
    for game_idx in tqdm(range(num_games)):
        # Random pass direction (skip KEEP for passing training)
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS])
        
        # Reset episode memory
        play_agent.reset_episode_memory()
        episode_history = []  # For building sequences
        
        # ========== PASSING PHASE ==========
        # We need to handle passing specially since game.run_game_training does it internally
        # Let's manually do the passing phase to collect data properly
        
        game.reset()
        game.pass_direction = pass_dir
        
        passed_cards_all = [[], [], [], []]
        
        for player_id in range(4):
            player = game.gamestate.players[player_id]
            hand = list(player.hand)
            
            # Build features for passing
            hand_vec = torch.zeros(52, dtype=torch.float32)
            for card in hand:
                hand_vec[card.to_id()] = 1.0
            
            pass_dir_vec = torch.zeros(4, dtype=torch.float32)
            pass_dir_vec[pass_dir.value] = 1.0
            
            hand_mask = torch.full((52,), float('-inf'), dtype=torch.float32)
            for card in hand:
                hand_mask[card.to_id()] = 0.0
            
            # Get Expert's choice (ALWAYS the label)
            class DummyPlayer:
                def __init__(self, h):
                    self.hand = h
            dummy = DummyPlayer(hand)
            info = {'pass_direction': pass_dir}
            expert_cards = ExpertPolicy.pass_policy(dummy, info)
            expert_ids = torch.tensor([c.to_id() for c in expert_cards], dtype=torch.long)
            
            # For player 0, decide whether AI or Expert actually plays
            if player_id == 0:
                if random.random() < beta:
                    # Expert plays
                    actual_cards = expert_cards
                else:
                    # AI plays
                    pass_model.eval()
                    with torch.no_grad():
                        h = hand_vec.unsqueeze(0).to(device)
                        p = pass_dir_vec.unsqueeze(0).to(device)
                        m = hand_mask.unsqueeze(0).to(device)
                        selected, _, _ = pass_model.select_three_cards(h, p, m, deterministic=False)
                        card_ids = selected[0].cpu().numpy()
                        actual_cards = [Card.from_id(int(idx)) for idx in card_ids]
                
                # Store passing data for player 0
                pass_data.append((hand_vec, pass_dir_vec, hand_mask, expert_ids))
            else:
                # Other players use Expert
                actual_cards = expert_cards
            
            passed_cards_all[player_id] = actual_cards
            
            # Remove cards from hand
            for c in actual_cards:
                player.hand.remove(c)
        
        # Execute card passing
        offsets = {PassDirection.LEFT: 1, PassDirection.RIGHT: 3, PassDirection.ACROSS: 2}
        offset = offsets[pass_dir]
        
        for i in range(4):
            source_id = (i - offset) % 4
            received = passed_cards_all[source_id]
            game.gamestate.players[i].hand.extend(received)
            game.gamestate.players[i].hand.sort()
        
        # Save pass info for feature engineering
        game.passed_cards = passed_cards_all
        for i in range(4):
            source_id = (i - offset) % 4
            game.received_cards[i] = passed_cards_all[source_id]
        
        # ========== PLAYING PHASE ==========
        # Now play the game and collect playing data for player 0
        
        current_player = game.gamestate.get_first_player()
        
        for trick_num in range(1, 14):
            game.gamestate.current_table = []
            game.gamestate.current_suit = None
            
            trick_cards = []
            
            for i in range(4):
                player_idx = (current_player + i) % 4
                player = game.gamestate.players[player_idx]
                
                info = game.get_player_info(player_idx)
                
                is_first_round = (trick_num == 1)
                from game import available_actions
                scored_hearts = (sum(p.points for p in game.gamestate.players) > 0)
                legal = available_actions(player, game.gamestate.current_suit, is_first_round, scored_hearts)
                
                # Get Expert's choice (ALWAYS the label for player 0)
                expert_action = ExpertPolicy.play_policy(player, info, legal, i)
                
                if player_idx == 0:
                    # Preprocess state for player 0
                    state_vec = play_agent.preprocess_obs(info)
                    episode_history.append(state_vec)
                    
                    # Build sequence
                    seq = torch.stack(episode_history)
                    
                    # Build mask
                    mask = torch.full((52,), float('-inf'))
                    legal_ids = [c.to_id() for c in legal]
                    mask[legal_ids] = 0
                    
                    # Global priv
                    global_priv = info.get('global_state', torch.zeros(156))
                    if not isinstance(global_priv, torch.Tensor):
                        global_priv = torch.zeros(156)
                    
                    # Store playing data (expert action is the label)
                    action_id = expert_action.to_id()
                    play_data.append((seq.clone(), mask.clone(), action_id, global_priv.clone()))
                    
                    # Decide who actually plays
                    if random.random() < beta:
                        chosen_card = expert_action
                    else:
                        # AI plays
                        chosen_card = play_agent.act(player, info, legal, i, training=False)
                else:
                    # Other players use Expert
                    chosen_card = expert_action
                
                # Execute the move
                if chosen_card not in legal:
                    chosen_card = random.choice(legal)
                
                player.hand.remove(chosen_card)
                game.gamestate.current_table.append((chosen_card, player_idx))
                game.gamestate.table.append((chosen_card, player_idx))
                trick_cards.append(chosen_card)
                
                if i == 0:
                    game.gamestate.current_suit = chosen_card.suit
                
                if chosen_card.suit == Suit.HEARTS:
                    game.gamestate.heart_broken = True
            
            # Determine winner
            winner_local_idx = 0
            best_val = -1
            lead_suit = game.gamestate.current_suit
            
            for idx, c in enumerate(trick_cards):
                if c.suit == lead_suit:
                    val = 14 if c.rank == 1 else c.rank
                    if val > best_val:
                        best_val = val
                        winner_local_idx = idx
            
            winner_idx = (current_player + winner_local_idx) % 4
            
            points = sum(c.value() for c in trick_cards)
            game.gamestate.players[winner_idx].table.extend(trick_cards)
            game.gamestate.players[winner_idx].points += points
            
            current_player = winner_idx
    
    print(f"Collected {len(pass_data)} passing samples, {len(play_data)} playing samples")
    return pass_data, play_data


def compute_pass_loss(model, hand_vec, pass_dir_vec, hand_mask, expert_cards, device):
    """Compute sequential cross-entropy loss for passing."""
    batch_size = hand_vec.size(0)
    
    selected_vec = torch.zeros(batch_size, 52, device=device)
    current_mask = hand_mask.clone()
    
    total_loss = 0
    correct = 0
    
    for step in range(3):
        logits, _ = model(hand_vec, selected_vec, pass_dir_vec, current_mask)
        target = expert_cards[:, step]
        
        loss = nn.functional.cross_entropy(logits, target)
        total_loss = total_loss + loss
        
        pred = logits.argmax(dim=-1)
        correct += (pred == target).sum().item()
        
        # Teacher forcing: use expert's choice for next step
        selected_vec = selected_vec.scatter(1, target.unsqueeze(1), 1.0)
        current_mask = current_mask.scatter(1, target.unsqueeze(1), float('-inf'))
    
    return total_loss / 3, correct / (3 * batch_size)


def compute_play_loss(model, batch_seqs, batch_masks, batch_actions, batch_global, device):
    """Compute cross-entropy loss for playing."""
    logits, _, _, _ = model(batch_seqs, batch_global)
    
    # Apply mask and compute loss
    masked_logits = logits + batch_masks
    loss = nn.functional.cross_entropy(masked_logits, batch_actions)
    
    # Accuracy
    pred = masked_logits.argmax(dim=-1)
    acc = (pred == batch_actions).float().mean().item()
    
    return loss, acc


def pretrain_joint(num_games=5000, epochs=15, batch_size=256, lr=1e-3, dagger_rounds=3):
    """
    Joint pre-training using DAgger.
    
    Both networks train together, seeing each other's decisions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Joint Pre-training on {device}")
    print(f"Games per round: {num_games}, Epochs: {epochs}, DAgger rounds: {dagger_rounds}")
    
    # Initialize models
    play_model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
    pass_model = PassingNetwork(hidden_dim=256).to(device)
    
    play_optimizer = optim.Adam(play_model.parameters(), lr=lr)
    pass_optimizer = optim.Adam(pass_model.parameters(), lr=lr)
    
    # Create agents
    play_agent = SimpleFCNAgent(play_model, device)
    pass_agent = PassingAgent(pass_model, device)
    play_agent.set_passing_agent(pass_agent)
    
    # Accumulators
    all_pass_data = []
    all_play_data = []
    
    best_play_acc = 0
    best_pass_acc = 0
    
    for dagger_round in range(dagger_rounds + 1):
        # Compute beta (probability of using Expert)
        if dagger_round == 0:
            beta = 1.0  # Pure behavior cloning
            games_this_round = num_games
        else:
            beta = max(0.2, 1.0 - dagger_round * 0.25)  # Gradually decrease
            games_this_round = num_games // 2  # Less data in later rounds
        
        print(f"\n{'='*60}")
        print(f"DAgger Round {dagger_round}: beta={beta:.2f}")
        print(f"{'='*60}")
        
        # Collect data by playing actual games
        pass_data, play_data = collect_joint_data(
            games_this_round, 
            play_model, pass_model,
            play_agent, pass_agent,
            device, beta
        )
        
        all_pass_data.extend(pass_data)
        all_play_data.extend(play_data)
        
        print(f"Total data: {len(all_pass_data)} passing, {len(all_play_data)} playing")
        
        # Create data loaders
        pass_loader = DataLoader(
            all_pass_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: (
                torch.stack([d[0] for d in x]),
                torch.stack([d[1] for d in x]),
                torch.stack([d[2] for d in x]),
                torch.stack([d[3] for d in x])
            )
        )
        
        play_loader = DataLoader(
            all_play_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pad_sequences
        )
        
        # Training epochs
        round_epochs = epochs if dagger_round == 0 else max(5, epochs // 2)
        
        for epoch in range(round_epochs):
            play_model.train()
            pass_model.train()
            
            # ===== Train Passing Network =====
            pass_loss_sum = 0
            pass_correct = 0
            pass_total = 0
            
            for hand_vec, pass_dir_vec, hand_mask, expert_cards in pass_loader:
                hand_vec = hand_vec.to(device)
                pass_dir_vec = pass_dir_vec.to(device)
                hand_mask = hand_mask.to(device)
                expert_cards = expert_cards.to(device)
                
                loss, acc = compute_pass_loss(
                    pass_model, hand_vec, pass_dir_vec, hand_mask, expert_cards, device
                )
                
                pass_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pass_model.parameters(), 1.0)
                pass_optimizer.step()
                
                bs = hand_vec.size(0)
                pass_loss_sum += loss.item() * bs
                pass_correct += acc * bs
                pass_total += bs
            
            pass_acc = pass_correct / pass_total if pass_total > 0 else 0
            
            # ===== Train Playing Network =====
            play_loss_sum = 0
            play_correct = 0
            play_total = 0
            
            for batch_seqs, batch_masks, batch_actions, batch_global in play_loader:
                batch_seqs = batch_seqs.to(device)
                batch_masks = batch_masks.to(device)
                batch_actions = batch_actions.to(device)
                batch_global = batch_global.to(device)
                
                loss, acc = compute_play_loss(
                    play_model, batch_seqs, batch_masks, batch_actions, batch_global, device
                )
                
                play_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(play_model.parameters(), 1.0)
                play_optimizer.step()
                
                bs = batch_seqs.size(0)
                play_loss_sum += loss.item() * bs
                play_correct += acc * bs
                play_total += bs
            
            play_acc = play_correct / play_total if play_total > 0 else 0
            
            print(f"Epoch {epoch+1}/{round_epochs} | "
                  f"Pass Acc: {pass_acc:.4f} | Play Acc: {play_acc:.4f}")
            
            # Save best models
            if pass_acc > best_pass_acc:
                best_pass_acc = pass_acc
                torch.save({
                    'model_state_dict': pass_model.state_dict(),
                    'optimizer_state_dict': pass_optimizer.state_dict(),
                    'accuracy': pass_acc,
                    'dagger_round': dagger_round,
                }, config.PASSING_PRETRAINED_PATH)
            
            if play_acc > best_play_acc:
                best_play_acc = play_acc
                torch.save({
                    'model_state_dict': play_model.state_dict(),
                    'optimizer_state_dict': play_optimizer.state_dict(),
                    'accuracy': play_acc,
                    'dagger_round': dagger_round,
                }, config.PRETRAINED_MODEL_PATH)
        
        # Quick evaluation
        print(f"\n--- Evaluation after Round {dagger_round} ---")
        evaluate_joint(play_model, pass_model, device, num_games=100)
    
    print(f"\n{'='*60}")
    print(f"Pre-training Complete!")
    print(f"Best Pass Accuracy: {best_pass_acc:.4f}")
    print(f"Best Play Accuracy: {best_play_acc:.4f}")
    print(f"Models saved to: {config.OUTPUT_DIR}")
    print(f"{'='*60}")


def evaluate_joint(play_model, pass_model, device, num_games=200):
    """Evaluate both networks together."""
    play_model.eval()
    pass_model.eval()
    
    play_agent = SimpleFCNAgent(play_model, device)
    pass_agent = PassingAgent(pass_model, device)
    play_agent.set_passing_agent(pass_agent)
    
    game = GameV2()
    
    # vs Expert
    scores_vs_expert = []
    for _ in range(num_games):
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, 
                                   PassDirection.ACROSS, PassDirection.KEEP])
        
        def ai_policy(p, i, l, o):
            return play_agent.act(p, i, l, o, training=False)
        
        policies = [ai_policy] + [ExpertPolicy.play_policy] * 3
        pass_policies = [play_agent.pass_policy] + [ExpertPolicy.pass_policy] * 3
        
        play_agent.reset_episode_memory()
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        scores_vs_expert.append(scores[0])
    
    avg_expert = sum(scores_vs_expert) / len(scores_vs_expert)
    
    # vs Random
    scores_vs_random = []
    for _ in range(num_games):
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, 
                                   PassDirection.ACROSS, PassDirection.KEEP])
        
        def ai_policy(p, i, l, o):
            return play_agent.act(p, i, l, o, training=False)
        def rand_policy(p, i, l, o):
            return random.choice(l)
        def rand_pass(p, i):
            return random.sample(list(p.hand), 3) if len(p.hand) >= 3 else list(p.hand)
        
        policies = [ai_policy] + [rand_policy] * 3
        pass_policies = [play_agent.pass_policy] + [rand_pass] * 3
        
        play_agent.reset_episode_memory()
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        scores_vs_random.append(scores[0])
    
    avg_random = sum(scores_vs_random) / len(scores_vs_random)
    
    print(f"vs Expert: {avg_expert:.2f} | vs Random: {avg_random:.2f}")
    
    return avg_expert, avg_random


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        # Evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        play_model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
        pass_model = PassingNetwork(hidden_dim=256).to(device)
        
        if os.path.exists(config.PRETRAINED_MODEL_PATH):
            checkpoint = torch.load(config.PRETRAINED_MODEL_PATH)
            play_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded play model (acc={checkpoint.get('accuracy', 'N/A')})")
        
        if os.path.exists(config.PASSING_PRETRAINED_PATH):
            checkpoint = torch.load(config.PASSING_PRETRAINED_PATH)
            pass_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pass model (acc={checkpoint.get('accuracy', 'N/A')})")
        
        print("\n===== Evaluation Results =====")
        evaluate_joint(play_model, pass_model, device, num_games=500)
        
    else:
        # Training mode
        pretrain_joint(
            num_games=5000,     # Games per DAgger round
            epochs=15,          # Epochs per round
            batch_size=256,
            lr=1e-3,
            dagger_rounds=3     # Total 4 rounds (0, 1, 2, 3)
        )


if __name__ == "__main__":
    main()
