"""
Imitation Learning Pre-training for Hearts AI

This script uses Behavior Cloning to pre-train the model on expert demonstrations.
After pre-training, use train.py for RL fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
from tqdm import tqdm
from collections import deque

from game import GameV2
from model import HeartsLSTM
from agent import SimpleFCNAgent
from data_structure import PassDirection, Card
from strategies import ExpertPolicy
import config


class ExpertDataset(Dataset):
    """Dataset of (state, expert_action) pairs."""
    
    def __init__(self, data):
        # data: list of (state_seq, mask, action_id)
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state_seq, mask, action_id, global_priv = self.data[idx]
        return state_seq, mask, action_id, global_priv


def pad_sequences(batch):
    """Collate function to pad sequences to the same length."""
    state_seqs, masks, action_ids, global_privs = zip(*batch)
    
    max_len = max(s.size(0) for s in state_seqs)
    max_len = max(max_len, 1)  # At least 1
    
    padded_seqs = []
    for seq in state_seqs:
        L = seq.size(0)
        if L < max_len:
            padding = torch.zeros(max_len - L, config.INPUT_DIM)
            padded = torch.cat([seq, padding], dim=0)
        else:
            padded = seq[:max_len]
        padded_seqs.append(padded)
    
    batch_seqs = torch.stack(padded_seqs)  # [B, MaxLen, Dim]
    batch_masks = torch.stack(masks)  # [B, 52]
    batch_actions = torch.tensor(action_ids, dtype=torch.long)  # [B]
    batch_global = torch.stack(global_privs)  # [B, 156]
    
    return batch_seqs, batch_masks, batch_actions, batch_global


def collect_expert_data(num_games=5000, use_dagger=False, model=None, agent=None, beta=0.5):
    """
    Run games with ExpertPolicy and collect state-action pairs.
    
    If use_dagger=True, uses DAgger algorithm:
    - Agent plays with probability (1-beta), Expert plays with probability beta
    - But we ALWAYS record the Expert's action as the label
    - This exposes the model to states it creates through its own mistakes
    """
    print(f"Collecting {num_games} games of expert data...")
    if use_dagger:
        print(f"  Using DAgger with beta={beta} (agent plays {100*(1-beta):.0f}% of the time)")
    
    device = torch.device("cpu")  # Collect on CPU
    
    # Temporary model just for preprocessing (if not using DAgger)
    if agent is None:
        temp_model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM)
        agent = SimpleFCNAgent(temp_model, device)
    
    game = GameV2()
    data = []
    
    for game_idx in tqdm(range(num_games)):
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        # We'll collect data for player 0 using expert policy
        episode_history = []
        
        def data_collecting_policy(player, info, legal, order):
            """Wrapper that collects data while using expert policy."""
            # Preprocess state
            state_vec = agent.preprocess_obs(info)
            episode_history.append(state_vec)
            
            # Build sequence
            seq = torch.stack(episode_history)  # [SeqLen, Dim]
            
            # Get expert action (this is ALWAYS the label)
            expert_action = ExpertPolicy.play_policy(player, info, legal, order)
            action_id = expert_action.to_id()
            
            # Build mask
            mask = torch.full((52,), float('-inf'))
            legal_ids = [c.to_id() for c in legal]
            mask[legal_ids] = 0
            
            # Global priv (dummy for now, or extract if available)
            global_priv = info.get('global_state', torch.zeros(156))
            if not isinstance(global_priv, torch.Tensor):
                global_priv = torch.zeros(156)
            
            # Store data (expert action is always the label)
            data.append((seq.clone(), mask.clone(), action_id, global_priv.clone()))
            
            # DAgger: Sometimes let the agent play to explore its own mistakes
            if use_dagger and random.random() > beta:
                # Agent plays (but expert action is still the label)
                return agent.act(player, info, legal, order, training=False)
            else:
                # Expert plays
                return expert_action
        
        # All players use expert policy, but we only collect data for player 0
        policies = [
            data_collecting_policy,
            lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
            lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
            lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
        ]
        
        pass_policies = [ExpertPolicy.pass_policy] * 4
        
        # Reset episode history for new game
        episode_history = []
        
        game.run_game_training(policies, pass_policies, pass_dir)
    
    print(f"Collected {len(data)} state-action pairs from {num_games} games")
    return data


def pretrain(num_games=5000, epochs=20, batch_size=256, lr=1e-3, dagger_rounds=3):
    """
    Pre-train the model using behavior cloning + DAgger.
    
    DAgger rounds progressively let the model explore more of its own trajectory
    while still learning from expert corrections.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pre-training on {device}")
    
    # Initialize model
    model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    agent = SimpleFCNAgent(model, device)
    
    best_acc = 0
    all_data = []
    
    for dagger_iter in range(dagger_rounds + 1):
        if dagger_iter == 0:
            print(f"\n=== Round 0: Pure Behavior Cloning ===")
            new_data = collect_expert_data(num_games, use_dagger=False)
        else:
            # Progressively decrease beta (let agent play more)
            beta = max(0.3, 1.0 - dagger_iter * 0.2)  # 0.8, 0.6, 0.4, 0.3...
            print(f"\n=== Round {dagger_iter}: DAgger with beta={beta} ===")
            new_data = collect_expert_data(
                num_games // 2,  # Fewer games per DAgger round
                use_dagger=True, 
                model=model, 
                agent=agent, 
                beta=beta
            )
        
        all_data.extend(new_data)
        print(f"Total dataset size: {len(all_data)}")
        
        # Create dataset and dataloader
        dataset = ExpertDataset(all_data)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=pad_sequences,
            num_workers=0
        )
        
        # Reduce epochs for DAgger rounds (fine-tuning)
        round_epochs = epochs if dagger_iter == 0 else max(5, epochs // 2)
        
        # Training loop
        for epoch in range(round_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
        
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{round_epochs}")
            for batch_seqs, batch_masks, batch_actions, batch_global in pbar:
                batch_seqs = batch_seqs.to(device)
                batch_masks = batch_masks.to(device)
                batch_actions = batch_actions.to(device)
                batch_global = batch_global.to(device)
                
                # Forward
                logits, _, _, _ = model(batch_seqs, batch_global, hidden=None)
                
                # Apply mask
                masked_logits = logits + batch_masks
                
                # Loss
                loss = criterion(masked_logits, batch_actions)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Stats
                total_loss += loss.item() * batch_seqs.size(0)
                preds = masked_logits.argmax(dim=1)
                correct += (preds == batch_actions).sum().item()
                total += batch_seqs.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})
            
            epoch_loss = total_loss / total
            epoch_acc = correct / total
            
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
            
            # Save best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': epoch_acc,
                }, config.PRETRAINED_MODEL_PATH)
                print(f"  -> Saved best model (acc={epoch_acc:.4f})")
        
        # Quick evaluation after each DAgger round
        print(f"\n--- Quick Eval after Round {dagger_iter} ---")
        quick_eval(model, agent, num_games=50)
    
    # Also save as the main model for RL training to continue from
    torch.save({
        'episode': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': float('inf'),
    }, config.MODEL_PATH)
    print(f"\nPre-training complete! Best accuracy: {best_acc:.4f}")
    print(f"Model saved to {config.MODEL_PATH}")
    print("Now run train.py to fine-tune with RL.")


def quick_eval(model, agent, num_games=50):
    """Quick evaluation during training."""
    game = GameV2()
    scores_vs_expert = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_games):
            agent.reset_episode_memory()
            
            def agent_policy(player, info, legal, order):
                return agent.act(player, info, legal, order, training=False)
            
            policies = [
                agent_policy,
                lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
                lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
                lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
            ]
            pass_policies = [ExpertPolicy.pass_policy] * 4
            pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
            
            scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
            scores_vs_expert.append(scores[0])
    
    model.train()
    avg_score = sum(scores_vs_expert) / len(scores_vs_expert)
    print(f"vs Expert ({num_games} games): Avg Score = {avg_score:.2f}")
    return avg_score


def evaluate_pretrained():
    """Evaluate the pretrained model against expert and random policies."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
    
    if os.path.exists(config.PRETRAINED_MODEL_PATH):
        checkpoint = torch.load(config.PRETRAINED_MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model (acc={checkpoint.get('accuracy', 'N/A')})")
    else:
        print("No pretrained model found!")
        return
    
    agent = SimpleFCNAgent(model, device)
    game = GameV2()
    
    # Test against experts
    scores_vs_expert = []
    scores_vs_random = []
    
    print("\nEvaluating vs Expert opponents (500 games)...")
    for _ in tqdm(range(500)):
        agent.reset_episode_memory()
        
        def agent_policy(player, info, legal, order):
            return agent.act(player, info, legal, order, training=False)
        
        policies = [
            agent_policy,
            lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
            lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
            lambda p, i, l, o: ExpertPolicy.play_policy(p, i, l, o),
        ]
        pass_policies = [ExpertPolicy.pass_policy] * 4
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        scores_vs_expert.append(scores[0])
    
    print("\nEvaluating vs Random opponents (500 games)...")
    for _ in tqdm(range(500)):
        agent.reset_episode_memory()
        
        def agent_policy(player, info, legal, order):
            return agent.act(player, info, legal, order, training=False)
        
        policies = [
            agent_policy,
            lambda p, i, l, o: random.choice(l),
            lambda p, i, l, o: random.choice(l),
            lambda p, i, l, o: random.choice(l),
        ]
        pass_policies = [ExpertPolicy.pass_policy] * 4
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        
        scores, _, _, _ = game.run_game_training(policies, pass_policies, pass_dir)
        scores_vs_random.append(scores[0])
    
    print(f"\n===== Evaluation Results =====")
    print(f"vs Expert: Avg Score = {sum(scores_vs_expert)/len(scores_vs_expert):.2f}")
    print(f"vs Random: Avg Score = {sum(scores_vs_random)/len(scores_vs_random):.2f}")
    print(f"(Lower is better. Random baseline ~6.5)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        evaluate_pretrained()
    else:
        pretrain(num_games=5000, epochs=20, batch_size=256, lr=1e-3)
