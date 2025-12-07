import torch
import torch.optim as optim
import torch.nn.functional as F
from game import GameV2
from model import HeartsProNet
from agent import SotaAgent
from strategies import ExpertPolicy
from data_structure import PassDirection
import random
import config
import os

def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pretraining on {device}")
    
    model = HeartsProNet(config.HIDDEN_DIM, config.LSTM_HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    agent = SotaAgent(model, device, use_pimc=False)
    game = GameV2()
    
    # Pretraining Hyperparameters
    PRETRAIN_EPISODES = 10000
    BATCH_SIZE = 32
    
    print(f"Starting Supervised Pretraining for {PRETRAIN_EPISODES} episodes...")
    
    log_data = {'episodes': [], 'loss': [], 'policy_acc': [], 'qs_acc': []}
    
    agent.reset() # Clear buffer initially
    
    for episode in range(PRETRAIN_EPISODES):
        # agent.reset() # REMOVED: Do not clear buffer here, or we lose data from previous episodes in the batch!
        
        # We need to collect data from Expert Play
        # But SotaAgent.act() uses the model.
        # We need a way to run the game with ExpertPolicy but RECORD the observations as if the Agent was playing.
        
        # Solution:
        # 1. Run game with 4 Expert Policies.
        # 2. Inside the game loop (or via a custom wrapper), for every move:
        #    a. Construct the observation (Agent's view).
        #    b. Get the Expert's action.
        #    c. Store (obs, expert_action) in the agent's buffer.
        
        # To do this cleanly without modifying game.py too much, we can use a "Recording Policy".
        
        def recording_expert_policy(player, info, legal, order):
            # 1. Get Expert Action
            expert_card = ExpertPolicy.play_policy(player, info, legal, order)
            
            # 2. Record Data (Only for Player 0 to simplify, or all players if we want more data)
            # Let's record for ALL players to maximize data efficiency!
            # We just need to treat each player's turn as a training sample for the shared model.
            
            # Preprocess Obs (using Agent's helper)
            static_obs, seq_cards, seq_players = agent.preprocess_obs(info)
            
            # Store in Agent's buffer
            agent.saved_static_obs.append(static_obs.to(device))
            agent.saved_seq_cards.append(seq_cards.to(device))
            agent.saved_seq_players.append(seq_players.to(device))
            
            # Store Expert Action Label
            agent.saved_actions.append(torch.tensor(expert_card.to_id(), device=device))
            
            # Store Aux Label (SQ Location)
            agent.saved_qs_labels.append(torch.tensor(info.get('sq_label', 4), device=device))
            
            # Store Mask (for valid loss calculation, though CrossEntropy handles classes, we should mask illegal logits if we were doing KL div)
            # For standard CrossEntropy(logits, target), we don't strictly need the mask if the target is always legal.
            # But to be safe and consistent with RL, let's store it.
            mask = torch.full((52,), float('-inf'), device=device)
            legal_indices = [c.to_id() for c in legal]
            mask[legal_indices] = 0
            agent.saved_masks.append(mask)
            
            return expert_card

        # Run Game with Recording Policy for ALL players
        policies = [recording_expert_policy] * 4
        pass_policies = [ExpertPolicy.pass_policy] * 4
        
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])
        game.run_game_training(policies, pass_policies, pass_dir)
        
        # Update Model
        if (episode + 1) % BATCH_SIZE == 0:
            loss, p_acc, q_acc = update_supervised(agent, optimizer)
            print(f"Pretrain Ep {episode+1} | Loss: {loss:.4f} | Acc: {p_acc:.2%} | QS Acc: {q_acc:.2%}")
            
            log_data['episodes'].append(episode + 1)
            log_data['loss'].append(loss)
            log_data['policy_acc'].append(p_acc)
            log_data['qs_acc'].append(q_acc)
            
            # Save Log
            try:
                import json
                with open(config.PRETRAIN_LOG_FILE, 'w') as f:
                    json.dump(log_data, f)
            except Exception as e:
                print(f"Error saving log: {e}")
            
    torch.save(model.state_dict(), config.PRETRAINED_MODEL_PATH)
    print(f"Pretraining Complete. Model saved to {config.PRETRAINED_MODEL_PATH}")

def update_supervised(agent, optimizer):
    if not agent.saved_static_obs:
        return 0.0
        
    # 1. Prepare Data
    b_static_obs = torch.stack(agent.saved_static_obs)
    b_seq_cards = torch.stack(agent.saved_seq_cards)
    b_seq_players = torch.stack(agent.saved_seq_players)
    b_masks = torch.stack(agent.saved_masks)
    b_actions = torch.stack(agent.saved_actions)
    b_qs_labels = torch.stack(agent.saved_qs_labels)
    
    dataset_size = b_static_obs.size(0)
    indices = list(range(dataset_size))
    
    total_loss_accum = 0
    total_p_acc = 0
    total_q_acc = 0
    steps = 0
    
    # 2. Mini-batch Training Loop
    MINI_BATCH_SIZE = 64
    
    for _ in range(config.PRETRAIN_EPOCHS):
        random.shuffle(indices)
        
        for start_idx in range(0, dataset_size, MINI_BATCH_SIZE):
            end_idx = min(start_idx + MINI_BATCH_SIZE, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Slice Batch
            mb_static = b_static_obs[batch_indices]
            mb_seq_cards = b_seq_cards[batch_indices]
            mb_seq_players = b_seq_players[batch_indices]
            mb_masks = b_masks[batch_indices]
            mb_actions = b_actions[batch_indices]
            mb_qs_labels = b_qs_labels[batch_indices]
            
            # Forward Pass
            logits, _, qs_pred = agent.model(mb_static, mb_seq_cards, mb_seq_players, global_priv_info=None)
            
            # Policy Loss
            masked_logits = logits.squeeze() + mb_masks
            policy_loss = F.cross_entropy(masked_logits, mb_actions)
            
            # Policy Accuracy
            pred_actions = torch.argmax(masked_logits, dim=1)
            p_acc = (pred_actions == mb_actions).float().mean().item()
            
            # Aux Loss
            qs_loss = F.cross_entropy(qs_pred, mb_qs_labels)
            
            # QS Accuracy
            pred_qs = torch.argmax(qs_pred, dim=1)
            q_acc = (pred_qs == mb_qs_labels).float().mean().item()
            
            loss = policy_loss + 0.5 * qs_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss_accum += loss.item()
            total_p_acc += p_acc
            total_q_acc += q_acc
            steps += 1
    
    agent.reset()
    if steps > 0:
        return total_loss_accum / steps, total_p_acc / steps, total_q_acc / steps
    return 0.0, 0.0, 0.0

if __name__ == "__main__":
    pretrain()
