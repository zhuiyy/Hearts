"""
Hearts AI - Complete Training Pipeline

This is the main entry point for training the Hearts AI.
Both the passing network and playing network are trained together.

================================================================================
TRAINING FLOW
================================================================================

Step 1: Joint Pre-training (DAgger)
    Both networks train together from the start.
    - PassingNetwork learns to select 3 cards to pass
    - HeartsLSTM learns to play cards
    - They see each other's decisions, so no distribution mismatch!
    
    Command: python main.py pretrain

Step 2: Joint RL Fine-tuning (PPO)
    Both networks are fine-tuned together using PPO.
    - Reward is based on final game score
    - Both networks share the same reward signal
    - They learn to work together to minimize score
    
    Command: python main.py train

Step 3: Evaluation
    Test the trained AI against Expert and Random opponents.
    
    Command: python main.py eval

================================================================================
ARCHITECTURE
================================================================================

PassingNetwork (MLP):
    Input:  Hand (52) + Selected (52) + Direction (4) = 108 dim
    Hidden: 256 → 256 → 256
    Output: 52 (score for each card)
    
    Sequential selection: Pick 3 cards one by one
    This captures dependencies like "if I passed Q♠, don't need to pass K♠"

HeartsLSTM:
    Input:  State (349 dim) including passed/received cards info
    Encoder: 349 → 512
    LSTM: 512 hidden, 1 layer
    Output: 52 (score for each card)
    
    The 349 features include:
    - Hand cards (52)
    - Table cards (52)
    - Play history (104)
    - Passed cards (52) ← knows what was passed!
    - Received cards (52) ← knows what was received!
    - Pass direction (4)
    - Other game state (33)

================================================================================
"""

import sys
import os


def print_usage():
    print("""
Hearts AI Training Pipeline
============================

Usage:
    python main.py pretrain    - Joint pre-training (DAgger)
    python main.py train       - Joint RL fine-tuning (PPO)  
    python main.py eval        - Evaluate the trained model
    python main.py all         - Run full pipeline (pretrain → train → eval)
    
Files generated:
    output/simple_fcn_pretrained.pth   - Pretrained playing network
    output/passing_pretrained.pth      - Pretrained passing network
    output/simple_fcn_model.pth        - RL-trained playing network
    output/passing_model.pth           - RL-trained passing network
""")


def run_pretrain():
    print("\n" + "="*60)
    print("STEP 1: Joint Pre-training (DAgger)")
    print("="*60 + "\n")
    
    from pretrain_joint import pretrain_joint
    pretrain_joint(
        num_games=5000,
        epochs=15,
        batch_size=256,
        lr=1e-3,
        dagger_rounds=3
    )


def run_train():
    print("\n" + "="*60)
    print("STEP 2: Joint RL Fine-tuning (PPO)")
    print("="*60 + "\n")
    
    from train_joint import train_joint
    train_joint()


def run_eval():
    print("\n" + "="*60)
    print("STEP 3: Evaluation")
    print("="*60 + "\n")
    
    from train_joint import evaluate_joint
    import torch
    from model import HeartsLSTM
    from passing_model import PassingNetwork
    import config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    play_model = HeartsLSTM(config.INPUT_DIM, config.HIDDEN_DIM).to(device)
    pass_model = PassingNetwork(hidden_dim=256).to(device)
    
    # Try to load RL-trained models first, then pretrained
    play_path = config.MODEL_PATH if os.path.exists(config.MODEL_PATH) else config.PRETRAINED_MODEL_PATH
    pass_path = config.PASSING_MODEL_PATH if os.path.exists(config.PASSING_MODEL_PATH) else config.PASSING_PRETRAINED_PATH
    
    if os.path.exists(play_path):
        checkpoint = torch.load(play_path)
        play_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded play model from {play_path}")
    else:
        print("No play model found!")
    
    if os.path.exists(pass_path):
        checkpoint = torch.load(pass_path)
        pass_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pass model from {pass_path}")
    else:
        print("No pass model found!")
    
    print("\n===== Evaluation Results =====")
    evaluate_joint(play_model, pass_model, device, num_games=500)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'pretrain':
        run_pretrain()
    elif command == 'train':
        run_train()
    elif command == 'eval':
        run_eval()
    elif command == 'all':
        run_pretrain()
        run_train()
        run_eval()
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
