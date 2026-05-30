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
    import config

    print("\n===== Evaluation Results =====")
    evaluate_joint(num_games=config.EVAL_GAMES, seed=config.SEED)


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
