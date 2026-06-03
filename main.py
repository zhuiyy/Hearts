"""
Hearts AI training pipeline.

Step 1: Joint pretraining
    python main.py pretrain

Step 2: PPO fine-tuning
    python main.py train

Step 3: Multi-seed evaluation
    python main.py eval
"""

import sys


def print_usage():
    print("""
Hearts AI Training Pipeline
============================

Usage:
    python main.py pretrain    - Joint pretraining with DAgger-style imitation
    python main.py train       - PPO fine-tuning
    python main.py eval        - Multi-seed evaluation
    python main.py all         - Run pretrain -> train -> eval

Files generated:
    output/simple_fcn_pretrained.pth   - Pretrained playing network
    output/passing_pretrained.pth      - Pretrained passing network
    output/simple_fcn_model.pth        - PPO-trained playing network
    output/passing_model.pth           - Passing network checkpoint
    output/training_log.json           - Training metrics and run config
""")


def run_pretrain():
    print("\n" + "=" * 60)
    print("STEP 1: Joint Pretraining")
    print("=" * 60 + "\n")

    from pretrain_joint import pretrain_joint
    pretrain_joint(
        num_games=5000,
        epochs=15,
        batch_size=256,
        lr=1e-3,
        dagger_rounds=3,
    )


def run_train():
    print("\n" + "=" * 60)
    print("STEP 2: PPO Fine-Tuning")
    print("=" * 60 + "\n")

    from train_joint import train_joint
    train_joint()


def run_eval():
    print("\n" + "=" * 60)
    print("STEP 3: Multi-Seed Evaluation")
    print("=" * 60 + "\n")

    from train_joint import evaluate_across_seeds
    evaluate_across_seeds()


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
