import torch
import os
import numpy as np
import random
from game import GameV2
from transformer import HeartsTransformer
from train import AIPlayer, HIDDEN_DIM, DROPOUT
from strategies import ExpertPolicy
from data_structure import PassDirection
import gpu_selector

def test_model(num_games=300):
    # 1. Setup Device and Model
    device = gpu_selector.select_device()
    print(f"Testing on {device}")

    model = HeartsTransformer(d_model=HIDDEN_DIM, dropout=DROPOUT).to(device)
    model_path = 'hearts_model.pth'

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    else:
        print("Error: No model found at hearts_model.pth")
        return

    # 2. Setup Players
    ai_player = AIPlayer(model, device)
    game = GameV2()

    # 3. Run Games
    ai_scores = []
    expert_scores = [] # Average of 3 experts

    print(f"Starting {num_games} games against 3 Experts...")

    for i in range(num_games):
        ai_player.reset()
        game.reset()

        # Policies
        # Player 0 is AI
        # Players 1, 2, 3 are Experts
        
        # Play Policies
        current_policies = [
            ai_player.play_policy,
            ExpertPolicy.play_policy,
            ExpertPolicy.play_policy,
            ExpertPolicy.play_policy
        ]

        # Pass Policies
        current_pass_policies = [
            ai_player.pass_policy,
            ExpertPolicy.pass_policy,
            ExpertPolicy.pass_policy,
            ExpertPolicy.pass_policy
        ]

        # Random pass direction
        pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP])

        # Run Game
        # run_game_training returns: scores, raw_scores, events, trick_history
        scores, raw_scores, events, trick_history = game.run_game_training(
            current_policies, 
            current_pass_policies, 
            pass_direction=pass_dir
        )

        ai_score = scores[0]
        avg_expert_score = sum(scores[1:]) / 3.0
        
        ai_scores.append(ai_score)
        expert_scores.append(avg_expert_score)

        if (i + 1) % 10 == 0:
            print(f"Game {i+1}/{num_games} - AI Score: {ai_score}, Expert Avg: {avg_expert_score:.2f} | Running AI Avg: {np.mean(ai_scores):.2f}")

    # 4. Results
    print("\n" + "="*30)
    print(f"Test Complete ({num_games} games)")
    print(f"AI Average Score: {np.mean(ai_scores):.2f}")
    print(f"Expert Average Score: {np.mean(expert_scores):.2f}")
    print(f"AI Win Rate (Score=0): {sum(1 for s in ai_scores if s == 0) / num_games:.2%}")
    print(f"AI < Expert Rate: {sum(1 for a, e in zip(ai_scores, expert_scores) if a < e) / num_games:.2%}")
    
    print("\nLast Game State:")
    print(f"Pass Direction: {pass_dir.name}")
    print(f"Final Scores: {scores}")
    print(f"Raw Scores: {raw_scores}")
    print("Trick History (Winner, Points):")
    for idx, t in enumerate(trick_history):
        print(f"  Trick {idx+1}: Winner P{t.winner}, Points {t.score}, Lead {t.lead_suit.name if t.lead_suit else 'None'}")

if __name__ == "__main__":
    test_model()
