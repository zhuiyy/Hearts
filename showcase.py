import torch
import random
import os
import sys
from game import GameV2
from transformer import HeartsTransformer
from data_structure import Card, PassDirection, Suit
from agent import AIPlayer
import config
import strategies
import gpu_selector

def print_card_list(cards):
    return ", ".join([str(c) for c in sorted(cards)])

def run_showcase():
    # 1. Load Model
    # Use gpu_selector if interactive, or just auto-detect if not
    if sys.stdin.isatty():
        device = gpu_selector.select_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Running on {device}")
    
    model = HeartsTransformer(d_model=config.HIDDEN_DIM).to(device)
    model_path = config.MODEL_PATH
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', '?')
                print(f"Model loaded from epoch {epoch}")
            else:
                state_dict = checkpoint
                print("Loaded legacy model format.")

            # Auto-Patching for Architecture Changes
            if 'input_projections.5.weight' in state_dict:
                old_weight = state_dict['input_projections.5.weight']
                if old_weight.shape[1] == 8:
                    print("Warning: Detected old model version (GameStats dim=8). Patching to dim=9...")
                    new_weight = torch.zeros((HIDDEN_DIM, 9), device=device)
                    new_weight[:, :8] = old_weight
                    state_dict['input_projections.5.weight'] = new_weight
                    
            model.load_state_dict(state_dict, strict=False)
            model.eval() # Set to eval mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Model not found! Please train first.")
        return

    ai_player = AIPlayer(model, device)
    game = GameV2()
    
    # 2. Setup Game
    print("\n" + "="*50)
    print(" STARTING SHOWCASE GAME ")
    print(" Player 0: AI Agent (Trained)")
    print(" Player 1: Expert Bot")
    print(" Player 2: Expert Bot")
    print(" Player 3: Expert Bot")
    print("="*50 + "\n")
    
    # Reset
    game.reset()
    ai_player.reset()
    
    # Determine Pass Direction
    pass_dir = random.choice([PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS])
    print(f"Pass Direction: {pass_dir.name}")
    
    # 3. Passing Phase
    print("\n--- Passing Phase ---")
    
    # Wrapper to print AI's passing decision
    def ai_pass_wrapper(player, info):
        print(f"AI Hand before pass: {print_card_list(player.hand)}")
        passed = ai_player.pass_policy(player, info)
        print(f"AI chose to pass:    {print_card_list(passed)}")
        return passed

    pass_policies = [
        ai_pass_wrapper, 
        strategies.ExpertPolicy.pass_policy, 
        strategies.ExpertPolicy.pass_policy, 
        strategies.ExpertPolicy.pass_policy
    ]
    
    # Execute passing
    game.perform_pass(pass_dir, pass_policies)
    
    print(f"AI Hand after pass:  {print_card_list(game.gamestate.players[0].hand)}")
    
    # 4. Play Phase
    print("\n--- Play Phase ---")
    
    current_trick_ai_value = 0.0
    
    # Wrapper to print AI's play decision
    def ai_play_wrapper(player, info, legal, order):
        nonlocal current_trick_ai_value
        card = ai_player.play_policy(player, info, legal, order)
        # Get the last value prediction if available
        current_trick_ai_value = ai_player.saved_values[-1].item() if ai_player.saved_values else 0.0
        return card

    # Opponents
    policies = [
        ai_play_wrapper,
        strategies.ExpertPolicy.play_policy,
        strategies.ExpertPolicy.play_policy,
        strategies.ExpertPolicy.play_policy
    ]
    
    first_player = game.gamestate.get_first_player()
    
    for round_num in range(1, 14):
        print(f"\n--- Trick {round_num} ---")
        winner, points, events = game.play_trick(first_player, policies, verbose=True)
        first_player = winner
        game.rounds += 1
        
        # Print AI Value Prediction
        print(f"AI Value Prediction: {current_trick_ai_value:.4f}")
        
    # Final Score
    scores = game.end_game_scoring()
    print("\n" + "="*50)
    print(" FINAL SCORES ")
    print(f" AI Agent: {scores[0]}")
    print(f" Expert1: {scores[1]}")
    print(f" Expert2: {scores[2]}")
    print(f" Expert3: {scores[3]}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_showcase()
