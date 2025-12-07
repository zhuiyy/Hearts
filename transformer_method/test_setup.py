
import torch
from game import GameV2
from transformer import HeartsTransformer
from strategies import ExpertPolicy, random_policy
from data_structure import PassDirection

def test_integration():
    print("Testing Game Logic...")
    game = GameV2()
    game.reset()
    
    # Test Showcase Run
    print("Running Showcase Game...")
    try:
        game.run_game_showcase([random_policy]*4, pass_direction=PassDirection.LEFT)
        print("Showcase Game Success!")
    except Exception as e:
        print(f"Showcase Game Failed: {e}")
        raise e

    print("\nTesting Transformer Model...")
    try:
        model = HeartsTransformer(d_model=64, num_layers=2)
        print("Model initialized.")
        
        # Create dummy input
        # We need to simulate the input assembly
        # Let's just try to run a training step simulation
        from train import AIPlayer
        ai = AIPlayer(model)
        
        # Run a partial game to get state
        game.reset()
        # We need to manually step to get info
        # Or just use the AIPlayer in a game
        
        print("Running Game with AI Player...")
        
        def ai_policy_wrapper(player, info, legal, order):
            return ai.play_policy(player, info, legal, order)
            
        def ai_pass_wrapper(player, info):
            return ai.pass_policy(player, info)
            
        policies = [ai_policy_wrapper, random_policy, random_policy, random_policy]
        pass_policies = [ai_pass_wrapper, lambda p,i: p.hand[:3], lambda p,i: p.hand[:3], lambda p,i: p.hand[:3]]
        
        game.run_game_training(policies, pass_policies, pass_direction=PassDirection.LEFT)
        print("AI Player Game Success!")
        
    except Exception as e:
        print(f"Transformer Test Failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_integration()
