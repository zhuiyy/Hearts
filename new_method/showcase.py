import torch
import random
import time
from game import GameV2
from model import HeartsProNet
from agent import SotaAgent
from strategies import ExpertPolicy
from data_structure import PassDirection, Card, Suit
import config
import os

def print_card(card):
    suit_symbols = {Suit.HEARTS: "♥", Suit.DIAMONDS: "♦", Suit.CLUBS: "♣", Suit.SPADES: "♠"}
    rank_str = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(card.rank, str(card.rank))
    return f"{suit_symbols[card.suit]}{rank_str}"

def run_custom_showcase(game, policies, pass_policies, pass_direction):
    game.reset()
    
    # --- Passing Phase ---
    if pass_direction != PassDirection.KEEP:
        print(f"\n{'='*20} Passing Phase ({pass_direction.name}) {'='*20}")
        
        # Show AI Hand Before Pass
        ai_hand = sorted(game.gamestate.players[0].hand)
        print(f"\n[AI Hand Before Pass]:")
        print("  " + " ".join([print_card(c) for c in ai_hand]))
        
        pass_events = game.perform_pass(pass_direction, pass_policies)
        
        # Find AI event
        ai_event = next(e for e in pass_events if e.player_id == 0)
        
        print(f"\n[AI Passing Action]:")
        print(f"  Passed   -> {' '.join([print_card(c) for c in ai_event.passed_cards])}")
        print(f"  Received <- {' '.join([print_card(c) for c in ai_event.received_cards])}")
        
        # Show AI Hand After Pass
        ai_hand_after = sorted(game.gamestate.players[0].hand)
        print(f"\n[AI Hand After Pass]:")
        print("  " + " ".join([print_card(c) for c in ai_hand_after]))
        print("-" * 60)

    first_player = game.gamestate.get_first_player()
    
    # --- Trick Phase ---
    print(f"\n{'='*20} Game Start {'='*20}")
    
    for round_num in range(1, 14):
        print(f"\n--- Trick {round_num} ---")
        
        # Print AI's hand
        ai_p = game.gamestate.players[0]
        print(f"AI Hand: {' '.join([print_card(c) for c in sorted(ai_p.hand)])}")
        
        # Play trick with verbose=False
        winner, points, events = game.play_trick(first_player, policies, verbose=False)
        
        # Print the plays in order
        for event in events:
            p_id = event.player_id
            card = event.card
            role = "AI" if p_id == 0 else f"P{p_id}"
            print(f"  {role:<4} plays {print_card(card)}")
            
        print(f"Winner: {'AI' if winner == 0 else f'P{winner}'} ({points} pts)")
        
        first_player = winner
        game.rounds += 1
        
    scores = game.end_game_scoring()
    print("\n" + "="*40)
    print(f" FINAL SCORES ")
    print(f" AI: {scores[0]} | P1: {scores[1]} | P2: {scores[2]} | P3: {scores[3]}")
    print("="*40 + "\n")
    return scores

def showcase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Showcase on {device}")
    
    model = HeartsProNet(config.HIDDEN_DIM, config.LSTM_HIDDEN).to(device)
    
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device)['model_state_dict'])
        print(f"Loaded model from {config.MODEL_PATH}")
    elif os.path.exists(config.PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location=device))
        print(f"Loaded pretrained model from {config.PRETRAINED_MODEL_PATH}")
    else:
        print("Warning: No trained model found. Using random weights.")
        
    # Enable PIMC for showcase!
    agent = SotaAgent(model, device, use_pimc=True) 
    game = GameV2()
    
    print("\n" + "="*40)
    print(" SOTA HEARTS AI - SHOWCASE MATCH ")
    print("="*40 + "\n")
    
    # Setup Players: Agent vs 3 Experts
    # We want to see the Agent's thought process
    
    def agent_policy_wrapper(player, info, legal, order):
        
        start_time = time.time()
        action = agent.act(player, info, legal, order, training=False)
        duration = time.time() - start_time
        
        print(f"(Thinking time: {duration:.2f}s)")
        return action
        
    def expert_policy_wrapper(player, info, legal, order):
        action = ExpertPolicy.play_policy(player, info, legal, order)
        # print(f"Expert {player.player_id} plays: {print_card(action)}")
        return action

    policies = [agent_policy_wrapper, expert_policy_wrapper, expert_policy_wrapper, expert_policy_wrapper]
    pass_policies = [agent.pass_policy, ExpertPolicy.pass_policy, ExpertPolicy.pass_policy, ExpertPolicy.pass_policy]
    
    pass_dir = PassDirection.LEFT
    
    # Use custom showcase loop
    run_custom_showcase(game, policies, pass_policies, pass_dir)
    
if __name__ == "__main__":
    showcase()
