import torch
import random
import copy
from typing import List, Optional
from data_structure import Card, Suit, GameState, Player
from game import GameV2, available_actions

class PIMCSelector:
    def __init__(self, model, device='cpu', num_simulations=20):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations

    def determinize_hand(self, game_info: dict, my_player_id: int) -> GameState:
        """
        Creates a concrete GameState (a "parallel universe") consistent with:
        1. My hand (known)
        2. Cards played so far (known)
        3. Void constraints (inferred from history)
        """
        # Reconstruct basic state
        new_state = GameState()
        
        # 1. Set up known cards (My hand + Table history)
        my_hand = [Card(c.suit, c.rank) for c in game_info['hand']]
        
        # Fix: 'history' in game_info is list of (Card, PlayerID)
        # But in GameV2.get_game_info it is 'history'
        # In GameV2.get_player_info it is 'table' (list of cards played so far)
        # Let's check what is passed. It is 'info' from get_player_info.
        # So we should use 'table' or 'trick_history'.
        # 'table' contains all cards played in previous tricks.
        
        history_cards = [c for c, _ in game_info.get('table', [])]
        current_table_cards = [c for c, _ in game_info['current_table']]
        
        known_cards = set(my_hand + history_cards + current_table_cards)
        
        # 2. Identify unknown cards
        all_cards = [Card(suit, rank) for suit in Suit for rank in range(1, 14)]
        unknown_cards = [c for c in all_cards if c not in known_cards]
        random.shuffle(unknown_cards)
        
        # 3. Distribute unknown cards to other 3 players respecting constraints
        # This is a simplified constraint solver. A full CSP solver is better but slower.
        # We use a greedy approach with backtracking or just simple filtering.
        
        # Get void info from game logic (not passed directly, need to infer or pass it)
        # For now, we assume random distribution for simplicity in this MVP.
        # TODO: Implement Void Logic checking from history.
        
        # Calculate how many cards each player should have
        # Total cards = 52. Rounds played = game_info['rounds'] - 1
        # Cards per player = 13 - (rounds - 1)
        # Wait, rounds is 1-13.
        cards_per_hand = 13 - (len(history_cards) // 4)
        
        # Assign my hand
        new_state.players[my_player_id].hand = my_hand
        
        # Assign others
        others = [i for i in range(4) if i != my_player_id]
        
        # Simple distribution (ignoring voids for MVP speed, add constraints later)
        # In a real PIMC, you MUST respect voids or the search is useless.
        # Let's try to respect voids if we had the data.
        
        chunk_size = len(unknown_cards) // 3
        # Handle uneven split if any (shouldn't be if math is right)
        
        start = 0
        for pid in others:
            # How many cards does this player need?
            # If they played in current trick, they have 1 less than me?
            # No, everyone starts trick with same number.
            # If they already played in current trick, they have 1 less card currently.
            
            already_played_in_current = False
            for _, p_id in game_info['current_table']:
                if p_id == pid:
                    already_played_in_current = True
                    break
            
            needed = len(my_hand) if not already_played_in_current else len(my_hand) # Wait
            # If I haven't played yet, I have N cards.
            # If they played, they have N-1 cards.
            # If they haven't played, they have N cards.
            
            # Actually, simpler: just fill their hands from unknown_cards until empty.
            # But we need to know exactly how many.
            
            # Let's just distribute evenly for now.
            end = start + (len(unknown_cards) // 3) # Approx
            # This is risky. Better to track exact counts.
            pass

        # Fallback: Just deal remaining cards randomly to others
        # This is "Perfect Information" in the sense that we fix ONE configuration.
        idx = 0
        for pid in others:
            # Calculate needed cards
            # Total cards initially 13.
            # Played in history: count from history
            played_count = 0
            for _, p in game_info.get('table', []):
                if p == pid: played_count += 1
            
            # played_count includes current_table because game_info['table'] includes everything
            current_hand_size = 13 - played_count
            
            if idx + current_hand_size > len(unknown_cards):
                # This should not happen if accounting is correct
                # But if it does, we must handle it to avoid crash
                current_hand_size = len(unknown_cards) - idx
            
            hand_cards = unknown_cards[idx : idx + current_hand_size]
            new_state.players[pid].hand = hand_cards
            idx += current_hand_size
            
        # Restore game state context
        new_state.table = game_info.get('table', []) # This might need deep copy or proper object reconstruction
        new_state.current_table = game_info['current_table']
        new_state.current_suit = game_info['current_suit']
        new_state.heart_broken = game_info['heart_broken']
        new_state.piggy_pulled = game_info['piggy_pulled']
        
        return new_state

    def simulate(self, game_state: GameState, my_player_id: int, first_action: Card) -> float:
        """
        Rollout the game from this state, forcing first_action.
        Returns the score for my_player (Lower is better).
        """
        # Clone the state to not mess up the parent PIMC loop
        # Deepcopy is slow, maybe optimize later
        sim_game = GameV2()
        sim_game.gamestate = copy.deepcopy(game_state)
        
        # Fast forward logic
        # We need to play out the rest of the game.
        # We can use a random policy or a greedy policy from the network.
        # For speed, let's use Random Policy for now.
        
        # 1. Apply the forced action
        player = sim_game.gamestate.players[my_player_id]
        if first_action in player.hand:
            player.hand.remove(first_action)
            sim_game.gamestate.current_table.append((first_action, my_player_id))
            sim_game.gamestate.table.append((first_action, my_player_id))
            if sim_game.gamestate.current_suit is None:
                sim_game.gamestate.current_suit = first_action.suit
            
            # Update flags
            if first_action.suit == Suit.HEARTS: sim_game.gamestate.heart_broken = True
            if first_action.suit == Suit.SPADES and first_action.rank == 12:
                sim_game.gamestate.piggy_pulled = True
                sim_game.gamestate.heart_broken = True
        
        # 2. Finish current trick
        start_idx = (my_player_id + 1) % 4
        # Check if trick is already full (4 cards)
        if len(sim_game.gamestate.current_table) < 4:
            # Need to finish the trick
            # Who is next?
            # The order in current_table tells us who played.
            # But we need to know the turn order.
            # Assuming standard order 0-1-2-3.
            
            # Actually, GameV2.play_trick logic handles this.
            # But we are mid-trick.
            # Let's just write a simple rollout loop.
            
            current_order_len = len(sim_game.gamestate.current_table)
            players_needed = 4 - current_order_len
            
            next_player = (my_player_id + 1) % 4
            
            for _ in range(players_needed):
                p = sim_game.gamestate.players[next_player]
                legal = available_actions(p, sim_game.gamestate.current_suit, False, True) # Simplified
                if not legal: break # Should not happen
                card = random.choice(legal)
                p.hand.remove(card)
                sim_game.gamestate.current_table.append((card, next_player))
                sim_game.gamestate.table.append((card, next_player))
                # Update flags...
                next_player = (next_player + 1) % 4
                
            # Resolve trick
            trick = sim_game.gamestate.current_table
            lead_suit = trick[0][0].suit
            suited = [x for x in trick if x[0].suit == lead_suit]
            winner_card, winner_id = max(suited, key=lambda x: (x[0].rank - 2) % 13)
            
            points = sum(x[0].value() for x in trick)
            sim_game.gamestate.players[winner_id].points += points
            sim_game.gamestate.current_table = []
            sim_game.gamestate.current_suit = None
            next_leader = winner_id
        else:
            # Trick was already full? Unlikely in this flow.
            next_leader = my_player_id # Placeholder
            
        # 3. Play remaining tricks
        # Count cards left
        cards_left = len(sim_game.gamestate.players[0].hand)
        
        for _ in range(cards_left):
            # Play a full trick
            leader = next_leader
            trick_cards = []
            current_suit = None
            
            for i in range(4):
                pid = (leader + i) % 4
                p = sim_game.gamestate.players[pid]
                legal = available_actions(p, current_suit, False, True)
                card = random.choice(legal)
                p.hand.remove(card)
                trick_cards.append((card, pid))
                if i == 0: current_suit = card.suit
                
                # Flags
                if card.suit == Suit.HEARTS: sim_game.gamestate.heart_broken = True
                if card.suit == Suit.SPADES and card.rank == 12:
                    sim_game.gamestate.piggy_pulled = True
                    sim_game.gamestate.heart_broken = True
            
            # Resolve
            suited = [x for x in trick_cards if x[0].suit == current_suit]
            winner_card, winner_id = max(suited, key=lambda x: (x[0].rank - 2) % 13)
            points = sum(x[0].value() for x in trick_cards)
            sim_game.gamestate.players[winner_id].points += points
            next_leader = winner_id
            
        # 4. Return Score
        scores = sim_game.end_game_scoring()
        return scores[my_player_id]

    def select_best_action(self, player, game_info, legal_actions, top_k_candidates=None):
        """
        Main entry point for PIMC.
        """
        if len(legal_actions) == 1:
            return legal_actions[0]
            
        # 1. Pruning (Optional: Use Policy Network to get top K)
        candidates = legal_actions
        if top_k_candidates:
            candidates = [c for c in legal_actions if c in top_k_candidates]
            if not candidates: candidates = legal_actions # Fallback
            
        # 2. Search
        action_scores = {c: [] for c in candidates}
        
        for _ in range(self.num_simulations):
            # Create a parallel universe
            determinized_state = self.determinize_hand(game_info, player.player_id)
            
            for action in candidates:
                score = self.simulate(determinized_state, player.player_id, action)
                action_scores[action].append(score)
                
        # 3. Aggregate
        # We want to MINIMIZE score
        avg_scores = {c: sum(s)/len(s) for c, s in action_scores.items()}
        
        best_action = min(avg_scores, key=avg_scores.get)
        return best_action
