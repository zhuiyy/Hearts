"""
Hearts Game Logic - Refactored Version
Focuses on Training and Showcase modes, removing human interaction.
"""

import random
from typing import List, Callable, Optional, Tuple, Dict, Any
from data_structure import Suit, Card, Player, GameState, TrickRecord, PlayEvent, PassDirection, PassEvent


def available_actions(player: Player, suit: Optional[Suit], is_first_round: bool, scored: bool) -> List[Card]:
    """
    Determine legal moves for a player.
    
    :param player: Current player
    :param suit: Leading suit of the current trick (None if leading)
    :param is_first_round: Whether this is the first trick of the game (2 of Clubs lead)
    :param scored: Whether any points have been scored (hearts broken condition check usually, but here 'scored' implies points on table?)
                   Actually in standard hearts, 'scored' usually refers to if hearts have been broken.
                   Let's stick to the logic from the original file.
    """
    hand = player.hand
    if is_first_round:
        if suit is None:
            # Must lead 2 of Clubs if holding it (standard rule, usually enforced by game start)
            # But if we are just checking general logic:
            # In standard hearts, the player with 2C leads it.
            # If this function is called for the leader of the first round, they MUST have 2C.
            # If for some reason they don't (e.g. custom deal), we might need fallback.
            # Assuming standard deal:
            c2 = Card(Suit.CLUBS, 2)
            if c2 in hand:
                return [c2]
            # If not holding 2C (shouldn't happen for leader), return all valid leads?
            # Original code returned [2C] if suit is None.
            return [c2] 
        else:
            # Following in first round
            suited = [c for c in hand if c.suit == suit]
            if suited:
                return sorted(suited)
            
            # Cannot play points (Hearts or QS) on first trick usually
            # Original code: non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
            # And filter out QS (Spades 12)
            non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
            safe_cards = [c for c in non_hearts if not (c.suit == Suit.SPADES and c.rank == 12)]
            
            if not scored and safe_cards:
                 return sorted(safe_cards)
            
            # If only have point cards, must play them (slough)
            if not scored and non_hearts:
                 return sorted(non_hearts)
                 
            return sorted(hand)
    else:
        if suit is None:
            # Leading
            if not scored: # "scored" here seems to map to "hearts broken" in original logic context?
                # Original logic: if not scored, cannot lead Hearts
                non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
                if non_hearts:
                    return sorted(non_hearts)
                else:
                    # Only have hearts, must lead hearts
                    return sorted(hand)
            else:
                # Hearts broken, can lead anything
                return sorted(hand)
        else:
            # Following
            suited = [c for c in hand if c.suit == suit]
            if suited:
                return sorted(suited)
            
            # Sloughing
            # Original logic:
            # non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
            # if not scored and non_hearts: return sorted(non_hearts)
            # return sorted(player.hand)
            
            # Wait, standard rules allow sloughing anything if void in suit.
            # The restriction "if not scored" usually applies to breaking hearts?
            # Or maybe "scored" means "points on table"?
            # Let's stick to original logic to be safe.
            non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
            if not scored and non_hearts:
                return sorted(non_hearts)
            return sorted(hand)


def card_value(card: Card) -> int:
    return card.value()


class GameV2:
    def __init__(self) -> None:
        self.gamestate = GameState()
        self.rounds = 1 # Actually this tracks the trick number (1-13)
        self.trick_history: List[TrickRecord] = []
        self.pass_direction = PassDirection.KEEP

    def reset(self) -> None:
        self.gamestate.reset()
        self.rounds = 1
        self.trick_history = []
        self.pass_direction = PassDirection.KEEP

    def get_game_info(self) -> Dict[str, Any]:
        # Construct a dictionary similar to original game_info
        # But we can use the GameState's method if available or build it here
        # Original game_info returned: rounds, players (snapshot), scoreboard, deck, history, current_table...
        
        players_snapshot = []
        for p in self.gamestate.players:
            players_snapshot.append({
                'player_id': p.player_id,
                'hand': list(p.hand),
                'points': p.points,
                'taken_tricks': list(p.table) # p.table stores cards taken?
            })

        return {
            'rounds': self.rounds,
            'players': players_snapshot,
            'scoreboard': [p.points for p in self.gamestate.players],
            'deck': list(self.gamestate.deck),
            'history': list(self.gamestate.table), # All cards played in order?
            'current_table': list(self.gamestate.current_table),
            'current_suit': self.gamestate.current_suit,
            'hearts_broken': self.gamestate.heart_broken,
            'piggy_pulled': self.gamestate.piggy_pulled,
            'current_order': len(self.gamestate.current_table)
        }

    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        # Similar to original player_info
        p = self.gamestate.players[player_id]
        
        # Gather public stats for all players
        players_stats = []
        for pl in self.gamestate.players:
            has_sq = False
            for c in pl.table:
                if c.suit == Suit.SPADES and c.rank == 12:
                    has_sq = True
                    break
            players_stats.append({
                'points': pl.points,
                'has_sq': has_sq
            })

        return {
            'player_id': p.player_id,
            'hand': list(p.hand),
            'points': p.points,
            'taken_tricks': list(p.table),
            'scoreboard': [pl.points for pl in self.gamestate.players],
            'players_stats': players_stats, # New field
            'table': list(self.gamestate.table),
            'current_table': list(self.gamestate.current_table),
            'current_suit': self.gamestate.current_suit,
            'heart_broken': self.gamestate.heart_broken,
            'piggy_pulled': self.gamestate.piggy_pulled,
            'rounds': self.rounds,
            'current_order': len(self.gamestate.current_table),
            'pass_direction': self.pass_direction # New field
        }

    def perform_pass(self, pass_direction: PassDirection, pass_policies: List[Callable]) -> List[PassEvent]:
        """
        Execute the passing phase.
        
        :param pass_direction: Direction to pass cards.
        :param pass_policies: List of 4 policy functions for passing. 
                              Signature: (player, player_info) -> List[Card] (3 cards)
        :return: List of PassEvent
        """
        if pass_direction == PassDirection.KEEP:
            return []

        pass_events = []
        passed_cards_map = {} # player_id -> cards passed BY them

        # 1. Select cards to pass
        for i, player in enumerate(self.gamestate.players):
            p_info = self.get_player_info(i)
            # Policy should return 3 cards to pass
            cards_to_pass = pass_policies[i](player, p_info)
            
            if len(cards_to_pass) != 3:
                raise ValueError(f"Player {i} selected {len(cards_to_pass)} cards to pass, expected 3.")
            
            for card in cards_to_pass:
                if card not in player.hand:
                    raise ValueError(f"Player {i} tried to pass card {card} not in hand.")
                player.hand.remove(card)
            
            passed_cards_map[i] = cards_to_pass

        # 2. Distribute cards
        for i in range(4):
            target_idx = -1
            if pass_direction == PassDirection.LEFT:
                target_idx = (i + 1) % 4
            elif pass_direction == PassDirection.RIGHT:
                target_idx = (i - 1) % 4
            elif pass_direction == PassDirection.ACROSS:
                target_idx = (i + 2) % 4
            
            received_cards = passed_cards_map[target_idx] # Wait, if I am i, I receive from someone who passed TO me.
            # Left (Clockwise): i passes to i+1. So i receives from i-1.
            # Right (Counter-Clockwise): i passes to i-1. So i receives from i+1.
            # Across: i passes to i+2. So i receives from i+2 (since +2 and -2 mod 4 are same).
            
            # Let's re-logic:
            # We iterate over players who ARE RECEIVING.
            # Or iterate over players who ARE PASSING and put into target.
            pass
        
        # Let's iterate over SENDERS to make it easier
        for sender_idx in range(4):
            target_idx = -1
            if pass_direction == PassDirection.LEFT:
                target_idx = (sender_idx + 1) % 4
            elif pass_direction == PassDirection.RIGHT:
                target_idx = (sender_idx - 1) % 4
            elif pass_direction == PassDirection.ACROSS:
                target_idx = (sender_idx + 2) % 4
            
            cards = passed_cards_map[sender_idx]
            self.gamestate.players[target_idx].hand.extend(cards)
            
            # We need to record what sender_idx passed and what they received.
            # But we can't record received until everyone has passed.
            # So we do this in two passes or just store it.
            
        # Sort hands after receiving
        for player in self.gamestate.players:
            player.hand.sort()

        # Create events
        for i in range(4):
            # Who sent to i?
            sender_to_i = -1
            if pass_direction == PassDirection.LEFT:
                sender_to_i = (i - 1) % 4
            elif pass_direction == PassDirection.RIGHT:
                sender_to_i = (i + 1) % 4
            elif pass_direction == PassDirection.ACROSS:
                sender_to_i = (i + 2) % 4
            
            received = passed_cards_map[sender_to_i]
            passed = passed_cards_map[i]
            
            event = PassEvent(
                player_id=i,
                direction=pass_direction,
                passed_cards=passed,
                received_cards=received
            )
            pass_events.append(event)
            
        return pass_events

    def play_trick(self, first_player_idx: int, policies: List[Callable], verbose: bool = False) -> Tuple[int, int, List[PlayEvent]]:
        """
        Play a single trick (one round of 4 cards).
        
        :param first_player_idx: Index of the player who leads.
        :param policies: List of 4 policy functions.
        :param verbose: Whether to print game state.
        :return: (winner_idx, points_in_trick, events_log)
        """
        self.gamestate.current_table = []
        self.gamestate.current_suit = None
        
        trick_events: List[PlayEvent] = []
        trick_cards: List[Tuple[Card, int]] = []
        
        if verbose:
            print(f'--- Trick {self.rounds} ---')
            for i, p in enumerate(self.gamestate.players):
                print(f'Player {i} hand: {sorted(p.hand)}')

        # Check if hearts broken or scored condition for available_actions
        # Original code used 'scored' = any(p.points > 0)
        # But usually available_actions checks 'heart_broken' state directly inside?
        # Wait, original available_actions takes 'scored' param.
        # And inside it uses 'scored' to decide if hearts can be played/led.
        # In original code: scored = any(p.points > 0 for p in self.gamestate.players)
        # This implies "points have been taken by someone".
        scored = any(p.points > 0 for p in self.gamestate.players)

        for i in range(4):
            current_player_idx = (first_player_idx + i) % 4
            player = self.gamestate.players[current_player_idx]
            is_first_trick = (self.rounds == 1)
            
            # Determine legal actions
            legal_moves = available_actions(player, self.gamestate.current_suit, is_first_trick, scored)
            
            if verbose:
                print(f"Player {current_player_idx} legal moves: {legal_moves}")

            # Get action from policy
            # Policy signature: (player, player_info, legal_moves, order_in_trick) -> Card
            # We need to construct player_info
            p_info = self.get_player_info(current_player_idx)
            selected_card = policies[current_player_idx](player, p_info, legal_moves, i)
            
            # Validate action
            if selected_card not in legal_moves:
                raise ValueError(f"Player {current_player_idx} played illegal card {selected_card}. Legal: {legal_moves}")

            # Update State
            player.hand.remove(selected_card)
            
            # Record Event
            event = PlayEvent(
                player_id=current_player_idx,
                card=selected_card,
                round_number=self.rounds, # Using rounds as trick number
                trick_number=self.rounds,
                is_lead=(i == 0),
                current_table=list(self.gamestate.current_table),
                heart_broken=self.gamestate.heart_broken,
                piggy_pulled=self.gamestate.piggy_pulled,
                legal_actions=legal_moves
            )
            trick_events.append(event)

            # Update Game Logic
            if not self.gamestate.piggy_pulled and (selected_card.suit == Suit.SPADES and selected_card.rank == 12):
                self.gamestate.piggy_pulled = True
                self.gamestate.heart_broken = True
            
            if selected_card.suit == Suit.HEARTS and not self.gamestate.heart_broken:
                self.gamestate.heart_broken = True

            # Add to tables
            # player.table.append(selected_card) # Wait, player.table usually stores taken cards? 
            # In original code: player.table.append(card) -> This seems to be "cards played by player"?
            # Let's check original code: 
            # player.table.append(card)
            # table.append((card, player_idx))
            # self.gamestate.table.append((card, player_idx))
            # self.gamestate.current_table.append((card, player_idx))
            
            # If player.table is "cards played by this player historically", then yes.
            player.table.append(selected_card) 
            
            self.gamestate.table.append((selected_card, current_player_idx))
            self.gamestate.current_table.append((selected_card, current_player_idx))
            trick_cards.append((selected_card, current_player_idx))

            if i == 0:
                self.gamestate.current_suit = selected_card.suit
            
            if verbose:
                print(f"Player {current_player_idx} plays {selected_card}")

        # Determine Winner
        lead_suit_cards = [(c, pid) for c, pid in trick_cards if c.suit == self.gamestate.current_suit]
        # Highest rank wins. Ace is 1? Wait, Card rank: 1-13.
        # Original code: key=lambda x: (x[0].rank - 2) % 13
        # If rank 1 (Ace), (1-2)%13 = -1%13 = 12 (Highest)
        # If rank 13 (King), (13-2)%13 = 11
        # If rank 2, (2-2)%13 = 0 (Lowest)
        # So Ace is high.
        winner_card, winner_idx = max(lead_suit_cards, key=lambda x: (x[0].rank - 2) % 13)
        
        # Calculate Points
        points = sum(card_value(c) for c, _ in trick_cards)
        
        # Update Winner Stats
        self.gamestate.players[winner_idx].points += points
        # In original code, winner also gets the cards added to their 'table'? 
        # No, original code: player.table.append(card) happened during play.
        # But wait, usually 'table' in Player struct might mean 'tricks taken'?
        # Original code: player.table.append(card) happens when they PLAY the card.
        # So player.table is history of played cards.
        
        # Record Trick History
        record = TrickRecord(winner=winner_idx, score=points, lead_suit=self.gamestate.current_suit)
        self.trick_history.append(record)
        
        if verbose:
            print(f"Player {winner_idx} wins trick with {points} points.")
            print("-" * 20)

        return winner_idx, points, trick_events

    def run_game_training(self, policies: List[Callable], pass_policies: Optional[List[Callable]] = None, pass_direction: PassDirection = PassDirection.KEEP) -> Tuple[List[int], List[int], List[Any], List[TrickRecord]]:
        """
        Run a full game for training purposes.
        Returns final scores, raw scores, list of all events, and trick history.
        """
        self.reset()
        all_events = []

        # Passing Phase
        if pass_direction != PassDirection.KEEP:
            self.pass_direction = pass_direction # Store it
            if pass_policies is None:
                # Default random pass policy if none provided
                def random_pass(p, info):
                    return random.sample(p.hand, 3)
                pass_policies = [random_pass] * 4
            
            pass_events = self.perform_pass(pass_direction, pass_policies)
            all_events.extend(pass_events)

        first_player = self.gamestate.get_first_player()
        
        for _ in range(13):
            winner, _, events = self.play_trick(first_player, policies, verbose=False)
            first_player = winner
            all_events.extend(events)
            self.rounds += 1
            
        scores = self.end_game_scoring()
        raw_scores = [p.points for p in self.gamestate.players]
        return scores, raw_scores, all_events, self.trick_history

    def run_game_showcase(self, policies: List[Callable], pass_policies: Optional[List[Callable]] = None, pass_direction: PassDirection = PassDirection.KEEP) -> List[int]:
        """
        Run a full game with verbose output for demonstration.
        """
        self.reset()
        
        # Passing Phase
        if pass_direction != PassDirection.KEEP:
            print(f"--- Passing Phase ({pass_direction.name}) ---")
            if pass_policies is None:
                def random_pass(p, info):
                    return random.sample(p.hand, 3)
                pass_policies = [random_pass] * 4
                
            pass_events = self.perform_pass(pass_direction, pass_policies)
            for event in pass_events:
                print(f"Player {event.player_id} passed {event.passed_cards} and received {event.received_cards}")
            print("-" * 20)

        first_player = self.gamestate.get_first_player()
        
        for _ in range(13):
            winner, _, _ = self.play_trick(first_player, policies, verbose=True)
            first_player = winner
            self.rounds += 1
            
        scores = self.end_game_scoring()
        print("Final Scores:", scores)
        return scores

    def end_game_scoring(self) -> List[int]:
        """
        Handle Shooting the Moon logic and return final scores.
        """
        scores = [p.points for p in self.gamestate.players]
        if 26 in scores:
            # Shooting the Moon
            new_scores = []
            for s in scores:
                if s == 26:
                    new_scores.append(0)
                else:
                    new_scores.append(26)
            return new_scores
        return scores

if __name__ == '__main__':
    # Simple test
    def random_policy(player, info, actions, order):
        return random.choice(actions)
    
    game = GameV2()
    print("Running Showcase Game with Passing (LEFT)...")
    game.run_game_showcase([random_policy] * 4, pass_direction=PassDirection.LEFT)
