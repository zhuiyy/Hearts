"""
Hearts Game Logic - Refactored Version
Focuses on Training and Showcase modes, removing human interaction.
"""

import random
from typing import List, Callable, Optional, Tuple, Dict, Any
from data_structure import Suit, Card, Player, GameState, TrickRecord, PlayEvent, PassDirection, PassEvent


def available_actions(player: Player, suit: Optional[Suit], is_first_round: bool, scored: bool) -> List[Card]:
    hand = player.hand
    if is_first_round:
        if suit is None:
            c2 = Card(Suit.CLUBS, 2)
            if c2 in hand:
                return [c2]
            return [c2] 
        else:
            suited = [c for c in hand if c.suit == suit]
            if suited:
                return sorted(suited)
            
            non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
            safe_cards = [c for c in non_hearts if not (c.suit == Suit.SPADES and c.rank == 12)]
            
            if not scored and safe_cards:
                 return sorted(safe_cards)
            
            if not scored and non_hearts:
                 return sorted(non_hearts)
                 
            return sorted(hand)
    else:
        if suit is None:
            if not scored: 
                non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
                if non_hearts:
                    return sorted(non_hearts)
                else:
                    return sorted(hand)
            else:
                return sorted(hand)
        else:
            suited = [c for c in hand if c.suit == suit]
            if suited:
                return sorted(suited)
            
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
        players_snapshot = []
        for p in self.gamestate.players:
            players_snapshot.append({
                'player_id': p.player_id,
                'hand': list(p.hand),
                'points': p.points,
                'taken_tricks': list(p.table)
            })

        return {
            'rounds': self.rounds,
            'players': players_snapshot,
            'scoreboard': [p.points for p in self.gamestate.players],
            'deck': list(self.gamestate.deck),
            'history': list(self.gamestate.table),
            'current_table': list(self.gamestate.current_table),
            'current_suit': self.gamestate.current_suit,
            'hearts_broken': self.gamestate.heart_broken,
            'piggy_pulled': self.gamestate.piggy_pulled,
            'current_order': len(self.gamestate.current_table)
        }

    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        p = self.gamestate.players[player_id]
        
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
            'my_id': p.player_id, # Added alias for convenience
            'hand': list(p.hand),
            'points': p.points,
            'taken_tricks': list(p.table),
            'scoreboard': [pl.points for pl in self.gamestate.players],
            'players_stats': players_stats,
            'table': list(self.gamestate.table),
            'current_table': list(self.gamestate.current_table),
            'current_suit': self.gamestate.current_suit,
            'heart_broken': self.gamestate.heart_broken,
            'piggy_pulled': self.gamestate.piggy_pulled,
            'rounds': self.rounds,
            'current_order': len(self.gamestate.current_table),
            'pass_direction': self.pass_direction,
            'trick_history': list(self.trick_history), # Added trick_history
            'game_state': self.gamestate # Added game_state object for direct access if needed
        }

    def perform_pass(self, pass_direction: PassDirection, pass_policies: List[Callable]) -> List[PassEvent]:
        if pass_direction == PassDirection.KEEP:
            return []

        pass_events = []
        passed_cards_map = {} 

        for i, player in enumerate(self.gamestate.players):
            p_info = self.get_player_info(i)
            cards_to_pass = pass_policies[i](player, p_info)
            
            if len(cards_to_pass) != 3:
                raise ValueError(f"Player {i} selected {len(cards_to_pass)} cards to pass, expected 3.")
            
            for card in cards_to_pass:
                if card not in player.hand:
                    raise ValueError(f"Player {i} tried to pass card {card} not in hand.")
                player.hand.remove(card)
            
            passed_cards_map[i] = cards_to_pass

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
            
        for player in self.gamestate.players:
            player.hand.sort()

        for i in range(4):
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
        self.gamestate.current_table = []
        self.gamestate.current_suit = None
        
        trick_events: List[PlayEvent] = []
        trick_cards: List[Tuple[Card, int]] = []
        
        if verbose:
            print(f'--- Trick {self.rounds} ---')
            for i, p in enumerate(self.gamestate.players):
                print(f'Player {i} hand: {sorted(p.hand)}')

        scored = any(p.points > 0 for p in self.gamestate.players)

        for i in range(4):
            current_player_idx = (first_player_idx + i) % 4
            player = self.gamestate.players[current_player_idx]
            is_first_trick = (self.rounds == 1)
            
            legal_moves = available_actions(player, self.gamestate.current_suit, is_first_trick, scored)
            
            if verbose:
                print(f"Player {current_player_idx} legal moves: {legal_moves}")

            p_info = self.get_player_info(current_player_idx)
            selected_card = policies[current_player_idx](player, p_info, legal_moves, i)
            
            if selected_card not in legal_moves:
                raise ValueError(f"Player {current_player_idx} played illegal card {selected_card}. Legal: {legal_moves}")

            player.hand.remove(selected_card)
            
            event = PlayEvent(
                player_id=current_player_idx,
                card=selected_card,
                round_number=self.rounds,
                trick_number=self.rounds,
                is_lead=(i == 0),
                current_table=list(self.gamestate.current_table),
                heart_broken=self.gamestate.heart_broken,
                piggy_pulled=self.gamestate.piggy_pulled,
                legal_actions=legal_moves
            )
            trick_events.append(event)

            if not self.gamestate.piggy_pulled and (selected_card.suit == Suit.SPADES and selected_card.rank == 12):
                self.gamestate.piggy_pulled = True
                self.gamestate.heart_broken = True
            
            if selected_card.suit == Suit.HEARTS and not self.gamestate.heart_broken:
                self.gamestate.heart_broken = True

            player.table.append(selected_card) 
            
            self.gamestate.table.append((selected_card, current_player_idx))
            self.gamestate.current_table.append((selected_card, current_player_idx))
            trick_cards.append((selected_card, current_player_idx))

            if i == 0:
                self.gamestate.current_suit = selected_card.suit
            
            if verbose:
                print(f"Player {current_player_idx} plays {selected_card}")

        lead_suit_cards = [(c, pid) for c, pid in trick_cards if c.suit == self.gamestate.current_suit]
        winner_card, winner_idx = max(lead_suit_cards, key=lambda x: (x[0].rank - 2) % 13)
        
        points = sum(card_value(c) for c, _ in trick_cards)
        
        self.gamestate.players[winner_idx].points += points
        
        record = TrickRecord(winner=winner_idx, score=points, lead_suit=self.gamestate.current_suit)
        record.cards = trick_cards # Dynamically add cards to record
        
        self.trick_history.append(record)
        
        if verbose:
            print(f"Player {winner_idx} wins trick with {points} points.")
            print("-" * 20)

        return winner_idx, points, trick_events

    def run_game_training(self, policies: List[Callable], pass_policies: Optional[List[Callable]] = None, pass_direction: PassDirection = PassDirection.KEEP):
        self.reset()
        all_events = []

        if pass_direction != PassDirection.KEEP:
            self.pass_direction = pass_direction
            if pass_policies is None:
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
        self.reset()
        
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
        scores = [p.points for p in self.gamestate.players]
        if 26 in scores:
            new_scores = []
            for s in scores:
                if s == 26:
                    new_scores.append(0)
                else:
                    new_scores.append(26)
            return new_scores
        return scores

if __name__ == '__main__':
    def random_policy(player, info, actions, order):
        return random.choice(actions)
    
    game = GameV2()
    print("Running Showcase Game with Passing (LEFT)...")
    game.run_game_showcase([random_policy] * 4, pass_direction=PassDirection.LEFT)
