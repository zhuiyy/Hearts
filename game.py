import random
from typing import List, Dict, Any, Optional, Tuple
from data_structure import Card, Suit, Player, GameState, TrickRecord, PassDirection

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
        self.rounds = 1 
        self.trick_history: List[TrickRecord] = []
        self.pass_direction = PassDirection.KEEP
        # 换牌信息: passed_cards[i] = 玩家i传出的牌, received_cards[i] = 玩家i收到的牌
        self.passed_cards: List[List[Card]] = [[], [], [], []]
        self.received_cards: List[List[Card]] = [[], [], [], []]

    def reset(self) -> None:
        self.gamestate.reset()
        self.rounds = 1
        self.trick_history = []
        self.pass_direction = PassDirection.KEEP
        self.passed_cards = [[], [], [], []]
        self.received_cards = [[], [], [], []]

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
        
    def get_player_info(self, player_id: int):
        info = self.gamestate.player_info(player_id)
        # Add labels for training if needed
        # We also need global state for Critic
        info['global_state'] = self.get_global_state_vector(player_id)
        
        # --- NEW: RAW History for Feature Engineering ---
        # Pass the full table history so agent can infer voids
        # format: List[Tuple[Card, player_id]]
        info['full_history'] = list(self.gamestate.table)
        
        # --- 换牌信息 ---
        info['passed_cards'] = self.passed_cards[player_id]      # 我传出去的3张牌
        info['received_cards'] = self.received_cards[player_id]  # 我收到的3张牌
        info['pass_direction'] = self.pass_direction              # 传牌方向
        
        return info

    def get_training_labels(self):
        sq_card = Card(Suit.SPADES, 12) 
        sq_label = 4
        for i, p in enumerate(self.gamestate.players):
             if sq_card in p.hand:
                 sq_label = i
                 break
        return sq_label

    def get_global_state_vector(self, perspective_player_id):
        # 156 dims: 3 other players * 52 cards (bool)
        import torch
        vec = []
        for i, p in enumerate(self.gamestate.players):
            if i == perspective_player_id:
                continue
            p_hand = torch.zeros(52)
            for c in p.hand:
                p_hand[c.to_id()] = 1.0
            vec.append(p_hand)
        return torch.cat(vec)

    def run_game_training(self, policies, pass_policies, pass_direction):
        self.reset()
        
        # 0. Pass Phase
        self.pass_direction = pass_direction
        if pass_direction != PassDirection.KEEP:
            passed_cards = [[], [], [], []]
            # Select cards to pass
            for i in range(4):
                pid = i
                p_info = self.get_player_info(pid)
                passed = pass_policies[i](self.gamestate.players[i], p_info)
                passed_cards[i] = passed
                
                # Remove from hand
                for c in passed:
                    self.gamestate.players[i].hand.remove(c)
            
            # Receive cards
            offsets = {PassDirection.LEFT: 1, PassDirection.RIGHT: 3, PassDirection.ACROSS: 2}
            offset = offsets[pass_direction]
            
            for i in range(4):
                source_id = (i - offset) % 4
                received = passed_cards[source_id]
                self.gamestate.players[i].hand.extend(received)
                self.gamestate.players[i].hand.sort()
                
            # 保存换牌信息供特征工程使用
            self.passed_cards = passed_cards
            for i in range(4):
                source_id = (i - offset) % 4
                self.received_cards[i] = passed_cards[source_id]

        # 1. Start Game
        current_player = self.gamestate.get_first_player()
        
        # Track rewards per player per trick
        # shape: [4, 13] (4 players, 13 tricks)
        # trick_rewards[player_id] = [r_trick1, r_trick2, ...]
        trick_rewards = [[0.0] * 13 for _ in range(4)]
        
        # 13 Rounds
        for trick_num in range(1, 14):
            self.gamestate.current_table = []
            self.gamestate.current_suit = None
            
            trick_cards = []
            player_order_in_trick = [] # To know who played what
            
            for i in range(4):
                player_idx = (current_player + i) % 4
                player_order_in_trick.append(player_idx)
                
                player = self.gamestate.players[player_idx]
                
                info = self.get_player_info(player_idx)
                # Determine Valid Moves
                is_first_round = (trick_num == 1)
                scored_hearts = (sum(p.points for p in self.gamestate.players) > 0) 
                
                legal = available_actions(player, self.gamestate.current_suit, is_first_round, scored_hearts)
                
                # SQ Label (for training)
                info['sq_label'] = self.get_training_labels()
                
                # Act
                chosen_card = policies[player_idx](player, info, legal, i) 
                
                if chosen_card not in legal:
                    chosen_card = random.choice(legal)
                
                player.hand.remove(chosen_card)
                self.gamestate.current_table.append((chosen_card, player_idx))
                self.gamestate.table.append((chosen_card, player_idx))
                trick_cards.append(chosen_card)
                
                if i == 0:
                    self.gamestate.current_suit = chosen_card.suit
                
                if chosen_card.suit == Suit.HEARTS:
                    self.gamestate.heart_broken = True
                    
            # End Trick
            winner_local_idx = 0
            best_val = -1
            lead_suit = self.gamestate.current_suit
            
            points = 0
            for idx, c in enumerate(trick_cards):
                if c.suit == lead_suit:
                    val = 14 if c.rank == 1 else c.rank
                    if val > best_val:
                        best_val = val
                        winner_local_idx = idx
                
                points += c.value()
                
            winner_idx = (current_player + winner_local_idx) % 4
            self.gamestate.players[winner_idx].table.extend(trick_cards)
            self.gamestate.players[winner_idx].points += points
            
            # Step Reward Logic
            # The winner gets penalty proportional to points taken
            # Everyone else gets 0 (or small positive for surviving)
            
            # Record points taken this trick for each player (only winner takes points)
            for i in range(4):
                if i == winner_idx:
                    trick_rewards[i][trick_num-1] = -points # Negative reward for taking points
                else:
                    trick_rewards[i][trick_num-1] = 0.0 # Neutral for safe pass
            
            current_player = winner_idx
            
        # End Game
        final_scores = [p.points for p in self.gamestate.players]
        
        # Shoot the Moon Check
        stm_success_player = -1
        for i in range(4):
            if final_scores[i] == 26:
                stm_success_player = i
                for j in range(4):
                    if i == j: 
                        final_scores[j] = 0 
                    else:
                        final_scores[j] = 26
                break
        
        # Adjust Trick Rewards for STM
        # If someone STM, their "penalties" during tricks should turn into massive rewards!
        if stm_success_player != -1:
             for t in range(13):
                 # Invert penalty for STM player: -point -> +point (simplified)
                 # Actually, success STM is just big bonus.
                 # Let's say we override step rewards.
                 # The STM player gets positive rewards for taking points!
                 
                 # Recalculate based on knowledge of STM success
                 # This is "Hindsight Experience Replay" concept basically
                 pass 
                 # For now, let's leave step rewards as "immediate pain" locally, 
                 # and let the final terminal reward fix it. 
                 # Or just clear penalties for STM player.
                 
        return final_scores, trick_rewards, [], []
