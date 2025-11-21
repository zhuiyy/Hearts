import random
from typing import List, Dict, Any, Optional, Tuple
from data_structure import Card, Suit

def get_card_strength(card: Card) -> int:
    if card.rank == 1:
        return 14
    return card.rank

def random_policy(player: Any, info: Dict[str, Any], legal_actions: List[Card], order: int) -> Card:
    return random.choice(legal_actions)

def random_pass_policy(player: Any, info: Dict[str, Any]) -> List[Card]:
    return random.sample(player.hand, 3)

def min_policy(player: Any, info: Dict[str, Any], legal_actions: List[Card], order: int) -> Card:
    return min(legal_actions, key=get_card_strength)

def max_policy(player: Any, info: Dict[str, Any], legal_actions: List[Card], order: int) -> Card:
    return max(legal_actions, key=get_card_strength)

class ExpertPolicy:
    @staticmethod
    def pass_policy(player: Any, info: Dict[str, Any]) -> List[Card]:
        hand = list(player.hand)
        
        def pass_score(card: Card) -> int:
            score = 0
            if card.suit == Suit.SPADES:
                if card.rank == 12: score += 100 # Queen
                elif card.rank == 13: score += 90 # King
                elif card.rank == 1: score += 95 # Ace
            elif card.suit == Suit.HEARTS:
                if card.rank == 1: score += 50 # Ace
                elif card.rank >= 10: score += card.rank + 20
                else: score += card.rank
            else:
                if card.rank == 1: score += 15
                else: score += card.rank
            return score
            
        hand.sort(key=pass_score, reverse=True)
        return hand[:3]

    @staticmethod
    def play_policy(player: Any, info: Dict[str, Any], legal_actions: List[Card], order: int) -> Card:
        current_table: List[Tuple[Card, int]] = info['current_table']
        current_suit: Optional[Suit] = info['current_suit']
        piggy_pulled: bool = info.get('piggy_pulled', False)
        
        # 1. Leading
        if not current_table:
            # Piggy Hunting Logic: If Pig is not out, and we have safe spades but no dangerous spades, lead low spade.
            if not piggy_pulled:
                my_spades = [c for c in legal_actions if c.suit == Suit.SPADES]
                dangerous_spades = [c for c in my_spades if c.rank in [1, 12, 13]]
                safe_spades = [c for c in my_spades if c.rank not in [1, 12, 13]]
                
                if safe_spades and not dangerous_spades:
                    return min(safe_spades, key=get_card_strength)

            # Standard Lead Logic
            safe_leads = [c for c in legal_actions if c.suit != Suit.SPADES and c.suit != Suit.HEARTS]
            if not safe_leads:
                safe_leads = [c for c in legal_actions if c.suit != Suit.SPADES]
            
            if not safe_leads:
                safe_leads = legal_actions
                
            return min(safe_leads, key=get_card_strength)

        # 2. Following
        else:
            following_cards = [c for c in legal_actions if c.suit == current_suit]
            
            if following_cards:
                winning_rank = -1
                for c, _ in current_table:
                    if c.suit == current_suit:
                        r = 14 if c.rank == 1 else c.rank
                        if r > winning_rank:
                            winning_rank = r
                            
                safe_follows = [c for c in following_cards if (14 if c.rank == 1 else c.rank) < winning_rank]
                
                if safe_follows:
                    return max(safe_follows, key=get_card_strength)
                else:
                    return max(following_cards, key=get_card_strength)
            
            else:
                # Void - Dump cards
                sq = next((c for c in legal_actions if c.suit == Suit.SPADES and c.rank == 12), None)
                if sq: return sq
                
                dangerous_spades = [c for c in legal_actions if c.suit == Suit.SPADES and c.rank in [1, 13]]
                if dangerous_spades: return max(dangerous_spades, key=get_card_strength)
                
                high_hearts = [c for c in legal_actions if c.suit == Suit.HEARTS]
                if high_hearts: return max(high_hearts, key=get_card_strength)
                
                return max(legal_actions, key=get_card_strength)
