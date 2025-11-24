import random
from typing import List, Dict, Any, Optional, Tuple
from data_structure import Card, Suit

def get_card_strength(card: Card) -> tuple:
    # Return a tuple (strength, suit_value) to ensure deterministic comparison
    # Strength: Ace (14) > King (13) > ... > 2
    # Suit: Spades (0) < Hearts (1) < Diamonds (2) < Clubs (3) (Arbitrary but fixed)
    strength = 14 if card.rank == 1 else card.rank
    return (strength, card.suit.value)

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
        
        # Shooting the Moon (STM) Detection
        # Heuristic: If we have many high cards (A, K, Q) and long suits, we might try STM.
        # Specifically, if we have the Spades Queen/King/Ace and good Hearts support.
        
        high_cards = [c for c in hand if c.rank >= 10 or c.rank == 1]
        spades = [c for c in hand if c.suit == Suit.SPADES]
        hearts = [c for c in hand if c.suit == Suit.HEARTS]
        
        has_sq = any(c.rank == 12 for c in spades)
        has_sk = any(c.rank == 13 for c in spades)
        has_sa = any(c.rank == 1 for c in spades)
        
        # Simple STM criteria: Have SQ, SK, SA and at least 3-4 high hearts
        attempt_stm = False
        if has_sq and has_sk and has_sa:
            high_hearts_count = sum(1 for c in hearts if c.rank >= 10 or c.rank == 1)
            if high_hearts_count >= 3:
                attempt_stm = True
                
        if attempt_stm:
            # Strategy: Keep high cards, pass low cards or holes.
            # Pass lowest cards to clear hand of losers.
            # But keep some low cards for lead control? No, in STM you want to win everything.
            # You want to pass cards that you CANNOT win with.
            # i.e. Low cards in suits where you don't have the Ace/King.
            
            def stm_pass_score(card: Card) -> int:
                # Higher score = More likely to pass (Bad for STM)
                # We want to keep High cards (Low score)
                # We want to pass Low cards (High score)
                
                rank_val = 14 if card.rank == 1 else card.rank
                return 15 - rank_val # 2 -> 13 (Pass), A -> 1 (Keep)
            
            hand.sort(key=stm_pass_score, reverse=True)
            return hand[:3]
            
        else:
            # Normal Strategy: Avoid taking points.
            # Pass dangerous high cards.
            def pass_score(card: Card) -> int:
                score = 0
                if card.suit == Suit.SPADES:
                    if card.rank == 12: score += 100 # Queen - DANGEROUS
                    elif card.rank == 13: score += 90 # King - Dangerous if Q is out
                    elif card.rank == 1: score += 95 # Ace - Dangerous if Q is out
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
        
        # Check if we are currently Shooting the Moon
        # Condition: We have taken all point cards so far, and there are point cards taken.
        # Or simply: We have points > 0, and no one else has points.
        # But 'info' might not have full player stats easily accessible in a clean way?
        # info['players_stats'] is available in AIPlayer context, let's assume it's passed here or we can infer.
        # The 'player' object has 'points'.
        # We need to know if OTHERS have points.
        # info['players_stats'] is a list of dicts or objects.
        
        others_have_points = False
        my_points = player.points
        
        # If we can access game state via info (hacky but possible if passed)
        # Usually info contains 'players_stats' from GameV2.get_player_info
        if 'players_stats' in info:
            for i, stats in enumerate(info['players_stats']):
                if i != info['player_id']: # Not me
                    if stats['points'] > 0:
                        others_have_points = True
                        break
        
        # STM Mode if I have points and no one else does (and points exist)
        # Or if no points taken yet, we might still be aiming for it (based on hand).
        # For simplicity, let's stick to "If I have points and others don't, try to keep winning".
        # But be careful not to ruin it.
        
        trying_stm = (my_points > 0 and not others_have_points)
        
        # 1. Leading
        if not current_table:
            if trying_stm:
                # Lead highest card to win
                return max(legal_actions, key=get_card_strength)
            
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
                
                # Can we win this trick?
                # If trying STM, we WANT to win if there are points or to keep control.
                if trying_stm:
                    # Try to win with highest card
                    high_card = max(following_cards, key=get_card_strength)
                    h_val = 14 if high_card.rank == 1 else high_card.rank
                    if h_val > winning_rank:
                        return high_card
                    else:
                        # Can't win, play lowest to save high cards? Or dump?
                        return min(following_cards, key=get_card_strength)

                # Normal: Avoid winning if points or dangerous
                # Is there a point card in the trick?
                points_in_trick = sum(c.value() for c, _ in current_table)
                
                safe_follows = [c for c in following_cards if (14 if c.rank == 1 else c.rank) < winning_rank]
                
                if safe_follows:
                    # If points in trick, definitely play safe.
                    # If no points, maybe play high to get rid of high cards (but stay under winner)
                    if points_in_trick > 0:
                        return max(safe_follows, key=get_card_strength) # Highest safe card
                    else:
                        return max(safe_follows, key=get_card_strength) # Still highest safe
                else:
                    # Must win (or play under if possible but we checked safe_follows)
                    # If we must win, win as cheaply as possible? Or get rid of high card?
                    # If points in trick, win high to save low for later? No, you eat points anyway.
                    # Win high to get lead and maybe lead safe next?
                    return max(following_cards, key=get_card_strength)
            
            else:
                # Void - Dump cards
                if trying_stm:
                    # Dump low cards? Or dump cards that are useless?
                    # Actually, if void, we can't win this trick (unless we trump? No trumps in Hearts).
                    # So we lose this trick.
                    # If we lose a trick with points, STM fails.
                    # If trick has points, and we can't win it, STM is dead (unless we are partner? No partners).
                    # So if we are void, and trick has points, we lost STM.
                    # Just play normally.
                    pass

                sq = next((c for c in legal_actions if c.suit == Suit.SPADES and c.rank == 12), None)
                if sq: return sq
                
                dangerous_spades = [c for c in legal_actions if c.suit == Suit.SPADES and c.rank in [1, 13]]
                if dangerous_spades: return max(dangerous_spades, key=get_card_strength)
                
                high_hearts = [c for c in legal_actions if c.suit == Suit.HEARTS]
                if high_hearts: return max(high_hearts, key=get_card_strength)
                
                return max(legal_actions, key=get_card_strength)
                
                return max(legal_actions, key=get_card_strength)
