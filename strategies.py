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
            def stm_pass_score(card: Card) -> int:
                rank_val = 14 if card.rank == 1 else card.rank
                return 15 - rank_val # 2 -> 13 (Pass), A -> 1 (Keep)
            
            hand.sort(key=stm_pass_score, reverse=True)
            return hand[:3]
            
        else:
            # Normal Strategy: Avoid taking points.
            # Pass dangerous high cards.
            # NEW: Prioritize creating voids (short suits)
            
            suit_counts = {s: 0 for s in Suit}
            for c in hand:
                suit_counts[c.suit] += 1
                
            def pass_score(card: Card) -> int:
                score = 0
                # Danger Score
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
                
                # Void Creation Bonus
                # If suit length is <= 3 and we don't have high cards in it, try to void it.
                # But don't void if we have dangerous cards (handled by danger score).
                count = suit_counts[card.suit]
                if count <= 3:
                    # Check if we have dangerous cards in this suit
                    has_danger = False
                    if card.suit == Suit.SPADES:
                        has_danger = any(c.rank in [12, 13, 1] for c in hand if c.suit == Suit.SPADES)
                    
                    if not has_danger:
                        score += (4 - count) * 10 # Shorter suit -> Higher bonus
                        
                return score
                
            hand.sort(key=pass_score, reverse=True)
            return hand[:3]

    @staticmethod
    def play_policy(player: Any, info: Dict[str, Any], legal_actions: List[Card], order: int) -> Card:
        current_table: List[Tuple[Card, int]] = info['current_table']
        current_suit: Optional[Suit] = info['current_suit']
        piggy_pulled: bool = info.get('piggy_pulled', False)
        
        # Helper to identify cards
        my_spades = [c for c in legal_actions if c.suit == Suit.SPADES]
        has_sq = any(c.rank == 12 for c in my_spades)
        
        # Check if we are currently Shooting the Moon
        others_have_points = False
        my_points = player.points
        
        # Check if SOMEONE ELSE is Shooting the Moon (STM Blocking)
        opponent_stm_threat = False
        if 'players_stats' in info:
            for i, stats in enumerate(info['players_stats']):
                if i != info['player_id']: # Not me
                    if stats['points'] > 0:
                        others_have_points = True
                    if stats['points'] >= 18: # Threat threshold
                        opponent_stm_threat = True
        
        trying_stm = (my_points > 0 and not others_have_points)
        
        # 1. Leading
        if not current_table:
            if trying_stm:
                return max(legal_actions, key=get_card_strength)
            
            # STM Blocking Lead
            if opponent_stm_threat:
                # If opponent is threatening STM, we should NOT lead a suit they are void in (if we know).
                # But we don't know voids easily.
                # Best bet: Lead a low heart if we have it? No, that gives them points.
                # Lead a suit where we have high cards to force them to play?
                # Actually, leading is hard to block STM with. Just play safe.
                pass

            # Piggy Hunting Logic
            if not piggy_pulled:
                # If we have S12, then S13 and S1 are NOT dangerous (we can't catch pig from others)
                # But S12 is VERY dangerous to lead.
                dangerous_ranks = [12] if has_sq else [12, 13, 1]
                
                dangerous_spades = [c for c in my_spades if c.rank in dangerous_ranks]
                safe_spades = [c for c in my_spades if c.rank not in dangerous_ranks]
                
                # Arching the Pig: If we have safe spades and no dangerous ones, lead high to force pig
                if safe_spades and not dangerous_spades:
                    return max(safe_spades, key=get_card_strength)

            # Standard Lead Logic
            safe_leads = [c for c in legal_actions if c.suit != Suit.SPADES and c.suit != Suit.HEARTS]
            if not safe_leads:
                safe_leads = [c for c in legal_actions if c.suit != Suit.SPADES]
            
            if not safe_leads:
                # Must lead Spades or Hearts
                # If we have S12, definitely avoid leading it.
                candidates = [c for c in legal_actions if not (c.suit == Suit.SPADES and c.rank == 12)]
                if not candidates:
                    candidates = legal_actions # Only have S12
                return min(candidates, key=get_card_strength)
                
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
                
                if trying_stm:
                    high_card = max(following_cards, key=get_card_strength)
                    h_val = 14 if high_card.rank == 1 else high_card.rank
                    if h_val > winning_rank:
                        return high_card
                    else:
                        return min(following_cards, key=get_card_strength)

                # STM Blocking (Following)
                if opponent_stm_threat:
                    # If there are points in the trick, we MUST try to win it if we can, 
                    # provided we don't give them the rest of the points.
                    # Actually, just taking ONE point stops STM.
                    points_in_trick = any(c.suit == Suit.HEARTS or (c.suit == Suit.SPADES and c.rank == 12) for c, _ in current_table)
                    
                    if points_in_trick:
                        # Try to win!
                        winning_cards = [c for c in following_cards if (14 if c.rank == 1 else c.rank) > winning_rank]
                        if winning_cards:
                            # Win with the lowest possible card that wins
                            return min(winning_cards, key=get_card_strength)
                
                # Normal Logic
                
                # 1. Feed the Pig (S12) Logic
                # If Spades led and K(13) or A(14) is played, dump S12 immediately
                if current_suit == Suit.SPADES and has_sq:
                    if winning_rank > 12:
                        return next(c for c in following_cards if c.rank == 12)
                    
                    # If we can't dump S12 safely, try to avoid playing it
                    safe_options = [c for c in following_cards if c.rank != 12]
                    if safe_options:
                        following_cards = safe_options

                # 1.5 Last Player Escape
                # If we are the last player and there are no points (Hearts or SQ) in the trick,
                # we can safely play a high card to escape it.
                is_last_player = (len(current_table) == 3)
                points_in_trick = any(c.suit == Suit.HEARTS or (c.suit == Suit.SPADES and c.rank == 12) for c, _ in current_table)
                
                if is_last_player and not points_in_trick:
                     return max(following_cards, key=get_card_strength)

                # 2. Dump High Cards if No Risk
                # If we can play under the winning rank, play the highest possible card
                safe_follows = [c for c in following_cards if (14 if c.rank == 1 else c.rank) < winning_rank]
                
                if safe_follows:
                    return max(safe_follows, key=get_card_strength)
                
                # 3. Must Win (or forced high)
                # If we must go over, play the highest card to dump it
                return max(following_cards, key=get_card_strength)
            
            else:
                # Void - Dump cards
                if trying_stm:
                    pass # Fall through to normal dump logic

                # STM Blocking (Void)
                if opponent_stm_threat:
                    # Dump a point card if possible to someone who is NOT the threat?
                    # Or just hold points?
                    # Actually, if we dump a heart to the threat, we help them!
                    # We should dump points to NON-threats.
                    # But we don't know who will win the trick yet (unless we are last).
                    pass

                # 1. Dump SQ (S12) - Always priority #1
                sq = next((c for c in legal_actions if c.suit == Suit.SPADES and c.rank == 12), None)
                if sq: return sq
                
                # 2. Dump Dangerous Spades (SA, SK) - Priority #2
                dangerous_spades = [c for c in legal_actions if c.suit == Suit.SPADES and c.rank in [1, 13]]
                if dangerous_spades: return max(dangerous_spades, key=get_card_strength)
                
                # 3. Dump High Hearts (Rank >= 10 or Ace) - Priority #3 (Points + High Rank)
                high_hearts = [c for c in legal_actions if c.suit == Suit.HEARTS and (c.rank >= 10 or c.rank == 1)]
                if high_hearts: return max(high_hearts, key=get_card_strength)
                
                # 4. Dump High Non-Point Cards (Rank >= J, Q, K, A) - Priority #4 (Avoid Lead)
                # Exclude Spades (handled) and Hearts (handled separately)
                high_others = [c for c in legal_actions if c.suit not in [Suit.SPADES, Suit.HEARTS] and (c.rank >= 11 or c.rank == 1)]
                if high_others: return max(high_others, key=get_card_strength)
                
                # 5. Dump Remaining Hearts - Priority #5 (Points)
                hearts = [c for c in legal_actions if c.suit == Suit.HEARTS]
                if hearts: return max(hearts, key=get_card_strength)
                
                # 6. Dump Max Remaining - Priority #6 (General High Cards)
                return max(legal_actions, key=get_card_strength)
