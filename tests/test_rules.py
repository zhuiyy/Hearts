import unittest

from hearts.cards import DECK, TWO_OF_CLUBS, Card, Suit
from hearts.rules import finish_trick, legal_plays, score_hand
from hearts.types import Play


class CardTests(unittest.TestCase):
    def test_ids_round_trip_and_point_total(self):
        self.assertEqual(len(set(DECK)), 52)
        self.assertEqual([card.id for card in DECK], list(range(52)))
        self.assertEqual(sum(card.points for card in DECK), 26)
        for card in DECK:
            self.assertEqual(Card.from_id(card.id), card)
        self.assertEqual(str(Card(Suit.SPADES, 12)), "SQ")
        self.assertEqual(str(Card(Suit.HEARTS, 14)), "HA")

    def test_invalid_card_values(self):
        for rank in (1, 15, True, 2.0):
            with self.assertRaises(ValueError):
                Card(Suit.HEARTS, rank)
        for cid in (-1, 52, True, 0.0):
            with self.assertRaises(ValueError):
                Card.from_id(cid)


class RuleTests(unittest.TestCase):
    def test_opening_must_be_club_two(self):
        self.assertEqual(legal_plays(DECK, (), 0, False), (TWO_OF_CLUBS,))

    def test_must_follow_suit_even_on_first_trick(self):
        club = Card(Suit.CLUBS, 14)
        hand = [club, Card(Suit.SPADES, 12)]
        self.assertEqual(legal_plays(hand, [Play(0, TWO_OF_CLUBS)], 0, False), (club,))

    def test_no_points_on_first_trick_if_safe_card_exists(self):
        safe = Card(Suit.SPADES, 14)
        hand = [Card(Suit.HEARTS, 2), Card(Suit.SPADES, 12), safe]
        self.assertEqual(legal_plays(hand, [Play(0, TWO_OF_CLUBS)], 0, False), (safe,))

    def test_only_points_exception_prevents_first_trick_deadlock(self):
        hand = [Card(Suit.HEARTS, 2), Card(Suit.SPADES, 12)]
        self.assertEqual(set(legal_plays(hand, [Play(0, TWO_OF_CLUBS)], 0, False)), set(hand))

    def test_can_discard_hearts_before_hearts_are_broken(self):
        heart, spade = Card(Suit.HEARTS, 2), Card(Suit.SPADES, 3)
        legal = legal_plays([heart, spade], [Play(0, TWO_OF_CLUBS)], 1, False)
        self.assertEqual(set(legal), {heart, spade})

    def test_cannot_lead_hearts_until_broken_unless_only_hearts(self):
        heart, spade = Card(Suit.HEARTS, 2), Card(Suit.SPADES, 12)
        self.assertEqual(legal_plays([heart, spade], (), 1, False), (spade,))
        self.assertEqual(legal_plays([heart], (), 1, False), (heart,))
        self.assertEqual(set(legal_plays([heart, spade], (), 1, True)), {heart, spade})

    def test_trick_winner_uses_only_lead_suit_and_ace_is_high(self):
        plays = (
            Play(2, Card(Suit.CLUBS, 10)),
            Play(3, Card(Suit.HEARTS, 14)),
            Play(0, Card(Suit.CLUBS, 14)),
            Play(1, Card(Suit.SPADES, 12)),
        )
        result = finish_trick(plays)
        self.assertEqual((result.winner, result.points), (0, 14))

    def test_scoring_and_moon(self):
        normal = score_hand((1, 2, 3, 20))
        self.assertEqual(normal.raw_scores, normal.scores)
        self.assertIsNone(normal.moon_shooter)
        for shooter in range(4):
            raw = tuple(26 if seat == shooter else 0 for seat in range(4))
            result = score_hand(raw)
            self.assertEqual(result.moon_shooter, shooter)
            self.assertEqual(
                result.scores, tuple(0 if seat == shooter else 26 for seat in range(4))
            )

    def test_invalid_scores_rejected(self):
        for scores in ((0, 0, 0, 0), (-1, 1, 0, 26), (True, 0, 0, 25), (1, 25)):
            with self.assertRaises(ValueError):
                score_hand(scores)
