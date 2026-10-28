import random
import unittest
from dataclasses import FrozenInstanceError, fields

from hearts.cards import TWO_OF_CLUBS, Card, Suit
from hearts.deal import Deal
from hearts.engine import HeartsHand, IllegalAction
from hearts.types import Observation, PassAction, PassDirection, Phase, PlayAction
from tests.helpers import deal_with_first_hand


class EngineTests(unittest.TestCase):
    def test_passing_cycle(self):
        for number in range(12):
            hand = HeartsHand(Deal.shuffled(1), hand_number=number)
            self.assertEqual(hand.public.pass_direction, PassDirection(number % 4))

    def test_passing_is_sealed_and_applied_in_each_direction(self):
        deal = Deal.shuffled(8)
        choices = [cards[:3] for cards in deal.hands]
        for direction in (PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS):
            hand = HeartsHand(deal, pass_direction=direction)
            for seat in range(4):
                hand.submit(seat, PassAction(choices[seat]))
                if seat < 3:
                    self.assertEqual(hand.public.hand_sizes, (13, 13, 13, 13))
                    for observer in range(4):
                        obs = hand.observe(observer)
                        self.assertEqual(obs.hand, deal.hands[observer])
                        self.assertEqual(obs.passed_cards, ())
                        self.assertEqual(obs.received_cards, ())
            for seat in range(4):
                source = (seat - direction.offset) % 4
                obs = hand.observe(seat)
                expected = set(deal.hands[seat]) - set(choices[seat]) | set(choices[source])
                self.assertEqual(set(obs.hand), expected)
                self.assertEqual(obs.passed_cards, choices[seat])
                self.assertEqual(obs.received_cards, choices[source])
            self.assertEqual(hand.public.phase, Phase.PLAYING)
            self.assertIn(TWO_OF_CLUBS, hand.observe(hand.public.current_player).hand)

    def test_invalid_actions_leave_state_unchanged(self):
        hand = HeartsHand(Deal.shuffled(7))
        own = hand.observe(0).hand
        other = hand.observe(1).hand
        actions = [
            (1, PassAction(other[:3])),
            (0, PassAction((own[0], own[0], own[1]))),
            (0, PassAction((own[0], own[1], other[0]))),
            (0, PassAction(own[:2])),
            (0, PassAction(list(own[:3]))),
            (0, PlayAction(own[0])),
        ]
        before = tuple(hand.observe(seat) for seat in range(4))
        for seat, action in actions:
            with self.assertRaises(IllegalAction):
                hand.submit(seat, action)
            self.assertEqual(tuple(hand.observe(seat) for seat in range(4)), before)
            self.assertEqual(hand.action_records, ())

        hand = HeartsHand(Deal.shuffled(7), pass_direction=PassDirection.KEEP)
        seat = hand.public.current_player
        before = hand.public
        illegal = next(card for card in hand.observe(seat).hand if card != TWO_OF_CLUBS)
        with self.assertRaises(IllegalAction):
            hand.submit(seat, PlayAction(illegal))
        self.assertEqual(hand.public, before)

    def test_observations_are_immutable_and_exclude_hidden_state(self):
        deal = Deal.shuffled(3)
        hand = HeartsHand(deal)
        obs = hand.observe(0)
        with self.assertRaises(FrozenInstanceError):
            obs.player_id = 1
        with self.assertRaises(FrozenInstanceError):
            obs.public.hearts_broken = True
        self.assertEqual(
            {field.name for field in fields(Observation)},
            {"player_id", "public", "hand", "legal_cards", "passed_cards", "received_cards"},
        )
        # Swapping two opponent hands cannot change P0's observation.
        swapped = Deal((deal.hands[0], deal.hands[2], deal.hands[1], deal.hands[3]))
        self.assertEqual(obs, HeartsHand(swapped).observe(0))
        hand.submit(0, PassAction(obs.hand[:3]))
        self.assertEqual(obs.public.current_player, 0)  # A snapshot, not a live reference.

    def test_queen_of_spades_does_not_break_hearts(self):
        hand = HeartsHand(Deal.shuffled(0), pass_direction=PassDirection.KEEP)
        rng = random.Random(15)
        while hand.result is None:
            seat = hand.public.current_player
            legal = hand.observe(seat).legal_cards
            queen = Card(Suit.SPADES, 12)
            options = [card for card in legal if card.suit != Suit.HEARTS]
            card = queen if queen in legal else rng.choice(options or legal)
            before = hand.public.hearts_broken
            hand.submit(seat, PlayAction(card))
            if card == queen:
                self.assertFalse(before, "fixture must play SQ before any hearts")
                self.assertFalse(hand.public.hearts_broken)
                return
        self.fail("SQ never appeared")

    def test_all_hearts_hand_can_finish(self):
        hearts = tuple(Card(Suit.HEARTS, rank) for rank in range(2, 15))
        hand = HeartsHand(deal_with_first_hand(hearts), pass_direction=PassDirection.KEEP)
        while hand.result is None:
            seat = hand.public.current_player
            hand.submit(seat, PlayAction(hand.observe(seat).legal_cards[0]))
        self.assertTrue(hand.public.hearts_broken)

    def test_many_complete_hands_conserve_cards_at_every_step(self):
        for seed in range(256):
            hand = HeartsHand(Deal.shuffled(seed), pass_direction=PassDirection(seed % 4))
            rng = random.Random(seed + 1000)
            while hand.result is None:
                seat = hand.public.current_player
                obs = hand.observe(seat)
                self.assertTrue(obs.legal_cards)
                if obs.public.phase == Phase.PASSING:
                    action = PassAction(tuple(rng.sample(obs.legal_cards, 3)))
                else:
                    action = PlayAction(rng.choice(obs.legal_cards))
                hand.submit(seat, action)
                self.assert_conservation(hand)
            self.assertEqual(len(hand.public.completed_tricks), 13)
            self.assertEqual(sum(hand.result.raw_scores), 26)
            self.assertEqual(
                sum(hand.result.scores), 78 if hand.result.moon_shooter is not None else 26
            )
            self.assertEqual(hand.public.hand_sizes, (0, 0, 0, 0))
            self.assertEqual(hand.public.phase, Phase.FINISHED)
            with self.assertRaises(IllegalAction):
                hand.submit(0, PlayAction(TWO_OF_CLUBS))

    def assert_conservation(self, hand):
        state = hand.public
        held = [card for seat in range(4) for card in hand.observe(seat).hand]
        played = [play.card for trick in state.completed_tricks for play in trick.plays]
        played.extend(play.card for play in state.current_trick)
        self.assertEqual(len(held) + len(played), 52)
        self.assertEqual(len(set(held + played)), 52)
        self.assertEqual(
            sum(state.raw_scores), sum(trick.points for trick in state.completed_tricks)
        )
