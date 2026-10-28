"""Automatic inputs for tests only; these are not shipped as platform strategies."""

import random

from hearts.agents import AgentContext, AgentSpec
from hearts.cards import DECK
from hearts.deal import Deal
from hearts.types import Observation, PassAction, Phase, PlayAction


class RandomInputs:
    def __init__(self, context: AgentContext) -> None:
        self.rng = random.Random(context.seed)

    def act(self, observation: Observation):
        if observation.public.phase == Phase.PASSING:
            return PassAction(tuple(self.rng.sample(observation.legal_cards, 3)))
        return PlayAction(self.rng.choice(observation.legal_cards))


def random_inputs(name="test-input") -> AgentSpec:
    return AgentSpec(name, RandomInputs)


def deal_with_first_hand(cards) -> Deal:
    cards = tuple(cards)
    remaining = tuple(card for card in DECK if card not in cards)
    return Deal((cards, remaining[:13], remaining[13:26], remaining[26:39]))
