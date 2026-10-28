"""Terminal input and display adapters for the player action protocol."""

import sys
from collections.abc import Callable

from .agents import AgentContext
from .types import Action, Observation, PassAction, Phase, PlayAction, PublicState


class UserQuit(Exception):
    pass


def render_public(state: PublicState) -> str:
    table = (
        "  ".join(f"P{play.player_id}: {play.card}" for play in state.current_trick) or "(empty)"
    )
    lines = [
        f"Hand {state.hand_number + 1} | pass={state.pass_direction.name} | phase={state.phase.value}",
        f"Match scores before this hand: {state.match_scores}",
        f"Raw hand points: {state.raw_scores} | hearts broken: {state.hearts_broken}",
        f"Tricks completed: {len(state.completed_tricks)}/13 | cards left: {state.hand_sizes}",
        f"Table: {table}",
    ]
    if state.completed_tricks:
        last = state.completed_tricks[-1]
        cards = "  ".join(f"P{play.player_id}:{play.card}" for play in last.plays)
        lines.append(f"Last trick: {cards} -> P{last.winner}, {last.points} points")
    return "\n".join(lines)


def parse_selection(text: str, count: int, available: int) -> tuple[int, ...]:
    """Convert one-based visible menu numbers to distinct zero-based indices."""
    if text.strip().lower() == "q":
        raise UserQuit
    try:
        indices = tuple(int(part) - 1 for part in text.replace(",", " ").split())
    except ValueError as error:
        raise ValueError("请输入选项编号，例如 1 或 1 3 5。") from error
    if len(indices) != count or len(set(indices)) != count:
        raise ValueError(f"请选择 {count} 个不同的编号。")
    if any(index < 0 or index >= available for index in indices):
        raise ValueError("编号超出范围。")
    return indices


class TerminalPlayer:
    def __init__(
        self,
        context: AgentContext,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        self.context = context
        self.read = read
        self.write = write

    def act(self, observation: Observation) -> Action:
        # Terminal clearing is a convenience, not a security boundary.
        if sys.stdout.isatty():
            self.write("\033[2J\033[H")
        self.read(f"请将终端交给 P{observation.player_id}，按 Enter 查看手牌；Ctrl+C 退出。")
        self.write(render_public(observation.public))
        self.write("你的手牌: " + " ".join(str(card) for card in observation.hand))
        if observation.passed_cards:
            self.write("你传出的牌: " + " ".join(str(card) for card in observation.passed_cards))
            self.write("你收到的牌: " + " ".join(str(card) for card in observation.received_cards))
        options = observation.legal_cards
        self.write("  ".join(f"{index + 1}:{card}" for index, card in enumerate(options)))
        count = 3 if observation.public.phase == Phase.PASSING else 1
        while True:
            try:
                text = self.read(f"选择 {count} 张牌的编号（q 退出）: ")
                indices = parse_selection(text, count, len(options))
                selected = tuple(options[index] for index in indices)
                return PassAction(selected) if count == 3 else PlayAction(selected[0])
            except ValueError as error:
                self.write(str(error))
