"""Command-line entry points for play, replay, scheduling and analysis."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .agents import AgentSpec
from .arena import run_hand, run_match
from .deal import Deal
from .engine import HeartsHand
from .evaluation import Comparison, make_schedule
from .replay import load_record, record_hand, replay_hand, save_record
from .rules import RULESET_VERSION
from .seeds import SEED_VERSION, Seeds
from .statistics import paired_t_test
from .terminal import TerminalPlayer, UserQuit, render_public


def play(args: argparse.Namespace) -> None:
    seeds = Seeds(args.seed)
    specs = [AgentSpec("human", TerminalPlayer) for _ in range(4)]
    print("手动轮流控制四个座位。C/D/S/H = 梅花/方块/黑桃/红心。")

    def finish(hand: HeartsHand) -> None:
        print(f"本副结算: {hand.result.scores}")
        if args.record_dir:
            path = Path(args.record_dir) / f"hand_{hand.public.hand_number:04d}.json"
            stream = ("match", hand.public.hand_number) if args.match else ("manual", 0)
            deal_seed = (
                seeds.derive("match", "deal", hand.public.hand_number)
                if args.match
                else seeds.derive("manual", "deal", 0)
            )
            save_record(
                path,
                record_hand(
                    hand,
                    {
                        "root_seed": seeds.root,
                        "deal_seed": deal_seed,
                        "agents": ["human"] * 4,
                        "agent_seeds": [seeds.derive(*stream, "agent", seat) for seat in range(4)],
                    },
                ),
            )
            print(f"已保存回放: {path}")

    if args.match:
        result = run_match(specs, seeds, target=args.target, on_hand_end=finish)
        print(f"比赛结束: {result.scores}; 获胜座位: {result.winners}")
    else:
        hand = HeartsHand(Deal.shuffled(seeds.derive("manual", "deal", 0)))
        run_hand(hand, specs, seeds, stream=("manual", 0))
        finish(hand)


def show_replay(args: argparse.Namespace) -> None:
    hand = replay_hand(load_record(args.path))
    print(render_public(hand.public))
    for index, trick in enumerate(hand.public.completed_tricks, 1):
        cards = " ".join(f"P{play.player_id}:{play.card}" for play in trick.plays)
        print(f"{index:2d}. {cards} -> P{trick.winner}, {trick.points} points")
    print(f"结算: {hand.result.scores}; 射月玩家: {hand.result.moon_shooter}")


def schedule(args: argparse.Namespace) -> None:
    seeds = Seeds(args.seed)
    trials = make_schedule(seeds, args.blocks, args.split)
    record = {
        "schedule_version": 1,
        "ruleset": RULESET_VERSION,
        "seed_version": SEED_VERSION,
        "root_seed": args.seed,
        "split": args.split,
        "blocks": args.blocks,
        "trials": [asdict(trial) for trial in trials],
    }
    save_record(args.output, record)
    print(
        f"已保存 {args.blocks} 个独立牌局块，{len(trials)} 个配对试验，执行时共 {2 * len(trials)} 副。"
    )
    print("通过 hearts.evaluation.compare 传入策略工厂，执行配对比较。")


def analyze(args: argparse.Namespace) -> None:
    comparison = Comparison.from_dict(load_record(args.path))
    result = paired_t_test(comparison.block_differences, confidence=args.confidence)
    print(f"独立牌局块: {result.blocks}")
    print(f"平均罚分改善（正值更好）: {result.mean_improvement:.4f}")
    print(f"t={result.statistic:.4f}, 双侧 p={result.pvalue:.6g}")
    print(f"{args.confidence:.1%} 置信区间: {result.confidence_interval}")
    print("这是单次比较的未校正结果；多个预先指定的比较可用 holm_adjust 统一校正。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hearts research platform")
    commands = parser.add_subparsers(dest="command", required=True)

    play_parser = commands.add_parser("play", help="manual four-seat terminal game")
    play_parser.add_argument("--seed", type=int, default=0)
    play_parser.add_argument("--match", action="store_true", help="play a cumulative match")
    play_parser.add_argument("--target", type=int, default=100)
    play_parser.add_argument("--record-dir")
    play_parser.set_defaults(handler=play)

    replay_parser = commands.add_parser("replay", help="validate and display a completed recording")
    replay_parser.add_argument("path")
    replay_parser.set_defaults(handler=show_replay)

    schedule_parser = commands.add_parser("schedule", help="generate a paired Monte Carlo schedule")
    schedule_parser.add_argument("--seed", type=int, default=0)
    schedule_parser.add_argument("--blocks", type=int, default=100)
    schedule_parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    schedule_parser.add_argument("--output", required=True)
    schedule_parser.set_defaults(handler=schedule)

    analyze_parser = commands.add_parser(
        "analyze", help="paired block-level t-test on a comparison file"
    )
    analyze_parser.add_argument("path")
    analyze_parser.add_argument("--confidence", type=float, default=0.95)
    analyze_parser.set_defaults(handler=analyze)

    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (UserQuit, KeyboardInterrupt, EOFError):
        print("\n已退出；未完成的牌局不会导出全信息回放。")
        return 130
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"错误: {error}\n")
    return 0
