#!/usr/bin/env python3
"""Train QualityMatrix scoring weights from profit cycle logs (dashboard logger).

Replays trade-like cycles from ``logs/profit_cycles_*.json`` (and Postgres
``profit_cycle_logs`` when configured) for the last N days, ingests each via
:mod:`core.quality.quality_learning`, refits bounded weights, and persists loop
patches to the active profit profile
(merged into an evolved profile, or ``{base}_qm_learned`` for built-ins).

Requires ``QUALITY_MATRIX_LEARN_FROM_HISTORY=1`` (or pass ``--enable-learning``).

Examples::

    python scripts/train_quality_matrix.py
    python scripts/train_quality_matrix.py --days 30 --dry-run
    python scripts/train_quality_matrix.py --enable-learning --reload-config
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train QualityMatrix weights from profit cycle logs",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        metavar="N",
        help="Replay cycle logs from the last N calendar days (default: 30)",
    )
    parser.add_argument(
        "--enable-learning",
        action="store_true",
        help="Set QUALITY_MATRIX_LEARN_FROM_HISTORY=1 for this run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List trade-like cycles only; do not learn or save",
    )
    parser.add_argument(
        "--no-save-profile",
        action="store_true",
        help="Learn and refit but do not write evolved_profiles.json",
    )
    parser.add_argument(
        "--reload-config",
        action="store_true",
        help="Call get_profit_config().reload() after saving profile patches",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="Write JSON training summary to PATH",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    return parser.parse_args(argv)


def _ensure_learning_enabled(enable_flag: bool) -> None:
    import os

    from core.central_profit_config import get_profit_config
    from core.memory_config import reload_memory_config

    if enable_flag:
        os.environ["QUALITY_MATRIX_LEARN_FROM_HISTORY"] = "1"
    reload_memory_config()
    get_profit_config().reload(dotenv=False)
    from core.quality.quality_learning import learning_enabled

    if not learning_enabled():
        raise SystemExit(
            "QualityMatrix learning is disabled. Set QUALITY_MATRIX_LEARN_FROM_HISTORY=1 "
            "or pass --enable-learning."
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    _ensure_learning_enabled(args.enable_learning)

    from core.profit_config_state import get_active_profile_label
    from core.profit_cycle_logger import load_entries_since
    from core.quality.quality_learning import (
        LEARNABLE_WEIGHT_KEYS,
        load_state,
        persist_learned_weights_to_active_profile,
        trade_outcomes_from_cycle_logs,
        train_from_trade_outcomes,
    )
    since = datetime.now(timezone.utc) - timedelta(days=max(1, args.days))
    entries = load_entries_since(since)
    outcomes = trade_outcomes_from_cycle_logs(entries)
    profile = get_active_profile_label()

    summary: dict = {
        "since": since.isoformat(),
        "days": args.days,
        "cycle_entries": len(entries),
        "trade_like_outcomes": len(outcomes),
        "active_profile": profile,
        "dry_run": args.dry_run,
    }

    print(f"Cycle log window: last {args.days} days ({since.date().isoformat()} .. now)")
    print(f"  Cycle entries loaded: {len(entries)}")
    print(f"  Trade-like outcomes:  {len(outcomes)}")
    print(f"  Active profile:       {profile}")

    if args.dry_run:
        for row in outcomes[:10]:
            print(
                f"    {row.get('ts')} {row.get('symbol')} "
                f"pnl={row.get('pnl_usd'):+.2f} action={row.get('action')}"
            )
        if len(outcomes) > 10:
            print(f"    ... and {len(outcomes) - 10} more")
        if args.output:
            Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0

    if not outcomes:
        print("No trade-like cycle outcomes in window; nothing to train.")
        if args.output:
            Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 1

    train_summary = train_from_trade_outcomes(outcomes, refit_at_end=True)
    summary["train"] = train_summary

    state = load_state()
    summary["learned_weights"] = {
        k: state.learned_weights.get(k) for k in LEARNABLE_WEIGHT_KEYS if k in state.learned_weights
    }
    summary["base_weights"] = {
        k: state.base_weights.get(k) for k in LEARNABLE_WEIGHT_KEYS if k in state.base_weights
    }
    summary["last_reward"] = state.last_reward

    if not args.no_save_profile:
        save_summary = persist_learned_weights_to_active_profile(profile_label=profile)
        summary["profile_save"] = save_summary
        if save_summary.get("saved"):
            print(f"\nSaved loop patches to evolved profile: {save_summary.get('profile')}")
            print(f"  Path: {save_summary.get('path')}")
            promote = save_summary.get("promote_env")
            if promote and promote != profile:
                print(f"  To use on next trader start: set PROFIT_PROFILE={promote}")
        else:
            print(f"\nProfile save skipped: {save_summary}")

    if args.reload_config:
        from core.central_profit_config import get_profit_config

        get_profit_config().reload()
        print("Reloaded ProfitConfig from environment.")

    from core.quality.quality_matrix import get_quality_matrix_service

    get_quality_matrix_service().populate()
    print(
        f"\nTraining complete: ingested={train_summary.get('ingested', len(outcomes))} "
        f"reward={state.last_reward:.4f}"
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"Wrote summary: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
