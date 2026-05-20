#!/usr/bin/env python3
"""Daily meta-optimization: Grok reviews live performance and proposes prompt/tool diffs.

Uses :data:`~core.prompt_config.PromptConfig.multi_agent_model` (MULTI_AGENT_MODEL /
``grok-4.20-multi-agent`` by default). **Review-only** — writes a unified diff and JSON
for manual inspection; never modifies ``prompt_config.py`` or ``tool_registry.py``.

Schedule (after market close, optional):

    EVOLVE_STRATEGY_ENABLED=1 python scripts/evolve_strategy.py --days 7

Examples::

    python scripts/evolve_strategy.py --days 7
    python scripts/evolve_strategy.py --dry-run
    python scripts/evolve_strategy.py -o data/evolve_strategy_latest.patch

Requires ``XAI_API_KEY`` or ``GROK_API_KEY``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta-optimize trader prompts/tools from cycle logs (review-only diff)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        metavar="N",
        help="Lookback for cycle logs and trade outcomes (default: 7)",
    )
    parser.add_argument(
        "--max-tools",
        type=int,
        default=30,
        help="Max tool schema lines included in corpus (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build performance digest and corpus only; skip LLM call",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="Patch output path (default: data/evolve_strategy_YYYYMMDD.patch)",
    )
    parser.add_argument(
        "--json-output",
        metavar="PATH",
        default=None,
        help="JSON bundle path (default: data/evolve_strategy_YYYYMMDD.json)",
    )
    parser.add_argument(
        "--require-enabled",
        action="store_true",
        help="Exit 2 unless EVOLVE_STRATEGY_ENABLED=1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    from core.strategy_evolution import (
        collect_performance_digest,
        evolution_enabled,
        format_review_report,
        run_strategy_evolution,
        snapshot_strategy_corpus,
    )

    if args.require_enabled and not evolution_enabled():
        print(
            "EVOLVE_STRATEGY_ENABLED is not set. "
            "Export EVOLVE_STRATEGY_ENABLED=1 or omit --require-enabled.",
            file=sys.stderr,
        )
        return 2

    days = max(1, int(args.days))
    out_dir = Path(args.output).parent if args.output else _REPO / "data"

    if args.dry_run:
        perf = collect_performance_digest(days=days)
        corpus = snapshot_strategy_corpus(max_tools=args.max_tools)
        print(f"Dry run: {perf.get('cycle_stats', {}).get('cycles', 0)} cycles in {days}d")
        print(f"  Profile: {corpus.active_profile}  tools in corpus: {len(corpus.tool_schemas)}")
        print("  Skipped LLM (use without --dry-run to generate diff).")
        return 0

    if not evolution_enabled():
        print(
            "Note: EVOLVE_STRATEGY_ENABLED is off — running anyway (review-only, no auto-apply).",
        )

    result = asyncio.run(
        run_strategy_evolution(
            days=days,
            max_tools=max(5, args.max_tools),
            output_dir=out_dir,
        )
    )

    if args.output:
        src = Path(result["patch_path"])
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        result["patch_path"] = str(dest)

    print(result["report"])
    print(f"\nPatch (review): {result['patch_path']}")
    print(f"JSON bundle:    {result['json_path']}")
    print("\nApply changes manually after review. Nothing was modified in the repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
