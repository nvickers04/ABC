"""
Research module entry point.

Usage:
    python -m research                   # Run strategy evolution loop
    python -m research --verbose         # Debug logging
    python -m research --dry-run         # Evaluate only, don't overwrite strategy.py
    python -m research --no-promotion    # Skip options promotion repricing
    python -m research --strict-slippage # Use conservative slippage presets
    python -m research --replay-gating   # Require replay pass before promotion
    python -m research --slots 1,3,5     # Run only specific slots (comma-separated)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_logging(verbose: bool = False):
    """Configure logging for research module."""
    from logging.handlers import TimedRotatingFileHandler

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        log_dir / "research.log",
        when="midnight",
        interval=1,
        backupCount=7,  # reduced; value preserved in DB snapshots/hypotheses
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    # Silence noisy libraries
    for lib in ("httpx", "httpcore", "grpc", "xai_sdk", "urllib3",
                "charset_normalizer", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Research Agent — Strategy Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate only, don't overwrite strategy.py")
    # ── Phase 5: Feature flag CLI args ──────────────────────
    parser.add_argument("--no-promotion", action="store_true",
                        help="Disable options promotion repricing (faster, less accurate)")
    parser.add_argument("--strict-slippage", action="store_true",
                        help="Use conservative slippage presets (promotion-grade path)")
    parser.add_argument("--replay-gating", action="store_true",
                        help="Require deterministic replay pass before accepting a strategy")
    parser.add_argument("--slots", type=str, default="",
                        help="Comma-separated list of slot numbers to run, e.g. '1,3,5'. "
                             "Defaults to all slots.")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Load environment
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # Apply feature flags before importing agent to allow module-level reads
    from research import agent as _agent
    if args.no_promotion:
        _agent.FEATURE_FLAGS["options_promotion_enabled"] = False
    if args.strict_slippage:
        _agent.FEATURE_FLAGS["strict_slippage"] = True
    if args.replay_gating:
        _agent.FEATURE_FLAGS["replay_gating"] = True

    # Parse slot filter
    slot_filter: list[int] | None = None
    if args.slots:
        try:
            slot_filter = [int(s.strip()) for s in args.slots.split(",") if s.strip()]
        except ValueError:
            print(f"ERROR: --slots must be comma-separated integers, got: {args.slots!r}")
            sys.exit(1)

    from research.agent import run_research
    asyncio.run(
        run_research(
            verbose=args.verbose,
            dry_run=args.dry_run,
            slot_filter=slot_filter,
        )
    )


if __name__ == "__main__":
    main()
