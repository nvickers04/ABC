"""
Research module entry point.

Usage:
    python -m research              # Run strategy evolution loop
    python -m research --verbose    # Debug logging
    python -m research --dry-run    # Evaluate only, don't overwrite strategy.py
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
        backupCount=30,
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
    parser = argparse.ArgumentParser(description="Research Agent — Strategy Evolution")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate only, don't overwrite strategy.py")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Load environment
    from dotenv import load_dotenv
    load_dotenv(override=True)

    from research.agent import run_research
    asyncio.run(run_research(verbose=args.verbose, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
