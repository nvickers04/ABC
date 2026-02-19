"""
Minimal Grok 4.2 Trader — Entry Point

Usage:
    python __main__.py                  # Run agent (paper trading)
    python __main__.py --test           # Test Grok connection
    python __main__.py --verbose        # Debug logging
    python __main__.py --account live   # Live trading (use with caution)

Built for Grok 4.2 — Alpha Arena winning style (pure autonomy, max WAIT, 0.5% risk)
"""

import asyncio
import argparse
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Configure logging with console + rotating file handler."""
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
        log_dir / "agent.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    # Suppress noisy IBKR logs
    logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
    logging.getLogger("ib_insync.ib").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def validate_startup():
    """Validate critical secrets before launch."""
    from dotenv import load_dotenv
    load_dotenv()

    errors = []

    # Check Grok API key
    grok_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not grok_key:
        errors.append("Missing GROK_API_KEY or XAI_API_KEY")

    # Check config files
    for path in ["config/trading.yaml"]:
        if not Path(path).exists():
            errors.append(f"Missing config: {path}")

    if errors:
        for err in errors:
            logger.error(f"Startup: {err}")
        logger.error("Fix the above errors and restart.")
        sys.exit(1)

    logger.info("Startup validation passed")


async def test_grok():
    """Test Grok LLM connection."""
    from core.grok_llm import get_grok_llm

    print("Testing Grok connection...")
    llm = get_grok_llm()

    response = await llm.client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "user", "content": "Say 'connected' if you can read this."}],
        max_tokens=50,
    )

    print(f"Response: {response.choices[0].message.content}")
    print("✓ Grok connected successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Minimal Grok 4.2 Trader")
    parser.add_argument("--test", action="store_true", help="Test Grok connection")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--account",
        choices=["paper", "live"],
        default="paper",
        help="Trading account type (default: paper)",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.test:
        asyncio.run(test_grok())
        return

    os.environ["IBKR_ACCOUNT_TYPE"] = args.account
    if args.account == "live":
        logger.warning("*** LIVE TRADING MODE ***")

    validate_startup()

    print(f"\nStarting Minimal Grok 4.2 Trader (account={args.account})...")
    print("Press Ctrl+C to stop\n")

    from core.agent import run_agent
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
