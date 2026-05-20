#!/usr/bin/env python3
"""One-click Docker deploy for ABC research + trader stacks.

Examples (from repo root):

  python infra/deploy.py --role research --env paper --with-postgres
  python infra/deploy.py --role trader --env paper
  python infra/deploy.py --role all --env dev --with-postgres --build
  python infra/deploy.py --role trader --env paper --registry ghcr.io/org/repo --tag latest --pull
  python infra/deploy.py --role trader --env paper --with-status

Dry-run prints compose commands without executing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.runtime.env_profile import VALID_ENVS, build_deploy_env  # noqa: E402

RUNTIME = REPO_ROOT / "infra" / "runtime"
EXAMPLES = RUNTIME / "examples"
POSTGRES_COMPOSE = REPO_ROOT / "infra" / "postgres" / "docker-compose.yml"
NETWORK_NAME = "postgres_default"


def _ensure_network(dry_run: bool) -> None:
    cmd = ["docker", "network", "create", NETWORK_NAME]
    if dry_run:
        print("[dry-run]", " ".join(cmd), "(ignored if exists)")
        return
    subprocess.run(cmd, check=False, capture_output=True)


def _compose_files(role: str, env_name: str, *, with_status: bool) -> list[Path]:
    env_name = env_name.lower()
    if role == "postgres":
        return [POSTGRES_COMPOSE]
    if role == "status":
        return [RUNTIME / "docker-compose.status.yml"]

    files: list[Path] = []
    if role == "research":
        files.append(RUNTIME / "docker-compose.research.yml")
        if env_name == "dev":
            files.append(EXAMPLES / "docker-compose.dev.yml")
        else:
            files.append(EXAMPLES / "docker-compose.prod.research.yml")
    elif role == "trader":
        files.append(RUNTIME / "docker-compose.trader.yml")
        if env_name == "dev":
            files.append(EXAMPLES / "docker-compose.dev.yml")
        else:
            files.append(EXAMPLES / "docker-compose.prod.trader.yml")
            overlay = EXAMPLES / f"docker-compose.env.{env_name}.yml"
            if overlay.is_file():
                files.append(overlay)
        if with_status:
            files.append(RUNTIME / "docker-compose.status.yml")
    return files


def _image_env(role: str, registry: str | None, tag: str, *, with_status: bool) -> dict[str, str]:
    if not registry:
        return {}
    reg = registry.rstrip("/")
    out: dict[str, str] = {}
    if role in ("research",):
        out["ABC_RESEARCH_IMAGE"] = f"{reg}/research:{tag}"
    if role in ("trader",):
        out["ABC_TRADER_IMAGE"] = f"{reg}/trader:{tag}"
    if role == "status" or with_status:
        out["ABC_STATUS_IMAGE"] = f"{reg}/status-api:{tag}"
    return out


def _run_compose(
    compose_files: list[Path],
    env_file: Path,
    *,
    build: bool,
    pull: bool,
    dry_run: bool,
    extra_env: dict[str, str],
) -> int:
    base = ["docker", "compose"]
    for f in compose_files:
        base.extend(["-f", str(f)])
    base.extend(["--env-file", str(env_file)])

    if pull:
        pull_cmd = base + ["pull", "--ignore-buildable"]
        if dry_run:
            print("[dry-run]", " ".join(pull_cmd))
        else:
            subprocess.run(pull_cmd, cwd=REPO_ROOT, env={**os.environ, **extra_env}, check=True)

    up_cmd = base + ["up", "-d"]
    if build:
        up_cmd.append("--build")
    if dry_run:
        print("[dry-run]", " ".join(up_cmd))
        for k, v in extra_env.items():
            print(f"  export {k}={v}")
        return 0
    return subprocess.run(up_cmd, cwd=REPO_ROOT, env={**os.environ, **extra_env}, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy ABC Docker stacks")
    parser.add_argument(
        "--role",
        choices=["postgres", "research", "trader", "status", "all"],
        required=True,
    )
    parser.add_argument("--env", choices=sorted(VALID_ENVS), default="paper")
    parser.add_argument("--with-postgres", action="store_true")
    parser.add_argument("--with-status", action="store_true")
    parser.add_argument("--registry", default=None, help="GHCR prefix, e.g. ghcr.io/org/repo")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.role == "all" and args.env != "dev":
        print("warning: --role all is for single-machine dev; use split roles in production.", file=sys.stderr)

    env_file, _ = build_deploy_env(REPO_ROOT, args.env)
    print(f"Using merged env: {env_file}")
    _ensure_network(args.dry_run)

    if args.with_postgres or args.role in ("postgres", "all"):
        pg_env = REPO_ROOT / "infra" / "postgres" / ".env"
        if not pg_env.is_file():
            print(f"Missing {pg_env} — copy infra/postgres/.env.example", file=sys.stderr)
            return 1
        rc = _run_compose([POSTGRES_COMPOSE], pg_env, build=False, pull=False, dry_run=args.dry_run, extra_env={})
        if rc != 0:
            return rc

    if args.role == "postgres":
        return 0

    stacks: list[str]
    if args.role == "all":
        stacks = ["research", "trader"]
    else:
        stacks = [args.role]

    for stack in stacks:
        files = _compose_files(stack, args.env, with_status=args.with_status and stack == "trader")
        ie = _image_env(stack, args.registry, args.tag, with_status=args.with_status and stack == "trader")
        rc = _run_compose(files, env_file, build=args.build, pull=args.pull, dry_run=args.dry_run, extra_env=ie)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
