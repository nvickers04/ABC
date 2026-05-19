"""Run status API: ``python -m infra.status_api``."""

from __future__ import annotations

import os

import uvicorn

from infra.status_api.app import app


def main() -> None:
    host = os.getenv("STATUS_API_HOST", "0.0.0.0")
    port = int(os.getenv("STATUS_API_PORT", "8790"))
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("STATUS_API_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
