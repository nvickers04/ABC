"""Run the profit API: ``python -m api``."""

from __future__ import annotations

import os

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("PROFIT_API_HOST", "127.0.0.1")
    port = int(os.getenv("PROFIT_API_PORT", "8787"))
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=os.getenv("PROFIT_API_RELOAD", "").lower() in ("1", "true", "yes"),
    )
