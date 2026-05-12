"""Normalize xAI SDK ``usage`` objects into per-bucket token counts for billing.

The Python SDK maps protobuf ``SamplingUsage`` fields onto ``response.usage``:

* ``prompt_tokens`` — total prompt tokens (telemetry / rollups).
* ``prompt_text_tokens`` / ``cached_prompt_text_tokens`` — text prompt split when
  the server reports caching (see xAI prompt-caching docs).
* ``prompt_image_tokens`` — image-side prompt tokens when applicable.
* ``completion_tokens`` — completion side (visible completion).
* ``reasoning_tokens`` — internal reasoning; xAI bills these at the **output** rate.

We treat **output-priced tokens** as ``completion_tokens + reasoning_tokens`` so
cost estimates align with published per-1M *output* pricing when both counters
are populated independently.
"""

from __future__ import annotations

from typing import Any


def extract_billing_token_counts(usage: Any) -> dict[str, int]:
    """Return non-negative integer counts for each billing bucket."""
    pt = int(getattr(usage, "prompt_tokens", None) or 0)
    ptext = int(getattr(usage, "prompt_text_tokens", None) or 0)
    pcached = int(getattr(usage, "cached_prompt_text_tokens", None) or 0)
    pimg = int(getattr(usage, "prompt_image_tokens", None) or 0)
    comp = int(getattr(usage, "completion_tokens", None) or 0)
    reas = int(getattr(usage, "reasoning_tokens", None) or 0)

    # Prefer explicit text split when present; otherwise approximate from totals.
    text_basis = ptext if ptext > 0 else pt
    noncached_text = max(0, text_basis - pcached)
    output_priced = max(0, comp + reas)

    return {
        "noncached_prompt_text": noncached_text,
        "cached_prompt_text": max(0, pcached),
        "prompt_image": max(0, pimg),
        "completion": max(0, comp),
        "reasoning": max(0, reas),
        "output_priced": output_priced,
        "prompt_total": max(0, pt),
    }
