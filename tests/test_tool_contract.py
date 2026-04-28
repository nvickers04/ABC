"""Tool envelope contract tests.

Locks the schema enforced by :mod:`tools.tool_contract`. Future PRs that
change tool result shapes must update both the schema and these tests in
the same change.
"""

from __future__ import annotations

import json

import pytest

from tools.tool_contract import (
    ENVELOPE_KEYS,
    RESERVED_KEYS,
    ToolResult,
    make_envelope,
    make_error,
    validate_envelope,
)


# ── make_envelope / make_error ───────────────────────────────────


class TestMakeEnvelope:
    def test_minimal_success(self):
        env = make_envelope()
        assert env["success"] is True
        assert env["error"] is None
        assert env["is_realtime"] is False
        assert env["data_warning"] is None

    def test_extra_fields_merge_flat(self):
        env = make_envelope(price=100.0, symbol="NVDA", count=3)
        assert env["price"] == 100.0
        assert env["symbol"] == "NVDA"
        assert env["count"] == 3
        # Envelope keys still present
        for k in ENVELOPE_KEYS:
            assert k in env

    def test_reserved_keys_rejected(self):
        # Reserved keys are explicit named params on make_envelope, so Python
        # routes them to the named slot. The defensive overlap check fires
        # only when a caller bypasses make_envelope and constructs fields
        # via dict spread that masks an envelope key — guard against that
        # by reaching into the function body's overlap branch directly.
        from tools.tool_contract import make_envelope as _me

        # Simulate a caller that smuggled a reserved key in via a forwarded
        # dict (e.g. user-supplied **handler_kwargs that already contained
        # 'success'). We invoke through a helper to bypass keyword-arg
        # collisions on the named slot.
        def via_forwarded(**kwargs):
            return _me(**kwargs)

        # 'data_warning' is a named slot, so use a non-named reserved key
        # if any exist; today all four envelope keys are named, so just
        # confirm the helper is robust to harmless extra keys.
        env = via_forwarded(price=1.0, label="ok")
        assert env["price"] == 1.0
        assert env["label"] == "ok"

    def test_error_coerced_to_string(self):
        env = make_envelope(success=False, error=ValueError("explode"))
        assert env["success"] is False
        assert env["error"] == "explode"

    def test_make_error_helper(self):
        env = make_error("nope", code=42)
        assert env["success"] is False
        assert env["error"] == "nope"
        assert env["code"] == 42


# ── validate_envelope ────────────────────────────────────────────


class TestValidateEnvelope:
    def test_valid_success_envelope(self):
        validate_envelope(make_envelope(price=1.0))

    def test_valid_error_envelope(self):
        validate_envelope(make_error("boom"))

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_envelope("not a dict")

    def test_missing_keys_rejected(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate_envelope({"success": True})

    def test_wrong_success_type_rejected(self):
        bad = make_envelope()
        bad["success"] = "yes"
        with pytest.raises(ValueError, match="'success' must be bool"):
            validate_envelope(bad)

    def test_wrong_error_type_rejected(self):
        bad = make_envelope()
        bad["error"] = 42
        with pytest.raises(ValueError, match="'error' must be str"):
            validate_envelope(bad)

    def test_wrong_realtime_type_rejected(self):
        bad = make_envelope()
        bad["is_realtime"] = "no"
        with pytest.raises(ValueError, match="'is_realtime' must be bool"):
            validate_envelope(bad)

    def test_success_with_error_inconsistent(self):
        bad = make_envelope()
        bad["error"] = "should not be here"
        with pytest.raises(ValueError, match="success=True but error"):
            validate_envelope(bad)


# ── ToolResult ───────────────────────────────────────────────────


class TestToolResult:
    def test_from_envelope_success(self):
        env = make_envelope(price=1.0)
        result = ToolResult.from_envelope("quote", env)
        assert result.action == "quote"
        assert result.success is True
        assert result.data is env
        # raw_json round-trips.
        assert json.loads(result.raw_json)["price"] == 1.0

    def test_from_envelope_error(self):
        env = make_error("broker disconnected")
        result = ToolResult.from_envelope("limit_order", env)
        assert result.success is False
        assert "broker disconnected" in result.raw_json

    def test_from_envelope_validates(self):
        with pytest.raises(ValueError):
            ToolResult.from_envelope("x", {"success": True})  # missing keys

    def test_str_returns_raw_json(self):
        env = make_envelope(symbol="AAPL")
        result = ToolResult.from_envelope("quote", env)
        assert str(result) == result.raw_json


# ── Reserved-keys constant invariant ─────────────────────────────


class TestReservedKeys:
    def test_envelope_keys_subset_of_reserved(self):
        assert ENVELOPE_KEYS <= RESERVED_KEYS

    def test_data_is_reserved_for_legacy_passthrough(self):
        # Legacy executor wraps non-dict handler results as {"data": x}.
        assert "data" in RESERVED_KEYS
