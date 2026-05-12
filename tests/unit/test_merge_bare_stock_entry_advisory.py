"""merge_bare_stock_entry_advisory — soft hint for naked stock entries."""

from __future__ import annotations

from tools.tools_executor import merge_bare_stock_entry_advisory


def test_no_warning_when_plan_context_present():
    from memory import set_pending_order_context, reset_state

    reset_state()
    set_pending_order_context("AAPL", {"intent": "entry", "atr_pct": 1.0})
    result = {"success": True, "order_id": 1}
    out = merge_bare_stock_entry_advisory(
        "market_order",
        {"symbol": "AAPL", "side": "BUY", "quantity": 1},
        result,
    )
    assert out.get("data_warning") is None
    reset_state()


def test_warning_on_bare_market_buy():
    from memory import reset_state

    reset_state()
    result = {"success": True, "order_id": 1}
    out = merge_bare_stock_entry_advisory(
        "market_order",
        {"symbol": "MSFT", "side": "BUY", "quantity": 1},
        result,
    )
    assert out.get("data_warning")
    assert "plan_order" in out["data_warning"]


def test_no_warning_when_disabled(monkeypatch):
    from memory import reset_state

    reset_state()
    monkeypatch.setenv("DISABLE_BARE_STOCK_ENTRY_ADVISORY", "true")
    result = {"success": True}
    out = merge_bare_stock_entry_advisory(
        "market_order",
        {"symbol": "QQQ", "side": "BUY", "quantity": 1},
        result,
    )
    assert out.get("data_warning") is None
    monkeypatch.delenv("DISABLE_BARE_STOCK_ENTRY_ADVISORY", raising=False)


def test_no_warning_for_exit_intent():
    from memory import reset_state

    reset_state()
    result = {"success": True}
    out = merge_bare_stock_entry_advisory(
        "market_order",
        {"symbol": "QQQ", "side": "SELL", "quantity": 1, "intent": "exit"},
        result,
    )
    assert out.get("data_warning") is None


def test_no_warning_for_option_like_symbol():
    from memory import reset_state

    reset_state()
    result = {"success": True}
    occ = "AAPL260321C00150000"
    out = merge_bare_stock_entry_advisory(
        "market_order",
        {"symbol": occ, "side": "SELL", "quantity": 1},
        result,
    )
    assert out.get("data_warning") is None


def test_skips_on_error_dict():
    out = merge_bare_stock_entry_advisory(
        "market_order",
        {"symbol": "X", "side": "BUY", "quantity": 1},
        {"error": "broker not connected"},
    )
    assert out.get("data_warning") is None
