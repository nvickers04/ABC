"""Unit tests for ``data.xai_usage`` token bucket extraction."""

from types import SimpleNamespace

from data.xai_usage import extract_billing_token_counts


def test_extract_prefers_text_minus_cached():
    u = SimpleNamespace(
        prompt_tokens=5000,
        prompt_text_tokens=1000,
        cached_prompt_text_tokens=200,
        prompt_image_tokens=10,
        completion_tokens=50,
        reasoning_tokens=30,
    )
    c = extract_billing_token_counts(u)
    assert c["noncached_prompt_text"] == 800
    assert c["cached_prompt_text"] == 200
    assert c["prompt_image"] == 10
    assert c["completion"] == 50
    assert c["reasoning"] == 30
    assert c["output_priced"] == 80


def test_extract_fallback_when_text_fields_zero():
    u = SimpleNamespace(
        prompt_tokens=900,
        prompt_text_tokens=0,
        cached_prompt_text_tokens=100,
        prompt_image_tokens=0,
        completion_tokens=5,
        reasoning_tokens=0,
    )
    c = extract_billing_token_counts(u)
    assert c["noncached_prompt_text"] == 800
    assert c["cached_prompt_text"] == 100
