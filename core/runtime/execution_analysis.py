"""Execution snapshot analysis — calibration, graduated params, LLM proposals.

Extracted from ``core.agent.TradingAgent`` so the agent module stays focused on
the ReAct loop. ``TradingAgent`` keeps thin methods that delegate here for
backward compatibility with characterization tests.
"""

from __future__ import annotations

import json
import logging
import statistics as _stats
from collections import defaultdict

from core.async_utils import safe_sleep as _safe_sleep
from core.json_parse import _parse_json_objects
from xai_sdk.chat import system as sdk_system, user as sdk_user

logger = logging.getLogger(__name__)


async def run_execution_analysis(agent) -> None:
    """Analyze execution snapshots: calibrate simulator slippage, propose graduated params.

    Runs when enough new snapshots have accumulated. Groups filled snapshots
    by (order_type, time_bucket, atr_bucket) and computes empirical medians.
    Then asks Grok to review the data and propose execution config changes.
    """
    try:
        from memory import (
            get_db,
            get_filled_snapshots,
            get_graduated_params,
            get_research_config,
            set_research_config,
            upsert_calibrated_slippage,
            insert_graduated_param,
            validate_param_key,
            deactivate_graduated_param,
            get_snapshots_for_param_review,
        )
        db = get_db()
        last_id = int(get_research_config("last_analysis_snapshot_id", 0.0))

        snapshots = get_filled_snapshots(since_id=last_id, limit=500)
        if len(snapshots) < 5:
            logger.info(f"Execution analysis: only {len(snapshots)} new snapshots, skipping")
            return

        groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        by_order_type: dict[str, list[float]] = defaultdict(list)

        for snap in snapshots:
            slip = snap.get("slippage_bps")
            if slip is None:
                continue
            ot = snap.get("order_type", "unknown") or "unknown"
            tb = snap.get("time_bucket", "all") or "all"
            ab = snap.get("atr_bucket", "all") or "all"
            groups[(ot, tb, ab)].append(abs(slip))
            by_order_type[ot].append(abs(slip))

        calibrated_count = 0
        for (ot, tb, ab), slips in groups.items():
            if len(slips) < 3:
                continue
            slips.sort()
            median = _stats.median(slips)
            p25 = slips[len(slips) // 4] if len(slips) >= 4 else slips[0]
            p75 = slips[3 * len(slips) // 4] if len(slips) >= 4 else slips[-1]
            upsert_calibrated_slippage(
                order_type=ot, time_bucket=tb, atr_bucket=ab,
                median_bps=round(median, 2), sample_count=len(slips),
                p25_bps=round(p25, 2), p75_bps=round(p75, 2),
            )
            calibrated_count += 1

        by_ot_tb: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (ot, tb, ab), slips in groups.items():
            by_ot_tb[(ot, tb)].extend(slips)
        for (ot, tb), slips in by_ot_tb.items():
            if len(slips) < 3:
                continue
            slips.sort()
            median = _stats.median(slips)
            upsert_calibrated_slippage(
                order_type=ot, time_bucket=tb, atr_bucket="all",
                median_bps=round(median, 2), sample_count=len(slips),
            )

        for ot, slips in by_order_type.items():
            if len(slips) < 3:
                continue
            slips.sort()
            median = _stats.median(slips)
            upsert_calibrated_slippage(
                order_type=ot, time_bucket="all", atr_bucket="all",
                median_bps=round(median, 2), sample_count=len(slips),
            )

        logger.info(
            f"Simulator calibration: {calibrated_count} bucket(s) updated from {len(snapshots)} snapshots"
        )

        active_params = get_graduated_params(active_only=True)
        for gp in active_params:
            review = get_snapshots_for_param_review(gp["id"], gp["ts"])
            after = review["after"]
            before = review["before"]
            if len(after) < 10:
                continue
            if not before:
                continue
            after_median = _stats.median(after)
            before_median = _stats.median(before)
            if after_median > before_median * 1.10:
                deactivate_graduated_param(
                    gp["id"],
                    f"Rolled back: after median {after_median:.1f}bps > before {before_median:.1f}bps "
                    f"(n_after={len(after)}, n_before={len(before)})",
                )
                logger.info(
                    f"Rolled back graduated param {gp['param_key']}: "
                    f"after={after_median:.1f}bps vs before={before_median:.1f}bps"
                )

        summary_lines = []
        for (ot, tb, ab), slips in sorted(groups.items(), key=lambda x: -len(x[1])):
            if len(slips) < 3:
                continue
            median = _stats.median(slips)
            mean = _stats.mean(slips)
            summary_lines.append(
                f"  {ot} | {tb} | {ab}: n={len(slips)}, "
                f"median={median:.1f}bps, mean={mean:.1f}bps, "
                f"range=[{min(slips):.1f}, {max(slips):.1f}]"
            )

        current_params = get_graduated_params(active_only=True)
        param_lines = [
            f"  {p['param_key']} = {p['param_value']} (since {p['ts'][:10]}, {p['improvement_bps']:.1f}bps improvement)"
            for p in current_params
        ] if current_params else ["  (none)"]

        if summary_lines:
            prompt = f"""Execution analysis review. {len(snapshots)} new order executions analyzed.

Slippage by (order_type | time_bucket | atr_bucket):
{chr(10).join(summary_lines[:20])}

Current graduated parameters:
{chr(10).join(param_lines)}

Based on this data, propose 0 to 2 execution config changes that would reduce slippage.
Only propose a change if the evidence is clear (n >= 10 for both comparison groups).
Changes must be specific and testable, e.g., "use adaptive orders instead of market orders during open for high-ATR stocks".

CRITICAL: param_key MUST use exactly this 4-part dot-separated format:
  {{order_type}}.{{intent}}.{{time_bucket}}.{{atr_bucket}}

The param_key describes the CONTEXT to match (the CURRENT order type and conditions),
NOT the new value. The new value goes in param_value.
Example: to switch from market to adaptive for entries at open with high ATR:
  param_key: "market.entry.open.high"  (matches current market orders in this context)
  param_value: "adaptive"  (the replacement order type)
  previous_value: "market"  (what it replaces)

Valid order_type: market, limit, stop_entry, bracket, trailing_stop, oca_exit, midprice, adaptive, vwap, twap, relative, snap_mid, moc, moo, loc, loo, all
Valid intent: entry, exit, stop, all
Valid time_bucket: open, morning, midday, close, extended, all
Valid atr_bucket: low, medium, high, all

Respond with ONLY a JSON array of objects:
  "param_key": "order_type.intent.time_bucket.atr_bucket",
  "param_value": "the new value",
  "previous_value": "what it replaces or null",
  "evidence_summary": "brief stats justification",
  "estimated_improvement_bps": number

If no changes warranted: []"""

            try:
                chat = agent.grok.client.chat.create(
                    model=agent.grok.model,
                    messages=[
                        sdk_system(
                            "You are an execution quality analyst. Review slippage data and propose "
                            "concrete parameter changes. Be conservative — only propose when evidence "
                            "is strong. Respond ONLY with a JSON array."
                        ),
                        sdk_user(prompt),
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                response = None
                for _api_attempt in range(3):
                    try:
                        response = await chat.sample()
                        break
                    except Exception:
                        if _api_attempt < 2:
                            await _safe_sleep(2 ** (_api_attempt + 1))
                        else:
                            raise

                usage = response.usage
                agent.cost_tracker.log_llm_usage(
                    agent.grok.model,
                    usage=usage,
                    purpose="execution_analysis",
                )

                raw = response.content or ""
                proposals = _parse_json_objects(raw)
                if not proposals:
                    try:
                        parsed = json.loads(raw.strip())
                        if isinstance(parsed, list):
                            proposals = parsed
                    except Exception:
                        pass
                if len(proposals) == 1 and isinstance(proposals[0], list):
                    proposals = proposals[0]

                for prop in proposals[:2]:
                    if not isinstance(prop, dict) or not prop.get("param_key"):
                        continue
                    key_err = validate_param_key(prop["param_key"])
                    if key_err:
                        logger.info(f"Proposal rejected (bad key): {key_err}")
                        continue
                    p_value = test_proposal(prop, groups)
                    if p_value is not None and p_value < 0.20:
                        insert_graduated_param(
                            param_key=prop["param_key"],
                            param_value=str(prop.get("param_value", "")),
                            previous_value=prop.get("previous_value"),
                            evidence_json=json.dumps(prop.get("evidence_summary", "")),
                            snapshots_analyzed=len(snapshots),
                            improvement_bps=float(prop.get("estimated_improvement_bps", 0)),
                            p_value=p_value,
                        )
                        logger.info(
                            f"Graduated param: {prop['param_key']} = {prop['param_value']} (p={p_value:.3f})"
                        )
                    else:
                        logger.info(f"Proposal rejected (p={p_value}): {prop.get('param_key')}")

            except Exception as llm_err:
                logger.warning(f"Execution analysis LLM call failed: {llm_err}")

        max_id = max(s["id"] for s in snapshots)
        set_research_config("last_analysis_snapshot_id", float(max_id), "execution_analysis_complete")
        logger.info(f"Execution analysis complete: analyzed {len(snapshots)} snapshots")

        try:
            from core.quality.quality_matrix import get_quality_matrix_service
            svc = get_quality_matrix_service()
            svc.update_from_execution_analysis(
                snapshots, calibrated_count, [], db
            )
        except Exception as _qm_err:
            logger.debug(f"QualityMatrix execution_analysis hook skipped: {_qm_err}")

    except Exception as e:
        logger.warning(f"Execution analysis failed: {e}", exc_info=True)


def test_proposal(proposal: dict, groups: dict) -> float | None:
    """Run Mann-Whitney U test on a proposal's implied comparison.

    Uses the structured param_key to identify the target bucket group
    and compares it against all other groups of the same order type.
    Returns p-value or None if insufficient data.
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        logger.info("scipy not available, cannot validate proposal statistically")
        return None

    key = proposal.get("param_key", "")
    parts = key.split(".")
    if len(parts) != 4:
        return None

    target_ot, target_intent, target_tb, target_ab = parts

    target_data = []
    other_data = []
    for (ot, tb, ab), slips in groups.items():
        if len(slips) < 3:
            continue
        ot_match = (target_ot == "all" or ot == target_ot)
        tb_match = (target_tb == "all" or tb == target_tb)
        ab_match = (target_ab == "all" or ab == target_ab)
        if ot_match and tb_match and ab_match:
            target_data.extend(slips)
        elif ot_match:
            other_data.extend(slips)

    if len(target_data) >= 5 and len(other_data) >= 5:
        try:
            _, p_value = mannwhitneyu(target_data, other_data, alternative='two-sided')
            return round(p_value, 4)
        except Exception:
            pass

    logger.debug(
        f"Proposal {key} rejected: insufficient data "
        f"(target={len(target_data)}, other={len(other_data)}, need 5 each)"
    )
    return None
