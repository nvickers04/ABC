# Research Program

This file is the human-edited control surface for the autonomous research loop.

## Objective

Improve the quality of intraday trading signals produced by the 50-signal combination engine.
Primary goal: higher out-of-sample expectancy after slippage.
Secondary goals: stable signal count, interpretable logic, and strategies that can survive live execution frictions.

## Hard Constraints

- Preserve the existing signal schema expected by the simulator.
- Prefer strategies that produce enough signals to evaluate meaningfully.
- Avoid overfitting to a single symbol or one unusual day.

## Research Priorities

1. Favor simple signals whose edge is obvious from the candle data.
2. Improve out-of-sample performance, not just in-sample metrics.
3. Use the current market-environment snapshot when choosing between momentum, mean reversion, VWAP, and options structures.
4. Penalize strategies that need implausibly precise fills.
5. Prefer robust stock proxies over fragile options logic unless the environment clearly favors options structures.

## Signal Templates

- Templates define parameter boundaries for combining the 50 base signals.
- Template evolution adjusts boundaries via walk-forward optimization.
- Keep portfolio diversity across momentum, mean reversion, VWAP, and options-based ideas.
- If a signal category repeatedly fails in the current regime, pivot instead of fine-tuning the same idea.

## Evaluation Guidance

- Focus on expectancy, profit factor, drawdown behavior, and signal count together.
- Be suspicious of high expectancy with very few trades.
- Tight stops and tight targets often create simulator artifacts; use realistic distances.
- Time-of-day effects matter. Opening behavior and midday behavior should not be treated as identical.

## Live Trading Gap

When simulated returns and real trade outcomes diverge, prioritize strategies that are more execution-tolerant.
Large execution gaps should be treated as evidence that the strategy is too brittle, even if the backtest looks good.

## Human Notes

- Edit this file when you want to steer the research system without rewriting Python prompts.
- Keep instructions concrete and trading-specific.
- Avoid long essays. Short directives are easier for the agent to follow consistently.