# Research Package

This package is the repo's trading-specific adaptation of the autoresearch pattern.

## Layout

- `__main__.py`: CLI entry point for `python -m research`
- `agent.py`: orchestration loop for evaluation, mutation, selection, persistence, and git versioning
- `config.py`: research universe, sandbox rules, prompt templates, and program-file loading
- `environment.py`: market-regime classification from shared candle data
- `simulator.py`: mechanical trade simulation and aggregate scoring
- `slots/`: mutable strategy artifacts, one file per slot

## Control Surfaces

- Top-level `program.md`: human-edited research brief
- `research/config.py`: fixed research parameters and prompt templates
- `research/slots/strategy_XX.py`: slot-local strategy code produced by the research loop

## Why This Is Not Karpathy's 3-File Layout

Karpathy's repo collapses the runtime into one mutable training file plus one instruction file.
This repo keeps the same core loop but splits responsibilities across:

- a trading simulator instead of a model-training loop
- multiple slot files instead of one evolving artifact
- market-environment analysis and selector logic
- SQLite persistence for evaluations and live-trade feedback

The behavior is the same in spirit: propose, evaluate, keep or discard, and iterate.
The code is just organized for a more complex domain.