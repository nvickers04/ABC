# Research Package

Configuration, simulation, and market-environment classification for the signal combination engine.

## Layout

- `config.py`: research universe, signal combination parameters, option schemas, and API budget settings
- `environment.py`: market-regime classification from shared candle data
- `simulator.py`: mechanical trade simulation and aggregate scoring
- `promoter.py`: options promotion repricing pipeline
- `replay.py`: deterministic replay for strategy validation

## Related Packages

- `signals/`: 50 base signal generators, combiner, templates, template evolution, scorer, and briefing
- Top-level `program.md`: human-edited research brief