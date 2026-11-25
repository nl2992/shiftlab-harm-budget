# A Pathwise Harm Budget for Reversible Model Promotion

A team that wants to replace a deployed model with a challenger must decide when to switch while feedback is delayed, purchased or missing. This work bounds the harm the switch actually causes (excess loss against never switching, plus switching fees). For a chosen budget R and level delta, a shield wrapped around any promotion rule keeps the realised harm below R at every round with probability at least 1 - delta, for every fixed loss sequence and every admissible delay rule. A matching lower bound covers the first promotion. The repository holds the TMLR manuscript, the shield and comparator implementations, the registered simulation, and the 14-stream and 100-stream real-data screens.

**Status.** Manuscript prepared for TMLR.

## Layout

- `paper/`: manuscript sources
- `papers/`: paper records and manuscript
- `src/shiftlab/`: implementation
- `scripts/`: experiment, analysis and audit scripts
- `tests/`: unit and regression tests
- `data/`: small tracked data and reports
- `configs/`: experiment configuration

The manuscript sources are in `paper/f3_tmlr`. The main implementation is `src/shiftlab/paper3/riskbudget`; scripts are under `scripts/paper3/riskbudget`. `papers/paper3_cost_aware_adaptation/` holds the Paper 3 programme record, including the design note for the risk budget (`2026_programme/03F_RISK_BUDGET_DESIGN.md`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Large raw data and run outputs are not tracked. Scripts regenerate them, and each protocol or results note under `docs/` states the seeds and commands it used.

## Related repositories

The Python package `shiftlab` started as one shared codebase and is now split by paper. Each repository carries the part of it that the paper needs.

## License

MIT. See `LICENSE`.
