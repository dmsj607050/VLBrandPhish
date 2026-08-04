# Stability with limited reference logo images

| | |
| --- | --- |
| Section | Appendix -- reference library size study |
| Setting | K=1..N reference logos per brand, VLBP-Full |
| Metrics | Match Acc (mean +/- std over samplings) |
| Paper table(s) | Reference library size table |

## Status

This ablation is **not yet ported** to the `run.py` framework. The existing helper modules
(`prompt_modes.py`, `prompt_library_sweep.py`) in the `prompt_ablation` directory are
kept for reference.

To implement: use the shared `experiments/_shared/e2e.py` engine with brand subsetting
via `--restrict-brands-csv` and `--max-logo-per-brand`.
