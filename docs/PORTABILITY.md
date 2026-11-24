# Portability & Environment Configuration

All scripts in `analysis/` read credentialed database locations from standard environment variables (defaulting to relative subdirectories if unset), containing **zero hardcoded personal paths**:

| Environment Variable | Default | Description |
|---|---|---|
| `TBI_ROOT` | `./data_workspace` | Workspace directory containing `derived/` cohort tables (see `docs/DATA_ACCESS.md`) |
| `MIMIC_RAW_DIR` | `./physionet/mimic` | Directory containing `mimiciv_v31/`, `mimic_iv_ecg/`, and `mimic_iv_note/` |
| `SICDB_DIR` | `./physionet/sicdb` | Directory containing extracted `SICdb v1.0.8` files (`laboratory.csv.gz`, `cases.csv.gz`) |

The standalone bedside calculator (`tools/score_tbi_six_test_states.py`) and automated verification suite (`tests/test_bedside_scorer_and_privacy.py`) require **only Python ≥ 3.9 standard library** and run immediately after `git clone` without credentialed datasets.
