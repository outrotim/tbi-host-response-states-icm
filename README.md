# Baseline and 48-Hour Dynamic Host-Response States From Six Routine Blood Tests After Traumatic Brain Injury

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.1.0-teal.svg)](https://github.com/outrotim/tbi-host-response-states-icm/releases)

Public reproducible analysis suite and standalone zero-refit bedside classifier accompanying the four-cohort intensive care study ($N = 8,130$ across **MIMIC-IV v3.1**, **eICU-CRD v2.0** [130 US hospitals], **SICdb v1.0.8** [Salzburg, Austria], and **HiRID v1.1.1** [Bern, Switzerland]).

---

## 1. Instant Zero-Dependency Bedside / CSV Scorer (No Credentialed Data Needed)

`tools/score_tbi_six_test_states.py` implements the exact closed-form frozen MIMIC-IV three-state Gaussian mixture classifier (`model/locked_mimic_tbi_endotype_k3.json`) using only Python's standard library:

```bash
git clone https://github.com/outrotim/tbi-host-response-states-icm.git
cd tbi-host-response-states-icm

# 1. Run automated mathematical-parity & privacy unit tests (6/6 tests):
python3 -m unittest discover -s tests -v

# 2. Score a single acute TBI blood panel (conventional units: WBC x10^9/L, Plt x10^9/L, Hgb g/dL, INR, Bicarb mmol/L, Glu mg/dL):
python3 tools/score_tbi_six_test_states.py --wbc 8.5 --plt 144 --hgb 10.2 --inr 1.50 --bicarb 23.0 --glu 133

# 3. Score European SI units (Hgb g/L, Glu mmol/L) or left-censored INR <= 1.20 (SICdb):
python3 tools/score_tbi_six_test_states.py --wbc 8.9 --plt 149 --hgb 101 --inr 1.20 --bicarb 22.6 --glu 7.05 --european-units --sicdb-censored-inr

# 4. Batch-score a local CSV file (columns: wbc, plt, hgb, inr, bicarb, glu):
python3 tools/score_tbi_six_test_states.py --input-csv my_cohort.csv --output-csv my_cohort_scored.csv
```

---

## 2. Repository Structure (`v1.1.0`)

```text
model/               Frozen 45-parameter MIMIC-IV v3.1 RobustScaler + 3-state diagonal GMM (JSON)
tools/               Standalone zero-dependency CLI & batch CSV six-test state calculator
tests/               Automated unit tests for frozen model parity, SI units, censored INR & zero-PHI compliance
analysis/            Complete reproducible Python scripts for four-cohort validation, DCA, ICU scores & figures
reference_outputs/   Aggregate summary verification tables only (zero patient-level rows)
docs/                PhysioNet credentialed data access, environment portability & SHA-256 manifest
.zenodo.json         Zenodo archival metadata
CITATION.cff         Machine-readable citation metadata
```

| Script | Reproduces |
|---|---|
| `analysis/01b_strict_48h_72h_sicdb070_and_component_blood.py` | Mutually exclusive 48h/72h ICU-resident disposition flow (`D1`), zero-carry-forward transition survival models with unified `(0.70, 1.20]` censored-INR integration (`D2`), and binary + separate `PRBC`/platelet/plasma-factor blood-product models (`D3`) |
| `analysis/02_ecg_confounding_and_4db_biomarkers.py` | MIMIC-IV 12-lead ECG pre-ECG confounding/narrow-QRS/timing models (`B1`), four-cohort tested biomarker denominators (`B2`), and unified 112-test Holm multiplicity ledger (`B3`) |
| `analysis/03_exploratory_ct_lax_and_verified_ice.py` | Serial Head CT strict timing & competing-death flow (`S4A–S4B`), 2-min/5-min `LAx` autoregulation models (`S5A`), and simulation-verified observed-past sequential `ICE` (`S5B`) |
| `analysis/04_external_dca_and_severity_benchmarks.py` | Zero-refit `M0–M4` calibration/AUC parity, external `eICU-CRD` (`n=3,640`) 5%–50% Decision Curve Analysis (`Table S1F`), and 11-row four-cohort benchmarks over `Motor GCS`, `Total GCS (3–15)`, `APACHE IV`, `APACHE IVa`, and `SAPS 3` (`Table S1G`) |
| `analysis/05a_render_main_figures_1_to_3.py` & `05b_render_figS1_and_vector_qa.py` | `180.0-mm` 600-DPI vector Figures 1–3, Supplementary Figure S1, Visual Abstract, and automated PyMuPDF zero-overlap QA |

---

## 3. Data Access & Strict PhysioNet DUA Compliance

**No patient-level data, identifiers, or row-level clinical extracts are included in this repository.** All four source datasets are available to credentialed researchers via PhysioNet under their respective Data Use Agreements (`MIMIC-IV v3.1`, `MIMIC-IV-ECG v1.0`, `MIMIC-IV-Note v2.2`, `eICU-CRD v2.0`, `SICdb v1.0.8`, and `HiRID v1.1.1`; see [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md)).

---

## 4. License

Code is released under the [MIT License](LICENSE); aggregate summary tables in `reference_outputs/` are released under CC BY 4.0.
