# Baseline and 48-Hour Dynamic Host-Response States After Traumatic Brain Injury

Analysis code accompanying the manuscript submitted to *Intensive Care Medicine*:

> **Baseline and 48-Hour Dynamic Host-Response States From Six Routine Blood Tests After
> Traumatic Brain Injury: Four-Database Validation, Transfusion Audit, and Multi-Organ Context**

This repository contains the **final analysis layer** that generates every table and figure
reported in the manuscript, plus the frozen classifier parameters required for zero-refit
scoring and the reference output tables (aggregate statistics only).

---

## What is here

```
analysis/            4 Python scripts (exact bytes that produced the published tables/figures)
model/               Frozen three-state Gaussian-mixture classifier parameters
reference_outputs/   Aggregate result tables (no patient-level rows) for verification
docs/                Data access, reproducibility and portability notes
```

| Script | Produces |
|---|---|
| `analysis/21_strict_48h_72h_icu_transitions_and_transfusion_v2.py` | Mutually exclusive ICU-landmark flow (D1), strict in-ICU landmark survival models (D2), US blood-product exposure and adjusted models (D3) |
| `analysis/22_ecg_confounding_and_4db_biomarkers_v2.py` | 12-lead ECG confounding/timing audit (B1), four-cohort biomarker denominators and Holm adjustment (B2), ECG clock/confounder coverage JSON |
| `analysis/23_exploratory_ct_lax_and_fixed_lmtp_v2.py` | Head-CT strict-timing and competing flow (S4A) and corrected exploratory summary (S4B); LAx / minimum-LAx 5-mmHg CPP bin 16-row table (S5A); corrected shifted-history sequential ICE estimator (S5B) |
| `analysis/24_render_icm_v2_3figures.py` | Main Figures 1–3 |

Every published number is reproduced **byte-for-byte** by these scripts (see
`docs/REPRODUCIBILITY.md`).

---

## Data are not in this repository

All four source databases are **credentialed, de-identified clinical databases** that cannot be
redistributed:

| Database | Access |
|---|---|
| MIMIC-IV v3.1, MIMIC-IV-ECG, MIMIC-IV-Note | PhysioNet — credentialed + DUA + CITI training |
| eICU-CRD v2.0 | PhysioNet — credentialed + DUA + CITI training |
| SICdb v1.0.8 | PhysioNet — credentialed + DUA |
| HiRID v1.1.1 | PhysioNet — credentialed + DUA |

No patient-level data, identifiers, or derived row-level tables are included. The files in
`reference_outputs/` are **aggregate summary statistics only** (counts, medians, adjusted odds
ratios, confidence intervals, P values) and contain **no subject-level rows**.

See **`docs/DATA_ACCESS.md`** for the access procedure and the exact input files the scripts expect.

---

## Quick start (for a credentialed user)

```bash
git clone https://github.com/outrotim/tbi-host-response-states-icm.git
cd tbi-host-response-states-icm
pip install -r requirements.txt      # Python 3.11+ recommended

# 1. Obtain the four credentialed databases and build the derived cohorts with your own
#    extraction pipeline (see docs/DATA_ACCESS.md for the required schemas).
# 2. Point the scripts at your local paths (see docs/PORTABILITY.md).
# 3. Run in order:
python analysis/21_strict_48h_72h_icu_transitions_and_transfusion_v2.py
python analysis/22_ecg_confounding_and_4db_biomarkers_v2.py
python analysis/23_exploratory_ct_lax_and_fixed_lmtp_v2.py
python analysis/24_render_icm_v2_3figures.py
```

Compare your outputs against `reference_outputs/` using the SHA-256 values listed in
`docs/REPRODUCIBILITY.md`.

---

## Key methodological features

1. **Zero-refit transport** — the three-state classifier is frozen in `model/` and applied to
   external cohorts without re-estimation.
2. **Strict ICU-resident landmarks** — admissions are partitioned at exact 48 h / 72 h into
   *died ≤ landmark*, *discharged alive ≤ landmark*, *still in ICU*; dynamic re-scoring requires
   all six tests freshly redrawn in `(24, 48]` or `(48, 72]` with zero carry-forward.
3. **Transfusion audit** — recorded 0–48 h / 0–72 h packed red cells, platelets, plasma/FFP and
   coagulation factors are extracted and used for adjusted and stratified models.
4. **Negation-aware radiology parsing** — clause-level rules with intra-sentence line-break
   unwrapping, extracranial bleeds excluded, and four built-in unit tests executed at runtime.
5. **Corrected longitudinal estimation** — the sequential-ICE estimator substitutes the shifted
   exposure history `d(A_s)` for `s < t`, and reports both unpenalised OLS and Ridge variants.

---

## Citation

If you use this code, please cite the manuscript. See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The reference output tables are released under CC BY 4.0.

## Contact

Corresponding author details are in the manuscript. For data-access questions, refer to
`docs/DATA_ACCESS.md`.
