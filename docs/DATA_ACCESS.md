# Data access and required input schemas

None of the source data are redistributed here. All four databases are credentialed releases
from PhysioNet and are governed by their own Data Use Agreements.

## 1. Obtain credentialed access

| Database | PhysioNet project | Requirement |
|---|---|---|
| MIMIC-IV v3.1 | `physionet.org/content/mimiciv/3.1/` | Credentialed user + CITI "Data or Specimens Only Research" + signed DUA |
| MIMIC-IV-ECG | `physionet.org/content/mimic-iv-ecg/1.0/` | Same credentialed account + DUA |
| MIMIC-IV-Note | `physionet.org/content/mimic-iv-note/2.2/` | Same credentialed account + DUA |
| eICU-CRD v2.0 | `physionet.org/content/eicu-crd/2.0/` | Same credentialed account + DUA |
| SICdb v1.0.8 | `physionet.org/content/sicdb/1.0.8/` | Same credentialed account + DUA |
| HiRID v1.1.1 | `physionet.org/content/hirid/1.1.1/` | Same credentialed account + DUA (project-specific request) |

Credentialing: <https://physionet.org/settings/credentialing/>

## 2. Raw tables the scripts read directly

```
<MIMIC>/mimiciv_v31/hosp/admissions.csv.gz
<MIMIC>/mimiciv_v31/hosp/diagnoses_icd.csv.gz
<MIMIC>/mimiciv_v31/hosp/d_labitems.csv.gz
<MIMIC>/mimiciv_v31/hosp/labevents.csv.gz
<MIMIC>/mimiciv_v31/hosp/prescriptions.csv.gz
<MIMIC>/mimiciv_v31/icu/d_items.csv.gz
<MIMIC>/mimiciv_v31/icu/inputevents.csv.gz
<MIMIC>/mimiciv_v31/icu/chartevents.csv.gz
<MIMIC>/mimiciv_v31/icu/procedureevents.csv.gz
<MIMIC>/mimiciv_v31/icu/icustays.csv.gz
<MIMIC>/mimic_iv_ecg/machine_measurements.csv
<MIMIC>/mimic_iv_note/radiology.csv.gz
<SICdb>/laboratory.csv.gz
```

## 3. Derived cohort files (must be built by the user)

The scripts consume four cohort files that are produced by the upstream feature-extraction
pipeline. They are **not** included because they contain subject-level rows. Schema essentials:

### `mimic_tbi_k3_scored.csv`, `eicu_tbi_k3_scored.csv`
One row per admission with: `subject_id`/`patientunitstayid`, `hadm_id`, `stay_id`, `intime`,
`outtime`, `deathtime`, `hospital_expire_flag`, `in_hospital_death`, `age`, `male`, `gcs_motor`,
`vent_0_24h`, `vaso_0_24h`, `endotype_k3` (E1/E2/E3 assignment under the frozen classifier),
plus the six routine tests used for scoring.

### `sicdb_icuoffset_censoredCDF_scored.csv`
One row per SICdb case with `phenotype` (E1/E2/E3), `in_hospital_death`, `icu_death`,
`died_by_365d`, `obs_ge_365d`, `AdmissionYear`, `saps3`, `age`, `male`, and the six tests.

### `hirid_true_neurotrauma_tbi_k3_scored.csv`
One row per HiRID admission with `endotype_k3`, `in_hospital_death`, `age`, `male`, `gcs_motor`,
`apache2_raw`, `apache4_raw`.

### Private intermediate parquet/CSV stores
`mimic_labs.csv.gz`, `eicu_labs.csv.gz`, `*_pairs.parquet`, `*_windows.parquet`,
`*_measured_map_hours.parquet` — high-frequency ICP/MAP/laboratory extracts built per database.
Their schemas are documented in the docstrings of the consuming scripts.

## 4. Reporting requirement

The DUAs require that you not attempt re-identification, not share the data with third parties,
and acknowledge the databases in any publication. All analyses in this repository are aggregate
and retrospective.
