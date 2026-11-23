#!/usr/bin/env python3
"""
ICM v3 — Pillar B (Supportive Simultaneous Multi-Organ & 12-Lead ECG Context):
1. MIMIC-IV 12-lead ECG valid-first selection (n=1,255), exact specimen-draw AND result-release
   (`storetime`) clock audit, systemic-only pre-ECG QT-drug extraction (excluding ophthalmic/otic/topical
   routes), exact n=1,255 ventricular rate regression, and non-nested pre-ECG confounding/rhythm stress-tests.
2. Four-database Fibrinogen (<200 and <150 mg/dL), aPTT, Lactate (>=2.0 mmol/L across all 4 cohorts),
   US NLR, Anion Gap, and CRP with exact tested denominators and a SINGLE UNIFIED 112-TEST HOLM LEDGER
   covering all 88 continuous/chi2 biomarker tests + 8 base ECG tests + 16 binary Fisher threshold tests
   (with 96-test Holm preserved side-by-side in V3_TableB3 for complete auditability).

Strictly read-only on all upstream datasets and all prior submission / review directories.
Writes exclusively to icm_submission_v3_20261123/tables/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

ROOT = Path("/Users/haoruijinlin/VItalDB/tbi_public_data_inventory_20261031")
EXT_MIMIC = Path("/Volumes/Extreme SSD/2T H2/mimic_downloads")
SIC_DIR = Path("/Volumes/Extreme SSD/SICdb数据库/extracted_core/salzburg-intensive-care-database-sicdb-a-freely-accessible-intensive-care-database-1.0.8")
K3_DIR = ROOT / "derived/closed_loop_v1/clinical_k3"
P1_DIR = ROOT / "derived/closed_loop_v1/revision_gatekeepers_v1/p1_endpoints_and_sicdb"
HIR_DIR = ROOT / "derived/closed_loop_v1/transatlantic_4db_v1/hirid"
CPP_PRIV = ROOT / "clinical_strengthening_20261119/cpp/private"

OUT_DIR = ROOT / "icm_submission_v3_20261123/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def holm_adjust(pvals: list) -> list:
    arr = np.asarray(pvals, dtype=float)
    valid = np.isfinite(arr)
    out = np.full(len(arr), np.nan)
    if not valid.any():
        return out.tolist()
    sub = arr[valid]
    m = len(sub)
    order = np.argsort(sub)
    adj = np.empty(m, dtype=float)
    rmax = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * sub[idx])
        rmax = max(rmax, val)
        adj[idx] = rmax
    out[valid] = adj
    return out.tolist()


def med_iqr(s: pd.Series, dec: int = 1) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return f"{q50:.{dec}f} ({q25:.{dec}f}–{q75:.{dec}f})"


def extract_6lab_draw_and_release_clocks_mimic(m_df: pd.DataFrame):
    MAP_M = {
        51301: "wbc", 51755: "wbc", 51265: "plt", 53189: "plt",
        51222: "hgb", 50811: "hgb", 51237: "inr", 51675: "inr",
        50882: "bicarb", 50803: "bicarb", 50931: "glu", 50809: "glu", 52569: "glu",
    }
    RANGES = {"wbc": (0.2, 120), "plt": (5, 1500), "hgb": (3, 24), "inr": (0.5, 15), "bicarb": (5, 50), "glu": (20, 1200)}
    cols_avail = pd.read_csv(CPP_PRIV / "mimic_labs.csv.gz", nrows=1).columns.tolist()
    use_c = [c for c in ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum", "labevent_id"] if c in cols_avail]
    ml = pd.read_csv(CPP_PRIV / "mimic_labs.csv.gz", usecols=use_c, low_memory=False)
    ml = ml.merge(m_df[["stay_id", "subject_id", "hadm_id", "intime"]], on="subject_id", suffixes=("", "_src"))
    ml = ml[ml["hadm_id"].isna() | (ml["hadm_id"] == ml["hadm_id_src"])].copy()
    ml["feat"] = ml["itemid"].map(MAP_M)
    ml = ml[ml["feat"].notna()].copy()
    in_dt = pd.to_datetime(ml["intime"])
    ml["h"] = (pd.to_datetime(ml["charttime"]) - in_dt).dt.total_seconds() / 3600.0
    if "storetime" in ml.columns:
        st_h = (pd.to_datetime(ml["storetime"], errors="coerce") - in_dt).dt.total_seconds() / 3600.0
        ml["h_avail"] = np.maximum(ml["h"], st_h.fillna(ml["h"]))
    else:
        ml["h_avail"] = ml["h"]
    ml["v"] = pd.to_numeric(ml["valuenum"], errors="coerce")
    ml = ml[ml["h"].between(-24.0, 6.0) & ml["v"].between(ml["feat"].map({k: v[0] for k, v in RANGES.items()}), ml["feat"].map({k: v[1] for k, v in RANGES.items()}))].copy()
    ml["dist_abs"] = ml["h"].abs()
    ml["post"] = ml["h"] > 0
    ml = ml.sort_values(["stay_id", "feat", "dist_abs", "post", "labevent_id"]).drop_duplicates(["stay_id", "feat"])
    return ml.groupby("stay_id")["h"].max().to_dict(), ml.groupby("stay_id")["h_avail"].max().to_dict()


def build_mimic_ecg_with_pre_ecg_confounders(m_df: pd.DataFrame):
    print("[1/3] Building MIMIC-IV 12-lead ECG cohort with pre-ECG comorbidities, electrolytes, systemic QT drugs, and rhythm QC...", flush=True)
    last_draw_h, last_avail_h = extract_6lab_draw_and_release_clocks_mimic(m_df)

    ecg = pd.read_csv(EXT_MIMIC / "mimic_iv_ecg/machine_measurements.csv", low_memory=False)
    rep_cols = [c for c in ecg.columns if c.startswith("report_")]
    ecg = ecg[ecg["subject_id"].isin(set(m_df["subject_id"]))].copy()

    merged = ecg.merge(
        m_df[["subject_id", "hadm_id", "stay_id", "admittime", "intime", "endotype_k3", "age", "male", "gcs_motor", "vent_0_24h", "afib", "chf"]],
        on="subject_id",
        how="inner",
    )
    merged["ecg_dt"] = pd.to_datetime(merged["ecg_time"], errors="coerce")
    merged["in_dt"] = pd.to_datetime(merged["intime"], errors="coerce")
    merged["adm_dt"] = pd.to_datetime(merged["admittime"], errors="coerce")
    merged["dt_h"] = (merged["ecg_dt"] - merged["in_dt"]).dt.total_seconds() / 3600.0
    cand = merged[merged["dt_h"].between(-24.0, 24.0)].copy()

    rr = pd.to_numeric(cand["rr_interval"], errors="coerce").where(lambda s: s.between(270, 2400))
    qt = (pd.to_numeric(cand["t_end"], errors="coerce") - pd.to_numeric(cand["qrs_onset"], errors="coerce")).where(lambda s: s.between(200, 700))
    qrs_dur = (pd.to_numeric(cand["qrs_end"], errors="coerce") - pd.to_numeric(cand["qrs_onset"], errors="coerce")).where(lambda s: s.between(30, 300))
    cand["ecg_hr_bpm"] = 60000.0 / rr
    cand["qtcf_ms"] = qt / np.cbrt(rr / 1000.0)
    cand["qtcb_ms"] = qt / np.sqrt(rr / 1000.0)
    cand["qrs_ms"] = qrs_dur

    qrs_ax = pd.to_numeric(cand["qrs_axis"], errors="coerce").where(lambda s: s.between(-180, 360))
    t_ax = pd.to_numeric(cand["t_axis"], errors="coerce").where(lambda s: s.between(-180, 360))
    diff_ax = (qrs_ax - t_ax).abs() % 360.0
    cand["qrs_t_angle"] = np.minimum(diff_ax, 360.0 - diff_ax)

    rep_txt = cand[rep_cols].fillna("").agg(" | ".join, axis=1).str.lower()
    cand["rhythm_af_pace_flag"] = rep_txt.str.contains(r"atrial fibrillation|atrial flutter|paced|pacemaker|ventricular rhythm").astype(int)
    cand["quality_artifact_flag"] = rep_txt.str.contains(r"poor data quality|artifact|lead reversal|misplaced|suspect").astype(int)

    cand["abs_dt"] = cand["dt_h"].abs()
    cand["post_icu"] = cand["dt_h"] > 0
    w24 = cand[cand["qtcf_ms"].notna()].sort_values(["stay_id", "abs_dt", "post_icu", "study_id"]).drop_duplicates("stay_id", keep="first").copy()
    w24["ph"] = w24["endotype_k3"].str[:2]
    w24["qtcf_prolonged"] = (((w24["male"] == 1) & (w24["qtcf_ms"] >= 450.0)) | ((w24["male"] == 0) & (w24["qtcf_ms"] >= 460.0))).astype(float)
    w24["wide_qrs_t_gt90"] = np.where(w24["qrs_t_angle"].notna(), (w24["qrs_t_angle"] > 90.0).astype(float), np.nan)
    w24["last_6lab_h"] = w24["stay_id"].map(last_draw_h)
    w24["last_6lab_avail_h"] = w24["stay_id"].map(last_avail_h)
    w24["ecg_after_final_6lab"] = (w24["dt_h"] > w24["last_6lab_h"]).astype(int)
    w24["ecg_after_6lab_avail"] = (w24["dt_h"] > w24["last_6lab_avail_h"]).astype(int)
    w24["ecg_after_hosp_admit"] = (w24["ecg_dt"] >= w24["adm_dt"]).astype(int)
    w24["narrow_clean_qrs"] = ((w24["qrs_ms"].between(50.0, 119.0)) & (w24["rhythm_af_pace_flag"] == 0) & (w24["quality_artifact_flag"] == 0)).astype(int)

    # 1. Pre-index hospital admission cardiac diagnoses strictly before index admittime
    adm = pd.read_csv(EXT_MIMIC / "mimiciv_v31/hosp/admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
    adm = adm[adm["subject_id"].isin(set(w24["subject_id"]))].copy()
    adm["disch_dt"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    dx = pd.read_csv(EXT_MIMIC / "mimiciv_v31/hosp/diagnoses_icd.csv.gz", usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    dx = dx[dx["subject_id"].isin(set(w24["subject_id"]))].merge(adm[["hadm_id", "disch_dt"]], on="hadm_id", how="left")
    code_s = dx["icd_code"].astype(str).str.upper()
    dx["is_af"] = code_s.str.startswith(("42731", "42732", "I48"))
    dx["is_hf"] = code_s.str.startswith(("428", "I50"))
    dx["is_cad"] = code_s.str.startswith(("410", "411", "412", "413", "414", "I20", "I21", "I22", "I24", "I25"))

    dx_w = dx.merge(w24[["stay_id", "subject_id", "hadm_id", "adm_dt"]], on="subject_id", suffixes=("_dx", "_idx"))
    prior_dx = dx_w[(dx_w["hadm_id_dx"] != dx_w["hadm_id_idx"]) & (dx_w["disch_dt"] <= dx_w["adm_dt"])]
    prior_adm = adm.merge(w24[["stay_id", "subject_id", "hadm_id", "adm_dt"]], on="subject_id", suffixes=("_a", "_idx"))
    prior_adm_set = set(prior_adm[(prior_adm["hadm_id_a"] != prior_adm["hadm_id_idx"]) & (prior_adm["disch_dt"] <= prior_adm["adm_dt"])]["stay_id"])

    w24["has_prior_adm"] = w24["stay_id"].isin(prior_adm_set).astype(float)
    w24["prior_af"] = w24["stay_id"].isin(set(prior_dx[prior_dx["is_af"]]["stay_id"])).astype(float)
    w24["prior_hf"] = w24["stay_id"].isin(set(prior_dx[prior_dx["is_hf"]]["stay_id"])).astype(float)
    w24["prior_cad"] = w24["stay_id"].isin(set(prior_dx[prior_dx["is_cad"]]["stay_id"])).astype(float)

    ever_dx = dx_w[(dx_w["hadm_id_dx"] == dx_w["hadm_id_idx"]) | (dx_w["disch_dt"] <= dx_w["adm_dt"])]
    w24["ever_af"] = w24["stay_id"].isin(set(ever_dx[ever_dx["is_af"]]["stay_id"])).astype(float)
    w24["ever_hf"] = w24["stay_id"].isin(set(ever_dx[ever_dx["is_hf"]]["stay_id"])).astype(float)
    w24["ever_cad"] = w24["stay_id"].isin(set(ever_dx[ever_dx["is_cad"]]["stay_id"])).astype(float)

    # 2. Pre-ECG [-24h, 0) electrolytes: K, Mg, Total Ca, Ionized Ca
    ELEC_IDS = {50971: "k", 50822: "k", 50960: "mg", 50893: "ca_tot", 50808: "ca_ion"}
    sids = set(w24["subject_id"])
    el_chunks = []
    for ch in pd.read_csv(
        EXT_MIMIC / "mimiciv_v31/hosp/labevents.csv.gz",
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        chunksize=1000000,
        low_memory=False,
    ):
        sub = ch[ch["subject_id"].isin(sids) & ch["itemid"].isin(ELEC_IDS.keys()) & ch["valuenum"].notna()].copy()
        if len(sub):
            el_chunks.append(sub)
    el_all = pd.concat(el_chunks, ignore_index=True).merge(w24[["stay_id", "subject_id", "ecg_dt"]], on="subject_id")
    el_all["dt_to_ecg_h"] = (pd.to_datetime(el_all["charttime"]) - el_all["ecg_dt"]).dt.total_seconds() / 3600.0
    el_pre = el_all[(el_all["dt_to_ecg_h"] >= -24.0) & (el_all["dt_to_ecg_h"] < 0.0)].copy()
    el_pre["etype"] = el_pre["itemid"].map(ELEC_IDS)
    el_pre = el_pre.sort_values(["stay_id", "etype", "dt_to_ecg_h"]).drop_duplicates(["stay_id", "etype"], keep="last")
    el_piv = el_pre.pivot(index="stay_id", columns="etype", values="valuenum")
    for col in ["k", "mg", "ca_tot", "ca_ion"]:
        w24[col] = w24["stay_id"].map(el_piv[col].to_dict()) if col in el_piv.columns else np.nan
    w24["k_low"] = (w24["k"] < 3.5).astype(float)
    w24["k_miss"] = w24["k"].isna().astype(float)
    w24["mg_low"] = (w24["mg"] < 1.7).astype(float)
    w24["mg_miss"] = w24["mg"].isna().astype(float)
    w24["ca_low"] = np.where(w24["ca_ion"].notna(), (w24["ca_ion"] < 1.12).astype(float), np.where(w24["ca_tot"].notna(), (w24["ca_tot"] < 8.5).astype(float), 0.0))
    w24["ca_miss"] = (w24["ca_ion"].isna() & w24["ca_tot"].isna()).astype(float)

    # 3. Pre-ECG [-48h, 0) SYSTEMIC administered QT-prolonging drugs (excluding ophthalmic/otic/topical routes!)
    QT_PAT = r"amiodarone|sotalol|dofetilide|procainamide|quinidine|ibutilide|ondansetron|droperidol|metoclopramide|haloperidol|quetiapine|ziprasidone|olanzapine|risperidone|citalopram|escitalopram|methadone|azithromycin|levofloxacin|ciprofloxacin|moxifloxacin|fluconazole|voriconazole"
    LOCAL_EXCL = r"ophth|otic|\beye\b|\bear\b|topical"
    hids = set(w24["hadm_id"])
    qt_stays = set()
    for ch in pd.read_csv(
        EXT_MIMIC / "mimiciv_v31/hosp/emar.csv.gz",
        usecols=["hadm_id", "charttime", "medication", "event_txt"],
        chunksize=800000,
        low_memory=False,
    ):
        med_s = ch["medication"].fillna("")
        sub = ch[ch["hadm_id"].isin(hids) & med_s.str.contains(QT_PAT, case=False) & ~med_s.str.contains(LOCAL_EXCL, case=False)].copy()
        if len(sub) == 0:
            continue
        ev = sub["event_txt"].fillna("").str.lower()
        sub = sub[ev.str.contains("administer|given|started|applied|confirmed|delayed") & ~ev.str.contains("not |hold|cancel|refus|pend")].copy()
        if len(sub):
            m_sub = sub.merge(w24[["stay_id", "hadm_id", "ecg_dt"]], on="hadm_id")
            dh = (pd.to_datetime(m_sub["charttime"]) - m_sub["ecg_dt"]).dt.total_seconds() / 3600.0
            qt_stays.update(m_sub[(dh >= -48.0) & (dh < 0.0)]["stay_id"].tolist())

    d_it = pd.read_csv(EXT_MIMIC / "mimiciv_v31/icu/d_items.csv.gz")
    lbl_it = d_it["label"].fillna("")
    qt_iids = set(d_it[lbl_it.str.contains(QT_PAT, case=False) & ~lbl_it.str.contains(LOCAL_EXCL, case=False)]["itemid"])
    ie = pd.read_csv(EXT_MIMIC / "mimiciv_v31/icu/inputevents.csv.gz", usecols=["stay_id", "itemid", "starttime", "amount", "statusdescription"], low_memory=False)
    ie = ie[ie["stay_id"].isin(set(w24["stay_id"])) & ie["itemid"].isin(qt_iids) & (pd.to_numeric(ie["amount"], errors="coerce") > 0)].copy()
    ie = ie[~ie["statusdescription"].fillna("").str.lower().isin(["rewritten", "cancelled", "canceled"])].merge(w24[["stay_id", "ecg_dt"]], on="stay_id")
    dh_ie = (pd.to_datetime(ie["starttime"]) - ie["ecg_dt"]).dt.total_seconds() / 3600.0
    qt_stays.update(ie[(dh_ie >= -48.0) & (dh_ie < 0.0)]["stay_id"].tolist())
    w24["pre_ecg_qt_drug"] = w24["stay_id"].isin(qt_stays).astype(float)

    return w24


def extract_sicdb_icuoffset_bridge_labs(s_df: pd.DataFrame) -> pd.DataFrame:
    lac_ids = {454, 465, 657}
    crp_ids = {341}
    fib_ids = {344, 345}
    aptt_ids = {597}
    target_ids = lac_ids | crp_ids | fib_ids | aptt_ids

    cids = set(s_df["CaseID"].astype(int))
    icu_off = s_df.set_index("CaseID")["ICUOffset"].fillna(0).to_dict()
    chunks = []
    for ch in pd.read_csv(SIC_DIR / "laboratory.csv.gz", usecols=["CaseID", "LaboratoryID", "Offset", "LaboratoryValue"], chunksize=600000):
        sub = ch[ch["CaseID"].isin(cids) & ch["LaboratoryID"].isin(target_ids) & ch["LaboratoryValue"].notna()].copy()
        if len(sub) == 0:
            continue
        sub["h_icu"] = (sub["Offset"] - sub["CaseID"].map(icu_off)) / 3600.0
        sub = sub[sub["h_icu"].between(-24.0, 72.0)].copy()
        if len(sub):
            chunks.append(sub)
    labs = pd.concat(chunks, ignore_index=True)
    labs["abs_h"] = labs["h_icu"].abs()
    labs["post"] = labs["h_icu"] > 0
    labs24 = labs[labs["h_icu"].between(-24.0, 24.0)].sort_values(["CaseID", "abs_h", "post"])

    out = s_df.copy()
    for col, id_set, (lo, hi), mult in [
        ("fibrinogen", fib_ids, (20.0, 1500.0), 1.0),
        ("aptt", aptt_ids, (10.0, 250.0), 1.0),
        ("lactate", lac_ids, (0.1, 35.0), 1.0),
        ("crp", crp_ids, (0.01, 60.0), 10.0),
    ]:
        s_sub = labs24[labs24["LaboratoryID"].isin(id_set) & labs24["LaboratoryValue"].between(lo, hi)].copy()
        first_s = (s_sub.groupby("CaseID")["LaboratoryValue"].first() * mult).rename(col)
        out = out.merge(first_s, on="CaseID", how="left")

    crp_late = labs[labs["LaboratoryID"].isin(crp_ids) & labs["h_icu"].between(24.0, 72.0) & labs["LaboratoryValue"].between(0.01, 60.0)]
    out = out.merge((crp_late.groupby("CaseID")["LaboratoryValue"].max() * 10.0).rename("crp_peak_24_72h"), on="CaseID", how="left")
    return out


def main():
    m_df = pd.read_csv(K3_DIR / "mimic_tbi_k3_scored.csv"); m_df["ph"] = m_df["endotype_k3"].str[:2]
    e_df = pd.read_csv(K3_DIR / "eicu_tbi_k3_scored.csv"); e_df["ph"] = e_df["endotype_k3"].str[:2]
    s_base = pd.read_csv(P1_DIR / "sicdb_icuoffset_censoredCDF_scored.csv"); s_base["ph"] = s_base["phenotype"].str[:2]
    s_df = extract_sicdb_icuoffset_bridge_labs(s_base)
    h_df = pd.read_csv(HIR_DIR / "hirid_true_neurotrauma_tbi_k3_scored.csv"); h_df["ph"] = h_df["endotype"].str[:2]

    w24 = build_mimic_ecg_with_pre_ecg_confounders(m_df)

    print("[2/3] Fitting pre-ECG confounding-adjusted & rhythm-filtered 12-lead ECG models + exact HR regression...", flush=True)
    w24["age_z"] = (w24["age"] - w24["age"].mean()) / w24["age"].std()
    w24["age_z2"] = w24["age_z"] ** 2
    w24["age_z3"] = w24["age_z"] ** 3
    w24["is_E2"] = (w24["ph"] == "E2").astype(float)
    w24["is_E3"] = (w24["ph"] == "E3").astype(float)
    for c in ["gcs_motor", "vent_0_24h"]:
        w24[c] = pd.to_numeric(w24[c], errors="coerce").fillna(w24[c].median())

    # Exact ventricular rate regression in n=1,255 valid-QTcF cohort
    X_hr = sm.add_constant(w24[["is_E2", "is_E3", "age", "male", "gcs_motor", "vent_0_24h"]].astype(float))
    fit_hr = sm.OLS(w24["ecg_hr_bpm"].astype(float), X_hr).fit(cov_type="HC1")
    ci_hr_e2 = fit_hr.conf_int().loc["is_E2"]

    pre_covs = ["age_z", "age_z2", "age_z3", "male", "has_prior_adm", "prior_af", "prior_hf", "prior_cad", "k_low", "k_miss", "mg_low", "mg_miss", "ca_low", "ca_miss", "pre_ecg_qt_drug"]
    ever_covs = ["age_z", "age_z2", "age_z3", "male", "ever_af", "ever_hf", "ever_cad", "k_low", "k_miss", "mg_low", "mg_miss", "ca_low", "ca_miss", "pre_ecg_qt_drug"]

    ecg_models = []
    for out_col, out_lbl in [("qtcf_prolonged", "Prolonged QTcF >=450/460 ms"), ("wide_qrs_t_gt90", "Frontal |QRS-T| > 90 deg")]:
        for spec_lbl, mask, covs in [
            ("Base clinical model (age, sex, motor GCS, day-1 ventilation)", w24[out_col].notna(), ["age", "male", "gcs_motor", "vent_0_24h"]),
            ("Pre-ECG model (cubic age, sex, prior-admission cardiac dx, pre-ECG K/Mg/Ca & systemic QT drugs)", w24[out_col].notna(), pre_covs),
            ("Pre-ECG model in narrow QRS (50–119 ms), no AF/flutter/pacing/artifact", w24[out_col].notna() & (w24["narrow_clean_qrs"] == 1), pre_covs),
            ("Pre-ECG model with index-admission cardiac diagnoses included", w24[out_col].notna(), ever_covs),
            ("Pre-ECG model in stays with complete K+Mg+Ca measured before ECG", w24[out_col].notna() & (w24["k_miss"] == 0) & (w24["mg_miss"] == 0) & (w24["ca_miss"] == 0), ["age_z", "age_z2", "age_z3", "male", "has_prior_adm", "prior_af", "prior_hf", "prior_cad", "k_low", "mg_low", "ca_low", "pre_ecg_qt_drug"]),
            ("Pre-ECG model in ECGs recorded at or after hospital admission", w24[out_col].notna() & (w24["ecg_after_hosp_admit"] == 1), pre_covs),
            ("Pre-ECG model in ECGs recorded strictly after final 6-lab specimen draw", w24[out_col].notna() & (w24["ecg_after_final_6lab"] == 1), pre_covs),
        ]:
            sub = w24[mask].copy()
            use_c = [c for c in covs if sub[c].nunique(dropna=True) >= 2 and sub[c].std() > 1e-6]
            X = sm.add_constant(sub[["is_E2", "is_E3"] + use_c].astype(float))
            res = sm.Logit(sub[out_col].astype(float), X).fit(disp=False, maxiter=200, cov_type="HC1")
            ci = np.exp(res.conf_int())
            wt_glob = res.wald_test("is_E2 = 0, is_E3 = 0", scalar=True)
            wt_32 = res.wald_test("is_E3 - is_E2 = 0", scalar=True)
            g1 = sub[sub["ph"] == "E1"]; g2 = sub[sub["ph"] == "E2"]; g3 = sub[sub["ph"] == "E3"]
            ecg_models.append({
                "endpoint": out_lbl,
                "specification": spec_lbl,
                "n_total": len(sub),
                "E1_events_N": f"{int(g1[out_col].sum())}/{len(g1)} ({100*g1[out_col].mean():.1f}%)",
                "E2_events_N": f"{int(g2[out_col].sum())}/{len(g2)} ({100*g2[out_col].mean():.1f}%)",
                "E3_events_N": f"{int(g3[out_col].sum())}/{len(g3)} ({100*g3[out_col].mean():.1f}%)",
                "E2_vs_E1_aOR_CI": f"{np.exp(res.params['is_E2']):.2f} ({ci.loc['is_E2',0]:.2f}–{ci.loc['is_E2',1]:.2f})",
                "p_E2_vs_E1": float(res.pvalues["is_E2"]),
                "E3_vs_E1_aOR_CI": f"{np.exp(res.params['is_E3']):.2f} ({ci.loc['is_E3',0]:.2f}–{ci.loc['is_E3',1]:.2f})",
                "p_E3_vs_E1": float(res.pvalues["is_E3"]),
                "p_E3_vs_E2": float(wt_32.pvalue),
                "p_global_2df": float(wt_glob.pvalue),
            })

    # Also compute exact strictly post-storetime (n=299) QTcF check for JSON summary
    sub_st = w24[w24["ecg_after_6lab_avail"] == 1].copy()
    use_st = [c for c in pre_covs if sub_st[c].nunique(dropna=True) >= 2 and sub_st[c].std() > 1e-6]
    res_st = sm.Logit(sub_st["qtcf_prolonged"].astype(float), sm.add_constant(sub_st[["is_E2", "is_E3"] + use_st].astype(float))).fit(disp=False, maxiter=200, cov_type="HC1")
    ci_st = np.exp(res_st.conf_int().loc["is_E3"])

    ecg_mod_df = pd.DataFrame(ecg_models)

    print("[3/3] Building 4-database biomarker table & SINGLE UNIFIED 112-TEST HOLM LEDGER...", flush=True)
    bio_rows = []
    specs = [
        ("MIMIC-IV", "[-24h, +6h]", m_df, "fibrinogen", "Fibrinogen (mg/dL)", 0, ("lt", 200.0, "<200 mg/dL")),
        ("MIMIC-IV", "[-24h, +6h]", m_df, "fibrinogen", "Fibrinogen <150 mg/dL sensitivity", 0, ("lt_only", 150.0, "<150 mg/dL")),
        ("MIMIC-IV", "[-24h, +6h]", m_df, "ptt", "Intrinsic aPTT (s)", 1, None),
        ("MIMIC-IV", "[-24h, +6h]", m_df, "lactate", "Blood lactate (mmol/L)", 2, ("ge", 2.0, ">=2.0 mmol/L")),
        ("MIMIC-IV", "[-24h, +6h]", m_df, "nlr", "Neutrophil-to-lymphocyte ratio (NLR)", 1, None),
        ("MIMIC-IV", "[-24h, +6h]", m_df, "anion_gap", "Serum anion gap (mEq/L)", 1, None),
        ("eICU-CRD", "[-24h, +6h]", e_df, "fibrinogen", "Fibrinogen (mg/dL)", 0, ("lt", 200.0, "<200 mg/dL")),
        ("eICU-CRD", "[-24h, +6h]", e_df, "fibrinogen", "Fibrinogen <150 mg/dL sensitivity", 0, ("lt_only", 150.0, "<150 mg/dL")),
        ("eICU-CRD", "[-24h, +6h]", e_df, "ptt", "Intrinsic aPTT (s)", 1, None),
        ("eICU-CRD", "[-24h, +6h]", e_df, "lactate", "Blood lactate (mmol/L)", 2, ("ge", 2.0, ">=2.0 mmol/L")),
        ("eICU-CRD", "[-24h, +6h]", e_df, "nlr", "Neutrophil-to-lymphocyte ratio (NLR)", 1, None),
        ("eICU-CRD", "[-24h, +6h]", e_df, "anion_gap", "Serum anion gap (mEq/L)", 1, None),
        ("SICdb", "[-24h, +24h] ICUOffset", s_df, "fibrinogen", "Fibrinogen (mg/dL)", 0, ("lt", 200.0, "<200 mg/dL")),
        ("SICdb", "[-24h, +24h] ICUOffset", s_df, "fibrinogen", "Fibrinogen <150 mg/dL sensitivity", 0, ("lt_only", 150.0, "<150 mg/dL")),
        ("SICdb", "[-24h, +24h] ICUOffset", s_df, "aptt", "Intrinsic aPTT (s)", 1, None),
        ("SICdb", "[-24h, +24h] ICUOffset", s_df, "lactate", "Blood lactate (mmol/L)", 2, ("ge", 2.0, ">=2.0 mmol/L")),
        ("SICdb", "[-24h, +24h] ICUOffset", s_df, "crp", "Admission C-reactive protein (mg/L)", 1, None),
        ("SICdb", "[+24h, +72h] ICUOffset", s_df, "crp_peak_24_72h", "Peak 24–72h C-reactive protein (mg/L)", 1, None),
        ("HiRID proxy", "[-12h, +24h]", h_df, "fibrinogen", "Fibrinogen (mg/dL)", 0, ("lt", 200.0, "<200 mg/dL")),
        ("HiRID proxy", "[-12h, +24h]", h_df, "fibrinogen", "Fibrinogen <150 mg/dL sensitivity", 0, ("lt_only", 150.0, "<150 mg/dL")),
        ("HiRID proxy", "[-12h, +24h]", h_df, "lactate", "Blood lactate (mmol/L)", 2, ("ge", 2.0, ">=2.0 mmol/L")),
        ("HiRID proxy", "[-12h, +24h]", h_df, "crp", "Early C-reactive protein (mg/L)", 1, None),
    ]
    for dbn, win_s, d, col, lbl, dec, thr in specs:
        sub = d.dropna(subset=[col]).copy()
        g1 = sub[sub["ph"] == "E1"][col].astype(float)
        g2 = sub[sub["ph"] == "E2"][col].astype(float)
        g3 = sub[sub["ph"] == "E3"][col].astype(float)

        if thr and thr[0] == "lt_only":
            cut = thr[1]
            c1, n1 = int((g1 < cut).sum()), len(g1)
            c2, n2 = int((g2 < cut).sum()), len(g2)
            c3, n3 = int((g3 < cut).sum()), len(g3)
            tab = np.array([[c1, n1 - c1], [c2, n2 - c2], [c3, n3 - c3]])
            p_glob = float(stats.chi2_contingency(tab)[1])
            p_21 = float(stats.fisher_exact([[c2, n2 - c2], [c1, n1 - c1]])[1])
            p_31 = float(stats.fisher_exact([[c3, n3 - c3], [c1, n1 - c1]])[1])
            p_32 = float(stats.fisher_exact([[c3, n3 - c3], [c2, n2 - c2]])[1])
            bio_rows.append({
                "database": dbn, "window": win_s, "biomarker": lbl,
                "measured_total_vs_cohort": f"{len(sub)}/{len(d)}",
                "E1_tested_summary": f"{c1}/{n1} ({100*c1/n1:.1f}%)",
                "E2_tested_summary": f"{c2}/{n2} ({100*c2/n2:.1f}%)",
                "E3_tested_summary": f"{c3}/{n3} ({100*c3/n3:.1f}%)",
                "threshold_counts_and_fisher": f"Fisher E3 vs E1 P={p_31:.2e}",
                "p_global": p_glob, "p_E2_vs_E1": p_21, "p_E3_vs_E1": p_31, "p_E3_vs_E2": p_32,
                "p_fisher_thr_E2_vs_E1": np.nan, "p_fisher_thr_E3_vs_E1": np.nan,
            })
        else:
            p_glob = float(stats.kruskal(g1, g2, g3).pvalue)
            p_21 = float(stats.mannwhitneyu(g2, g1).pvalue)
            p_31 = float(stats.mannwhitneyu(g3, g1).pvalue)
            p_32 = float(stats.mannwhitneyu(g3, g2).pvalue)
            thr_str = "—"
            pf21, pf31 = np.nan, np.nan
            if thr:
                op, cut, t_lbl = thr
                b1 = (g1 < cut) if op == "lt" else (g1 >= cut)
                b2 = (g2 < cut) if op == "lt" else (g2 >= cut)
                b3 = (g3 < cut) if op == "lt" else (g3 >= cut)
                c1, n1 = int(b1.sum()), len(g1)
                c2, n2 = int(b2.sum()), len(g2)
                c3, n3 = int(b3.sum()), len(g3)
                pf31 = float(stats.fisher_exact([[c3, n3 - c3], [c1, n1 - c1]])[1])
                pf21 = float(stats.fisher_exact([[c2, n2 - c2], [c1, n1 - c1]])[1])
                thr_str = f"{t_lbl}: E1 {c1}/{n1} ({100*c1/n1:.1f}%), E2 {c2}/{n2} ({100*c2/n2:.1f}%), E3 {c3}/{n3} ({100*c3/n3:.1f}%) [Fisher E2vE1 P={pf21:.2e}, E3vE1 P={pf31:.2e}]"
            bio_rows.append({
                "database": dbn, "window": win_s, "biomarker": lbl,
                "measured_total_vs_cohort": f"{len(sub)}/{len(d)}",
                "E1_tested_summary": f"{med_iqr(g1, dec)} [n={len(g1)}]",
                "E2_tested_summary": f"{med_iqr(g2, dec)} [n={len(g2)}]",
                "E3_tested_summary": f"{med_iqr(g3, dec)} [n={len(g3)}]",
                "threshold_counts_and_fisher": thr_str,
                "p_global": p_glob, "p_E2_vs_E1": p_21, "p_E3_vs_E1": p_31, "p_E3_vs_E2": p_32,
                "p_fisher_thr_E2_vs_E1": pf21, "p_fisher_thr_E3_vs_E1": pf31,
            })

    bio_df = pd.DataFrame(bio_rows)

    # Build SINGLE UNIFIED 112-TEST LEDGER (88 continuous/chi2 biomarker + 8 base ECG + 16 Fisher binary threshold = 112)
    ledger_items = []
    for r_i, row in bio_df.iterrows():
        for c_type, c_col in [("global", "p_global"), ("pairwise_E2_vs_E1", "p_E2_vs_E1"), ("pairwise_E3_vs_E1", "p_E3_vs_E1"), ("pairwise_E3_vs_E2", "p_E3_vs_E2")]:
            ledger_items.append({
                "source_table": "TableB2_Biomarkers", "source_row_idx": int(r_i),
                "database": row["database"], "variable": row["biomarker"],
                "test_category": c_type, "in_96_family": 1, "raw_p": float(row[c_col]),
            })

    ecg_base_indices = ecg_mod_df.index[ecg_mod_df["specification"].str.startswith("Base")].tolist()
    for e_i in ecg_base_indices:
        erow = ecg_mod_df.loc[e_i]
        for c_type, c_col in [("global_2df", "p_global_2df"), ("pairwise_E2_vs_E1", "p_E2_vs_E1"), ("pairwise_E3_vs_E1", "p_E3_vs_E1"), ("pairwise_E3_vs_E2", "p_E3_vs_E2")]:
            ledger_items.append({
                "source_table": "TableB1_ECG_Base", "source_row_idx": int(e_i),
                "database": "MIMIC-IV ECG", "variable": erow["endpoint"],
                "test_category": c_type, "in_96_family": 1, "raw_p": float(erow[c_col]),
            })

    for r_i, row in bio_df.iterrows():
        if np.isfinite(row["p_fisher_thr_E2_vs_E1"]):
            ledger_items.append({
                "source_table": "TableB2_BinaryThreshold", "source_row_idx": int(r_i),
                "database": row["database"], "variable": row["biomarker"] + " binary threshold",
                "test_category": "threshold_fisher_E2_vs_E1", "in_96_family": 0, "raw_p": float(row["p_fisher_thr_E2_vs_E1"]),
            })
            ledger_items.append({
                "source_table": "TableB2_BinaryThreshold", "source_row_idx": int(r_i),
                "database": row["database"], "variable": row["biomarker"] + " binary threshold",
                "test_category": "threshold_fisher_E3_vs_E1", "in_96_family": 0, "raw_p": float(row["p_fisher_thr_E3_vs_E1"]),
            })

    ledger_df = pd.DataFrame(ledger_items)
    assert len(ledger_df) == 112 and int(ledger_df["in_96_family"].sum()) == 96
    ledger_df["holm_112"] = holm_adjust(ledger_df["raw_p"].tolist())
    p96_input = [p if f96 == 1 else np.nan for p, f96 in zip(ledger_df["raw_p"], ledger_df["in_96_family"])]
    ledger_df["holm_96"] = holm_adjust(p96_input)
    ledger_df.to_csv(OUT_DIR / "V3_TableB3_Unified_112Test_Holm_Ledger.csv", index=False)

    # Map unified 112-test Holm P values back into TableB1 (base rows) and TableB2 (all rows + binary thresholds)
    for c_type, out_col in [
        ("global_2df", "holm_112_global_2df"),
        ("pairwise_E2_vs_E1", "holm_112_E2_vs_E1"),
        ("pairwise_E3_vs_E1", "holm_112_E3_vs_E1"),
        ("pairwise_E3_vs_E2", "holm_112_E3_vs_E2"),
    ]:
        ecg_mod_df[out_col] = np.nan
        sub_l = ledger_df[(ledger_df["source_table"] == "TableB1_ECG_Base") & (ledger_df["test_category"] == c_type)]
        for _, lr in sub_l.iterrows():
            ecg_mod_df.loc[int(lr["source_row_idx"]), out_col] = float(lr["holm_112"])
    ecg_mod_df.to_csv(OUT_DIR / "V3_TableB1_MIMIC_ECG_Confounding_and_Timing_Audit.csv", index=False)

    for c_type, out_col in [
        ("global", "holm_112_global"),
        ("pairwise_E2_vs_E1", "holm_112_E2_vs_E1"),
        ("pairwise_E3_vs_E1", "holm_112_E3_vs_E1"),
        ("pairwise_E3_vs_E2", "holm_112_E3_vs_E2"),
    ]:
        sub_l = ledger_df[(ledger_df["source_table"] == "TableB2_Biomarkers") & (ledger_df["test_category"] == c_type)].set_index("source_row_idx")
        bio_df[out_col] = bio_df.index.map(sub_l["holm_112"].to_dict())

    for c_type, out_col in [
        ("threshold_fisher_E2_vs_E1", "holm_112_fisher_thr_E2_vs_E1"),
        ("threshold_fisher_E3_vs_E1", "holm_112_fisher_thr_E3_vs_E1"),
    ]:
        sub_l = ledger_df[(ledger_df["source_table"] == "TableB2_BinaryThreshold") & (ledger_df["test_category"] == c_type)].set_index("source_row_idx")
        bio_df[out_col] = bio_df.index.map(sub_l["holm_112"].to_dict())

    bio_df.to_csv(OUT_DIR / "V3_TableB2_4DB_Biomarkers_FullDenominators_and_Holm.csv", index=False)

    ecg_clock_summary = {
        "n_valid_qtcf_stays": int(len(w24)),
        "median_age_by_state": {k: float(w24[w24["ph"] == k]["age"].median()) for k in ["E1", "E2", "E3"]},
        "ecg_median_hours_from_icu": round(float(w24["dt_h"].median()), 2),
        "ecg_before_icu_n": int((w24["dt_h"] < 0).sum()),
        "ecg_strictly_before_final_6lab_draw_n": int((w24["dt_h"] < w24["last_6lab_h"]).sum()),
        "ecg_simultaneous_with_final_6lab_draw_n": int((w24["dt_h"] == w24["last_6lab_h"]).sum()),
        "ecg_before_or_at_final_6lab_draw_n": int((w24["dt_h"] <= w24["last_6lab_h"]).sum()),
        "ecg_strictly_after_final_6lab_draw_n": int((w24["dt_h"] > w24["last_6lab_h"]).sum()),
        "ecg_strictly_after_all_6lab_storetime_n": int(w24["ecg_after_6lab_avail"].sum()),
        "qtcf_post_storetime_n299_E3_vs_E1_aOR_CI": f"{np.exp(res_st.params['is_E3']):.2f} ({ci_st[0]:.2f}–{ci_st[1]:.2f})",
        "qtcf_post_storetime_n299_p": float(res_st.pvalues["is_E3"]),
        "ecg_at_or_after_hosp_admit_n": int(w24["ecg_after_hosp_admit"].sum()),
        "narrow_clean_qrs_n": int(w24["narrow_clean_qrs"].sum()),
        "pre_ecg_systemic_qt_drug_n": int(w24["pre_ecg_qt_drug"].sum()),
        "prior_hosp_adm_n": int(w24["has_prior_adm"].sum()),
        "hr_bpm_E2_vs_E1_diff": round(float(fit_hr.params["is_E2"]), 2),
        "hr_bpm_E2_vs_E1_ci": [round(float(ci_hr_e2[0]), 2), round(float(ci_hr_e2[1]), 2)],
        "hr_bpm_E2_vs_E1_p": float(fit_hr.pvalues["is_E2"]),
        "qtcf_base_E3_vs_E1_holm_112": float(ecg_mod_df.loc[0, "holm_112_E3_vs_E1"]),
        "qtcf_base_E3_vs_E1_holm_96": float(ledger_df[(ledger_df["source_table"] == "TableB1_ECG_Base") & (ledger_df["source_row_idx"] == 0) & (ledger_df["test_category"] == "pairwise_E3_vs_E1")]["holm_96"].iloc[0]),
    }
    (OUT_DIR / "V3_ECG_Clock_and_Confounder_Coverage.json").write_text(json.dumps(ecg_clock_summary, indent=2))

    print("\n=== ECG CLOCK & HR SUMMARY ===")
    print(json.dumps(ecg_clock_summary, indent=2))
    print("\n=== ECG CONFOUNDING STRESS-TESTS (V3) ===")
    print(ecg_mod_df[["endpoint", "specification", "n_total", "E3_vs_E1_aOR_CI", "p_E3_vs_E1", "holm_112_E3_vs_E1"]].to_string(index=False))


if __name__ == "__main__":
    main()
