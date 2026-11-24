#!/usr/bin/env python3
"""
ICM v8 — Supplementary Priority B Quantitative Strengthening Script:
1. Reproduces exact frozen MIMIC-IV -> zero-refit eICU-CRD (n = 3,640) predicted mortality
   probabilities (M0 Clinical Base, M1 + Discrete 3-State Labels, M2 + GMM Posterior Probs,
   M3 + 6 Linear Labs, M4 + 12-df Cubic Regression Splines), verifying 100% parity with Table S1E.
2. Computes external Decision Curve Analysis (DCA; Vickers & Elkin, Med Decis Making 2006) across
   threshold probabilities 1%–60% (primary clinical decision window 5%–50%), reporting Net Benefit,
   Standardised Net Benefit, Net Interventions Avoided per 100 patients, and Incremental Net Benefit
   per 1,000 admissions (with 500-resample hospital-clustered bootstrap 95% CIs).
3. Computes head-to-head and multivariable-incremental prognostic benchmarks against validated
   neurological and physiological ICU severity scores across all four databases (N = 8,130):
   - MIMIC-IV v3.1 (USA): Motor GCS (n=3,483), Total GCS 3–15 (n=3,481),
                          and Age + Sex + Total GCS + Day-1 Ventilation + Cranial CT lesions (n=3,481)
   - eICU-CRD v2.0 (USA, 130 hospitals): Motor GCS (n=3,640), Total GCS 3–15 (n=3,640; plus zero-refit),
                          APACHE IV physiological score (n=3,297), and APACHE IVa predicted hospital mortality (n=3,273)
   - SICdb v1.0.8 (Austria): SAPS 3 score for hospital mortality (n=516) and verified 1-year mortality (n=457)
   - HiRID v1.1.1 proxy (Switzerland): Motor GCS (n=491) and Age + Sex + Motor GCS + Day-1 Ventilation (n=491)
4. Strictly read-only on all upstream datasets and all prior v1–v7 directories.
   Writes aggregate tables only (zero patient-level rows) to icm_submission_v8_20261124/tables/.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("TBI_ROOT", "./data_workspace")))
K3_DIR = ROOT / "derived/closed_loop_v1/clinical_k3"
P1_DIR = ROOT / "derived/closed_loop_v1/revision_gatekeepers_v1/p1_endpoints_and_sicdb"
HIR_DIR = ROOT / "derived/closed_loop_v1/transatlantic_4db_v1/hirid"

V7_TAB_DIR = ROOT / "icm_submission_v7_20261124/tables"
OUT_DIR = ROOT / "icm_submission_v8_20261124/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_lab_trans(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "t_wbc": np.log1p(np.clip(df["wbc"].values.astype(float), 0.0, 200.0)),
        "t_plt": df["plt"].values.astype(float),
        "t_hgb": df["hgb"].values.astype(float),
        "t_inr": np.log(np.clip(df["inr"].values.astype(float), 0.5, 15.0)),
        "t_bic": df["bicarb"].values.astype(float),
        "t_glu": np.log1p(np.clip(df["glu"].values.astype(float), 20.0, 1500.0)),
    })


def net_benefit(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    n = float(len(y))
    pred_pos = p >= pt
    tp = float(np.sum((y == 1) & pred_pos))
    fp = float(np.sum((y == 0) & pred_pos))
    w = pt / (1.0 - pt)
    return (tp / n) - (fp / n) * w


def net_benefit_treat_all(y: np.ndarray, pt: float) -> float:
    prev = float(np.mean(y))
    return prev - (1.0 - prev) * (pt / (1.0 - pt))


def calc_idi_pp(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> float:
    ev = y == 1
    nev = y == 0
    return 100.0 * float((np.mean(p_new[ev]) - np.mean(p_old[ev])) - (np.mean(p_new[nev]) - np.mean(p_old[nev])))


def fit_severity_comparison(
    df: pd.DataFrame,
    y_col: str,
    base_covs: list[str],
    cohort_label: str,
    endpoint_label: str,
    score_label: str,
    cluster_col: str | None = None,
    precomputed_prob_col: str | None = None,
    n_boot: int = 500,
    seed: int = 20261124,
) -> dict:
    cols_needed = (
        [y_col, "is_E2", "is_E3"]
        + base_covs
        + ([cluster_col] if cluster_col else [])
        + ([precomputed_prob_col] if precomputed_prob_col else [])
    )
    sub = df.dropna(subset=cols_needed).copy().reset_index(drop=True)
    y = sub[y_col].astype(int).to_numpy()
    n = len(sub)
    deaths = int(y.sum())

    if precomputed_prob_col is not None:
        p_raw = np.clip(sub[precomputed_prob_col].astype(float).to_numpy(), 1e-5, 1.0 - 1e-5)
        lp = np.log(p_raw / (1.0 - p_raw))
        X0 = sm.add_constant(lp[:, None], has_constant="add")
        X1 = sm.add_constant(np.column_stack([lp, sub["is_E2"].to_numpy(), sub["is_E3"].to_numpy()]), has_constant="add")
        auc_raw_score = float(roc_auc_score(y, p_raw))
    else:
        Xb = sub[base_covs].astype(float).to_numpy()
        X0 = sm.add_constant(Xb, has_constant="add")
        X1 = sm.add_constant(np.hstack([Xb, sub[["is_E2", "is_E3"]].astype(float).to_numpy()]), has_constant="add")
        auc_raw_score = np.nan

    if cluster_col is not None:
        groups = sub[cluster_col].to_numpy()
        f0 = sm.Logit(y, X0).fit(disp=False, cov_type="cluster", cov_kwds={"groups": groups})
        f1 = sm.Logit(y, X1).fit(disp=False, cov_type="cluster", cov_kwds={"groups": groups})
    else:
        f0 = sm.Logit(y, X0).fit(disp=False, cov_type="HC1")
        f1 = sm.Logit(y, X1).fit(disp=False, cov_type="HC1")

    p0 = f0.predict(X0)
    p1 = f1.predict(X1)
    auc0 = float(roc_auc_score(y, p0))
    auc1 = float(roc_auc_score(y, p1))
    d_auc = auc1 - auc0
    brier0 = float(brier_score_loss(y, p0))
    brier1 = float(brier_score_loss(y, p1))
    lrt_chi2 = max(0.0, float(2.0 * (f1.llf - f0.llf)))
    lrt_p = float(stats.chi2.sf(lrt_chi2, df=2))

    # 2-df robust/clustered Wald test for [is_E2, is_E3]
    R = np.zeros((2, X1.shape[1]))
    R[0, -2] = 1.0
    R[1, -1] = 1.0
    w_test = f1.wald_test(R, scalar=True)
    wald_p = float(w_test.pvalue)

    b_e2, se_e2 = float(f1.params[-2]), float(f1.bse[-2])
    b_e3, se_e3 = float(f1.params[-1]), float(f1.bse[-1])
    p_e2, p_e3 = float(f1.pvalues[-2]), float(f1.pvalues[-1])

    or_e2, lo_e2, hi_e2 = np.exp(b_e2), np.exp(b_e2 - 1.959964 * se_e2), np.exp(b_e2 + 1.959964 * se_e2)
    or_e3, lo_e3, hi_e3 = np.exp(b_e3), np.exp(b_e3 - 1.959964 * se_e3), np.exp(b_e3 + 1.959964 * se_e3)

    rng = np.random.default_rng(seed)
    b0_list, b1_list, bd_list = [], [], []
    if cluster_col is not None:
        uniq_cl = np.unique(sub[cluster_col].to_numpy())
        cl_to_idx = {c: np.where(sub[cluster_col].to_numpy() == c)[0] for c in uniq_cl}
        for _ in range(n_boot):
            samp_cl = rng.choice(uniq_cl, size=len(uniq_cl), replace=True)
            idx = np.concatenate([cl_to_idx[c] for c in samp_cl])
            if len(np.unique(y[idx])) < 2:
                continue
            a0_i = roc_auc_score(y[idx], p0[idx])
            a1_i = roc_auc_score(y[idx], p1[idx])
            b0_list.append(a0_i)
            b1_list.append(a1_i)
            bd_list.append(a1_i - a0_i)
    else:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) < 2:
                continue
            a0_i = roc_auc_score(y[idx], p0[idx])
            a1_i = roc_auc_score(y[idx], p1[idx])
            b0_list.append(a0_i)
            b1_list.append(a1_i)
            bd_list.append(a1_i - a0_i)

    a0_lo, a0_hi = np.percentile(b0_list, [2.5, 97.5])
    a1_lo, a1_hi = np.percentile(b1_list, [2.5, 97.5])
    d_lo, d_hi = np.percentile(bd_list, [2.5, 97.5])
    se_d = float(np.std(bd_list, ddof=1))
    p_dauc = float(2.0 * stats.norm.sf(abs(d_auc / se_d))) if se_d > 0 else np.nan

    idi_pp = calc_idi_pp(y, p0, p1)

    return {
        "database": cohort_label,
        "endpoint": endpoint_label,
        "conventional_comparator": score_label,
        "n": n,
        "deaths": deaths,
        "raw_proprietary_auc": round(auc_raw_score, 4) if not np.isnan(auc_raw_score) else np.nan,
        "comparator_auc": round(auc0, 4),
        "comparator_auc_lo": round(float(a0_lo), 4),
        "comparator_auc_hi": round(float(a0_hi), 4),
        "plus_states_auc": round(auc1, 4),
        "plus_states_auc_lo": round(float(a1_lo), 4),
        "plus_states_auc_hi": round(float(a1_hi), 4),
        "delta_auc": round(d_auc, 4),
        "delta_auc_lo": round(float(d_lo), 4),
        "delta_auc_hi": round(float(d_hi), 4),
        "p_delta_auc": p_dauc,
        "brier_comparator": round(brier0, 4),
        "brier_plus_states": round(brier1, 4),
        "lrt_chi2_2df": round(lrt_chi2, 2),
        "lrt_p_2df": lrt_p,
        "global_2df_wald_p": wald_p,
        "idi_pp": round(idi_pp, 2),
        "e2_vs_e1_aor": round(float(or_e2), 2),
        "e2_vs_e1_lo": round(float(lo_e2), 2),
        "e2_vs_e1_hi": round(float(hi_e2), 2),
        "e2_vs_e1_p": p_e2,
        "e3_vs_e1_aor": round(float(or_e3), 2),
        "e3_vs_e1_lo": round(float(lo_e3), 2),
        "e3_vs_e1_hi": round(float(hi_e3), 2),
        "e3_vs_e1_p": p_e3,
    }


def main():
    for csv_p in sorted(V7_TAB_DIR.glob("*")):
        if csv_p.is_file():
            shutil.copy2(csv_p, OUT_DIR / csv_p.name)

    m_cc = pd.read_csv(K3_DIR / "mimic_tbi_k3_scored.csv")
    e_cc = pd.read_csv(K3_DIR / "eicu_tbi_k3_scored.csv")
    s_cc = pd.read_csv(P1_DIR / "sicdb_icuoffset_censoredCDF_scored.csv")
    h_cc = pd.read_csv(HIR_DIR / "hirid_true_neurotrauma_tbi_k3_scored.csv")

    if "age" not in e_cc.columns:
        e_cc["age"] = e_cc["age_num"]
    m_cc["vent"] = m_cc["vent_0_24h"].astype(float)
    e_cc["vent"] = e_cc["vent_24h"].astype(float)
    h_cc["vent"] = h_cc["vent_24h"].astype(float)

    # =========================================================================
    # 1. REPRODUCE EXACT FROZEN M0-M4 MODELS (MIMIC-IV -> ZERO-REFIT eICU-CRD)
    # =========================================================================
    m_lt = make_lab_trans(m_cc)
    e_lt = make_lab_trans(e_cc)
    sp_m_cols, sp_e_cols = [], []
    for col in m_lt.columns:
        q50 = float(np.percentile(m_lt[col], 50))
        lo_b, hi_b = np.percentile(m_lt[col], [1, 99])
        vm = np.clip(m_lt[col].values, lo_b, hi_b)
        ve = np.clip(e_lt[col].values, lo_b, hi_b)
        dm_m = np.asarray(dmatrix(f"cr(x, knots=[{q50}], lower_bound={lo_b}, upper_bound={hi_b}) - 1", {"x": vm}))
        dm_e = np.asarray(dmatrix(f"cr(x, knots=[{q50}], lower_bound={lo_b}, upper_bound={hi_b}) - 1", {"x": ve}))
        sp_m_cols.append(dm_m)
        sp_e_cols.append(dm_e)
    X_sp_m = np.hstack(sp_m_cols)
    X_sp_e = np.hstack(sp_e_cols)

    base_cols = ["age", "male", "gcs_motor", "vent"]
    Xm_base = m_cc[base_cols].astype(float).values
    Xe_base = e_cc[base_cols].astype(float).values
    ym = m_cc["in_hospital_death"].values.astype(int)
    ye = e_cc["in_hospital_death"].values.astype(int)

    specs = [
        ("M0_Clinical_Base", Xm_base, Xe_base, 0.8571, 0.0834, 1.095),
        ("M1_Plus_Discrete_States", np.hstack([Xm_base, m_cc[["is_E2", "is_E3"]].values]), np.hstack([Xe_base, e_cc[["is_E2", "is_E3"]].values]), 0.8655, 0.0802, 1.045),
        ("M2_Plus_Posterior_Probs", np.hstack([Xm_base, m_cc[["prob_E2", "prob_E3"]].values]), np.hstack([Xe_base, e_cc[["prob_E2", "prob_E3"]].values]), 0.8685, 0.0795, 1.039),
        ("M3_Plus_6Lab_Linear", np.hstack([Xm_base, m_lt.values]), np.hstack([Xe_base, e_lt.values]), 0.8735, 0.0782, 1.040),
        ("M4_Plus_12df_Cubic_Splines", np.hstack([Xm_base, X_sp_m]), np.hstack([Xe_base, X_sp_e]), 0.8760, 0.0772, 1.023),
    ]

    preds_e = {}
    for mname, Xm, Xe, exp_auc, exp_brier, exp_slp in specs:
        fit = sm.Logit(ym, sm.add_constant(Xm, has_constant="add")).fit(disp=False)
        pe = np.clip(fit.predict(sm.add_constant(Xe, has_constant="add")), 1e-6, 1.0 - 1e-6)
        preds_e[mname] = pe
        auc = round(float(roc_auc_score(ye, pe)), 4)
        brier = round(float(brier_score_loss(ye, pe)), 4)
        lp = np.log(pe / (1.0 - pe))
        slp = round(float(sm.Logit(ye, sm.add_constant(lp)).fit(disp=False).params[1]), 3)
        assert auc == exp_auc and brier == exp_brier and slp == exp_slp, (
            f"Parity mismatch on {mname}: got ({auc}, {brier}, {slp}) vs expected ({exp_auc}, {exp_brier}, {exp_slp})"
        )
        print(f"[Verified Table S1E Parity] {mname:28s}: AUC={auc:.4f}, Brier={brier:.4f}, CalSlope={slp:.3f}")

    idi_m1_m0 = calc_idi_pp(ye, preds_e["M0_Clinical_Base"], preds_e["M1_Plus_Discrete_States"])
    idi_m4_m0 = calc_idi_pp(ye, preds_e["M0_Clinical_Base"], preds_e["M4_Plus_12df_Cubic_Splines"])
    print(f"[External IDI in eICU n=3,640] M1 vs M0 = {idi_m1_m0:+.2f} pp; M4 vs M0 = {idi_m4_m0:+.2f} pp")

    # =========================================================================
    # 2. EXTERNAL DECISION CURVE ANALYSIS (DCA) IN eICU-CRD (n = 3,640)
    # =========================================================================
    thresholds_dense = np.round(np.arange(0.01, 0.605, 0.005), 4)
    prev_e = float(np.mean(ye))

    rng = np.random.default_rng(20261124)
    hosp_ids = e_cc["hospitalid"].to_numpy()
    uniq_h = np.unique(hosp_ids)
    h_to_idx = {h: np.where(hosp_ids == h)[0] for h in uniq_h}
    n_boot = 500
    boot_indices = [
        np.concatenate([h_to_idx[h] for h in rng.choice(uniq_h, size=len(uniq_h), replace=True)])
        for _ in range(n_boot)
    ]

    curve_rows = []
    for pt in thresholds_dense:
        nb_all = net_benefit_treat_all(ye, pt)
        nb_m0 = net_benefit(ye, preds_e["M0_Clinical_Base"], pt)
        nb_m1 = net_benefit(ye, preds_e["M1_Plus_Discrete_States"], pt)
        nb_m2 = net_benefit(ye, preds_e["M2_Plus_Posterior_Probs"], pt)
        nb_m3 = net_benefit(ye, preds_e["M3_Plus_6Lab_Linear"], pt)
        nb_m4 = net_benefit(ye, preds_e["M4_Plus_12df_Cubic_Splines"], pt)

        d_m1_m0_1000 = 1000.0 * (nb_m1 - nb_m0)
        d_m4_m0_1000 = 1000.0 * (nb_m4 - nb_m0)
        d_m4_m1_1000 = 1000.0 * (nb_m4 - nb_m1)

        b_d10, b_d40, b_d41 = [], [], []
        for idx in boot_indices:
            yb = ye[idx]
            nb0_b = net_benefit(yb, preds_e["M0_Clinical_Base"][idx], pt)
            nb1_b = net_benefit(yb, preds_e["M1_Plus_Discrete_States"][idx], pt)
            nb4_b = net_benefit(yb, preds_e["M4_Plus_12df_Cubic_Splines"][idx], pt)
            b_d10.append(1000.0 * (nb1_b - nb0_b))
            b_d40.append(1000.0 * (nb4_b - nb0_b))
            b_d41.append(1000.0 * (nb4_b - nb1_b))

        d10_lo, d10_hi = np.percentile(b_d10, [2.5, 97.5])
        d40_lo, d40_hi = np.percentile(b_d40, [2.5, 97.5])
        d41_lo, d41_hi = np.percentile(b_d41, [2.5, 97.5])

        nia_m0 = 100.0 * (nb_m0 - nb_all) * ((1.0 - pt) / pt)
        nia_m1 = 100.0 * (nb_m1 - nb_all) * ((1.0 - pt) / pt)
        nia_m4 = 100.0 * (nb_m4 - nb_all) * ((1.0 - pt) / pt)

        curve_rows.append({
            "threshold_prob": float(pt),
            "threshold_pct": round(100.0 * float(pt), 1),
            "nb_treat_none": 0.0,
            "nb_treat_all": round(nb_all, 6),
            "nb_M0_clinical_base": round(nb_m0, 6),
            "nb_M1_discrete_states": round(nb_m1, 6),
            "nb_M2_posterior_probs": round(nb_m2, 6),
            "nb_M3_linear_6labs": round(nb_m3, 6),
            "nb_M4_12df_splines": round(nb_m4, 6),
            "snb_M0_clinical_base": round(nb_m0 / prev_e, 5),
            "snb_M1_discrete_states": round(nb_m1 / prev_e, 5),
            "snb_M4_12df_splines": round(nb_m4 / prev_e, 5),
            "delta_nb_M1_vs_M0_per_1000": round(d_m1_m0_1000, 2),
            "delta_nb_M1_vs_M0_lo": round(float(d10_lo), 2),
            "delta_nb_M1_vs_M0_hi": round(float(d10_hi), 2),
            "delta_nb_M4_vs_M0_per_1000": round(d_m4_m0_1000, 2),
            "delta_nb_M4_vs_M0_lo": round(float(d40_lo), 2),
            "delta_nb_M4_vs_M0_hi": round(float(d40_hi), 2),
            "delta_nb_M4_vs_M1_per_1000": round(d_m4_m1_1000, 2),
            "delta_nb_M4_vs_M1_lo": round(float(d41_lo), 2),
            "delta_nb_M4_vs_M1_hi": round(float(d41_hi), 2),
            "net_interventions_avoided_per_100_M0": round(nia_m0, 2),
            "net_interventions_avoided_per_100_M1": round(nia_m1, 2),
            "net_interventions_avoided_per_100_M4": round(nia_m4, 2),
        })

    df_curve = pd.DataFrame(curve_rows)
    df_curve.to_csv(OUT_DIR / "V8_DCA_Curve_eICU_n3640.csv", index=False)

    key_pts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    df_key_dca = df_curve[df_curve["threshold_prob"].isin(key_pts)].copy().reset_index(drop=True)
    df_key_dca.to_csv(OUT_DIR / "V8_TableS1F_eICU_DCA_KeyThresholds.csv", index=False)

    # =========================================================================
    # 3. VALIDATED NEUROLOGICAL & PHYSIOLOGICAL ICU SEVERITY-SCORE BENCHMARKS
    # =========================================================================
    e_ap4 = e_cc[e_cc["apachescore"] >= 0].copy()
    e_ap4a = e_cc[(e_cc["predictedhospitalmortality"] > 0) & (e_cc["predictedhospitalmortality"] < 1)].copy()
    s_1yr = s_cc[(s_cc["AdmissionYear"] <= 2020) & (s_cc["obs_ge_365d"] == 1)].copy()

    sev_rows = [
        # MIMIC-IV v3.1 (USA)
        fit_severity_comparison(
            m_cc, "in_hospital_death", ["gcs_motor"],
            "MIMIC-IV v3.1 (USA)", "Hospital mortality", "Motor GCS alone"
        ),
        fit_severity_comparison(
            m_cc, "in_hospital_death", ["gcs_total"],
            "MIMIC-IV v3.1 (USA)", "Hospital mortality", "Total GCS (3–15) alone"
        ),
        fit_severity_comparison(
            m_cc, "in_hospital_death", ["age", "male", "gcs_total", "vent", "ct_ich_any", "ct_mls", "ct_mass_hern"],
            "MIMIC-IV v3.1 (USA)", "Hospital mortality", "Age + Sex + Total GCS + Vent + Cranial CT"
        ),
        # eICU-CRD v2.0 (USA, 130 hospitals, hospital-clustered variance)
        fit_severity_comparison(
            e_cc, "in_hospital_death", ["gcs_motor"],
            "eICU-CRD v2.0 (USA, 130 hosp.)", "Hospital mortality", "Motor GCS alone",
            cluster_col="hospitalid"
        ),
        fit_severity_comparison(
            e_cc, "in_hospital_death", ["gcs_total"],
            "eICU-CRD v2.0 (USA, 130 hosp.)", "Hospital mortality", "Total GCS (3–15) alone",
            cluster_col="hospitalid"
        ),
        fit_severity_comparison(
            e_ap4, "in_hospital_death", ["apachescore"],
            "eICU-CRD v2.0 (USA, 130 hosp.)", "Hospital mortality", "APACHE IV physiological score (0–169)",
            cluster_col="hospitalid"
        ),
        fit_severity_comparison(
            e_ap4a, "in_hospital_death", [],
            "eICU-CRD v2.0 (USA, 130 hosp.)", "Hospital mortality", "APACHE IVa predicted hospital mortality",
            cluster_col="hospitalid", precomputed_prob_col="predictedhospitalmortality"
        ),
        # SICdb v1.0.8 (Austria)
        fit_severity_comparison(
            s_cc, "in_hospital_death", ["saps3"],
            "SICdb v1.0.8 (Austria)", "Hospital mortality", "SAPS 3 physiological score (16–107)"
        ),
        fit_severity_comparison(
            s_1yr, "died_by_365d", ["saps3"],
            "SICdb v1.0.8 (Austria, 2013–2020)", "Verified 1-year mortality", "SAPS 3 physiological score (16–107)"
        ),
        # HiRID v1.1.1 proxy (Switzerland)
        fit_severity_comparison(
            h_cc, "in_hospital_death", ["gcs_m_imp"],
            "HiRID v1.1.1 proxy (Switzerland)", "ICU mortality", "Motor GCS alone"
        ),
        fit_severity_comparison(
            h_cc, "in_hospital_death", ["age", "male", "gcs_m_imp", "vent"],
            "HiRID v1.1.1 proxy (Switzerland)", "ICU mortality", "Age + Sex + Motor GCS + Day-1 Vent"
        ),
    ]

    df_sev = pd.DataFrame(sev_rows)
    df_sev.to_csv(OUT_DIR / "V8_TableS1G_Conventional_Severity_Score_Benchmarks.csv", index=False)

    print("\n=== TABLE S1G: VALIDATED NEUROLOGICAL & ICU SEVERITY-SCORE BENCHMARKS (11 ROWS ACROSS 4 DBs) ===")
    for _, r in df_sev.iterrows():
        print(
            f"  {r['database']:33s} | {r['conventional_comparator']:42s} | n={int(r['n']):4d} ({int(r['deaths']):3d} d) | "
            f"AUC {r['comparator_auc']:.4f} -> {r['plus_states_auc']:.4f} (dAUC={r['delta_auc']:+.4f}, p={r['p_delta_auc']:.3g}) | "
            f"LRT chi2={r['lrt_chi2_2df']:6.2f} (P={r['lrt_p_2df']:.3g}), Wald P={r['global_2df_wald_p']:.3g} | "
            f"IDI={r['idi_pp']:+.2f} pp | E2 aOR={r['e2_vs_e1_aor']:.2f} ({r['e2_vs_e1_lo']:.2f}-{r['e2_vs_e1_hi']:.2f}), "
            f"E3 aOR={r['e3_vs_e1_aor']:.2f} ({r['e3_vs_e1_lo']:.2f}-{r['e3_vs_e1_hi']:.2f})"
        )


if __name__ == "__main__":
    main()
