#!/usr/bin/env python3
"""
ICM v7 — Pillar D Re-Verification & Unification Script:
1. Unifies SICdb left-censored INR (<= 1.20) lower integration bound to exact (0.70, 1.20]
   (matching locked baseline script 218a line 99 AND CLIP_BOUNDS inr lower bound 0.70!),
   ensuring 100% identical mathematical definition across baseline AND 48h/72h repeat scoring.
2. Explicitly labels binary 'any recorded 0–48h blood product' multivariable adjustment AND
   adds component-level multivariable adjustment (separate PRBC, platelet, and plasma/factor
   indicators entered simultaneously) in V7_TableD3.
3. Strictly read-only on all upstream datasets and all prior v1–v6 directories.
   Writes exclusively to icm_submission_v7_20261124/tables/.
"""
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

ROOT = Path(os.environ.get("TBI_ROOT", "./data_workspace"))
EXT_MIMIC = Path(os.environ.get("MIMIC_RAW_DIR", "./physionet/mimic"))
SIC_DIR = Path(os.environ.get("SICDB_DIR", "./physionet/sicdb"))
K3_DIR = ROOT / "derived/closed_loop_v1/clinical_k3"
P1_DIR = ROOT / "derived/closed_loop_v1/revision_gatekeepers_v1/p1_endpoints_and_sicdb"
HIR_DIR = ROOT / "derived/closed_loop_v1/transatlantic_4db_v1/hirid"
CPP_PRIV = ROOT / "clinical_strengthening_20261119/cpp/private"

V3_TAB_DIR = ROOT / "icm_submission_v3_20261123/tables"
OUT_DIR = ROOT / "icm_submission_v7_20261124/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATS6 = ["wbc", "plt", "hgb", "inr", "bicarb", "glu"]
RANGES = {"wbc": (0.2, 120.0), "plt": (5.0, 1500.0), "hgb": (3.0, 24.0), "inr": (0.5, 15.0), "bicarb": (5.0, 50.0), "glu": (20.0, 1200.0)}
CLIP_BOUNDS = [(0.2, 100.0), (10.0, 1000.0), (4.0, 22.0), (0.7, 10.0), (6.0, 45.0), (30.0, 900.0)]


def score_frozen_070(df_vals: pd.DataFrame, cfg: dict, sicdb_censored_inr: bool = False) -> np.ndarray:
    """Exact frozen 6-test diagonal GMM scorer with unified (0.70, 1.20] censored INR bound."""
    vals = [df_vals[f].astype(float).clip(lo, hi).to_numpy() for f, (lo, hi) in zip(FEATS6, CLIP_BOUNDS)]
    wbc, plt_v, hgb, inr, bicarb, glu = vals
    u = np.column_stack([np.log1p(wbc), -plt_v, -hgb, np.log(inr), -bicarb, np.log1p(glu)])
    center = np.asarray(cfg["scaler_center"])
    scale = np.asarray(cfg["scaler_scale"])
    x = (u - center) / scale
    means = np.asarray(cfg["gmm_means"])
    covars = np.asarray(cfg["gmm_covariances"])
    weights = np.asarray(cfg["gmm_weights"])

    logp = np.zeros((len(x), 3))
    for k in range(3):
        ll_other = np.zeros(len(x))
        for j in range(6):
            if sicdb_censored_inr and j == 3:
                is_cens = df_vals["inr"].astype(float).to_numpy() <= 1.2001
                x_hi = (np.log(1.20) - center[3]) / scale[3]
                x_lo = (np.log(0.70) - center[3]) / scale[3]
                sd = np.sqrt(covars[k, 3])
                cdf_diff = np.maximum(ndtr((x_hi - means[k, 3]) / sd) - ndtr((x_lo - means[k, 3]) / sd), 1e-12)
                pt_ll = -0.5 * (np.log(2 * np.pi * covars[k, 3]) + ((x[:, 3] - means[k, 3]) ** 2) / covars[k, 3])
                ll_other += np.where(is_cens, np.log(cdf_diff), pt_ll)
            else:
                ll_other += -0.5 * (np.log(2 * np.pi * covars[k, j]) + ((x[:, j] - means[k, j]) ** 2) / covars[k, j])
        logp[:, k] = np.log(weights[k]) + ll_other

    comp_map = {v: k[:2] for k, v in cfg["component_map"].items()}
    return np.array([comp_map[int(j)] for j in logp.argmax(axis=1)])


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
s31 = import_module("01a_landmark_window_engine_base")


def build_strict_icu_window_states_v7(base_df: pd.DataFrame, id_col: str, long_labs: pd.DataFrame, cfg: dict, sicdb_censored: bool = False):
    # Temporarily bind s31.score_frozen to score_frozen_070 so 100% of window logic is identical except x_lo=0.70!
    orig_fn = s31.score_frozen
    s31.score_frozen = score_frozen_070
    try:
        return s31.build_strict_icu_window_states(base_df, id_col, long_labs, cfg, sicdb_censored=sicdb_censored)
    finally:
        s31.score_frozen = orig_fn


def main():
    # Copy unchanged non-D tables from V3_TAB_DIR to OUT_DIR first
    for csv_p in sorted(V3_TAB_DIR.glob("*.csv")):
        if not csv_p.name.startswith("V3_TableD"):
            shutil.copy2(csv_p, OUT_DIR / csv_p.name)

    cfg = json.loads((K3_DIR / "locked_mimic_tbi_endotype_k3.json").read_text())

    # Verify baseline SICdb 516/516 exact parity with score_frozen_070!
    s_df = pd.read_csv(P1_DIR / "sicdb_icuoffset_censoredCDF_scored.csv")
    s_df["ph"] = s_df["phenotype"].str[:2]
    s_test = s_df.rename(columns={"inr_raw": "inr"}).copy()
    re_ph = score_frozen_070(s_test, cfg, sicdb_censored_inr=True)
    exact_match = int((re_ph == s_df["ph"].to_numpy()).sum())
    assert exact_match == 516, f"Expected 516/516 SICdb baseline match under (0.70, 1.20], got {exact_match}"
    s_plug = s_df.copy()
    s_plug["inr"] = np.where(s_plug["inr_raw"] <= 1.2001, 1.20, s_plug["inr_raw"])
    plug_ph = score_frozen_070(s_plug, cfg, sicdb_censored_inr=False)
    plug_agree = int((re_ph == plug_ph).sum())
    print(f"[Verified] SICdb baseline (0.70, 1.20] exact match = {exact_match}/516; 1.20 plug-in agreement = {plug_agree}/516 ({100*plug_agree/516:.1f}%)")

    m_df = pd.read_csv(K3_DIR / "mimic_tbi_k3_scored.csv")
    m_df["ph"] = m_df["endotype_k3"].str[:2]
    m_df["vent"] = m_df["vent_0_24h"].astype(float)
    m_df["icu_exit_h"] = (pd.to_datetime(m_df["outtime"]) - pd.to_datetime(m_df["intime"])).dt.total_seconds() / 3600.0
    dt_death = pd.to_datetime(m_df["deathtime"], errors="coerce")
    dt_disch = pd.to_datetime(m_df["dischtime"], errors="coerce")
    dt_in = pd.to_datetime(m_df["intime"])
    m_df["death_h_exact"] = np.where(
        m_df["in_hospital_death"] == 1,
        np.where(dt_death.notna(), (dt_death - dt_in).dt.total_seconds() / 3600.0, (dt_disch - dt_in).dt.total_seconds() / 3600.0),
        np.inf,
    )

    e_df = pd.read_csv(K3_DIR / "eicu_tbi_k3_scored.csv")
    e_df["ph"] = e_df["endotype_k3"].str[:2]
    e_df["age"] = e_df["age_num"].astype(float)
    e_df["vent"] = e_df["vent_24h"].astype(float)
    e_df["icu_exit_h"] = e_df["unitdischargeoffset"].astype(float) / 60.0
    e_df["hosp_exit_h"] = e_df["hospitaldischargeoffset"].astype(float) / 60.0
    e_df["death_h_exact"] = np.where(
        e_df["in_hospital_death"] == 1,
        np.where(e_df["icu_death"] == 1, e_df["icu_exit_h"], e_df["hosp_exit_h"]),
        np.inf,
    )

    s_df["vent"] = s_df["vent_24h"].astype(float)
    s_df["icu_exit_h"] = (s_df["TimeOfStay"].fillna(0) - s_df["ICUOffset"].fillna(0)) / 3600.0
    s_df["death_h_exact"] = np.where(
        s_df["in_hospital_death"] == 1,
        np.where(s_df["OffsetOfDeath"].notna(), (s_df["OffsetOfDeath"] - s_df["ICUOffset"].fillna(0)) / 3600.0, s_df["HospitalStayDays"].fillna(99) * 24.0),
        np.inf,
    )

    h_df = pd.read_csv(HIR_DIR / "hirid_true_neurotrauma_tbi_k3_scored.csv")
    h_df["ph"] = h_df["endotype"].str[:2]
    h_df["vent"] = h_df["vent_24h"].astype(float)
    h_df["icu_exit_h"] = h_df["max_obs_h"].astype(float)
    h_df["death_h_exact"] = np.where(h_df["in_hospital_death"] == 1, h_df["max_obs_h"].astype(float), np.inf)

    ml, el, sl, hl = s31.load_all_longitudinal_labs(m_df, e_df, s_df, h_df)
    m_ev, e_ev = s31.extract_us_blood_products(m_df, e_df)

    m_st = build_strict_icu_window_states_v7(m_df, "stay_id", ml, cfg, sicdb_censored=False)
    e_st = build_strict_icu_window_states_v7(e_df, "patientunitstayid", el, cfg, sicdb_censored=False)
    s_st = build_strict_icu_window_states_v7(s_df, "CaseID", sl, cfg, sicdb_censored=True)
    h_st = build_strict_icu_window_states_v7(h_df, "patientid", hl, cfg, sicdb_censored=False)

    # Compare SICdb repeat states between 0.50 (v3) and 0.70 (v7)
    s_st_old = s31.build_strict_icu_window_states(s_df, "CaseID", sl, cfg, sicdb_censored=True)
    for col_chk in ["ph_all6_win_48h", "ph_all6_avail_48h", "ph_win_bounded_48h", "ph_latest_inicu_48h",
                    "ph_all6_win_72h", "ph_all6_avail_72h", "ph_win_bounded_72h", "ph_latest_inicu_72h"]:
        both_valid = s_st[col_chk].notna() & s_st_old[col_chk].notna()
        n_diff = int((s_st.loc[both_valid, col_chk] != s_st_old.loc[both_valid, col_chk]).sum())
        print(f"[SICdb 0.50->0.70 check] {col_chk}: valid={int(both_valid.sum())}, changed={n_diff}")

    cats4 = ["rbc", "plt", "plasma", "factor"]
    for lm_h in [48, 72]:
        m_zero = m_ev[(m_ev["h"] >= 0.0) & (m_ev["h"] <= float(lm_h)) & (m_ev["cat"].isin(cats4))]
        m_min6 = m_ev[(m_ev["h"] >= -6.0) & (m_ev["h"] <= float(lm_h)) & (m_ev["cat"].isin(cats4))]
        m_st[f"blood_any_{lm_h}h"] = m_st["stay_id"].isin(set(m_zero["stay_id"])).astype(float)
        m_st[f"blood_any_minus6_{lm_h}h"] = m_st["stay_id"].isin(set(m_min6["stay_id"])).astype(float)
        for cat in cats4:
            m_st[f"blood_{cat}_{lm_h}h"] = m_st["stay_id"].isin(set(m_zero[m_zero["cat"] == cat]["stay_id"])).astype(int)
            m_st[f"blood_{cat}_minus6_{lm_h}h"] = m_st["stay_id"].isin(set(m_min6[m_min6["cat"] == cat]["stay_id"])).astype(int)
        m_st[f"blood_plasma_or_fac_{lm_h}h"] = ((m_st[f"blood_plasma_{lm_h}h"] == 1) | (m_st[f"blood_factor_{lm_h}h"] == 1)).astype(float)

        m_j = m_zero.merge(m_st[["stay_id", f"last_draw_all6_{lm_h}h"]], on="stay_id")
        m_trc = m_j[m_j["h"] <= m_j[f"last_draw_all6_{lm_h}h"]]
        m_st[f"blood_any_predraw_{lm_h}h"] = m_st["stay_id"].isin(set(m_trc["stay_id"])).astype(float)

        e_zero = e_ev[(e_ev["src"] == "recorded") & (e_ev["h"] >= 0.0) & (e_ev["h"] <= float(lm_h)) & (e_ev["cat"].isin(cats4))]
        e_min6 = e_ev[(e_ev["src"] == "recorded") & (e_ev["h"] >= -6.0) & (e_ev["h"] <= float(lm_h)) & (e_ev["cat"].isin(cats4))]
        e_plus = e_ev[(e_ev["h"] >= 0.0) & (e_ev["h"] <= float(lm_h)) & (e_ev["cat"].isin(cats4))]
        e_st[f"blood_any_{lm_h}h"] = e_st["patientunitstayid"].isin(set(e_zero["patientunitstayid"])).astype(float)
        e_st[f"blood_any_minus6_{lm_h}h"] = e_st["patientunitstayid"].isin(set(e_min6["patientunitstayid"])).astype(float)
        e_st[f"blood_any_plus_{lm_h}h"] = e_st["patientunitstayid"].isin(set(e_plus["patientunitstayid"])).astype(float)
        for cat in cats4:
            e_st[f"blood_{cat}_{lm_h}h"] = e_st["patientunitstayid"].isin(set(e_zero[e_zero["cat"] == cat]["patientunitstayid"])).astype(int)
            e_st[f"blood_{cat}_minus6_{lm_h}h"] = e_st["patientunitstayid"].isin(set(e_min6[e_min6["cat"] == cat]["patientunitstayid"])).astype(int)
        e_st[f"blood_plasma_or_fac_{lm_h}h"] = ((e_st[f"blood_plasma_{lm_h}h"] == 1) | (e_st[f"blood_factor_{lm_h}h"] == 1)).astype(float)

        e_j = e_zero.merge(e_st[["patientunitstayid", f"last_draw_all6_{lm_h}h"]], on="patientunitstayid")
        e_trc = e_j[e_j["h"] <= e_j[f"last_draw_all6_{lm_h}h"]]
        e_st[f"blood_any_predraw_{lm_h}h"] = e_st["patientunitstayid"].isin(set(e_trc["patientunitstayid"])).astype(float)

    flow_rows = []
    for dbn, d in [("MIMIC-IV", m_st), ("eICU-CRD", e_st), ("SICdb", s_st), ("HiRID proxy", h_st)]:
        for lm_h in [48, 72]:
            for grp in ["All", "E1", "E2", "E3"]:
                sub = d if grp == "All" else d[d["ph_W0"] == grp]
                died_pre = (sub["death_h_exact"] <= float(lm_h))
                unk_exit_alive = (~died_pre) & (sub["icu_exit_h"].isna())
                left_icu_alive = (~died_pre) & (sub["icu_exit_h"] <= float(lm_h))
                in_icu = (~died_pre) & (sub["icu_exit_h"] > float(lm_h))
                assert int(died_pre.sum() + unk_exit_alive.sum() + left_icu_alive.sum() + in_icu.sum()) == len(sub)
                icu_sub = sub[in_icu]

                c_all6 = f"ph_all6_win_{lm_h}h"
                c_av6 = f"ph_all6_avail_{lm_h}h"
                c_wbnd = f"ph_win_bounded_{lm_h}h"
                c_lat = f"ph_latest_inicu_{lm_h}h"

                s6 = icu_sub[icu_sub[c_all6].notna()]
                sav = icu_sub[icu_sub[c_av6].notna()]
                sw = icu_sub[icu_sub[c_wbnd].notna()]
                slat = icu_sub[icu_sub[c_lat].notna()]

                flow_rows.append({
                    "database": dbn,
                    "landmark_h": lm_h,
                    "baseline_group": grp,
                    "n_baseline": len(sub),
                    "died_at_or_before_landmark": int(died_pre.sum()),
                    "alive_at_landmark_exited_icu_le_landmark": int(left_icu_alive.sum()),
                    "unknown_icu_exit_alive_n": int(unk_exit_alive.sum()),
                    "remained_in_icu_at_landmark": int(in_icu.sum()),
                    "in_icu_unscored_post_baseline": int(len(icu_sub) - len(slat)),
                    "in_icu_latest_scored_n": len(slat),
                    "in_icu_active_window_bounded_n": len(sw),
                    "in_icu_all6_fresh_in_window_n": len(s6),
                    "in_icu_all6_fresh_verified_avail_n": len(sav),
                    "all6_to_E1_n": int((s6[c_all6] == "E1").sum()),
                    "all6_to_E1_pct": round(100.0 * (s6[c_all6] == "E1").mean(), 1) if len(s6) else np.nan,
                    "all6_to_E2_n": int((s6[c_all6] == "E2").sum()),
                    "all6_to_E3_n": int((s6[c_all6] == "E3").sum()),
                    "all6_to_E3_pct": round(100.0 * (s6[c_all6] == "E3").mean(), 1) if len(s6) else np.nan,
                    "bounded_win_to_E1_n": int((sw[c_wbnd] == "E1").sum()),
                    "bounded_win_to_E1_pct": round(100.0 * (sw[c_wbnd] == "E1").mean(), 1) if len(sw) else np.nan,
                    "latest_inicu_to_E1_n": int((slat[c_lat] == "E1").sum()),
                    "latest_inicu_to_E1_pct": round(100.0 * (slat[c_lat] == "E1").mean(), 1) if len(slat) else np.nan,
                })
    flow_df = pd.DataFrame(flow_rows)
    flow_df.to_csv(OUT_DIR / "V3_TableD1_MutuallyExclusive_ICU_Landmark_Flow.csv", index=False)

    surv_rows = []
    for lm_h in [48, 72]:
        for rule_code, col_tpl, rule_lbl in [
            ("strict_icu_all6_fresh", f"ph_all6_win_{lm_h}h", f"Strict in-ICU, all 6 tests freshly redrawn in active window ({lm_h-24}–{lm_h}h)"),
            ("strict_icu_all6_avail", f"ph_all6_avail_{lm_h}h", f"Strict in-ICU, all 6 fresh + result release <= {lm_h}h"),
            ("strict_icu_active_window", f"ph_win_bounded_{lm_h}h", f"Strict in-ICU, active-window bounded (>=3 fresh in {lm_h-24}–{lm_h}h)"),
            ("strict_icu_latest", f"ph_latest_inicu_{lm_h}h", f"Strict in-ICU, most recent post-baseline pre-{lm_h}h state"),
        ]:
            per_db_results = {}
            for dbn, d, sev_c, endp_lbl, cl_col in [
                ("MIMIC-IV", m_st, "gcs_motor", "Post-landmark hospital death", None),
                ("eICU-CRD", e_st, "gcs_motor", "Post-landmark hospital death", "hospitalid"),
                ("SICdb", s_st, "saps3", "Post-landmark hospital death", None),
                ("HiRID proxy", h_st, "gcs_motor", "Post-landmark ICU death", None),
            ]:
                icu_alive = d[(d["death_h_exact"] > float(lm_h)) & (d["icu_exit_h"] > float(lm_h))].copy()
                covs = ["age", "male", sev_c, "vent"]
                for c in covs:
                    icu_alive[c] = pd.to_numeric(icu_alive[c], errors="coerce").fillna(icu_alive[c].median())

                e3_p = icu_alive[(icu_alive["ph_W0"] == "E3") & icu_alive[col_tpl].isin(["E1", "E3"])].copy()
                if len(e3_p) >= 20:
                    e3_p["res_E1"] = (e3_p[col_tpl] == "E1").astype(float)
                    n_ref = int((e3_p["res_E1"] == 0).sum()); d_ref = int(e3_p.loc[e3_p["res_E1"] == 0, "in_hospital_death"].sum())
                    n_tr = int((e3_p["res_E1"] == 1).sum()); d_tr = int(e3_p.loc[e3_p["res_E1"] == 1, "in_hospital_death"].sum())
                    if n_tr >= 3 and d_ref >= 2 and d_tr >= 1:
                        fit = s31._safe_fit_logit(e3_p, "in_hospital_death", "res_E1", covs, cl_col)
                        if fit is not None:
                            ci = np.exp(fit.conf_int())
                            r_obj = {
                                "database": dbn, "endpoint": endp_lbl, "landmark_h": lm_h, "rule_code": rule_code, "rule_label": rule_lbl,
                                "contrast": "Resolved E3 -> E1 vs Persistent E3 -> E3",
                                "n_pair": len(e3_p), "ref_deaths_N": f"{d_ref}/{n_ref} ({100*d_ref/n_ref:.1f}%)",
                                "trans_deaths_N": f"{d_tr}/{n_tr} ({100*d_tr/n_tr:.1f}%)",
                                "aOR": float(np.exp(fit.params["res_E1"])), "ci_lo": float(ci.loc["res_E1", 0]), "ci_hi": float(ci.loc["res_E1", 1]),
                                "log_or": float(fit.params["res_E1"]), "se_log_or": float(fit.bse["res_E1"]), "p_raw": float(fit.pvalues["res_E1"]),
                            }
                            surv_rows.append(r_obj)
                            per_db_results.setdefault("Resolved E3 -> E1 vs Persistent E3 -> E3", []).append(r_obj)

                for tgt_s, c_lbl in [("E3", "De novo E1 -> E3 vs Stable E1 -> E1"), ("E2", "De novo E1 -> E2 vs Stable E1 -> E1 (exploratory)")]:
                    e1_p = icu_alive[(icu_alive["ph_W0"] == "E1") & icu_alive[col_tpl].isin(["E1", tgt_s])].copy()
                    if len(e1_p) >= 30:
                        e1_p["tr"] = (e1_p[col_tpl] == tgt_s).astype(float)
                        n_ref = int((e1_p["tr"] == 0).sum()); d_ref = int(e1_p.loc[e1_p["tr"] == 0, "in_hospital_death"].sum())
                        n_tr = int((e1_p["tr"] == 1).sum()); d_tr = int(e1_p.loc[e1_p["tr"] == 1, "in_hospital_death"].sum())
                        if n_tr >= 5 and d_ref >= 2 and d_tr >= 1:
                            fit = s31._safe_fit_logit(e1_p, "in_hospital_death", "tr", covs, cl_col)
                            if fit is not None:
                                ci = np.exp(fit.conf_int())
                                r_obj = {
                                    "database": dbn, "endpoint": endp_lbl, "landmark_h": lm_h, "rule_code": rule_code, "rule_label": rule_lbl,
                                    "contrast": c_lbl,
                                    "n_pair": len(e1_p), "ref_deaths_N": f"{d_ref}/{n_ref} ({100*d_ref/n_ref:.1f}%)",
                                    "trans_deaths_N": f"{d_tr}/{n_tr} ({100*d_tr/n_tr:.1f}%)",
                                    "aOR": float(np.exp(fit.params["tr"])), "ci_lo": float(ci.loc["tr", 0]), "ci_hi": float(ci.loc["tr", 1]),
                                    "log_or": float(fit.params["tr"]), "se_log_or": float(fit.bse["tr"]), "p_raw": float(fit.pvalues["tr"]),
                                }
                                surv_rows.append(r_obj)
                                per_db_results.setdefault(c_lbl, []).append(r_obj)

                all_s = icu_alive[icu_alive[col_tpl].notna()].copy()
                if len(all_s) >= 80 and int(all_s["in_hospital_death"].sum()) >= 15:
                    all_s["b_E2"] = (all_s["ph_W0"] == "E2").astype(float)
                    all_s["b_E3"] = (all_s["ph_W0"] == "E3").astype(float)
                    all_s["c_E2"] = (all_s[col_tpl] == "E2").astype(float)
                    all_s["c_E3"] = (all_s[col_tpl] == "E3").astype(float)
                    fit_j = s31._safe_fit_logit(all_s, "in_hospital_death", "c_E3", ["c_E2", "b_E2", "b_E3"] + covs, cl_col)
                    if fit_j is not None:
                        ci = np.exp(fit_j.conf_int())
                        wt_2df = fit_j.wald_test("c_E2 = 0, c_E3 = 0", scalar=True)
                        p_wald_2df = float(wt_2df.pvalue)
                        n_ref = int((all_s[col_tpl] == "E1").sum()); d_ref = int(all_s.loc[all_s[col_tpl] == "E1", "in_hospital_death"].sum())
                        n_tr = int((all_s[col_tpl] == "E3").sum()); d_tr = int(all_s.loc[all_s[col_tpl] == "E3", "in_hospital_death"].sum())
                        X0 = sm.add_constant(all_s[["b_E2", "b_E3"] + covs].astype(float))
                        X1 = sm.add_constant(all_s[["c_E2", "c_E3", "b_E2", "b_E3"] + covs].astype(float))
                        m0 = sm.Logit(all_s["in_hospital_death"].astype(float), X0).fit(disp=False, maxiter=200)
                        m1 = sm.Logit(all_s["in_hospital_death"].astype(float), X1).fit(disp=False, maxiter=200)
                        auc0 = roc_auc_score(all_s["in_hospital_death"], m0.predict(X0))
                        auc1 = roc_auc_score(all_s["in_hospital_death"], m1.predict(X1))
                        lr_stat = max(0.0, 2.0 * (m1.llf - m0.llf))
                        lr_p = float(1.0 - sp_stats.chi2.cdf(lr_stat, df=2))
                        r_obj = {
                            "database": dbn, "endpoint": endp_lbl, "landmark_h": lm_h, "rule_code": rule_code, "rule_label": rule_lbl,
                            "contrast": f"Dynamic {lm_h}h E3 vs E1 (adj. W0; AUC {auc0:.3f}->{auc1:.3f}, 2df Wald P={p_wald_2df:.3g}, LRT P={lr_p:.3g})",
                            "n_pair": len(all_s), "ref_deaths_N": f"{d_ref}/{n_ref} ({100*d_ref/n_ref:.1f}%)",
                            "trans_deaths_N": f"{d_tr}/{n_tr} ({100*d_tr/n_tr:.1f}%)",
                            "aOR": float(np.exp(fit_j.params["c_E3"])), "ci_lo": float(ci.loc["c_E3", 0]), "ci_hi": float(ci.loc["c_E3", 1]),
                            "log_or": float(fit_j.params["c_E3"]), "se_log_or": float(fit_j.bse["c_E3"]), "p_raw": float(fit_j.pvalues["c_E3"]),
                        }
                        surv_rows.append(r_obj)
                        per_db_results.setdefault("Dynamic Landmark E3 vs E1 (adjusted for Baseline W0)", []).append(r_obj)

            for c_key, r_list in per_db_results.items():
                hosp_r = [r for r in r_list if r["database"] in ("MIMIC-IV", "eICU-CRD", "SICdb")]
                if len(hosp_r) >= 2:
                    m_res = s31.dl_meta(hosp_r)
                    surv_rows.append({
                        "database": f"Pooled Hospital Meta ({'+'.join(r['database'] for r in hosp_r)})",
                        "endpoint": "Post-landmark hospital death",
                        "landmark_h": lm_h,
                        "rule_code": rule_code,
                        "rule_label": rule_lbl + f" [DL Meta, I2={m_res['I2_pct']}%]",
                        "contrast": c_key,
                        "n_pair": int(sum(r["n_pair"] for r in hosp_r)),
                        "ref_deaths_N": f"I2={m_res['I2_pct']}%",
                        "trans_deaths_N": f"k={m_res['k']} cohorts (FE aOR {m_res['fe_aOR']:.2f})",
                        "aOR": m_res["aOR"],
                        "ci_lo": m_res["ci_lo"],
                        "ci_hi": m_res["ci_hi"],
                        "log_or": np.nan,
                        "se_log_or": np.nan,
                        "p_raw": m_res["p_raw"],
                    })

    surv_df = pd.DataFrame(surv_rows)
    surv_df.to_csv(OUT_DIR / "V3_TableD2_Strict_InICU_Landmark_Survival_Models.csv", index=False)

    bp_rows = []
    for lm_h in [48, 72]:
        for rule_code, col_tpl in [
            ("strict_icu_all6_fresh", f"ph_all6_win_{lm_h}h"),
            ("strict_icu_latest", f"ph_latest_inicu_{lm_h}h"),
        ]:
            us_per_db_adj = []
            us_per_db_comp = []
            us_per_db_with = []
            us_per_db_without = []
            us_per_db_predraw = []
            us_per_db_minus6 = []

            for dbn, d, cl_col in [("MIMIC-IV", m_st, None), ("eICU-CRD", e_st, "hospitalid")]:
                icu_alive = d[(d["death_h_exact"] > float(lm_h)) & (d["icu_exit_h"] > float(lm_h))].copy()
                e3_p = icu_alive[(icu_alive["ph_W0"] == "E3") & icu_alive[col_tpl].isin(["E1", "E3"])].copy()
                for c in ["age", "male", "gcs_motor", "vent"]:
                    e3_p[c] = pd.to_numeric(e3_p[c], errors="coerce").fillna(e3_p[c].median())
                e3_p["res_E1"] = (e3_p[col_tpl] == "E1").astype(float)
                b_col = f"blood_any_{lm_h}h"
                e3_p["any_blood_to_lm"] = e3_p[b_col].astype(float)

                g_res = e3_p[e3_p["res_E1"] == 1]
                g_per = e3_p[e3_p["res_E1"] == 0]
                res_bp = f"{int(g_res[b_col].sum())}/{len(g_res)} ({100*g_res[b_col].mean():.1f}%)" if len(g_res) else "0/0"
                per_bp = f"{int(g_per[b_col].sum())}/{len(g_per)} ({100*g_per[b_col].mean():.1f}%)" if len(g_per) else "0/0"

                # 1. Binary any-blood-product indicator adjustment (exact primary model)
                fit_adj = s31._safe_fit_logit(e3_p, "in_hospital_death", "res_E1", ["age", "male", "gcs_motor", "vent", "any_blood_to_lm"], cl_col)
                if fit_adj is not None:
                    ci = np.exp(fit_adj.conf_int())
                    obj = {
                        "database": dbn, "landmark_h": lm_h, "rule_code": rule_code,
                        "specification": f"Adjusted for any recorded 0–{lm_h}h blood product (binary indicator)",
                        "n_pair": len(e3_p),
                        "resolved_blood_exposed": res_bp,
                        "persistent_blood_exposed": per_bp,
                        "resolved_PRBC_Plt_Plasma_Factor": f"{int(g_res[f'blood_rbc_{lm_h}h'].sum())}/{int(g_res[f'blood_plt_{lm_h}h'].sum())}/{int(g_res[f'blood_plasma_{lm_h}h'].sum())}/{int(g_res[f'blood_factor_{lm_h}h'].sum())}",
                        "persistent_PRBC_Plt_Plasma_Factor": f"{int(g_per[f'blood_rbc_{lm_h}h'].sum())}/{int(g_per[f'blood_plt_{lm_h}h'].sum())}/{int(g_per[f'blood_plasma_{lm_h}h'].sum())}/{int(g_per[f'blood_factor_{lm_h}h'].sum())}",
                        "aOR": float(np.exp(fit_adj.params["res_E1"])), "ci_lo": float(ci.loc["res_E1", 0]), "ci_hi": float(ci.loc["res_E1", 1]),
                        "log_or": float(fit_adj.params["res_E1"]), "se_log_or": float(fit_adj.bse["res_E1"]), "p_raw": float(fit_adj.pvalues["res_E1"]),
                    }
                    bp_rows.append(obj)
                    us_per_db_adj.append(obj)

                # 2. Separate component-level multivariable adjustment (PRBC + Platelets + Plasma/Factor entered simultaneously!)
                comp_cols = [f"blood_rbc_{lm_h}h", f"blood_plt_{lm_h}h", f"blood_plasma_or_fac_{lm_h}h"]
                fit_comp = s31._safe_fit_logit(e3_p, "in_hospital_death", "res_E1", ["age", "male", "gcs_motor", "vent"] + comp_cols, cl_col)
                if fit_comp is not None:
                    ci_c = np.exp(fit_comp.conf_int())
                    c_obj = {
                        "database": dbn, "landmark_h": lm_h, "rule_code": rule_code,
                        "specification": f"Sensitivity: adjusted for separate 0–{lm_h}h PRBC, platelet, and plasma/factor indicators",
                        "n_pair": len(e3_p),
                        "resolved_blood_exposed": res_bp,
                        "persistent_blood_exposed": per_bp,
                        "resolved_PRBC_Plt_Plasma_Factor": f"{int(g_res[f'blood_rbc_{lm_h}h'].sum())}/{int(g_res[f'blood_plt_{lm_h}h'].sum())}/{int(g_res[f'blood_plasma_{lm_h}h'].sum())}/{int(g_res[f'blood_factor_{lm_h}h'].sum())}",
                        "persistent_PRBC_Plt_Plasma_Factor": f"{int(g_per[f'blood_rbc_{lm_h}h'].sum())}/{int(g_per[f'blood_plt_{lm_h}h'].sum())}/{int(g_per[f'blood_plasma_{lm_h}h'].sum())}/{int(g_per[f'blood_factor_{lm_h}h'].sum())}",
                        "aOR": float(np.exp(fit_comp.params["res_E1"])), "ci_lo": float(ci_c.loc["res_E1", 0]), "ci_hi": float(ci_c.loc["res_E1", 1]),
                        "log_or": float(fit_comp.params["res_E1"]), "se_log_or": float(fit_comp.bse["res_E1"]), "p_raw": float(fit_comp.pvalues["res_E1"]),
                    }
                    bp_rows.append(c_obj)
                    us_per_db_comp.append(c_obj)

                for b_val, st_lbl, bucket in [
                    (1.0, f"Subgroup WITH recorded 0–{lm_h}h blood product", us_per_db_with),
                    (0.0, f"Subgroup WITHOUT recorded 0–{lm_h}h blood product", us_per_db_without),
                ]:
                    sub_b = e3_p[e3_p["any_blood_to_lm"] == b_val].copy()
                    if len(sub_b) >= 25 and int(sub_b.loc[sub_b["res_E1"] == 1, "in_hospital_death"].sum()) >= 1 and int(sub_b.loc[sub_b["res_E1"] == 0, "in_hospital_death"].sum()) >= 2:
                        fit_s = s31._safe_fit_logit(sub_b, "in_hospital_death", "res_E1", ["age", "male", "gcs_motor", "vent"], cl_col)
                        if fit_s is not None:
                            ci_s = np.exp(fit_s.conf_int())
                            sr = sub_b[sub_b["res_E1"] == 1]
                            sp = sub_b[sub_b["res_E1"] == 0]
                            s_obj = {
                                "database": dbn, "landmark_h": lm_h, "rule_code": rule_code, "specification": st_lbl,
                                "n_pair": len(sub_b),
                                "resolved_blood_exposed": f"{int(sr['in_hospital_death'].sum())}/{len(sr)} deaths",
                                "persistent_blood_exposed": f"{int(sp['in_hospital_death'].sum())}/{len(sp)} deaths",
                                "resolved_PRBC_Plt_Plasma_Factor": "—", "persistent_PRBC_Plt_Plasma_Factor": "—",
                                "aOR": float(np.exp(fit_s.params["res_E1"])), "ci_lo": float(ci_s.loc["res_E1", 0]), "ci_hi": float(ci_s.loc["res_E1", 1]),
                                "log_or": float(fit_s.params["res_E1"]), "se_log_or": float(fit_s.bse["res_E1"]), "p_raw": float(fit_s.pvalues["res_E1"]),
                            }
                            bp_rows.append(s_obj)
                            bucket.append(s_obj)

                if rule_code == "strict_icu_all6_fresh":
                    for sens_col, sens_lbl, sens_bucket in [
                        (f"blood_any_predraw_{lm_h}h", "Sensitivity: adjusted for 0h to final-redraw blood product", us_per_db_predraw),
                        (f"blood_any_minus6_{lm_h}h", f"Sensitivity: adjusted for [-6, +{lm_h}]h pre-ICU-inclusive blood product", us_per_db_minus6),
                    ]:
                        e3_s = e3_p.copy()
                        e3_s["any_blood_to_lm"] = e3_s[sens_col].astype(float)
                        fit_sn = s31._safe_fit_logit(e3_s, "in_hospital_death", "res_E1", ["age", "male", "gcs_motor", "vent", "any_blood_to_lm"], cl_col)
                        if fit_sn is not None:
                            ci_sn = np.exp(fit_sn.conf_int())
                            sn_obj = {
                                "database": dbn, "landmark_h": lm_h, "rule_code": rule_code, "specification": sens_lbl,
                                "n_pair": len(e3_s),
                                "resolved_blood_exposed": f"{int(g_res[sens_col].sum())}/{len(g_res)} ({100*g_res[sens_col].mean():.1f}%)",
                                "persistent_blood_exposed": f"{int(g_per[sens_col].sum())}/{len(g_per)} ({100*g_per[sens_col].mean():.1f}%)",
                                "resolved_PRBC_Plt_Plasma_Factor": "—", "persistent_PRBC_Plt_Plasma_Factor": "—",
                                "aOR": float(np.exp(fit_sn.params["res_E1"])), "ci_lo": float(ci_sn.loc["res_E1", 0]), "ci_hi": float(ci_sn.loc["res_E1", 1]),
                                "log_or": float(fit_sn.params["res_E1"]), "se_log_or": float(fit_sn.bse["res_E1"]), "p_raw": float(fit_sn.pvalues["res_E1"]),
                            }
                            bp_rows.append(sn_obj)
                            sens_bucket.append(sn_obj)

                    if dbn == "eICU-CRD":
                        e3_pl = e3_p.copy()
                        p_col = f"blood_any_plus_{lm_h}h"
                        e3_pl["any_blood_to_lm"] = e3_pl[p_col].astype(float)
                        fit_pl = s31._safe_fit_logit(e3_pl, "in_hospital_death", "res_E1", ["age", "male", "gcs_motor", "vent", "any_blood_to_lm"], cl_col)
                        if fit_pl is not None:
                            ci_pl = np.exp(fit_pl.conf_int())
                            bp_rows.append({
                                "database": dbn, "landmark_h": lm_h, "rule_code": rule_code,
                                "specification": f"Sensitivity: adjusted for 0–{lm_h}h recorded + structured treatment entries",
                                "n_pair": len(e3_pl),
                                "resolved_blood_exposed": f"{int(g_res[p_col].sum())}/{len(g_res)} ({100*g_res[p_col].mean():.1f}%)",
                                "persistent_blood_exposed": f"{int(g_per[p_col].sum())}/{len(g_per)} ({100*g_per[p_col].mean():.1f}%)",
                                "resolved_PRBC_Plt_Plasma_Factor": "—", "persistent_PRBC_Plt_Plasma_Factor": "—",
                                "aOR": float(np.exp(fit_pl.params["res_E1"])), "ci_lo": float(ci_pl.loc["res_E1", 0]), "ci_hi": float(ci_pl.loc["res_E1", 1]),
                                "log_or": float(fit_pl.params["res_E1"]), "se_log_or": float(fit_pl.bse["res_E1"]), "p_raw": float(fit_pl.pvalues["res_E1"]),
                            })

            for lbl_m, b_list in [
                (f"Pooled US (MIMIC+eICU) — Adjusted for any 0–{lm_h}h blood product", us_per_db_adj),
                (f"Pooled US (MIMIC+eICU) — Adjusted for separate 0–{lm_h}h PRBC/Plt/Plasma-Factor indicators", us_per_db_comp),
                (f"Pooled US (MIMIC+eICU) — Subgroup WITH 0–{lm_h}h blood product", us_per_db_with),
                (f"Pooled US (MIMIC+eICU) — Subgroup WITHOUT 0–{lm_h}h blood product", us_per_db_without),
                ("Pooled US (MIMIC+eICU) — Sensitivity: 0h to final-redraw blood product", us_per_db_predraw),
                (f"Pooled US (MIMIC+eICU) — Sensitivity: [-6, +{lm_h}]h pre-ICU-inclusive blood product", us_per_db_minus6),
            ]:
                if len(b_list) == 2:
                    mm = s31.dl_meta(b_list)
                    bp_rows.append({
                        "database": "Pooled US (MIMIC-IV+eICU-CRD)", "landmark_h": lm_h, "rule_code": rule_code, "specification": lbl_m,
                        "n_pair": int(sum(x["n_pair"] for x in b_list)),
                        "resolved_blood_exposed": f"I2={mm['I2_pct']}%", "persistent_blood_exposed": "k=2 US cohorts",
                        "resolved_PRBC_Plt_Plasma_Factor": "—", "persistent_PRBC_Plt_Plasma_Factor": "—",
                        "aOR": mm["aOR"], "ci_lo": mm["ci_lo"], "ci_hi": mm["ci_hi"],
                        "log_or": np.nan, "se_log_or": np.nan, "p_raw": mm["p_raw"],
                    })

    bp_df = pd.DataFrame(bp_rows)
    bp_df.to_csv(OUT_DIR / "V3_TableD3_US_BloodProduct_Exposure_and_Adjusted_Models.csv", index=False)

    print("\n=== POOLED SURVIVAL MODELS (V7 UNIFIED 0.70) ===")
    print(surv_df[surv_df["database"].str.startswith("Pooled")][["database", "landmark_h", "rule_code", "contrast", "n_pair", "aOR", "ci_lo", "ci_hi", "p_raw"]].to_string(index=False))
    print("\n=== US 0–48H BLOOD PRODUCT MODELS (BINARY + COMPONENT-LEVEL) ===")
    print(bp_df[bp_df["landmark_h"] == 48][["database", "rule_code", "specification", "n_pair", "aOR", "ci_lo", "ci_hi", "p_raw"]].to_string(index=False))


if __name__ == "__main__":
    main()
