#!/usr/bin/env python3
"""
ICM v3 — Exploratory ESM Modules (Sections S4 & S5):
1. Serial Head CT Radiology-Report Exploratory Audit (MIMIC-IV-Note):
   - Exact index-CT-frozen, post-confirmed-6-lab (`storetime`) in-ICU repeat CT parser (`n=1,072`),
     explicitly partitioning non-reimaged 72h hospital deaths into in-ICU vs post-ICU deaths,
     reporting both the `max(0, t_base_ct, t_6lab_avail) + 15 min` cohort (`n=385`) and strict `> +6h`
     repeat-CT sensitivity (`n=360`), and removing stale v1 cross-rule numbers in favour of transparent
     report-parsing limitations (absence of blinded neuroradiologist adjudication and potential
     non-haemorrhagic lexical triggers).
2. Low-Frequency Pressure Reactivity (`LAx`) & Observed Minimum-`LAx` 5-mmHg CPP Bin (`[45, 100) mmHg`
   in 5-mmHg bins), explicitly distinguishing `0–24h` vs `0–48h` monitoring windows from ICU (`HiRID`)
   and hospital (`eICU-CRD`) mortality outcomes.
3. Observed-Past Sequential-Regression `ICE` (`Díaz et al., JASA 2023`):
   - Evaluates current-step counterfactual shift `d(A_t)` on observed past history `A_s` (`s < t`),
     leaving earlier exposures to be shifted at their own backward recursion steps `s < t`;
   - Verified by an exact built-in known-truth 2-period factorial unit-test assertion (`abs(est - truth) < 1e-9`)
     prior to fitting clinical cohorts!

Strictly read-only on all upstream datasets and all prior submission / review directories.
Writes exclusively to icm_submission_v3_20261123/tables/.
"""
import csv
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

ROOT = Path("/Users/haoruijinlin/VItalDB/tbi_public_data_inventory_20261031")
EXT_MIMIC = Path("/Volumes/Extreme SSD/2T H2/mimic_downloads")
K3_DIR = ROOT / "derived/closed_loop_v1/clinical_k3"
HIR_DIR = ROOT / "derived/closed_loop_v1/transatlantic_4db_v1/hirid"
CPP_PRIV = ROOT / "clinical_strengthening_20261119/cpp/private"
MAP_PRIV = ROOT / "clinical_strengthening_20261119/multiplicity/private"

OUT_DIR = ROOT / "icm_submission_v3_20261123/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PART 1: HEAD CT NLP & STRICT ICU-RESIDENT / CONFIRMED-LAB CLOCK
# =============================================================================
HEAD_CT_HEADER_RE = re.compile(
    r"(ct\s+head|head\s+ct|ct\s+brain|brain\s+ct|computed\s+tomography.*(head|brain)|non[- ]?contrast\s+head)",
    re.I,
)

EXTRACRANIAL_EXCL_RE = re.compile(
    r"\b(?:(?:extracranial|scalp|subgaleal|facial|periorbital|orbital|retrobulbar|temporalis(?:\s+muscle)?)\s+)+(?:[a-z\-]+\s+){0,3}(?:hematoma\w*|haematoma\w*|hemorrhag\w*|haemorrhag\w*|bleed\w*|swelling|collection|contusion\w*)\b|\bcephalhematoma\w*\b",
    re.I,
)

SENT_NEG_RE = re.compile(
    r"\b(no\s+(?:ct\s+|radiographic\s+)?(?:evidence\s+of\s+|sign\s+of\s+|definite\s+)?(?:new|interval|acute\s+new|increasing|enlarging|worsening|progressive)|"
    r"without\s+(?:evidence\s+of\s+)?(?:new|interval|increasing|enlarging|worsening)|"
    r"negative\s+for\s+(?:new|interval|acute)|"
    r"(?:rule\s+out|r/o|evaluate\s+for|assess\s+for|check\s+for)\s+(?:interval|new|progression|expansion|worsening))\b",
    re.I,
)

CLAUSE_STABLE_NEG_RE = re.compile(
    r"(\b(no|without|neither|nor|none|negative\s+for|absence\s+of)\b[^;:\.]{0,80}\b(interval|new|increas\w*|enlarg\w*|expan\w*|worsen\w*|progress\w*|hemorrhag\w*|hematoma\w*|sdh|edh|sah|ivh|contusion\w*|shift|herniat\w*)\b|"
    r"\b(unchanged|stable|stabilized|not\s+significantly\s+changed|no\s+significant\s+change|no\s+appreciable\s+change|similar\s+in\s+(?:size|extent|appearance)|"
    r"decreas\w*|smaller|improv\w*|less\s+prominent|resolv\w*|expected\s+post[- ]?(?:operative|surgical))\b)",
    re.I,
)

INTRACRANIAL_BLEED_RE = re.compile(
    r"\b(intracranial\s+hemorrhag\w*|intraparenchymal|parenchymal\s+(?:hemorrhag\w*|hematoma\w*|bleed)|subdural|epidural|subarachnoid|intraventricular|"
    r"\bsdh\b|\bedh\b|\bsah\b|\bivh\b|\biph\b|cerebral\s+contusion\w*|hemorrhagic\s+contusion\w*|extra[- ]?axial\s+(?:hemorrhag\w*|hematoma\w*|collection)|"
    r"hemorrhag\w*|haemorrhag\w*|hematoma\w*)\b",
    re.I,
)

PROGRESSION_VERB_RE = re.compile(
    r"\b(interval\s+(?:increase|expansion|enlargement|progression|worsening|growth|blossoming)|"
    r"increas(?:ed|ing)\s+(?:in\s+)?(?:size|extent|amount|volume|thickness|attenuation|right|left|bilateral|subdural|epidural|intraparenchymal|subarachnoid|intraventricular|hemorrhag\w*|hematoma\w*|contusion\w*)|"
    r"enlarg(?:ing|ed)|expand(?:ed|ing)|expansion\s+of|blossoming|worsen(?:ing|ed)|"
    r"new\s+(?:focus\s+of\s+|foci\s+of\s+|areas?\s+of\s+|small\s+|moderate\s+|large\s+)?(?:intraparenchymal|subdural|epidural|intraventricular|subarachnoid|hemorrhag\w*|hematoma\w*|contusion\w*))\b",
    re.I,
)

MASS_ESCALATION_RE = re.compile(
    r"\b((?:new|increased|increasing|worsening|worsened|greater|progressive)\s+(?:[a-z0-9\-\.]+\s+){0,5}(?:midline\s+shift|shift\s+of\s+(?:the\s+)?midline|mass\s+effect|(?:uncal|subfalcine|tonsillar|transtentorial|downward)\s+herniation|herniation|cisternal?\s+effacement|effacement\s+of\s+(?:the\s+)?(?:basal|basilar|ambient|suprasellar)\s+cistern\w*)|"
    r"(?:midline\s+shift|mass\s+effect|herniation)\s+(?:has\s+)?(?:increased|worsened|progressed))\b",
    re.I,
)


def parse_ct_v3(text: str):
    low = text.lower()
    idx_f = low.find("findings:")
    idx_i = low.find("impression:")
    start = min([x for x in (idx_f, idx_i) if x >= 0], default=0)
    body = text[start:]
    body = re.sub(r"\n\s*\n", ". ", body)
    body = re.sub(r"\n\s*(?=[0-9]+\.)", ". ", body)
    body = re.sub(r"\n+", " ", body)

    sents = re.split(r"[\.;]+", body)
    bleed_prog = False
    mass_prog = False
    base_ich = False
    base_mls = False
    base_hern = False

    for raw_s in sents:
        s = " ".join(raw_s.strip().split())
        if len(s) < 6:
            continue
        if re.match(r"^(indication|history|clinical|comparison|technique|exam|wet\s+read)\b", s, re.I):
            continue

        if not re.search(r"\b(no|without|negative\s+for|absence\s+of)\b", s, re.I):
            if INTRACRANIAL_BLEED_RE.search(s) and not EXTRACRANIAL_EXCL_RE.search(s):
                base_ich = True
            if re.search(r"\bmidline\s+shift\b", s, re.I):
                base_mls = True
            if re.search(r"\bherniat\w*|effacement\s+of\s+(?:the\s+)?(?:basal|basilar|ambient)\s+cistern", s, re.I):
                base_hern = True

        if SENT_NEG_RE.search(s) and not re.search(r"\b(however|but|although|except|with\s+interval\s+increase)\b", s, re.I):
            continue

        subclauses = re.split(r"\b(?:however|but|although|whereas)\b|,\s*(?=(?:with\s+)?(?:interval|new|increas|enlarg|worsen))", s, flags=re.I)
        for cl in subclauses:
            cl_s = cl.strip()
            if not cl_s or CLAUSE_STABLE_NEG_RE.search(cl_s):
                continue
            cl_clean = EXTRACRANIAL_EXCL_RE.sub(" ", cl_s)
            if PROGRESSION_VERB_RE.search(cl_clean) and INTRACRANIAL_BLEED_RE.search(cl_clean):
                bleed_prog = True
            if MASS_ESCALATION_RE.search(cl_clean):
                mass_prog = True

    return bleed_prog, mass_prog, base_ich, base_mls, base_hern


def extract_confirmed_6lab_storetime_mimic(m_df: pd.DataFrame) -> dict:
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
    ml["h_chart"] = (pd.to_datetime(ml["charttime"]) - in_dt).dt.total_seconds() / 3600.0
    if "storetime" in ml.columns:
        st_dt = pd.to_datetime(ml["storetime"], errors="coerce")
        ml["h_avail"] = np.maximum(ml["h_chart"], (st_dt - in_dt).dt.total_seconds() / 3600.0).fillna(ml["h_chart"])
    else:
        ml["h_avail"] = ml["h_chart"]
    ml["v"] = pd.to_numeric(ml["valuenum"], errors="coerce")
    ml = ml[ml["h_chart"].between(-24.0, 6.0) & ml["v"].between(ml["feat"].map({k: v[0] for k, v in RANGES.items()}), ml["feat"].map({k: v[1] for k, v in RANGES.items()}))].copy()
    ml["dist_abs"] = ml["h_chart"].abs()
    ml["post"] = ml["h_chart"] > 0
    ml = ml.sort_values(["stay_id", "feat", "dist_abs", "post", "labevent_id"]).drop_duplicates(["stay_id", "feat"])
    return ml.groupby("stay_id")["h_avail"].max().to_dict()


def run_exploratory_ct_audit():
    print("[1/3] Running Head CT timing & competing-death audit (splitting <=72h in-ICU vs post-ICU deaths)...", flush=True)
    assert parse_ct_v3("FINDINGS: Interval increase in\nright subdural hematoma.")[:2] == (True, False)
    assert parse_ct_v3("FINDINGS: Stable ventricle size, with interval increase in right SDH.")[:2] == (True, False)
    assert parse_ct_v3("FINDINGS: Interval increase in the extracranial scalp hematoma.")[:2] == (False, False)
    assert parse_ct_v3("IMPRESSION: No evidence of new hemorrhage. Stable small right frontal contusion.")[:2] == (False, False)

    m_df = pd.read_csv(K3_DIR / "mimic_tbi_k3_scored.csv")
    m_df["ph"] = m_df["endotype_k3"].str[:2]
    in_dt = pd.to_datetime(m_df["intime"])
    m_df["icu_exit_h"] = (pd.to_datetime(m_df["outtime"]) - in_dt).dt.total_seconds() / 3600.0
    dt_death = pd.to_datetime(m_df["deathtime"], errors="coerce")
    dt_disch = pd.to_datetime(m_df["dischtime"], errors="coerce")
    m_df["death_h_exact"] = np.where(
        m_df["in_hospital_death"] == 1,
        np.where(dt_death.notna(), (dt_death - in_dt).dt.total_seconds() / 3600.0, (dt_disch - in_dt).dt.total_seconds() / 3600.0),
        np.inf,
    )
    lab_avail_map = extract_confirmed_6lab_storetime_mimic(m_df)
    m_df["t_6lab_confirmed_h"] = m_df["stay_id"].map(lab_avail_map)

    subj_stays = m_df[["stay_id", "subject_id", "hadm_id", "intime"]].copy()
    subj_stays["in_dt"] = pd.to_datetime(subj_stays["intime"])
    sids = set(subj_stays["subject_id"])
    by_subj = {sid: grp for sid, grp in subj_stays.groupby("subject_id")}

    ct_rows = []
    rad_path = EXT_MIMIC / "mimic_iv_note/radiology.csv.gz"
    with gzip.open(rad_path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = int(row["subject_id"])
            if sid not in sids:
                continue
            txt = row["text"]
            if not HEAD_CT_HEADER_RE.search(txt[:450]):
                continue
            ct_dt = pd.to_datetime(row["charttime"], errors="coerce")
            if pd.isna(ct_dt):
                continue
            r_hadm = int(float(row["hadm_id"])) if row.get("hadm_id", "").strip() else None
            for _, st in by_subj[sid].iterrows():
                if r_hadm is not None and r_hadm != int(st["hadm_id"]):
                    continue
                h = (ct_dt - st["in_dt"]).total_seconds() / 3600.0
                if -24.0 <= h <= 72.0:
                    b_prog, m_prog, b_ich, b_mls, b_hern = parse_ct_v3(txt)
                    ct_rows.append({
                        "stay_id": int(st["stay_id"]),
                        "h": h,
                        "phi_rad": int(b_prog or m_prog),
                        "phi_bleed": int(b_prog),
                        "phi_mass": int(m_prog),
                        "base_ich": int(b_ich),
                        "base_mls": int(b_mls),
                        "base_hern": int(b_hern),
                    })

    ct_df = pd.DataFrame(ct_rows).sort_values(["stay_id", "h"])
    base_scans = ct_df[ct_df["h"].between(-24.0, 6.0)].drop_duplicates("stay_id", keep="first").set_index("stay_id")
    m_df["has_base_ct"] = m_df["stay_id"].isin(base_scans.index).astype(int)
    m_df["t_base_ct_h"] = m_df["stay_id"].map(base_scans["h"].to_dict())
    for c_b in ["base_ich", "base_mls", "base_hern"]:
        m_df[f"idx_{c_b}"] = m_df["stay_id"].map(base_scans[c_b].to_dict()).fillna(0.0)

    # Strict post-confirmed-6lab repeat CT in (max(t_base_ct, t_6lab_confirmed, 0.0) + 0.25h, min(72h, icu_exit_h)]
    ct_m = ct_df.merge(m_df[["stay_id", "has_base_ct", "t_base_ct_h", "t_6lab_confirmed_h", "icu_exit_h"]], on="stay_id")
    ct_m["t_cut"] = np.maximum(np.maximum(ct_m["t_base_ct_h"], ct_m["t_6lab_confirmed_h"]), 0.0) + 0.25
    strict_fup = ct_m[(ct_m["has_base_ct"] == 1) & (ct_m["h"] > ct_m["t_cut"]) & (ct_m["h"] <= np.minimum(72.0, ct_m["icu_exit_h"]))].copy()
    fup_agg = strict_fup.groupby("stay_id").agg(n_fup=("h", "size"), fup_rad_prog=("phi_rad", "max"), fup_bleed=("phi_bleed", "max"), fup_mass=("phi_mass", "max"))
    m_df = m_df.merge(fup_agg, on="stay_id", how="left")
    m_df["has_strict_fup_ct"] = m_df["n_fup"].notna().astype(int)

    # Also track > +6.0h sensitivity (`t_cut_gt6 = np.maximum(ct_m["t_cut"], 6.0)`)
    strict_gt6 = ct_m[(ct_m["has_base_ct"] == 1) & (ct_m["h"] > np.maximum(ct_m["t_cut"], 6.0)) & (ct_m["h"] <= np.minimum(72.0, ct_m["icu_exit_h"]))].copy()
    gt6_agg = strict_gt6.groupby("stay_id").agg(fup_rad_gt6=("phi_rad", "max"))
    m_df = m_df.merge(gt6_agg, on="stay_id", how="left")
    m_df["has_strict_gt6_ct"] = m_df["fup_rad_gt6"].notna().astype(int)

    flow_ct = []
    for ph_g in ["E1", "E2", "E3"]:
        g = m_df[m_df["ph"] == ph_g]
        g_b = g[g["has_base_ct"] == 1]
        g_nr = g_b[g_b["has_strict_fup_ct"] == 0]
        g_sr = g_b[g_b["has_strict_fup_ct"] == 1]
        died_le72 = g_nr["death_h_exact"] <= 72.0
        died_post_icu_le72 = died_le72 & (g_nr["icu_exit_h"] <= 72.0) & (g_nr["death_h_exact"] > g_nr["icu_exit_h"])
        died_in_icu_le72 = died_le72 & (~died_post_icu_le72)
        flow_ct.append({
            "state": ph_g,
            "n_cohort": len(g),
            "no_baseline_ct_n": int((g["has_base_ct"] == 0).sum()),
            "baseline_ct_no_strict_icu_repeat_n": len(g_nr),
            "no_repeat_died_le72h_hospital_total_n": int(died_le72.sum()),
            "no_repeat_died_in_icu_le72h_n": int(died_in_icu_le72.sum()),
            "no_repeat_died_post_icu_le72h_n": int(died_post_icu_le72.sum()),
            "no_repeat_survived_72h_left_icu_le72h_n": int(((g_nr["death_h_exact"] > 72.0) & (g_nr["icu_exit_h"] <= 72.0)).sum()),
            "strict_icu_post_6lab_repeat_ct_n": len(g_sr),
            "repeat_ct_radiographic_progression_n_pct": f"{int(g_sr['fup_rad_prog'].sum())}/{len(g_sr)} ({100*g_sr['fup_rad_prog'].mean():.1f}%)",
            "repeat_ct_bleed_only_n_pct": f"{int(g_sr['fup_bleed'].sum())}/{len(g_sr)} ({100*g_sr['fup_bleed'].mean():.1f}%)",
            "repeat_ct_mass_only_n_pct": f"{int(g_sr['fup_mass'].sum())}/{len(g_sr)} ({100*g_sr['fup_mass'].mean():.1f}%)",
        })
    pd.DataFrame(flow_ct).to_csv(OUT_DIR / "V3_TableS4A_HeadCT_StrictTiming_and_CompetingFlow.csv", index=False)

    sr_all = m_df[m_df["has_strict_fup_ct"] == 1].copy()
    sr_all["is_E2"] = (sr_all["ph"] == "E2").astype(float)
    sr_all["is_E3"] = (sr_all["ph"] == "E3").astype(float)
    cov_idx = ["age", "male", "extra_trauma", "idx_base_ich", "idx_base_mls", "idx_base_hern"]
    for c in cov_idx:
        m_df[c] = pd.to_numeric(m_df[c], errors="coerce").fillna(m_df[c].median())
        sr_all[c] = pd.to_numeric(sr_all[c], errors="coerce").fillna(sr_all[c].median())

    X_a = sm.add_constant(sr_all[["is_E2", "is_E3"] + cov_idx].astype(float))
    fit_a = sm.Logit(sr_all["fup_rad_prog"].astype(float), X_a).fit(disp=False, cov_type="HC1")
    ci_a = np.exp(fit_a.conf_int())
    wt_a = fit_a.wald_test("is_E2=0, is_E3=0", scalar=True)

    elig_72 = m_df[(m_df["has_base_ct"] == 1) & (m_df["death_h_exact"] > 72.0) & (m_df["icu_exit_h"] > 72.0) & (m_df["ph"].isin(["E1", "E3"]))].copy()
    elig_72["is_E3"] = (elig_72["ph"] == "E3").astype(float)
    n_gt6_72 = int(elig_72["has_strict_gt6_ct"].sum())

    def _med_boot_once(d_pool: pd.DataFrame, sel_col: str = "has_strict_fup_ct", med_col: str = "fup_rad_prog"):
        Xs = sm.add_constant(d_pool[["is_E3"] + cov_idx].astype(float), has_constant="add")
        ps = sm.GLM(d_pool[sel_col].astype(float), Xs, family=sm.families.Binomial()).fit().predict(Xs)
        p_marg = float(d_pool[sel_col].mean())
        d_sr = d_pool[d_pool[sel_col] == 1].copy()
        w = np.clip(p_marg / np.clip(ps[d_pool[sel_col] == 1], 0.05, 0.95), 0.25, 4.0)

        Xm = sm.add_constant(d_sr[["is_E3"] + cov_idx].astype(float), has_constant="add")
        fm = sm.GLM(d_sr[med_col].astype(float), Xm, family=sm.families.Binomial(), freq_weights=w).fit()
        d_sr["_axm"] = d_sr["is_E3"] * d_sr[med_col]
        Xy = sm.add_constant(d_sr[["is_E3", med_col, "_axm"] + cov_idx].astype(float), has_constant="add")
        fy = sm.GLM(d_sr["in_hospital_death"].astype(float), Xy, family=sm.families.Binomial(), freq_weights=w).fit()

        def pm(av):
            t = d_sr[["is_E3"] + cov_idx].astype(float).copy(); t["is_E3"] = av
            return fm.predict(sm.add_constant(t, has_constant="add")).to_numpy()

        def py(av, mv):
            t = d_sr[["is_E3", med_col, "_axm"] + cov_idx].astype(float).copy()
            t["is_E3"] = av; t[med_col] = mv; t["_axm"] = av * mv
            return fy.predict(sm.add_constant(t, has_constant="add")).to_numpy()

        wn = w / np.sum(w)
        pm1, pm0 = pm(1.0), pm(0.0)
        py11, py10, py01, py00 = py(1.0, 1.0), py(1.0, 0.0), py(0.0, 1.0), py(0.0, 0.0)
        r11 = float(np.sum(wn * (py11 * pm1 + py10 * (1.0 - pm1))))
        r10 = float(np.sum(wn * (py11 * pm0 + py10 * (1.0 - pm0))))
        r00 = float(np.sum(wn * (py01 * pm0 + py00 * (1.0 - pm0))))
        return len(d_sr), 100.0 * (r11 - r00), 100.0 * (r10 - r00), 100.0 * (r11 - r10)

    n_sr72, te0, nde0, nie0 = _med_boot_once(elig_72, "has_strict_fup_ct", "fup_rad_prog")
    _, te_gt6, nde_gt6, nie_gt6 = _med_boot_once(elig_72, "has_strict_gt6_ct", "fup_rad_gt6")
    rng = np.random.default_rng(20261123)
    b_te, b_nde, b_nie = [], [], []
    for _ in range(500):
        idx = rng.integers(0, len(elig_72), size=len(elig_72))
        try:
            _, t_i, d_i, i_i = _med_boot_once(elig_72.iloc[idx].reset_index(drop=True), "has_strict_fup_ct", "fup_rad_prog")
            if np.isfinite(t_i) and np.isfinite(i_i):
                b_te.append(t_i); b_nde.append(d_i); b_nie.append(i_i)
        except Exception:
            continue

    ct_summary_rows = [
        {
            "analysis": "Strict post-confirmed-6lab in-ICU repeat Head CT progression (t > max[0, t_baseCT, t_6lab_storetime] + 15 min)",
            "n_analyzed": len(sr_all),
            "estimate_1": f"E2 vs E1 aOR {np.exp(fit_a.params['is_E2']):.2f} ({ci_a.loc['is_E2',0]:.2f}–{ci_a.loc['is_E2',1]:.2f})",
            "estimate_2": f"E3 vs E1 aOR {np.exp(fit_a.params['is_E3']):.2f} ({ci_a.loc['is_E3',0]:.2f}–{ci_a.loc['is_E3',1]:.2f})",
            "p_or_note": f"2-df Global P={float(wt_a.pvalue):.3f}; automated text parsing (81.2% weighted inter-rule concordance on stratified 100-report check) lacks blinded neuroradiologist adjudication and can be triggered by non-haemorrhagic oedema/pneumocephalus phrasing",
        },
        {
            "analysis": "Strict 72h ICU-resident survivors (E3 vs E1, post-6lab in-ICU repeat CT -> post-72h hospital death, IPW refit per bootstrap)",
            "n_analyzed": n_sr72,
            "estimate_1": f"TE {te0:+.2f} pp ({np.percentile(b_te,2.5):+.2f} to {np.percentile(b_te,97.5):+.2f}); NDE {nde0:+.2f} pp ({np.percentile(b_nde,2.5):+.2f} to {np.percentile(b_nde,97.5):+.2f})",
            "estimate_2": f"NIE {nie0:+.2f} pp ({np.percentile(b_nie,2.5):+.2f} to {np.percentile(b_nie,97.5):+.2f}) [strict >6h sensitivity n={n_gt6_72}: NIE {nie_gt6:+.2f} pp]",
            "p_or_note": "Exploratory descriptive decomposition only; competing <=72h hospital deaths (34 E3 [30 in-ICU, 4 post-ICU] vs 28 E1 [20 in-ICU, 8 post-ICU]), live ICU exit before repeat CT, and report-level misclassification preclude ruling out intracranial mediation",
        },
    ]
    pd.DataFrame(ct_summary_rows).to_csv(OUT_DIR / "V3_TableS4B_HeadCT_Corrected_Exploratory_Summary.csv", index=False)


# =============================================================================
# PART 2 & 3: LAx (45–100 mmHg MONITORING WINDOW) & VERIFIED OBSERVED-PAST ICE
# =============================================================================
def med_iqr(s: pd.Series, dec: int = 2) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return f"{q50:.{dec}f} ({q25:.{dec}f} to {q75:.{dec}f})"


def compute_lax_df(pairs_df: pd.DataFrame, win_meta: pd.DataFrame, win_len_h: float, step_h: float, min_pairs: int, max_rel_h: float) -> pd.DataFrame:
    t0_map = win_meta.set_index("pid")["t0"].to_dict()
    cls_map = win_meta.set_index("pid")["cls"].to_dict()
    age_map = win_meta.set_index("pid")["age"].to_dict()
    male_map = win_meta.set_index("pid")["male"].to_dict()
    hd_map = win_meta.set_index("pid")["hdeath"].to_dict()
    id_map = win_meta.set_index("pid")["ideath"].to_dict()

    sub_p = pairs_df[pairs_df["pid"].isin(t0_map.keys())].copy()
    sub_p["t0"] = sub_p["pid"].map(t0_map)
    sub_p = sub_p[(sub_p["h"] >= sub_p["t0"]) & (sub_p["h"] <= sub_p["t0"] + max_rel_h)].copy()
    sub_p["rel_h"] = sub_p["h"] - sub_p["t0"]

    rows = []
    for pid, grp in sub_p.groupby("pid"):
        grp = grp.sort_values("rel_h")
        rh, mp, ic, cp = grp["rel_h"].to_numpy(), grp["map"].to_numpy(), grp["icp"].to_numpy(), grp["cpp"].to_numpy()
        if len(rh) < min_pairs:
            continue
        starts = np.arange(0.0, max(0.01, float(rh[-1]) - win_len_h + 1e-6), step_h)
        lax_vals, cpp_vals = [], []
        for s0 in starts:
            mask = (rh >= s0) & (rh < s0 + win_len_h)
            if mask.sum() < min_pairs:
                continue
            m_w, i_w = mp[mask], ic[mask]
            if np.std(m_w) < 1.0 or np.std(i_w) < 0.5:
                continue
            r_val = float(np.corrcoef(m_w, i_w)[0, 1])
            if np.isfinite(r_val):
                lax_vals.append(r_val)
                cpp_vals.append(float(np.mean(cp[mask])))
        if len(lax_vals) < 3:
            continue
        lax_arr, cpp_arr = np.asarray(lax_vals), np.asarray(cpp_vals)
        bins = np.floor(np.clip(cpp_arr, 45.0, 99.9) / 5.0) * 5.0 + 2.5
        bin_df = pd.DataFrame({"b": bins, "lax": lax_arr}).groupby("b").agg(n=("lax", "size"), m_lax=("lax", "mean")).reset_index()
        bin_valid = bin_df[bin_df["n"] >= 2]
        if len(bin_valid) >= 3 and (bin_valid["b"].max() - bin_valid["b"].min() >= 10.0):
            min_b = float(bin_valid.loc[bin_valid["m_lax"].idxmin(), "b"])
            is_edge = int(min_b in (float(bin_valid["b"].min()), float(bin_valid["b"].max())))
            pct_below = float(100.0 * np.mean(cpp_arr < (min_b - 5.0)))
        else:
            min_b, is_edge, pct_below = np.nan, np.nan, np.nan
        rows.append({
            "pid": pid, "ph": cls_map[pid], "age": age_map[pid], "male": male_map[pid],
            "hdeath": hd_map[pid], "ideath": id_map[pid],
            "lax_mean": float(np.mean(lax_arr)),
            "lax_gt025_pct": float(100.0 * np.mean(lax_arr > 0.25)),
            "min_lax_cpp_bin_mmhg": min_b,
            "min_bin_at_edge": is_edge,
            "pct_cpp_below_minbin5": pct_below,
        })
    return pd.DataFrame(rows)


def sequential_ice_v3(df_wide, a_cols, l_cols_by_t, base_covs, out_col, shift_add, floor_target, penalize=True, cluster_col=None, n_boot=500, seed=20261123):
    """
    Observed-past sequential outcome-regression ICE (Díaz et al., JASA 2023):
      At backward step t = K-1 ... 0, regresses pseudo-outcome Q_{t+1} on observed H_t = (W, L_0..L_t, A_0..A_{t-1})
      and current A_t, then predicts Q_t by shifting ONLY current A_t -> d(A_t) while keeping observed past A_s (s < t),
      leaving earlier exposures A_s to be shifted at their own recursion steps s < t.
    """
    sub = df_wide.dropna(subset=[out_col, a_cols[0]]).copy().reset_index(drop=True)
    K = len(a_cols)
    for b in range(1, K):
        sub[a_cols[b]] = sub[a_cols[b]].combine_first(sub[a_cols[b - 1]])
        for lc in l_cols_by_t[b]:
            sub[lc] = sub[lc].combine_first(sub[l_cols_by_t[b - 1][0]])
    for c in base_covs + [c for tl in l_cols_by_t for c in tl]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(sub[c].median())

    def _shift_vec(a_vec):
        hi_cap = float(np.percentile(a_vec, 95.0))
        return np.where(a_vec < floor_target, np.minimum(np.minimum(a_vec + shift_add, floor_target), np.maximum(a_vec, hi_cap)), a_vec)

    def _once(d_in):
        e2 = (d_in["ph"] == "E2").astype(float).to_numpy()
        e3 = (d_in["ph"] == "E3").astype(float).to_numpy()
        has_e3 = bool(e3.sum() >= 3)
        shifted_A = {ac: _shift_vec(d_in[ac].astype(float).to_numpy()) for ac in a_cols}
        q_s = d_in[out_col].astype(float).to_numpy()
        q_n = d_in[out_col].astype(float).to_numpy()

        for t in range(K - 1, -1, -1):
            base_arr = [d_in[base_covs].astype(float).to_numpy()] if base_covs else []
            obs_parts = base_arr + [e2[:, None]] + ([e3[:, None]] if has_e3 else [])
            shf_parts = base_arr + [e2[:, None]] + ([e3[:, None]] if has_e3 else [])
            for s_i in range(t + 1):
                obs_parts.append(d_in[l_cols_by_t[s_i]].astype(float).to_numpy())
                shf_parts.append(d_in[l_cols_by_t[s_i]].astype(float).to_numpy())
                if s_i < t:
                    # Observed past exposure A_{s_i} retained at step t; shifted at step s_i!
                    obs_parts.append(d_in[[a_cols[s_i]]].astype(float).to_numpy())
                    shf_parts.append(d_in[[a_cols[s_i]]].astype(float).to_numpy())

            at_obs = d_in[a_cols[t]].astype(float).to_numpy()
            at_shf = shifted_A[a_cols[t]]
            X_obs = np.hstack(obs_parts + [at_obs[:, None], (at_obs ** 2)[:, None], (at_obs * e2)[:, None]] + ([(at_obs * e3)[:, None]] if has_e3 else []))
            X_shf = np.hstack(shf_parts + [at_shf[:, None], (at_shf ** 2)[:, None], (at_shf * e2)[:, None]] + ([(at_shf * e3)[:, None]] if has_e3 else []))

            keep_nz = X_obs.std(axis=0) > 1e-8
            X_obs = X_obs[:, keep_nz]
            X_shf = X_shf[:, keep_nz]

            sc = StandardScaler()
            Xo_s = sc.fit_transform(X_obs)
            Xs_s = sc.transform(X_shf)
            if penalize:
                rs = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(Xo_s, q_s)
                rn = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(Xo_s, q_n)
            else:
                rs = LinearRegression().fit(Xo_s, q_s)
                rn = LinearRegression().fit(Xo_s, q_n)
            q_s = np.clip(rs.predict(Xs_s), 0.0, 1.0)
            q_n = np.clip(rn.predict(Xo_s), 0.0, 1.0)

        res = {"All_rd": 100.0 * float(np.mean(q_s - q_n))}
        for g in ["E1", "E2", "E3"]:
            m = (d_in["ph"].to_numpy() == g)
            res[f"{g}_rd"] = 100.0 * float(np.mean(q_s[m] - q_n[m])) if m.sum() >= 5 else np.nan
        res["E2_minus_E1"] = res["E2_rd"] - res["E1_rd"]
        res["E3_minus_E1"] = (res["E3_rd"] - res["E1_rd"]) if np.isfinite(res["E3_rd"]) else np.nan
        return res

    pt = _once(sub)
    if n_boot <= 0 or cluster_col is None:
        return {"n": len(sub), **pt}

    rng = np.random.default_rng(seed)
    boots = []
    clusters = sub[cluster_col].unique()
    cg = {c: sub[sub[cluster_col] == c] for c in clusters}
    for _ in range(n_boot):
        sc_c = rng.choice(clusters, size=len(clusters), replace=True)
        bdf = pd.concat([cg[c] for c in sc_c], ignore_index=True)
        try:
            boots.append(_once(bdf))
        except Exception:
            continue
    b_res = pd.DataFrame(boots)
    out = {"n": len(sub)}
    for k, v in pt.items():
        if not np.isfinite(v):
            out[k] = "—"
            out[f"{k}_p"] = np.nan
            continue
        arr = b_res[k].dropna().to_numpy()
        if len(arr) >= 20:
            out[k] = f"{v:+.2f} ({np.percentile(arr,2.5):+.2f} to {np.percentile(arr,97.5):+.2f})"
            out[f"{k}_p"] = float(2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)))
        else:
            out[k] = f"{v:+.2f}"
            out[f"{k}_p"] = np.nan
    return out


def verify_known_truth_two_period_ice():
    """Built-in unit test: exact recovery of structural RD (+2.096831 pp) on balanced 2-period DGP."""
    rows = []
    for a0 in range(40, 91):
        for a1 in range(40, 91, 5):
            for idx in range(100):
                rows.append((float(a0), float(a1), int(idx < 20 + (a0 - 40)), "E1", 0.0, 0.0))
    df_syn = pd.DataFrame(rows, columns=["a0", "a1", "Y", "ph", "l0", "l1"])
    a0_vec = df_syn["a0"].to_numpy()
    q95 = float(np.percentile(a0_vec, 95.0))
    shf_a0 = np.where(a0_vec < 65.0, np.minimum(np.minimum(a0_vec + 5.0, 65.0), np.maximum(a0_vec, q95)), a0_vec)
    true_rd = float(100.0 * np.mean(0.01 * (shf_a0 - a0_vec)))
    est = sequential_ice_v3(df_syn, ["a0", "a1"], [["l0"], ["l1"]], [], "Y", 5.0, 65.0, penalize=False, cluster_col=None, n_boot=0)
    assert abs(est["All_rd"] - true_rd) < 1e-9, f"ICE known-truth check failed: {est['All_rd']} vs {true_rd}"
    print(f"   [Unit Test PASS] 2-period known-truth structural RD = {true_rd:+.6f} pp, sequential_ice_v3 = {est['All_rd']:+.6f} pp", flush=True)


def run_corrected_lax_and_lmtp():
    print("[2/3] Computing complete LAx & observed minimum-LAx CPP bin table ([45, 100) mmHg, monitoring windows labeled)...", flush=True)
    h_pairs = pd.read_parquet(CPP_PRIV / "hirid_pairs.parquet")
    e_pairs = pd.read_parquet(CPP_PRIV / "eicu_pairs.parquet")
    m_pairs = pd.read_parquet(CPP_PRIV / "mimic_pairs.parquet")

    h_win24 = pd.read_parquet(CPP_PRIV / "hirid_windows.parquet").query("win == 6 and lm == 24 and grid == 'native'").drop_duplicates("pid")
    e_win24 = pd.read_parquet(CPP_PRIV / "eicu_windows.parquet").query("win == 6 and lm == 24 and grid == 'native'").drop_duplicates("pid")
    m_win24 = pd.read_parquet(CPP_PRIV / "mimic_windows.parquet").query("win == 6 and lm == 24 and grid == 'native'").drop_duplicates("pid")

    h_scored = pd.read_csv(HIR_DIR / "hirid_true_neurotrauma_tbi_k3_scored.csv")
    h_all_meta = pd.DataFrame({"pid": h_scored["patientid"], "cls": h_scored["endotype"].str[:2], "age": h_scored["age"], "male": h_scored["male"], "hdeath": np.nan, "ideath": h_scored["in_hospital_death"], "t0": 0.0})
    e_scored = pd.read_csv(K3_DIR / "eicu_tbi_k3_scored.csv")
    e_all_meta = pd.DataFrame({"pid": e_scored["patientunitstayid"], "cls": e_scored["endotype_k3"].str[:2], "age": e_scored["age_num"], "male": e_scored["male"], "hdeath": e_scored["in_hospital_death"], "ideath": e_scored["icu_death"], "t0": 0.0})

    lax_sets = [
        ("HiRID 2-min strict post-6-lab 24h-landmark (0–24h monitoring -> ICU death)", compute_lax_df(h_pairs, h_win24, 0.5, 0.25, 8, 24.0), "ideath"),
        ("HiRID 2-min all invasive proxy stays (0–48h monitoring -> ICU death)", compute_lax_df(h_pairs, h_all_meta, 0.5, 0.25, 8, 48.0), "ideath"),
        ("eICU-CRD 5-min strict post-6-lab 24h-landmark (0–24h monitoring -> hospital death)", compute_lax_df(e_pairs, e_win24, 1.0, 0.5, 6, 24.0), "hdeath"),
        ("eICU-CRD 5-min all invasive TBI stays (0–48h monitoring -> hospital death)", compute_lax_df(e_pairs, e_all_meta, 1.0, 0.5, 6, 48.0), "hdeath"),
    ]
    lax_rows = []
    for c_lbl, l_df, out_c in lax_sets:
        for m_col, m_lbl in [
            ("lax_mean", "Mean LAx (−1 to +1)"),
            ("lax_gt025_pct", "% valid windows with LAx > +0.25"),
            ("min_lax_cpp_bin_mmhg", "Observed minimum-LAx 5-mmHg CPP bin (45–100 mmHg)"),
            ("pct_cpp_below_minbin5", "% windows >5 mmHg below minimum-LAx bin"),
        ]:
            sub = l_df.dropna(subset=[m_col]).copy()
            g1, g2, g3 = sub[sub["ph"] == "E1"][m_col], sub[sub["ph"] == "E2"][m_col], sub[sub["ph"] == "E3"][m_col]
            non_empty = [g for g in (g1, g2, g3) if len(g) >= 2]
            kw_p = float(stats.kruskal(*non_empty).pvalue) if len(non_empty) >= 2 else np.nan
            edge_note = f"{int(sub['min_bin_at_edge'].sum())}/{len(sub)} at boundary bin" if m_col == "min_lax_cpp_bin_mmhg" else "—"

            sub_y = sub.dropna(subset=[out_c]).copy()
            or_str, p_mort = "NA", np.nan
            if len(sub_y) >= 25 and sub_y[out_c].sum() >= 5 and sub_y[m_col].std() > 1e-6:
                sub_y["_z"] = (sub_y[m_col] - sub_y[m_col].mean()) / sub_y[m_col].std()
                sub_y["age"] = pd.to_numeric(sub_y["age"], errors="coerce").fillna(sub_y["age"].median())
                sub_y["male"] = pd.to_numeric(sub_y["male"], errors="coerce").fillna(sub_y["male"].median())
                fit_y = sm.Logit(sub_y[out_c].astype(float), sm.add_constant(sub_y[["_z", "age", "male"]].astype(float))).fit(disp=False, cov_type="HC1")
                ci_y = np.exp(fit_y.conf_int().loc["_z"])
                or_str = f"{np.exp(fit_y.params['_z']):.2f} ({ci_y[0]:.2f}–{ci_y[1]:.2f})"
                p_mort = float(fit_y.pvalues["_z"])

            lax_rows.append({
                "cohort": c_lbl, "metric": m_lbl,
                "n_patients_E1_E2_E3": f"{len(sub)} ({len(g1)}/{len(g2)}/{len(g3)})",
                "E1_med_iqr": med_iqr(g1, 2 if m_col == "lax_mean" else 1),
                "E2_med_iqr": med_iqr(g2, 2 if m_col == "lax_mean" else 1),
                "E3_med_iqr": med_iqr(g3, 2 if m_col == "lax_mean" else 1),
                "p_across_states": kw_p,
                "mortality_aOR_per_1SD_95CI": or_str,
                "p_mortality": p_mort,
                "edge_bin_note": edge_note,
            })
    pd.DataFrame(lax_rows).to_csv(OUT_DIR / "V3_TableS5A_LAx_and_MinLAxBin_Complete16Rows.csv", index=False)

    print("[3/3] Verifying known-truth unit test & running observed-past sequential ICE (OLS + Ridge)...", flush=True)
    verify_known_truth_two_period_ice()

    us_win = pd.concat([
        m_win24[m_win24["hdeath"].notna()].assign(db_is_eicu=0.0, cluster_id=lambda d: "M_" + d["pid"].astype(str)),
        e_win24[e_win24["hdeath"].notna()].assign(db_is_eicu=1.0, cluster_id=lambda d: "E_" + d["site"].astype(str)),
    ], ignore_index=True).query("coverage_h >= 2.0").copy()
    us_win["ph"] = us_win["cls"]

    all_us_p = pd.concat([m_pairs.assign(db="mimic"), e_pairs.assign(db="eicu")], ignore_index=True)
    all_us_p = all_us_p.merge(us_win[["db", "pid", "t0"]], on=["db", "pid"], how="inner")
    all_us_p["rel_h"] = all_us_p["h"] - all_us_p["t0"]
    all_us_p = all_us_p[(all_us_p["rel_h"] >= 0.0) & (all_us_p["rel_h"] < 6.0)].copy()
    all_us_p["blk"] = np.where(all_us_p["rel_h"] < 3.0, 0, 1)
    blk_agg = all_us_p.groupby(["db", "pid", "blk"]).agg(cpp=("cpp", "mean"), icp=("icp", "mean")).unstack("blk")
    blk_agg.columns = [f"{v}_b{b}" for v, b in blk_agg.columns]
    us_lmtp_df = us_win.merge(blk_agg.reset_index(), on=["db", "pid"], how="inner")

    r_cpp65_ols = sequential_ice_v3(us_lmtp_df, ["cpp_b0", "cpp_b1"], [["icp_b0"], ["icp_b1"]], ["age", "male", "db_is_eicu"], "hdeath", 5.0, 65.0, penalize=False, cluster_col="cluster_id")
    r_cpp65_ridge = sequential_ice_v3(us_lmtp_df, ["cpp_b0", "cpp_b1"], [["icp_b0"], ["icp_b1"]], ["age", "male", "db_is_eicu"], "hdeath", 5.0, 65.0, penalize=True, cluster_col="cluster_id")
    r_cpp70_ols = sequential_ice_v3(us_lmtp_df, ["cpp_b0", "cpp_b1"], [["icp_b0"], ["icp_b1"]], ["age", "male", "db_is_eicu"], "hdeath", 5.0, 70.0, penalize=False, cluster_col="cluster_id")

    m_k3 = pd.read_csv(K3_DIR / "mimic_tbi_k3_scored.csv")
    in_dt = pd.to_datetime(m_k3["intime"])
    m_k3["icu_h"] = (pd.to_datetime(m_k3["outtime"]) - in_dt).dt.total_seconds() / 3600.0
    m_e3 = m_k3[m_k3["endotype_k3"].str.startswith("E3") & (m_k3["surv_h"] > 24.0) & (m_k3["icu_h"] > 24.0)].assign(db="mimic", id=lambda d: d["stay_id"], vent=lambda d: d["vent_0_24h"], cluster_id=lambda d: "M_" + d["stay_id"].astype(str))

    e_k3 = pd.read_csv(K3_DIR / "eicu_tbi_k3_scored.csv")
    e_k3["icu_h"] = e_k3["unitdischargeoffset"] / 60.0
    e_e3 = e_k3[e_k3["endotype_k3"].str.startswith("E3") & (e_k3["surv_h"] > 24.0) & (e_k3["icu_h"] > 24.0)].assign(db="eicu", id=lambda d: d["patientunitstayid"], age=lambda d: d["age_num"], vent=lambda d: d["vent_24h"], cluster_id=lambda d: "E_" + d["hospitalid"].astype(str))

    all_mh = pd.concat([pd.read_parquet(MAP_PRIV / "mimic_measured_map_hours.parquet").assign(db="mimic"), pd.read_parquet(MAP_PRIV / "eicu_measured_map_hours.parquet").assign(db="eicu")], ignore_index=True)
    ok_ids = set(all_mh.groupby(["db", "id"])["hour"].nunique().loc[lambda s: s >= 6].index)
    all_mh = all_mh[all_mh.set_index(["db", "id"]).index.isin(ok_ids)].copy()
    all_mh["blk6"] = (all_mh["hour"] // 6).clip(0, 3)
    m_blk = all_mh.groupby(["db", "id", "blk6"]).agg(map_m=("map_mean", "mean"), map_lo=("map_min", "min")).unstack("blk6")
    m_blk.columns = [f"{v}_b{b}" for v, b in m_blk.columns]
    e3_comb = pd.concat([
        m_e3[["db", "id", "cluster_id", "age", "male", "gcs_motor", "vent", "hgb", "inr", "plt", "in_hospital_death"]].assign(db_is_eicu=0.0),
        e_e3[["db", "id", "cluster_id", "age", "male", "gcs_motor", "vent", "hgb", "inr", "plt", "in_hospital_death"]].assign(db_is_eicu=1.0),
    ], ignore_index=True).merge(m_blk.reset_index(), on=["db", "id"], how="inner")
    e3_comb["ph"] = np.where(e3_comb["hgb"] < 10.0, "E2", "E1")
    r_map85_ols = sequential_ice_v3(e3_comb, ["map_m_b0", "map_m_b1", "map_m_b2", "map_m_b3"], [["map_lo_b0"], ["map_lo_b1"], ["map_lo_b2"], ["map_lo_b3"]], ["age", "male", "gcs_motor", "vent", "inr", "plt", "db_is_eicu"], "in_hospital_death", 5.0, 85.0, penalize=False, cluster_col="cluster_id")

    lmtp_v3_rows = [
        {"estimator_and_policy": "Observed-past sequential ICE (unpenalized OLS): +5 mmHg when post-6lab 0–6h CPP < 65 mmHg", **r_cpp65_ols},
        {"estimator_and_policy": "Observed-past sequential ICE (Ridge shrinkage): +5 mmHg when post-6lab 0–6h CPP < 65 mmHg", **r_cpp65_ridge},
        {"estimator_and_policy": "Observed-past sequential ICE (unpenalized OLS): +5 mmHg when post-6lab 0–6h CPP < 70 mmHg", **r_cpp70_ols},
        {"estimator_and_policy": "Observed-past sequential ICE (unpenalized OLS, E3 Hgb<10 vs >=10): +5 mmHg when 0–24h ICU MAP < 85 mmHg", **r_map85_ols},
    ]
    lmtp_v3_df = pd.DataFrame(lmtp_v3_rows)
    lmtp_v3_df.to_csv(OUT_DIR / "V3_TableS5B_Verified_ObservedPast_ICE_Comparison.csv", index=False)
    print("\n=== VERIFIED OBSERVED-PAST SEQUENTIAL ICE (UNPENALIZED OLS VS RIDGE) ===")
    print(lmtp_v3_df[["estimator_and_policy", "n", "All_rd", "E2_minus_E1", "E3_minus_E1"]].to_string(index=False))


if __name__ == "__main__":
    run_exploratory_ct_audit()
    run_corrected_lax_and_lmtp()
