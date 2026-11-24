#!/usr/bin/env python3
"""
ICM v7 Editorial Visual System — True 180-mm Double-Column Publication Figures + Visual Abstract:
Incorporates all cross-adjudicated visual & statistical upgrades from Codex CLI + DeepSeek V4 Pro:
  1. Reads unified v7 tables (SICdb INR (0.70, 1.20] locked identically at baseline and 48h/72h + separate US blood-component models).
  2. Fig 1:
     - (a) Marks E1->E2 surge as exploratory.
     - (c) Displays all 5 external eICU benchmark models with explicit sample sizes (Clinical base 0.857, + 3 states 0.866,
           + Posterior probs 0.869, + 12-df splines 0.876 [n=3,640], and Folweiler vs ours 0.866/0.865 [n=3,550]).
     - (d) Explicitly scopes equal median motor GCS (6 in US, 4 in HiRID; SAPS 3 in SICdb).
  3. Fig 2:
     - (a) Strictly proportional alluvial bar & ribbon heights (zero artificial minimum height inflation on Shifted E3->E2 n=17),
           with non-overlapping right-hand typography and updated 4-cohort fresh resolution rates (30.0% / 29.7% / 22.6% / 34.4%).
     - (b) Displays all 6 post-48h bars in 3 matched comparator pairs, including US Stable E1->E1 (45/570, 7.9%) alongside
           exploratory De novo E1->E2* (9/32, 28.1%), separated cleanly from 3-cohort E1->E3 and E3->E1 contrasts.
     - (d) Enlarged ~5.95-pt forest-plot typography, pre-specified E1->E3 before exploratory E1->E2*, and both binary any-blood
           (aOR 0.47) and separate component-level blood adjustment (aOR 0.51).
  4. Fig 3:
     - (a) Explicit 'Fibrinogen <200 mg/dL (%)' y-axis and '*SICdb cont. Fib raw P=0.69' sub-track label.
     - (c, d) Enlarged ~5.95-pt tabular forest-plot typography; marks E3 0–24h MAP overlap ('*') in (d);
       replaces unsupported LAx interaction P claim in bottom banner with exact HiRID LAx & eICU low-CPP findings.
  5. Visual Abstract:
     - Adds explicit cohort tags to all metrics, scopes equal median motor GCS to US/HiRID, replaces 'fewer transfusions'
       with 'lower recorded 0–48h blood exposure', notes day-1 clinical covariates in spline model, and aligns LAx/CPP text.
"""
import os
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(os.environ.get("TBI_ROOT", "./data_workspace"))
K3_DIR = ROOT / "derived/closed_loop_v1/clinical_k3"
P1_DIR = ROOT / "derived/closed_loop_v1/revision_gatekeepers_v1/p1_endpoints_and_sicdb"
P2_DIR = ROOT / "derived/closed_loop_v1/revision_gatekeepers_v1/p2_competing_explanations"
HIR_DIR = ROOT / "derived/closed_loop_v1/transatlantic_4db_v1/hirid"

V7_DIR = ROOT / "icm_submission_v7_20261124"
TAB_DIR = V7_DIR / "tables"
FIG_DIR = V7_DIR / "figures"
VA_DIR = V7_DIR / "visual_abstract"
QA_DIR = V7_DIR / "qa_visual_proofs"
ART_DIR = Path(os.environ.get("TBI_ARTIFACTS_DIR", "./figures_preview"))

for d in (TAB_DIR, FIG_DIR, VA_DIR, QA_DIR):
    d.mkdir(parents=True, exist_ok=True)

MM_TO_IN = 1.0 / 25.4
W_180MM = 180.0 * MM_TO_IN  # 7.086614 in exact

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.2,
    "axes.titlesize": 7.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 6.3,
    "xtick.labelsize": 5.9,
    "ytick.labelsize": 5.9,
    "legend.fontsize": 5.9,
    "axes.linewidth": 0.65,
    "axes.edgecolor": "#334155",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

C_E1 = "#1B5E8A"       # Deep Steel Blue (State E1)
C_E2 = "#D55E00"       # Okabe-Ito Vermilion (State E2)
C_E3 = "#6A2C70"       # Royal Plum (State E3)
C_TEAL = "#0D7A68"     # Deep Clinical Teal (Resolved E3 -> E1)
C_CRIMSON = "#991B1B"  # Early death / Conflated comparator
C_SLATE = "#64748B"    # Neutral muted slate (Attenuated / Non-sig)
C_LIGHT = "#CBD5E1"
C_INK = "#0F172A"
C_SUB = "#475569"
C_GRID = "#E8ECF1"


def fmt_sci_p(p_val: float, sig_digits: int = 2) -> str:
    if p_val < 1e-80:
        return r"<$10^{-80}$"
    if p_val >= 0.001:
        return f"={p_val:.3f}"
    exp = int(np.floor(np.log10(p_val)))
    mant = p_val / (10.0 ** exp)
    return rf"={mant:.{sig_digits - 1}f}×$10^{{{exp}}}$"


def wilson_ci(k: int, n: int):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959964
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = (z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / den
    return 100.0 * p, 100.0 * max(0.0, ctr - half), 100.0 * min(1.0, ctr + half)


def parse_rd_ci(s: str):
    m = re.match(r"([+\-]?[0-9\.]+)\s*\(([+\-]?[0-9\.]+)\s*to\s*([+\-]?[0-9\.]+)\)", str(s).strip())
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_or_ci(s: str):
    m = re.match(r"([0-9\.]+)\s*\(([0-9\.]+)[–\-]([0-9\.]+)\)", str(s).strip())
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_med_iqr(s: str):
    m = re.match(r"([0-9\.]+)\s*\(([0-9\.]+)[–\-]([0-9\.]+)\)", str(s).strip())
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def panel_header(fig, x_fig: float, y_fig: float, letter: str, title: str):
    fig.text(x_fig, y_fig, letter, fontsize=9.2, fontweight="bold", color=C_INK, ha="left", va="baseline")
    fig.text(x_fig + 0.022, y_fig, title, fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="baseline")


def save_editorial(fig, out_dir: Path, stem: str):
    pdf_p = out_dir / f"{stem}.pdf"
    svg_p = out_dir / f"{stem}.svg"
    png_600 = out_dir / f"{stem}.png"
    fig.savefig(pdf_p)
    fig.savefig(svg_p)
    fig.savefig(png_600, dpi=600)
    plt.close(fig)
    shutil.copy2(png_600, ART_DIR / png_600.name)


def draw_bezier_ribbon(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.30):
    xm = 0.5 * (x0 + x1)
    verts = [
        (x0, y0_bot),
        (xm, y0_bot), (xm, y1_bot), (x1, y1_bot),
        (x1, y1_top),
        (xm, y1_top), (xm, y0_top), (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    patch = PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=2)
    ax.add_patch(patch)


def main():
    # Copy benchmark CSV for reproducible archival in TAB_DIR (do NOT overwrite v7 D1/D2/D3!)
    shutil.copy2(P2_DIR / "P2_1_frozen_prediction_and_folweiler_benchmark.csv",
                 TAB_DIR / "P2_1_frozen_prediction_and_folweiler_benchmark.csv")

    m_df = pd.read_csv(K3_DIR / "mimic_tbi_k3_scored.csv"); m_df["ph"] = m_df["endotype_k3"].str[:2]
    e_df = pd.read_csv(K3_DIR / "eicu_tbi_k3_scored.csv"); e_df["ph"] = e_df["endotype_k3"].str[:2]
    s_df = pd.read_csv(P1_DIR / "sicdb_icuoffset_censoredCDF_scored.csv"); s_df["ph"] = s_df["phenotype"].str[:2]
    h_df = pd.read_csv(HIR_DIR / "hirid_true_neurotrauma_tbi_k3_scored.csv"); h_df["ph"] = h_df["endotype"].str[:2]

    bench_df = pd.read_csv(TAB_DIR / "P2_1_frozen_prediction_and_folweiler_benchmark.csv").set_index("model")
    flow_df = pd.read_csv(TAB_DIR / "V3_TableD1_MutuallyExclusive_ICU_Landmark_Flow.csv")
    surv_df = pd.read_csv(TAB_DIR / "V3_TableD2_Strict_InICU_Landmark_Survival_Models.csv")
    bp_df = pd.read_csv(TAB_DIR / "V3_TableD3_US_BloodProduct_Exposure_and_Adjusted_Models.csv")
    ecg_df = pd.read_csv(TAB_DIR / "V3_TableB1_MIMIC_ECG_Confounding_and_Timing_Audit.csv")
    bio_df = pd.read_csv(TAB_DIR / "V3_TableB2_4DB_Biomarkers_FullDenominators_and_Holm.csv")
    lmtp_df = pd.read_csv(TAB_DIR / "V3_TableS5B_Verified_ObservedPast_ICE_Comparison.csv")

    # =========================================================================
    # FIGURE 1 (180 mm × 154 mm)
    # =========================================================================
    fig1 = plt.figure(figsize=(W_180MM, 154.0 * MM_TO_IN), facecolor="white")

    panel_header(fig1, 0.015, 0.972, "a", "Four-cohort zero-refit validation & 48h ICU landmark design (N = 8,130)")
    axa = fig1.add_axes([0.015, 0.825, 0.970, 0.135])
    axa.set_xlim(0, 100); axa.set_ylim(0, 24); axa.axis("off")

    def card(ax, x, y, w, h, hdr_col, title_s, line1, line2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.1",
                                    fc="#F8FAFC", ec=hdr_col, lw=0.85, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y + h - 5.2), w, 5.2, boxstyle="round,pad=0.1,rounding_size=0.9",
                                    fc=hdr_col, ec="none", zorder=3))
        ax.text(x + w / 2, y + h - 2.7, title_s, color="white", fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(x + w / 2, y + 8.6, line1, color=C_INK, fontsize=5.85, fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(x + w / 2, y + 3.4, line2, color=C_SUB, fontsize=5.65, ha="center", va="center", zorder=4)

    card(axa, 0.4, 2.0, 21.4, 19.5, C_E1, "1. Derivation (Locked GMM)", "MIMIC-IV · USA (n = 3,483)", "6 routine tests [-24h, +6h] · ARI 0.772")
    axa.annotate("", xy=(26.8, 11.2), xytext=(22.4, 11.2), arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.45", lw=1.0, color=C_INK))
    axa.text(24.6, 15.0, "0-refit", fontsize=5.65, fontweight="bold", color=C_INK, ha="center")

    card(axa, 27.4, 2.0, 44.2, 19.5, "#334155", "2. Zero-Refit External Transport (n = 4,647)",
         "eICU-CRD (n=3,640) · SICdb (n=516) · HiRID proxy (n=491)",
         "Separated endpoints: Hospital · Verified 1-year · ICU death")
    axa.annotate("", xy=(76.4, 11.2), xytext=(72.2, 11.2), arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.45", lw=1.0, color=C_TEAL))
    axa.text(74.3, 15.0, "48h", fontsize=5.65, fontweight="bold", color=C_TEAL, ha="center")

    card(axa, 77.0, 2.0, 22.6, 19.5, C_TEAL, "3. 48h ICU Retesting", "Fresh (24, 48] h 6-lab panel", "E3→E1 recovery · E1→E2* surge")

    # Panel b: Six-test multi-cohort horizontal dot-range profile
    panel_header(fig1, 0.015, 0.794, "b", "Conserved six-test signatures across 4 cohorts")
    axb = fig1.add_axes([0.118, 0.470, 0.378, 0.275])
    feats = ["wbc", "plt", "hgb", "inr", "bicarb", "glu"]
    f_lbls = ["WBC", "Platelets", "Haemoglobin", "INR*", "Bicarbonate", "Glucose"]
    m_med = m_df[feats].median()
    m_iqr = (m_df[feats].quantile(0.75) - m_df[feats].quantile(0.25)).replace(0, 1.0)
    s_plot = s_df.copy(); s_plot["inr"] = s_plot["inr_raw"]
    cohorts_b = [("MIMIC", m_df, "o"), ("eICU", e_df, "s"), ("SICdb", s_plot, "^"), ("HiRID", h_df, "D")]

    y6 = np.arange(len(feats))[::-1]
    offsets_state = {"E1": +0.20, "E2": 0.0, "E3": -0.20}
    cols_state = {"E1": C_E1, "E2": C_E2, "E3": C_E3}
    axb.axvline(0.0, color="#94A3B8", lw=0.75, ls="--", zorder=1)
    for f_i, f_col in enumerate(feats):
        yi = y6[f_i]
        if f_i % 2 == 1:
            axb.axhspan(yi - 0.44, yi + 0.44, color="#F8FAFC", zorder=0)
        for st in ("E1", "E2", "E3"):
            z_vals = [float((cdf[cdf["ph"] == st][f_col].median() - m_med[f_col]) / m_iqr[f_col]) for _, cdf, _ in cohorts_b]
            yy = yi + offsets_state[st]
            axb.plot([min(z_vals), max(z_vals)], [yy, yy], color=cols_state[st], lw=1.35, alpha=0.65, zorder=2)
            for (_, _, mk), zv in zip(cohorts_b, z_vals):
                axb.scatter(zv, yy, marker=mk, s=15, color=cols_state[st], edgecolors="white", linewidths=0.35, zorder=3)

    axb.set_yticks(y6); axb.set_yticklabels(f_lbls, fontweight="bold")
    axb.set_xlim(-1.62, 1.82)
    axb.set_xlabel("Standardised median (MIMIC IQR units)")
    axb.grid(axis="x", color=C_GRID, lw=0.5)

    axb_note = fig1.add_axes([0.500, 0.470, 0.095, 0.275])
    axb_note.set_xlim(0, 1); axb_note.set_ylim(axb.get_ylim()); axb_note.axis("off")
    axb_note.text(0.0, 5.48, "MIMIC med.", fontsize=5.7, fontweight="bold", color=C_SUB)
    unit_notes = [
        "10.5 / 16.3 / 8.5",
        "208 / 302 / 144",
        "12.6 / 12.5 / 10.2",
        "1.1 / 1.1 / 1.5",
        "24 / 22 / 23",
        "119 / 172 / 133",
    ]
    for yi, u_txt in zip(y6, unit_notes):
        axb_note.text(0.0, yi, u_txt, fontsize=5.7, color=C_INK, va="center")

    axb_key = fig1.add_axes([0.118, 0.756, 0.390, 0.024])
    axb_key.set_xlim(0, 100); axb_key.set_ylim(0, 10); axb_key.axis("off")
    for xk, st_lbl, st_c in [(0.0, "E1", C_E1), (11.5, "E2", C_E2), (23.0, "E3", C_E3)]:
        axb_key.plot([xk, xk + 3.6], [5.0, 5.0], color=st_c, lw=2.2)
        axb_key.text(xk + 4.8, 5.0, st_lbl, fontsize=5.65, fontweight="bold", color=st_c, va="center")
    axb_key.text(35.0, 5.0, "|", fontsize=5.7, color="#94A3B8", va="center")
    for xk, c_lbl, mk_c in [(39.5, "MIMIC", "o"), (56.5, "eICU", "s"), (71.5, "SICdb", "^"), (87.5, "HiRID", "D")]:
        axb_key.scatter(xk, 5.0, marker=mk_c, s=14, color="#334155", zorder=3)
        axb_key.text(xk + 2.5, 5.0, c_lbl, fontsize=5.65, color=C_SUB, va="center")

    # Panel c: External calibration & complete 5-row benchmark key (including Posterior probs 0.869 & sample sizes)
    panel_header(fig1, 0.608, 0.794, "c", "External calibration & AUC (eICU)")
    axc = fig1.add_axes([0.662, 0.470, 0.318, 0.275])
    m_fit = m_df.copy()
    for c in ["age", "male", "gcs_motor", "vent_0_24h"]:
        m_fit[c] = pd.to_numeric(m_fit[c], errors="coerce").fillna(m_fit[c].median())
    m_fit["is_E2"] = (m_fit["ph"] == "E2").astype(float)
    m_fit["is_E3"] = (m_fit["ph"] == "E3").astype(float)
    Xm = sm.add_constant(m_fit[["age", "male", "gcs_motor", "vent_0_24h", "is_E2", "is_E3"]].astype(float))
    mod_m1 = sm.Logit(m_fit["in_hospital_death"].astype(float), Xm).fit(disp=False)

    e_cal = e_df.copy()
    e_cal["age"] = pd.to_numeric(e_cal["age_num"], errors="coerce").fillna(e_cal["age_num"].median())
    e_cal["male"] = pd.to_numeric(e_cal["male"], errors="coerce").fillna(e_cal["male"].median())
    e_cal["gcs_motor"] = pd.to_numeric(e_cal["gcs_motor"], errors="coerce").fillna(e_cal["gcs_motor"].median())
    e_cal["vent_0_24h"] = pd.to_numeric(e_cal["vent_24h"], errors="coerce").fillna(e_cal["vent_24h"].median())
    e_cal["is_E2"] = (e_cal["ph"] == "E2").astype(float)
    e_cal["is_E3"] = (e_cal["ph"] == "E3").astype(float)
    Xe = sm.add_constant(e_cal[["age", "male", "gcs_motor", "vent_0_24h", "is_E2", "is_E3"]].astype(float))
    e_cal["pred"] = mod_m1.predict(Xe)
    e_cal["dec"] = pd.qcut(e_cal["pred"], 10, labels=False, duplicates="drop")
    dec_df = e_cal.groupby("dec").agg(n=("in_hospital_death", "size"), k=("in_hospital_death", "sum"), mp=("pred", "mean")).reset_index()
    obs_p, lo_p, hi_p = zip(*[wilson_ci(int(r.k), int(r.n)) for _, r in dec_df.iterrows()])
    pred_p = 100.0 * dec_df["mp"].to_numpy()
    obs_p = np.asarray(obs_p)

    axc.plot([0, 68.5], [0, 68.5], ls="--", color="#94A3B8", lw=0.85, zorder=1)
    axc.errorbar(pred_p, obs_p, yerr=[obs_p - np.asarray(lo_p), np.asarray(hi_p) - obs_p],
                 fmt="o-", color=C_E1, lw=1.35, markersize=3.7, capsize=2.0, zorder=3)
    axc.set_xlim(0, 68.5); axc.set_ylim(0, 68.5)
    axc.set_xlabel("Predicted mortality, frozen MIMIC (%)")
    axc.set_ylabel("Observed eICU mortality (%)")
    axc.grid(color=C_GRID, lw=0.5)

    axc.text(2.2, 63.2, "Slope = 1.045 (10 deciles)", fontsize=5.95, fontweight="bold", color=C_E1)
    bench_lines = [
        ("Clin. base (3,640)", f"{bench_df.loc['M0_Frozen_Clinical_Baseline', 'auc']:.3f}"),
        ("+ 3 states (3,640)", f"{bench_df.loc['M1_Baseline_plus_3Class_Phenotypes', 'auc']:.3f}"),
        ("+ Post. probs (3,640)", f"{bench_df.loc['M2_Baseline_plus_Posterior_Probs', 'auc']:.3f}"),
        ("+ 12-df splines (3,640)", f"{bench_df.loc['M4_Baseline_plus_6Lab_CubicSplines', 'auc']:.3f}"),
        ("Folweiler/ours (3,550)", f"{bench_df.loc['Folweiler_6Lab_K3', 'auc']:.3f}/{bench_df.loc['Ours_6Lab_K3_SameSubset', 'auc']:.3f}"),
    ]
    axc.text(29.8, 22.8, "Model (n)", fontsize=5.65, fontweight="bold", color=C_SUB)
    axc.text(67.3, 22.8, "AUC", fontsize=5.65, fontweight="bold", color=C_SUB, ha="right")
    for idx_b, (m_n, a_v) in enumerate(bench_lines):
        yy = 18.5 - idx_b * 4.15
        axc.text(29.8, yy, m_n, fontsize=5.6, color=C_INK)
        axc.text(67.3, yy, a_v, fontsize=5.6,
                 fontweight="bold" if ("3 states" in m_n or "splines" in m_n) else "normal",
                 color=C_INK, ha="right")

    # Panel d: Hero separated mortality dot-interval + slim stem plot
    panel_header(fig1, 0.015, 0.382, "d", "Separated mortality gradients (equal median motor GCS across E1 & E3: 6 in US, 4 in HiRID; SAPS 3 in SICdb)")
    axd = fig1.add_axes([0.078, 0.068, 0.902, 0.262])
    s_1yr = s_df[s_df["obs_ge_365d"] == 1]
    s1y_counts = [(int(s_1yr[s_1yr["ph"] == p]["died_by_365d"].sum()), int((s_1yr["ph"] == p).sum())) for p in ["E1", "E2", "E3"]]
    groups_meta = [
        ("MIMIC-IV\nHospital\n(n=3,483)", [(144, 2111), (89, 546), (195, 826)]),
        ("eICU-CRD\nHospital\n(n=3,640)", [(160, 2274), (120, 576), (196, 790)]),
        ("SICdb\nHospital\n(n=516)", [(16, 403), (5, 40), (18, 73)]),
        ("SICdb\n1-Year\n(n=457)", s1y_counts),
        ("MIMIC-IV\nICU\n(n=3,481)", [(78, 2110), (62, 546), (133, 825)]),
        ("eICU-CRD\nICU\n(n=3,640)", [(87, 2274), (76, 576), (136, 790)]),
        ("SICdb\nICU known\n(n=478)", [(8, 375), (4, 36), (11, 67)]),
        ("HiRID proxy\nICU only\n(n=491)", [(35, 291), (16, 41), (49, 159)]),
    ]
    xg = np.arange(len(groups_meta))
    st_specs = [
        ("State E1 (Compensated ref.)", C_E1, "o", -0.24),
        ("State E2 (Hypermetabolic)", C_E2, "^", 0.00),
        ("State E3 (Coagulopathic)", C_E3, "D", +0.24),
    ]
    for k_i, (lbl_s, col_s, mk_s, dx_s) in enumerate(st_specs):
        rates, los, his = [], [], []
        for _, cells in groups_meta:
            r, lo, hi = wilson_ci(cells[k_i][0], cells[k_i][1])
            rates.append(r); los.append(r - lo); his.append(hi - r)
        pos = xg + dx_s
        axd.bar(pos, rates, width=0.19, color=col_s, alpha=0.22, edgecolor="none", zorder=2)
        axd.errorbar(pos, rates, yerr=[los, his], fmt=mk_s, color=col_s, markersize=4.3,
                     ecolor=col_s, elinewidth=1.05, capsize=2.0, label=lbl_s, zorder=4)
        for g_idx, (xi, ri, hi_e) in enumerate(zip(pos, rates, his)):
            x_nudge = -0.04 if (g_idx == 6 and k_i == 1) else (+0.05 if (g_idx == 6 and k_i == 2) else 0.0)
            y_nudge = 2.0 if (g_idx == 6 and k_i == 2) else 1.2
            axd.text(xi + x_nudge, ri + hi_e + y_nudge, f"{ri:.1f}", ha="center", va="bottom",
                     fontsize=5.7, fontweight="bold" if k_i == 2 else "normal", color=C_INK)

    axd.axvline(3.5, color="#64748B", lw=0.9, ls="--", zorder=3)
    axd.text(1.5, 62.5, "Hospital & Verified 1-Year Mortality (%)", ha="center", va="top", fontsize=6.1, fontweight="bold", color=C_INK)
    axd.text(5.55, 62.5, "ICU Mortality Analyzed Separately (%)", ha="center", va="top", fontsize=6.1, fontweight="bold", color=C_INK)
    axd.set_xticks(xg); axd.set_xticklabels([g[0] for g in groups_meta], fontsize=5.8)
    axd.set_ylim(0, 66); axd.set_ylabel("Crude mortality (%, 95% CI)")
    axd.grid(axis="y", color=C_GRID, lw=0.5)
    axd.legend(loc="upper right", bbox_to_anchor=(0.995, 1.16), ncol=3, frameon=False, handletextpad=0.35, columnspacing=1.1, fontsize=5.9)

    save_editorial(fig1, FIG_DIR, "ICM_v7_Fig1_Baseline_Invariance_Calibration_and_Mortality")

    # =========================================================================
    # FIGURE 2 (180 mm × 156 mm)
    # =========================================================================
    fig2 = plt.figure(figsize=(W_180MM, 156.0 * MM_TO_IN), facecolor="white")

    # Panel a: Strictly Proportional Alluvial Transition Ribbon (ZERO artificial height inflation!)
    panel_header(fig2, 0.015, 0.960, "a", "48h ICU disposition & transition flow (Admission E3, n = 1,848)")
    ax2a = fig2.add_axes([0.015, 0.555, 0.525, 0.382])
    dbs4 = ["MIMIC-IV", "eICU-CRD", "SICdb", "HiRID proxy"]
    f48_e3 = flow_df[(flow_df["landmark_h"] == 48) & (flow_df["baseline_group"] == "E3")].set_index("database").loc[dbs4]

    tot_e3 = float(f48_e3["n_baseline"].sum())  # 1,848
    n_fresh_tot = int(f48_e3["in_icu_all6_fresh_in_window_n"].sum())  # 637
    n_died = int(f48_e3["died_at_or_before_landmark"].sum())
    n_exit = int(f48_e3["alive_at_landmark_exited_icu_le_landmark"].sum())
    n_lt6 = int((f48_e3["remained_in_icu_at_landmark"] - f48_e3["in_icu_all6_fresh_in_window_n"]).sum())
    n_pe3 = int(f48_e3["all6_to_E3_n"].sum())
    n_se2 = int(f48_e3["all6_to_E2_n"].sum())
    n_re1 = int(f48_e3["all6_to_E1_n"].sum())

    res_rates_str = " / ".join(f"{float(f48_e3.loc[d, 'all6_to_E1_pct']):.1f}%" for d in dbs4) + " of fresh"
    nodes_r = [
        ("Resolved E3 → E1", n_re1, C_TEAL, res_rates_str),
        ("Shifted E3 → E2", n_se2, C_E2, f"{100*n_se2/n_fresh_tot:.1f}% of fresh (n={n_se2})"),
        ("Persistent E3 → E3", n_pe3, C_E3, f"{100*n_pe3/n_fresh_tot:.1f}% of fresh (n={n_pe3})"),
        ("In ICU, <6 fresh labs", n_lt6, "#64748B", f"{100*n_lt6/tot_e3:.1f}% of E3 (n={n_lt6})"),
        ("Alive, left ICU ≤48h", n_exit, "#94A3B8", f"{100*n_exit/tot_e3:.1f}% of E3 (n={n_exit})"),
        ("Died ≤48h", n_died, C_CRIMSON, f"{100*n_died/tot_e3:.1f}% of E3 (n={n_died})"),
    ]
    ax2a.set_xlim(0, 1.0); ax2a.set_ylim(0, 114); ax2a.axis("off")

    src_y0, src_h = 21.0, 72.0
    ax2a.add_patch(Rectangle((0.010, src_y0), 0.160, src_h, fc=C_E3, ec="none", zorder=3))
    ax2a.text(0.090, src_y0 + src_h + 3.0, "Admission E3\n(n = 1,848)", ha="center", va="bottom",
              fontsize=6.0, fontweight="bold", color=C_E3)
    cohort_Left_txt = "MIMIC 826\neICU 790\nSICdb 73\nHiRID 159"
    ax2a.text(0.090, src_y0 + src_h / 2, cohort_Left_txt, ha="center", va="center",
              fontsize=5.65, color="white", fontweight="bold", linespacing=1.35, zorder=4)

    # Strictly proportional heights (h_seg = total_r_h * cnt / tot_e3 for EVERY node — zero artificial inflation!)
    gaps = [7.8, 4.8, 4.2, 3.6, 3.6]
    total_r_h = 65.0
    cur_src_top = src_y0 + src_h
    cur_dst_top = 104.0
    for idx_n, (n_lbl, cnt, col_n, sub_s) in enumerate(nodes_r):
        h_seg = total_r_h * (cnt / tot_e3)
        h_src = src_h * (cnt / tot_e3)
        y_s_top = cur_src_top
        y_s_bot = cur_src_top - h_src
        y_d_top = cur_dst_top
        y_d_bot = cur_dst_top - h_seg

        draw_bezier_ribbon(ax2a, 0.170, y_s_bot, y_s_top, 0.405, y_d_bot, y_d_top,
                           col_n, alpha=0.45 if idx_n in (0, 2) else 0.28)
        ax2a.add_patch(Rectangle((0.405, y_d_bot), 0.042, h_seg, fc=col_n, ec="none", zorder=3))
        ym = 0.5 * (y_d_top + y_d_bot)
        ax2a.text(0.462, ym + (1.9 if idx_n == 0 else 0.0),
                  f"{n_lbl} (n={cnt})" if idx_n == 0 else f"{n_lbl}",
                  va="center", ha="left", fontsize=5.85,
                  fontweight="bold" if idx_n in (0, 2) else "normal",
                  color=col_n if idx_n == 0 else C_INK)
        if idx_n == 0:
            ax2a.text(0.462, ym - 3.3, sub_s, va="center", ha="left", fontsize=5.6, color=C_TEAL, fontweight="bold")
        else:
            ax2a.text(0.995, ym, sub_s, va="center", ha="right", fontsize=5.6, color=C_SUB)

        cur_src_top = y_s_bot
        if idx_n < len(gaps):
            cur_dst_top = y_d_bot - gaps[idx_n]

    ax2a.plot([0.010, 0.995], [10.8, 10.8], color=C_GRID, lw=0.8)
    ax2a.text(0.010, 4.2, "Fresh (24, 48] h 6-lab ICU residents: n = 637 (MIMIC 320 · eICU 222 · SICdb 31 · HiRID 64)",
              fontsize=5.65, color=C_SUB, va="center")

    # Panel b: Bidirectional post-48h hospital mortality — 6 bars in 3 exact matched comparator pairs!
    panel_header(fig2, 0.565, 0.960, "b", "Bidirectional post-48h hospital death")
    ax2b = fig2.add_axes([0.612, 0.592, 0.372, 0.332])
    s48_fresh = surv_df[(surv_df["landmark_h"] == 48) & (surv_df["rule_code"] == "strict_icu_all6_fresh") & (surv_df["database"].isin(["MIMIC-IV", "eICU-CRD", "SICdb"]))]

    def sum_counts(contrast_prefix: str, col_name: str, dbs_filter=None):
        sub_c = s48_fresh[s48_fresh["contrast"].str.startswith(contrast_prefix)]
        if dbs_filter is not None:
            sub_c = sub_c[sub_c["database"].isin(dbs_filter)]
        d_sum, n_sum = 0, 0
        for s_val in sub_c[col_name]:
            m = re.match(r"(\d+)/(\d+)", str(s_val))
            if m:
                d_sum += int(m.group(1)); n_sum += int(m.group(2))
        return d_sum, n_sum

    d_e1e1_k3, n_e1e1_k3 = sum_counts("De novo E1 -> E3", "ref_deaths_N")
    d_e1e3_k3, n_e1e3_k3 = sum_counts("De novo E1 -> E3", "trans_deaths_N")
    d_e1e1_us, n_e1e1_us = sum_counts("De novo E1 -> E2", "ref_deaths_N", ["MIMIC-IV", "eICU-CRD"])
    d_e1e2_us, n_e1e2_us = sum_counts("De novo E1 -> E2", "trans_deaths_N", ["MIMIC-IV", "eICU-CRD"])
    d_e3e3_k3, n_e3e3_k3 = sum_counts("Resolved E3 -> E1", "ref_deaths_N")
    d_e3e1_k3, n_e3e1_k3 = sum_counts("Resolved E3 -> E1", "trans_deaths_N")

    bars_b = [
        ("Stable\nE1→E1", d_e1e1_k3, n_e1e1_k3, C_E1, "o"),
        ("De novo\nE1→E3", d_e1e3_k3, n_e1e3_k3, C_SLATE, "D"),
        ("US Ref\nE1→E1", d_e1e1_us, n_e1e1_us, C_E1, "o"),
        ("De novo\nE1→E2*", d_e1e2_us, n_e1e2_us, C_E2, "^"),
        ("Persist.\nE3→E3", d_e3e3_k3, n_e3e3_k3, C_E3, "D"),
        ("Resolved\nE3→E1", d_e3e1_k3, n_e3e1_k3, C_TEAL, "s"),
    ]
    xb = np.arange(len(bars_b))
    for i, (lbl, dk, nk, col, mk) in enumerate(bars_b):
        r, lo, hi = wilson_ci(dk, nk)
        ax2b.bar(i, r, width=0.46, color=col, alpha=0.22, zorder=2)
        ax2b.errorbar(i, r, yerr=[[r - lo], [hi - r]], fmt=mk, color=col, markersize=4.4,
                      ecolor=col, lw=1.15, capsize=2.1, zorder=4)
        ax2b.text(i, hi + 1.4, f"{r:.1f}%\n{dk}/{nk}", ha="center", va="bottom",
                  fontsize=5.6, fontweight="bold" if i in (3, 5) else "normal", color=C_INK, linespacing=1.10)
    ax2b.axvline(1.5, color="#CBD5E1", ls=":", lw=0.85)
    ax2b.axvline(3.5, color="#64748B", ls="--", lw=0.85)
    ax2b.text(0.5, 53.0, "E1→E3 (k=3)", ha="center", fontsize=5.65, fontweight="bold", color=C_SUB)
    ax2b.text(2.5, 53.0, "US E1→E2*", ha="center", fontsize=5.65, fontweight="bold", color=C_E2)
    ax2b.text(4.5, 53.0, "E3→E1 (k=3)", ha="center", fontsize=5.65, fontweight="bold", color=C_TEAL)
    ax2b.set_xticks(xb); ax2b.set_xticklabels([b[0] for b in bars_b], fontsize=5.65)
    ax2b.set_xlim(-0.58, 5.58); ax2b.set_ylim(0, 58)
    ax2b.set_ylabel("Post-48h hospital mortality (%)")
    ax2b.grid(axis="y", color=C_GRID, lw=0.5)

    # Panel c: Paired Cleveland Dumbbell of 0–48h blood-product exposure
    panel_header(fig2, 0.015, 0.495, "c", "0–48h blood-product exposure")
    ax2c = fig2.add_axes([0.132, 0.075, 0.222, 0.372])
    bp_m = bp_df[(bp_df["database"] == "MIMIC-IV") & (bp_df["landmark_h"] == 48) & (bp_df["rule_code"] == "strict_icu_all6_fresh") & (bp_df["specification"].str.startswith("Adjusted"))].iloc[0]
    bp_e = bp_df[(bp_df["database"] == "eICU-CRD") & (bp_df["landmark_h"] == 48) & (bp_df["rule_code"] == "strict_icu_all6_fresh") & (bp_df["specification"].str.startswith("Adjusted"))].iloc[0]

    def pct_from_frac(s_val):
        m = re.search(r"\(([0-9\.]+)%\)", str(s_val))
        return float(m.group(1)) if m else 0.0

    m_res_n = int(str(bp_m["resolved_blood_exposed"]).split("/")[1].split()[0])
    m_per_n = int(str(bp_m["persistent_blood_exposed"]).split("/")[1].split()[0])
    e_res_n = int(str(bp_e["resolved_blood_exposed"]).split("/")[1].split()[0])
    e_per_n = int(str(bp_e["persistent_blood_exposed"]).split("/")[1].split()[0])
    m_rc = [int(x) for x in str(bp_m["resolved_PRBC_Plt_Plasma_Factor"]).split("/")]
    m_pc = [int(x) for x in str(bp_m["persistent_PRBC_Plt_Plasma_Factor"]).split("/")]
    e_rc = [int(x) for x in str(bp_e["resolved_PRBC_Plt_Plasma_Factor"]).split("/")]
    e_pc = [int(x) for x in str(bp_e["persistent_PRBC_Plt_Plasma_Factor"]).split("/")]

    cats_c = ["MIMIC Any", "MIMIC PRBC", "MIMIC FFP", "eICU Any", "eICU PRBC", "eICU FFP"]
    res_pcts = [pct_from_frac(bp_m["resolved_blood_exposed"]), 100*m_rc[0]/m_res_n, 100*m_rc[2]/m_res_n,
                pct_from_frac(bp_e["resolved_blood_exposed"]), 100*e_rc[0]/e_res_n, 100*e_rc[2]/e_res_n]
    per_pcts = [pct_from_frac(bp_m["persistent_blood_exposed"]), 100*m_pc[0]/m_per_n, 100*m_pc[2]/m_per_n,
                pct_from_frac(bp_e["persistent_blood_exposed"]), 100*e_pc[0]/e_per_n, 100*e_pc[2]/e_per_n]

    yc2 = np.arange(len(cats_c))[::-1]
    for yi, vp, vr in zip(yc2, per_pcts, res_pcts):
        ax2c.plot([vr, vp], [yi, yi], color="#94A3B8", lw=1.4, zorder=2)
        ax2c.scatter(vp, yi, marker="D", s=22, color=C_E3, zorder=3, label="Persistent E3→E3" if yi == yc2[0] else None)
        ax2c.scatter(vr, yi, marker="o", s=24, color=C_TEAL, zorder=4, label="Resolved E3→E1" if yi == yc2[0] else None)
        lo_v, hi_v = min(vp, vr), max(vp, vr)
        ax2c.text(lo_v - 2.6, yi, f"{lo_v:.1f}", va="center", ha="right", fontsize=5.7,
                  fontweight="bold" if vr <= vp else "normal", color=C_TEAL if vr <= vp else C_E3)
        ax2c.text(hi_v + 2.6, yi, f"{hi_v:.1f}", va="center", ha="left", fontsize=5.7,
                  fontweight="bold" if vr > vp else "normal", color=C_E3 if vp >= vr else C_TEAL)

    ax2c.axhline(2.5, color=C_GRID, lw=0.8)
    ax2c.set_yticks(yc2); ax2c.set_yticklabels(cats_c, fontsize=5.85, fontweight="bold")
    ax2c.set_xlim(-4, 71); ax2c.set_xticks([0, 20, 40, 60]); ax2c.set_ylim(-0.6, 5.7)
    ax2c.set_xlabel("Recorded 0–48h exposure (%)")
    ax2c.grid(axis="x", color=C_GRID, lw=0.5)
    ax2c.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=2, frameon=False, fontsize=5.7, handletextpad=0.2, columnspacing=0.6)

    # Panel d: Hero Table-Aligned Forest Plot (10 rows, enlarged ~5.95-pt typography, both blood adjustments!)
    panel_header(fig2, 0.388, 0.495, "d", "Multivariable ICU-resident landmark survival models")
    ax2d = fig2.add_axes([0.642, 0.075, 0.180, 0.375])

    def pick_surv(lm_h, rule_c, contrast_sub):
        r = surv_df[(surv_df["database"].str.startswith("Pooled")) & (surv_df["landmark_h"] == lm_h) & (surv_df["rule_code"] == rule_c) & (surv_df["contrast"].str.contains(contrast_sub, regex=False))].iloc[0]
        return float(r["aOR"]), float(r["ci_lo"]), float(r["ci_hi"]), int(r["n_pair"])

    def pick_bp(lm_h, rule_c, spec_sub):
        r = bp_df[(bp_df["database"].str.startswith("Pooled")) & (bp_df["landmark_h"] == lm_h) & (bp_df["rule_code"] == rule_c) & (bp_df["specification"].str.contains(spec_sub, regex=False))].iloc[0]
        return float(r["aOR"]), float(r["ci_lo"]), float(r["ci_hi"]), int(r["n_pair"])

    s1 = pick_surv(48, "strict_icu_all6_fresh", "Resolved E3 -> E1")
    s2 = pick_surv(48, "strict_icu_active_window", "Resolved E3 -> E1")
    b_any = pick_bp(48, "strict_icu_all6_fresh", "Adjusted for any 0–48h")
    b_cmp = pick_bp(48, "strict_icu_all6_fresh", "separate 0–48h PRBC")
    b_no = pick_bp(48, "strict_icu_all6_fresh", "WITHOUT 0–48h")
    b_yes = pick_bp(48, "strict_icu_all6_fresh", "WITH 0–48h")
    s3 = pick_surv(48, "strict_icu_all6_fresh", "De novo E1 -> E3")
    s_e12 = pick_surv(48, "strict_icu_all6_fresh", "De novo E1 -> E2")
    s4 = pick_surv(48, "strict_icu_all6_fresh", "Dynamic Landmark E3")
    s5 = pick_surv(72, "strict_icu_all6_fresh", "Resolved E3 -> E1")

    forest_d = [
        ("48h E3→E1 (all 6 fresh, k=3)", s1[3], s1[0], s1[1], s1[2], C_TEAL, "D", True),
        ("48h E3→E1 (active-window, k=3)", s2[3], s2[0], s2[1], s2[2], C_TEAL, "o", False),
        ("48h E3→E1 + any 0–48h blood", b_any[3], b_any[0], b_any[1], b_any[2], C_TEAL, "s", True),
        ("48h E3→E1 + blood components", b_cmp[3], b_cmp[0], b_cmp[1], b_cmp[2], C_TEAL, "o", False),
        ("48h E3→E1 (no 0–48h blood)", b_no[3], b_no[0], b_no[1], b_no[2], C_TEAL, "o", False),
        ("48h E3→E1 (with 0–48h blood)", b_yes[3], b_yes[0], b_yes[1], b_yes[2], C_TEAL, "o", False),
        ("48h De novo E1→E3 (k=3)", s3[3], s3[0], s3[1], s3[2], C_SLATE, "o", False),
        ("48h De novo E1→E2* (explor., US)", s_e12[3], s_e12[0], s_e12[1], s_e12[2], C_E2, "^", True),
        ("48h Dynamic E3 (adj. baseline)", s4[3], s4[0], s4[1], s4[2], C_E3, "D", True),
        ("72h E3→E1 (all 6 fresh, k=3)", s5[3], s5[0], s5[1], s5[2], C_SLATE, "o", False),
    ]
    yd = np.arange(len(forest_d))[::-1]
    ax2d.axvline(1.0, color="#64748B", ls="--", lw=0.85, zorder=1)
    for yi, (lbl, n_v, or_v, lo_v, hi_v, col, mk, is_sig) in zip(yd, forest_d):
        mfc = col if is_sig or col == C_TEAL else "white"
        ax2d.errorbar(or_v, yi, xerr=[[or_v - lo_v], [hi_v - or_v]], fmt=mk, color=col,
                      markerfacecolor=mfc, markeredgecolor=col, markersize=4.1, lw=1.1, capsize=1.9, zorder=3)

    ax2d.set_xscale("log"); ax2d.set_xlim(0.035, 9.5); ax2d.set_ylim(-0.6, len(forest_d) - 0.3)
    ax2d.set_xticks([0.05, 0.2, 0.5, 1.0, 2.0, 5.0])
    ax2d.set_xticklabels(["0.05", "0.2", "0.5", "1", "2", "5"])
    ax2d.xaxis.minorticks_off()
    ax2d.set_yticks(yd); ax2d.set_yticklabels([""] * len(yd))
    ax2d.spines["left"].set_visible(False); ax2d.tick_params(axis="y", length=0)
    ax2d.set_xlabel("Adjusted OR (log scale)")
    ax2d.grid(axis="x", color=C_GRID, lw=0.5)

    ax2d_tbl = fig2.add_axes([0.388, 0.075, 0.597, 0.398])
    ax2d_tbl.set_xlim(0, 1); ax2d_tbl.set_ylim(-0.6, len(forest_d) + 0.30); ax2d_tbl.axis("off")
    ax2d_tbl.text(0.0, len(forest_d) - 0.12, "Contrast / Rule", fontsize=6.0, fontweight="bold", color=C_SUB)
    ax2d_tbl.text(0.395, len(forest_d) - 0.12, "N", fontsize=6.0, fontweight="bold", color=C_SUB, ha="right")
    ax2d_tbl.text(0.995, len(forest_d) - 0.12, "aOR (95% CI)", fontsize=6.0, fontweight="bold", color=C_SUB, ha="right")
    for yi, (lbl, n_v, or_v, lo_v, hi_v, col, mk, is_sig) in zip(yd, forest_d):
        fw = "bold" if is_sig else "normal"
        ax2d_tbl.text(0.0, yi, lbl, va="center", ha="left", fontsize=5.95, fontweight=fw, color=C_INK)
        ax2d_tbl.text(0.395, yi, f"{n_v:,}", va="center", ha="right", fontsize=5.9, color=C_SUB)
        ax2d_tbl.text(0.995, yi, f"{or_v:.2f} ({lo_v:.2f}–{hi_v:.2f})", va="center", ha="right", fontsize=5.95, fontweight=fw, color=col if is_sig else C_SUB)

    save_editorial(fig2, FIG_DIR, "ICM_v7_Fig2_Bidirectional_48h_ICU_Dynamic_Restratification")

    # =========================================================================
    # FIGURE 3 (180 mm × 158 mm)
    # =========================================================================
    fig3 = plt.figure(figsize=(W_180MM, 158.0 * MM_TO_IN), facecolor="white")

    def parse_thr_pcts(t_str):
        ms = re.findall(r"E[123]\s+(\d+)/(\d+)\s+\(([0-9\.]+)%\)", str(t_str))
        return {f"E{idx+1}": (int(k), int(n), float(p)) for idx, (k, n, p) in enumerate(ms)}

    def fmt_frac(s_frac):
        a, b = str(s_frac).split("/")
        return f"{int(a):,}/{int(b):,}"

    # Panel a upper: Hypofibrinogenaemia <200 mg/dL (%)
    panel_header(fig3, 0.015, 0.964, "a", "Haemostatic vulnerability: fibrinogen & aPTT")
    ax3a_top = fig3.add_axes([0.088, 0.702, 0.388, 0.220])
    fib_rows = bio_df[bio_df["biomarker"] == "Fibrinogen (mg/dL)"].set_index("database").loc[dbs4]
    aptt_rows = bio_df[bio_df["biomarker"] == "Intrinsic aPTT (s)"].set_index("database")
    fib_parsed = {d: parse_thr_pcts(fib_rows.loc[d, "threshold_counts_and_fisher"]) for d in dbs4}
    xf = np.arange(len(dbs4))
    for k_i, (st, col, mk) in enumerate([("E1", C_E1, "o"), ("E2", C_E2, "^"), ("E3", C_E3, "D")]):
        pos = xf + (k_i - 1) * 0.25
        vals = [fib_parsed[d][st][2] for d in dbs4]
        ax3a_top.bar(pos, vals, width=0.22, color=col, alpha=0.88, label=f"State {st}")
        for xi, vi in zip(pos, vals):
            ax3a_top.text(xi, vi + 1.5, f"{vi:.1f}", ha="center", va="bottom",
                          fontsize=5.7, fontweight="bold" if st == "E3" else "normal", color=C_INK)
    ax3a_top.set_xticks(xf)
    ax3a_top.set_xticklabels([f"{d.replace(' proxy','')}\n({fmt_frac(fib_rows.loc[d,'measured_total_vs_cohort'])})" for d in dbs4], fontsize=5.7)
    ax3a_top.set_ylim(0, 79)
    ax3a_top.set_ylabel("Fibrinogen <200 mg/dL (%)")
    ax3a_top.grid(axis="y", color=C_GRID, lw=0.5)
    ax3a_top.legend(loc="upper left", ncol=3, frameon=False, fontsize=5.75, handlelength=1.0, handletextpad=0.3, columnspacing=0.7)

    # Panel a lower sub-track: 3-cohort Intrinsic aPTT Median & IQR
    ax3a_bot = fig3.add_axes([0.088, 0.540, 0.388, 0.105])
    dbs_aptt = ["MIMIC-IV", "eICU-CRD", "SICdb"]
    ya_sub = np.arange(len(dbs_aptt))[::-1]
    for idx_d, d_n in enumerate(dbs_aptt):
        yi = ya_sub[idx_d]
        for st, col, mk, dy in [("E1", C_E1, "o", +0.18), ("E2", C_E2, "^", 0.0), ("E3", C_E3, "D", -0.18)]:
            med_v, q1_v, q3_v = parse_med_iqr(aptt_rows.loc[d_n, f"{st}_tested_summary"])
            ax3a_bot.plot([q1_v, q3_v], [yi + dy, yi + dy], color=col, lw=1.1, alpha=0.75)
            ax3a_bot.scatter(med_v, yi + dy, marker=mk, s=16, color=col, zorder=3)
        p_h = float(aptt_rows.loc[d_n, "holm_112_global"])
        ax3a_bot.text(55.2, yi, f"P{fmt_sci_p(p_h, 2)}", va="center", ha="right", fontsize=5.75, color=C_SUB)
    ax3a_bot.set_yticks(ya_sub); ax3a_bot.set_yticklabels(["MIMIC", "eICU", "SICdb"], fontsize=5.8, fontweight="bold")
    ax3a_bot.set_xlim(22, 56.0); ax3a_bot.set_xticks([25, 30, 35, 40, 45]); ax3a_bot.set_ylim(-0.55, 2.55)
    ax3a_bot.set_xlabel("Intrinsic aPTT median & IQR (s; *SICdb cont. Fib raw P=0.69)", fontsize=5.75)
    ax3a_bot.grid(axis="x", color=C_GRID, lw=0.5)

    # Panel b upper: Early blood lactate >=2.0 mmol/L (%)
    panel_header(fig3, 0.512, 0.964, "b", "Metabolic–inflammatory stress: lactate & NLR")
    ax3b_top = fig3.add_axes([0.580, 0.706, 0.395, 0.216])
    lac_rows = bio_df[bio_df["biomarker"] == "Blood lactate (mmol/L)"].set_index("database").loc[dbs4]
    nlr_rows = bio_df[bio_df["biomarker"] == "Neutrophil-to-lymphocyte ratio (NLR)"].set_index("database")
    lac_parsed = {d: parse_thr_pcts(lac_rows.loc[d, "threshold_counts_and_fisher"]) for d in dbs4}
    xl = np.arange(len(dbs4))
    for k_i, (st, col) in enumerate([("E1", C_E1), ("E2", C_E2), ("E3", C_E3)]):
        pos = xl + (k_i - 1) * 0.25
        vals = [lac_parsed[d][st][2] for d in dbs4]
        ax3b_top.bar(pos, vals, width=0.22, color=col, alpha=0.88, label=f"State {st}")
        for xi, vi in zip(pos, vals):
            ax3b_top.text(xi, vi + 1.5, f"{vi:.1f}", ha="center", va="bottom",
                          fontsize=5.7, fontweight="bold" if vi >= 65.0 else "normal", color=C_INK)
    ax3b_top.set_xticks(xl)
    ax3b_top.set_xticklabels([f"{d.replace(' proxy','')}\n({fmt_frac(lac_rows.loc[d,'measured_total_vs_cohort'])})" for d in dbs4], fontsize=5.7)
    ax3b_top.set_ylim(0, 82)
    ax3b_top.set_ylabel("Lactate ≥2.0 mmol/L (%)")
    ax3b_top.grid(axis="y", color=C_GRID, lw=0.5)

    # Panel b lower sub-track: US NLR Median & IQR
    ax3b_bot = fig3.add_axes([0.580, 0.540, 0.395, 0.105])
    dbs_nlr = ["MIMIC-IV", "eICU-CRD"]
    yb_sub = np.arange(len(dbs_nlr))[::-1]
    for idx_d, d_n in enumerate(dbs_nlr):
        yi = yb_sub[idx_d]
        for st, col, mk, dy in [("E1", C_E1, "o", +0.18), ("E2", C_E2, "^", 0.0), ("E3", C_E3, "D", -0.18)]:
            med_v, q1_v, q3_v = parse_med_iqr(nlr_rows.loc[d_n, f"{st}_tested_summary"])
            ax3b_bot.plot([q1_v, q3_v], [yi + dy, yi + dy], color=col, lw=1.1, alpha=0.75)
            ax3b_bot.scatter(med_v, yi + dy, marker=mk, s=16, color=col, zorder=3)
        p_h = float(nlr_rows.loc[d_n, "holm_112_global"])
        ax3b_bot.text(21.4, yi, f"P{fmt_sci_p(p_h, 2)}", va="center", ha="right", fontsize=5.75, color=C_SUB)
    ax3b_bot.set_yticks(yb_sub); ax3b_bot.set_yticklabels(["MIMIC", "eICU"], fontsize=5.8, fontweight="bold")
    ax3b_bot.set_xlim(2.0, 21.8); ax3b_bot.set_xticks([4, 8, 12, 16]); ax3b_bot.set_ylim(-0.6, 1.6)
    ax3b_bot.set_xlabel("US neutrophil-to-lymphocyte ratio median & IQR", fontsize=5.75)
    ax3b_bot.grid(axis="x", color=C_GRID, lw=0.5)

    # Panel c: Table-Aligned 12-lead ECG QTcF Forest Plot (enlarged ~5.95-pt typography)
    panel_header(fig3, 0.015, 0.474, "c", "12-lead ECG QTcF prolongation (MIMIC E3 vs E1)*")
    ax3c = fig3.add_axes([0.242, 0.120, 0.125, 0.312])
    qt_sub = ecg_df[ecg_df["endpoint"].str.startswith("Prolonged QTcF")].reset_index(drop=True)
    ecg_rows_meta = [
        ("Base clinical model", int(qt_sub.loc[0, "n_total"]), 0),
        ("Pre-ECG cardiac/K-Mg-Ca/QT", int(qt_sub.loc[1, "n_total"]), 1),
        ("Narrow QRS, no AF/pace", int(qt_sub.loc[2, "n_total"]), 2),
        ("+ Index cardiac diagnoses", int(qt_sub.loc[3, "n_total"]), 3),
        ("Complete pre-ECG K+Mg+Ca", int(qt_sub.loc[4, "n_total"]), 4),
        ("Post-admission ECGs", int(qt_sub.loc[5, "n_total"]), 5),
        ("Strictly post-6-lab ECGs", int(qt_sub.loc[6, "n_total"]), 6),
    ]
    yc3 = np.arange(len(ecg_rows_meta))[::-1]
    ax3c.axvline(1.0, color="#64748B", ls="--", lw=0.85, zorder=1)
    for yi, (lbl, n_v, idx_q) in zip(yc3, ecg_rows_meta):
        or_v, lo_v, hi_v = parse_or_ci(qt_sub.loc[idx_q, "E3_vs_E1_aOR_CI"])
        sig = lo_v > 1.0
        col = C_E3 if sig else C_SLATE
        mk = "D" if idx_q <= 2 else "o"
        ax3c.errorbar(or_v, yi, xerr=[[or_v - lo_v], [hi_v - or_v]], fmt=mk, color=col,
                      markerfacecolor=col if sig else "white", markeredgecolor=col,
                      markersize=4.2, lw=1.15, capsize=2.0, zorder=3)
    ax3c.set_xscale("log"); ax3c.set_xlim(0.42, 4.4); ax3c.set_ylim(-0.6, len(ecg_rows_meta) - 0.3)
    ax3c.set_xticks([0.5, 1.0, 2.0, 4.0])
    ax3c.set_xticklabels(["0.5", "1", "2", "4"])
    ax3c.xaxis.minorticks_off()
    ax3c.set_yticks(yc3); ax3c.set_yticklabels([""] * len(yc3))
    ax3c.spines["left"].set_visible(False); ax3c.tick_params(axis="y", length=0)
    ax3c.set_xlabel("aOR (95% CI)")
    ax3c.grid(axis="x", color=C_GRID, lw=0.5)

    ax3c_tbl = fig3.add_axes([0.015, 0.120, 0.470, 0.334])
    ax3c_tbl.set_xlim(0, 1); ax3c_tbl.set_ylim(-0.6, len(ecg_rows_meta) + 0.28); ax3c_tbl.axis("off")
    ax3c_tbl.text(0.0, len(ecg_rows_meta) - 0.12, "ECG model / subset", fontsize=5.95, fontweight="bold", color=C_SUB)
    ax3c_tbl.text(0.465, len(ecg_rows_meta) - 0.12, "n", fontsize=5.95, fontweight="bold", color=C_SUB, ha="right")
    ax3c_tbl.text(0.995, len(ecg_rows_meta) - 0.12, "aOR (95% CI)", fontsize=5.95, fontweight="bold", color=C_SUB, ha="right")
    for yi, (lbl, n_v, idx_q) in zip(yc3, ecg_rows_meta):
        or_v, lo_v, hi_v = parse_or_ci(qt_sub.loc[idx_q, "E3_vs_E1_aOR_CI"])
        sig = lo_v > 1.0
        fw = "bold" if idx_q <= 2 else "normal"
        ax3c_tbl.text(0.0, yi, lbl, va="center", ha="left", fontsize=5.9, fontweight=fw, color=C_INK if sig else C_SUB)
        ax3c_tbl.text(0.465, yi, f"{n_v:,}", va="center", ha="right", fontsize=5.85, color=C_SUB)
        ax3c_tbl.text(0.995, yi, f"{or_v:.2f} ({lo_v:.2f}–{hi_v:.2f})", va="center", ha="right", fontsize=5.9, fontweight=fw, color=C_E3 if sig else C_SUB)

    # Panel d: Table-Aligned Landmark Haemodynamic Forest Plot (enlarged ~5.95-pt typography + '*' on 0–24h MAP)
    panel_header(fig3, 0.512, 0.474, "d", "Landmark haemodynamic evaluations (ΔRD, pp)")
    ax3d = fig3.add_axes([0.745, 0.120, 0.120, 0.312])
    r_ols65 = parse_rd_ci(lmtp_df.loc[0, "E2_minus_E1"])
    r_rdg65 = parse_rd_ci(lmtp_df.loc[1, "E2_minus_E1"])
    r_map85 = parse_rd_ci(lmtp_df.loc[3, "E2_minus_E1"])
    n_map85 = int(lmtp_df.loc[3, "n"])

    comp_rows = [
        ("Overlapping CPP <60 (conflated)", 579, +29.3, +10.3, +48.3, C_CRIMSON, "s", 1, False),
        ("24h-landmark CPP <60 (OW)", 199, -4.9, -38.2, +29.2, C_E1, "D", 1, True),
        ("Observed-past OLS ICE (CPP <65)", 199, r_ols65[0], r_ols65[1], r_ols65[2], C_TEAL, "o", 2, True),
        ("Observed-past Ridge ICE (CPP <65)", 199, r_rdg65[0], r_rdg65[1], r_rdg65[2], C_SLATE, "o", 2, True),
        ("24h-landmark E3 MAP ≥85 × Hgb", 763, -11.3, -21.8, +0.3, C_E3, "D", 1, True),
        ("Observed-past ICE E3 MAP <85*", n_map85, r_map85[0], r_map85[1], r_map85[2], C_E3, "o", 2, True),
    ]
    yd3 = np.arange(len(comp_rows))[::-1]
    ax3d.axvline(0.0, color="#64748B", ls="--", lw=0.85, zorder=1)
    for yi, (lbl, n_v, pt_v, lo_v, hi_v, col, mk, dec_d, filled) in zip(yd3, comp_rows):
        ax3d.errorbar(pt_v, yi, xerr=[[pt_v - lo_v], [hi_v - pt_v]], fmt=mk, color=col,
                      markerfacecolor=col if filled else "white", markeredgecolor=col,
                      markersize=4.2, lw=1.15, capsize=2.0, zorder=3)
    ax3d.set_xlim(-42, 54); ax3d.set_ylim(-0.6, len(comp_rows) - 0.3)
    ax3d.set_xticks([-30, 0, 30])
    ax3d.set_yticks(yd3); ax3d.set_yticklabels([""] * len(yd3))
    ax3d.spines["left"].set_visible(False); ax3d.tick_params(axis="y", length=0)
    ax3d.set_xlabel("Interaction ΔRD (pp)")
    ax3d.grid(axis="x", color=C_GRID, lw=0.5)

    ax3d_tbl = fig3.add_axes([0.512, 0.120, 0.473, 0.334])
    ax3d_tbl.set_xlim(0, 1); ax3d_tbl.set_ylim(-0.6, len(comp_rows) + 0.28); ax3d_tbl.axis("off")
    ax3d_tbl.text(0.0, len(comp_rows) - 0.12, "Design / Estimator", fontsize=5.95, fontweight="bold", color=C_SUB)
    ax3d_tbl.text(0.475, len(comp_rows) - 0.12, "n", fontsize=5.95, fontweight="bold", color=C_SUB, ha="right")
    ax3d_tbl.text(0.995, len(comp_rows) - 0.12, "ΔRD (95% CI)", fontsize=5.95, fontweight="bold", color=C_SUB, ha="right")
    for yi, (lbl, n_v, pt_v, lo_v, hi_v, col, mk, dec_d, filled) in zip(yd3, comp_rows):
        ax3d_tbl.text(0.0, yi, lbl, va="center", ha="left", fontsize=5.85, color=C_CRIMSON if not filled else C_INK)
        ax3d_tbl.text(0.475, yi, f"{n_v:,}", va="center", ha="right", fontsize=5.85, color=C_SUB)
        ci_s = f"{pt_v:+.1f} ({lo_v:+.1f}, {hi_v:+.1f})" if dec_d == 1 else f"{pt_v:+.2f} ({lo_v:+.2f}, {hi_v:+.2f})"
        ax3d_tbl.text(0.995, yi, ci_s, va="center", ha="right", fontsize=5.85, color=C_CRIMSON if not filled else C_SUB)

    # Bottom 2-line synthesis banner (accurate cohort-scoped LAx & CPP findings; zero interaction P claim!)
    ax3_syn = fig3.add_axes([0.015, 0.008, 0.970, 0.052])
    ax3_syn.set_xlim(0, 100); ax3_syn.set_ylim(0, 10); ax3_syn.axis("off")
    ax3_syn.add_patch(FancyBboxPatch((0.2, 0.5), 99.6, 9.0, boxstyle="round,pad=0.1,rounding_size=0.9", fc="#F8FAFC", ec="#CBD5E1", lw=0.65))
    ax3_syn.text(50.0, 6.8, "Complementary bedside axes: 0–48h higher LAx predicts HiRID ICU death (aOR 1.73–1.78/SD) and low-CPP time predicts eICU hospital death (aOR 2.12),",
                 ha="center", va="center", fontsize=5.65, fontweight="bold", color=C_INK)
    ax3_syn.text(50.0, 3.0, "supporting universal autoregulation-guided neuroprotection while admission State E3 prompts targeted fibrinogen, intrinsic aPTT, and arrival QTcF checks.",
                 ha="center", va="center", fontsize=5.65, color=C_SUB)

    save_editorial(fig3, FIG_DIR, "ICM_v7_Fig3_MultiOrgan_ECG_and_Haemodynamic_Evaluations")

    # =========================================================================
    # STANDALONE BONUS: Intensive Care Medicine Visual Abstract (180 mm × 98 mm)
    # =========================================================================
    fig_va = plt.figure(figsize=(W_180MM, 98.0 * MM_TO_IN), facecolor="white")
    axv = fig_va.add_axes([0.0, 0.0, 1.0, 1.0])
    axv.set_xlim(0, 180); axv.set_ylim(0, 98); axv.axis("off")

    axv.add_patch(FancyBboxPatch((2.0, 84.5), 176.0, 11.5, boxstyle="round,pad=0.2,rounding_size=1.8", fc="#0F172A", ec="none"))
    axv.text(90.0, 91.6, "Baseline & 48-Hour Dynamic Host-Response States After Traumatic Brain Injury",
             color="white", fontsize=8.4, fontweight="bold", ha="center", va="center")
    axv.text(90.0, 87.0, "8,130 ICU admissions · 4 databases (MIMIC-IV, eICU-CRD, SICdb, HiRID) · 6 universal blood tests (WBC, Plt, Hgb, INR, Bicarb, Glu)",
             color="#CBD5E1", fontsize=6.0, ha="center", va="center")

    # Column 1: ICU Admission
    axv.add_patch(FancyBboxPatch((2.5, 13.5), 55.5, 68.0, boxstyle="round,pad=0.2,rounding_size=2.0", fc="#F8FAFC", ec=C_E1, lw=0.9))
    axv.text(30.25, 77.2, "1. Zero-Refit Admission Staging", color=C_E1, fontsize=7.0, fontweight="bold", ha="center")
    axv.text(30.25, 72.2, "3.5× US hosp. risk gap at equal motor GCS (6)", color=C_INK, fontsize=5.75, fontweight="bold", ha="center")
    axv.scatter(6.5, 65.1, marker="o", s=18, color=C_E1, zorder=4)
    axv.text(9.0, 64.5, "E1 Compensated (US 6.8–7.0% · SICdb 4.0%)", color=C_E1, fontsize=5.7, fontweight="bold")
    axv.scatter(6.5, 58.6, marker="^", s=20, color=C_E2, zorder=4)
    axv.text(9.0, 58.0, "E2 Hypermetabolic (US 16.3–20.8% death)", color=C_E2, fontsize=5.7, fontweight="bold")
    axv.text(9.0, 53.6, "MIMIC WBC 16.3 · Glu 172 · 4-DB lactate peak", color=C_SUB, fontsize=5.6)
    axv.scatter(6.5, 47.4, marker="D", s=18, color=C_E3, zorder=4)
    axv.text(9.0, 46.8, "E3 Coagulopathic (US 23.6–24.8% · HiRID 30.8%)", color=C_E3, fontsize=5.7, fontweight="bold")
    axv.text(9.0, 42.4, "MIMIC Hgb 10.2 · Plt 144 · SICdb 1-yr 40.6%", color=C_SUB, fontsize=5.6)
    axv.add_patch(FancyBboxPatch((5.0, 16.5), 50.5, 21.5, boxstyle="round,pad=0.2,rounding_size=1.4", fc="#F3E8FF", ec=C_E3, lw=0.7))
    axv.text(30.25, 34.0, "State E3 Targeted Bedside Prompt", color=C_E3, fontsize=5.95, fontweight="bold", ha="center")
    axv.text(30.25, 28.4, "• Fibrinogen <200 mg/dL: 28.2–67.0% (4 DBs)", color=C_INK, fontsize=5.65, ha="center")
    axv.text(30.25, 23.4, r"• Prolonged intrinsic aPTT (3 DBs, P ≤ 4.9×$10^{-18}$)", color=C_INK, fontsize=5.65, ha="center")
    axv.text(30.25, 18.6, "• MIMIC arrival QTcF prolongation: aOR 2.08", color=C_INK, fontsize=5.65, ha="center")

    # Column 2: 48-Hour Retesting
    axv.add_patch(FancyBboxPatch((61.5, 13.5), 57.0, 68.0, boxstyle="round,pad=0.2,rounding_size=2.0", fc="#F8FAFC", ec=C_TEAL, lw=0.9))
    axv.text(90.0, 77.2, "2. 48-Hour ICU Retesting Radar", color=C_TEAL, fontsize=7.0, fontweight="bold", ha="center")
    axv.text(90.0, 72.2, "All 6 tests freshly redrawn in (24, 48] h", color=C_INK, fontsize=5.8, fontweight="bold", ha="center")

    axv.add_patch(FancyBboxPatch((64.0, 44.5), 52.0, 23.5, boxstyle="round,pad=0.2,rounding_size=1.4", fc="#ECFDF5", ec=C_TEAL, lw=0.75))
    axv.text(90.0, 63.8, "Resolved E3 → E1 (22.6–34.4%, 4 DBs)", color=C_TEAL, fontsize=6.0, fontweight="bold", ha="center")
    axv.text(90.0, 58.3, "Post-48h hosp. death: 14.2% vs 26.4% (3 DBs)", color=C_INK, fontsize=5.7, fontweight="bold", ha="center")
    axv.text(90.0, 53.0, "Pooled aOR 0.48 · US blood-adj. 0.47–0.51", color=C_TEAL, fontsize=5.7, fontweight="bold", ha="center")
    axv.text(90.0, 47.6, "Lower 0–48h blood exposure (MIMIC 40.6% vs 54.2%)", color=C_SUB, fontsize=5.6, ha="center")

    axv.add_patch(FancyBboxPatch((64.0, 16.5), 52.0, 23.5, boxstyle="round,pad=0.2,rounding_size=1.4", fc="#FFF7ED", ec=C_E2, lw=0.75))
    axv.text(90.0, 35.8, "De novo E1 → E2* (Exploratory, US)", color=C_E2, fontsize=6.0, fontweight="bold", ha="center")
    axv.text(90.0, 30.3, "Post-48h US hosp. death: 28.1% vs 7.9%", color=C_INK, fontsize=5.75, fontweight="bold", ha="center")
    axv.text(90.0, 25.0, "US aOR 3.38 (P = 0.0075; E1→E3 null 1.28)", color=C_E2, fontsize=5.7, fontweight="bold", ha="center")
    axv.text(90.0, 19.6, "Flags occult metabolic–inflammatory surge", color=C_SUB, fontsize=5.6, ha="center")

    # Column 3: Complementary Integration
    axv.add_patch(FancyBboxPatch((122.0, 13.5), 55.5, 68.0, boxstyle="round,pad=0.2,rounding_size=2.0", fc="#F8FAFC", ec="#334155", lw=0.9))
    axv.text(149.75, 77.2, "3. Brain–Body Bedside Partnership", color=C_INK, fontsize=7.0, fontweight="bold", ha="center")
    axv.text(149.75, 72.2, "Complementary roles — never replacement", color=C_SUB, fontsize=5.8, fontweight="bold", ha="center")
    axv.text(124.8, 64.0, "• Clin. base + 12-df splines (eICU 0.876):", color=C_INK, fontsize=5.7, fontweight="bold")
    axv.text(126.8, 59.4, "Automated EHR day-1 + 6-lab probability", color=C_SUB, fontsize=5.65)
    axv.text(124.8, 52.5, "• Discrete states (AUC 0.866, slope 1.045):", color=C_INK, fontsize=5.7, fontweight="bold")
    axv.text(126.8, 47.9, "Bedside staging, 48h tracking & trial enrichment", color=C_SUB, fontsize=5.65)
    axv.text(124.8, 41.0, "• Invasive LAx & CPP (HiRID & eICU):", color=C_INK, fontsize=5.7, fontweight="bold")
    axv.text(126.8, 36.4, "HiRID LAx aOR 1.73–1.78 · eICU CPP aOR 2.12", color=C_SUB, fontsize=5.65)
    axv.plot([126.0, 173.5], [30.5, 30.5], ls="--", color="#94A3B8", lw=0.8)
    axv.text(149.75, 25.0, "Strict 24h landmarks prevent confounded", color=C_SUB, fontsize=5.7, ha="center")
    axv.text(149.75, 20.0, "blood-based alteration of CPP targets", color=C_INK, fontsize=5.75, fontweight="bold", ha="center")

    axv.add_patch(FancyBboxPatch((2.0, 2.0), 176.0, 8.8, boxstyle="round,pad=0.2,rounding_size=1.5", fc="#F1F5F9", ec="#CBD5E1", lw=0.7))
    axv.text(90.0, 6.4, "Take-home: Six routine ICU blood tests stratify TBI mortality (at equal median motor GCS in US/HiRID), flag E3 coagulopathy/QTc risk, and update 48h prognosis alongside ICP care.",
             color=C_INK, fontsize=5.75, fontweight="bold", ha="center", va="center")

    save_editorial(fig_va, VA_DIR, "ICM_v7_Visual_Abstract")
    print("All ICM v7 180-mm editorial figures and Visual Abstract rendered.")


if __name__ == "__main__":
    main()
