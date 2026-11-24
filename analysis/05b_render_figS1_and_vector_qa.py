#!/usr/bin/env python3
"""
ICM v8 — Render Supplementary Figure S1 (External eICU-CRD Decision Curve Analysis & Four-Cohort
Conventional ICU Severity-Score Benchmarks) at exact 180.0-mm double-column width (600 DPI PNG +
Type-42 TrueType vector PDF + SVG) AND run the automated PyMuPDF vector QA gate across all 5 figures
(Fig 1, Fig 2, Fig 3, Visual Abstract, and Fig S1).
"""
import json
import os
import re
import shutil
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

fitz.TOOLS.set_small_glyph_heights(True)

ROOT = Path(os.environ.get("TBI_ROOT", "./data_workspace")))
V7_DIR = ROOT / "icm_submission_v7_20261124"
V8_DIR = ROOT / "icm_submission_v8_20261124"
TAB_DIR = V8_DIR / "tables"
FIG_DIR = V8_DIR / "figures"
VA_DIR = V8_DIR / "visual_abstract"
QA_DIR = V8_DIR / "qa_visual_proofs"
ART_DIR = Path(os.environ.get("TBI_ARTIFACTS_DIR", "./figures_preview"))

for d in (FIG_DIR, VA_DIR, QA_DIR):
    d.mkdir(parents=True, exist_ok=True)

MM_TO_IN = 1.0 / 25.4
PT_TO_MM = 25.4 / 72.0
W_180MM = 180.0 * MM_TO_IN

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.1,
    "axes.titlesize": 7.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 6.2,
    "xtick.labelsize": 5.85,
    "ytick.labelsize": 5.85,
    "legend.fontsize": 5.75,
    "axes.linewidth": 0.65,
    "axes.edgecolor": "#334155",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

C_BASE = "#475569"     # Slate grey-blue (Clinical base / Score alone)
C_M1 = "#6A2C70"       # Royal Plum (+ Frozen 3 States)
C_M2 = "#0D7A68"       # Deep Teal (+ Posterior probs)
C_M4 = "#D55E00"       # Okabe-Ito Vermilion (+ 12-df Splines)
C_ALL = "#94A3B8"      # Muted slate (Treat All / None)
C_INK = "#0F172A"
C_GRID = "#E2E8F0"


def fmt_p(p: float) -> str:
    if p < 1e-15:
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return f"{mant:.1f}e{exp}"
    if p < 0.001:
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return f"{mant:.1f}e{exp}"
    return f"{p:.3f}"


def copy_v7_figures_to_v8():
    # Copy verified v7 main figures & visual abstract as ICM_v8_* without touching v7!
    for src in sorted((V7_DIR / "figures").glob("ICM_v7_Fig*")):
        dst_name = src.name.replace("ICM_v7_", "ICM_v8_")
        shutil.copy2(src, FIG_DIR / dst_name)
    for src in sorted((V7_DIR / "visual_abstract").glob("ICM_v7_Visual_Abstract*")):
        dst_name = src.name.replace("ICM_v7_", "ICM_v8_")
        shutil.copy2(src, VA_DIR / dst_name)


def render_fig_s1():
    dca = pd.read_csv(TAB_DIR / "V8_DCA_Curve_eICU_n3640.csv")
    sev = pd.read_csv(TAB_DIR / "V8_TableS1G_Conventional_Severity_Score_Benchmarks.csv")

    fig = plt.figure(figsize=(W_180MM, 96.0 * MM_TO_IN), facecolor="white")

    # Left panel: Decision Curve Analysis (5% to 50%) — ends at x=0.405 for wide central gutter
    ax_a = fig.add_axes([0.072, 0.140, 0.333, 0.725])
    sub_dca = dca[(dca["threshold_prob"] >= 0.05) & (dca["threshold_prob"] <= 0.50)].copy()
    x_pct = sub_dca["threshold_pct"].to_numpy()

    # Shaded E2/E3 mortality decision zone (20% to 30%)
    ax_a.axvspan(20.0, 30.0, color="#F3E8FF", alpha=0.55, zorder=0)
    ax_a.grid(True, linestyle=":", linewidth=0.5, color=C_GRID, zorder=1)

    ax_a.axhline(0.0, color="#64748B", linestyle=":", linewidth=0.85, label="Treat none (NB = 0)", zorder=2)
    ax_a.plot(x_pct, sub_dca["nb_treat_all"], color=C_ALL, linestyle="--", linewidth=0.95, label="Treat all", zorder=2)
    ax_a.plot(x_pct, sub_dca["nb_M0_clinical_base"], color=C_BASE, linestyle="-", linewidth=1.15,
              label="M0: Clinical base (AUC 0.857)", zorder=3)
    ax_a.plot(x_pct, sub_dca["nb_M2_posterior_probs"], color=C_M2, linestyle="-.", linewidth=1.05,
              label="M2: + Posterior probs (0.869)", zorder=4)
    ax_a.plot(x_pct, sub_dca["nb_M4_12df_splines"], color=C_M4, linestyle="--", linewidth=1.20,
              label="M4: + 12-df splines (0.876)", zorder=5)
    ax_a.plot(x_pct, sub_dca["nb_M1_discrete_states"], color=C_M1, linestyle="-", linewidth=1.55,
              label="M1: + Discrete 3 states (0.866)", zorder=6)

    ax_a.set_xlim(5.0, 50.0)
    ax_a.set_ylim(-0.025, 0.112)
    ax_a.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax_a.set_xlabel("Hospital-mortality threshold probability (%)", fontweight="bold", color=C_INK, labelpad=4)
    ax_a.set_ylabel("Net benefit (TP equivalents / stay)", fontweight="bold", color=C_INK, labelpad=4)
    ax_a.set_title("a   External decision curves (eICU-CRD, n = 3,640)", loc="left", pad=12, color=C_INK, x=-0.10)

    ax_a.text(
        19.6, 0.097,
        "E2/E3 zone (20–30%)\nM1 vs M0: +3.1 to +7.3 / 1,000",
        fontsize=5.65, color="#4C1D95", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#C4B5FD", linewidth=0.55),
        zorder=7,
    )

    ax_a.legend(
        loc="lower left", bbox_to_anchor=(0.01, 0.02),
        frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96,
        handlelength=1.7, labelspacing=0.24, borderpad=0.28, fontsize=5.65,
    )

    # Right panel: Paired Conventional Severity-Score Benchmark across all 4 databases
    keep_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    s_plot = sev.iloc[keep_idx].copy().reset_index(drop=True)
    short_labels = [
        "MIMIC-IV: Motor GCS (3,483)",
        "MIMIC-IV: Total GCS (3,481)",
        "eICU-CRD: Motor GCS (3,640)",
        "eICU-CRD: Total GCS (3,640)",
        "eICU-CRD: APACHE IV (3,297)",
        "eICU-CRD: APACHE IVa (3,273)",
        "SICdb: SAPS 3 hosp. (516)",
        "SICdb: SAPS 3 1-yr (457)",
        "HiRID: Motor GCS ICU (491)",
    ]

    ax_b = fig.add_axes([0.640, 0.140, 0.215, 0.725])
    y_pos = np.arange(len(s_plot))[::-1]

    ax_b.grid(True, axis="x", linestyle=":", linewidth=0.5, color=C_GRID, zorder=1)
    for yi, (_, row), lbl in zip(y_pos, s_plot.iterrows(), short_labels):
        a0 = float(row["comparator_auc"])
        a1 = float(row["plus_states_auc"])
        ax_b.plot([a0, a1], [yi, yi], color="#94A3B8", linewidth=1.25, zorder=2)
        ax_b.scatter([a0], [yi], s=20, facecolor="white", edgecolor=C_BASE, linewidth=1.1, zorder=3,
                     label="Score alone" if yi == y_pos[0] else None)
        ax_b.scatter([a1], [yi], s=22, facecolor=C_M1, edgecolor=C_M1, linewidth=0.8, zorder=4,
                     label="Score + states (E1–E3)" if yi == y_pos[0] else None)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(short_labels, fontsize=5.8, color=C_INK)
    ax_b.set_xlim(0.74, 0.91)
    ax_b.set_xticks([0.75, 0.80, 0.85, 0.90])
    ax_b.set_ylim(-0.65, len(s_plot) - 0.30)
    ax_b.set_xlabel("Discrimination AUC", fontweight="bold", color=C_INK, labelpad=4)
    ax_b.set_title("b   Gain over ICU severity scores (4 cohorts)", loc="left", pad=12, color=C_INK, x=-0.88)

    # Place legend cleanly centered beneath Panel b title (wide clearance from right-hand column!)
    ax_b.legend(
        loc="lower left", bbox_to_anchor=(-0.58, 1.005), ncol=2,
        frameon=False, handletextpad=0.25, columnspacing=0.80, borderaxespad=0.0, fontsize=5.65,
    )

    # Right-hand tabular column for 2-df Wald P & E3 aOR (zero overlap)
    ax_txt = fig.add_axes([0.865, 0.140, 0.128, 0.725])
    ax_txt.axis("off")
    ax_txt.set_ylim(-0.65, len(s_plot) - 0.30)
    ax_txt.set_xlim(0.0, 1.0)
    ax_txt.text(0.02, len(s_plot) - 0.05, "2-df P  |  E3 aOR", fontsize=5.8, fontweight="bold", color=C_INK, va="bottom")
    for yi, (_, row) in zip(y_pos, s_plot.iterrows()):
        p_str = fmt_p(float(row["global_2df_wald_p"]))
        e3_str = f"{row['e3_vs_e1_aor']:.2f}"
        ax_txt.text(0.02, yi, f"{p_str} | {e3_str}", fontsize=5.75, color=C_INK, va="center")

    stem = "ICM_v8_FigS1_External_DCA_and_Severity_Benchmarks"
    pdf_p = FIG_DIR / f"{stem}.pdf"
    png_p = FIG_DIR / f"{stem}.png"
    svg_p = FIG_DIR / f"{stem}.svg"
    fig.savefig(pdf_p, dpi=600)
    fig.savefig(png_p, dpi=600)
    fig.savefig(svg_p)
    plt.close(fig)
    shutil.copy2(png_p, ART_DIR / f"{stem}.png")
    return pdf_p, png_p


def inspect_pdf_qa(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    assert len(doc) == 1, f"{pdf_path.name}: expected 1 page"
    page = doc[0]
    w_pt, h_pt = float(page.rect.width), float(page.rect.height)
    w_mm, h_mm = w_pt * PT_TO_MM, h_pt * PT_TO_MM

    fonts = page.get_fonts()
    has_type3 = any("Type3" in f[2] for f in fonts)

    raw_dict = page.get_text("dict")
    spans_records = []
    line_boxes = []

    for b_idx, blk in enumerate(raw_dict["blocks"]):
        if "lines" not in blk:
            continue
        for l_idx, ln in enumerate(blk["lines"]):
            l_texts, x0s, y0s, x1s, y1s = [], [], [], [], []
            for s_idx, sp in enumerate(ln["spans"]):
                txt = sp["text"]
                if not txt.strip():
                    continue
                bx0, by0, bx1, by1 = [float(v) for v in sp["bbox"]]
                sz = float(sp["size"])
                is_exp = sz < 5.0 and bool(re.fullmatch(r"[−\-]?\d+", txt.strip()))
                spans_records.append({
                    "text": txt, "size_pt": round(sz, 2), "is_exponent": is_exp,
                    "bbox": [round(bx0, 2), round(by0, 2), round(bx1, 2), round(by1, 2)],
                })
                l_texts.append(txt)
                x0s.append(bx0); y0s.append(by0); x1s.append(bx1); y1s.append(by1)
            if l_texts:
                line_boxes.append({
                    "text": "".join(l_texts),
                    "bbox": [min(x0s), min(y0s), max(x1s), max(y1s)],
                })

    non_exp_sizes = [s["size_pt"] for s in spans_records if not s["is_exponent"]]
    min_pt = min(non_exp_sizes) if non_exp_sizes else 0.0

    margin_violations = []
    pad_pt = 2.0
    for lb in line_boxes:
        x0, y0, x1, y1 = lb["bbox"]
        if x0 < pad_pt or y0 < pad_pt or x1 > (w_pt - pad_pt) or y1 > (h_pt - pad_pt):
            margin_violations.append({"text": lb["text"], "bbox": lb["bbox"]})

    overlaps = []
    for i in range(len(line_boxes)):
        ax0, ay0, ax1, ay1 = line_boxes[i]["bbox"]
        for j in range(i + 1, len(line_boxes)):
            bx0, by0, bx1, by1 = line_boxes[j]["bbox"]
            dx = min(ax1, bx1) - max(ax0, bx0)
            dy = min(ay1, by1) - max(ay0, by0)
            if dx > 0.8 and dy > 0.8:
                overlaps.append((line_boxes[i]["text"], line_boxes[j]["text"], round(dx, 2), round(dy, 2)))

    pix = page.get_pixmap(dpi=300, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(QA_DIR / f"{pdf_path.stem}_300dpi_QA.png")
    doc.close()

    res = {
        "file": pdf_path.name,
        "width_mm": round(w_mm, 2),
        "height_mm": round(h_mm, 2),
        "has_type3_fonts": has_type3,
        "span_count": len(spans_records),
        "min_non_exponent_pt": round(min_pt, 2),
        "margin_violation_count": len(margin_violations),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "margin_violations": margin_violations,
    }
    assert abs(w_mm - 180.0) <= 0.15, f"{pdf_path.name}: width {w_mm} != 180.0 mm"
    assert not has_type3, f"{pdf_path.name}: Type3 font found"
    assert min_pt >= 5.55, f"{pdf_path.name}: min font {min_pt} < 5.55 pt"
    assert len(margin_violations) == 0, f"{pdf_path.name}: margin violations {margin_violations}"
    assert len(overlaps) == 0, f"{pdf_path.name}: text overlaps {overlaps}"
    return res


def main():
    copy_v7_figures_to_v8()
    render_fig_s1()

    pdfs = sorted(list(FIG_DIR.glob("ICM_v8_Fig*.pdf")) + list(VA_DIR.glob("ICM_v8_Visual_Abstract.pdf")))
    assert len(pdfs) == 5, f"Expected 5 vector PDFs (Fig 1-3, Fig S1, Visual Abstract), got {len(pdfs)}"

    qa_results = []
    for p in pdfs:
        r = inspect_pdf_qa(p)
        qa_results.append(r)
        print(f"[QA PASS] {r['file']:56s} | {r['width_mm']:.1f}x{r['height_mm']:.1f} mm | spans={r['span_count']:3d} | min_pt={r['min_pt'] if 'min_pt' in r else r['min_non_exponent_pt']:.2f} | overlaps=0 | margins=0")

    (V8_DIR / "FIGURE_QA_REPORT_v8.json").write_text(json.dumps(qa_results, indent=2), encoding="utf-8")
    print("ALL_5_V8_FIGURES_VECTOR_QA_PASS")


if __name__ == "__main__":
    main()
