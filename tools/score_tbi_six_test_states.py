#!/usr/bin/env python3
"""
Standalone Zero-Refit Six-Test Host-Response State Scorer for Acute Traumatic Brain Injury (TBI).

Implements the exact closed-form frozen MIMIC-IV v3.1 classifier (n = 3,483; Table S1.1) using
only Python's standard library (zero external dependencies required for single-patient or CSV scoring):
  1. White blood cell count (WBC, x10^9/L)
  2. Platelet count (Plt, x10^9/L)
  3. Haemoglobin (Hgb, g/dL; or g/L with --european-units)
  4. International normalised ratio (INR, dimensionless; optional --sicdb-censored-inr for INR <= 1.20)
  5. Serum bicarbonate (HCO3, mmol/L)
  6. Blood glucose (Glu, mg/dL; or mmol/L with --european-units)

Outputs:
  - Assigned host-response state: E1 (Compensated), E2 (Hypermetabolic), or E3 (Coagulopathic)
  - Posterior membership probabilities: P(E1), P(E2), P(E3)
  - Non-causal clinical diagnostic vigilance prompt (ESM 1: Section S1.1)

Examples:
  # Single patient at ICU admission or 48-hour landmark:
  python3 tools/score_tbi_six_test_states.py --wbc 8.5 --plt 144 --hgb 10.2 --inr 1.50 --bicarb 23.0 --glu 133

  # Batch CSV scoring (columns: wbc, plt, hgb, inr, bicarb, glu):
  python3 tools/score_tbi_six_test_states.py --input-csv patients.csv --output-csv patients_scored.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "locked_mimic_tbi_endotype_k3.json"

CLIP_BOUNDS = {
    "wbc": (0.2, 100.0),
    "plt": (10.0, 1000.0),
    "hgb": (4.0, 22.0),
    "inr": (0.70, 10.0),
    "bicarb": (6.0, 45.0),
    "glu": (30.0, 900.0),
}

CLINICAL_PROMPTS = {
    "E1": (
        "Compensated State E1 (reference risk stratum): maintain standard neurocritical ICP/autoregulation "
        "protection; repeat full six-test panel at 48-hour ICU landmark if still in ICU to monitor for "
        "secondary metabolic-inflammatory escalation."
    ),
    "E2": (
        "Hypermetabolic State E2 (elevated WBC/glucose/platelets, lower bicarbonate, intact INR): "
        "assess systemic Catecholamine/glycolytic stress and secondary infection/extracranial injury; "
        "repeat full six-test panel at 48-hour ICU landmark."
    ),
    "E3": (
        "Coagulopathic State E3 (anaemia, thrombocytopenia, elevated INR): prompt targeted check of "
        "fibrinogen (<200/<150 mg/dL), intrinsic aPTT, electrolytes (K/Mg/Ca), QT-prolonging drugs, "
        "and an arrival 12-lead ECG (QTcF), alongside standard neuromonitoring; repeat all six tests "
        "freshly at 48 hours in ICU residents to assess E3->E1 physiological resolution."
    ),
}


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def load_frozen_model(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    return json.loads(Path(model_path).read_text(encoding="utf-8"))


def score_single_panel(
    wbc: float,
    plt: float,
    hgb: float,
    inr: float,
    bicarb: float,
    glu: float,
    cfg: dict | None = None,
    european_units: bool = False,
    sicdb_censored_inr: bool = False,
) -> Dict[str, object]:
    """Score one six-test blood panel without parameter refitting."""
    if cfg is None:
        cfg = load_frozen_model()

    hgb_conv = float(hgb) / 10.0 if european_units else float(hgb)
    glu_conv = float(glu) * 18.0182 if european_units else float(glu)
    raw_inr = float(inr)

    wbc_c = min(max(float(wbc), CLIP_BOUNDS["wbc"][0]), CLIP_BOUNDS["wbc"][1])
    plt_c = min(max(float(plt), CLIP_BOUNDS["plt"][0]), CLIP_BOUNDS["plt"][1])
    hgb_c = min(max(hgb_conv, CLIP_BOUNDS["hgb"][0]), CLIP_BOUNDS["hgb"][1])
    inr_c = min(max(raw_inr, CLIP_BOUNDS["inr"][0]), CLIP_BOUNDS["inr"][1])
    bic_c = min(max(float(bicarb), CLIP_BOUNDS["bicarb"][0]), CLIP_BOUNDS["bicarb"][1])
    glu_c = min(max(glu_conv, CLIP_BOUNDS["glu"][0]), CLIP_BOUNDS["glu"][1])

    u = [
        math.log1p(wbc_c),
        -plt_c,
        -hgb_c,
        math.log(inr_c),
        -bic_c,
        math.log1p(glu_c),
    ]
    centers = cfg["scaler_center"]
    scales = cfg["scaler_scale"]
    x = [(u[j] - centers[j]) / scales[j] for j in range(6)]

    means = cfg["gmm_means"]
    covars = cfg["gmm_covariances"]
    weights = cfg["gmm_weights"]

    logp = [0.0, 0.0, 0.0]
    for k in range(3):
        ll = 0.0
        for j in range(6):
            if sicdb_censored_inr and j == 3 and raw_inr <= 1.2001:
                x_hi = (math.log(1.20) - centers[3]) / scales[3]
                x_lo = (math.log(0.70) - centers[3]) / scales[3]
                sd = math.sqrt(covars[k][3])
                cdf_diff = max(_norm_cdf((x_hi - means[k][3]) / sd) - _norm_cdf((x_lo - means[k][3]) / sd), 1e-12)
                ll += math.log(cdf_diff)
            else:
                ll += -0.5 * (math.log(2.0 * math.pi * covars[k][j]) + ((x[j] - means[k][j]) ** 2) / covars[k][j])
        logp[k] = math.log(weights[k]) + ll

    max_lp = max(logp)
    exps = [math.exp(v - max_lp) for v in logp]
    denom = sum(exps)
    post_raw = [v / denom for v in exps]

    comp_map = {int(idx): name[:2] for name, idx in cfg["component_map"].items()}
    probs_by_state = {comp_map[k]: post_raw[k] for k in range(3)}
    best_state = comp_map[int(max(range(3), key=lambda k: logp[k]))]

    return {
        "assigned_state": best_state,
        "prob_E1": round(probs_by_state["E1"], 6),
        "prob_E2": round(probs_by_state["E2"], 6),
        "prob_E3": round(probs_by_state["E3"], 6),
        "max_posterior": round(max(probs_by_state.values()), 6),
        "clinical_prompt": CLINICAL_PROMPTS[best_state],
    }


def score_csv_file(
    input_csv: Path,
    output_csv: Path,
    cfg: dict,
    european_units: bool = False,
    sicdb_censored_inr: bool = False,
) -> int:
    with Path(input_csv).open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        assert reader.fieldnames is not None, "Input CSV must contain a header row."
        req = ["wbc", "plt", "hgb", "inr", "bicarb", "glu"]
        missing = [c for c in req if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")
        out_fields = list(reader.fieldnames) + ["assigned_state", "prob_E1", "prob_E2", "prob_E3", "max_posterior"]
        rows_out: List[dict] = []
        for row in reader:
            res = score_single_panel(
                wbc=float(row["wbc"]),
                plt=float(row["plt"]),
                hgb=float(row["hgb"]),
                inr=float(row["inr"]),
                bicarb=float(row["bicarb"]),
                glu=float(row["glu"]),
                cfg=cfg,
                european_units=european_units,
                sicdb_censored_inr=sicdb_censored_inr,
            )
            merged = dict(row)
            for k in ("assigned_state", "prob_E1", "prob_E2", "prob_E3", "max_posterior"):
                merged[k] = res[k]
            rows_out.append(merged)

    with Path(output_csv).open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-refit six-test host-response state calculator after acute TBI.")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to locked_mimic_tbi_endotype_k3.json")
    ap.add_argument("--wbc", type=float, help="White blood cell count (x10^9/L)")
    ap.add_argument("--plt", type=float, help="Platelet count (x10^9/L)")
    ap.add_argument("--hgb", type=float, help="Haemoglobin (g/dL; or g/L with --european-units)")
    ap.add_argument("--inr", type=float, help="International normalised ratio (INR)")
    ap.add_argument("--bicarb", type=float, help="Serum bicarbonate (mmol/L)")
    ap.add_argument("--glu", type=float, help="Blood glucose (mg/dL; or mmol/L with --european-units)")
    ap.add_argument("--input-csv", type=Path, help="Batch input CSV with columns: wbc,plt,hgb,inr,bicarb,glu")
    ap.add_argument("--output-csv", type=Path, help="Batch output CSV path")
    ap.add_argument("--european-units", action="store_true", help="Convert Hgb from g/L (/10) and Glu from mmol/L (*18.0182)")
    ap.add_argument("--sicdb-censored-inr", action="store_true", help="Integrate normal CDF over (0.70, 1.20] when INR <= 1.20")
    args = ap.parse_args()

    cfg = load_frozen_model(args.model)
    if args.input_csv is not None:
        if args.output_csv is None:
            ap.error("--output-csv is required when --input-csv is provided.")
        n = score_csv_file(args.input_csv, args.output_csv, cfg, args.european_units, args.sicdb_censored_inr)
        print(json.dumps({"status": "ok", "rows_scored": n, "output_csv": str(args.output_csv)}, indent=2))
        return

    single_vals = [args.wbc, args.plt, args.hgb, args.inr, args.bicarb, args.glu]
    if any(v is None for v in single_vals):
        ap.error("Provide either --input-csv/--output-csv OR all six test values (--wbc --plt --hgb --inr --bicarb --glu).")

    out = score_single_panel(
        wbc=args.wbc,
        plt=args.plt,
        hgb=args.hgb,
        inr=args.inr,
        bicarb=args.bicarb,
        glu=args.glu,
        cfg=cfg,
        european_units=args.european_units,
        sicdb_censored_inr=args.sicdb_censored_inr,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
