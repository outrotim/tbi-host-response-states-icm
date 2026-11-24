#!/usr/bin/env python3
"""Automated unit, mathematical-parity, and zero-PHI repository verification suite."""
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import score_tbi_six_test_states as scorer  # noqa: E402


class TestFrozenTbiSixTestClassifier(unittest.TestCase):
    def setUp(self):
        self.model_path = REPO_ROOT / "model" / "locked_mimic_tbi_endotype_k3.json"
        self.cfg = scorer.load_frozen_model(self.model_path)

    def test_frozen_model_sha256_integrity(self):
        expected_sha = "0410cac81115c7f37998860b97a2278214982250d9e62b8ebd84fe6fe4b42c50"
        actual_sha = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.assertEqual(actual_sha, expected_sha)

    def test_archetypal_clinical_medians_classify_into_E1_E2_E3(self):
        # MIMIC-IV published state medians (ESM 1: Table S1A)
        r_e1 = scorer.score_single_panel(wbc=10.5, plt=208, hgb=12.6, inr=1.10, bicarb=24.0, glu=119, cfg=self.cfg)
        r_e2 = scorer.score_single_panel(wbc=16.3, plt=302, hgb=12.5, inr=1.10, bicarb=22.0, glu=172, cfg=self.cfg)
        r_e3 = scorer.score_single_panel(wbc=8.5, plt=144, hgb=10.2, inr=1.50, bicarb=23.0, glu=133, cfg=self.cfg)

        self.assertEqual(r_e1["assigned_state"], "E1")
        self.assertEqual(r_e2["assigned_state"], "E2")
        self.assertEqual(r_e3["assigned_state"], "E3")

        for r in (r_e1, r_e2, r_e3):
            total_p = r["prob_E1"] + r["prob_E2"] + r["prob_E3"]
            self.assertAlmostEqual(total_p, 1.0, places=5)
            self.assertGreater(r["max_posterior"], 0.60)

    def test_european_si_unit_conversion_parity(self):
        us = scorer.score_single_panel(wbc=8.5, plt=144, hgb=10.2, inr=1.50, bicarb=23.0, glu=133.0, cfg=self.cfg)
        eu = scorer.score_single_panel(
            wbc=8.5, plt=144, hgb=102.0, inr=1.50, bicarb=23.0, glu=133.0 / 18.0182,
            cfg=self.cfg, european_units=True,
        )
        self.assertEqual(us["assigned_state"], eu["assigned_state"])
        self.assertAlmostEqual(us["prob_E3"], eu["prob_E3"], places=5)

    def test_sicdb_left_censored_inr_070_to_120_integral(self):
        cens = scorer.score_single_panel(
            wbc=9.8, plt=205, hgb=13.0, inr=1.20, bicarb=23.5, glu=118,
            cfg=self.cfg, sicdb_censored_inr=True,
        )
        uncens = scorer.score_single_panel(
            wbc=9.8, plt=205, hgb=13.0, inr=1.20, bicarb=23.5, glu=118,
            cfg=self.cfg, sicdb_censored_inr=False,
        )
        self.assertEqual(cens["assigned_state"], "E1")
        self.assertEqual(uncens["assigned_state"], "E1")
        self.assertNotAlmostEqual(cens["prob_E1"], uncens["prob_E1"], places=4)

    def test_batch_csv_scoring_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_p = Path(tmp) / "synthetic_in.csv"
            out_p = Path(tmp) / "synthetic_out.csv"
            with in_p.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["case_label", "wbc", "plt", "hgb", "inr", "bicarb", "glu"])
                w.writeheader()
                w.writerow({"case_label": "syn_E1", "wbc": 10.5, "plt": 208, "hgb": 12.6, "inr": 1.10, "bicarb": 24.0, "glu": 119})
                w.writerow({"case_label": "syn_E3", "wbc": 8.5, "plt": 110, "hgb": 9.5, "inr": 1.65, "bicarb": 21.0, "glu": 140})
            n = scorer.score_csv_file(in_p, out_p, self.cfg)
            self.assertEqual(n, 2)
            with out_p.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["assigned_state"], "E1")
            self.assertEqual(rows[1]["assigned_state"], "E3")

    def test_zero_personal_paths_and_zero_patient_level_files(self):
        forbidden_tokens = ["/Users/", "/Volumes/", "subject_id,", "patientunitstayid,", "hadm_id,"]
        for f in REPO_ROOT.rglob("*"):
            if not f.is_file() or ".git" in f.parts or "__pycache__" in f.parts:
                continue
            if f.name == "test_bedside_scorer_and_privacy.py":
                continue
            txt = f.read_text(encoding="utf-8", errors="ignore")
            for tok in forbidden_tokens:
                self.assertNotIn(tok, txt, f"Forbidden token {tok!r} found in {f.relative_to(REPO_ROOT)}")
        for csv_f in (REPO_ROOT / "reference_outputs").glob("*.csv"):
            with csv_f.open("r", encoding="utf-8") as fh:
                n_rows = sum(1 for _ in fh) - 1
            self.assertLessEqual(n_rows, 300, f"Aggregate table {csv_f.name} has {n_rows} rows (expected <= 300 summary rows)")


if __name__ == "__main__":
    unittest.main()
