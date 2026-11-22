# Portability: three path constants

The scripts were executed on macOS with absolute paths. They are published unmodified so that
their hashes remain identical to the versions that produced the published tables. To run them
elsewhere, change only the path constants at the top of each script:

| Constant | Meaning | Example |
|---|---|---|
| `ROOT` | Project root containing `derived/` | `/data/tbi_project` |
| `EXT_MIMIC` | Directory containing `mimiciv_v31/`, `mimic_iv_ecg/`, `mimic_iv_note/` | `/data/mimic` |
| `SIC_DIR` | Extracted SICdb 1.0.8 directory | `/data/sicdb` |

`safe_patch_paths.py` rewrites these three constants in a copy of the scripts and prints the
before/after SHA-256 so the change is auditable:

```bash
python tools/safe_patch_paths.py --root /data/tbi_project --mimic /data/mimic --sicdb /data/sicdb
```

## Known portability notes

- Scripts 21–24 assume a POSIX filesystem and Python ≥ 3.9 (tested on 3.9 and 3.12).
- `23_*.py` reads MIMIC-IV-Note `radiology.csv.gz` by streaming; it needs ~2 GB free RAM.
- `22_*.py` scans `labevents.csv.gz` (hundreds of millions of rows) — allow 5–15 minutes.
- `21_*.py` writes its outputs to `OUT_DIR`; point it at a writable directory.
- All random seeds are fixed in-script; reruns are deterministic on the same input files.

## Note on absolute paths

`analysis/` scripts are published **unmodified** so that their SHA-256 values match the versions
that produced the published tables (see `docs/REPRODUCIBILITY.md`). As a consequence they still
contain the original execution environment's absolute paths, including the author's home
directory and volume names. These are inert strings; they are not credentials and grant no
access. Run `tools/safe_patch_paths.py` to rewrite them for your own environment.
