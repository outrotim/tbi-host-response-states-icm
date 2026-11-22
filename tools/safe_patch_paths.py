#!/usr/bin/env python3
"""Rewrite the three path constants in a copy of the analysis scripts, audibly.

Prints the SHA-256 of each file before and after patching so the modification is traceable.
"""
import argparse
import hashlib
import re
import shutil
from pathlib import Path

KEYS = ("ROOT", "EXT_MIMIC", "SIC_DIR")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--sicdb", required=True)
    ap.add_argument("--src", default="analysis")
    ap.add_argument("--dst", default="analysis_patched")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    repl = {"ROOT": a.root, "EXT_MIMIC": a.mimic, "SIC_DIR": a.sicdb}
    for f in sorted(src.glob("*.py")):
        text = f.read_text()
        before = hashlib.sha256(text.encode()).hexdigest()
        for key, val in repl.items():
            text = re.sub(rf'^{key} = Path\("[^"]*"\)',
                          f'{key} = Path("{val}")', text, flags=re.M)
        out = dst / f.name
        out.write_text(text)
        print(f"{f.name}: {before[:16]} -> {sha(out)[:16]}")
    print(f"patched copies in {dst}")


if __name__ == "__main__":
    main()
