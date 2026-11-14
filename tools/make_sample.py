"""A random sample of real 2025 test applications for the website (public HMDA rows, no identifiers).

Used for: the hero animation (each dot is a real application, sorted by FairKey's risk) and the
"real look-alikes" check (how many of the most similar real applications were actually denied).

    .venv/bin/python tools/make_sample.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hmda_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = 50000

P = pd.read_parquet(ROOT / "data/predictions.parquet")
te = P[P["year"] == 2025].sample(N, random_state=7).reset_index(drop=True)
out = {
    "n": N,
    "note": "Random sample of 2025 HMDA test applications (US home purchase, 50 states + DC). p = FairKey's predicted chance of denial x1000; st = index into states.",
    "states": hmda_data.STATES,
    "st": te["state"].astype(str).map({s: i for i, s in enumerate(hmda_data.STATES)}).astype(int).tolist(),
    "income": te["income_k"].round().astype(int).tolist(),
    "price": te["price_k"].round().astype(int).tolist(),
    "ltv": (te["ltv"] * 10).round().astype(int).tolist(),
    "dti": te["dti"].round().astype(int).tolist(),
    "type": te["loan_type"].astype(int).tolist(),
    "term": te["term_years"].round().astype(int).tolist(),
    "coapp": te["coapp"].astype(int).tolist(),
    "manuf": te["manufactured"].astype(int).tolist(),
    "denied": te["denied"].astype(int).tolist(),
    "p": (te["p"] * 1000).round().astype(int).tolist(),
}
path = ROOT / "web/data/sample.json"
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"{N} rows · denied {np.mean(out['denied']):.3f} · {path.stat().st_size / 1024:.0f} KB → {path}")
