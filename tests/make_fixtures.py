"""Write tests/fixtures.json: 500 real 2025 applications + LightGBM's own answers, for checking the browser engine.

Reads the saved test predictions (train.py writes them) and the category lists the website uses (web/data/meta.json),
so it doesn't need to reload the 15 million raw rows.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb, numpy as np, pandas as pd
import features as F

meta = json.loads((ROOT / "web/data/meta.json").read_text())
cats = meta["cats"]
P = pd.read_parquet(ROOT / "data/predictions.parquet")
te = P[P.year == 2025].sample(500, random_state=1)
X = F.as_model_frame(te, meta["features"], cats)
risk = lgb.Booster(model_file=str(ROOT / "models/risk.txt"))
reasons = lgb.Booster(model_file=str(ROOT / "models/reasons.txt"))
enc = np.column_stack([X[c].cat.codes.replace(-1, np.nan).to_numpy(float) if c in cats else X[c].to_numpy(float)
                       for c in meta["features"]])
out = {"x": [[None if np.isnan(v) else float(v) for v in row] for row in enc],
       "raw": risk.predict(X, raw_score=True).tolist(),
       "contrib": risk.predict(X, pred_contrib=True).tolist(),
       "reasons": reasons.predict(X).tolist()}
(ROOT / "tests/fixtures.json").write_text(json.dumps(out))
print("fixtures", len(out["x"]))
