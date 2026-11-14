"""Check FairKey's promises against the files that are actually published (run: .venv/bin/python tests/check_guards.py).

1. The model's inputs are only what an applicant knows before applying: no race, ethnicity, sex or age, no field that
   is only known after the decision (interest rate, pre-approval code, underwriting result, denial reasons...), and no
   place below the state (county, census tract).
2. The browser gets the same model as Python: same inputs, same number of trees.
3. Monotone: on real 2025 applications, raising debt-to-income or loan-to-value never lowers the predicted risk.
4. The split is by time: training years < validation years < test years, no overlap.
5. The website's three bands keep their order and hold from 2024 to 2025.
6. The public sample (web/data/sample.json) has no identifiers and no protected attributes.
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb, numpy as np, pandas as pd
import features as F

meta = json.loads((ROOT / "web/data/meta.json").read_text())
ev = json.loads((ROOT / "web/data/evidence.json").read_text())
browser = json.loads((ROOT / "web/data/model.json").read_text())
risk = lgb.Booster(model_file=str(ROOT / "models/risk.txt"))
reasons = lgb.Booster(model_file=str(ROOT / "models/reasons.txt"))
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" · {detail}" if detail else ""))


# 1. inputs ---------------------------------------------------------------------------------------------------------
FORBIDDEN = re.compile(r"race|ethnic|sex|gender|age|interest|rate_spread|preappro|aus|credit_score|denial|reason|"
                       r"action|loan_cost|origination|discount|lender_credit|hoepa|purchaser|county|tract|zip|census",
                       re.I)
for name, feats in (("risk model", risk.feature_name()), ("reason model", reasons.feature_name()),
                    ("browser model", meta["features"])):
    bad = [f for f in feats if FORBIDDEN.search(f)]
    check(f"{name}: only pre-application inputs", not bad and list(feats) == list(F.FEATURES),
          f"{len(feats)} inputs: {', '.join(feats)}" + (f" · FORBIDDEN: {bad}" if bad else ""))

# 2. browser == Python ----------------------------------------------------------------------------------------------
for key, booster in (("risk", risk), ("reasons", reasons)):
    nb, npy = len(browser[key]["trees"]), booster.num_trees()
    check(f"browser {key} model has the same trees as LightGBM", nb == npy, f"{nb} vs {npy}")

# 3. monotone -------------------------------------------------------------------------------------------------------
test_years = [int(y) for y in meta["test_years"]]
P = pd.read_parquet(ROOT / "data/predictions.parquet", columns=list(meta["features"]) + ["year"],
                    filters=[("year", "in", test_years)])
te = P.sample(2000, random_state=7)
del P
X = F.as_model_frame(te, meta["features"], meta["cats"])
for col, grid in (("dti", np.arange(10, 71, 1.0)), ("ltv", np.arange(20, 121, 2.5))):
    if col not in meta.get("monotone", {}):
        check(f"monotone in {col}", False, "not declared monotone in meta.json")
        continue
    big = pd.concat([X.assign(**{col: v}) for v in grid], ignore_index=True)
    raw = risk.predict(big, raw_score=True).reshape(len(grid), len(X))
    worst = float(np.diff(raw, axis=0).min())
    check(f"risk never goes down when {col} goes up", worst >= -1e-9,
          f"{len(X):,} real 2025 applications × {len(grid)} values · largest step down {min(worst, 0):.1e}")

# 4. split by time --------------------------------------------------------------------------------------------------
tr, va, ts = ([int(y) for y in meta[k]] for k in ("train_years", "valid_years", "test_years"))
check("train < validation < test, by year", max(tr) < min(va) and max(va) < min(ts), f"{tr} → {va} → {ts}")

# 5. bands ----------------------------------------------------------------------------------------------------------
b = {(r["band"], r["year"]): r["denied"] for r in ev["bands_check"]}
order = all(b[("low", y)] < b[("medium", y)] < b[("high", y)] for y in ("2024", "2025"))
drift = max(abs(b[(k, "2024")] - b[(k, "2025")]) for k in ("low", "medium", "high"))
check("bands keep their order and hold from 2024 to 2025", order and drift < 0.02,
      " · ".join(f"{k} {b[(k, '2024')]:.1%}→{b[(k, '2025')]:.1%}" for k in ("low", "medium", "high")))

# 6. public sample --------------------------------------------------------------------------------------------------
s = json.loads((ROOT / "web/data/sample.json").read_text())
leaky = [k for k in s if FORBIDDEN.search(k) or re.search(r"lei|lender|id$|^id", k, re.I)]
check("public sample has no identifiers or protected attributes", not leaky, f"fields: {', '.join(s)}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
