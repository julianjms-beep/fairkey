"""Train FairKey: would a US lender deny this home-purchase mortgage — and why?  (50 states + DC)

    python train.py        # trains, tests once on 2025, writes models/ and web/data/{model,meta,evidence}.json

Honest time split: learn from 2022-2023, tune and calibrate on 2024, test ONCE on 2025 (applications it never saw).
The model in the browser is exactly the model tested here.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer

import features as F
import hmda_data

TRAIN, VALID, TEST = (2022, 2023), (2024,), (2025,)
SEED = 0
N_LENDERS = 50
LOGIT_SAMPLE = 1_500_000   # the logistic baseline (~110 parameters) learns from a random 1.5M training rows
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, cat_l2=10, max_cat_to_onehot=8,
              seed=SEED, verbose=-1, deterministic=True, force_col_wise=True, num_threads=8)
# Trust: more debt or a smaller down payment can never LOWER the predicted risk, and vice versa.
MONOTONE = {"dti": 1, "ltv": 1}
WEB = Path("web/data")
MODELS = Path("models")


# ---------------------------------------------------------------- data
RAW_COLS = ["activity_year", "action_taken", "income", "loan_amount", "property_value", "combined_loan_to_value_ratio",
            "debt_to_income_ratio", "loan_term", "loan_type", "co_applicant_age", "construction_method", "preapproval",
            "total_units", "state_code", "county_code", "lei", "tract_to_msa_income_percentage",
            "tract_minority_population_percent", "tract_median_age_of_housing_units", "derived_race",
            "derived_ethnicity", "derived_sex", "applicant_age", "denial_reason_1", "interest_rate"]


def load_data() -> tuple[pd.DataFrame, dict]:
    raw = hmda_data.load(RAW_COLS)
    d = F.prepare(raw)
    del raw
    tops = d.loc[d.year.isin(TRAIN), "lei"].value_counts().head(N_LENDERS).index.tolist()
    d["lender"] = np.where(d["lei"].isin(tops), d["lei"], "other")
    for c in ("state", "county", "lei", "lender", "units", "race", "sex", "age"):      # compact: 15M rows
        d[c] = d[c].astype("category")
    cats = {"loan_type": [1, 2, 3, 4], "state": list(hmda_data.STATES),
            "county": sorted(map(str, d["county"].cat.categories)), "lender": tops + ["other"]}
    return d, cats


def split(d):
    return d[d.year.isin(TRAIN)], d[d.year.isin(VALID)], d[d.year.isin(TEST)]


# ---------------------------------------------------------------- models
def fit_lgb(feats, tr, va, cats, y="denied", params=PARAMS, monotone=MONOTONE, rounds=4000):
    Xtr, Xva = F.as_model_frame(tr, feats, cats), F.as_model_frame(va, feats, cats)
    p = {**params, "metric": ["average_precision", "binary_logloss"],
         "monotone_constraints": [monotone.get(f, 0) for f in feats], "monotone_constraints_method": "advanced"}
    ds = lgb.Dataset(Xtr, tr[y], categorical_feature=[f for f in feats if f in cats], free_raw_data=False)
    dv = lgb.Dataset(Xva, va[y], reference=ds)
    return lgb.train(p, ds, rounds, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])


def raw_score(b, d, feats, cats):
    return b.predict(F.as_model_frame(d, feats, cats), raw_score=True, num_iteration=b.best_iteration)


def fit_platt(raw, y):
    """Calibration on 2024: p = sigmoid(a * raw + b). Two numbers, smooth, keeps the ranking."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(raw.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_baseline(tr, cats):
    """The classic credit-scoring model: logistic regression (with smooth curves for the numbers)."""
    logn = ["income_k", "loan_k", "price_k", "loan_to_income"]
    oh = [c for c in ("loan_type", "county", "lender") if c in F.FEATURES]
    lin = ["ltv", "dti", "term_years"]
    ct = ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p), SplineTransformer(n_knots=6, degree=3)), logn),
        ("spl", SplineTransformer(n_knots=6, degree=3), lin),
        ("oh", OneHotEncoder(categories=[cats[c] for c in oh], handle_unknown="ignore"), oh),
        ("flags", "passthrough", F.FLAGS)])
    m = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=3000))
    s = tr.sample(min(LOGIT_SAMPLE, len(tr)), random_state=SEED)
    m.fit(s[F.FEATURES], s["denied"])
    return m


def predict_batches(m, X, size=400_000):
    return np.concatenate([m.predict_proba(X.iloc[i:i + size])[:, 1] for i in range(0, len(X), size)])


def rule_flag(d):
    """Rule of thumb: debt-to-income above 43% (the old 'Qualified Mortgage' limit)."""
    return (d["dti"] > 43).astype(int).to_numpy()


# ---------------------------------------------------------------- measures
def ap_fast(score, y, w=None):
    """Average precision (area under the precision-recall curve), tie-aware, with optional bootstrap weights."""
    o = np.argsort(-score, kind="mergesort")
    s, yy = score[o], y[o]
    ww = np.ones_like(s) if w is None else w[o]
    tp, fp = np.cumsum(ww * yy), np.cumsum(ww * (1 - yy))
    last = np.r_[np.diff(s) != 0, True]
    tp, fp = tp[last], fp[last]
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(prec * np.diff(np.r_[0, tp])) / tp[-1])


def bootstrap(scores: dict, y, n=1000, seed=SEED):
    """Poisson bootstrap of PR-AUC. Each score is sorted once; every replicate only re-weights the rows."""
    rng = np.random.default_rng(seed)
    pre = {}
    for k, s in scores.items():
        o = np.argsort(-s, kind="mergesort")
        ss = s[o]
        pre[k] = (o, y[o], np.r_[np.diff(ss) != 0, True])
    out = {k: np.empty(n) for k in scores}
    for i in range(n):
        w = rng.poisson(1.0, len(y)).astype(np.float32)
        for k, (o, yy, last) in pre.items():
            ww = w[o]
            tp, fp = np.cumsum(ww * yy)[last], np.cumsum(ww * (1 - yy))[last]
            prec = tp / np.maximum(tp + fp, 1e-12)
            out[k][i] = np.sum(prec * np.diff(np.r_[0, tp])) / tp[-1]
    return out


def pr_points(score, y, n=60):
    o = np.argsort(-score, kind="mergesort")
    yy = y[o]
    tp = np.cumsum(yy)
    k = np.unique(np.linspace(1, len(y), 400).astype(int)) - 1
    rec, prec = tp[k] / tp[-1], tp[k] / (k + 1)
    keep = np.unique(np.searchsorted(rec, np.linspace(0.01, 1, n)).clip(0, len(rec) - 1))
    return [[round(float(rec[i]), 4), round(float(prec[i]), 4)] for i in keep]


def calibration_bins(p, y, edges=(0, .03, .05, .08, .12, .16, .2, .3, .4, .5, .7, 1.0001)):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 50:
            out.append({"lo": lo, "hi": round(min(hi, 1), 2), "n": int(m.sum()), "pred": round(float(p[m].mean()), 4),
                        "actual": round(float(y[m].mean()), 4)})
    return out


def at_recall(score, y, target):
    """Threshold that catches `target` of the denials; return precision and share of applications flagged."""
    o = np.argsort(-score, kind="mergesort")
    tp = np.cumsum(y[o])
    k = int(np.searchsorted(tp, target * tp[-1])) + 1
    return {"precision": float(tp[k - 1] / k), "recall": float(tp[k - 1] / tp[-1]), "flagged": k / len(y),
            "threshold": float(score[o][k - 1])}


# ---------------------------------------------------------------- export for the browser
def nice_name(raw: str, overrides: dict) -> str:
    """'ROCKET MORTGAGE, LLC' → 'Rocket Mortgage' (hand-written names first)."""
    if raw in overrides:
        return overrides[raw]
    n = str(raw).strip()
    for suf in (", LLC", " LLC", ", L.L.C.", ", Inc.", ", INC.", " INC.", ", INC", " INC", " Inc.", ", N.A.", " N.A.",
                ", LTD.", " LTD.", " CORPORATION", " Corporation", " CORP", ", a Corporation", ","):
        if n.endswith(suf):
            n = n[: -len(suf)].strip()
    if n.isupper():
        n = " ".join(w if len(w) <= 3 and w in ("FHA", "VA", "USA", "PNC", "TD", "BMO") else w.capitalize() for w in n.split())
    return n


def export_trees(b, n_iter=None, counts=True):
    """LightGBM trees → compact arrays the browser walks (and explains with TreeSHAP, which needs the counts)."""
    dump = b.dump_model(num_iteration=n_iter)
    trees = []
    for t in dump["tree_info"]:
        F_, T, L, R, V, C, DL, MT, CS = [], [], [], [], [], [], [], [], []

        def walk(n):
            i = len(F_)
            for a in (F_, T, L, R, V, C, DL, MT):
                a.append(0)
            if "leaf_value" in n or "split_feature" not in n:
                F_[i], V[i], C[i] = -1, n.get("leaf_value", 0.0), n.get("leaf_count", 0)
                L[i] = R[i] = -1
                return i
            F_[i], V[i], C[i] = n["split_feature"], 0, n["internal_count"]
            DL[i] = 1 if n["default_left"] else 0
            MT[i] = {"None": 0, "Zero": 1, "NaN": 2}[n["missing_type"]]
            if n["decision_type"] == "==":
                CS.append(sorted(int(c) for c in str(n["threshold"]).split("||")))
                T[i] = -(len(CS))          # negative = index into category sets (1-based)
            else:
                T[i] = float(f"{float(n['threshold']):.9g}")
            L[i] = walk(n["left_child"])
            R[i] = walk(n["right_child"])
            return i

        walk(t["tree_structure"])
        t = {"f": F_, "t": T, "l": L, "r": R, "v": [round(x, 7) for x in V], "dl": DL, "mt": MT, "cs": CS}
        if counts:
            t["c"] = C
        trees.append(t)
    return {"num_class": dump.get("num_class", 1), "trees": trees}


def main():
    MODELS.mkdir(exist_ok=True)
    WEB.mkdir(parents=True, exist_ok=True)
    d, cats = load_data()
    tr, va, te = split(d)
    n_complete = len(d)
    del d
    feats = F.FEATURES
    print(f"train {len(tr):,} · valid {len(va):,} · test {len(te):,} · denial rates "
          f"{tr.denied.mean():.3f} / {va.denied.mean():.3f} / {te.denied.mean():.3f}")

    # 1) the risk model
    b = fit_lgb(feats, tr, va, cats)
    print("best iteration", b.best_iteration)
    a_, b_ = fit_platt(raw_score(b, va, feats, cats), va.denied.to_numpy())
    p_te = sigmoid(a_ * raw_score(b, te, feats, cats) + b_)
    p_va = sigmoid(a_ * raw_score(b, va, feats, cats) + b_)
    b.save_model(MODELS / "risk.txt", num_iteration=b.best_iteration)

    # 2) baselines
    lr = logistic_baseline(tr, cats)
    lr_te = predict_batches(lr, te[F.FEATURES])
    y = te.denied.to_numpy()
    dti_score = te["dti"].to_numpy() + 1e-6 * te["ltv"].to_numpy()       # rule of thumb as a ranking
    rule = rule_flag(te)

    scores = {"fairkey": p_te, "logistic": lr_te, "dti": dti_score}
    bs = bootstrap(scores, y.astype(float), n=1000)
    metrics = {}
    for k, s in scores.items():
        metrics[k] = {"ap": ap_fast(s, y), "ap_ci": [float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))],
                      "roc_auc": float(roc_auc_score(y, s))}
    for k in ("fairkey", "logistic"):
        s = scores[k]
        metrics[k].update(brier=float(brier_score_loss(y, s)), logloss=float(log_loss(y, s)))
    metrics["base_rate"] = float(y.mean())
    metrics["brier_constant"] = float(brier_score_loss(y, np.full_like(p_te, tr.denied.mean())))
    diff = bs["fairkey"] - bs["logistic"]
    metrics["fairkey_minus_logistic"] = {"mean": float(diff.mean()), "ci": [float(np.percentile(diff, 2.5)),
                                                                            float(np.percentile(diff, 97.5))]}
    diff2 = bs["fairkey"] - bs["dti"]
    metrics["fairkey_minus_dti"] = {"mean": float(diff2.mean()), "ci": [float(np.percentile(diff2, 2.5)),
                                                                        float(np.percentile(diff2, 97.5))]}
    # accuracy is useless here: say "approved" to everyone and you are right 87% of the time
    metrics["accuracy_all_approve"] = float(1 - y.mean())

    rule_prec, rule_rec = float(y[rule == 1].mean()), float(rule[y == 1].mean())
    same = at_recall(p_te, y, rule_rec)
    lr_same = at_recall(lr_te, y, rule_rec)
    ops = {"rule": {"precision": rule_prec, "recall": rule_rec, "flagged": float(rule.mean())},
           "fairkey_same_recall": same, "logistic_same_recall": lr_same}
    per1000 = {"denials": round(1000 * y.mean()),
               "rule_flags": round(1000 * rule.mean()), "rule_false": round(1000 * ((rule == 1) & (y == 0)).mean()),
               "fk_flags": round(1000 * same["flagged"]),
               "fk_false": round(1000 * same["flagged"] * (1 - same["precision"]))}
    ops["per1000"] = per1000

    # risk bands shown in the tool, chosen on 2024 and checked on 2025
    bands = {"low": 0.08, "high": 0.25}
    band_check = []
    for name, lo, hi in (("low", 0, bands["low"]), ("medium", bands["low"], bands["high"]), ("high", bands["high"], 1.01)):
        for yr, pp, yy in (("2024", p_va, va.denied.to_numpy()), ("2025", p_te, y)):
            m = (pp >= lo) & (pp < hi)
            band_check.append({"band": name, "year": yr, "share": float(m.mean()), "denied": float(yy[m].mean())})

    # 3) why: which reason a lender would most likely give, if it said no
    trd, vad, ted = (x[x.denied.eq(1) & x.reason.notna()].copy() for x in (tr, va, te))
    rmap = {r: i for i, r in enumerate(F.REASON_ORDER)}
    for x in (trd, vad, ted):
        x["reason_id"] = x["reason"].map(rmap)
    rp = {**PARAMS, "objective": "multiclass", "num_class": len(F.REASON_ORDER), "metric": "multi_logloss",
          "num_leaves": 31, "min_child_samples": 100}
    Xr = F.as_model_frame(trd, feats, cats)
    dr = lgb.Dataset(Xr, trd.reason_id, categorical_feature=[f for f in feats if f in cats], free_raw_data=False)
    dvr = lgb.Dataset(F.as_model_frame(vad, feats, cats), vad.reason_id, reference=dr)
    rb = lgb.train(rp, dr, 3000, valid_sets=[dvr], callbacks=[lgb.early_stopping(100, verbose=False)])
    pr = rb.predict(F.as_model_frame(ted, feats, cats), num_iteration=rb.best_iteration)
    yr_ = ted.reason_id.to_numpy()
    prior = trd.reason_id.value_counts(normalize=True).sort_index().reindex(range(len(F.REASON_ORDER)), fill_value=0)
    top2 = np.argsort(-pr, axis=1)[:, :2]
    reason_metrics = {
        "n_test": int(len(ted)), "top1": float((pr.argmax(1) == yr_).mean()),
        "top2": float(((top2[:, 0] == yr_) | (top2[:, 1] == yr_)).mean()),
        "majority": float((yr_ == int(prior.idxmax())).mean()),
        "logloss": float(log_loss(yr_, pr, labels=range(len(F.REASON_ORDER)))),
        "logloss_prior": float(log_loss(yr_, np.tile(prior.to_numpy(), (len(yr_), 1)), labels=range(len(F.REASON_ORDER)))),
        "prior": {r: float(prior[i]) for i, r in enumerate(F.REASON_ORDER)},
        "per_reason_recall": {r: float((pr.argmax(1)[yr_ == i] == i).mean()) for i, r in enumerate(F.REASON_ORDER)},
    }
    rb.save_model(MODELS / "reasons.txt", num_iteration=rb.best_iteration)

    # 4) importance (gain), for the README and the page
    gain = dict(zip(feats, b.feature_importance("gain", iteration=b.best_iteration)))
    tot = sum(gain.values())
    importance = sorted(((k, v / tot) for k, v in gain.items()), key=lambda kv: -kv[1])

    # ---- export
    lenders = pd.read_csv("data/lenders.csv").set_index("lei")["respondent_name"].to_dict()
    NICE = json.loads(Path("data/lender_names.json").read_text())       # display names for the top lenders
    lender_share = tr["lender"].value_counts(normalize=True).to_dict()
    costs = pd.read_csv("data/state_costs.csv").set_index("state")
    states = {}
    for st, g in tr.groupby("state", observed=True):                   # where people in each state actually apply
        sh = g["lender"].value_counts(normalize=True)
        c = costs.loc[st]
        states[st] = {"name": c["name"], "n": int(len(g)), "denied": round(float(g.denied.mean()), 4),
                      "median_price_k": float(g.price_k.median()), "median_income_k": float(g.income_k.median()),
                      "tax_rate": float(c.tax_rate), "ins_rate": float(c.ins_rate),
                      "lw": {str(k): round(float(v), 5) for k, v in sh.items() if v > 0}}
    cats_web = {k: v for k, v in cats.items() if k != "county"}      # 3,000+ counties: only for the audit
    meta = {
        "features": feats, "cats": cats_web, "monotone": MONOTONE,
        "platt": [a_, b_], "bands": bands, "base_rate_train": float(tr.denied.mean()),
        "expected_raw": float(np.mean(raw_score(b, tr, feats, cats))),
        "lenders": {k: nice_name(lenders.get(k, k), NICE) for k in cats["lender"] if k != "other"},
        "lender_share": {str(k): round(float(v), 5) for k, v in lender_share.items()},
        "states": states,
        "reasons": F.REASON_ORDER, "reason_prior": reason_metrics["prior"],
        "train_years": list(TRAIN), "valid_years": list(VALID), "test_years": list(TEST),
    }
    model = {"risk": export_trees(b, b.best_iteration), "reasons": export_trees(rb, rb.best_iteration, counts=False)}
    evidence = {
        "n": {"train": len(tr), "valid": len(va), "test": len(te), "all_raw": int(hmda_data.count()),
              "complete": n_complete, "states": len(states), "lenders_named": N_LENDERS},
        "metrics": metrics, "operating": ops, "bands_check": band_check,
        "pr": {k: pr_points(s, y) for k, s in scores.items()},
        "calibration": calibration_bins(p_te, y), "calibration_logistic": calibration_bins(lr_te, y),
        "reasons": reason_metrics, "importance": [[k, round(float(v), 4)] for k, v in importance],
        "best_iteration": b.best_iteration, "reason_iterations": rb.best_iteration,
    }
    (WEB / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    (WEB / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    (WEB / "evidence.json").write_text(json.dumps(evidence, indent=1))

    # predictions for the fairness audit (2024 + 2025 only — never the training years)
    keep = ["year", "denied", "race", "sex", "age", "state", "county", "lender", "loan_type", "income_k", "loan_k", "price_k",
            "ltv", "dti", "coapp", "manufactured", "preapproval", "term_years", "loan_to_income",
            "tract_income_pct", "tract_minority_pct", "tract_housing_age", "reason"]
    out = pd.concat([va[keep].assign(p=p_va), te[keep].assign(p=p_te, p_logistic=lr_te)])
    out.to_parquet("data/predictions.parquet", index=False)

    m = metrics
    print(f"TEST 2025  base rate {m['base_rate']:.3f}")
    for k in ("fairkey", "logistic", "dti"):
        print(f"  {k:9s} PR-AUC {m[k]['ap']:.3f} [{m[k]['ap_ci'][0]:.3f}-{m[k]['ap_ci'][1]:.3f}]  ROC {m[k]['roc_auc']:.3f}")
    print(f"  brier fairkey {m['fairkey']['brier']:.4f}  logistic {m['logistic']['brier']:.4f}  constant {m['brier_constant']:.4f}")
    print("  ops", json.dumps(per1000), f"rule prec {rule_prec:.3f} rec {rule_rec:.3f}; fk prec {same['precision']:.3f}")
    print("  bands", [(x["band"], x["year"], round(x["share"], 3), round(x["denied"], 3)) for x in band_check])
    print("  reasons top1 %.3f top2 %.3f majority %.3f  logloss %.3f vs prior %.3f" % (
        reason_metrics["top1"], reason_metrics["top2"], reason_metrics["majority"], reason_metrics["logloss"],
        reason_metrics["logloss_prior"]))
    print("  importance", [(k, round(v, 3)) for k, v in importance])
    print("  model.json KB", round((WEB / "model.json").stat().st_size / 1024))


def export_only():
    """Rewrite web/data/model.json from the saved models (no retraining)."""
    b, rb = lgb.Booster(model_file=str(MODELS / "risk.txt")), lgb.Booster(model_file=str(MODELS / "reasons.txt"))
    model = {"risk": export_trees(b), "reasons": export_trees(rb, counts=False)}
    (WEB / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    print("model.json KB", round((WEB / "model.json").stat().st_size / 1024))


if __name__ == "__main__":
    import sys
    export_only() if "--export" in sys.argv else main()
