"""FairKey fairness audit. The model never sees race, ethnicity, sex or age — this script checks it anyway.

    python audit.py        # needs train.py first; writes web/data/fairness.json and figures/fairness_*.png

1. Same risk, same outcome?  For applicants FairKey rates the same, are some groups denied more often in real life?
   (observed ÷ expected denials, with 95% bootstrap intervals — tested on 2025, never used for training)
2. Same mistakes?            Who gets wrongly flagged as high risk, and whose denials does it miss?
3. Hidden proxies?           Can a second model guess an applicant's race from FairKey's inputs alone?
4. Where you live:           versions that may see the state, the county or census-tract data vs the one we ship.
5. State by state:           the lender gap (Black and Hispanic vs White applicants with the same predicted risk).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import features as F  # noqa: E402
import train as T  # noqa: E402

FIG = Path("figures")
OUT = Path("web/data/fairness.json")
B = 1000
HIGH = None   # set from meta bands


def boot_sums(y, p, rng, k=4000):
    """Bootstrap of sum(y)/sum(p) without a B × n matrix: rows go into k random blocks, blocks are resampled
    (Poisson weights). Exact for small groups (one row per block), and fine for the 1.5M-row national ones."""
    n = len(y)
    if n <= k:
        Y, P, N = y, p, np.ones(n)
    else:
        blk = rng.integers(0, k, n)
        Y, P, N = np.bincount(blk, y, k), np.bincount(blk, p, k), np.bincount(blk, None, k).astype(float)
    W = rng.poisson(1.0, (B, len(Y))).astype(float)
    return (W @ Y) / (W @ N), (W @ P) / (W @ N)


def oe_table(d: pd.DataFrame, col: str, order: list[str], seed=0) -> list[dict]:
    """Observed vs expected denials per group, with Poisson-bootstrap 95% intervals."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in order:
        m = d[col].eq(g).to_numpy()
        if m.sum() < 200:
            continue
        y, p = d.loc[m, "denied"].to_numpy().astype(float), d.loc[m, "p"].to_numpy()
        obs_b, exp_b = boot_sums(y, p, rng)
        ratio_b = obs_b / exp_b
        flag = p >= HIGH
        neg, pos = y == 0, y == 1
        rows.append({
            "group": g, "n": int(m.sum()), "observed": float(y.mean()), "expected": float(p.mean()),
            "ratio": float(y.mean() / p.mean()), "ratio_ci": [float(np.percentile(ratio_b, 2.5)),
                                                              float(np.percentile(ratio_b, 97.5))],
            "gap_pts": float(100 * (y.mean() - p.mean())),
            "false_flag": float(flag[neg].mean()),      # approved, but FairKey said "high risk"
            "caught": float(flag[pos].mean()),          # denied, and FairKey had said "high risk"
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        })
    return rows


def calib_by_group(d, col, groups, q=8):
    out = {}
    edges = np.quantile(d["p"], np.linspace(0, 1, q + 1))
    edges[-1] += 1e-9
    for g in groups:
        s = d[d[col].eq(g)]
        b = np.digitize(s["p"], edges[1:-1])
        out[g] = [{"pred": float(s.p[b == i].mean()), "actual": float(s.denied[b == i].mean()), "n": int((b == i).sum())}
                  for i in range(q) if (b == i).sum() >= 50]
    return out


def proxy_test(d, cats, sets: dict) -> dict:
    """Train a classifier to guess 'Black or Hispanic' vs 'White' from each set of inputs. AUC 0.5 = no clue."""
    tr, va, te = T.split(d[d.race.isin(["White", "Black", "Hispanic"])])
    res = {}
    for name, feats in sets.items():
        tr_, va_, te_ = (x.assign(minority=x.race.ne("White").astype(int)) for x in (tr, va, te))
        b = T.fit_lgb(feats, tr_, va_, cats, y="minority", monotone={}, rounds=2000)
        s = b.predict(F.as_model_frame(te_, feats, cats), num_iteration=b.best_iteration)
        res[name] = {"auc": float(roc_auc_score(te_.minority, s)), "features": feats}
        print(f"  proxy {name:22s} AUC {res[name]['auc']:.3f}")
    return res


FIN = F.NUMERIC + F.FLAGS + ["loan_type"]
VARIANTS = {
    "finances": FIN,
    "shipped": F.FEATURES,                                   # finances + lender
    "shipped+state": F.FEATURES + ["state"],
    "shipped+county": F.FEATURES + ["county"],
    "neighbourhood": F.FEATURES + ["county"] + F.LOCAL,
}


def by_state(te: pd.DataFrame, seed=1) -> dict:
    """Per state, 2025: for applicants with the same predicted risk, how much more often were Black and Hispanic
    applicants denied than White applicants? (observed ÷ expected for the group, divided by the same for White)"""
    rng = np.random.default_rng(seed)
    out = {}
    for st, g in te.groupby("state", observed=True):
        row = {"n": int(len(g)), "observed": float(g.denied.mean()), "expected": float(g.p.mean())}
        w = g[g.race.eq("White")]
        yw, pw = w.denied.to_numpy().astype(float), w.p.to_numpy()
        ow, ew = boot_sums(yw, pw, rng)
        row["white"] = {"n": int(len(w)), "obs": float(yw.mean()) if len(w) else None,
                        "exp": float(pw.mean()) if len(w) else None}
        for grp, key in (("Black", "black"), ("Hispanic", "hispanic")):
            x = g[g.race.eq(grp)]
            row["n_" + key] = int(len(x))
            if len(x) < 300 or len(w) < 300:
                continue
            y, p = x.denied.to_numpy().astype(float), x.p.to_numpy()
            o, e = boot_sums(y, p, rng)
            rb = (o / e) / (ow / ew)
            row[key] = {"ratio": float((y.mean() / p.mean()) / (yw.mean() / pw.mean())),
                        "ci": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                        "obs": float(y.mean()), "exp": float(p.mean())}
        out[str(st)] = row
    return out


def neighbourhood_test(d, cats) -> dict:
    """Ship-or-not experiment: does adding census-tract data buy accuracy, and at what fairness cost?"""
    tr, va, te = T.split(d)
    variants = VARIANTS
    y = te.denied.to_numpy()
    out = {}
    white, black, hisp = (te.race.eq(g).to_numpy() for g in ("White", "Black", "Hispanic"))
    for name, feats in variants.items():
        b = T.fit_lgb(feats, tr, va, cats)
        a_, c_ = T.fit_platt(T.raw_score(b, va, feats, cats), va.denied.to_numpy())
        p = T.sigmoid(a_ * T.raw_score(b, te, feats, cats) + c_)
        r = {"ap": T.ap_fast(p, y), "features": feats,
             "pred_gap_black": float(100 * (p[black].mean() - p[white].mean())),
             "pred_gap_hispanic": float(100 * (p[hisp].mean() - p[white].mean())),
             "ratio": {g: float(y[m].mean() / p[m].mean()) for g, m in (("White", white), ("Black", black),
                                                                       ("Hispanic", hisp))},
             "false_flag": {g: float((p[m & (y == 0)] >= HIGH).mean()) for g, m in (("White", white),
                                                                                   ("Black", black),
                                                                                   ("Hispanic", hisp))}}
        if "tract_minority_pct" in feats:
            # twin test: the same families, moved between a mostly-white and a mostly-minority census tract
            lo, hi = np.percentile(d["tract_minority_pct"].dropna(), [10, 90])
            X = F.as_model_frame(te, feats, cats)
            p_lo = T.sigmoid(a_ * b.predict(X.assign(tract_minority_pct=lo), raw_score=True,
                                            num_iteration=b.best_iteration) + c_)
            p_hi = T.sigmoid(a_ * b.predict(X.assign(tract_minority_pct=hi), raw_score=True,
                                            num_iteration=b.best_iteration) + c_)
            r["twin"] = {"minority_lo": float(lo), "minority_hi": float(hi), "risk_lo": float(p_lo.mean()),
                         "risk_hi": float(p_hi.mean()), "pts": float(100 * (p_hi - p_lo).mean()),
                         "share_higher": float((p_hi > p_lo + 1e-9).mean())}
        out[name] = r
        print(f"  variant {name:16s} PR-AUC {r['ap']:.4f}  predicted gap Black {r['pred_gap_black']:+.2f} pts"
              f"  ratios {', '.join(f'{k} {v:.2f}' for k, v in r['ratio'].items())}"
              f"  false-flag {', '.join(f'{k} {v:.3f}' for k, v in r['false_flag'].items())}"
              + (f"  twin +{r['twin']['pts']:.2f} pts" if "twin" in r else ""))
    out["actual_gap_black"] = float(100 * (y[black].mean() - y[white].mean()))
    out["actual_gap_hispanic"] = float(100 * (y[hisp].mean() - y[white].mean()))
    return out


def figure(res, cal):
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    rows = [r for r in res["race"] if r["group"] in ("White", "Asian", "Hispanic", "Black")]
    names = [r["group"] for r in rows]
    obs = [100 * r["observed"] for r in rows]
    exp = [100 * r["expected"] for r in rows]
    x = np.arange(len(rows))
    ax[0].bar(x - .18, exp, .36, color="#9fb3c8", label="Expected from finances (FairKey)")
    ax[0].bar(x + .18, obs, .36, color="#0f4c5c", label="Actually denied (2025)")
    for i, r in enumerate(rows):
        ax[0].text(i + .18, obs[i] + .4, f"×{r['ratio']:.2f}", ha="center", fontsize=10, color="#0f4c5c")
    ax[0].set_xticks(x, names)
    ax[0].set_ylabel("% of applications denied")
    ax[0].set_title("Same finances, different outcomes")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].spines[["top", "right"]].set_visible(False)
    colors = {"White": "#9fb3c8", "Hispanic": "#e9a03b", "Black": "#0f4c5c", "Asian": "#7a9e7e"}
    for g, pts in cal.items():
        ax[1].plot([100 * p["pred"] for p in pts], [100 * p["actual"] for p in pts], "o-", color=colors.get(g, "#888"),
                   label=g, lw=2, ms=4)
    lim = max(max(100 * p["actual"] for pts in cal.values() for p in pts), 40)
    ax[1].plot([0, lim], [0, lim], "--", color="#bbb", lw=1)
    ax[1].set_xlabel("FairKey's predicted risk (%)")
    ax[1].set_ylabel("Actually denied (%)")
    ax[1].set_title("For the same predicted risk, who is denied more?")
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "HMDA 2025, US home-purchase applications (50 states + DC) the model never saw. "
             "FairKey does not see race or ethnicity.", fontsize=8.5, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fairness_race.png", dpi=160)
    plt.close(fig)


def main():
    global HIGH
    meta = json.loads(Path("web/data/meta.json").read_text())
    HIGH = meta["bands"]["high"]
    pr = pd.read_parquet("data/predictions.parquet")
    te = pr[pr.year.isin(T.TEST)]
    res = {"high_threshold": HIGH, "n_test": int(len(te)),
           "race": oe_table(te, "race", F.RACE_ORDER), "sex": oe_table(te, "sex", F.SEX_ORDER),
           "age": oe_table(te, "age", F.AGE_ORDER)}
    for k in ("race", "sex", "age"):
        for r in res[k]:
            print(f"  {k:4s} {r['group']:14s} n={r['n']:6d} obs {r['observed']:.3f} exp {r['expected']:.3f} "
                  f"×{r['ratio']:.2f} [{r['ratio_ci'][0]:.2f}-{r['ratio_ci'][1]:.2f}] "
                  f"false-flag {r['false_flag']:.3f} caught {r['caught']:.3f} AUC {r['auc']:.3f}")
    cal = calib_by_group(te, "race", ["White", "Asian", "Hispanic", "Black"])
    res["calibration_race"] = cal
    figure(res, cal)

    res["states"] = by_state(te)
    for st, r in sorted(res["states"].items(), key=lambda kv: -kv[1]["n"])[:12]:
        b = r.get("black")
        print(f"  state {st} n={r['n']:7d} obs {r['observed']:.3f} exp {r['expected']:.3f}"
              + (f"  Black×{b['ratio']:.2f} [{b['ci'][0]:.2f}-{b['ci'][1]:.2f}]" if b else ""))
    OUT.write_text(json.dumps(res, indent=1))             # partial results first: the experiments below take a while

    d, cats = T.load_data()
    res["proxy"] = proxy_test(d, cats, VARIANTS)
    OUT.write_text(json.dumps(res, indent=1))
    res["neighbourhood"] = neighbourhood_test(d, cats)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
