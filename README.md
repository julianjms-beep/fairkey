# FairKey 🔑

**Will a lender say yes to your mortgage — and if not, why, and what would change it?**

> **This project is a prototype and is not financial advice.**

FairKey is a machine-learning tool for people buying a home anywhere in the United States (all 50 states and DC).
You enter what you know *before* you apply (state, income, price, down payment, other debts, loan type, term, and
optionally the lender) and it:

1. **estimates the chance a lender says no**, as a calibrated percentage you can believe;
2. **explains why** with exact TreeSHAP, computed on your own device;
3. **predicts the reason** a lender would most likely give (a second model trained on 613,955 real denials);
4. **suggests what would change it** (more down, fewer debts, a cheaper home, another loan type or lender) and
   draws your **safe zone**: the risk for every price and down payment;
5. **checks itself against reality**: next to your number it shows how many of the 200 most similar *real* 2025
   applications were actually denied (from your own state first);
6. **shows the AI at work**: "Inside the AI" replays the 1,231 decision trees adding up to your answer, and opens any
   tree to show its real yes/no questions, your answers and its vote;
7. and **audits itself for fairness**, nationally and **state by state**: FairKey never sees race, ethnicity, sex or
   age, but we used those columns to check whether it, or the lenders, treat groups differently.

To get started fast there are six example applicants in six states (every result is the live model), a **Watch a
demo** button that walks through them, and a "tell it in your own words" box. That box fills the form with simple word
rules (regular expressions), not a language model; it shows what it understood so you can check it.

Everything runs in the browser: no server, no account, nothing you type leaves your device. English and Spanish.

**Track:** Finance · **Data:** public (HMDA) · **Live demo:** <https://julianjms-beep.github.io/fairkey/> · **Evidence:** [`FairKey.ipynb`](FairKey.ipynb) ([open in Colab](https://colab.research.google.com/github/julianjms-beep/fairkey/blob/main/FairKey.ipynb))

---

## The results, honestly measured

Trained on 2022–2023 (6.8 million applications), calibrated on 2024 (3.0 million), tested **once** on
**3,017,153 applications from 2025** it had never seen.

| | PR-AUC (95% CI) | ROC-AUC | Brier |
|---|---|---|---|
| **FairKey** (LightGBM, monotone) | **0.569** (0.567–0.571) | 0.822 | **0.0529** |
| Logistic regression, same inputs | 0.503 (0.501–0.505) | 0.793 | 0.0582 |
| Debt-to-income alone | 0.335 (0.333–0.337) | 0.688 | — |
| Always guess the average | 0.088 | 0.500 | 0.0800 |

FairKey minus logistic regression: **+0.066 (0.065–0.067)**, 1,000 bootstrap resamples.

**Why not accuracy?** Only 8.8% of applications are denied, so saying "approved" to everyone is 91% accurate while
warning nobody who gets denied. We judge FairKey on what matters to a person:

* **Finding denials without crying wolf (PR-AUC).** Per 1,000 applications, catching the same 63% of denials as
  the common "debt-to-income above 43%" rule, the rule wrongly warns **343** families who were approved; FairKey
  warns **121**.
* **Percentages you can believe (calibration).** Applications FairKey called *likely yes* were denied 3.7% of the time
  in 2025, *uncertain* 11.0%, *high risk* 65.5%. In 2024 the same bands gave 3.7%, 11.1% and 66.5%.
* **The reason model** gets the lender's main reason right **57.5%** of the time (top 2: 73.6%) out of six,
  vs. 37.4% by always guessing the most common reason.

## The three leaks we refused

A model can look brilliant by reading the answer off the form. In HMDA we found three ways:

| Leak | What happens | What we did |
|---|---|---|
| **Interest rate** | HMDA records a rate only for loans that were made: a model that reads it is "99.98% accurate" and useless | never a feature |
| **Blank fields** | 15% of records are incomplete and **41%** of those are denials (the lender stopped filling it in) | complete records only |
| **Pre-approval** | denied pre-approval requests are filed under another code, so **all 740,803** pre-approved applications were approved | removed (it first showed up as a "→ 0%" suggestion in our own what-if tool) |

We also left out the automated-underwriting result and the credit-score model field, which are only known after applying.

## The fairness audit

FairKey is blind to race, ethnicity, sex and age, but **blind is not the same as fair**: other inputs can stand in
for them. On 2025 applications we compared, for every group, how often it was **actually denied** with what FairKey
**expected** from its finances:

| Group | Applications | Denied | Expected | Ratio (95% CI) | Wrongly flagged |
|---|---:|---:|---:|---|---:|
| White | 1,596,980 | 6.4% | 7.4% | ×0.86 (0.85–0.86) | 2.2% |
| Black | 229,912 | 17.3% | 13.9% | **×1.24 (1.23–1.25)** | 4.0% |
| Hispanic | 417,443 | 12.8% | 11.6% | ×1.11 (1.10–1.11) | 2.8% |
| Asian | 200,697 | 8.5% | 8.0% | ×1.07 (1.05–1.08) | 1.4% |

*Wrongly flagged* = approved families FairKey had marked "high risk". The sex and age tables and calibration by group
are on the site and in the notebook: every age group is between ×0.92 and ×1.08 and every sex group between ×0.96 and
×1.01, except applicants who did not give their sex (×1.17).

**What we found:** relative to what their finances predict, real lenders denied Black applicants **×1.45** as often
as White applicants, and Hispanic applicants ×1.29. **State by state** (a tile map on the site), the Black–White gap
is above 1 in every one of the 39 states with enough applications to measure, and the 95% interval is entirely above
1 in all 39. FairKey gives the same answer to the same finances, whoever you are, but it cannot make lenders do the
same. Public data has no credit scores, so part of this gap could be credit history; we can't rule that out, and we
say so.

### Which inputs we let in: one rule, five versions

**Rule: keep an input only if it adds accuracy without mainly encoding race.** We trained five versions on the same
years and tested each on 2025. For each set of inputs, a *proxy test* also tries to guess whether an applicant is Black
or Hispanic rather than White from those inputs alone (0.5 = a coin flip, 1.0 = perfect).

| Version | PR-AUC | Proxy AUC (guesses race) | Wrongly flagged: Black · Hispanic · White | |
|---|---:|---:|---|---|
| Finances only | 0.543 | 0.746 | 3.6% · 2.6% · 2.1% |  |
| **+ lender (shipped)** | **0.569** | **0.757** | **4.0% · 2.8% · 2.2%** | ✓ kept |
| + lender + state | 0.574 | 0.813 | 4.4% · 3.0% · 2.1% | ✗ left out |
| + lender + county | 0.576 | 0.851 | 4.7% · 3.2% · 2.1% | ✗ left out |
| + lender + county + neighbourhood (census tract) | 0.577 | 0.870 | 4.8% · 3.3% · 2.1% | ✗ left out |

* **The lender passed:** +0.027 PR-AUC, and it barely helps guess race
  (0.746 → 0.757). It isn't free (approved Black families wrongly flagged
  3.6% → 4.0%), so we publish that cost.
* **Place failed.** The state adds only +0.005 PR-AUC while the proxy AUC jumps to
  0.813 and wrong flags for approved Black families rise to
  4.4%; the county adds +0.007, with
  0.851 and 4.7%. The neighbourhood version adds +0.001 more; moving the same 2025 families from a census tract with 10% minority residents to one with 75% changes their predicted risk by +1.14 points on average.
* **A trap worth knowing:** place makes the group ratios look *better* (Black ×1.24 →
  ×1.21 with the state, ×1.19 with the
  county) because the model learns to expect more denials where minority applicants live. That copies the lenders' gap
  into the prediction instead of measuring it, and it wrongly flags more Black families. So your state sets taxes,
  insurance and the lender list on the website, but the model never sees where you live.

No model can satisfy every definition of fairness at once when groups have different denial rates. We chose **the
same answer for the same finances**: FairKey never uses race, not even to be "more accurate" for one group, because that
would copy the lenders' gap into the prediction. The price is visible on the site: for Black and Hispanic applicants,
real denials run above FairKey's number at every level of risk ("For the same predicted risk, who is denied more?").
We refuse inputs that act as stand-ins for race, and we **publish the gaps that remain**.

## How the AI works

```
HMDA 2022–2025, 50 states + DC: 15.0M decisions → 12.85M complete applications
   │  home purchase · first lien · owner-occupied · single-family · approved or denied
   │  complete records only (leak) · only what an applicant knows before applying
   ▼
LightGBM risk model ── monotone: more debt-to-income or loan-to-value can never lower the risk
   │  early stopping on PR-AUC (2024) · Platt calibration (2024) · tested once on 2025
   ├── LightGBM reason model (debt-to-income, credit, collateral, cash, paperwork, other)
   ├── logistic-regression and 43%-rule baselines, bootstrap confidence intervals
   └── fairness audit: observed/expected by group and by state, calibration by group, proxy test, 5 versions
   ▼
trees exported to a JavaScript engine in the browser (1,231 risk trees, 1,338 reason trees)
   ├── prediction (0.5 ms) and exact path-dependent TreeSHAP (32 ms), equal to LightGBM within 3e-6
   ├── "lender not chosen": the average over the lenders people in your state use, by market share,
   │     computed for all 50 lenders in one walk per tree (2 ms), with the exact TreeSHAP of that average (49 ms)
   ├── what-if search over realistic changes · safe-zone heat map (864 predictions per redraw)
   ├── per-lender comparison for the same application, in your state
   ├── "Inside the AI": the running total tree by tree, and the path through any tree (its real thresholds)
   └── reality check: nearest of 50,000 sampled 2025 applications (same loan type, home type and DTI band)
```

* **11 features:** income, loan amount, home price, loan-to-value, debt-to-income, loan-to-income, term, co-borrower,
  manufactured home, loan type, lender (the 50 biggest nationally + "other").
* **Your state is used, but not as a model input:** it sets the property-tax and insurance estimate in your monthly
  payment, the lender mix when you haven't picked one, the lender list, and the reality check. We tested letting the
  model itself see the state: it failed the fairness rule (see "Which inputs we let in" above), so it stays out.
* **The browser engine is checked, not trusted:** `tests/check_engine.mjs` compares it with LightGBM's own
  predictions and SHAP values (`pred_contrib=True`) on 500 test applications (max error 2.4e-6 for the prediction,
  1.7e-6 for SHAP), and checks the "all lenders at once" functions against one-lender-at-a-time runs and against
  brute-force Shapley values over all 1,024 coalitions (max error 8e-15).
* **The promises are checked in code, not just written down:** `tests/check_guards.py` reads the published model
  files and fails if any input could reveal race, ethnicity, sex or age, is only known after the decision (interest
  rate, pre-approval code, underwriting result, denial reasons) or is a place below the state (county, census tract);
  if the browser's trees differ from LightGBM's; if raising debt-to-income or loan-to-value ever lowers the risk (2,000
  real 2025 applications × every value); if the years overlap; if the bands drift between 2024 and 2025; or if the
  public sample holds an identifier. 10/10 pass.
* **The form speaks HMDA:** what you type is converted the way lenders report it (debt-to-income bands, prices in
  $10k buckets), and the monthly payment includes taxes, insurance, mortgage insurance and FHA/VA/USDA fees. Taxes and
  insurance default to your state's averages (Census ACS 2024 property-tax rate, NAIC 2023 homeowners premium).

## Limits (said plainly)

* **No credit scores** in public data. If your credit is strong, your real risk is lower; if it is weak, higher.
* **Lenders differ in who applies to them.** The lender comparison is a lead for shopping around, not a ranking of
  who is fair, and not proof that switching lender changes the outcome.
* **Observational data:** "what would change it" is the model's estimate, not a promise.
* **The reality check is a simple nearest-neighbour count**, not a second model. It can disagree with FairKey (for
  example when your exact mix of inputs is rare); we show both rather than hide the gap. If fewer than 40 real
  applications match your loan type, home type and debt-to-income band, the band is widened.
* **Calibration is not perfect everywhere.** Overall a FairKey percentage is within a fraction of a point of reality,
  but between 12% and 30% it runs about 2 points high, and between 40% and 50% about 7 points low (it said 44%,
  reality was 52%); few applications fall there.
* **Small states** have too few Black or Hispanic applicants in one year to measure a gap fairly; the state map greys
  them out instead of guessing.
* **Home purchases, 2022–2025 only.** Rules and markets change; retrain each year when HMDA publishes.
* It is a prototype for learning and exploring. **It is not financial advice.**

## Run it

```bash
# the website (no build step)
cd web && python3 -m http.server 8000      # then open http://localhost:8000

# rebuild everything from public data (~3 GB download, ~6 GB of memory for training)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python hmda_data.py              # download + filter HMDA → data/us/<year>.parquet
.venv/bin/python tools/make_state_costs.py # state property-tax and insurance rates → data/state_costs.csv
.venv/bin/python train.py                  # models, baselines, evidence → web/data/
.venv/bin/python audit.py                  # fairness audit → web/data/fairness.json, figures/
.venv/bin/python tools/make_sample.py      # 50,000 real 2025 applications for the hero and the reality check
.venv/bin/python tests/make_fixtures.py && node tests/check_engine.mjs   # browser engine == LightGBM
.venv/bin/python tests/check_guards.py      # no protected or leaky inputs, monotone, split by time
.venv/bin/python tools/make_notebook.py    # re-run the evidence notebook
```

| File | What it does |
|---|---|
| `hmda_data.py` | downloads the CFPB/FFIEC national snapshots and keeps home-purchase applications |
| `features.py` | turns HMDA rows into model inputs; race, sex and age are kept apart, for the audit only |
| `train.py` | risk and reason models, baselines, bootstrap, calibration, state costs, export for the browser |
| `audit.py` | fairness audit (by group and by state), proxy test, five model versions |
| `tools/make_state_costs.py` | property-tax rate (ACS 2024, tables B25103/B25077) and home-insurance rate (NAIC 2023) per state |
| `tools/make_sample.py` | a random sample of 2025 test applications (`web/data/sample.json`, no identifiers) |
| `tests/` | `check_engine.mjs` (browser engine == LightGBM), `check_guards.py` (the promises above, on the published files) |
| `web/` | the site: `fk-model.js` (tree engine + TreeSHAP + tree paths), `app.js`, `i18n.js` (English/Spanish) |
| `FairKey.ipynb` | the evidence notebook: every number above, recomputed |

## Sources and credits

* **Data:** Home Mortgage Disclosure Act (HMDA) snapshot loan-level datasets and Transmittal Sheets, 2022–2025,
  published by the CFPB/FFIEC (public U.S. government data). <https://ffiec.cfpb.gov/data-publication/>
* **State costs:** U.S. Census Bureau, American Community Survey 2024 1-year (median real estate taxes and home
  values); NAIC, *Dwelling Fire, Homeowners Owner-Occupied, and Homeowners Tenant and Condominium/Cooperative Unit
  Owner's Insurance Report* (2023 data, HO-3 average premium).
* **Context:** The Markup, "The Secret Bias Hidden in Mortgage-Approval Algorithms" (2021); the Equal Credit
  Opportunity Act; the CFPB complaint portal.
* **Libraries:** LightGBM, scikit-learn, pandas, NumPy, PyArrow, SciPy, Matplotlib, Jupyter (nbformat, nbclient);
  fonts Fraunces and Inter (Google Fonts). TreeSHAP follows Lundberg, Erion and Lee, *Consistent Individualized
  Feature Attribution for Tree Ensembles* (2018).
* **AI assistance:** code, analysis and writing were developed with Claude (Anthropic) in Claude Code. Every number
  comes from the scripts in this repository.

Built for the Veritas AI International High School Hackathon 2026 · Finance track.

> **This project is a prototype and is not financial advice.**
