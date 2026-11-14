# FairKey — will a lender say yes to your mortgage?

**Track:** Finance · **Website:** <https://julianjms-beep.github.io/fairkey/> · **Code and notebook:** <https://github.com/julianjms-beep/fairkey> ([`FairKey.ipynb`](https://colab.research.google.com/github/julianjms-beep/fairkey/blob/main/FairKey.ipynb))

**The problem.** A mortgage denial costs a family months, application fees and a mark on their credit file, and it
lands hardest on first-time buyers who have no one to ask. Before applying, people get rules of thumb ("keep
debt-to-income under 43%") that ignore everything else. Our question: **before you apply, how likely is a "no", why,
and what would change it — and is the answer fair to everyone?**

**What we built.** A free, bilingual (English/Spanish) website for home buyers in all 50 U.S. states and DC. You enter
your state, income, home price, down payment, other debts, loan type, term and, if you know it, the lender. It gives:
- the **chance a lender says no**, calibrated, with a band (*likely yes / uncertain / high risk*) and how often that
  band was really denied
- **why**: each input's push up or down in percentage points (exact TreeSHAP, computed on your device)
- **the reason you would most likely hear** (debt-to-income, credit, appraisal, cash, paperwork, other)
- **what would change it**: more down payment, fewer debts, a cheaper home, another loan type or lender, and the best
  combination
- a **safe-zone map** of the risk for every price and down payment, and the risk for the same application at
  the biggest lenders in your state
- a **reality check**: how many of the 200 most similar real 2025 applications were denied, next to FairKey's number
- **"Inside the AI"**: the 1,231 trees adding up to your answer, and any tree's real questions and vote
- a **state-by-state fairness map**, what the law says (Equal Credit Opportunity Act) and where to file a complaint.

Six example applicants in six states, a demo mode and a "tell it in your own words" box (simple word rules, not AI)
make it quick to try.

**How it works.**
- **Data:** 12.85 million complete home-purchase applications from every state, 2022–2025, from public HMDA records
  (CFPB/FFIEC). Taxes and insurance in the monthly payment use each state's averages (Census ACS 2024, NAIC 2023).
- **Only what you know before applying:** we found and refused three leaks. Interest rates exist only for approved
  loans; incomplete records (15%) are 41% denials; and all 740,803 "pre-approved" applications were approved because
  denied ones are filed elsewhere. Pre-approval first appeared in our own what-if tool as "→ 0% risk", which is how we
  caught it.
- **Models:** LightGBM with monotone constraints (all else equal, a higher debt-to-income or loan-to-value can never
  lower the risk), Platt calibration, and a second LightGBM model for the denial reason.
- **Honest split by time:** train 2022–2023 (6.8 million), choose and calibrate on 2024, test once on 3,017,153
  applications from 2025.
- **No server:** the trees run in a JavaScript engine we wrote, with exact TreeSHAP. On 500 test applications it
  matches LightGBM within 0.000003. When you haven't picked a lender, it averages over your state's lenders by market
  share; we made that exact and fast by walking each tree once for all 50 lenders (2 ms instead of 23 ms), including
  the SHAP values of the average, checked against brute-force Shapley values.

**Results (2025, never seen).**

| | PR-AUC (95% CI) | Wrong warnings per 1,000, catching 63% of denials |
|---|---|---|
| Debt-to-income above 43% (rule of thumb) | 0.335 (0.333–0.337) | 343 |
| Logistic regression, same inputs | 0.503 (0.501–0.505) | 156 |
| **FairKey** | **0.569 (0.567–0.571)** | **121** |

- **Why not accuracy:** "approve everyone" is 91% accurate and useless, so we measure how well it finds denials
  without false alarms (PR-AUC) and whether its percentages are true. Applications it called *likely yes* were denied
  3.7% of the time, *uncertain* 11.0%, *high risk* 65.5% (2024: 3.7%, 11.1%, 66.5%).
- The reason model names the lender's main reason 57.5% of the time (six options), vs. 37.4% by always guessing the
  most common one.

**Fairness.** FairKey never sees race, ethnicity, sex or age; we kept them only to audit.
- **Lenders:** Black applicants were denied 17.3% of the time when their finances predicted 13.9% (×1.24); White
  applicants 6.4% when their finances predicted 7.4% (×0.86). For the same finances, that is a **×1.45 gap**
  (Hispanic: ×1.29). It is not one state's quirk: the gap is clearly above 1 in **all 39 states** with enough data.
  Public data has no credit scores, so part of it could be credit history. We say so.
- **FairKey itself:** one rule for every input: keep it only if it adds accuracy without mainly encoding race. We
  trained five versions and ran a proxy test (can the inputs alone guess an applicant's race?). The lender passed:
  +0.027 PR-AUC, proxy AUC 0.746 → 0.757. Place failed: the state would add only +0.005 PR-AUC but push the proxy AUC
  to 0.813 and wrong flags for approved Black families from 4.0% to 4.4%; the county, 0.851 and 4.7%. So your state
  sets taxes, insurance and the lender list, but the model never sees where you live.
- Sex and age: every group within ×0.92–×1.08, except applicants who didn't give their sex (×1.17).

**Where it fails.**
- No credit scores: for someone with very strong or very weak credit, the estimate can be far off.
- Lenders serve different customers, so "apply with lender X" is a lead, not a promise; the data cannot show cause.
- The same answer for the same finances means real denials run above FairKey's number for Black and Hispanic
  applicants, and wrong "high risk" flags fall unequally: 4.0% of approved Black families vs. 2.2% of White families.
- Between 40% and 50% it runs low in 2025 (it said 44.5%, reality was 51.5%).
- Home purchases, 2022–2025 only; small states have too few applicants to measure a gap.

**What we'd do next.** Test it with a housing counsellor and first-time buyers; add refinancing; retrain every year
when HMDA publishes; let people add a self-reported credit range.

**Sources.**
- HMDA snapshot loan-level datasets and Transmittal Sheets 2022–2025, CFPB/FFIEC.
- U.S. Census Bureau, ACS 2024 1-year (B25103, B25077); NAIC Homeowners Insurance Report (2023 data).
- The Markup, "The Secret Bias Hidden in Mortgage-Approval Algorithms" (2021); Equal Credit Opportunity Act.
- Lundberg, Erion & Lee (2018), TreeSHAP.
- Software: Python, LightGBM, scikit-learn, pandas, NumPy, PyArrow, SciPy, Matplotlib, Jupyter; fonts Fraunces and Inter.
- AI tools: Claude Code (Anthropic, Claude Opus 5.5) helped write and debug the code and texts.

*This project is a prototype and is not financial advice.*
