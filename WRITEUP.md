# FairKey — will a lender say yes to your mortgage?

**Track:** Finance · **Team:** solo entry · **Website:** [julianjms-beep.github.io/fairkey](https://julianjms-beep.github.io/fairkey/) · **Code:** [GitHub](https://github.com/julianjms-beep/fairkey) · **Notebook:** [Colab](https://colab.research.google.com/github/julianjms-beep/fairkey/blob/main/FairKey.ipynb)

> **Main takeaway:** before you apply, FairKey tells you the chance a lender says no, why, and what would change it,
> with nearly 3× fewer false alarms than the 43% rule. And lenders deny Black applicants ×1.45 as often as White ones
> with the same finances.

## The problem
María, a nurse in Florida, is about to apply for her first home loan. All she has is a rule of thumb: "keep
debt-to-income under 43%". A "no" costs her months, fees and a mark on her credit file. In 2025, 1 in 11
home-purchase applications in our data was denied (264,695), and denials do not fall equally on every group.

## What we built
A free website in English and Spanish for all 50 states and DC. You enter your state, income, home price, down
payment, debts, loan type and, if you know it, the lender. It shows **the chance of a "no"** with a band (*likely yes /
uncertain / high risk*); **why** (each input's push in points, by exact TreeSHAP, and the lender's most likely reason);
**what would change it** (more down payment, fewer debts, a cheaper home, another loan type or lender); a **reality
check** against the 200 most similar real 2025 applications; the trees inside the AI; and a state fairness map. For
María ($62k income, $285k home, 3.5% down, FHA) it says 20%; 35 of her 200 look-alikes were denied.

## What we added
Our own idea. **Own data:** 12.85 million real U.S. mortgage applications, not a sample dataset. **Own user:**
first-time buyers, before they apply. **Own question:** does the model treat groups unfairly when race is not an
input, and which inputs secretly stand in for race?

## How it works
**Data:** public HMDA records (CFPB/FFIEC), 2022–2025: home purchases, complete records only. We refused three leaks:
interest rates (known only after approval), incomplete records (41% denials) and "pre-approved" files (all
740,803 approved). **Model:** LightGBM with monotone constraints, so more debt can never lower the risk; Platt
calibration; a second model for the reason. It beats logistic regression here and can be explained exactly. **Split by time:** trained on 2022–2023, tuned on 2024, tested once on 3,017,153
applications from 2025. The trees run in our own JavaScript engine (nothing leaves your device), matching
LightGBM within 0.000003.

## Results
| 2025 test (never seen) | PR-AUC (95% CI) | False alarms per 1,000, catching 63% of denials |
|---|---|---|
| Rule: debt-to-income above 43% | 0.335 (0.333–0.337) | 343 |
| Logistic regression | 0.503 (0.501–0.505) | 156 |
| **FairKey** | **0.569 (0.567–0.571)** | **121** |

"Approve everyone" is 91% accurate and useless, so we use PR-AUC. Denial rates by band: *likely yes* 3.7%,
*uncertain* 11.0%, *high risk* 65.5%.

**Fairness.** FairKey never sees race, sex or age; we used them only to audit it.
- **Lenders:** with the same finances, Black applicants are denied ×1.45 as often as White applicants (Hispanic
  ×1.29), clearly above 1 in all 39 states with enough data. Part of it could be credit history, which public data lacks.
- **Sex and age:** every group is denied within ×0.92–×1.08 of what its finances predict (except applicants who didn't
  give their sex, ×1.17), and FairKey ranks risk about as well in each group (AUC 0.80–0.84).
- **FairKey's inputs:** an input stays only if it adds accuracy without mainly encoding race. Of five versions, the
  lender passed (+0.027 PR-AUC). The state failed: +0.005 PR-AUC, but the model got better at guessing race (proxy AUC
  0.757 → 0.813) and wrongly flagged more approved Black families (4.0% → 4.4%). So it never sees where you live.

**Where it fails, and why we think so.**
- **A real miss:** a Florida couple earning $178k, buying a $675k home with a VA loan (no down payment, 33%
  debt-to-income). FairKey said 0.3%. The lender said no, citing credit history, which public data does not have.
- **The pattern:** 33% of 2025 denials (87,186) had been rated *likely yes*. In nearly 4 of 5, the reason was
  something the application numbers can't show: the appraisal (23%), paperwork (22%), credit history (19%) or
  another reason (16%). Debt-to-income explains 62% of the denials FairKey caught, but only 10% of those it missed.
- **Fairness:** real denials run above FairKey's number for Black and Hispanic applicants, and its wrong "high risk"
  flags hit 4.0% of approved Black families vs. 2.2% of White families. The site shows this.

## What we'd do next
Test it with a housing counsellor and first-time buyers, and let people add a self-reported credit range.

## Sources
[HMDA 2022–2025](https://ffiec.cfpb.gov/data-publication/) (CFPB/FFIEC) · [Census ACS 2024](https://data.census.gov/)
(B25103, B25077) · [NAIC](https://content.naic.org/) home insurance (2023) ·
[The Markup (2021)](https://themarkup.org/denied/2021/08/25/the-secret-bias-hidden-in-mortgage-approval-algorithms) ·
[TreeSHAP](https://arxiv.org/abs/1802.03888) · [LightGBM](https://github.com/lightgbm-org/LightGBM).
**AI tools:** [Claude Code](https://claude.com/product/claude-code) (Anthropic) was our AI coding partner: it wrote
most of the code and texts, while we chose the problem, directed the work and made the final decisions.
[ElevenLabs](https://elevenlabs.io/) voiced the video. The "own words" box uses word rules, not AI.

*This project is a prototype and is not financial advice.*
