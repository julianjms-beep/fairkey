We shipped **finances + lender**. The lender passed the rule: «d_lender» PR-AUC, and it barely helps guess race
(proxy AUC «px_finances» → «px_shipped»). It is not free: approved Black families wrongly flagged went from
«ff_finances_Black» to «ff_shipped_Black», so we publish that cost. **Place failed.** The state would add only «d_state»
PR-AUC, yet the proxy AUC jumps to «px_shipped_state» and wrong flags for approved Black families rise to
«ff_shipped_state_Black»; the county adds «d_county», with a proxy AUC of «px_shipped_county» and «ff_shipped_county_Black».
Across the U.S., where you live stands in for race. Place even makes the group ratios at the start of section 7 look *better* (Black ×«rB_shipped» →
×«rB_shipped_state» with the state, ×«rB_shipped_county» with the county), because the model learns to expect more
denials where minority applicants live: it copies the lenders' gap into the prediction instead of measuring it, and
wrongly flags more Black families while doing so. So on the website your state sets taxes, insurance and the lender
list, but the model never sees where you live.
