"""Turn raw HMDA rows into the numbers FairKey learns from.

Rule: the model only gets things an applicant knows before applying (income, price, down payment, debts, loan type,
lender, and where they live was tested). Race, ethnicity, sex and age are kept in separate columns and used ONLY to
audit fairness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# HMDA reports debt-to-income in bands below 36% and above 49%; inside 36-49 it is the exact percent.
DTI_BANDS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}
LOAN_TYPES = {1: "Conventional", 2: "FHA", 3: "VA", 4: "USDA"}

# Why a lender said no (HMDA denial_reason codes) — grouped into the reasons an applicant can act on.
REASONS = {1: "dti", 2: "other", 3: "credit", 4: "collateral", 5: "cash", 6: "paperwork", 7: "paperwork",
           8: "other", 9: "other"}
REASON_ORDER = ["dti", "credit", "collateral", "cash", "paperwork", "other"]

AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
RACE_ORDER = ["White", "Black", "Hispanic", "Asian", "Other or two+", "Not provided"]
SEX_ORDER = ["Male", "Female", "Joint", "Not provided"]

NUMERIC = ["income_k", "loan_k", "price_k", "ltv", "dti", "loan_to_income", "term_years"]
CATEGORICAL = ["loan_type", "state", "county", "lender"]
# "preapproval" is NOT here: denied pre-approval requests are filed under another action code, so in this data every
# pre-approved application was approved (2022–2025, 50 states + DC, complete records: 0 of 740,803 denied). A model that sees it learns
# a leak, not a pattern.
FLAGS = ["coapp", "manufactured"]
LOCAL = ["tract_income_pct", "tract_minority_pct", "tract_housing_age"]   # neighbourhood: tested, then left out
# State, county and census-tract data were tested in audit.py against one rule: keep an input only if it adds accuracy
# without mainly encoding race (results in the README and FairKey.ipynb, section 7b).
FEATURES = NUMERIC + FLAGS + ["loan_type", "lender"]                     # what the published model uses


def _num(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype):            # national data comes as categories: convert each label once
        lab = pd.to_numeric(pd.Series(s.cat.categories.astype(str)), errors="coerce").to_numpy(dtype=float)
        codes = s.cat.codes.to_numpy()
        return pd.Series(np.where(codes >= 0, lab[np.maximum(codes, 0)], np.nan), index=s.index)
    return pd.to_numeric(s, errors="coerce")


def _str(s: pd.Series) -> pd.Series:
    return s.astype(str) if isinstance(s.dtype, pd.CategoricalDtype) else s


def dti_number(s: pd.Series) -> pd.Series:
    mapped = _str(s).map(DTI_BANDS).astype(float)
    return mapped.fillna(_num(s))


def race_group(d: pd.DataFrame) -> pd.Series:
    """One label per application, from HMDA's derived race and ethnicity (applicant + co-applicant)."""
    r, e = d["derived_race"], d["derived_ethnicity"]
    g = pd.Series("Other or two+", index=d.index)
    g[r.eq("Race Not Available") & ~e.eq("Hispanic or Latino")] = "Not provided"
    g[r.eq("White") & e.eq("Not Hispanic or Latino")] = "White"
    g[r.eq("Black or African American") & ~e.eq("Hispanic or Latino")] = "Black"
    g[r.eq("Asian") & ~e.eq("Hispanic or Latino")] = "Asian"
    g[e.eq("Hispanic or Latino")] = "Hispanic"
    return g


def sex_group(d: pd.DataFrame) -> pd.Series:
    return _str(d["derived_sex"]).map({"Male": "Male", "Female": "Female", "Joint": "Joint"}).fillna("Not provided")


def prepare(raw: pd.DataFrame, top_lenders: list[str] | None = None) -> pd.DataFrame:
    """Clean rows → model table. Keeps only complete applications (see README: why)."""
    d = pd.DataFrame(index=raw.index)
    d["year"] = raw["activity_year"].astype(int)
    d["denied"] = (raw["action_taken"] == "3").astype(int)
    d["income_k"] = _num(raw["income"])
    d["loan_k"] = _num(raw["loan_amount"]) / 1000
    d["price_k"] = _num(raw["property_value"]) / 1000
    d["ltv"] = _num(raw["combined_loan_to_value_ratio"])
    d["dti"] = dti_number(raw["debt_to_income_ratio"])
    d["term_years"] = _num(raw["loan_term"]) / 12
    d["loan_type"] = _num(raw["loan_type"]).astype("Int64")
    d["coapp"] = (raw["co_applicant_age"] != "9999").astype(int)
    d["manufactured"] = (raw["construction_method"] == "2").astype(int)
    d["preapproval"] = (raw["preapproval"] == "1").astype(int)
    d["units"] = _str(raw["total_units"])
    d["state"] = _str(raw["state_code"]) if "state_code" in raw else "FL"
    d["county"] = _str(raw["county_code"])
    d["lei"] = _str(raw["lei"])
    d["tract_income_pct"] = _num(raw["tract_to_msa_income_percentage"])
    d["tract_minority_pct"] = _num(raw["tract_minority_population_percent"])
    d["tract_housing_age"] = _num(raw["tract_median_age_of_housing_units"])
    # audit-only columns
    d["race"] = race_group(raw)
    d["sex"] = sex_group(raw)
    age = _str(raw["applicant_age"])
    d["age"] = age.where(age.isin(AGE_ORDER), "Not provided")
    d["reason"] = _num(raw["denial_reason_1"]).map(REASONS)
    # leak-check-only columns
    d["interest_rate"] = _num(raw["interest_rate"])

    ok = (d["income_k"] > 0) & d["loan_k"].gt(0) & d["price_k"].gt(0) & d["ltv"].between(1, 150) \
        & d["dti"].notna() & d["loan_type"].notna() & d["term_years"].between(5, 40) & d["county"].str.len().eq(5) \
        & d["units"].eq("1")
    d = d[ok].copy()
    d["loan_type"] = d["loan_type"].astype(int)
    d["loan_to_income"] = d["loan_k"] / d["income_k"]
    if top_lenders is not None:
        d["lender"] = d["lei"].where(d["lei"].isin(top_lenders), "other")
    return d


def top_lenders(d: pd.DataFrame, n: int = 40) -> list[str]:
    return d["lei"].value_counts().head(n).index.tolist()


def as_model_frame(d: pd.DataFrame, feats: list[str], cats: dict[str, list]) -> pd.DataFrame:
    """Same column order and category codes every time (training, testing and the browser)."""
    X = d[feats].copy()
    for c in feats:
        if c in cats:
            X[c] = pd.Categorical(X[c], categories=cats[c])
    return X
