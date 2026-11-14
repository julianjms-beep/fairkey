"""HMDA (Home Mortgage Disclosure Act) data: every mortgage application in the US, published every year by the CFPB.

We keep one decision, everywhere in the country: home-purchase, first-lien mortgages for a home the family will live
in, in the 50 states and DC, where the lender either approved or denied the application.

    python hmda_data.py            # download 2022-2025 (about 3 GB of zips) and write data/us/<year>.parquet

Source: Consumer Financial Protection Bureau / FFIEC, HMDA Snapshot National Loan-Level Dataset (public domain).
https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

YEARS = (2022, 2023, 2024, 2025)
URL = "https://files.ffiec.cfpb.gov/static-data/snapshot/{y}/{y}_public_lar_csv.zip"
RAW = Path("data/raw")
OUT = Path("data/us")
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
          "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
          "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

KEEP = [
    "activity_year", "lei", "county_code", "census_tract", "action_taken", "preapproval", "loan_type",
    "loan_amount", "property_value", "combined_loan_to_value_ratio", "loan_term", "income", "debt_to_income_ratio",
    "construction_method", "total_units", "conforming_loan_limit", "submission_of_application",
    "co_applicant_sex", "co_applicant_age",
    # only for the fairness check — never used as model inputs
    "derived_race", "derived_ethnicity", "derived_sex", "applicant_age", "applicant_age_above_62",
    # only for explaining denials afterwards — never used as model inputs
    "denial_reason_1", "denial_reason_2", "denial_reason_3",
    # neighbourhood (census tract) description
    "tract_minority_population_percent", "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
    "tract_median_age_of_housing_units",
    # kept only to check for leaks: these exist only after a loan is made
    "interest_rate", "purchaser_type", "aus_1", "applicant_credit_score_type",
]
FILTER = ["state_code", "loan_purpose", "lien_status", "occupancy_type", "business_or_commercial_purpose",
          "reverse_mortgage", "open_end_line_of_credit"]


def download(years=YEARS) -> None:
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for y in years:
        f = RAW / f"{y}_public_lar_csv.zip"
        if f.exists() and zipfile.is_zipfile(f):
            continue
        with requests.get(URL.format(y=y), stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(f, "wb") as out:
                for chunk in r.iter_content(1 << 22):
                    out.write(chunk)


def read_year(y: int) -> int:
    """Stream one national file and write its home-purchase decisions (approved or denied) to data/us/<year>.parquet.
    Written chunk by chunk, so memory stays small even though the national file has ~16 million rows."""
    z = zipfile.ZipFile(RAW / f"{y}_public_lar_csv.zip")
    name = [n for n in z.namelist() if n.endswith(".csv") or n.endswith(".txt")][0]
    cols = [c for c in KEEP if c != "activity_year"] + ["state_code"]
    schema = pa.schema([(c, pa.string()) for c in cols] + [("activity_year", pa.int16())])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{y}.parquet.part"
    n = 0
    with z.open(name) as fh, pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for c in pd.read_csv(fh, usecols=lambda col: col in KEEP + FILTER, dtype=str, chunksize=500_000,
                             keep_default_na=False):
            c = c[c["state_code"].isin(STATES) & (c["loan_purpose"] == "1") & (c["lien_status"] == "1")
                  & (c["occupancy_type"] == "1") & (c["business_or_commercial_purpose"] != "1")
                  & (c["reverse_mortgage"] != "1") & (c["open_end_line_of_credit"] != "1")
                  & c["action_taken"].isin(["1", "2", "3"])]
            t = pa.table({**{k: pa.array(c[k].to_numpy(dtype=object), pa.string()) for k in cols},
                          "activity_year": pa.array(np.full(len(c), y, dtype=np.int16))}, schema=schema)
            w.write_table(t)
            n += len(c)
    tmp.rename(OUT / f"{y}.parquet")
    return n


def build() -> None:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(YEARS)) as ex:          # one national file per core
        counts = dict(zip(YEARS, ex.map(read_year, YEARS)))
    print(f"{sum(counts.values()):,} applications · " + " · ".join(f"{y}: {n:,}" for y, n in counts.items()))


def files() -> list[Path]:
    return [OUT / f"{y}.parquet" for y in YEARS]


def count() -> int:
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files())


def load(columns: list[str] | None = None) -> pd.DataFrame:
    """All years, with text columns as categories (a few bytes per row instead of a Python string)."""
    cols = columns or [f.name for f in pq.read_schema(files()[0])]
    strs = [c for c in cols if c != "activity_year"]
    t = pa.concat_tables([pq.read_table(f, columns=cols, read_dictionary=strs) for f in files()], promote_options="permissive")
    return t.to_pandas()


if __name__ == "__main__":
    if "--no-download" not in sys.argv:
        download()
    build()
