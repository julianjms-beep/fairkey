"""Default taxes + insurance for each state, used ONLY to turn what you type into a monthly payment (and so a
debt-to-income ratio). You can always type your own number in "Payment assumptions".

    python tools/make_state_costs.py      # writes data/state_costs.csv

* Property tax: median real estate taxes paid ÷ median value of owner-occupied homes, US Census Bureau,
  American Community Survey 2024 1-year, tables B25103 and B25077 (the Tax Foundation's method).
* Home insurance: average HO-3 premium per state, NAIC "Dwelling Fire, Homeowners Owner-Occupied, and Homeowners
  Tenant and Condominium/Cooperative Unit Owner's Insurance Report: Data for 2023" (July 2026), Table 4, "Total" rows,
  HO-3 column. Turned into a rate by dividing by the state's median home value (ACS above).
"""
from __future__ import annotations

import io
from pathlib import Path

import urllib.request

import pandas as pd

ACS = "https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/1YRData/acsdt1y2024-{t}.dat"
FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
        "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
        "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
        "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
        "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
NAMES = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
         "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
         "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
         "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
         "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
         "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
         "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
         "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
         "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
         "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
# NAIC 2023 data, Table 4, HO-3 average written premium (US average $1,737). Checked against the report's pages.
HO3 = {"AL": 1906, "AK": 1216, "AZ": 1194, "AR": 1870, "CA": 1655, "CO": 2492, "CT": 2036, "DE": 1196, "DC": 1545,
       "FL": 2779, "GA": 1828, "HI": 1549, "ID": 1135, "IL": 1480, "IN": 1259, "IA": 1342, "KS": 1733, "KY": 1525,
       "LA": 3027, "ME": 1150, "MD": 1578, "MA": 2134, "MI": 1110, "MN": 1988, "MS": 2029, "MO": 1589, "MT": 1768,
       "NE": 2142, "NV": 1013, "NH": 1300, "NJ": 1551, "NM": 1490, "NY": 1801, "NC": 1852, "ND": 1414, "OH": 1116,
       "OK": 2486, "OR": 1003, "PA": 1217, "RI": 2396, "SC": 1753, "SD": 1614, "TN": 1649, "TX": 2864, "UT": 1107,
       "VT": 1215, "VA": 1537, "WA": 1232, "WV": 1179, "WI": 923, "WY": 1853}


def acs(table: str, col: str) -> pd.Series:
    with urllib.request.urlopen(ACS.format(t=table), timeout=120) as r:
        t = pd.read_csv(io.StringIO(r.read().decode()), sep="|", dtype=str)
    t = t[t.GEO_ID.str.startswith("0400000US")]
    return pd.Series(t[col].astype(float).to_numpy(), index=t.GEO_ID.str[-2:].map(FIPS)).dropna()


def main():
    tax, value = acs("b25103", "B25103_E001"), acs("b25077", "B25077_E001")
    rows = []
    for st in sorted(NAMES):
        v = value[st]
        rows.append({"state": st, "name": NAMES[st], "median_value": int(v), "median_tax": int(tax[st]),
                     "tax_rate": round(100 * tax[st] / v, 3), "ho3_premium": HO3[st],
                     "ins_rate": round(100 * HO3[st] / v, 3)})
    out = pd.DataFrame(rows)
    out["total_rate"] = (out.tax_rate + out.ins_rate).round(3)
    Path("data").mkdir(exist_ok=True)
    out.to_csv("data/state_costs.csv", index=False)
    print(out.sort_values("total_rate").to_string(index=False))


if __name__ == "__main__":
    main()
