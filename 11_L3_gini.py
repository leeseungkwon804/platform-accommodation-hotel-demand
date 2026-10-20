"""L3: regional seasonal concentration (Gini of 12 monthly nights, all establishments I551-I553) vs platform penetration.
Region + year FE, SE clustered by region. usage: python -I 11_L3_gini.py LONG_DIR DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
L, D, O = map(pathlib.Path, sys.argv[1:4])
m = pd.read_csv(L / "tour_occ_nin2m.csv.gz", dtype={"time": str, "flag": str}, low_memory=False)
m = m[(m.unit == "NR") & (m.c_resid == "TOTAL") & (m.month != "TOTAL") & (m.geo.str.len() == 4)].dropna(subset=["value"])
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())
g = m.groupby(["geo", "time"]).filter(lambda x: len(x) == 12).groupby(["geo", "time"]).value.apply(gini).rename("gini").reset_index()
g["year"] = g.time.astype(int)
P = pd.read_csv(D / "L1_panel.csv.gz")[["geo", "year", "pen", "pen_fixed", "plat_per_cap"]]
d = g.merge(P, on=["geo", "year"], how="inner")
res = dict(region_years_gini=len(g), regions_gini=int(g.geo.nunique()), gini_quantiles={f"p{q}": float(g.gini.quantile(q / 100)) for q in (10, 50, 90)})
for tag, sub in [("excl 2020-21", d[~d.year.isin([2020, 2021])]), ("all 2020-25", d)]:
    for tr in ["pen_fixed", "pen"]:
        s = sub.dropna(subset=[tr]); s = s[s.groupby("geo").year.transform("count") >= 2]
        r = smf.ols(f"gini ~ {tr} + C(geo) + C(year)", s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.geo)[0]})
        res[f"{tag} | {tr}"] = dict(n=len(s), regions=int(s.geo.nunique()), beta_per10pp=float(r.params[tr] * 0.1),
                                   ci=[float(x * 0.1) for x in r.conf_int().loc[tr]], p=float(r.pvalues[tr]), sd_gini=float(s.gini.std()))
json.dump(res, open(O / "L3_gini.json", "w"), indent=1); print(json.dumps(res, indent=1))
