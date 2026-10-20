"""C1 (L3): does current platform penetration 'predict' PAST seasonal concentration? Gini_{t-k} ~ P_t + region FE + year FE,
t in 2022-2025, k = 1, 2. SE clustered by region. usage: python -I 15_L3_placebo.py LONG_DIR DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
L, D, O = map(pathlib.Path, sys.argv[1:4])
m = pd.read_csv(L / "tour_occ_nin2m.csv.gz", dtype={"time": str, "flag": str}, low_memory=False)
m = m[(m.unit == "NR") & (m.c_resid == "TOTAL") & (m.month != "TOTAL") & (m.geo.str.len() == 4)].dropna(subset=["value"])
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())
g = m.groupby(["geo", "time"]).filter(lambda x: len(x) == 12).groupby(["geo", "time"]).value.apply(gini).rename("gini").reset_index()
g["year"] = g.time.astype(int); G = g.set_index(["geo", "year"]).gini
P = pd.read_csv(D / "L1_panel.csv.gz")[["geo", "year", "pen", "pen_fixed"]]
P = P[P.year.between(2022, 2025)].copy()
P["gini"] = [G.get((a, y), np.nan) for a, y in zip(P.geo, P.year)]
for k in (1, 2): P[f"gini_lag{k}"] = [G.get((a, y - k), np.nan) for a, y in zip(P.geo, P.year)]
res = {}
for out in ["gini", "gini_lag1", "gini_lag2"]:
    for tr in ["pen_fixed", "pen"]:
        s = P.dropna(subset=[out, tr]); s = s[s.groupby("geo").year.transform("count") >= 2]
        r = smf.ols(f"{out} ~ {tr} + C(geo) + C(year)", s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.geo)[0]})
        res[f"{out} ~ {tr}"] = dict(n=len(s), regions=int(s.geo.nunique()), beta_per10pp=float(r.params[tr] * .1),
                                    ci=[float(x * .1) for x in r.conf_int().loc[tr]], p=float(r.pvalues[tr]))
json.dump(res, open(O / "C1_L3_placebo.json", "w"), indent=1)
for k, v in res.items(): print(f"{k}: n={v['n']} regions={v['regions']} b/10pp={v['beta_per10pp']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3f}")
