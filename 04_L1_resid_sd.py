"""Decision-4 gate: L1 (NUTS2 x year) residual SD after region FE + year FE. usage: python -I 04_L1_resid_sd.py DERIVED_DIR"""
import sys, pathlib, numpy as np, pandas as pd
D = pathlib.Path(sys.argv[1])
def twfe_resid(df, y, a="geo", b="time", it=500):
    r = df[y] - df[y].mean()
    for _ in range(it):
        r = r - r.groupby(df[a]).transform("mean"); r = r - r.groupby(df[b]).transform("mean")
    return r
d = pd.read_csv(D / "panel_nuts2_year.csv.gz", dtype={"time": str})
d = d[(d.I551 > 0) & (d.plat > 0)].copy()
d["pen_hotel"] = d.plat / (d.plat + d.I551)          # main denominator
d["pen_all"] = d.plat / (d.plat + d["I551-I553"])    # alt denominator (robustness 3)
d["ly"] = np.log(d.I551)
rows = []
for tag, sub in [("all 2018-2025", d), ("excl 2020-2021", d[~d.time.isin(["2020", "2021"])]),
                 ("balanced excl COVID", None)]:
    if sub is None:
        s = d[~d.time.isin(["2020", "2021"])]; n_t = s.time.nunique()
        sub = s[s.groupby("geo").time.transform("count") == n_t]
    for v in ["pen_hotel", "pen_all", "ly"]:
        s = sub.dropna(subset=[v]); r = twfe_resid(s, v)
        between = s.groupby("geo")[v].mean().std()
        rows.append(dict(sample=tag, var=v, n=len(s), regions=s.geo.nunique(), sd_raw=s[v].std(),
                         sd_between_region=between, sd_resid=r.std(), within_share=r.var() / s[v].var()))
out = pd.DataFrame(rows).round(4); print(out.to_string(index=False))
out.to_csv(D / "L1_resid_sd.csv", index=False)
