"""Within (two-way FE) variation of penetration and log hotel nights. usage: python -I 03_within_variation.py DERIVED_DIR"""
import sys, pathlib, numpy as np, pandas as pd
D = pathlib.Path(sys.argv[1])
def twfe_resid(df, y, a, b, it=200):
    r = df[y] - df[y].mean()
    for _ in range(it):
        r = r - r.groupby(df[a]).transform("mean"); r = r - r.groupby(df[b]).transform("mean")
    return r
for name, t in [("panel_country_month", "ym"), ("panel_nuts2_year", "time")]:
    d = pd.read_csv(D / f"{name}.csv.gz", dtype={"time": str, "ym": str})
    d = d[(d.I551 > 0) & (d.plat > 0)].copy()
    d["pen"] = d.plat / (d.plat + d.I551); d["ly"] = np.log(d.I551); d["lp"] = np.log(d.plat)
    if t == "ym": d["mo"] = d.ym.str[-2:]
    out = {}
    for v in ["pen", "ly", "lp"]:
        r = twfe_resid(d, v, "geo", t)
        out[v] = dict(sd_raw=round(d[v].std(), 4), sd_within=round(r.std(), 4), share_within=round(r.var() / d[v].var(), 3))
    # range of pen within units
    rng = d.groupby("geo").pen.agg(lambda s: s.max() - s.min())
    print(name, "n", len(d), out, "| within-unit pen range median %.3f p90 %.3f" % (rng.median(), rng.quantile(.9)))
    if t == "ym":
        ov = d[d.I552_I553 > 0]; ov = ov.assign(r=ov.plat / ov.I552_I553)
        print("  overlap ratio plat/I552_I553 by country median:", ov.groupby("geo").r.median().round(2).sort_values().to_dict())

# country x month: add geo x calendar-month FE (country-specific seasonality) on top of geo + ym FE
d = pd.read_csv(D / "panel_country_month.csv.gz", dtype={"ym": str})
d = d[(d.I551 > 0) & (d.plat > 0)].copy()
d["pen"] = d.plat / (d.plat + d.I551); d["ly"] = np.log(d.I551); d["gm"] = d.geo + d.ym.str[-2:]
for tag, sub in [("all", d), ("excl COVID 2020-03..2021-12", d[(d.ym < "2020-03") | (d.ym > "2021-12")])]:
    for v in ["pen", "ly"]:
        r = twfe_resid(sub, v, "gm", "ym")
        print(f"country_month [{tag}] geo*cal-month + ym FE: {v} n={len(sub)} sd_within={r.std():.4f} share_within={r.var()/sub[v].var():.3f}")
