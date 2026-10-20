"""Share of residual variance of log hotel nights that falls in the COVID-19 window (2020-03..2021-12).
L2: residuals after country x calendar-month effects. L1: residuals after region effects (no year effects).
usage: python -I 17_covid_share.py DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, numpy as np, pandas as pd
D, O = map(pathlib.Path, sys.argv[1:3])
c = pd.read_csv(D / "panel_country_month.csv.gz")
c = c[c.I551 > 0].copy(); c["t"] = pd.to_datetime(c.ym.astype(str)); c["y"] = np.log(c.I551)
c["r"] = c.y - c.groupby([c.geo, c.t.dt.month]).y.transform("mean")
cov = c.t.between("2020-03-01", "2021-12-01")
p = pd.read_csv(D / "L1_panel.csv.gz"); p = p[p.I551 > 0].copy()
p["r"] = p.log_I551 - p.groupby("geo").log_I551.transform("mean")
res = dict(L2=dict(cells=int(len(c)), covid_cells=int(cov.sum()), covid_share_ssr=float((c.r[cov] ** 2).sum() / (c.r ** 2).sum())),
           L1=dict(cells=int(len(p)), covid_cells=int(p.year.isin([2020, 2021]).sum()),
                   covid_share_ssr=float((p.r[p.year.isin([2020, 2021])] ** 2).sum() / (p.r ** 2).sum())))
json.dump(res, open(O / "covid_share.json", "w"), indent=1); print(res)
