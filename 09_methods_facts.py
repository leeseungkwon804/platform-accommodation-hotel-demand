"""Collect every number quoted in Methods directly from data files. usage: python -I 09_methods_facts.py DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, numpy as np, pandas as pd
D, O = map(pathlib.Path, sys.argv[1:3])
f = {}
c = pd.read_csv(D / "panel_country_month.csv.gz", dtype={"ym": str})
f["L2"] = dict(countries=int(c.geo.nunique()), months=int(c.ym.nunique()), first=c.ym.min(), last=c.ym.max(), cells=len(c),
               cells_excl_covid=int(((c.ym < "2020-03") | (c.ym > "2021-12")).sum()))
ov = c[(c.I552_I553 > 0)]; r = ov.plat / ov.I552_I553
byc = r.groupby(ov.geo).median()
f["overlap"] = dict(n=len(r), median=float(r.median()), p10=float(r.quantile(.1)), p90=float(r.quantile(.9)),
                    countries_median_gt1=int((byc > 1).sum()), countries=int(len(byc)),
                    country_median_min=[byc.idxmin(), float(byc.min())], country_median_max=[byc.idxmax(), float(byc.max())])
n = pd.read_csv(D / "panel_nuts2_year.csv.gz", dtype={"time": str})
f["L1"] = dict(regions=int(n.geo.nunique()), years=sorted(n.time.unique().tolist()), cells=len(n),
               cells_excl_covid=int((~n.time.isin(["2020", "2021"])).sum()))
sd = pd.read_csv(D / "L1_resid_sd.csv")
m = sd[(sd["sample"] == "excl 2020-2021") & (sd["var"] == "pen_hotel")].iloc[0]
f["L1_variation"] = dict(sd_between=float(m.sd_between_region), sd_within=float(m.sd_resid), within_share=float(m.within_share),
                         between_share=float(1 - m.within_share))
pen = n.plat / (n.plat + n.I551)
f["L1_pen_quantiles"] = {f"p{int(q*100)}": float(pen.quantile(q)) for q in [.1, .5, .9]}
s = (D / "feasibility_summary.txt").read_text()
import re
mm = re.search(r"OPTION 4.*?grid cells: (\d+)\ncomplete cells.*?: (\d+) missing rate ([\d.]+)%", s, re.S)
f["L3"] = dict(grid=int(mm.group(1)), cells=int(mm.group(2)), missing_pct=float(mm.group(3)))
L = pd.read_csv(D / "L1_panel.csv.gz", dtype={"time": str})
COVS = ["beds_per_1000", "hotel_bed_share", "log_gdp_pps_pc", "log_density", "log_pop", "city_night_share", "coastal_night_share", "season_gini_2022_23"]
cc = L.dropna(subset=COVS)
f["L1_covariates"] = dict(regions_complete=int(cc.geo.nunique()), rows_complete=len(cc),
                          rows_complete_excl_covid=int((~cc.year.isin([2020, 2021])).sum()))
json.dump(f, open(O / "facts_methods.json", "w"), indent=1, default=float); print(json.dumps(f, indent=1, default=float))
