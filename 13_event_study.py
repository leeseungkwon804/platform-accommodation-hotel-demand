"""Robustness 2: stacked event study around STR regulation entry into force (region-level events, primary-source dates).
Treat year = effective year if effective in Jan-Jun, else next year. Controls: regions in countries with no listed event.
Each treated region is stacked with all controls; region x stack and year x stack FE; SE clustered by region.
usage: python -I 13_event_study.py DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
D, O = map(pathlib.Path, sys.argv[1:3])
p = pd.read_csv(D / "L1_panel.csv.gz")[["geo", "year", "plat", "I551"]]
# NUTS 2024 split of Lisbon (PT1A + PT1B = PT17) re-aggregated for the event study only
lx = p[p.geo.isin(["PT1A", "PT1B"])].groupby("year")[["plat", "I551"]].sum().reset_index().assign(geo="PT17")
p = pd.concat([p[~p.geo.isin(["PT1A", "PT1B"])], lx], ignore_index=True)
p = p[~p.year.isin([2020, 2021])]
p["log_I551"], p["log_plat"] = np.log(p.I551), np.log(p.plat)
ev = pd.read_csv(D / "regulation_events.csv")
ev = ev[(ev.source_level.str.strip() == "1차") & (ev.nuts2.str.len() == 4)].copy()
ev["eff"] = pd.to_datetime(ev.date_effective)
ev["treat_year"] = ev.eff.dt.year + (ev.eff.dt.month > 6).astype(int)
first = ev.sort_values("eff").groupby("nuts2").first().reset_index()[["nuts2", "city", "date_effective", "treat_year"]]
event_countries = set(pd.read_csv(D / "regulation_events.csv").nuts2.str[:2])
controls = sorted(g for g in p.geo.unique() if g[:2] not in event_countries)
stacks = []
for _, e in first.iterrows():
    s = p[p.geo.isin(controls + [e.nuts2])].copy()
    s["stack"] = e.nuts2; s["treated"] = (s.geo == e.nuts2).astype(int); s["rel"] = s.year - e.treat_year
    stacks.append(s)
S = pd.concat(stacks, ignore_index=True)
S["sg"], S["sy"] = S["stack"] + "_" + S.geo, S["stack"] + "_" + S.year.astype(str)
S["post"] = S.treated * (S.rel >= 0)
S["pre_far"] = S.treated * (S.rel <= -2)
S["post_0_1"] = S.treated * S.rel.between(0, 1)
S["post_2p"] = S.treated * (S.rel >= 2)
res = dict(events=first.to_dict("records"), n_treated=int(len(first)), n_controls=len(controls),
           control_countries=sorted(set(g[:2] for g in controls)), rows=len(S))
for y in ["log_I551", "log_plat"]:
    d = S.dropna(subset=[y])
    cl = pd.factorize(d.geo)[0]
    m1 = smf.ols(f"{y} ~ post + C(sg) + C(sy)", d).fit(cov_type="cluster", cov_kwds={"groups": cl})
    m2 = smf.ols(f"{y} ~ pre_far + post_0_1 + post_2p + C(sg) + C(sy)", d).fit(cov_type="cluster", cov_kwds={"groups": cl})
    res[y] = dict(did=dict(b=float(m1.params["post"]), se=float(m1.bse["post"]), p=float(m1.pvalues["post"])),
                  event={k: dict(b=float(m2.params[k]), se=float(m2.bse[k]), p=float(m2.pvalues[k])) for k in ["pre_far", "post_0_1", "post_2p"]})
json.dump(res, open(O / "L1_event_study.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
