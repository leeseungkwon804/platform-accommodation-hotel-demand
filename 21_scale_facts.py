"""c10dec6 decision 6: scale of treated vs control regions in the event-study sample (2018-19 mean nights).
Same region definitions as 16 (PT17 re-aggregated; controls = regions in countries without listed regulation).
usage: python -I 21_scale_facts.py DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, pandas as pd
D, O = map(pathlib.Path, sys.argv[1:3])
p = pd.read_csv(D / "L1_panel.csv.gz")[["geo", "year", "plat", "I551"]]
lx = p[p.geo.isin(["PT1A", "PT1B"])].groupby("year")[["plat", "I551"]].sum().reset_index().assign(geo="PT17")
p = pd.concat([p[~p.geo.isin(["PT1A", "PT1B"])], lx], ignore_index=True)
ev = pd.read_csv(D / "regulation_events.csv")
tr = set(ev[(ev.source_level.str.strip() == "1차") & (ev.nuts2.str.len() == 4)].nuts2)
ec = set(ev.nuts2.str[:2])
m = p[p.year.isin([2018, 2019])].groupby("geo")[["plat", "I551"]].mean()
T, C = m[m.index.isin(tr)], m[~m.index.str[:2].isin(ec)]
r = dict(years="2018-2019 mean", n_treated_with_data=len(T), n_controls_with_data=len(C))
for v in ["plat", "I551"]:
    r[v] = dict(treated_mean=float(T[v].mean()), treated_median=float(T[v].median()),
                control_mean=float(C[v].mean()), control_median=float(C[v].median()),
                ratio_mean=float(T[v].mean() / C[v].mean()), ratio_median=float(T[v].median() / C[v].median()))
json.dump(r, open(O / "scale_facts.json", "w"), indent=1); print(json.dumps(r, indent=1))
