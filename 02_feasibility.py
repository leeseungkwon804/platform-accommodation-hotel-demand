"""§B feasibility counts. usage: python -I 02_feasibility.py LONG_DIR OUT_DIR"""
import sys, pathlib, numpy as np, pandas as pd
L, O = map(pathlib.Path, sys.argv[1:3]); O.mkdir(exist_ok=True, parents=True)
rd = lambda c: pd.read_csv(L / f"{c}.csv.gz", dtype={"time": str, "flag": str}, keep_default_na=False, na_values=[""], low_memory=False)
Q = [.10, .25, .50, .75, .90, .95, .99]
lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

ce = rd("tour_ce_omn12"); ce = ce[(ce.indic_to == "NGT_SP") & (ce.c_resid == "TOTAL") & (ce.unit == "NR")]
ce_m = ce[ce.month != "TOTAL"].copy(); ce_m["ym"] = ce_m.time + "-" + ce_m.month.str[1:]
ce_m = ce_m[["geo", "ym", "value"]].rename(columns={"value": "plat"})
occm = rd("tour_occ_nim"); occm = occm[(occm.c_resid == "TOTAL") & (occm.unit == "NR")]
nin2m = rd("tour_occ_nin2m"); nin2m = nin2m[(nin2m.c_resid == "TOTAL") & (nin2m.unit == "NR")]
nin2 = rd("tour_occ_nin2"); nin2 = nin2[(nin2.c_resid == "TOTAL") & (nin2.unit == "NR")]

p("## B1 NACE split by table/geo level (non-missing geos)")
for name, d in [("tour_occ_nin2m", nin2m), ("tour_occ_nin2", nin2), ("tour_occ_nim", occm)]:
    d2 = d[d.value.notna()]
    p(name, d2.groupby([d2.geo.str.len(), "nace_r2"]).geo.nunique().unstack(fill_value=0).to_dict("index"))
p("tour_occ_nin2m months range:", nin2m[nin2m.value.notna()].time.min(), nin2m[nin2m.value.notna()].time.max())
p("tour_ce_omn12 platform NGT_SP monthly ym range:", ce_m.dropna().ym.min(), ce_m.dropna().ym.max())

def summarize(tag, df, unit_cols, cols, start=None):
    p(f"\n## {tag}")
    full = df.copy()
    n_units = full[unit_cols].nunique() if isinstance(unit_cols, str) else None
    p("units:", n_units, "periods:", full.t.nunique(), f"({full.t.min()}..{full.t.max()})",
      "grid cells:", n_units * full.t.nunique())
    ok = full.dropna(subset=cols)
    p("complete cells (all of", cols, "non-missing):", len(ok), "missing rate %.1f%%" % (100 * (1 - len(ok) / (n_units * full.t.nunique()))))
    return ok

def pen(tag, ok, num, den_cols):
    den = ok[den_cols].sum(axis=1)
    r = ok[num] / den
    qs = r.quantile(Q)
    p(f"penetration {tag} = {num}/({'+'.join(den_cols)}) quantiles:", {f"p{int(q*100)}": round(v, 4) for q, v in qs.items()},
      "| share <5%%: %.1f%%" % (100 * (r < .05).mean()), "| share >20%%: %.1f%%" % (100 * (r > .20).mean()))
    return r

# --- Option 1: country x month (I551 monthly available) ---
o = occm[occm.geo.str.len() == 2].copy(); o["ym"] = o.time
w = o.pivot_table(index=["geo", "ym"], columns="nace_r2", values="value").reset_index()
c1 = ce_m[ce_m.geo.str.len() == 2].merge(w, on=["geo", "ym"], how="outer")
END1 = min(ce_m.dropna().ym.max(), w.dropna(subset=["I551"]).ym.max())
p("I551 monthly last ym:", w.dropna(subset=["I551"]).ym.max(), "-> common end", END1)
c1 = c1[(c1.ym >= "2018-01") & (c1.ym <= END1)]
c1["t"] = c1.ym
geos = set(ce_m[ce_m.geo.str.len()==2].dropna().geo) & set(w.dropna(subset=["I551"]).geo)
c1 = c1[c1.geo.isin(geos)]
ok1 = summarize("OPTION 1: country x month (tour_ce_omn12 x tour_occ_nim)", c1, "geo", ["plat", "I551"])
p("countries:", sorted(geos))
r1 = pen("P_hotel", ok1, "plat", ["plat", "I551"])
pen("P_all", ok1.dropna(subset=["I551-I553"]), "plat", ["plat", "I551-I553"])
cov = ok1[(ok1.ym < "2020-03") | (ok1.ym > "2021-12")]
p("excluding COVID 2020-03..2021-12: cells", len(cov))
ov = ok1.dropna(subset=["I552_I553"]); ov = ov[ov.I552_I553 > 0]
rr = ov.plat / ov.I552_I553
p("OVERLAP platform/I552_I553 nights (country-month) quantiles:", {f"p{int(q*100)}": round(v, 3) for q, v in rr.quantile(Q).items()}, "n", len(rr))
ok1.to_csv(O / "panel_country_month.csv.gz", index=False)

# --- Option 2: NUTS2 x year (I551 annual) ---
ce_a = ce[ce.month == "TOTAL"][["geo", "time", "value"]].rename(columns={"value": "plat"})
n2 = nin2[nin2.geo.str.len() == 4].pivot_table(index=["geo", "time"], columns="nace_r2", values="value").reset_index()
c2 = ce_a[ce_a.geo.str.len() == 4].merge(n2, on=["geo", "time"], how="inner")
c2 = c2[(c2.time >= "2018") & (c2.time <= "2025")]; c2["t"] = c2.time
g2 = set(c2.dropna(subset=["plat", "I551"]).geo); c2 = c2[c2.geo.isin(g2)]
ok2 = summarize("OPTION 2: NUTS2 x year (tour_ce_omn12 TOTAL x tour_occ_nin2)", c2, "geo", ["plat", "I551"])
pen("P_hotel", ok2, "plat", ["plat", "I551"])
p("excluding 2020-2021 years: cells", (~ok2.time.isin(["2020","2021"])).sum())
ok2.to_csv(O / "panel_nuts2_year.csv.gz", index=False)

# --- Option 3: NUTS3 x year (tour_ce_oan3 x tour_occ_nin2) ---
oa = rd("tour_ce_oan3"); oa = oa[(oa.c_resid == "TOTAL")][["geo", "time", "value"]].rename(columns={"value": "plat"})
n3 = nin2[nin2.geo.str.len() == 5].pivot_table(index=["geo", "time"], columns="nace_r2", values="value").reset_index()
p("\n## OPTION 3: NUTS3 x year -> tour_occ_nin2 NUTS3 columns:", [c for c in n3.columns if c not in ("geo","time")], "(I551 absent at NUTS3 -> infeasible for hotel)")

# --- Option 4: NUTS2 x month, total only (overlap-contaminated) ---
n2m = nin2m[nin2m.month != "TOTAL"].copy(); n2m["ym"] = n2m.time + "-" + n2m.month.str[1:]
n2m = n2m[n2m.geo.str.len() == 4][["geo", "ym", "value"]].rename(columns={"value": "tot"})
c4 = ce_m[ce_m.geo.str.len() == 4].merge(n2m, on=["geo", "ym"], how="inner"); c4["t"] = c4.ym
g4 = set(c4.dropna().geo); c4 = c4[c4.geo.isin(g4)]
ok4 = summarize("OPTION 4: NUTS2 x month, I551-I553 total only (no hotel split)", c4, "geo", ["plat", "tot"])
pen("P_all", ok4, "plat", ["plat", "tot"])
p("  of which non-COVID (2022-01..):", (ok4.ym >= "2022-01").sum())
(O / "feasibility_summary.txt").write_text("\n".join(lines))
