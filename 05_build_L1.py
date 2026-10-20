"""Build L1 analysis panel (NUTS2 x year) with baseline covariates.
usage: python -I 05_build_L1.py LONG_DIR DERIVED_DIR"""
import sys, pathlib, numpy as np, pandas as pd
L, D = map(pathlib.Path, sys.argv[1:3])
rd = lambda c: pd.read_csv(L / f"{c}.csv.gz", dtype={"time": str, "flag": str}, keep_default_na=False, na_values=[""], low_memory=False)
n2 = lambda d: d[d.geo.str.len() == 4]

p = pd.read_csv(D / "panel_nuts2_year.csv.gz", dtype={"time": str})
p = p[(p.I551 > 0) & (p.plat > 0)][["geo", "time", "plat", "I551", "I551-I553", "I552_I553"]].copy()
p["year"] = p.time.astype(int)

cap = n2(rd("tour_cap_nuts2")); cap = cap[cap.unit == "NR"]
pop = n2(rd("demo_r_d2jan")); pop = pop[(pop.sex == "T") & (pop.age == "TOTAL")][["geo", "time", "value"]].rename(columns={"value": "pop"})
gdp = n2(rd("nama_10r_2gdp")); gdp = gdp[gdp.unit == "PPS_EU27_2020_HAB"][["geo", "time", "value"]].rename(columns={"value": "gdp_pps"})
dens = n2(rd("demo_r_d3dens"))[["geo", "time", "value"]].rename(columns={"value": "dens"})
dc = n2(rd("tour_occ_nin2dc")); dc = dc[(dc.unit == "NR") & (dc.c_resid == "TOTAL") & (dc.nace_r2 == "I551-I553")]
m = n2(rd("tour_occ_nin2m")); m = m[(m.unit == "NR") & (m.c_resid == "TOTAL") & (m.month != "TOTAL")]

def base(df, col, years):  # mean over baseline years, per region
    return df[df.time.isin(years)].groupby("geo")[col].mean()

B = ["2018", "2019"]
cov = pd.DataFrame(index=sorted(p.geo.unique()))
bed_all = cap[(cap.accomunit == "BEDPL") & (cap.nace_r2 == "I551-I553")][["geo", "time", "value"]]
bed_hot = cap[(cap.accomunit == "BEDPL") & (cap.nace_r2 == "I551")][["geo", "time", "value"]]
cov["pop_2018_19"] = base(pop, "pop", B)
cov["beds_per_1000"] = base(bed_all, "value", B) / cov.pop_2018_19 * 1000
cov["hotel_bed_share"] = base(bed_hot, "value", B) / base(bed_all, "value", B)
cov["gdp_pps_pc"] = base(gdp, "gdp_pps", B)
cov["pop_density"] = base(dens, "dens", B)
tot = dc[dc.terrtypo == "TOTAL"].set_index(["geo", "time"]).value
for t, name in [("DEG1", "city_night_share"), ("CST_A", "coastal_night_share")]:
    s = dc[dc.terrtypo == t].set_index(["geo", "time"]).value / tot
    cov[name] = s.reset_index().pipe(lambda x: x[x.time.isin(B)]).groupby("geo").value.mean()
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())
g = m.dropna(subset=["value"]).groupby(["geo", "time"]).filter(lambda x: len(x) == 12)
gg = g.groupby(["geo", "time"]).value.apply(gini).reset_index()
cov["season_gini_2022_23"] = gg[gg.time.isin(["2022", "2023"])].groupby("geo").value.mean()
cov.index.name = "geo"
cov["log_pop"] = np.log(cov.pop_2018_19); cov["log_gdp_pps_pc"] = np.log(cov.gdp_pps_pc); cov["log_density"] = np.log(cov.pop_density)

# placebo outcome: hotel (I551) bedrooms 4 years earlier
rm = cap[(cap.accomunit == "BEDRM") & (cap.nace_r2 == "I551")][["geo", "time", "value"]].copy()
rm["year"] = rm.time.astype(int) + 4; rm = rm.rename(columns={"value": "I551_bedrooms_tm4"})[["geo", "year", "I551_bedrooms_tm4"]]

p = p.merge(cov.reset_index(), on="geo", how="left").merge(rm, on=["geo", "year"], how="left")
p["pen"] = p.plat / (p.plat + p.I551)
p["pen_all"] = p.plat / (p.plat + p["I551-I553"])
p["log_I551"] = np.log(p.I551)
# mechanical-free penetration: denominator fixed at the region's 2018-19 market size (only platform nights move it)
b = p[p.time.isin(B)].groupby("geo")[["plat", "I551"]].mean()
p["pen_fixed"] = p.plat / p.geo.map(b.plat + b.I551)
p["plat_per_cap"] = p.plat / p.pop_2018_19   # platform nights per resident (2018-19 population)
p.to_csv(D / "L1_panel.csv.gz", index=False)
covs = ["beds_per_1000", "hotel_bed_share", "log_gdp_pps_pc", "log_density", "log_pop", "city_night_share", "coastal_night_share", "season_gini_2022_23"]
print("rows", len(p), "regions", p.geo.nunique())
print("covariate non-missing regions:\n", cov[covs].notna().sum().to_string())
print("rows complete on all covs:", p.dropna(subset=covs).shape[0], "regions", p.dropna(subset=covs).geo.nunique())
print("placebo non-missing rows:", p.I551_bedrooms_tm4.notna().sum())
print(cov[covs].describe().round(3).T.to_string())
