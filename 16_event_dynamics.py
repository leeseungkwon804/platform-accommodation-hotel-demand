"""Claim A support: event-time coefficients, 95% CI and minimum detectable effect (MDE) for the stacked event study.
Same sample/stacking as 13_event_study.py. Event-time bins: <=-4, -3, -2, [-1 reference], 0, 1, 2, >=3.
MDE (two-sided alpha 0.05, power 0.80) = (z_0.975 + z_0.80) * SE of the pooled DiD coefficient.
Full-transfer benchmark: pre-period platform/hotel nights ratio x estimated platform drop.
Wild cluster bootstrap (Rademacher, null imposed, clusters = region, 9999 draws) for the pooled DiD coefficient.
usage: python -I 16_event_dynamics.py DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
D, O = map(pathlib.Path, sys.argv[1:3])
p = pd.read_csv(D / "L1_panel.csv.gz")[["geo", "year", "plat", "I551"]]
lx = p[p.geo.isin(["PT1A", "PT1B"])].groupby("year")[["plat", "I551"]].sum().reset_index().assign(geo="PT17")
p = pd.concat([p[~p.geo.isin(["PT1A", "PT1B"])], lx], ignore_index=True)
p = p[~p.year.isin([2020, 2021])]
p["log_I551"], p["log_plat"] = np.log(p.I551), np.log(p.plat)
ev = pd.read_csv(D / "regulation_events.csv")
ev = ev[(ev.source_level.str.strip() == "1차") & (ev.nuts2.str.len() == 4)].copy()
ev["eff"] = pd.to_datetime(ev.date_effective)
ev["treat_year"] = ev.eff.dt.year + (ev.eff.dt.month > 6).astype(int)
first = ev.sort_values("eff").groupby("nuts2").first().reset_index()[["nuts2", "treat_year"]]
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
BINS = {"m4": lambda r: r <= -4, "m3": lambda r: r == -3, "m2": lambda r: r == -2,
        "p0": lambda r: r == 0, "p1": lambda r: r == 1, "p2": lambda r: r == 2, "p3": lambda r: r >= 3}
LAB = {"m4": "<=-4", "m3": "-3", "m2": "-2", "p0": "0", "p1": "1", "p2": "2", "p3": ">=3"}
for k, f in BINS.items():
    S[k] = S.treated * f(S.rel).astype(int)

def demean2(d, cols, it=300):
    X = d[cols].astype(float).copy()
    for _ in range(it):
        X = X - X.groupby(d.sg.values).transform("mean"); X = X - X.groupby(d.sy.values).transform("mean")
    return X

def wild_p(d, y, x, B=9999, seed=0):
    Z = demean2(d, [y, x]); Y, X = Z[y].values, Z[x].values
    g = pd.factorize(d.geo)[0]; G = g.max() + 1
    def tstat(Yb):
        b = (X @ Yb) / (X @ X); e = Yb - X * b
        sc = np.zeros(G); np.add.at(sc, g, X * e)
        return b / np.sqrt((sc ** 2).sum() / (X @ X) ** 2)
    t0 = tstat(Y); e0 = Y  # null imposed: beta = 0, restricted residual = Y (FE already partialled out)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], G)[g]
        cnt += abs(tstat(e0 * w)) >= abs(t0)
    return float((cnt + 1) / (B + 1))

z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
res = dict(n_treated=int(len(first)), n_controls=len(controls), rows=len(S), mde_multiplier=float(z),
           cells_by_bin={LAB[k]: int(S[k].sum()) for k in BINS},
           years_in_bin_le_m4=sorted(int(y) for y in S.loc[S["m4"] == 1, "year"].unique()),
           treat_years_in_bin_le_m4=sorted(int(y) for y in (S.loc[S["m4"] == 1, "year"] - S.loc[S["m4"] == 1, "rel"]).unique()))
for y in ["log_I551", "log_plat"]:
    d = S.dropna(subset=[y]).reset_index(drop=True)
    cl = pd.factorize(d.geo)[0]
    m1 = smf.ols(f"{y} ~ post + C(sg) + C(sy)", d).fit(cov_type="cluster", cov_kwds={"groups": cl})
    m2 = smf.ols(f"{y} ~ " + " + ".join(BINS) + " + C(sg) + C(sy)", d).fit(cov_type="cluster", cov_kwds={"groups": cl})
    b, se = float(m1.params["post"]), float(m1.bse["post"])
    lo, hi = b - 1.96 * se, b + 1.96 * se
    res[y] = dict(did=dict(b=b, se=se, p=float(m1.pvalues["post"]), ci95=[lo, hi], ci95_pct=[np.expm1(lo) * 100, np.expm1(hi) * 100],
                           pct=float(np.expm1(b) * 100), mde_log=float(z * se), mde_pct=float(np.expm1(z * se) * 100),
                           wild_p=wild_p(d, y, "post")),
                  event=[dict(rel=LAB[k], b=float(m2.params[k]), se=float(m2.bse[k]), p=float(m2.pvalues[k]),
                              lo=float(m2.params[k] - 1.96 * m2.bse[k]), hi=float(m2.params[k] + 1.96 * m2.bse[k])) for k in BINS],
                  pre_joint_p=float(m2.f_test("m4 = 0, m3 = 0, m2 = 0").pvalue))
# benchmark: hotel change implied if every lost platform night had moved to hotels (pre-period plat/I551 ratio x platform drop)
drop = -np.expm1(res["log_plat"]["did"]["b"])
pre = S[(S.treated == 1) & (S.rel < 0)].groupby("geo")[["plat", "I551"]].mean()
ratio = pre.plat / pre.I551
res["full_transfer_benchmark"] = dict(platform_drop_pct=float(drop * 100), ratio_pre_by_region=ratio.round(4).to_dict(),
                                      ratio_median=float(ratio.median()), implied_hotel_pct_median=float(ratio.median() * drop * 100),
                                      implied_hotel_pct_min=float(ratio.min() * drop * 100), implied_hotel_pct_max=float(ratio.max() * drop * 100))
h = res["log_I551"]["did"]; b0 = float(np.log1p(ratio.median() * drop))  # H0: hotel effect equals median full-transfer benchmark
tt = (h["b"] - b0) / h["se"]
res["full_transfer_benchmark"].update(test_b0_log=b0, test_t=float(tt), test_p_two_sided=float(2 * stats.norm.sf(abs(tt))),
                                      test_p_one_sided_below=float(stats.norm.cdf(tt)))
json.dump(res, open(O / "L1_event_dynamics.json", "w"), indent=1); print(json.dumps(res, indent=1))
