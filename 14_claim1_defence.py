"""C2: defend claim 1 (within-region association, region+year FE). Plus mechanical-coupling checks.
Sample = main L1 sample (excl 2020-21, complete covariates, >=2 years). SE clustered by region; wild cluster bootstrap
(Rademacher, null imposed, 9999 draws). usage: python -I 14_claim1_defence.py DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, numpy as np, pandas as pd
D, O = map(pathlib.Path, sys.argv[1:3])
COVS = ["beds_per_1000", "hotel_bed_share", "log_gdp_pps_pc", "log_density", "log_pop", "city_night_share", "coastal_night_share", "season_gini_2022_23"]
A = pd.read_csv(D / "L1_panel.csv.gz")
A["log_plat"] = np.log(A.plat); A["log_bedrm_tm4"] = np.log(A.I551_bedrooms_tm4)
lead = A.set_index(["geo", "year"])
for k in (1, 2):
    for v in ["pen", "pen_fixed"]:
        A[f"{v}_lead{k}"] = [lead[v].get((g, y + k), np.nan) for g, y in zip(A.geo, A.year)]
S0 = A[~A.year.isin([2020, 2021])].dropna(subset=COVS + ["log_I551", "pen"])
S0 = S0[S0.groupby("geo").year.transform("count") >= 2]

def demean(df, cols, it=500):
    X = df[cols].astype(float).copy()
    for _ in range(it):
        X = X - X.groupby(df.geo.values).transform("mean"); X = X - X.groupby(df.year.values).transform("mean")
    return X

def fe_reg(df, y, xs, wild=True, B=9999, seed=0):
    d = df.dropna(subset=[y] + xs); d = d[d.groupby("geo").year.transform("count") >= 2].reset_index(drop=True)
    Z = demean(d, [y] + xs); Y = Z[y].values; X = Z[xs].values
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ Y; e = Y - X @ b
    g = pd.factorize(d.geo)[0]; G = g.max() + 1
    meat = np.zeros((len(xs), len(xs)))
    sc = np.zeros((G, len(xs)))
    np.add.at(sc, g, X * e[:, None])
    meat = sc.T @ sc
    k = len(xs) + G + d.year.nunique() - 1
    adj = G / (G - 1) * (len(d) - 1) / (len(d) - k)
    V = adj * XtX_inv @ meat @ XtX_inv; se = np.sqrt(np.diag(V))
    from scipy import stats
    out = dict(n=len(d), regions=int(G), coef={x: float(b[i]) for i, x in enumerate(xs)}, se={x: float(se[i]) for i, x in enumerate(xs)},
               p={x: float(2 * (1 - stats.t.cdf(abs(b[i] / se[i]), G - 1))) for i, x in enumerate(xs)})
    if wild:  # WCR bootstrap for first regressor
        x0 = xs[0]; others = [i for i in range(len(xs)) if xs[i] != x0]
        Xr = X[:, others] if others else np.zeros((len(Y), 0))
        br = np.linalg.lstsq(Xr, Y, rcond=None)[0] if others else np.zeros(0)
        er = Y - (Xr @ br if others else 0); fit_r = Y - er
        t0 = b[0] / se[0]; rng = np.random.default_rng(seed); cnt = 0
        for _ in range(B):
            w = rng.choice([-1.0, 1.0], G)[g]; Ys = fit_r + er * w
            bs = XtX_inv @ X.T @ Ys; es = Ys - X @ bs
            scs = np.zeros((G, len(xs))); np.add.at(scs, g, X * es[:, None])
            Vs = adj * XtX_inv @ (scs.T @ scs) @ XtX_inv
            if abs(bs[0] / np.sqrt(Vs[0, 0])) >= abs(t0): cnt += 1
        out["wild_p"] = (cnt + 1) / (B + 1)
    return out

R = {}
R["claim1_pen (conventional share, region+year FE)"] = fe_reg(S0, "log_I551", ["pen"])
R["lead: pen_t + pen_t+1"] = fe_reg(S0, "log_I551", ["pen", "pen_lead1"])
R["lead: pen_t + pen_t+1 + pen_t+2"] = fe_reg(S0, "log_I551", ["pen", "pen_lead1", "pen_lead2"])
R["within placebo: log I551 bedrooms t-4 ~ pen"] = fe_reg(S0, "log_bedrm_tm4", ["pen"])
R["denominator I551-I553: pen_all"] = fe_reg(S0, "log_I551", ["pen_all"])
R["MECH-FREE: pen_fixed (2018-19 denominator)"] = fe_reg(S0, "log_I551", ["pen_fixed"])
R["MECH-FREE: log platform nights (elasticity)"] = fe_reg(S0, "log_I551", ["log_plat"])
R["MECH-FREE: platform nights per resident"] = fe_reg(S0, "log_I551", ["plat_per_cap"])
R["MECH-FREE lead: pen_fixed_t + pen_fixed_t+1"] = fe_reg(S0, "log_I551", ["pen_fixed", "pen_fixed_lead1"])
R["MECH-FREE within placebo: bedrooms t-4 ~ pen_fixed"] = fe_reg(S0, "log_bedrm_tm4", ["pen_fixed"])

# permutation: shuffle platform nights across years within region (no platform-hotel link), rebuild conventional share
rng = np.random.default_rng(1); perm = []
base = S0[["geo", "year", "plat", "I551", "log_I551"]].reset_index(drop=True)
for _ in range(500):
    pl = base.groupby("geo").plat.transform(lambda s: rng.permutation(s.values))
    d = base.assign(pen_perm=pl / (pl + base.I551))
    perm.append(fe_reg(d, "log_I551", ["pen_perm"], wild=False)["coef"]["pen_perm"])
perm = np.array(perm)
R["permutation (platform shuffled within region; conventional share)"] = dict(
    draws=500, mean_beta=float(perm.mean()), p2_5=float(np.percentile(perm, 2.5)), p97_5=float(np.percentile(perm, 97.5)),
    share_beta_le_observed=float((perm <= R["claim1_pen (conventional share, region+year FE)"]["coef"]["pen"]).mean()))
json.dump(R, open(O / "C2_claim1_defence.json", "w"), indent=1)
for k, v in R.items():
    if "coef" in v:
        x = list(v["coef"])[0]
        extra = " | ".join(f"{z}: {v['coef'][z]:+.4f} (se {v['se'][z]:.4f}, p {v['p'][z]:.3f})" for z in list(v["coef"])[1:])
        print(f"{k}: n={v['n']} G={v['regions']} b={v['coef'][x]:+.4f} se={v['se'][x]:.4f} p={v['p'][x]:.4f} wild_p={v.get('wild_p', float('nan')):.4f}" + (f" || {extra}" if extra else ""))
    else:
        print(k, v)
