"""Significance 1 (L1 threshold): GBM PDP/SHAP vs quadratic, region-block bootstrap; robustness 1/3, placebo, COVID sensitivity.
usage: python -I 06_L1_threshold.py DERIVED_DIR OUT_DIR [N_BOOT]"""
import sys, json, pathlib, numpy as np, pandas as pd, xgboost as xgb, shap
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
D, O = map(pathlib.Path, sys.argv[1:3]); NB = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
COVS = ["beds_per_1000", "hotel_bed_share", "log_gdp_pps_pc", "log_density", "log_pop",
        "city_night_share", "coastal_night_share", "season_gini_2022_23"]
P0 = pd.read_csv(D / "L1_panel.csv.gz", dtype={"time": str})
PARAMS = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, n_jobs=1, random_state=0)

def twfe(df, y, it=300):
    r = df[y] - df[y].mean()
    for _ in range(it):
        r = r - r.groupby(df.geo).transform("mean"); r = r - r.groupby(df.year).transform("mean")
    return r.values

def prep(df, outcome, treat):
    d = df.dropna(subset=COVS + [outcome, treat]).copy()
    d = d[d.groupby("geo").year.transform("count") >= 2]
    d["y"] = twfe(d, outcome)
    d["T"] = d[treat]; d["T_dev"] = d[treat] - d.groupby("geo")[treat].transform("mean")
    return d.reset_index(drop=True)

def feats(level=True): return (["T"] if level else []) + ["T_dev"] + COVS + ["year"]

def fit_gbm(d, F, depth, n):
    return xgb.XGBRegressor(max_depth=depth, n_estimators=n, **PARAMS).fit(d[F], d.y)

def pdp(model, d, F, var, grid):
    X = d[F].copy(); out = []
    for v in grid:
        X[var] = v; out.append(model.predict(X).mean())
    out = np.array(out); return out - out.mean()

def turning(grid, curve):
    """argmax of PDP; interior & sign change = rising slope before, falling after (linear fits on each side)."""
    k = int(np.argmax(curve)); n = len(grid)
    if k < 2 or k > n - 3: return dict(tp=np.nan, interior=False, k=k)
    sl = np.polyfit(grid[:k + 1], curve[:k + 1], 1)[0]; sr = np.polyfit(grid[k:], curve[k:], 1)[0]
    return dict(tp=float(grid[k]), interior=bool(sl > 0 and sr < 0), k=k, slope_left=float(sl), slope_right=float(sr),
                drop_right=float(curve[k] - curve[-1]), rise_left=float(curve[k] - curve[0]))

def quad(d, var):
    X = np.column_stack([np.ones(len(d)), d[var], d[var] ** 2, d[COVS], pd.get_dummies(d.year, drop_first=True).astype(float)])
    b = np.linalg.lstsq(X, d.y.values, rcond=None)[0]
    tp = -b[1] / (2 * b[2]) if b[2] != 0 else np.nan
    return dict(b1=b[1], b2=b[2], tp=tp), X, b

def cv_compare(d, F, var, depth, n):
    gkf = GroupKFold(5); res = {"gbm": [], "quad": [], "linear": []}; yy = []
    for tr, te in gkf.split(d, groups=d.geo):
        a, b = d.iloc[tr], d.iloc[te]
        res["gbm"].append(fit_gbm(a, F, depth, n).predict(b[F]))
        for k, pw in [("quad", 2), ("linear", 1)]:
            cols = lambda z: np.column_stack([np.ones(len(z))] + [z[var] ** j for j in range(1, pw + 1)] + [z[COVS]] +
                                             [pd.get_dummies(z.year).reindex(columns=sorted(d.year.unique()), fill_value=0).astype(float).values[:, 1:]])
            beta = np.linalg.lstsq(cols(a), a.y.values, rcond=None)[0]; res[k].append(cols(b) @ beta)
        yy.append(b.y.values)
    y = np.concatenate(yy); sst = ((y - y.mean()) ** 2).sum()
    return {k: dict(r2=float(1 - ((y - np.concatenate(v)) ** 2).sum() / sst),
                    rmse=float(np.sqrt(((y - np.concatenate(v)) ** 2).mean()))) for k, v in res.items()}

def tune(d, F):
    best = None
    for depth in [2, 3, 4]:
        for n in [200, 500]:
            gkf = GroupKFold(5); err = []
            for tr, te in gkf.split(d, groups=d.geo):
                m = fit_gbm(d.iloc[tr], F, depth, n); err.append(((m.predict(d.iloc[te][F]) - d.iloc[te].y) ** 2).sum())
            e = np.sqrt(sum(err) / len(d))
            if best is None or e < best[0]: best = (e, depth, n)
    return best

def boot_one(df_raw, outcome, treat, level, var, grid, depth, n, seed):
    rng = np.random.default_rng(seed); geos = df_raw.geo.unique()
    pick = rng.choice(geos, len(geos), replace=True)
    parts = [df_raw[df_raw.geo == g].assign(geo=f"{g}_{i}") for i, g in enumerate(pick)]
    b = pd.concat(parts, ignore_index=True)
    d = prep(b, outcome, treat); F = feats(level)
    m = fit_gbm(d, F, depth, n); c = pdp(m, d, F, var, grid); t = turning(grid, c)
    q, _, _ = quad(d, var)
    return dict(tp=t["tp"], interior=t["interior"], q_b2=q["b2"], q_tp=q["tp"], curve=c.tolist())

def run(tag, df_raw, outcome, treat, level=True, nb=NB):
    d = prep(df_raw, outcome, treat); F = feats(level); var = "T" if level else "T_dev"
    lo, hi = d[var].quantile([.05, .95]); grid = np.linspace(lo, hi, 41)
    rmse, depth, n = tune(d, F)
    m = fit_gbm(d, F, depth, n); curve = pdp(m, d, F, var, grid); t = turning(grid, curve)
    sv = shap.TreeExplainer(m).shap_values(d[F]); imp = dict(zip(F, np.abs(sv).mean(0).round(5).tolist()))
    q, X, b = quad(d, var)
    cv = cv_compare(d, F, var, depth, n)
    ols_slope = None
    if not level:  # within-only linear slope, region-clustered SE
        import statsmodels.api as sm
        Xo = sm.add_constant(d[["T_dev"]]); r = sm.OLS(d.y, Xo).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.geo)[0]})
        ols_slope = dict(beta=float(r.params["T_dev"]), se=float(r.bse["T_dev"]), p=float(r.pvalues["T_dev"]))
    bs = Parallel(n_jobs=100)(delayed(boot_one)(df_raw, outcome, treat, level, var, grid, depth, n, s) for s in range(nb))
    tps = np.array([x["tp"] for x in bs]); inter = np.array([x["interior"] for x in bs])
    qb2 = np.array([x["q_b2"] for x in bs]); qtp = np.array([x["q_tp"] for x in bs])
    curves = np.array([x["curve"] for x in bs])
    q_in = (qb2 < 0) & (qtp >= lo) & (qtp <= hi)
    res = dict(tag=tag, outcome=outcome, treat=treat, level_feature=level, n=len(d), regions=int(d.geo.nunique()),
               years=sorted(map(int, d.year.unique())), sd_y=float(d.y.std()), sd_T_dev=float(d.T_dev.std()),
               grid_p5_p95=[float(lo), float(hi)], gbm_depth=depth, gbm_trees=n, gbm_cv_rmse=float(rmse), cv=cv,
               gbm_turning=t, pdp_range=float(curve.max() - curve.min()),
               shap_mean_abs=imp, quad=dict(b1=float(q["b1"]), b2=float(q["b2"]), tp=float(q["tp"])),
               ols_within=ols_slope,
               boot=dict(n=nb, share_interior_signchange=float(inter.mean()),
                         tp_ci_interior=[float(np.nanpercentile(tps[inter], 2.5)), float(np.nanmedian(tps[inter])), float(np.nanpercentile(tps[inter], 97.5))] if inter.sum() > 10 else None,
                         tp_ci_all=[float(np.nanpercentile(tps, 2.5)), float(np.nanpercentile(tps, 97.5))] if np.isfinite(tps).sum() > 10 else None,
                         quad_share_b2neg=float((qb2 < 0).mean()), quad_share_tp_in_range=float(q_in.mean()),
                         quad_tp_ci=[float(np.percentile(qtp[q_in], 2.5)), float(np.median(qtp[q_in])), float(np.percentile(qtp[q_in], 97.5))] if q_in.sum() > 10 else None,
                         pdp_band_2_5=np.percentile(curves, 2.5, 0).round(5).tolist(), pdp_band_97_5=np.percentile(curves, 97.5, 0).round(5).tolist()),
               pdp_grid=grid.round(5).tolist(), pdp=curve.round(5).tolist())
    print(json.dumps({k: v for k, v in res.items() if k not in ("pdp_grid", "pdp")} | {"boot": {k: v for k, v in res["boot"].items() if not k.startswith("pdp_band")}}, default=float, indent=None))
    return res

noCOVID = P0[~P0.year.isin([2020, 2021])]
RUNSET = sys.argv[4] if len(sys.argv) > 4 else "v1"
if RUNSET == "fixed":  # penetration with denominator fixed at 2018-19 market size: no mechanical link to current I551
    runs = [run("MAIN-F pen_fixed=plat/(plat+I551)_2018-19, excl 2020-21", noCOVID, "log_I551", "pen_fixed"),
            run("ROB1-F within-deviation only", noCOVID, "log_I551", "pen_fixed", level=False),
            run("ROB3-F platform nights per resident", noCOVID, "log_I551", "plat_per_cap"),
            run("PLACEBO-F outcome=log I551 bedrooms t-4", noCOVID.assign(lp=np.log(noCOVID.I551_bedrooms_tm4)), "lp", "pen_fixed"),
            run("SENS-F excl 2025 (3 platforms)", noCOVID[noCOVID.year != 2025], "log_I551", "pen_fixed")]
    json.dump(runs, open(O / "L1_threshold_fixed.json", "w"), default=float, indent=1); sys.exit()
runs = [run("MAIN pen=plat/(plat+I551), excl 2020-21", noCOVID, "log_I551", "pen"),
        run("ROB1 within-deviation only", noCOVID, "log_I551", "pen", level=False),
        run("ROB3 pen_all=plat/(plat+I551-I553)", noCOVID, "log_I551", "pen_all"),
        run("PLACEBO outcome=log I551 bedrooms t-4", noCOVID.assign(lp=np.log(noCOVID.I551_bedrooms_tm4)), "lp", "pen"),
        run("SENS incl 2020-21", P0, "log_I551", "pen")]
json.dump(runs, open(O / "L1_threshold.json", "w"), default=float, indent=1)
