"""Significance 3 (L1 heterogeneity): CausalForestDML (econml). Y = log I551 nights residualised on region+year FE,
T = within-region deviation of penetration, X = baseline covariates + region-mean penetration level, W = year dummies.
Cross-fitting grouped by region. usage: python -I 08_L1_causal_forest.py DERIVED_DIR OUT_DIR [TREAT] [OUTCOME: log_I551 | placebo_bedrooms_tm4]"""
import sys, json, pathlib, numpy as np, pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
D, O = map(pathlib.Path, sys.argv[1:3]); TR = sys.argv[3] if len(sys.argv) > 3 else "pen"; SUF = "" if TR == "pen" else f"_{TR}"
OUTC = sys.argv[4] if len(sys.argv) > 4 else "log_I551"
if OUTC != "log_I551": SUF += "_placebo"
COVS = ["beds_per_1000", "hotel_bed_share", "log_gdp_pps_pc", "log_density", "log_pop",
        "city_night_share", "coastal_night_share", "season_gini_2022_23"]
P = pd.read_csv(D / "L1_panel.csv.gz", dtype={"time": str})
P["placebo_bedrooms_tm4"] = np.log(P.I551_bedrooms_tm4)
P = P[~P.year.isin([2020, 2021])].dropna(subset=COVS + [OUTC, TR])
P = P[P.groupby("geo").year.transform("count") >= 2].reset_index(drop=True)
def twfe(df, y, it=300):
    r = df[y] - df[y].mean()
    for _ in range(it):
        r = r - r.groupby(df.geo).transform("mean"); r = r - r.groupby(df.year).transform("mean")
    return r.values
P["y"] = twfe(P, OUTC); P["pen_level"] = P.groupby("geo")[TR].transform("mean"); P["T"] = P[TR] - P.pen_level
XC = COVS + ["pen_level"]
W = pd.get_dummies(P.year, drop_first=True).astype(float).values
gb = lambda: GradientBoostingRegressor(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8, random_state=0)
est = CausalForestDML(model_y=gb(), model_t=gb(), discrete_treatment=False, cv=5, n_estimators=2000,
                      min_samples_leaf=10, max_samples=0.45, honest=True, inference=True, random_state=0, n_jobs=-1)
est.fit(P.y.values, P["T"].values, X=P[XC].values, W=W, groups=pd.factorize(P.geo)[0])
cate = est.effect(P[XC].values) * 0.1            # effect of +10 pp penetration on log hotel nights
lo, hi = [z * 0.1 for z in est.effect_interval(P[XC].values, alpha=0.05)]
P["cate10"], P["cate10_lo"], P["cate10_hi"] = cate, lo, hi
reg = P.groupby("geo").agg(cate10=("cate10", "mean"), lo=("cate10_lo", "mean"), hi=("cate10_hi", "mean"),
                           pen_level=("pen_level", "first"), n_years=("year", "count")).reset_index()
reg.to_csv(O / f"L1_cate_by_region{SUF}.csv", index=False)
ate = est.ate_inference(X=P[XC].values)
imp = dict(zip(XC, np.round(est.feature_importances_, 4).tolist()))
# CATE by penetration-level quintile (region level)
reg["q"] = pd.qcut(reg.pen_level, 5, labels=[f"Q{i}" for i in range(1, 6)])
byq = reg.groupby("q", observed=True).agg(n=("geo", "count"), pen_min=("pen_level", "min"), pen_max=("pen_level", "max"),
                                         cate10_mean=("cate10", "mean"), share_neg=("cate10", lambda s: (s < 0).mean())).round(4)
# best linear projection of CATE on standardized covariates (region level)
import statsmodels.api as sm
Z = P[XC].apply(lambda c: (c - c.mean()) / c.std())
blp = sm.OLS(P.cate10, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(P.geo)[0]})
sig = reg[(reg.lo > 0) | (reg.hi < 0)]
res = dict(treat=TR, outcome=OUTC, n=len(P), regions=int(P.geo.nunique()), years=sorted(map(int, P.year.unique())), sd_T=float(P["T"].std()),
           ate_per10pp=dict(est=float(ate.mean_point) * 0.1, ci=[float(z) * 0.1 for z in ate.conf_int_mean(alpha=0.05)], p=float(ate.pvalue())),
           cate_region=dict(mean=float(reg.cate10.mean()), sd=float(reg.cate10.std()),
                            p10=float(reg.cate10.quantile(.1)), p90=float(reg.cate10.quantile(.9)),
                            width_p10_p90=float(reg.cate10.quantile(.9) - reg.cate10.quantile(.1)),
                            share_negative=float((reg.cate10 < 0).mean()),
                            n_ci_excludes_0=int(len(sig)), n_ci_below_0=int((reg.hi < 0).sum()), n_ci_above_0=int((reg.lo > 0).sum())),
           feature_importance=imp,
           blp=dict(coef={k: float(v) for k, v in blp.params.items()}, p={k: float(v) for k, v in blp.pvalues.items()}),
           by_pen_quintile=byq.reset_index().astype({"q": str}).to_dict("records"))
json.dump(res, open(O / f"L1_causal_forest{SUF}.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
