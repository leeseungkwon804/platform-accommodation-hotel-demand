"""Significance 2 (L2 forecast value): rolling-origin h=3,6 on country x month log hotel (I551) nights.
Models with / without platform features: SARIMAX, country-FE linear (direct), GBM (direct); seasonal naive reference.
Evaluation targets 2023-01..2026-03 (seasonal lag year never inside COVID 2020-03..2021-12).
usage: python -I 07_L2_forecast.py DERIVED_DIR OUT_DIR [PUB_LAG]
PUB_LAG = months by which platform features are lagged (0 = same-month information set; 3 = Eurostat release lag of about one quarter)."""
import sys, json, warnings, pathlib, numpy as np, pandas as pd, xgboost as xgb
from joblib import Parallel, delayed
from scipy import stats
warnings.filterwarnings("ignore")
D, O = map(pathlib.Path, sys.argv[1:3]); LAG = int(sys.argv[3]) if len(sys.argv) > 3 else 0; SUF = f"_lag{LAG}" if LAG else ""
p = pd.read_csv(D / "panel_country_month.csv.gz", dtype={"ym": str})
p = p[(p.I551 > 0) & (p.plat > 0)].copy()
p["t"] = pd.PeriodIndex(p.ym, freq="M"); p = p.sort_values(["geo", "t"])
p["y"] = np.log(p.I551); p["x"] = np.log(p.plat); p["pen"] = p.plat / (p.plat + p.I551)
geos = sorted(p.geo.unique())
T0, T1 = pd.Period("2018-01", "M"), pd.Period("2026-03", "M")
full = pd.MultiIndex.from_product([geos, pd.period_range(T0, T1, freq="M")], names=["geo", "t"])
W = p.set_index(["geo", "t"])[["y", "x", "pen"]].reindex(full)
C0, C1 = pd.Period("2020-03", "M"), pd.Period("2021-12", "M")
covid = lambda t: (t >= C0) & (t <= C1)
EVAL0 = pd.Period("2023-01", "M")

def series(g):
    s = W.loc[g]; s.index = s.index.to_timestamp(); return s

# ---------- SARIMAX (per country, refit each origin) ----------
from statsmodels.tsa.statespace.sarimax import SARIMAX
def sarimax_one(g, h, origin, with_plat):
    s = series(g); y = s.y; tr_end = origin.to_timestamp()
    d = pd.DataFrame({"y": y, "cov": covid(s.index.to_period("M")).astype(float), "xl": s.x.shift(h + LAG)})
    d = d[d.index >= (T0 + h + LAG).to_timestamp()]
    tr = d[d.index <= tr_end]; fut_idx = pd.date_range(tr_end, periods=h + 1, freq="MS")[1:]
    if tr.y.isna().any() or tr.xl.isna().any() or fut_idx[-1] not in d.index: return None
    cols = ["cov", "xl"] if with_plat else ["cov"]
    exf = d.loc[fut_idx, cols]
    if exf.isna().any().any(): return None
    try:
        m = SARIMAX(tr.y, exog=tr[cols], order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="c").fit(disp=False, maxiter=200)
        return float(m.forecast(h, exog=exf).iloc[-1])
    except Exception:
        return None

# ---------- direct panel features ----------
def build_direct(h):
    rows = []
    for g in geos:
        s = W.loc[g]
        for t in s.index:
            tgt = t + h
            if tgt > T1 or t - 12 - LAG < T0: continue
            r = dict(geo=g, origin=t, target=tgt, month=tgt.month,
                     z=s.y.get(tgt) - s.y.get(tgt - 12) if tgt - 12 >= T0 else np.nan,
                     base=s.y.get(tgt - 12) if tgt - 12 >= T0 else np.nan, ytrue=s.y.get(tgt),
                     a=s.y[t] - s.y[t - 12], gx=s.x[t - LAG] - s.x[t - 12 - LAG], pen=s.pen[t - LAG], dpen=s.pen[t - LAG] - s.pen[t - 12 - LAG])
            r["covid_touch"] = any(covid(q) for q in [t, t - 12, tgt, tgt - 12])
            rows.append(r)
    return pd.DataFrame(rows)

def fe_design(df, plat, geos_cols):
    X = [np.ones(len(df))]
    X += [(df.geo == g).astype(float).values for g in geos_cols[1:]]
    X += [(df.month == m).astype(float).values for m in range(2, 13)]
    X += [df.a.values]
    if plat: X += [df.gx.values, df.pen.values, df.dpen.values]
    return np.column_stack(X)

def direct_forecasts(h):
    dd = build_direct(h).dropna(subset=["a", "base"])
    dd = dd.dropna(subset=["gx", "pen", "dpen"])
    ev = dd[(dd.target >= EVAL0)].copy()
    out = []
    for origin in sorted(ev.origin.unique()):
        tr = dd[(dd.target <= origin) & (~dd.covid_touch)].dropna(subset=["z"])
        te = ev[ev.origin == origin]
        for plat in [False, True]:
            Xtr, Xte = fe_design(tr, plat, geos), fe_design(te, plat, geos)
            b = np.linalg.lstsq(Xtr, tr.z.values, rcond=None)[0]
            fe = te.base.values + Xte @ b
            F = ["a", "month", "geo_code"] + (["gx", "pen", "dpen"] if plat else [])
            trg = tr.assign(geo_code=pd.Categorical(tr.geo, categories=geos).codes); teg = te.assign(geo_code=pd.Categorical(te.geo, categories=geos).codes)
            m = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8, min_child_weight=5, n_jobs=4, random_state=0).fit(trg[F], trg.z)
            gb = te.base.values + m.predict(teg[F])
            for i, (_, r) in enumerate(te.iterrows()):
                out.append(dict(geo=r.geo, target=str(r.target), h=h, model="FE_linear", plat=plat, yhat=fe[i], ytrue=r.ytrue))
                out.append(dict(geo=r.geo, target=str(r.target), h=h, model="GBM", plat=plat, yhat=gb[i], ytrue=r.ytrue))
    return out

def snaive(h):
    out = []
    for g in geos:
        s = W.loc[g]
        for tgt in pd.period_range(EVAL0, T1, freq="M"):
            if (tgt - h) < T0: continue
            out.append(dict(geo=g, target=str(tgt), h=h, model="SNaive", plat=False, yhat=s.y.get(tgt - 12), ytrue=s.y.get(tgt)))
    return out

recs = []
for h in [3, 6]:
    recs += snaive(h); recs += direct_forecasts(h)
    origins = pd.period_range(EVAL0 - h, T1 - h, freq="M")
    jobs = [(g, h, o, wp) for g in geos for o in origins for wp in [False, True]]
    res = Parallel(n_jobs=120)(delayed(sarimax_one)(*j) for j in jobs)
    for (g, h_, o, wp), r in zip(jobs, res):
        recs.append(dict(geo=g, target=str(o + h_), h=h_, model="SARIMAX", plat=wp, yhat=r, ytrue=W.loc[(g, o + h_), "y"]))
F = pd.DataFrame(recs); F["e"] = F.ytrue - F.yhat
F.to_csv(O / f"L2_forecasts{SUF}.csv.gz", index=False)

def dm(e1, e2, h, key):  # e1 = without, e2 = with; loss differential per target month averaged across countries
    d = pd.concat([e1.rename("a"), e2.rename("b")], axis=1).dropna()
    lf = (lambda z: z ** 2) if key == "mse" else np.abs
    dl = (lf(d.a) - lf(d.b)).groupby(level="target").mean().sort_index().values
    T = len(dl); mu = dl.mean(); lag = h - 1
    gam = [np.mean((dl[k:] - mu) * (dl[:T - k] - mu)) for k in range(lag + 1)]
    var = (gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, lag + 1))) / T
    stat = mu / np.sqrt(var) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)  # Harvey-Leybourne-Newbold
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), T - 1))), T

summary = []
for h in [3, 6]:
    Fh = F[F.h == h]
    # common evaluation set: (geo,target) present for every model/variant
    key = Fh.dropna(subset=["e"]).groupby(["geo", "target"]).size()
    nmod = Fh.groupby(["model", "plat"]).ngroups
    common = key[key == nmod].index
    E = Fh.dropna(subset=["e"]).set_index(["geo", "target"]).loc[lambda z: z.index.isin(common)]
    sn = E[E.model == "SNaive"].e
    row0 = dict(h=h, model="SNaive", n=len(sn), countries=sn.index.get_level_values(0).nunique(),
                rmse=float(np.sqrt((sn ** 2).mean())), mae=float(sn.abs().mean()))
    summary.append(row0)
    for mname in ["SARIMAX", "FE_linear", "GBM"]:
        a = E[(E.model == mname) & (~E.plat)].e; b = E[(E.model == mname) & (E.plat)].e
        r = dict(h=h, model=mname, n=len(a), countries=a.index.get_level_values(0).nunique(), targets=a.index.get_level_values(1).nunique(),
                 rmse_without=float(np.sqrt((a ** 2).mean())), rmse_with=float(np.sqrt((b ** 2).mean())),
                 mae_without=float(a.abs().mean()), mae_with=float(b.abs().mean()))
        r["rmse_reduction_pct"] = 100 * (r["rmse_without"] - r["rmse_with"]) / r["rmse_without"]
        r["mae_reduction_pct"] = 100 * (r["mae_without"] - r["mae_with"]) / r["mae_without"]
        r["dm_mse_stat"], r["dm_mse_p"], r["dm_T"] = dm(a, b, h, "mse")
        r["dm_mae_stat"], r["dm_mae_p"], _ = dm(a, b, h, "mae")
        r["rmse_with_vs_snaive_pct"] = 100 * (row0["rmse"] - r["rmse_with"]) / row0["rmse"]
        r["rmse_without_vs_snaive_pct"] = 100 * (row0["rmse"] - r["rmse_without"]) / row0["rmse"]
        summary.append(r)
S = pd.DataFrame(summary); S["pub_lag"] = LAG; S.to_csv(O / f"L2_forecast_summary{SUF}.csv", index=False)
pd.set_option("display.width", 250); print(S.round(4).to_string(index=False))
