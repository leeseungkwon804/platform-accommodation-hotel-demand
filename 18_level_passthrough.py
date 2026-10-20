"""c10dec5 item 1: level (nights) stacked event study and pass-through share pi = -D_hotel / D_plat.
Sample, stacking, controls, fixed effects (region x stack, year x stack) and clustering (region) are identical to 16_event_dynamics.py;
only the outcome changes from log to level: plat, I551 (hotel) and total = plat + I551.
Because the three regressions share sample, regressors and fixed effects, OLS is linear in y and D_total = D_hotel + D_plat exactly.
  full transfer  H0: D_total = 0       (<=> D_hotel = -D_plat)  -> null-imposed wild bootstrap on the total regression
  no substitution H0: D_total = D_plat (<=> D_hotel = 0)        -> null-imposed wild bootstrap on the hotel regression
Wild cluster bootstrap: Rademacher, 9,999 draws, clusters = region, CR1 scaling (G/(G-1)), restricted residuals re-projected on the FE.
pi CI: region-level block bootstrap stratified by treated (12) / control (143), 9,999 draws, percentile 95%.
Fieller interval for pi from the cluster-robust joint covariance of (D_hotel, D_plat) is reported in every case.
Two weightings: none (primary, as in 16) and 2018-19 mean hotel nights of the region.
Validation: the same code on log outcomes must reproduce the 16 DiD coefficients.
usage: python -I 18_level_passthrough.py DERIVED_DIR OUT_DIR"""
import sys, json, pathlib, numpy as np, pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
D, O = map(pathlib.Path, sys.argv[1:3])
B = int(sys.argv[3]) if len(sys.argv) > 3 else 9999
# ---- sample and stacking: verbatim from 16_event_dynamics.py ----
p = pd.read_csv(D / "L1_panel.csv.gz")[["geo", "year", "plat", "I551"]]
lx = p[p.geo.isin(["PT1A", "PT1B"])].groupby("year")[["plat", "I551"]].sum().reset_index().assign(geo="PT17")
p = pd.concat([p[~p.geo.isin(["PT1A", "PT1B"])], lx], ignore_index=True)
w1819 = p[p.year.isin([2018, 2019])].groupby("geo").I551.mean()
p = p[~p.year.isin([2020, 2021])]
p["log_I551"], p["log_plat"] = np.log(p.I551), np.log(p.plat)
p["total"] = p.plat + p.I551
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
S["post"] = S.treated * (S.rel >= 0)
S["w"] = S.geo.map(w1819)  # mean of available 2018-19 years; regions with neither year get NaN and drop out of the weighted run only
S["w"] = S.w.fillna(0.0)
assert S[["plat", "I551", "log_plat", "log_I551"]].notna().all().all()
Y_LEV, Y_LOG = ["plat", "I551", "total"], ["log_plat", "log_I551"]
treated = list(first.nuts2); TY = dict(zip(first.nuts2, first.treat_year))

def arrays(S):
    sg = pd.factorize(S["stack"] + "_" + S.geo)[0]; sy = pd.factorize(S["stack"] + "_" + S.year.astype(str))[0]
    cl = pd.factorize(S.geo)[0]
    return sg, sy, cl

def fwl(sg, sy, x, Y, w):
    """Partial region x stack (exact demeaning) and year x stack (explicit dummies after demeaning) out of x and Y (weighted)."""
    ws = np.bincount(sg, w)
    dm = lambda M: M - (np.stack([np.bincount(sg, w * M[:, j]) for j in range(M.shape[1])], 1) / ws[:, None])[sg]
    Dm = np.zeros((len(sy), sy.max() + 1)); Dm[np.arange(len(sy)), sy] = 1
    Dm = dm(Dm); sw = np.sqrt(w)
    Q, R = np.linalg.qr(Dm * sw[:, None]); keep = np.abs(np.diag(R)) > 1e-9 * np.abs(np.diag(R)).max()
    Q = Q[:, keep]
    proj = lambda M: (lambda Ms: (Ms - Q @ (Q.T @ Ms)) / sw[:, None])(dm(M) * sw[:, None])
    return proj(x[:, None])[:, 0], proj(Y), proj, int(sg.max() + 1 + Q.shape[1])

def est(xt, Yt, w):
    return (w * xt) @ Yt / ((w * xt) @ xt)

def cluster_cov(xt, Yt, w, cl, k_fe):
    """CR1 joint covariance of the post coefficients across columns of Y (same regressor, same sample)."""
    b = est(xt, Yt, w); E = Yt - np.outer(xt, b)
    G = cl.max() + 1; N = len(xt); K = k_fe + 1
    sc = np.stack([np.bincount(cl, w * xt * E[:, j], G) for j in range(E.shape[1])], 1) / ((w * xt) @ xt)
    c = G / (G - 1) * (N - 1) / (N - K)
    return b, c * sc.T @ sc, G

def wild(xt, y0t, proj, w, cl, seed):
    """Null-imposed (beta = 0) wild cluster bootstrap p; y0t = y residualised on the FE only (restricted residual)."""
    G = cl.max() + 1; den = (w * xt) @ xt
    def tstat(Ymat):
        b = (w * xt) @ Ymat / den
        Yp = proj(Ymat); E = Yp - np.outer(xt, (w * xt) @ Yp / den)
        sc = np.stack([np.bincount(cl, w * xt * E[:, j], G) for j in range(E.shape[1])], 1)
        return b / np.sqrt((sc ** 2).sum(0) / den ** 2)
    t0 = tstat(y0t[:, None])[0]
    rng = np.random.default_rng(seed); cnt = 0
    for i in range(0, B, 500):
        V = rng.choice([-1.0, 1.0], (G, min(500, B - i)))
        cnt += (np.abs(tstat(y0t[:, None] * V[cl])) >= abs(t0)).sum()
    return float((cnt + 1) / (B + 1)), float(t0)

def fit(S, ycols, weighted):
    if weighted: S = S[S.w > 0].reset_index(drop=True)
    sg, sy, cl = arrays(S); w = S.w.values.astype(float) if weighted else np.ones(len(S))
    xt, Yt, proj, k_fe = fwl(sg, sy, S.post.values.astype(float), S[ycols].values.astype(float), w)
    return sg, sy, cl, w, xt, Yt, proj, k_fe, S

# ---- bootstrap: region blocks, stratified treated / control ----
PA, YA, WA = S.post.values.astype(float), S[Y_LEV].values.astype(float), S.w.values.astype(float)
rows_tc = {(e, c): S.index[(S["stack"] == e) & (S.geo == c)].values for e in treated for c in controls + [e]}
def boot_chunk(args):
    seed, n = args; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        tdraw = rng.choice(len(treated), len(treated)); cdraw = rng.choice(len(controls), len(controls))
        idx, sg, sy, cl = [], [], [], []; gid = 0
        for k, ti in enumerate(tdraw):
            e = treated[ti]
            for j, ci in enumerate(list(cdraw) + [None]):
                r = rows_tc[(e, controls[ci] if ci is not None else e)]
                idx.append(r); sg.append(np.full(len(r), gid)); gid += 1
                cl.append(np.full(len(r), j if ci is not None else len(controls) + k))
                sy.append(k * 100 + (S.year.values[r] - 2000))
        idx = np.concatenate(idx); sg = np.concatenate(sg); cl = np.concatenate(cl); sy = pd.factorize(np.concatenate(sy))[0]
        res = []
        for weighted in (False, True):
            m = WA[idx] > 0 if weighted else np.ones(len(idx), bool)
            w = WA[idx][m] if weighted else np.ones(len(idx))
            xt, Yt, _, _ = fwl(pd.factorize(sg[m])[0], pd.factorize(sy[m])[0], PA[idx][m], YA[idx][m], w)
            res.append(est(xt, Yt, w))
        out.append(np.concatenate(res))
    return np.array(out)

if __name__ == "__main__":
    res = dict(spec="identical to 16_event_dynamics.py (stacks, 143 controls, region x stack + year x stack FE, cluster = region, 2020-21 excluded); outcome in levels",
               n_treated=len(treated), n_controls=len(controls), rows=len(S), B=B,
               weights_note="weighted = region mean hotel nights (I551) 2018-19", identity="D_total = D_hotel + D_plat (exact)")
    # validation against 16 (log outcomes, unweighted)
    sg, sy, cl, w, xt, Yt, proj, _, _ = fit(S, Y_LOG, False)
    e16 = json.load(open(O / "L1_event_dynamics.json"))
    res["validation_log_vs_16"] = {y: dict(here=float(est(xt, Yt[:, i], w)), script16=e16[y]["did"]["b"]) for i, y in enumerate(Y_LOG)}
    for y in Y_LOG: assert abs(res["validation_log_vs_16"][y]["here"] - res["validation_log_vs_16"][y]["script16"]) < 1e-8
    # pre-period means of treated regions (scale)
    pre = S[(S.treated == 1) & (S.rel < 0)].groupby("geo")[["plat", "I551"]].mean()
    res["treated_pre_mean_nights"] = dict(plat_mean=float(pre.plat.mean()), I551_mean=float(pre.I551.mean()),
                                          plat_median=float(pre.plat.median()), I551_median=float(pre.I551.median()))
    with ProcessPoolExecutor(min(100, 128)) as ex:
        chunks = [(1000 + i, n) for i, n in enumerate([B // 100] * 100 + ([B % 100] if B % 100 else [])) if n]
        BS = np.vstack(list(ex.map(boot_chunk, chunks)))
    assert len(BS) == B
    np.save(O / "L1_level_passthrough_boot.npy", BS)
    for wi, weighted in enumerate((False, True)):
        key = "weighted" if weighted else "unweighted"
        sg, sy, cl, w, xt, Yt, proj, k_fe, Sx = fit(S, Y_LEV, weighted)
        b, V, G = cluster_cov(xt, Yt, w, cl, k_fe)
        q = stats.t.ppf(0.975, G - 1)
        # restricted residuals (FE only) for null-imposed wild bootstrap
        r = {}
        for i, y in enumerate(Y_LEV):
            y0t = proj(Sx[[y]].values.astype(float))[:, 0]  # FE-only residual (post is not in proj) = restricted residual under beta = 0
            wp, t0 = wild(xt, y0t, proj, w, cl, seed=10 + i)
            se = float(np.sqrt(V[i, i]))
            r[y] = dict(b=float(b[i]), se=se, t=float(b[i] / se), p_t_G1=float(2 * stats.t.sf(abs(b[i] / se), G - 1)),
                        ci95=[float(b[i] - q * se), float(b[i] + q * se)], wild_p_H0_zero=wp)
        assert abs(r["total"]["b"] - r["plat"]["b"] - r["I551"]["b"]) < 1e-6 * abs(r["plat"]["b"])
        r["total"]["hypothesis_zero"] = "full transfer (D_total = 0)"
        r["I551"]["hypothesis_zero"] = "no substitution (D_hotel = 0 <=> D_total = D_plat)"
        r["benchmark_tests"] = dict(full_transfer_wild_p=r["total"]["wild_p_H0_zero"], no_substitution_wild_p=r["I551"]["wild_p_H0_zero"],
                                    full_transfer_t_p=r["total"]["p_t_G1"], no_substitution_t_p=r["I551"]["p_t_G1"])
        bh, bp = b[1], b[0]; pi = float(-bh / bp)
        bs = BS[:, wi * 3:(wi + 1) * 3]; pib = -bs[:, 1] / bs[:, 0]
        plat_ci = np.percentile(bs[:, 0], [2.5, 97.5])
        # Fieller for pi = -bh/bp: solve (bh + pi*bp)^2 = q^2 (Vhh + 2 pi Vhp + pi^2 Vpp)
        Vh, Vp, Vhp = V[1, 1], V[0, 0], V[0, 1]
        a = bp ** 2 - q ** 2 * Vp; bb = 2 * (bh * bp - q ** 2 * Vhp); c = bh ** 2 - q ** 2 * Vh
        disc = bb ** 2 - 4 * a * c
        fq = lambda x: a * x ** 2 + bb * x + c  # pi is in the Fieller set iff fq(pi) <= 0
        fl_in = dict(contains_0=bool(fq(0.0) <= 0), contains_1=bool(fq(1.0) <= 0))
        if a > 0 and disc >= 0:
            fl = dict(type="bounded", ci95=sorted([float((-bb - np.sqrt(disc)) / (2 * a)), float((-bb + np.sqrt(disc)) / (2 * a))]))
        else:
            fl = dict(type="unbounded" if a <= 0 else "no_real_roots", a=float(a), disc=float(disc))
        fl.update(fl_in)
        boot_ok = bool(plat_ci[1] < 0 or plat_ci[0] > 0)
        pct = np.percentile(pib, [2.5, 97.5])
        r["pi"] = dict(estimate=pi, boot_pct_ci95=[float(pct[0]), float(pct[1])], boot_median=float(np.median(pib)),
                       boot_share_pi_ge_1=float((pib >= 1).mean()), boot_share_pi_le_0=float((pib <= 0).mean()),
                       boot_D_plat_ci95=[float(plat_ci[0]), float(plat_ci[1])], boot_D_plat_excludes_zero=boot_ok,
                       boot_D_hotel_ci95=[float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])],
                       boot_D_total_ci95=[float(x) for x in np.percentile(bs[:, 2], [2.5, 97.5])],
                       fieller=fl, primary_ci="bootstrap percentile" if boot_ok else "Fieller (bootstrap D_plat CI covers 0)",
                       q_t_G1=float(q), G=int(G), cov_hotel_plat=float(Vhp), corr_hotel_plat=float(Vhp / np.sqrt(Vh * Vp)))
        if boot_ok:
            ci = r["pi"]["boot_pct_ci95"]; inc0, inc1 = ci[0] <= 0 <= ci[1], ci[0] <= 1 <= ci[1]
        else:
            inc0, inc1 = fl["contains_0"], fl["contains_1"]
        r["pi"]["D_plat_sign"] = "negative (platform nights fell)" if bp < 0 else "POSITIVE: level DiD shows no platform drop; pi is not a pass-through share"
        # diagnostic only: level change implied by the log estimate at the treated pre-period mean
        r["diag_log_implied_level_plat"] = float(np.expm1(e16["log_plat"]["did"]["b"]) * res["treated_pre_mean_nights"]["plat_mean"])
        r["diag_log_implied_level_hotel"] = float(np.expm1(e16["log_I551"]["did"]["b"]) * res["treated_pre_mean_nights"]["I551_mean"])
        r["pi"]["verdict"] = ("cannot distinguish (CI contains 0 and 1)" if inc0 and inc1 else
                              "partial substitution (CI excludes 0 and 1)" if not inc0 and not inc1 else
                              "full transfer rejected (CI excludes 1, contains 0)" if inc0 else
                              "no-substitution rejected, full transfer not rejected (CI contains 1, excludes 0)")
        r["n_rows"], r["n_regions"] = int(len(Sx)), int(Sx.geo.nunique())
        r["regions_dropped"] = sorted(set(S.geo) - set(Sx.geo))
        res[key] = r
    import statsmodels.formula.api as smf
    BINS = {"m4": lambda r: r <= -4, "m3": lambda r: r == -3, "m2": lambda r: r == -2,
            "p0": lambda r: r == 0, "p1": lambda r: r == 1, "p2": lambda r: r == 2, "p3": lambda r: r >= 3}
    E = S.copy(); E["sg"], E["sy"] = E["stack"] + "_" + E.geo, E["stack"] + "_" + E.year.astype(str)
    for k_, f_ in BINS.items(): E[k_] = E.treated * f_(E.rel).astype(int)
    res["diag_level_event_time"] = {}
    for weighted in (False, True):
        d = E[E.w > 0].reset_index(drop=True) if weighted else E
        for y in Y_LEV:
            m = smf.wls(f"{y} ~ " + " + ".join(BINS) + " + C(sg) + C(sy)", d, weights=d.w if weighted else np.ones(len(d))).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d.geo)[0]})
            res["diag_level_event_time"][f"{'weighted' if weighted else 'unweighted'}_{y}"] = dict(
                event={k_: dict(b=float(m.params[k_]), se=float(m.bse[k_]), p=float(m.pvalues[k_])) for k_ in BINS},
                pre_joint_p=float(m.f_test("m4 = 0, m3 = 0, m2 = 0").pvalue))
    json.dump(res, open(O / "L1_level_passthrough.json", "w"), indent=1); print(json.dumps(res, indent=1))
