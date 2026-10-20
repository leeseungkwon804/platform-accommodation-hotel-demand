"""c10dec6 decision 2: normalised-level stacked event study (run once, specified before the run on 2026-10-16).
Outcome y_n = nights / region's fixed 2018-19 mean hotel nights (I551); no time-varying denominator.
Sample, stacking, FE (region x stack, year x stack), clustering (region), 2020-21 exclusion identical to 16/18; unweighted only.
Regions without 2018-19 I551 (NO02, NO06-NO0A) drop out.
Stage "gate": pre-period joint p (bins <=-4, -3, -2; cluster Wald F as in 16) for plat_n, hotel_n, total_n.
  Gate fails if any p < 0.10 -> post coefficients are recorded but flagged unusable; no pi bootstrap.
Stage "full": DiD coefficients, CR1 SE, null-imposed wild bootstrap (9,999); if gate passed and D_plat < 0:
  pi = -D_hotel/D_plat with stratified region block bootstrap (9,999) and Fieller.
usage: python -I 19_normalised_level.py DERIVED_DIR OUT_DIR {gate|full} [B]"""
import sys, json, pathlib, numpy as np, pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
D, O = map(pathlib.Path, sys.argv[1:3])
STAGE = sys.argv[3]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 9999
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
DROPPED = sorted(S.loc[S.w == 0, "geo"].unique())
S = S[S.w > 0].reset_index(drop=True)
for c_ in ["plat", "I551"]: S[c_ + "_n"] = S[c_] / S.w      # fixed 2018-19 denominator
S["total_n"] = S.plat_n + S.I551_n
controls = [c_ for c_ in controls if c_ not in DROPPED]
Y_LEV, Y_LOG = ["plat_n", "I551_n", "total_n"], ["log_plat", "log_I551"]
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
        w = np.ones(len(idx))
        xt, Yt, _, _ = fwl(sg, sy, PA[idx], YA[idx], w)
        res.append(est(xt, Yt, w))
        out.append(np.concatenate(res))
    return np.array(out)


BINS = {"m4": lambda r: r <= -4, "m3": lambda r: r == -3, "m2": lambda r: r == -2,
        "p0": lambda r: r == 0, "p1": lambda r: r == 1, "p2": lambda r: r == 2, "p3": lambda r: r >= 3}

if __name__ == "__main__":
    import statsmodels.formula.api as smf
    G_ = O / "L1_normlevel_gate.json"
    if STAGE == "gate":
        E = S.copy(); E["sg"], E["sy"] = E["stack"] + "_" + E.geo, E["stack"] + "_" + E.year.astype(str)
        for k_, f_ in BINS.items(): E[k_] = E.treated * f_(E.rel).astype(int)
        g = dict(spec="normalised level: nights / fixed 2018-19 mean I551; unweighted; otherwise identical to 16",
                 dropped_regions=DROPPED, n_controls=len(controls), n_treated=len(treated), rows=len(S), G=int(S.geo.nunique()),
                 threshold=0.10, pre_joint_p={}, pre_coef={})
        for y in Y_LEV:
            m = smf.ols(f"{y} ~ " + " + ".join(BINS) + " + C(sg) + C(sy)", E).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(E.geo)[0]})
            g["pre_joint_p"][y] = float(m.f_test("m4 = 0, m3 = 0, m2 = 0").pvalue)
            g["pre_coef"][y] = {k_: dict(b=float(m.params[k_]), p=float(m.pvalues[k_])) for k_ in ["m4", "m3", "m2"]}
        g["gate_passed"] = bool(all(v >= 0.10 for v in g["pre_joint_p"].values()))
        json.dump(g, open(G_, "w"), indent=1); print(json.dumps(g, indent=1)); sys.exit()
    g = json.load(open(G_))
    res = dict(gate=g, n_rows=len(S), B=B)
    sg, sy, cl, w, xt, Yt, proj, k_fe, Sx = fit(S, Y_LEV, False)
    b, V, G = cluster_cov(xt, Yt, w, cl, k_fe); q = stats.t.ppf(0.975, G - 1)
    res["identity_max_abs_diff"] = float(abs(b[2] - b[0] - b[1]))
    res["identity_rowwise_max_abs_diff"] = float(np.abs(S.total_n - S.plat_n - S.I551_n).max())
    for i, y in enumerate(Y_LEV):
        y0t = proj(Sx[[y]].values.astype(float))[:, 0]
        wp, t0 = wild(xt, y0t, proj, w, cl, seed=10 + i); se = float(np.sqrt(V[i, i]))
        res[y] = dict(b=float(b[i]), se=se, p_t_G1=float(2 * stats.t.sf(abs(b[i] / se), G - 1)),
                      ci95=[float(b[i] - q * se), float(b[i] + q * se)], wild_p_H0_zero=wp)
    res["G"] = int(G)
    res["usable_for_claims"] = bool(g["gate_passed"])
    bh, bp = b[1], b[0]
    res["premise_D_plat_negative"] = bool(bp < 0)
    if g["gate_passed"] and bp < 0:
        with ProcessPoolExecutor(13) as ex:
            chunks = [(2000 + i, n) for i, n in enumerate([B // 100] * 100 + ([B % 100] if B % 100 else [])) if n]
            BS = np.vstack(list(ex.map(boot_chunk, chunks)))
        np.save(O / "L1_normlevel_boot.npy", BS)
        pib = -BS[:, 1] / BS[:, 0]; plat_ci = np.percentile(BS[:, 0], [2.5, 97.5])
        Vh, Vp, Vhp = V[1, 1], V[0, 0], V[0, 1]
        a = bp ** 2 - q ** 2 * Vp; bb = 2 * (bh * bp - q ** 2 * Vhp); c = bh ** 2 - q ** 2 * Vh; disc = bb ** 2 - 4 * a * c
        fq = lambda x: a * x ** 2 + bb * x + c
        fl = dict(contains_0=bool(fq(0.0) <= 0), contains_1=bool(fq(1.0) <= 0))
        fl.update(dict(type="bounded", ci95=sorted([float((-bb - np.sqrt(disc)) / (2 * a)), float((-bb + np.sqrt(disc)) / (2 * a))]))
                  if a > 0 and disc >= 0 else dict(type="unbounded" if a <= 0 else "no_real_roots"))
        boot_ok = bool(plat_ci[1] < 0 or plat_ci[0] > 0); pct = np.percentile(pib, [2.5, 97.5])
        if boot_ok: inc0, inc1 = pct[0] <= 0 <= pct[1], pct[0] <= 1 <= pct[1]
        else: inc0, inc1 = fl["contains_0"], fl["contains_1"]
        res["pi"] = dict(estimate=float(-bh / bp), boot_pct_ci95=[float(pct[0]), float(pct[1])], boot_D_plat_ci95=[float(x) for x in plat_ci],
                         fieller=fl, primary_ci="bootstrap percentile" if boot_ok else "Fieller",
                         verdict=("cannot distinguish (CI contains 0 and 1)" if inc0 and inc1 else
                                  "partial substitution (CI excludes 0 and 1)" if not inc0 and not inc1 else
                                  "full transfer rejected (CI excludes 1, contains 0)" if inc0 else
                                  "no-substitution rejected, full transfer not rejected"))
    else:
        res["pi"] = dict(not_computed="gate failed" if not g["gate_passed"] else "D_plat >= 0: pi is not a pass-through share")
    json.dump(res, open(O / "L1_normlevel.json", "w"), indent=1); print(json.dumps(res, indent=1))
