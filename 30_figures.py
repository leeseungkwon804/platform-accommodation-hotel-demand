"""Figures 1-4 and Supplementary Figures S1-S2 (A4 180 mm native width, 600 dpi, Arial).
Every plotted number is read from out/*.json|csv or recomputed from data/derived/L1_panel.csv.gz; plotted values and
derived numbers are written to figs/figure_facts.json. Panel letters are placed in figure coordinates on a fixed grid.
usage: python -I 30_figures.py PROJECT_DIR"""
import sys, json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
fm._load_fontmanager(try_read_cache=False)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm, Normalize

R = pathlib.Path(sys.argv[1]); O = R / "out"; FG = R / "figs"; FG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 600, "savefig.dpi": 600, "legend.frameon": False})
INK, GREY, LGREY, RULE = "#22262b", "#5a6069", "#8c929b", "#d7dade"
MAIN, PLAC, COUP, FREE = "#2166ac", "#b2182b", "#b2182b", "#2166ac"
W = 180.0  # mm
MM = 1 / 25.4
FACTS, LETTERS = {}, {}
def fp(p):   # p-value text
    return "< 0.001" if p < 0.001 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}")
def fs(x, d=4):  # signed number with typographic minus
    return f"{x:+.{d}f}".replace("-", "\u2212")

def new_fig(h):
    assert h <= 230
    return plt.figure(figsize=(W * MM, h * MM)), h

def ax_mm(fig, H, x, y_top, w, h):
    """axes from mm, y measured from the top of the figure"""
    return fig.add_axes([x / W, (H - y_top - h) / H, w / W, h / H])

def letter(fig, H, name, ch, x, y_top):
    fig.text(x / W, (H - y_top) / H, ch, fontsize=10, fontweight="bold", color=INK, ha="left", va="top")
    LETTERS.setdefault(name, []).append(dict(letter=ch, x_mm=x, y_top_mm=y_top))

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FG / f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)

# grid: 12 columns. left margin 14 mm, right 4 mm, gutter 14 mm between half-width panels
LM, RM, GUT = 14.0, 4.0, 14.0
HALF = (W - LM - RM - GUT) / 2
COLX = [LM, LM + HALF + GUT]
LETX = [2.0, LM + HALF + GUT - 12.0]

# ------------------------------------------------------------------ data
P = pd.read_csv(R / "data/derived/L1_panel.csv.gz")
ev = json.load(open(O / "L1_event_study.json"))
EVR = [e["nuts2"] for e in ev["events"]]
COVS = ["beds_per_1000", "hotel_bed_share", "log_gdp_pps_pc", "log_density", "log_pop", "city_night_share", "coastal_night_share", "season_gini_2022_23"]
S0 = P[~P.year.isin([2020, 2021])].dropna(subset=COVS + ["log_I551", "pen"])
S0 = S0[S0.groupby("geo").year.transform("count") >= 2].copy()

def demean(df, cols, it=500):
    X = df[cols].astype(float).copy()
    for _ in range(it):
        X = X - X.groupby(df.geo.values).transform("mean"); X = X - X.groupby(df.year.values).transform("mean")
    return X

# =================================================================== Figure 1
F = json.load(open(O / "facts_methods.json"))
fig, H = new_fig(224)
ax = ax_mm(fig, H, LM, 6, W - LM - RM, 34); ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
boxes = [
    ("L2  country × month", [f"{F['L2']['countries']} countries, {F['L2']['first']} to {F['L2']['last']}",
                             f"{F['L2']['cells']:,} cells ({F['L2']['cells_excl_covid']:,} outside 2020–2021)".replace(",", ",") ,
                             "hotels (I551) and platform nights"], "Forecasting experiment"),
    ("L1  NUTS 2 region × year", [f"{F['L1']['regions']} regions, 2018 to 2025",
                                  f"{F['L1']['cells']:,} cells; main sample {F['L1_covariates']['rows_complete_excl_covid']:,}",
                                  f"region-years in {F['L1_covariates']['regions_complete']} regions"], "Penetration measures, regulation event study"),
    ("L3  NUTS 2 region × month", [f"{F['L3']['cells']:,} cells",
                                   "all accommodation (I551–I553);", "no hotel split published"], "Seasonal concentration (Limitations)"),
]
bw, gap = 30.0, 5.0
for i, (head, lines, use) in enumerate(boxes):
    x0 = i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x0, 12), bw, 22, boxstyle="round,pad=0,rounding_size=1.2", fc="#f3f5f8", ec=GREY, lw=0.6))
    ax.text(x0 + 1.5, 32.0, head, fontsize=7.5, fontweight="bold", va="top", color=INK)
    for j, t in enumerate(lines):
        ax.text(x0 + 1.5, 26.5 - j * 4.6, t, fontsize=6.6, va="top", color=INK)
    ax.text(x0 + bw / 2, 9.5, "Used for", fontsize=6.4, ha="center", va="top", color=GREY)
    ax.text(x0 + bw / 2, 5.0, use, fontsize=6.6, ha="center", va="top", color=INK, wrap=True)
letter(fig, H, "Figure1", "a", LETX[0], 4)

# (b) data map: region-mean fixed-denominator penetration, regulated regions outlined
G2 = json.load(open(R / "data/raw/gisco/NUTS_RG_20M_2024_3035_LEVL_2.geojson"))
G0 = json.load(open(R / "data/raw/gisco/NUTS_RG_20M_2024_3035_LEVL_0.geojson"))
def _polys(geom):
    cs = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [np.array(c[0]) for c in cs]
rm_map = S0.groupby("geo").pen_fixed.mean()
axm = ax_mm(fig, H, 25, 48, 112, 112); axm.set_aspect("equal"); axm.axis("off")
axm.set_xlim(2.5e6, 6.6e6); axm.set_ylim(1.35e6, 5.45e6)
_vmax = float(np.nanpercentile(rm_map.values, 99))
_norm = Normalize(vmin=0, vmax=_vmax); _cmap = plt.get_cmap("Blues")
_out, _in, _v, _tr = [], [], [], []
for f_ in G2["features"]:
    _g = f_["properties"]["NUTS_ID"]
    for _pts in _polys(f_["geometry"]):
        if _g in rm_map.index:
            _in.append(Polygon(_pts)); _v.append(rm_map.loc[_g])
            if _g in EVR: _tr.append(Polygon(_pts))
        else:
            _out.append(Polygon(_pts))
axm.add_collection(PatchCollection(_out, fc="#eceef1", ec="white", lw=0.15))
_coll = PatchCollection(_in, cmap=_cmap, norm=_norm, ec="white", lw=0.15)
_coll.set_array(np.array(_v)); axm.add_collection(_coll)
axm.add_collection(PatchCollection(_tr, fc="none", ec=PLAC, lw=0.9))
for f_ in G0["features"]:
    for _pts in _polys(f_["geometry"]):
        axm.add_patch(Polygon(_pts, fc="none", ec=GREY, lw=0.3))
_cax = ax_mm(fig, H, 142, 68, 3.5, 72)
_cb = fig.colorbar(_coll, cax=_cax, extend="max"); _cb.outline.set_linewidth(0.4)
_cb.set_label("Region-mean fixed-denominator penetration", fontsize=6.4)
axm.plot([], [], marker="s", ls="none", mfc="none", mec=PLAC, mew=0.9, ms=6,
         label=f"short-term rental regulation (n = {len(EVR)})")
axm.legend(loc="lower left", fontsize=6.4, handletextpad=0.6, borderpad=0.3)
letter(fig, H, "Figure1", "b", LETX[0], 44)
FACTS["Figure1_map"] = dict(regions_mapped=int(rm_map.size), treated_outlined=len(EVR),
                            colour_vmax_p99=_vmax)

# (c) region-mean fixed-denominator penetration, regulated regions marked
lx = P[P.geo.isin(["PT1A", "PT1B"])]
base = lx[lx.year.isin([2018, 2019])].groupby("geo")[["plat", "I551"]].mean().sum()
pt17 = lx[~lx.year.isin([2020, 2021])].groupby("year").plat.sum() / (base.plat + base.I551)
rm = S0.groupby("geo").pen_fixed.mean()
ev_vals = {g: (float(pt17.mean()) if g == "PT17" else float(P[(P.geo == g) & ~P.year.isin([2020, 2021])].pen_fixed.mean())) for g in EVR}
ax = ax_mm(fig, H, COLX[0], 170, HALF, 44)
bins = np.arange(0, 1.05, 0.04)
ax.hist(rm.values, bins=bins, color=MAIN, alpha=0.85, ec="white", lw=0.4)
ymax = ax.get_ylim()[1] * 1.18; ax.set_ylim(0, ymax)
ax.plot(list(ev_vals.values()), [ymax * 0.93] * len(ev_vals), "v", ms=3.6, color=PLAC, mec="white", mew=0.3, clip_on=False)
ax.text(0.98, 0.83, f"▼ regulated regions (n = {len(ev_vals)})", transform=ax.transAxes, ha="right", fontsize=6.4, color=PLAC)
ax.set_xlabel("Fixed-denominator penetration, region mean"); ax.set_ylabel("Regions")
ax.set_xlim(0, 1.04)
letter(fig, H, "Figure1", "c", LETX[0], 166)
# (d) within-region deviation (two-way demeaned) for both measures
ax = ax_mm(fig, H, COLX[1], 170, HALF, 44)
Z = demean(S0, ["pen", "pen_fixed"])
bins = np.arange(-0.3, 0.3001, 0.015)
ax.hist(Z.pen_fixed.values, bins=bins, color=MAIN, alpha=0.8, ec="white", lw=0.3, label="fixed denominator")
ax.hist(Z.pen.values, bins=bins, histtype="step", color=INK, lw=0.8, label="conventional share")
ax.set_xlabel("Within-region deviation (region and year effects removed)"); ax.set_ylabel("Region-years")
ax.legend(loc="upper right", fontsize=6.4, handlelength=1.2)
letter(fig, H, "Figure1", "d", LETX[1], 166)
tot = lambda v: S0.groupby("geo")[v].transform("mean")
FACTS["Figure1"] = dict(region_means_n=int(rm.size), regulated_region_mean_pen_fixed=ev_vals,
                        pen_fixed_region_mean_p10_p50_p90=[float(x) for x in rm.quantile([.1, .5, .9])],
                        sd_within_twoway=dict(pen=float(Z.pen.std()), pen_fixed=float(Z.pen_fixed.std())),
                        sd_between_region_means=dict(pen=float(S0.groupby("geo").pen.mean().std()), pen_fixed=float(rm.std())),
                        n_region_years=int(len(S0)))
save(fig, "Figure1")

# =================================================================== Figure 2 (regulation event study)
E = json.load(open(O / "L1_event_dynamics.json"))
order = ["<=-4", "-3", "-2", "-1", "0", "1", "2", ">=3"]; xt = ["≤−4", "−3", "−2", "−1", "0", "1", "2", "≥3"]
fig, H = new_fig(78)
FACTS["Figure2"] = {}
for k, (y, ttl, col) in enumerate([("log_plat", "Platform nights (log)", PLAC), ("log_I551", "Hotel nights, I551 (log)", MAIN)]):
    ax = ax_mm(fig, H, COLX[k], 10, HALF, 54)
    em = {e["rel"]: e for e in E[y]["event"]}
    xs, bs, lo, hi = [], [], [], []
    for i, r in enumerate(order):
        if r == "-1": xs.append(i); bs.append(0.0); lo.append(0.0); hi.append(0.0); continue
        xs.append(i); bs.append(em[r]["b"]); lo.append(em[r]["lo"]); hi.append(em[r]["hi"])
    if y == "log_I551":
        mde = E[y]["did"]["mde_log"]; ftb = E["full_transfer_benchmark"]["test_b0_log"]
        ax.axhspan(-mde, mde, color=MAIN, alpha=0.10, lw=0)
        ax.axhline(ftb, color=GREY, lw=0.7, ls=(0, (3, 2)))
        ax.text(0.0, ftb, "full transfer of lost platform nights", ha="left", va="bottom", fontsize=6.0, color=GREY)
    ax.axhline(0, color=INK, lw=0.5); ax.axvline(3.5, ymin=0.25, color=LGREY, lw=0.6, ls=":")
    ax.vlines(xs, lo, hi, color=col, lw=1.0)
    ax.plot(xs, bs, "o", color=col, ms=3.6, mec="white", mew=0.3)
    ax.plot([3], [0], "o", mfc="white", mec=col, ms=3.6, mew=0.8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(xt)
    ax.set_xlabel("Years relative to regulation"); ax.set_title(ttl, fontsize=7.2, loc="left")
    if k == 0: ax.set_ylabel("Coefficient vs. year −1 (log points)")
    did = E[y]["did"]
    ax.text(0.02, 0.03, f"pooled post: {fs(did['b'])} (SE {did['se']:.4f}, p {fp(did['p'])})\npre-period joint p {fp(E[y]['pre_joint_p'])}"
            + (f"\nshaded: minimum detectable effect ±{E[y]['did']['mde_log']:.3f} (80% power)" if y == "log_I551" else ""),
            transform=ax.transAxes, fontsize=6.0, va="bottom", color=INK)
    letter(fig, H, "Figure2", "ab"[k], LETX[k], 4)
    FACTS["Figure2"][y] = dict(event=E[y]["event"], did=did, pre_joint_p=E[y]["pre_joint_p"])
ylo = min(a.get_ylim()[0] for a in fig.axes); yhi = max(a.get_ylim()[1] for a in fig.axes)
for a in fig.axes: a.set_ylim(ylo, yhi)
FACTS["Figure2"]["cells_by_bin_treated"] = E["cells_by_bin"]
save(fig, "Figure2")

# =================================================================== Figure 3
C2 = json.load(open(O / "C2_claim1_defence.json"))
S0["log_plat"] = np.log(S0.plat)
S0["log_bedrm_tm4"] = np.log(S0.I551_bedrooms_tm4)
def sd_within(var, outcome="log_I551"):
    d = S0.dropna(subset=[var, outcome]); d = d[d.groupby("geo").year.transform("count") >= 2]
    return float(demean(d, [var])[var].std())
rows_l = [("Conventional share", C2["claim1_pen (conventional share, region+year FE)"], "pen", "pen", "log_I551"),
          ("Denominator I551–I553", C2["denominator I551-I553: pen_all"], "pen_all", "pen_all", "log_I551"),
          ("Within placebo: bedrooms t−4", C2["within placebo: log I551 bedrooms t-4 ~ pen"], "pen", "pen", "log_bedrm_tm4")]
rows_r = [("Fixed denominator", C2["MECH-FREE: pen_fixed (2018-19 denominator)"], "pen_fixed", "pen_fixed", "log_I551"),
          ("Log platform nights", C2["MECH-FREE: log platform nights (elasticity)"], "log_plat", "log_plat", "log_I551"),
          ("Platform nights per resident", C2["MECH-FREE: platform nights per resident"], "plat_per_cap", "plat_per_cap", "log_I551"),
          ("Lead: fixed denominator t+1", C2["MECH-FREE lead: pen_fixed_t + pen_fixed_t+1"], "pen_fixed_lead1", "pen_fixed", "log_I551"),
          ("Within placebo: bedrooms t−4", C2["MECH-FREE within placebo: bedrooms t-4 ~ pen_fixed"], "pen_fixed", "pen_fixed", "log_bedrm_tm4")]
perm = C2["permutation (platform shuffled within region; conventional share)"]
fig, H = new_fig(76)
FACTS["Figure3"] = dict(left=[], right=[])
def forest(ax, rows, col, side, extra=None):
    ys = []
    for i, (lab, r, coefk, sdk, outc) in enumerate(rows):
        sd = sd_within(sdk, outc); b = r["coef"][coefk] * sd; se = r["se"][coefk] * sd
        y = len(rows) - i + (1 if extra else 0)
        ax.plot([b - 1.96 * se, b + 1.96 * se], [y, y], color=col, lw=1.0); ax.plot(b, y, "o", color=col, ms=3.6)
        rawb, p = r["coef"][coefk], r["p"][coefk]
        ax.text(0.985, y, f"β {fs(rawb)}, p {fp(p)}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.9, color=GREY)
        ys.append((y, lab))
        FACTS["Figure3"][side].append(dict(label=lab, raw_beta=rawb, raw_se=r["se"][coefk], p=p, sd_within=sd, std_beta=b, std_ci=[b - 1.96 * se, b + 1.96 * se]))
    if extra:
        sd = sd_within("pen"); m, lo, hi = perm["mean_beta"] * sd, perm["p2_5"] * sd, perm["p97_5"] * sd
        ax.plot([lo, hi], [1, 1], color=LGREY, lw=2.4, solid_capstyle="butt"); ax.plot(m, 1, "D", color=GREY, ms=3.2)
        ax.text(0.985, 1, f"β {fs(perm['mean_beta'])} [{fs(perm['p2_5'], 3)}, {fs(perm['p97_5'], 3)}]", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.9, color=GREY)
        ys.append((1, "Permutation null (no relation)"))
        FACTS["Figure3"][side].append(dict(label="permutation null", raw_mean=perm["mean_beta"], raw_ci=[perm["p2_5"], perm["p97_5"]], sd_within=sd, std_mean=m, std_ci=[lo, hi]))
    ax.set_yticks([y for y, _ in ys]); ax.set_yticklabels([l for _, l in ys], fontsize=6.4)
    ax.axvline(0, color=INK, lw=0.6); ax.set_ylim(0.4, max(y for y, _ in ys) + 0.6)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax1 = ax_mm(fig, H, 42, 12, 46, 52); forest(ax1, rows_l, COUP, "left", extra=True)
ax2 = ax_mm(fig, H, 130, 12, 46, 52); forest(ax2, rows_r, FREE, "right")
xl = (min(ax1.get_xlim()[0], ax2.get_xlim()[0]), max(ax1.get_xlim()[1], ax2.get_xlim()[1]))
for a in (ax1, ax2):
    a.set_xlim(xl[0], xl[1] + (xl[1] - xl[0]) * 0.75)
    tk = [t for t in np.round(np.arange(-0.03, 0.0301, 0.01), 2) if xl[0] <= t <= xl[1] + 1e-9]
    a.set_xticks(tk); a.set_xticklabels([("0" if t == 0 else f"{t:.2f}".replace("-", "\u2212")) for t in tk], fontsize=6.4)
    a.set_xlabel("Change in log hotel outcome per\none within-region SD of the measure")
ax1.set_title("Hotel nights inside the denominator", fontsize=7.2, loc="left", color=COUP)
ax2.set_title("Hotel nights outside the denominator", fontsize=7.2, loc="left", color=FREE)
letter(fig, H, "Figure3", "a", 2.0, 5); letter(fig, H, "Figure3", "b", 92.0, 5)
save(fig, "Figure3")

# =================================================================== Figure 4 (partial dependence vs placebo)
TH = {r["tag"].split()[0]: r for r in json.load(open(O / "L1_threshold.json"))}
TF = {r["tag"].split()[0]: r for r in json.load(open(O / "L1_threshold_fixed.json"))}
fig, H = new_fig(80)
panels = [("a", TH["MAIN"], TH["PLACEBO"], "Conventional share  platform / (platform + hotel)"),
          ("b", TF["MAIN-F"], TF["PLACEBO-F"], "Fixed denominator  platform / 2018–2019 total")]
FACTS["Figure4"] = {}
for k, (ch, m, pl, xl) in enumerate(panels):
    ax = ax_mm(fig, H, COLX[k], 10, HALF, 54)
    for r, col, lab in [(m, MAIN, "hotel nights (main)"), (pl, PLAC, "hotel bedrooms four years earlier (placebo)")]:
        g = np.array(r["pdp_grid"]); y = np.array(r["pdp"])
        ax.fill_between(g, r["boot"]["pdp_band_2_5"], r["boot"]["pdp_band_97_5"], color=col, alpha=0.14, lw=0)
        ax.plot(g, y, color=col, lw=1.2, label=lab)
        tp = r["gbm_turning"].get("tp")
        if r["gbm_turning"].get("interior"):
            ax.axvline(tp, color=col, lw=0.7, ls=(0, (3, 2)))
    ax.axhline(0, color=LGREY, lw=0.5)
    ax.set_xlabel(xl); ax.set_ylabel("Centred partial dependence (log points)") if k == 0 else None
    txt = []
    for r, lab in [(m, "main"), (pl, "placebo")]:
        tp = r["gbm_turning"].get("tp"); it = r["gbm_turning"].get("interior")
        txt.append(f"{lab}: turning point {tp:.3f}" if it else f"{lab}: no interior turning point")
        txt[-1] += f"; bootstrap sign change {r['boot']['share_interior_signchange'] * 100:.1f}%"
    ax.text(0.02, 0.03, "\n".join(txt), transform=ax.transAxes, fontsize=6.0, color=INK, va="bottom")
    if k == 0: ax.legend(loc="upper left", fontsize=6.2, handlelength=1.4)
    letter(fig, H, "Figure4", ch, LETX[k], 4)
    FACTS["Figure4"][ch] = dict(main=m["tag"], placebo=pl["tag"], main_tp=m["gbm_turning"].get("tp"), placebo_tp=pl["gbm_turning"].get("tp"),
                                main_interior=m["gbm_turning"].get("interior"), placebo_interior=pl["gbm_turning"].get("interior"),
                                main_boot_sign=m["boot"]["share_interior_signchange"], placebo_boot_sign=pl["boot"]["share_interior_signchange"])
for a in fig.axes: a.set_ylim(-0.062, 0.05)
save(fig, "Figure4")

# =================================================================== Figure S1: CATE map (supplementary only)
cate = pd.read_csv(O / "L1_cate_by_region_pen_fixed.csv").set_index("geo")
cfp = json.load(open(O / "L1_causal_forest_pen_fixed_placebo.json"))
G2 = json.load(open(R / "data/raw/gisco/NUTS_RG_20M_2024_3035_LEVL_2.geojson"))
G0 = json.load(open(R / "data/raw/gisco/NUTS_RG_20M_2024_3035_LEVL_0.geojson"))
def polys(geom):
    cs = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [np.array(c[0]) for c in cs]
fig, H = new_fig(166)
ax = ax_mm(fig, H, 4, 4, 150, 150); ax.set_aspect("equal"); ax.axis("off")
XL, YL = (2.5e6, 6.6e6), (1.35e6, 5.45e6)
vmax = float(np.nanmax(np.abs(cate.cate10.values)))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax); cmap = plt.get_cmap("RdBu")
pg, pc, gg = [], [], []
for f in G2["features"]:
    gid = f["properties"]["NUTS_ID"]
    for pts in polys(f["geometry"]):
        if gid in cate.index:
            pc.append(Polygon(pts)); gg.append(cate.loc[gid, "cate10"])
        else:
            pg.append(Polygon(pts))
ax.add_collection(PatchCollection(pg, fc="#eceef1", ec="white", lw=0.15))
coll = PatchCollection(pc, cmap=cmap, norm=norm, ec="white", lw=0.15); coll.set_array(np.array(gg)); ax.add_collection(coll)
for f in G0["features"]:
    for pts in polys(f["geometry"]):
        ax.add_patch(Polygon(pts, fc="none", ec=GREY, lw=0.3))
ax.set_xlim(*XL); ax.set_ylim(*YL)
cax = ax_mm(fig, H, 158, 30, 3.5, 90)
cb = fig.colorbar(coll, cax=cax); cb.outline.set_linewidth(0.4)
cb.set_label("Region-level CATE, log hotel nights per +10 pp fixed-denominator penetration", fontsize=6.4)
fig.text(4 / W, (H - 152) / H,
         "Based on between-region variation; not a causal estimate for individual regions. Region-level confidence intervals are\n"
         f"under-estimated: with a placebo outcome (hotel bedrooms four years earlier) {cfp['cate_region']['n_ci_below_0']} of {cfp['regions']} regions have an interval below zero.",
         fontsize=6.4, color=INK, va="top")
FACTS["FigureS1"] = dict(regions_mapped=int(len(gg) and cate.index.isin([f['properties']['NUTS_ID'] for f in G2['features']]).sum()),
                         vmax=vmax, placebo_ci_below0=cfp["cate_region"]["n_ci_below_0"], placebo_regions=cfp["regions"])
save(fig, "FigureS1")

# =================================================================== Figure S2: CATE by penetration quintile, main vs placebo
cfm = json.load(open(O / "L1_causal_forest_pen_fixed.json"))
fig, H = new_fig(70)
ax = ax_mm(fig, H, LM + 4, 8, 100, 50)
qm, qp = cfm["by_pen_quintile"], cfp["by_pen_quintile"]
x = np.arange(5)
ax.bar(x - 0.19, [q["cate10_mean"] for q in qm], 0.36, color=MAIN, label="hotel nights (main)")
ax.bar(x + 0.19, [q["cate10_mean"] for q in qp], 0.36, color=PLAC, label="hotel bedrooms four years earlier (placebo)")
ax.axhline(0, color=INK, lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([f"{q['q']}\n{q['pen_min']:.2f}–{q['pen_max']:.2f}" for q in qm])
ax.set_xlabel("Quintile of region-mean fixed-denominator penetration (main-sample range)")
ax.set_ylabel("Mean region-level CATE\nper +10 pp")
ax.legend(loc="upper right", fontsize=6.4)
ax.text(1.03, 0.98, f"BLP penetration-level term\nmain {fs(cfm['blp']['coef']['pen_level'])} (p {fp(cfm['blp']['p']['pen_level'])})\n"
        f"placebo {fs(cfp['blp']['coef']['pen_level'])} (p {fp(cfp['blp']['p']['pen_level'])})\n\nATE per +10 pp\n"
        f"main {fs(cfm['ate_per10pp']['est'])} (p {fp(cfm['ate_per10pp']['p'])})",
        transform=ax.transAxes, fontsize=6.4, va="top", color=INK)
FACTS["FigureS2"] = dict(main=qm, placebo=qp)
save(fig, "FigureS2")

json.dump(dict(facts=FACTS, letters=LETTERS), open(FG / "figure_facts.json", "w"), indent=1, default=float)
for k, v in LETTERS.items():
    print(k, [(l["letter"], l["x_mm"], l["y_top_mm"]) for l in v])
