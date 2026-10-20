"""Data map: region-mean fixed-denominator platform penetration across NUTS 2 regions
of the main sample, with the 12 regulated regions outlined. This maps a MEASURED
quantity, not an estimate, so it carries none of the identification caveats of Figure S1.
House style copied from 30_figures.py. usage: python -I 31_map_penetration.py PROJECT_DIR"""
import sys, json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
fm._load_fontmanager(try_read_cache=False)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

R = pathlib.Path(sys.argv[1]); O = R / "out"; FG = R / "figs"
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.6, "figure.dpi": 600, "savefig.dpi": 600, "legend.frameon": False})
INK, GREY = "#22262b", "#5a6069"
MARK = "#b2182b"
W, MM = 180.0, 1 / 25.4

def new_fig(h):
    return plt.figure(figsize=(W * MM, h * MM)), h

def ax_mm(fig, H, x, y_top, w, h):
    return fig.add_axes([x / W, (H - y_top - h) / H, w / W, h / H])

def polys(geom):
    cs = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [np.array(c[0]) for c in cs]

# ---- data: same main-sample definition as 30_figures.py
P = pd.read_csv(R / "data/derived/L1_panel.csv.gz")
ev = json.load(open(O / "L1_event_study.json"))
EVR = [e["nuts2"] for e in ev["events"]]
COVS = ["beds_per_1000", "hotel_bed_share", "log_gdp_pps_pc", "log_density", "log_pop",
        "city_night_share", "coastal_night_share", "season_gini_2022_23"]
S0 = P[~P.year.isin([2020, 2021])].dropna(subset=COVS + ["log_I551", "pen"])
rm = S0.groupby("geo").pen_fixed.mean()

G2 = json.load(open(R / "data/raw/gisco/NUTS_RG_20M_2024_3035_LEVL_2.geojson"))
G0 = json.load(open(R / "data/raw/gisco/NUTS_RG_20M_2024_3035_LEVL_0.geojson"))

fig, H = new_fig(166)
ax = ax_mm(fig, H, 4, 4, 150, 150); ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(2.5e6, 6.6e6); ax.set_ylim(1.35e6, 5.45e6)

# sequential single hue: penetration is a magnitude, not a polarity
norm = Normalize(vmin=0, vmax=float(np.nanpercentile(rm.values, 99)))
cmap = plt.get_cmap("Blues")

out_of_sample, in_sample, vals, treated = [], [], [], []
for f in G2["features"]:
    gid = f["properties"]["NUTS_ID"]
    for pts in polys(f["geometry"]):
        if gid in rm.index:
            in_sample.append(Polygon(pts)); vals.append(rm.loc[gid])
            if gid in EVR:
                treated.append(Polygon(pts))
        else:
            out_of_sample.append(Polygon(pts))
ax.add_collection(PatchCollection(out_of_sample, fc="#eceef1", ec="white", lw=0.15))
coll = PatchCollection(in_sample, cmap=cmap, norm=norm, ec="white", lw=0.15)
coll.set_array(np.array(vals)); ax.add_collection(coll)
ax.add_collection(PatchCollection(treated, fc="none", ec=MARK, lw=0.9))
for f in G0["features"]:
    for pts in polys(f["geometry"]):
        ax.add_patch(Polygon(pts, fc="none", ec=GREY, lw=0.3))

cax = ax_mm(fig, H, 158, 30, 3.5, 90)
cb = fig.colorbar(coll, cax=cax, extend="max"); cb.outline.set_linewidth(0.4)
cb.set_label("Region-mean fixed-denominator platform penetration", fontsize=6.4)
ax.plot([], [], marker="s", ls="none", mfc="none", mec=MARK, mew=0.9, ms=6,
        label=f"short-term rental regulation (n = {len(EVR)})")
ax.legend(loc="lower left", fontsize=6.6, handletextpad=0.6, borderpad=0.4)
fig.text(4 / W, (H - 152) / H,
         f"{rm.size} NUTS 2 regions of the main sample; grey, not in sample. Penetration is platform nights divided by the region's "
         "mean platform\nplus hotel nights in 2018–2019, averaged over 2018–2025 excluding 2020–2021. Measured values, not estimates. "
         "Boundaries: Eurostat GISCO, NUTS 2024.",
         fontsize=6.4, color=INK, va="top")
for ext in ("png", "pdf"):
    fig.savefig(FG / f"Figure1d_map.{ext}", dpi=600, facecolor="white")
plt.close(fig)
facts = dict(regions_mapped=int(rm.size), treated_outlined=len(EVR),
             pen_p10_p50_p90=[float(x) for x in rm.quantile([.1, .5, .9])],
             vmax_p99=float(np.nanpercentile(rm.values, 99)))
json.dump(facts, open(FG / "Figure1d_facts.json", "w"), indent=1)
print(json.dumps(facts, indent=1))
