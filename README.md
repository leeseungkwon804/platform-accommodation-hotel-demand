# Platform accommodation and hotel demand in European regions

Analysis code for the manuscript *Platform Accommodation and Hotel Demand in European
Regions with Evidence from Regulation and Placebo Tests*.

The study asks whether restricting short-term rentals moves visitors into hotels, using
Eurostat statistics on platform-booked nights and hotel nights for 264 NUTS 2 regions over
2018–2025, and examines how much of a reported substitution effect the conventional
penetration measure produces by construction.

## Data

**No data are redistributed here.** Every input is an official statistic published by
Eurostat and was downloaded through the Eurostat dissemination API (SDMX 2.1, TSV) on
15 October 2026. The dataset codes used are:

| Code | Content |
|---|---|
| `tour_ce_omn12` | nights spent in collaborative-economy platforms |
| `tour_occ_nim` | nights spent at tourist accommodation establishments, monthly |
| `tour_occ_nin2` | nights spent, by NUTS 2 region |
| `tour_occ_nin2m` | nights spent, by NUTS 2 region, monthly |

Running `01_parse.py` against a local copy of those files reproduces the analysis inputs.
The processed results underlying every table and figure in the manuscript are provided as
File S1 in the article's Supplementary Materials.

## Scripts

Numbered in execution order. `01`–`21` produce the results; `30`–`31` draw the figures.

| | |
|---|---|
| `01`–`05` | parse Eurostat files, feasibility checks, build the regional panel |
| `06`–`08` | penetration threshold, forecasting comparison, causal forest |
| `09`–`12` | methods facts, Crossref verification, concentration, significance tables |
| `13`–`19` | event study, dynamics, placebo outcome, level and normalised pass-through |
| `20`–`21` | assemble the reported numbers, scale facts |
| `30`–`31` | figures and the penetration map |

Each script takes the project root as its first argument and reads from `data/`, writing to
`out/`. No number reported in the manuscript is typed by hand; each is computed by one of
these scripts from the downloaded Eurostat files.

## Reproducing

```bash
python3 01_parse.py  <project-root>
python3 02_feasibility.py <project-root>
# … in numbered order
```

Python 3.11 with pandas, statsmodels, scikit-learn, XGBoost, SHAP and EconML.

## Note on this copy

Two docstrings were edited before publication to remove the filenames of internal working
notes. No computation, parameter or output was changed.

## Licence

MIT. See `LICENSE`.
