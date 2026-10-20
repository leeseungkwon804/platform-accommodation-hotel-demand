"""Eurostat SDMX TSV -> long csv.gz. usage: python -I 01_parse.py RAW_DIR OUT_DIR"""
import sys, gzip, pandas as pd, pathlib, re
raw, out = map(pathlib.Path, sys.argv[1:3]); out.mkdir(exist_ok=True, parents=True)
for f in sorted(raw.glob("*.tsv.gz")):
    code = f.name.split(".")[0]
    df = pd.read_csv(f, sep="\t", dtype=str)
    k = df.columns[0]; dims = k.split("\\")[0].split(",")
    df[dims] = df[k].str.split(",", expand=True); df = df.drop(columns=k)
    df.columns = [c.strip() for c in df.columns]
    long = df.melt(id_vars=dims, var_name="time", value_name="raw")
    long["raw"] = long["raw"].str.strip()
    long["flag"] = long["raw"].str.extract(r"([a-z]+)$")[0]
    long["value"] = pd.to_numeric(long["raw"].str.extract(r"^([0-9.\-]+)")[0], errors="coerce")
    long = long.drop(columns="raw")
    long.to_csv(out / f"{code}.csv.gz", index=False)
    print(code, dims, "rows", len(long), "nonmissing", long.value.notna().sum())
    for d in dims:
        if d not in ("geo",): print("   ", d, sorted(long[d].unique())[:20])
    g = long.loc[long.value.notna(), "geo"].str.len().value_counts().sort_index().to_dict()
    print("    geo code length -> n geos:", {L: long.loc[long.value.notna() & (long.geo.str.len()==L), "geo"].nunique() for L in g})
