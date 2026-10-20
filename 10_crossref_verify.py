"""Verify DOIs against Crossref; write manuscript/src/refs_verified.json. usage: python -I 10_crossref_verify.py OUT_JSON DOI..."""
import sys, json, urllib.request, urllib.parse, datetime
out = sys.argv[1]; res = {}
try: res = json.load(open(out))
except Exception: pass
for doi in sys.argv[2:]:
    u = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    try:
        m = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "c10-verify (mailto:none)"}), timeout=60))["message"]
    except Exception as e:
        print("FAIL", doi, e); continue
    r = dict(doi=doi, authors=[(a.get("family", ""), a.get("given", "")) for a in m.get("author", [])],
             title=m["title"][0], journal=(m.get("container-title") or [""])[0], short=(m.get("short-container-title") or [""])[0],
             year=(m.get("issued", {}).get("date-parts") or [[None]])[0][0], volume=m.get("volume"), issue=m.get("issue"),
             page=m.get("page"), year_print=(m.get("published-print", {}).get("date-parts") or [[None]])[0][0], article_number=m.get("article-number"), type=m.get("type"),
             verified=f"Crossref {datetime.date.today()}")
    res[doi] = r; print(doi, "|", "; ".join(a[0] for a in r["authors"]), "|", r["title"], "|", r["journal"], r["year"], r["volume"], r["issue"], r["page"])
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
