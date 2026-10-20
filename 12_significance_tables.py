"""Generate the numeric significance tables from out/*, printed to stdout. No number is typed by hand.
usage: python -I 12_significance_tables.py PROJECT_DIR > tables.md"""
import sys, json, pathlib
import pandas as pd
R = pathlib.Path(sys.argv[1]); O = R / "out"
f3 = lambda x: "–" if x is None else f"{x:.3f}"
f4 = lambda x: "–" if x is None else f"{x:.4f}"
pc = lambda x: "–" if x is None else f"{100 * x:.1f}%"
out = []
w = out.append

def threshold_table(path, title):
    runs = json.load(open(path))
    w(f"\n### {title}\n")
    w("| 실행 | n | 지역 | 침투율 격자 p5–p95 | GBM 전환점 (점추정) | 부트스트랩: 내부 전환점 비율 | 전환점 95% 구간 [2.5, 중앙, 97.5] | PDP 폭 | 2차항 b2 | 2차항 전환점 | 2차항: b2<0 & 범위내 비율 | 2차항 전환점 95% 구간 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        t = r["gbm_turning"]; b = r["boot"]
        tp = f3(t["tp"]) + (" (부호전환)" if t.get("interior") else " (없음)") if t["tp"] == t["tp"] else "없음(경계)"
        ci = "[" + ", ".join(f3(x) for x in b["tp_ci_interior"]) + "]" if b["tp_ci_interior"] else "–(내부 전환점 10회 이하)"
        qci = "[" + ", ".join(f3(x) for x in b["quad_tp_ci"]) + "]" if b["quad_tp_ci"] else "–"
        w(f"| {r['tag']} | {r['n']} | {r['regions']} | {f3(r['grid_p5_p95'][0])}–{f3(r['grid_p5_p95'][1])} | {tp} | "
          f"{pc(b['share_interior_signchange'])} | {ci} | {f4(r['pdp_range'])} | {r['quad']['b2']:.4f} | {f3(r['quad']['tp'])} | "
          f"{pc(b['quad_share_tp_in_range'])} | {qci} |")
    w("\n지역 그룹 5-fold CV (잔차화된 log I551):\n")
    w("| 실행 | GBM R² | 2차항 R² | 선형 R² | GBM RMSE | 2차항 RMSE | GBM 설정 (깊이/트리) |")
    w("|---|---|---|---|---|---|---|")
    for r in runs:
        c = r["cv"]
        w(f"| {r['tag']} | {c['gbm']['r2']:.4f} | {c['quad']['r2']:.4f} | {c['linear']['r2']:.4f} | {c['gbm']['rmse']:.4f} | {c['quad']['rmse']:.4f} | {r['gbm_depth']}/{r['gbm_trees']} |")
    w("\nSHAP 평균 |기여| (침투율 수준 T vs 지역내 편차 T_dev, 상위 공변량):\n")
    w("| 실행 | T (수준) | T_dev (지역내) | 상위 3개 |")
    w("|---|---|---|---|")
    for r in runs:
        s = r["shap_mean_abs"]; top = sorted(((k, v) for k, v in s.items() if k not in ("T", "T_dev")), key=lambda kv: -kv[1])[:3]
        w(f"| {r['tag']} | {f4(s.get('T'))} | {f4(s.get('T_dev'))} | " + ", ".join(f"{k} {v:.4f}" for k, v in top) + " |")
    ow = [r for r in runs if r.get("ols_within")]
    for r in ow:
        o = r["ols_within"]
        w(f"\n{r['tag']} — 지역내 편차의 선형 기울기(지역 군집 SE): β = {o['beta']:.4f} (SE {o['se']:.4f}, p = {o['p']:.3f}); 지역내 편차 SD {r['sd_T_dev']:.4f}")

for p, t in [(O / "L1_threshold_fixed.json", "① 임계값 — 고정분모 침투율 (기계적 결합 없음, 제안 주분석)"),
             (O / "L1_threshold.json", "① 임계값 — 관례 침투율 plat/(plat+I551) (분모에 결과변수 포함)")]:
    if p.exists(): threshold_table(p, t)

w("\n### ② 예측가치 (L2, 국가×월, 목표월 2023-01 이후, 로그 숙박일수 오차)\n")
for suf, lab in [("", "같은 달 정보(공표지연 무시)"), ("_lag3", "실시간: 플랫폼 피처 3개월 추가 지연(Eurostat 공표 지연)")]:
    p = O / f"L2_forecast_summary{suf}.csv"
    if not p.exists(): continue
    S = pd.read_csv(p)
    w(f"\n**{lab}**\n")
    w("| h | 모형 | n (국가×목표월) | 국가 | RMSE 없음 | RMSE 있음 | RMSE 감소율 | MAE 없음 | MAE 있음 | MAE 감소율 | DM(MSE) p | DM(MAE) p | 계절naive 대비 RMSE 개선(있음) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in S.iterrows():
        if r.model == "SNaive":
            w(f"| {r.h} | 계절 naive (기준) | {r.n} | {r.countries} | {r.rmse:.4f} | – | – | {r.mae:.4f} | – | – | – | – | – |"); continue
        w(f"| {r.h} | {r.model} | {r.n} | {r.countries} | {r.rmse_without:.4f} | {r.rmse_with:.4f} | {r.rmse_reduction_pct:+.2f}% | "
          f"{r.mae_without:.4f} | {r.mae_with:.4f} | {r.mae_reduction_pct:+.2f}% | {r.dm_mse_p:.3f} | {r.dm_mae_p:.3f} | {r.rmse_with_vs_snaive_pct:+.1f}% |")

w("\n### ③ 이질성 (L1, causal forest, +10%p 침투율당 log 호텔 숙박일수)\n")
w("| 처치 정의 | n | 지역 | ATE [95% CI] | p | 지역 CATE 평균 | SD | p10 | p90 | p10–p90 폭 | CATE<0 지역 비율 | CI>0 지역 | CI<0 지역 |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
cfs = [(O / "L1_causal_forest_pen_fixed.json", "고정분모 (제안 주분석)"), (O / "L1_causal_forest.json", "관례 share (분모에 I551)")]
for p, lab in cfs:
    if not p.exists(): continue
    r = json.load(open(p)); a = r["ate_per10pp"]; c = r["cate_region"]
    w(f"| {lab} | {r['n']} | {r['regions']} | {a['est']:.4f} [{a['ci'][0]:.4f}, {a['ci'][1]:.4f}] | {a['p']:.3f} | {c['mean']:.4f} | {c['sd']:.4f} | "
      f"{c['p10']:.4f} | {c['p90']:.4f} | {c['width_p10_p90']:.4f} | {pc(c['share_negative'])} | {c['n_ci_above_0']} | {c['n_ci_below_0']} |")
for p, lab in cfs:
    if not p.exists(): continue
    r = json.load(open(p))
    w(f"\n{lab} — 변수 중요도(분할 기반) / BLP 계수(표준화 공변량 1SD 당, 지역군집 p):\n")
    w("| 공변량 | 중요도 | BLP 계수 | p |"); w("|---|---|---|---|")
    for k, v in sorted(r["feature_importance"].items(), key=lambda kv: -kv[1]):
        w(f"| {k} | {v:.4f} | {r['blp']['coef'][k]:+.4f} | {r['blp']['p'][k]:.3f} |")
    w("\n침투율 수준 5분위별 지역 CATE:\n")
    w("| 분위 | 지역 수 | 침투율 범위 | CATE 평균 | CATE<0 비율 |"); w("|---|---|---|---|---|")
    for q in r["by_pen_quintile"]:
        w(f"| {q['q']} | {q['n']} | {q['pen_min']:.3f}–{q['pen_max']:.3f} | {q['cate10_mean']:+.4f} | {pc(q['share_neg'])} |")

p = O / "L3_gini.json"
if p.exists():
    g = json.load(open(p))
    w("\n### 참고: L3 계절집중도 (NUTS2×월, 숙박 전체 I551–I553 — 업종 분리 없음)\n")
    w(f"Gini 지역-연 {g['region_years_gini']}개, {g['regions_gini']}지역; 분위 p10/p50/p90 = "
      f"{g['gini_quantiles']['p10']:.3f}/{g['gini_quantiles']['p50']:.3f}/{g['gini_quantiles']['p90']:.3f}\n")
    w("| 표본 · 처치 | n | 지역 | +10%p 당 Gini 변화 [95% CI] | p | Gini SD |"); w("|---|---|---|---|---|---|")
    for k, v in g.items():
        if "|" in k:
            w(f"| {k} | {v['n']} | {v['regions']} | {v['beta_per10pp']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['p']:.3f} | {v['sd_gini']:.3f} |")
p = O / "C2_claim1_defence.json"
if p.exists():
    C = json.load(open(p))
    w("\n### C2 주장 1 방어 — 지역FE+연FE, 지역 군집 SE (G−1 자유도), 와일드 군집 부트스트랩(Rademacher, 귀무 부과, 9,999회)\n")
    w("| 명세 | n | 지역 | 계수 | SE | p (군집) | p (와일드) | 추가 계수 |"); w("|---|---|---|---|---|---|---|---|")
    for k, v in C.items():
        if "coef" not in v: continue
        x = list(v["coef"])[0]
        extra = "; ".join(f"{z} {v['coef'][z]:+.4f} (p {v['p'][z]:.3f})" for z in list(v["coef"])[1:]) or "–"
        wp = f"{v['wild_p']:.4f}" if "wild_p" in v else "–"
        w(f"| {k} | {v['n']} | {v['regions']} | {v['coef'][x]:+.4f} | {v['se'][x]:.4f} | {v['p'][x]:.4f} | {wp} | {extra} |")
    pm = C["permutation (platform shuffled within region; conventional share)"]
    w(f"\n순열 귀무(지역 내에서 플랫폼 숙박일수를 연도 간 섞어 실제 연결을 끊고 관례 share 재구성, {pm['draws']}회): "
      f"평균 β {pm['mean_beta']:+.4f}, 2.5–97.5% [{pm['p2_5']:+.4f}, {pm['p97_5']:+.4f}]; 관측 β 이하 비율 {pm['share_beta_le_observed']:.3f}")
p = O / "L1_event_study.json"
if p.exists():
    E = json.load(open(p))
    w(f"\n### C3 규제 event study (stacked, clean controls) — 처치 {E['n_treated']}지역, 대조 {E['n_controls']}지역({len(E['control_countries'])}개국), 행 {E['rows']}\n")
    w("| 결과 | DiD (post) | SE | p | pre(≤−2) | p | post 0–1 | p | post ≥2 | p |"); w("|---|---|---|---|---|---|---|---|---|---|")
    for y in ["log_I551", "log_plat"]:
        r = E[y]; ev = r["event"]
        w(f"| {y} | {r['did']['b']:+.4f} | {r['did']['se']:.4f} | {r['did']['p']:.4f} | {ev['pre_far']['b']:+.4f} | {ev['pre_far']['p']:.4f} | "
          f"{ev['post_0_1']['b']:+.4f} | {ev['post_0_1']['p']:.4f} | {ev['post_2p']['b']:+.4f} | {ev['post_2p']['p']:.4f} |")
    w("\n처치 지역·처치연도(상반기 시행=당해, 하반기=익년): " + ", ".join(f"{e['nuts2']} {e['treat_year']}" for e in E["events"]))
w("\n### C1 플라시보 확장 — causal forest (결과 = I551 객실수 t−4)\n")
w("| 처치 · 결과 | n | ATE/10%p [95% CI] | p | CI<0 지역 | CI>0 지역 | BLP pen_level | p | 분위 Q1→Q5 CATE |"); w("|---|---|---|---|---|---|---|---|---|")
for fn in ["L1_causal_forest.json", "L1_causal_forest_placebo.json", "L1_causal_forest_pen_fixed.json", "L1_causal_forest_pen_fixed_placebo.json"]:
    p = O / fn
    if not p.exists(): continue
    r = json.load(open(p)); a = r["ate_per10pp"]; c = r["cate_region"]
    w(f"| {r.get('treat', 'pen')} · {r.get('outcome', 'log_I551')} | {r['n']} | {a['est']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}] | {a['p']:.3f} | "
      f"{c['n_ci_below_0']} | {c['n_ci_above_0']} | {r['blp']['coef']['pen_level']:+.4f} | {r['blp']['p']['pen_level']:.4f} | "
      + " / ".join(f"{q['cate10_mean']:+.4f}" for q in r["by_pen_quintile"]) + " |")
p = O / "C1_L3_placebo.json"
if p.exists():
    g = json.load(open(p))
    w("\n### C1 플라시보 확장 — L3 Gini: 현재 침투율 → 과거 Gini (지역FE+연FE, t = 2022–2025)\n")
    w("| 회귀 | n | 지역 | +10%p 당 [95% CI] | p |"); w("|---|---|---|---|---|")
    for k, v in g.items():
        w(f"| {k} | {v['n']} | {v['regions']} | {v['beta_per10pp']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['p']:.3f} |")
    w("\n주의: t=2022 의 Gini(t−1)은 2021(COVID 구간) 값이다.")
print("\n".join(out))
