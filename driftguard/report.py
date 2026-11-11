"""Self-contained HTML report (no JS libraries, works offline).

Charts are hand-built inline SVG with native hover tooltips, each paired
with a data table, colour-blind-validated palette, light + dark mode.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--neutral:#8a8983;--good:#0ca30c;--warn:#fab219;--serious:#ec835a;--crit:#d03b3b;--card:#ffffff}
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--neutral:#8f8e86;--card:#222220}}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--surface);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;padding:32px 20px}
.wrap{max-width:980px;margin:0 auto}h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 6px}
.sub{color:var(--ink2);margin:0 0 20px}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:14px}
@media (max-width:640px){.kpis{grid-template-columns:1fr 1fr}}
.kpi b{display:block;font-size:26px}.kpi span{color:var(--ink2);font-size:13px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:16px;margin:10px 0}
.note{color:var(--ink2);font-size:13.5px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}
details summary{cursor:pointer;color:var(--ink2);font-size:13.5px;margin-top:8px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:6px 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
svg text{fill:var(--ink2);font-size:12px}svg .lbl{fill:var(--ink)}
svg .hit:hover+.mark,svg .mark:hover{opacity:.75}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt and pd.notna(v):
                v = fmt[c](v)
            elif isinstance(v, float):
                v = "–" if pd.isna(v) else f"{v:.3f}"
            cells.append(f"<td>{_esc(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _pct(v):
    return f"{v:.0%}"


def hbar(labels, values, title_fmt=_pct, color="var(--s1)", width=900, maxv=1.0) -> str:
    """Horizontal bars, 0..maxv, label left, value right, hover tooltip."""
    row, left, right = 26, 170, 60
    h = row * len(labels) + 24
    plot = width - left - right
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot * t
        out.append(f'<line x1="{x}" x2="{x}" y1="0" y2="{h-20}" stroke="var(--grid)"/>'
                   f'<text x="{x}" y="{h-4}" text-anchor="middle">{title_fmt(t * maxv)}</text>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = i * row + 4
        w = 0 if pd.isna(v) else max(plot * v / maxv, 2)
        txt = "n/a" if pd.isna(v) else title_fmt(v)
        out.append(f'<text class="lbl" x="{left-8}" y="{y+13}" text-anchor="end">{_esc(lab)}</text>'
                   f'<g><title>{_esc(lab)}: {txt}</title><rect class="mark" x="{left}" y="{y}" width="{w}" height="{row-10}" '
                   f'rx="4" fill="{color}"/></g><text x="{left+w+6}" y="{y+13}">{txt}</text>')
    out.append("</svg>")
    return "".join(out)


def line_chart(x, series: dict, ref: float | None, width=900, height=300, ylab="WAPE") -> str:
    left, right, top, bot = 56, 150, 14, 34
    pw, ph = width - left - right, height - top - bot
    allv = [v for s in series.values() for v in s["y"]] + ([ref] if ref else [])
    lo, hi = min(allv) * 0.97, max(allv) * 1.03
    X = lambda v: left + pw * (v - min(x)) / (max(x) - min(x))  # noqa: E731
    Y = lambda v: top + ph * (1 - (v - lo) / (hi - lo))           # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for t in np.linspace(lo, hi, 5):
        out.append(f'<line x1="{left}" x2="{left+pw}" y1="{Y(t)}" y2="{Y(t)}" stroke="var(--grid)"/>'
                   f'<text x="{left-8}" y="{Y(t)+4}" text-anchor="end">{t:.1%}</text>')
    for v in x:
        out.append(f'<text x="{X(v)}" y="{height-12}" text-anchor="middle">{v:.0%}</text>')
    out.append(f'<text x="{left+pw/2}" y="{height}" text-anchor="middle">share of batches with an injected fault</text>')
    if ref:
        out.append(f'<line x1="{left}" x2="{left+pw}" y1="{Y(ref)}" y2="{Y(ref)}" stroke="var(--neutral)" '
                   f'stroke-width="2" stroke-dasharray="5 4"/><text x="{left+pw+10}" y="{Y(ref)+18}">oracle {ref:.1%}</text>')
    for name, s in series.items():
        pts = " ".join(f"{X(a)},{Y(b)}" for a, b in zip(x, s["y"]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2"/>')
        for a, b in zip(x, s["y"]):
            out.append(f'<g><title>{_esc(name)} @ {a:.0%} faults: {b:.2%} {ylab}</title>'
                       f'<circle cx="{X(a)}" cy="{Y(b)}" r="12" fill="transparent"/>'
                       f'<circle cx="{X(a)}" cy="{Y(b)}" r="4.5" fill="{s["color"]}" stroke="var(--surface)" stroke-width="2"/></g>')
        out.append(f'<text class="lbl" x="{X(x[-1])+10}" y="{Y(s["y"][-1])+4}">{_esc(name)}</text>')
    out.append("</svg>")
    return "".join(out)


STATUS_COLOR = {"ACCEPTED": "var(--good)", "ACCEPTED_REPAIRED": "var(--s1)", "HELD": "var(--warn)",
                "REJECTED": "var(--crit)", "WARMUP": "var(--neutral)"}
STATUS_ICON = {"ACCEPTED": "✓", "ACCEPTED_REPAIRED": "↺", "HELD": "‖", "REJECTED": "✕", "WARMUP": "·"}


def timeline(df: pd.DataFrame, width=900) -> str:
    n = len(df)
    cw = min(26, (width - 10) / max(n, 1))
    out = [f'<svg viewBox="0 0 {width} 52" width="100%" role="img">']
    for i, r in enumerate(df.itertuples()):
        c = STATUS_COLOR.get(r.status, "var(--neutral)")
        tip = f"day {r.day}: injected={r.fault} → {r.status}; checks: {r.checks or 'none'}"
        out.append(f'<g><title>{_esc(tip)}</title><rect class="mark" x="{i*cw+1}" y="4" width="{cw-2}" height="30" rx="3" fill="{c}"/>'
                   f'<text x="{i*cw+cw/2}" y="24" text-anchor="middle" style="fill:#fff;font-size:11px">{STATUS_ICON.get(r.status,"")}</text></g>')
        if r.fault != "clean":
            out.append(f'<rect x="{i*cw+1}" y="38" width="{cw-2}" height="4" rx="2" fill="var(--ink2)"/>')
    out.append("</svg>")
    return "".join(out)


def build_report(results_dir: str | Path, out_path: str | Path) -> Path:
    R = Path(results_dir)
    e1 = pd.read_csv(R / "e1_schema.csv")
    e2 = pd.read_csv(R / "e2_batches.csv")
    e2s = pd.read_csv(R / "e2_summary.csv")
    e3 = pd.read_csv(R / "e3_downstream.csv")
    meta = json.loads((R / "meta.json").read_text())

    # ---- KPIs
    det = e2s.dropna(subset=["detected"])
    detect_rate = (det["detected"] * det["n"]).sum() / det["n"].sum()
    cause_rate = (det["cause_ok"] * det["n"]).sum() / det["n"].sum()
    fa = meta["false_alarms"] / max(meta["clean_batches"], 1)
    g = e3[e3["pipeline"] != "oracle (true data)"].groupby(["pipeline", "rate"])["wape"].mean().unstack(0)
    oracle = e3[e3["pipeline"] == "oracle (true data)"]["wape"].mean()
    worst = g.index.max()
    damage_naive = g.loc[worst, "naive"] - oracle
    damage_guard = g.loc[worst, "DriftGuard"] - oracle
    recovered = 1 - damage_guard / damage_naive if damage_naive > 0 else np.nan
    e1g = e1.groupby(["headers", "evidence"])[["correct", "n_cols", "silent_wrong", "flagged"]].sum().reset_index()
    full = e1g[e1g["evidence"] == "name + stats + invariants"]
    acc_named = (full[full.headers == "named headers"].eval("correct / n_cols")).iloc[0]
    acc_obf = (full[full.headers == "meaningless headers"].eval("correct / n_cols")).iloc[0]

    kpis = [(f"{detect_rate:.0%}", "faults detected"), (f"{cause_rate:.0%}", "with the correct cause named"),
            (f"{fa:.1%}", f"false-alarm rate ({meta['false_alarms']}/{meta['clean_batches']} clean batches)"),
            (f"{recovered:.0%}", f"of forecast damage prevented at {worst:.0%} fault rate"),
            (f"{acc_named:.0%} / {acc_obf:.0%}", "columns identified (named / meaningless headers)"),
            (str(int(e1g['silent_wrong'].sum())), "silent column mis-mappings, all configurations")]
    kpi_html = "".join(f'<div class="kpi"><b>{a}</b><span>{_esc(b)}</span></div>' for a, b in kpis)

    # ---- E1
    e1t = e1g.assign(accuracy=lambda d: d.correct / d.n_cols)[["headers", "evidence", "accuracy", "flagged", "silent_wrong"]]
    e1t.columns = ["headers", "evidence used", "columns correct", "flagged for review", "silent errors"]
    obf = e1t[e1t["headers"] == "meaningless headers"]
    e1_chart = hbar(list(obf["evidence used"]), list(obf["columns correct"]))

    # ---- E2
    det_sorted = det.sort_values("detected")
    e2_chart = hbar(list(det_sorted["fault"]), list(det_sorted["cause_ok"]))
    e2t = e2s[["fault", "n", "detected", "cause_ok", "auto_repaired", "held_or_rejected", "map_correct", "repair_ks"]].copy()
    e2t.columns = ["fault", "n", "detected", "correct cause", "auto-repaired", "held/rejected",
                   "mapping accuracy", "max KS vs truth after repair"]
    seed0 = e2[e2["seed"] == e2["seed"].min()].reset_index(drop=True)

    # ---- E3
    rates = list(g.index)
    e3_chart = line_chart(rates, {"naive": {"y": list(g["naive"]), "color": "var(--s2)"},
                                  "DriftGuard": {"y": list(g["DriftGuard"]), "color": "var(--s1)"}}, oracle)
    e3t = g.reset_index().rename(columns={"rate": "fault rate"})
    e3t["oracle"] = oracle

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriftGuard — benchmark report</title><style>{CSS}</style></head><body><div class="viz-root"><div class="wrap">
<h1>DriftGuard benchmark report</h1>
<p class="sub">Schema-drift-tolerant ingestion with cause-attributing data-quality checks. Synthetic NYC-taxi-like data with known ground truth;
seeds {meta['seeds_e1_e2']} (E1, E2) and {meta['seeds_e3']} (E3).</p>
<div class="kpis">{kpi_html}</div>

<h2>E1 · Can it recognise columns it has never seen?</h2>
<p class="note">Six header eras (2009 → 2023 layouts), each also scrambled to meaningless names (<code>col_00…</code>) in random order.
Bars: share of columns identified correctly under meaningless headers, by evidence used.
Invariants don't add accuracy here; they <em>resolve near-ties</em>, cutting how many columns a human must review.</p>
<div class="card">{e1_chart}<details><summary>Table view</summary>{_table(e1t, {"columns correct": _pct})}</details></div>

<h2>E2 · Does it catch bad batches, and name the cause?</h2>
<p class="note">Balanced schedule: every fault type injected {int(det['n'].iloc[0])}× ({len(meta['seeds_e1_e2'])} seeds × 2).
Bars: share of injected faults where the <em>correct cause</em> was named.</p>
<div class="card">{e2_chart}<details open><summary>Table view</summary>{_table(e2t, {k: _pct for k in ["detected", "correct cause", "auto-repaired", "held/rejected", "mapping accuracy"]})}</details></div>
<p class="note">Batch timeline, seed {int(seed0['seed'].iloc[0])} (hover a cell). Dark tick under a cell = a fault was injected that day.</p>
<div class="legend">{''.join(f'<span><i class="sw" style="background:{c}"></i>{STATUS_ICON[s]} {s.lower().replace("_", " ")}</span>' for s, c in STATUS_COLOR.items() if s != "WARMUP")}</div>
<div class="card">{timeline(seed0)}</div>

<h2>E3 · What does bad data cost a model?</h2>
<p class="note">Day-ahead hourly demand per zone (gradient boosting, Poisson loss). Each pipeline's model trains on <em>its own</em> warehouse;
error is scored against the true demand. Dashed line: model trained on perfect data.</p>
<div class="legend"><span><i class="sw" style="background:var(--s2)"></i>naive pipeline</span><span><i class="sw" style="background:var(--s1)"></i>DriftGuard</span><span><i class="sw" style="background:var(--neutral)"></i>oracle</span></div>
<div class="card">{e3_chart}<details><summary>Table view</summary>{_table(e3t, {c: (lambda v: f"{v:.2%}") for c in ["DriftGuard", "naive", "oracle"]} | {"fault rate": _pct})}</details></div>
<p class="note">Synthetic results establish that the method works when the truth is known. Re-run on real TLC files before
quoting numbers about real data (see README → "Running on real data").</p>
</div></div></body></html>"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
