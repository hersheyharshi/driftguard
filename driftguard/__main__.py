"""DriftGuard command line.

  python -m driftguard demo                     # synthetic run -> SQLite warehouse + lineage
  python -m driftguard bench [--quick]          # all three experiments -> results/
  python -m driftguard report                   # results/ -> results/report.html
  python -m driftguard map FILE                 # show how a raw file's columns get mapped
  python -m driftguard ingest FILES... --db warehouse.sqlite [--zones taxi_zones.shp]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)  # needs pyarrow
    return pd.read_csv(p, low_memory=False)


def _print_reports(reports):
    for r in reports:
        if r.status == "WARMUP":
            continue
        print(f"  {r.batch_id:>14}  {r.status:<18} loaded={r.rows_loaded:>7,}  "
              f"quarantined={r.rows_quarantined:>6,}  {', '.join(sorted(r.checks())) or '-'}")


def cmd_demo(a):
    from .faults import inject
    from .pipeline import DriftGuard
    from .registry import SchemaRegistry
    from .synth import TripGenerator, to_raw, zone_centroids
    from .transform import ZoneResolver, load_schema
    from .warehouse import open_warehouse

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "demo.sqlite"
    db.unlink(missing_ok=True)
    schema = load_schema()
    g = TripGenerator(a.rows, seed=a.seed)
    reg = SchemaRegistry(schema, memory_path=out / "registry_memory.json").fit_reference(g.day("2023-12-01"))
    dg = DriftGuard(reg, schema, warehouse=open_warehouse(str(db)), zones=ZoneResolver(zone_centroids()))
    days = pd.date_range("2024-01-01", periods=28)
    plan = {15: "unit_km", 17: "clock_shift", 19: "duplicate_batch", 21: "obfuscated_headers",
            23: "column_swap", 25: "subtle_scale", 27: "type_corruption"}
    rng = np.random.default_rng(a.seed)
    prev = None
    print(f"DriftGuard demo · run {dg.run_id} · warehouse {db}")
    for i, d in enumerate(days):
        clean_raw = to_raw(g.day(d))
        raw = clean_raw
        injected = plan.get(i)
        if injected:
            raw, _ = inject(injected, raw, rng, previous=prev)
        rep = dg.process(raw, f"{d.date()}", d, warmup=i < 14)
        if i == 13:
            dg.end_warmup()
            print("  warm-up complete: thresholds calibrated on 14 trusted days")
        if i >= 14:
            print(f"  {d.date()}  injected={injected or '-':<18} -> {rep.status:<18} "
                  f"{', '.join(sorted(rep.checks())) or '-'}")
        prev = clean_raw
    manifest = out / f"lineage_{dg.run_id}.json"
    manifest.write_text(json.dumps([{"batch": r.batch_id, "status": r.status, "repairs": r.repairs,
                                     "findings": r.findings, "lineage": r.lineage} for r in dg.reports],
                                   indent=2, default=str))
    print(f"\nTables: {dg.wh.tables()}\nLineage manifest: {manifest}")
    print(dg.wh.query("SELECT status, COUNT(*) AS batches, SUM(rows_loaded) AS rows FROM batches GROUP BY status"))


def cmd_bench(a):
    script = Path(__file__).resolve().parents[1] / "experiments" / "run_all.py"
    sys.exit(subprocess.call([sys.executable, "-W", "ignore", str(script)] + (["--quick"] if a.quick else [])
                             + ["--out", a.out]))


def cmd_report(a):
    from .report import build_report
    p = build_report(a.results, Path(a.results) / "report.html")
    print(f"wrote {p}")


def cmd_map(a):
    from .registry import SchemaRegistry
    from .transform import load_schema
    schema = load_schema()
    reg = SchemaRegistry(schema)
    if a.reference:
        ref = _read(a.reference)
        m0 = reg.infer(ref, use_stats=False)
        from .transform import coerce_to_canonical
        reg.fit_reference(coerce_to_canonical(ref, m0.mapping, schema)[0])
    df = _read(a.file)
    m = reg.infer(df.sample(min(len(df), 50000), random_state=0))
    print(f"fingerprint {m.fingerprint}   needs review: {m.needs_review}")
    for c, v in m.scores.items():
        print(f"  {c:<22} <- {v['source']:<26} score={v['score']:.2f} name={v['name']:.2f} "
              f"stat={v['stat'] if v['stat'] is not None else '-'}  [{v['status']}]")
    if m.ambiguous:
        print("  ambiguous:", m.ambiguous)
    if m.resolved_by_invariant:
        print("  resolved by invariants:", m.resolved_by_invariant)
    print("  unmapped source columns:", m.unmapped_sources)
    print("  missing required:", m.missing_required)


def cmd_ingest(a):
    from .pipeline import DriftGuard
    from .registry import SchemaRegistry
    from .transform import ZoneResolver, coerce_to_canonical, load_schema
    from .warehouse import open_warehouse

    schema = load_schema()
    reg = SchemaRegistry(schema, memory_path=Path(a.db).with_suffix(".registry.json"))
    zones = ZoneResolver(shapefile=a.zones) if a.zones else None
    dg = DriftGuard(reg, schema, warehouse=open_warehouse(a.db), zones=zones)
    seen_days = 0
    for k, path in enumerate(a.files):
        df = _read(path)
        print(f"{path}: {len(df):,} rows")
        if k == 0:  # first file = trusted reference; its names must map exactly
            m0 = reg.infer(df.sample(min(len(df), 50000), random_state=0), use_stats=False)
            reg.fit_reference(coerce_to_canonical(df.sample(min(len(df), 50000), random_state=0), m0.mapping, schema)[0])
        if a.split == "daily":
            m = reg.infer(df.sample(min(len(df), 50000), random_state=0))
            ts = pd.to_datetime(df[m.mapping["pickup_ts"]], errors="coerce").dt.normalize()
            month = ts.mode().iloc[0].to_period("M")
            for day, part in df.groupby(ts):
                if day.to_period("M") != month:
                    continue  # stray dates are routed by their own month's file
                warm = seen_days < a.warmup
                dg.process(part.reset_index(drop=True), f"{day.date()}", day, warmup=warm)
                seen_days += 1
                if seen_days == a.warmup:
                    dg.end_warmup()
        else:
            d = pd.Timestamp(Path(path).stem[-7:] + "-01") if Path(path).stem[-7:-3].isdigit() else pd.Timestamp.today()
            dg.process(df, Path(path).stem, d, warmup=seen_days < a.warmup)
            seen_days += 1
        _print_reports(dg.reports[-40:])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="driftguard")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("demo"); p.add_argument("--out", default="out"); p.add_argument("--rows", type=int, default=6000)
    p.add_argument("--seed", type=int, default=7); p.set_defaults(fn=cmd_demo)
    p = sp.add_parser("bench"); p.add_argument("--quick", action="store_true"); p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_bench)
    p = sp.add_parser("report"); p.add_argument("--results", default="results"); p.set_defaults(fn=cmd_report)
    p = sp.add_parser("map"); p.add_argument("file"); p.add_argument("--reference"); p.set_defaults(fn=cmd_map)
    p = sp.add_parser("ingest"); p.add_argument("files", nargs="+"); p.add_argument("--db", default="out/warehouse.sqlite")
    p.add_argument("--zones"); p.add_argument("--split", choices=["daily", "file"], default="daily")
    p.add_argument("--warmup", type=int, default=14); p.set_defaults(fn=cmd_ingest)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
