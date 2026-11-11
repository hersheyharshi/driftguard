"""Storage backends. Same interface; pick by what's installed.

  MemoryWarehouse  - for benchmarks (fast, nothing on disk)
  SQLiteWarehouse  - zero-dependency persistent store (stdlib sqlite3)
  DuckDBWarehouse  - recommended for real TLC volumes (`pip install duckdb`)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from .synth import CANONICAL_ORDER

TS_COLS = ("pickup_ts", "dropoff_ts")


def _normalise(table: str, df: pd.DataFrame) -> pd.DataFrame:
    if table in ("trips", "quarantine"):
        extra = [c for c in df.columns if c not in CANONICAL_ORDER]
        df = df.reindex(columns=CANONICAL_ORDER + extra)
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (dict, list))).any():
            out[c] = out[c].map(json.dumps)
    return out


class MemoryWarehouse:
    def __init__(self):
        self._t: dict[str, list[pd.DataFrame]] = {}

    def append(self, table: str, df: pd.DataFrame):
        if len(df):
            self._t.setdefault(table, []).append(_normalise(table, df))

    def read(self, table: str) -> pd.DataFrame:
        parts = self._t.get(table, [])
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def tables(self):
        return list(self._t)


class SQLiteWarehouse:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path))
        self._cols: dict[str, list[str]] = {}

    def append(self, table: str, df: pd.DataFrame):
        if not len(df):
            return
        out = _normalise(table, df)
        for c in TS_COLS:
            if c in out:
                out[c] = out[c].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")
        cols = self._cols.get(table)
        if cols is None:
            try:
                cols = [r[1] for r in self.con.execute(f"PRAGMA table_info({table})")]
            except sqlite3.Error:
                cols = []
        new = [c for c in out.columns if cols and c not in cols]
        for c in new:
            self.con.execute(f'ALTER TABLE {table} ADD COLUMN "{c}"')
        out.to_sql(table, self.con, if_exists="append", index=False)
        self._cols[table] = list(dict.fromkeys((cols or []) + list(out.columns)))
        self.con.commit()

    def read(self, table: str) -> pd.DataFrame:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", self.con)
        except Exception:
            return pd.DataFrame()
        for c in TS_COLS:
            if c in df:
                df[c] = pd.to_datetime(df[c])
        return df

    def query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self.con)

    def tables(self):
        return [r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")]


class DuckDBWarehouse:
    def __init__(self, path: str | Path):
        import duckdb  # optional
        self.con = duckdb.connect(str(path))

    def append(self, table: str, df: pd.DataFrame):
        if not len(df):
            return
        out = _normalise(table, df)  # noqa: F841  (referenced by duckdb by name)
        exists = self.con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()[0]
        if exists:
            self.con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM out")
        else:
            self.con.execute(f"CREATE TABLE {table} AS SELECT * FROM out")

    def read(self, table: str) -> pd.DataFrame:
        return self.con.execute(f"SELECT * FROM {table}").df()

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def tables(self):
        return [r[0] for r in self.con.execute("SHOW TABLES").fetchall()]


class AggWarehouse:
    """Memory-light store for multi-million-row studies: trips are kept only as
    (hour, pickup zone) aggregates; quarantined rows only as counts per reason.
    Everything else (batches, findings, lineage) is kept in full."""

    def __init__(self):
        self._agg: list[pd.DataFrame] = []
        self._reasons: dict[str, int] = {}
        self._other = MemoryWarehouse()

    def append(self, table: str, df: pd.DataFrame):
        if not len(df):
            return
        if table == "trips":
            t = df[df["pickup_ts"].notna() & df["pu_zone"].notna()]
            g = t.groupby([t["pickup_ts"].dt.floor("h"), t["pu_zone"]])
            a = pd.DataFrame({"n": g.size(), "dist_sum": g["trip_distance_mi"].sum(),
                              "fare_sum": g["fare_amount"].sum()}).reset_index()
            a["batch_id"] = df["batch_id"].iloc[0] if "batch_id" in df else None
            self._agg.append(a)
        elif table == "quarantine":
            for r, n in df["reasons"].astype(str).str.split(";").explode().value_counts().items():
                key = r.split(":")[0] if r.startswith("batch_") else r
                self._reasons[key] = self._reasons.get(key, 0) + int(n)
        else:
            self._other.append(table, df)

    def read(self, table: str) -> pd.DataFrame:
        if table == "trips":
            if not self._agg:
                return pd.DataFrame()
            a = pd.concat(self._agg, ignore_index=True)
            a = a.groupby(["pickup_ts", "pu_zone"], as_index=False)[["n", "dist_sum", "fare_sum"]].sum()
            a["trip_distance_mi"] = a["dist_sum"] / a["n"]
            a["fare_amount"] = a["fare_sum"] / a["n"]
            return a
        if table == "quarantine_reasons":
            return pd.DataFrame(sorted(self._reasons.items(), key=lambda kv: -kv[1]), columns=["reason", "rows"])
        return self._other.read(table)

    def tables(self):
        return ["trips", "quarantine_reasons"] + self._other.tables()


def open_warehouse(path: str | None):
    if not path:
        return MemoryWarehouse()
    if str(path).endswith(".duckdb"):
        return DuckDBWarehouse(path)
    return SQLiteWarehouse(path)
