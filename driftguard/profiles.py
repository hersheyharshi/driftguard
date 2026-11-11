"""Column profiling: a compact statistical fingerprint of a column.

Profiles let the schema registry recognise a column by *what its values look
like*, not only by its header. A column called `col_07` whose values look like
fares (heavy-tailed, positive, 2 decimals, ~$15 median) is probably a fare.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

QS = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
_JUNK = re.compile(r"[,\s$€£]")


# --------------------------------------------------------------------------
# Coercion helpers (also used by the pipeline, so repairs are counted once)
# --------------------------------------------------------------------------
def clean_numeric(s: pd.Series) -> tuple[pd.Series, int]:
    """Coerce to float64, repairing formatting damage like '$1,234.50'.
    Returns (values, number_of_values_that_needed_repair)."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("float64"), 0
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64"), 0
    raw = s.astype("string")
    out = _to_float(pd.to_numeric(raw, errors="coerce"))
    need = np.isnan(out) & raw.notna().to_numpy()
    n_rep = 0
    if need.any():
        fixed = _to_float(pd.to_numeric(raw[need].str.replace(_JUNK, "", regex=True), errors="coerce"))
        n_rep = int((~np.isnan(fixed)).sum())
        out[need] = fixed
    return pd.Series(out, index=s.index, name=s.name), n_rep


def _to_float(x) -> np.ndarray:
    return pd.array(x).to_numpy(dtype="float64", na_value=np.nan).copy()


def parse_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s.astype("datetime64[ns]")
    raw = s.astype("string")
    out = pd.to_datetime(raw, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    miss = out.isna() & raw.notna()
    if miss.any():
        out[miss] = pd.to_datetime(raw[miss], errors="coerce", format="mixed")
    return out.astype("datetime64[ns]")


def norm_value(v) -> str:
    """Normalise a categorical value for vocabulary matching."""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}"
    return str(v).strip().lower()


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


# --------------------------------------------------------------------------
@dataclass
class ColumnProfile:
    name: str
    kind: str                     # numeric | datetime | text | empty
    null_rate: float
    n_unique: int
    q: np.ndarray | None = None   # signed-log quantiles (numeric) or hours-of-day (datetime)
    int_frac: float = 0.0
    zero_frac: float = 0.0
    neg_frac: float = 0.0
    freq: dict = field(default_factory=dict)  # top normalised values -> share

    @property
    def low_card(self) -> bool:
        return self.n_unique <= 12


def profile_column(s: pd.Series, name: str | None = None, sample: int = 5000) -> ColumnProfile:
    name = name if name is not None else str(s.name)
    n = len(s)
    nn = s.dropna()
    null_rate = 1 - len(nn) / n if n else 1.0
    if len(nn) == 0:
        return ColumnProfile(name, "empty", null_rate, 0)
    x = nn.sample(min(len(nn), sample), random_state=0) if len(nn) > sample else nn

    kind = "text"
    vals = None
    if pd.api.types.is_datetime64_any_dtype(x):
        kind = "datetime"
    elif pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
        kind, vals = "numeric", x.astype("float64")
    else:
        num, _ = clean_numeric(x)
        if num.notna().mean() > 0.95:
            kind, vals = "numeric", num.dropna()
        else:
            xs = x.astype("string")
            looks_dt = xs.str.len().between(8, 35).mean() > 0.95 and xs.str.contains(r"\d{1,4}[-/:]").mean() > 0.95
            if looks_dt and parse_datetime(xs.head(500)).notna().mean() > 0.95:
                kind = "datetime"

    if kind == "datetime":
        dt = parse_datetime(x).dropna()
        hours = (dt.dt.hour + dt.dt.minute / 60).to_numpy()
        return ColumnProfile(name, "datetime", null_rate, int(dt.nunique()),
                             q=np.quantile(hours, QS) if len(hours) else None)

    if kind == "numeric":
        v = vals.to_numpy()
        nu = int(pd.Series(v).nunique())
        prof = ColumnProfile(name, "numeric", null_rate, nu, q=np.quantile(_slog(v), QS),
                             int_frac=float(np.mean(np.isclose(v, np.round(v)))),
                             zero_frac=float(np.mean(v == 0)), neg_frac=float(np.mean(v < 0)))
    else:
        prof = ColumnProfile(name, "text", null_rate, int(x.nunique()))
    vc = x.map(norm_value).value_counts(normalize=True).head(20)
    prof.freq = {str(k): float(v) for k, v in vc.items()}
    return prof


def profile_frame(df: pd.DataFrame, sample: int = 5000) -> dict[str, ColumnProfile]:
    return {c: profile_column(df[c], c, sample) for c in df.columns}


# --------------------------------------------------------------------------
def _tv(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def stat_similarity(src: ColumnProfile, ref: ColumnProfile, value_map: dict | None = None) -> float:
    """0..1 similarity between a source column profile and a canonical reference profile."""
    if src.kind == "empty" or ref.kind == "empty":
        return 0.0
    if value_map:
        # Vocabulary evidence: how much of the source maps into the canonical vocab,
        # and does the mapped frequency distribution look like the reference?
        mapped: dict[str, float] = {}
        cover = 0.0
        for k, share in src.freq.items():
            if k in value_map:
                cover += share
                mapped[value_map[k]] = mapped.get(value_map[k], 0) + share
        if cover == 0:
            return 0.0
        return float(0.5 * cover + 0.5 * (1 - _tv(mapped, ref.freq)))
    if src.kind != ref.kind:
        return 0.0
    null_sim = 1 - abs(src.null_rate - ref.null_rate)
    if src.kind == "datetime":
        if src.q is None or ref.q is None:
            return 0.5
        d = float(np.mean(np.abs(src.q - ref.q)))  # hours
        return float(0.85 * np.exp(-d / 2.0) + 0.15 * null_sim)
    if src.kind == "text":
        return float(0.8 * (1 - _tv(src.freq, ref.freq)) + 0.2 * null_sim)
    # numeric
    dq = float(np.mean(np.abs(src.q - ref.q)))
    s_q = np.exp(-2.0 * dq)
    la, lb = np.log1p(src.n_unique), np.log1p(ref.n_unique)
    s_card = 1 - abs(la - lb) / max(la, lb, 1.0)
    s_frac = 1 - np.mean([abs(src.int_frac - ref.int_frac), abs(src.zero_frac - ref.zero_frac),
                          abs(src.neg_frac - ref.neg_frac)])
    if src.low_card and ref.low_card:
        s_set = 1 - _tv(src.freq, ref.freq)
        return float(0.35 * s_q + 0.35 * s_set + 0.15 * s_frac + 0.15 * s_card)
    return float(0.55 * s_q + 0.2 * s_frac + 0.15 * s_card + 0.1 * null_sim)
