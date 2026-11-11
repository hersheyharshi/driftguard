"""Schema registry: maps any raw header layout onto the canonical schema.

Three kinds of evidence, combined:
  1. Name similarity   - aliases, abbreviation expansion, fuzzy string match.
  2. Statistical fit   - does the column's value fingerprint look like the
                         canonical column's reference profile?
  3. Invariants        - when two assignments are near-ties (e.g. pickup vs
                         dropoff timestamps under meaningless headers), try both
                         and keep the one that satisfies cross-field physics
                         (dropoff after pickup, plausible speed, totals add up).

The global assignment is solved jointly with the Hungarian algorithm, so one
raw column can never be claimed by two canonical columns.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .profiles import ColumnProfile, profile_column, profile_frame, stat_similarity
from .transform import coerce_to_canonical, derive, invariant_rates

ABBREV = {"amt": "amount", "lon": "longitude", "lng": "longitude", "long": "longitude",
          "lat": "latitude", "dist": "distance", "pu": "pickup", "do": "dropoff",
          "start": "pickup", "end": "dropoff", "dt": "time", "datetime": "time", "ts": "time",
          "date": "time", "fwd": "forward", "pax": "passenger", "num": "count", "cnt": "count",
          "location": "zone", "loc": "zone", "charge": "fee", "ratecode": "rate code",
          "vendorid": "vendor", "mi": "", "trip": "", "tpep": "", "lpep": "", "id": "", "the": ""}


def _tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    out = []
    for t in re.split(r"[^A-Za-z0-9]+", s.lower()):
        if not t:
            continue
        t = ABBREV.get(t, t)
        out.extend(x for x in t.split() if x)
    return out


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def name_similarity(src: str, canon: str, aliases: list[str]) -> float:
    cs = _compact(src)
    cands = [canon] + list(aliases or [])
    if any(cs == _compact(a) for a in cands):
        return 1.0
    ts = set(_tokens(src))
    best = 0.0
    for a in cands:
        ta = set(_tokens(a))
        jac = len(ts & ta) / len(ts | ta) if ts | ta else 0.0
        seq = difflib.SequenceMatcher(None, "".join(sorted(ts)), "".join(sorted(ta))).ratio()
        best = max(best, 0.6 * jac + 0.4 * seq)
    return float(best)


@dataclass
class MappingResult:
    mapping: dict                      # canonical -> raw column
    scores: dict                       # canonical -> {score, name, stat, status}
    unmapped_sources: list
    missing_required: list
    missing_optional: list
    ambiguous: list = field(default_factory=list)            # unresolved near-ties
    resolved_by_invariant: list = field(default_factory=list)
    fingerprint: str = ""
    from_memory: bool = False

    @property
    def needs_review(self) -> bool:
        return bool(self.ambiguous) or any(v["status"] == "review" for v in self.scores.values())

    def to_dict(self):
        return asdict(self)


class SchemaRegistry:
    def __init__(self, schema: dict, memory_path: str | Path | None = None,
                 name_weight: float = 0.6, accept: float = 0.70, review: float = 0.45,
                 margin: float = 0.06, uninformative: float = 0.35):
        self.schema = schema
        self.canons = list(schema["columns"].keys())
        self.ref: dict[str, ColumnProfile] = {}
        self.w, self.accept, self.review, self.margin = name_weight, accept, review, margin
        self.uninformative = uninformative
        self.memory_path = Path(memory_path) if memory_path else None
        self.memory = {}
        if self.memory_path and self.memory_path.exists():
            self.memory = json.loads(self.memory_path.read_text())

    # ---------------------------------------------------------------- reference
    def fit_reference(self, *canonical_frames: pd.DataFrame) -> "SchemaRegistry":
        """Learn canonical column fingerprints from trusted, already-canonical data.
        Several frames (e.g. different eras) can be given; later ones win."""
        for f in canonical_frames:
            for c in self.canons:
                if c in f and f[c].notna().any():
                    self.ref[c] = profile_column(f[c], c)
        return self

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def fingerprint(df: pd.DataFrame) -> str:
        sig = json.dumps(sorted((str(c), str(t)) for c, t in df.dtypes.items()))
        return hashlib.sha1(sig.encode()).hexdigest()[:16]

    def score_matrices(self, df: pd.DataFrame, use_name=True, use_stats=True):
        srcs = list(df.columns)
        profs = profile_frame(df)
        N = np.zeros((len(srcs), len(self.canons)))
        T = np.full((len(srcs), len(self.canons)), np.nan)
        vmaps = self.schema.get("value_maps", {})
        for i, s in enumerate(srcs):
            for j, c in enumerate(self.canons):
                N[i, j] = name_similarity(s, c, self.schema["columns"][c].get("aliases", []))
                if c in self.ref:
                    T[i, j] = stat_similarity(profs[s], self.ref[c], vmaps.get(c))
        if not use_stats:
            S = N.copy()
        elif not use_name:
            S = np.nan_to_num(T, nan=0.0)
        else:
            informative = N.max(axis=1, keepdims=True) >= self.uninformative
            Tn = np.nan_to_num(T, nan=0.0)
            blended = np.where(np.isnan(T), N, self.w * N + (1 - self.w) * Tn)
            S = np.where(informative, blended, Tn)
            S = np.where(N >= 0.999, np.maximum(S, 0.9), S)
        return srcs, N, T, S

    # ---------------------------------------------------------------- inference
    def infer(self, df: pd.DataFrame, use_name=True, use_stats=True, use_invariants=True,
              use_memory=True) -> MappingResult:
        fp = self.fingerprint(df)
        if use_memory and fp in self.memory and self.memory[fp].get("approved"):
            m = self.memory[fp]["mapping"]
            res = MappingResult(mapping=m, scores={c: {"score": 1.0, "status": "memory"} for c in m},
                                unmapped_sources=[s for s in df.columns if s not in m.values()],
                                missing_required=[c for c in self.canons if c not in m and
                                                  self.schema["columns"][c].get("required")],
                                missing_optional=[c for c in self.canons if c not in m and
                                                  not self.schema["columns"][c].get("required")],
                                fingerprint=fp, from_memory=True)
            return res

        srcs, N, T, S = self.score_matrices(df, use_name, use_stats)
        rows, cols = linear_sum_assignment(1 - S)
        assign = {self.canons[j]: srcs[i] for i, j in zip(rows, cols) if S[i, j] >= self.review}
        sidx = {s: i for i, s in enumerate(srcs)}
        cidx = {c: j for j, c in enumerate(self.canons)}

        # --- near-tie detection
        pairs = set()
        for c, s in assign.items():
            i, j = sidx[s], cidx[c]
            for j2, c2 in enumerate(self.canons):
                if j2 != j and S[i, j2] >= S[i, j] - self.margin and S[i, j2] >= self.review:
                    pairs.add(tuple(sorted((c, c2))))
        ambiguous, resolved = [], []
        for c1, c2 in sorted(pairs):
            if c1 in assign and c2 in assign:
                s1, s2 = assign[c1], assign[c2]
                # both directions must be near-ties for a swap to be plausible
                if not (S[sidx[s2], cidx[c1]] >= S[sidx[s2], cidx[c2]] - self.margin):
                    continue
                if use_invariants:
                    alt = dict(assign)
                    alt[c1], alt[c2] = s2, s1
                    a, b = self._invariant_score(df, assign), self._invariant_score(df, alt)
                    if b - a > 0.15:
                        assign = alt
                        resolved.append([c1, c2])
                        continue
                    if a - b > 0.15:
                        resolved.append([c1, c2])
                        continue
                ambiguous.append([c1, c2])
            else:
                # competitor unassigned: we cannot test a swap, so just flag it
                if c1 in assign or c2 in assign:
                    ambiguous.append([c1, c2])

        scores = {}
        amb_cols = {c for p in ambiguous for c in p}
        for c, s in assign.items():
            i, j = sidx[s], cidx[c]
            st = "auto" if S[i, j] >= self.accept and c not in amb_cols else "review"
            scores[c] = {"source": s, "score": round(float(S[i, j]), 3), "name": round(float(N[i, j]), 3),
                         "stat": None if np.isnan(T[i, j]) else round(float(T[i, j]), 3), "status": st}
        res = MappingResult(
            mapping=assign, scores=scores,
            unmapped_sources=[s for s in srcs if s not in assign.values()],
            missing_required=[c for c in self.canons if c not in assign and self.schema["columns"][c].get("required")],
            missing_optional=[c for c in self.canons if c not in assign and not self.schema["columns"][c].get("required")],
            ambiguous=ambiguous, resolved_by_invariant=resolved, fingerprint=fp)
        return res

    def _invariant_score(self, df: pd.DataFrame, mapping: dict, n: int = 3000) -> float:
        sample = df.sample(min(len(df), n), random_state=0)
        cdf, _ = coerce_to_canonical(sample, mapping, self.schema)
        r = invariant_rates(derive(cdf, self.schema))
        return float(np.mean(list(r.values()))) if r else 0.0

    # ---------------------------------------------------------------- memory
    def remember(self, res: MappingResult, approved: bool = True, note: str = ""):
        self.memory[res.fingerprint] = {"mapping": res.mapping, "approved": approved, "note": note}
        if self.memory_path:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_path.write_text(json.dumps(self.memory, indent=2))
