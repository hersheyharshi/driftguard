"""Row-level data contracts with a quarantine (bad rows are kept, never deleted)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transform import load_yaml


class Contracts:
    def __init__(self, rules: list[dict] | None = None):
        self.rules = rules if rules is not None else load_yaml("contracts.yaml")["rules"]

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Returns (clean_rows, quarantined_rows_with_reasons, stats)."""
        env = {c: df[c] for c in df.columns}
        reasons = pd.Series("", index=df.index, dtype=object)
        bad = np.zeros(len(df), bool)
        stats = {"violations": {}, "tagged": {}, "skipped": []}
        for r in self.rules:
            try:
                ok = eval(r["expr"], {"__builtins__": {}, "np": np}, env)  # trusted config
            except (KeyError, NameError):
                stats["skipped"].append(r["name"])   # column not present in this era
                continue
            ok = pd.Series(ok, index=df.index).fillna(False).astype(bool).to_numpy()
            n_bad = int((~ok).sum())
            if r.get("severity", "quarantine") == "quarantine":
                stats["violations"][r["name"]] = n_bad
                if n_bad:
                    reasons[~ok] = reasons[~ok] + r["name"] + ";"
                    bad |= ~ok
            else:
                stats["tagged"][r["name"]] = n_bad
        q = df[bad].copy()
        q["reasons"] = reasons[bad].str.rstrip(";")
        return df[~bad].copy(), q, stats
