"""Run all three experiments and write results/ (JSON + CSV).

    python experiments/run_all.py            # full run (~6 min on a laptop)
    python experiments/run_all.py --quick    # 1 seed, smoke test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftguard.benchmarks import downstream_benchmark, fault_benchmark, schema_benchmark  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(exist_ok=True)
S = (0,) if a.quick else (0, 1, 2, 3, 4)
S3 = (0,) if a.quick else (0, 1, 2)

e1 = schema_benchmark(S)
e1.to_csv(out / "e1_schema.csv", index=False)
e2, e2s = fault_benchmark(S)
e2.to_csv(out / "e2_batches.csv", index=False)
e2s.to_csv(out / "e2_summary.csv", index=False)
e3 = downstream_benchmark(S3)
e3.to_csv(out / "e3_downstream.csv", index=False)

clean = e2[e2["fault"] == "clean"]
meta = {"seeds_e1_e2": list(S), "seeds_e3": list(S3),
        "clean_batches": int(len(clean)), "false_alarms": int(clean["alarm"].sum())}
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
