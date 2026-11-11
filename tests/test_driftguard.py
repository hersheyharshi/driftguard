import numpy as np
import pandas as pd
import pytest

from driftguard.contracts import Contracts
from driftguard.faults import inject
from driftguard.pipeline import DriftGuard, NaivePipeline
from driftguard.profiles import clean_numeric
from driftguard.registry import SchemaRegistry, name_similarity
from driftguard.synth import ERAS, TripGenerator, era_truth, to_raw, zone_centroids
from driftguard.transform import ZoneResolver, coerce_to_canonical, derive, load_schema

SCHEMA = load_schema()


@pytest.fixture(scope="module")
def gen():
    return TripGenerator(3000, seed=11)


@pytest.fixture(scope="module")
def guard(gen):
    """A DriftGuard instance after 14 trusted warm-up days, plus the next day's truth."""
    reg = SchemaRegistry(SCHEMA).fit_reference(gen.day("2023-12-01"))
    dg = DriftGuard(reg, SCHEMA, zones=ZoneResolver(zone_centroids()))
    days = pd.date_range("2024-01-01", periods=15)
    truths = [gen.day(d) for d in days]
    for i in range(14):
        dg.process(to_raw(truths[i]), f"w{i}", days[i], warmup=True)
    dg.end_warmup()
    return dg, truths, days


def _fresh(guard_tuple, n_extra=0):
    dg, truths, days = guard_tuple
    return dg, to_raw(truths[13]), to_raw(truths[14]), days[14]


# ---------------------------------------------------------------- profiles
def test_clean_numeric_repairs_formatting():
    v, n = clean_numeric(pd.Series(["$1,234.50", "12.5", None, "abc"]))
    assert v.iloc[0] == 1234.5 and v.iloc[1] == 12.5 and np.isnan(v.iloc[2]) and np.isnan(v.iloc[3])
    assert n == 1


def test_name_similarity_aliases_and_fuzzy():
    assert name_similarity("Airport_fee", "airport_fee", ["airport_fee"]) == 1.0
    assert name_similarity("PULocationID", "pu_zone", ["pulocationid"]) == 1.0
    assert name_similarity("pickup_zone", "pu_zone", []) > 0.8
    assert name_similarity("col_07", "fare_amount", ["fare_amt"]) < 0.35


# ---------------------------------------------------------------- registry
@pytest.mark.parametrize("era", list(ERAS))
def test_registry_maps_every_named_era_exactly(gen, era):
    reg = SchemaRegistry(SCHEMA).fit_reference(gen.day("2024-02-01"))
    raw = to_raw(gen.day("2024-02-02"), era)
    m = reg.infer(raw)
    assert m.mapping == ERAS[era]["names"]
    assert not m.missing_required


def test_registry_never_silently_wrong_on_meaningless_headers(gen):
    reg = SchemaRegistry(SCHEMA).fit_reference(gen.day("2024-02-01"))
    raw = to_raw(gen.day("2024-02-03"))
    ob, lab = inject("obfuscated_headers", raw, np.random.default_rng(0))
    m = reg.infer(ob)
    flagged = {c for p in m.ambiguous for c in p} | {c for c, v in m.scores.items() if v["status"] != "auto"}
    for c, r in ERAS["2023"]["names"].items():
        if c in m.mapping and m.mapping[c] != lab["truth"][r]:
            assert c in flagged, f"{c} silently mis-mapped"
    # timestamps: identical stats, only invariants can order them
    assert m.mapping["pickup_ts"] == lab["truth"]["tpep_pickup_datetime"]


def test_value_vocabulary_is_normalised(gen):
    raw = to_raw(gen.day("2024-02-04"), "2009")
    cdf, rep = coerce_to_canonical(raw, {"payment_type": "Payment_Type"}, SCHEMA)
    assert set(cdf["payment_type"].dropna()) <= {"card", "cash", "no_charge", "dispute"}
    assert not rep.unknown_vocab


def test_zone_resolver_recovers_zones_from_coordinates(gen):
    t = gen.day("2024-02-05")
    z = ZoneResolver(zone_centroids()).resolve(t["pickup_lon"], t["pickup_lat"])
    assert (z == t["pu_zone"]).mean() > 0.98


# ---------------------------------------------------------------- contracts
def test_contracts_quarantine_with_reasons(gen):
    t = era_truth(gen.day("2024-02-06"), "2023").head(100).copy()
    t.loc[0, "fare_amount"] = -5
    t.loc[1, "dropoff_ts"] = t.loc[1, "pickup_ts"] - pd.Timedelta(minutes=3)
    clean, q, stats = Contracts().apply(derive(t, SCHEMA))
    assert len(q) == 2 and len(clean) == 98
    assert "fare_non_negative" in q["reasons"].iloc[0]


# ---------------------------------------------------------------- sentinel + repairs
@pytest.mark.parametrize("fault,check,status", [
    ("unit_km", "unit_change", "ACCEPTED_REPAIRED"),
    ("cents", "unit_change", "ACCEPTED_REPAIRED"),
    ("clock_shift", "clock_shift", "ACCEPTED_REPAIRED"),
    ("column_swap", "column_swap", "ACCEPTED_REPAIRED"),
    ("duplicate_batch", "duplicate_batch", "REJECTED"),
    ("partial_duplicates", "partial_duplicates", "ACCEPTED_REPAIRED"),
    ("null_burst", "null_burst", None),
    ("type_corruption", "type_repair", "ACCEPTED"),
    ("dropped_column", "missing_column", None),
])
def test_sentinel_attributes_cause(guard, fault, check, status):
    dg, prev, today, d = _fresh(guard)
    # copy so tests don't contaminate each other's reference
    import copy
    dg = copy.deepcopy(dg)
    raw, _ = inject(fault, today, np.random.default_rng(3), previous=prev)
    rep = dg.process(raw, f"t-{fault}", d)
    assert check in rep.checks(), rep.findings
    if status:
        assert rep.status == status


def test_unit_repair_restores_truth(guard):
    import copy
    dg, prev, today, d = _fresh(guard)
    dg = copy.deepcopy(dg)
    raw, _ = inject("unit_km", today, np.random.default_rng(3))
    rep = dg.process(raw, "t-repair", d)
    loaded = dg.wh.read("trips")
    loaded = loaded[loaded.batch_id == "t-repair"]["trip_distance_mi"].median()
    truth = guard[1][14]["trip_distance_mi"].median()
    assert abs(loaded / truth - 1) < 0.02
    assert rep.repairs[0]["divided_by"] == pytest.approx(1.609344)


def test_clean_batch_raises_no_alarm(guard):
    import copy
    dg, prev, today, d = _fresh(guard)
    dg = copy.deepcopy(dg)
    rep = dg.process(today, "t-clean", d)
    assert rep.status == "ACCEPTED"
    assert not [f for f in rep.findings if f["severity"] in ("warn", "hold", "reject")]


def test_lineage_records_every_step(guard):
    import copy
    dg, prev, today, d = _fresh(guard)
    dg = copy.deepcopy(dg)
    rep = dg.process(today, "t-lineage", d)
    steps = [s["step"] for s in rep.lineage]
    assert steps[:3] == ["map_schema", "coerce", "derive"] and "load" in steps
    assert all(s["digest_out"] for s in rep.lineage if s["rows_out"])


def test_naive_pipeline_crashes_on_unknown_headers(gen):
    raw = to_raw(gen.day("2024-02-07"))
    ob, _ = inject("obfuscated_headers", raw, np.random.default_rng(1))
    assert NaivePipeline(SCHEMA).process(ob, "x", "2024-02-07").status == "CRASHED"


def test_stable_vocabulary_oddity_is_not_an_alarm_but_a_jump_is(gen):
    """Real 2023+ files carry payment_type=0 every day: that's baseline, not drift."""
    import copy
    reg = SchemaRegistry(SCHEMA).fit_reference(gen.day("2023-12-01"))
    dg = DriftGuard(reg, SCHEMA)
    days = pd.date_range("2024-03-04", periods=16)
    rng = np.random.default_rng(0)

    def odd(d, share):
        raw = to_raw(gen.day(d))
        raw.loc[rng.random(len(raw)) < share, "payment_type"] = 0
        return raw
    for i in range(14):
        dg.process(odd(days[i], 0.01), f"v{i}", days[i], warmup=True)
    dg.end_warmup()
    calm = copy.deepcopy(dg).process(odd(days[14], 0.01), "calm", days[14])
    jump = copy.deepcopy(dg).process(odd(days[15], 0.10), "jump", days[15])
    assert "vocabulary_drift" not in calm.checks()
    assert "vocabulary_drift" in jump.checks()


def test_agg_warehouse_feeds_forecast_panel(gen):
    from driftguard.forecast import hourly_panel
    from driftguard.warehouse import AggWarehouse
    t = era_truth(gen.day("2024-03-01"), "2023")
    wh = AggWarehouse()
    wh.append("trips", t.assign(batch_id="a"))
    days = pd.date_range("2024-03-01", periods=1)
    full = hourly_panel(t, days, 60)["count"]
    agg = hourly_panel(wh.read("trips"), days, 60)["count"]
    assert np.allclose(full, agg)
