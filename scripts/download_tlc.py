"""Download real NYC TLC yellow-taxi files + taxi-zone shapes.

    python scripts/download_tlc.py --months 2023-01 2023-02
    python scripts/download_tlc.py --months 2009-01 2012-06 2015-03 2016-09 2019-05 2023-04   # one per era

Files: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
Note: the TLC now serves the whole history as Parquet; the old-era column
names still differ, which is exactly what the registry has to handle.
"""
import argparse
import urllib.request
from pathlib import Path

BASE = "https://d37ci6vzurychx.cloudfront.net"

ap = argparse.ArgumentParser()
ap.add_argument("--months", nargs="+", required=True)
ap.add_argument("--out", default="data/raw")
ap.add_argument("--zones", action="store_true", help="also fetch taxi_zones.zip + lookup csv")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
urls = [f"{BASE}/trip-data/yellow_tripdata_{m}.parquet" for m in a.months]
if a.zones:
    urls += [f"{BASE}/misc/taxi_zones.zip", f"{BASE}/misc/taxi+_zone_lookup.csv"]
for u in urls:
    dest = out / u.rsplit("/", 1)[1]
    if dest.exists():
        print("have", dest)
        continue
    print("get ", u)
    urllib.request.urlretrieve(u, dest)
print("done ->", out)
