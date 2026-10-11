"""Merge chunked outputs <name>.partXXXofYYY.csv into <name>.csv (sorted by cell index)."""
import sys, pathlib, re
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _chunk import out_dir

d = out_dir()
groups = {}
for f in sorted((d / "parts").glob("*.part*of*.csv")):
    m = re.match(r"(.+)\.part(\d+)of(\d+)\.csv$", f.name)
    groups.setdefault((m.group(1), int(m.group(3))), []).append(f)
for (name, n), files in groups.items():
    if len(files) != n:
        print(f"{name}: only {len(files)}/{n} parts present, not merged"); continue
    df = pd.concat([pd.read_csv(f) for f in files]).sort_values("cell").reset_index(drop=True)
    df.to_csv(d / f"{name}.csv", index=False)
    print(f"{name}: merged {n} parts, {len(df)} rows")
