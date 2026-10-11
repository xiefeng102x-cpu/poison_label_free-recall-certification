"""Shared helpers: chunked execution with per-cell seeding, so results do not depend
on how the grid is split across processes (needed on machines that limit the run
time of a single command).  Output directory can be redirected with P14CERT_OUT."""
import argparse, csv, os, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def out_dir():
    d = pathlib.Path(os.environ.get("P14CERT_OUT", ROOT / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse(extra_positional=None):
    ap = argparse.ArgumentParser()
    if extra_positional:
        ap.add_argument("what", nargs="?", default=extra_positional[0], choices=extra_positional)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    return ap.parse_args()


def mine(idx, a):
    return idx % a.nparts == a.part


def cell_rng(base_seed, idx):
    # spawn_key keeps every cell's stream distinct from the plain default_rng(base_seed)
    # stream (default_rng([base, 0]) would coincide with default_rng(base)).
    return np.random.default_rng(np.random.SeedSequence(base_seed, spawn_key=(idx,)))


def write(name, rows, a):
    if not rows:
        return None
    if a.nparts == 1:
        fn = out_dir() / f"{name}.csv"
    else:
        (out_dir() / "parts").mkdir(exist_ok=True)
        fn = out_dir() / "parts" / f"{name}.part{a.part:03d}of{a.nparts:03d}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return fn
