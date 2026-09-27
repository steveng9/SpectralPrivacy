"""Fixed-pool (LiRA-style) TabDDPM releases on adult, for per-record leakage that is estimable across releases.

Records: the 5,000 training records of reconstruction_data/adult/size_1000 (samples 00-04).
  pool   = 1,000 of them (seeded), the records we measure;
  filler = the other 4,000, used only to pad training sets.
Run r: exactly 500 pool records are members (mask shared seed scheme with pool_experiment.py) + 500 filler
records -> TabDDPM (the reconstruction project's generator and default config, 200k iterations) -> 1,000 synthetic rows.
Each pool record is then a member in ~R/2 runs and a non-member in ~R/2 runs.

  python tabddpm_pool.py run --runs 32 --concurrent 8 --gpu 1      # resumable
  python tabddpm_pool.py one <r> <out_dir> [iterations]            # (internal) one release
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path("/home/golobs/data/reconstruction_data")
RECON = Path("/home/golobs/Reconstruction")
WORK = HERE / "workspace" / "tabddpm_pool"
PY = sys.executable
POOL_SIZE, N_MEMBERS, N_TRAIN = 1000, 500, 1000
POOL_SEED, MASK_SEED = 20260926, 7


def records(dataset="adult"):
    root = DATA / dataset / "size_1000"
    return pd.concat([pd.read_csv(root / f"sample_{j:02d}" / "train.csv") for j in range(5)], ignore_index=True)


def pool_ids(n):
    return np.sort(np.random.default_rng(POOL_SEED).choice(n, POOL_SIZE, replace=False))


def membership_mask(r):
    rng = np.random.default_rng([MASK_SEED, r])
    m = np.zeros(POOL_SIZE, dtype=np.int8)
    m[rng.choice(POOL_SIZE, N_MEMBERS, replace=False)] = 1
    return m


def one(r, out_dir, iterations=None, dataset="adult"):
    """Train one TabDDPM release for run r (runs in its own process so GPU memory is freed on exit)."""
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("RECON_DATA_ROOT", str(DATA))
    sys.path.insert(0, str(RECON))
    from sdg import get_sdg

    out_dir = Path(out_dir)
    R = records(dataset)
    pid = pool_ids(len(R))
    m = membership_mask(r)
    filler_ids = np.setdiff1d(np.arange(len(R)), pid)
    rng = np.random.default_rng([MASK_SEED, 1000 + r])
    fill = rng.choice(filler_ids, N_TRAIN - N_MEMBERS, replace=False)
    train = pd.concat([R.iloc[pid[m == 1]], R.iloc[fill]]).sample(frac=1, random_state=r).reset_index(drop=True)
    meta = json.loads((DATA / dataset / "meta.json").read_text())
    cfg = {"workspace_dir": str(out_dir / "tabddpm_ws")}
    if iterations:
        cfg["iterations"] = int(iterations)
    t0 = time.time()
    synth = get_sdg("TabDDPM")(train, meta, **cfg)
    synth.to_csv(out_dir / "synth.tmp.csv", index=False)
    np.save(out_dir / "membership.npy", m)
    np.save(out_dir / "filler_ids.npy", fill)
    import shutil
    shutil.rmtree(out_dir / "tabddpm_ws", ignore_errors=True)
    os.replace(out_dir / "synth.tmp.csv", out_dir / "synth.csv")
    print(f"run {r}: {len(synth)} rows in {time.time() - t0:.0f}s", flush=True)


def run(runs, concurrent, gpu, iterations=None, dataset="adult"):
    WORK.mkdir(parents=True, exist_ok=True)
    R = records(dataset)
    np.save(WORK / "pool_ids.npy", pool_ids(len(R)))
    todo = [r for r in range(runs) if not (WORK / f"run_{r:03d}" / "synth.csv").exists()]
    print(f"{len(todo)} TabDDPM releases to train, {concurrent} at a time on GPU {gpu}", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "PYTHONWARNINGS": "ignore", "PYKEOPS_VERBOSE": "0",
           "RECON_DATA_ROOT": str(DATA)}

    def launch(r):
        d = WORK / f"run_{r:03d}"
        d.mkdir(exist_ok=True)
        cmd = ["nice", "-n", "5", PY, str(Path(__file__)), "one", str(r), str(d)] + ([str(iterations)] if iterations else [])
        with open(d / "log.txt", "w") as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(RECON)).returncode
        return r, rc

    with ThreadPoolExecutor(concurrent) as ex:
        futs = [ex.submit(launch, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            r, rc = f.result()
            print(f"[{i}/{len(todo)}] run {r} {'ok' if rc == 0 else f'FAILED rc={rc} (see run_{r:03d}/log.txt)'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--concurrent", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=None)
    a = ap.parse_args()
    run(a.runs, a.concurrent, a.gpu, a.iterations)
