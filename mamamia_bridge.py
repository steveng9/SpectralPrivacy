"""Import MAMA-MIA code from ../SyntheticData_MIA without modifying that repo.

The old code base hard-codes a Mac DATA_DIR and uses paths relative to its own root,
so we chdir there, patch DATA_DIR, and point artifacts at this repo's workspace.
"""
import os
import sys
from pathlib import Path

MIA_ROOT = Path("/home/golobs/SyntheticData_MIA")
HERE = Path(__file__).resolve().parent
WORK = HERE / "workspace"          # DATA_DIR for the old code: needs SNAKE/ and experiment_artifacts/
FP_DIR = WORK / "experiment_artifacts" / "focalpoints"

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

FP_DIR.mkdir(parents=True, exist_ok=True)
if not (WORK / "SNAKE").exists():
    (WORK / "SNAKE").symlink_to(MIA_ROOT / "SNAKE")

os.chdir(MIA_ROOT)
sys.path.insert(0, str(MIA_ROOT))
sys.path.append(str(MIA_ROOT / "reprosyn-main/src/reprosyn/methods/mbi/"))

import encode_data  # noqa: E402

encode_data.DATA_DIR = str(WORK) + "/"

import mst  # noqa: E402
import privbayes  # noqa: E402
import util  # noqa: E402
from util import Config, get_data, determine_weight_threshold, activate_3, area_under_curve  # noqa: E402
from encode_data import encode_data_all_numeric  # noqa: E402

N_TRAIN = 1000


def fo(eps):
    return "{0:.2f}".format(eps)


def fp_path(sdg, eps, n=N_TRAIN, data="snake"):
    return FP_DIR / f"FP4_{sdg}_e{fo(eps)}_n{n}_{data}"


_dump_artifact = encode_data.dump_artifact


def _dump_if_missing(artifact, name):
    # get_data() re-fits and re-dumps deterministic encoders on every call; with many
    # concurrent workers that races (a reader sees a half-written pickle). Write once only.
    if not (WORK / "experiment_artifacts" / name).exists():
        _dump_artifact(artifact, name)


encode_data.dump_artifact = _dump_if_missing


def load_snake():
    cfg = Config("snake", set_MI=False, train_size=N_TRAIN, overlapping_aux=True, check_arbitrary_fps=False)
    _, aux, columns, meta, _ = get_data(cfg)
    return cfg, aux, columns, meta
