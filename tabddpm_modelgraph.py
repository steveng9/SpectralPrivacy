"""Is the leftover per-record TabDDPM leakage structured in the MODEL's geometry rather than in Gower geometry?

The fixed-pool analysis (tabddpm_pool_analysis.py) found per-record leakage v whose density residual is reliable but
~white on a Gower kNN graph. This script rebuilds the record graph the way TabDDPM sees records and reruns the same
spectral statistics on it. Note how the Reconstruction pipeline feeds TabDDPM: every column is declared discrete,
label-encoded as a STRING (so fnlwgt / capital-gain / hours-per-week are sorted lexicographically: "20" sits next to
"2", not "19"), then quantile-normalised and diffused as a continuous 15-d vector.

Reference models: 4 TabDDPMs (same pipeline + default config as the releases) trained on disjoint 1,000-row slices of
the 4,000 FILLER records, so no pool record is ever in a reference model and graphs carry no membership information.

Graphs over the 1,000 pool records (kNN, Euclidean):
  ddpm_input   the model's input encoding x0 (lexicographic label codes -> quantile-normal), all refs concatenated
  ddpm_hidden  the denoiser's last hidden layer (512-d) at several noise levels, mean over fixed noise draws, PCA-64
  ddpm_loss    the record's denoising-loss profile across noise levels (log MSE), mean over refs
  gower        the original graph, for a like-for-like comparison with the same density covariates
Density covariates for the residual: Gower within-pool and population kNN distance + within-pool kNN distance in
each model space.

  python tabddpm_modelgraph.py refs --gpu 1        # train the 4 reference models (resumable, ~25 min)
  python tabddpm_modelgraph.py embed --gpu 1       # embeddings + loss profiles of the pool records
  python tabddpm_modelgraph.py analyze [--k 10]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd

import tabddpm_pool as TP

REFS = TP.WORK / "reference_models"
OUT = TP.HERE / "results" / "tabddpm_pool"
N_REFS = 4
T_GRID = [0, 5, 20, 50, 100, 200, 400, 700, 1000, 1500]
N_NOISE = 16


def ref_train_ids(j):
    R = TP.records()
    filler = np.setdiff1d(np.arange(len(R)), TP.pool_ids(len(R)))
    return np.sort(np.random.default_rng([TP.MASK_SEED, 5000]).permutation(filler)[j * TP.N_TRAIN:(j + 1) * TP.N_TRAIN])


def one_ref(j, out_dir):
    """Train reference model j on filler only; keep its checkpoint (runs in its own process)."""
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(TP.RECON))
    from sdg import get_sdg

    out_dir = Path(out_dir)
    R = TP.records()
    ids = ref_train_ids(j)
    meta = json.loads((TP.DATA / "adult" / "meta.json").read_text())
    t0 = time.time()
    get_sdg("TabDDPM")(R.iloc[ids].reset_index(drop=True), meta, workspace_dir=str(out_dir / "tabddpm_ws"))
    np.save(out_dir / "train_ids.npy", ids)
    print(f"ref {j}: trained in {time.time() - t0:.0f}s", flush=True)


def ckpt(j):
    return REFS / f"ref_{j}" / "tabddpm_ws" / "sdg_run" / "models" / "None_data_ckpt.pkl"


def refs(gpu):
    REFS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in range(N_REFS) if not (REFS / f"ref_{j}" / "train_ids.npy").exists()]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "PYTHONWARNINGS": "ignore", "RECON_DATA_ROOT": str(TP.DATA)}

    def launch(j):
        d = REFS / f"ref_{j}"
        d.mkdir(exist_ok=True)
        with open(d / "log.txt", "w") as log:
            rc = subprocess.run(["nice", "-n", "5", sys.executable, str(Path(__file__)), "one", str(j), str(d)],
                                env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(TP.RECON)).returncode
        print(f"ref {j} {'ok' if rc == 0 else f'FAILED rc={rc}'}", flush=True)

    with ThreadPoolExecutor(N_REFS) as ex:
        list(ex.map(launch, todo))


def encode_for(model, df):
    """Encode records exactly as the reference model's training data was encoded (string label codes ->
    quantile-normal). Values unseen by the model get the half-way code between their sorted neighbours."""
    cols = model["df_info"]["cat_cols"]
    codes = np.zeros((len(df), len(cols)))
    unseen = 0
    for i, c in enumerate(cols):
        classes = model["label_encoders"][i].classes_
        vals = df[c].to_numpy(dtype=np.str_)
        pos = np.searchsorted(classes, vals)
        hit = (pos < len(classes)) & (classes[np.minimum(pos, len(classes) - 1)] == vals)
        codes[:, i] = np.where(hit, pos, pos - 0.5)
        unseen += (~hit).sum()
    return model["dataset"].num_transform.transform(codes).astype(np.float32), unseen / codes.size


def embed(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(TP.RECON))
    import sdg.tabddpm_method  # noqa: F401  (puts the TabDDPM pipeline on sys.path)
    import torch
    from midst_models.single_table_TabDDPM.complex_pipeline import CustomUnpickler

    R = TP.records()
    P = R.iloc[TP.pool_ids(len(R))].reset_index(drop=True)
    for j in range(N_REFS):
        out = REFS / f"ref_{j}" / "pool_embedding.npz"
        if out.exists():
            continue
        with open(ckpt(j), "rb") as f:
            model = CustomUnpickler(f).load()
        diff = model["diffusion"].to("cuda").eval()
        net = diff._denoise_fn
        x0, unseen = encode_for(model, P)
        x0 = torch.tensor(x0, device="cuda")
        grab = {}
        h = net.mlp.blocks[-1].register_forward_hook(lambda m, i, o: grab.__setitem__("h", o.detach()))
        g = torch.Generator(device="cuda").manual_seed(1234)
        hidden, loss = [], []
        with torch.no_grad():
            for t in T_GRID:
                tt = torch.full((len(x0),), t, device="cuda", dtype=torch.long)
                hs, ls = 0, []
                for _ in range(N_NOISE):
                    eps = torch.randn(x0.shape, generator=g, device="cuda")
                    out_ = net(diff.gaussian_q_sample(x0, tt, noise=eps), tt)
                    hs = hs + grab["h"] / N_NOISE
                    ls.append(((out_ - eps) ** 2).mean(1))
                hidden.append(hs.cpu().numpy())
                loss.append(torch.stack(ls).mean(0).cpu().numpy())
        h.remove()
        np.savez(out, x0=x0.cpu().numpy(), hidden=np.stack(hidden), loss=np.stack(loss, 1), t=np.array(T_GRID),
                 unseen_frac=unseen)
        print(f"ref {j}: embedded pool, {100 * unseen:.1f}% of cell values unseen by this model", flush=True)
        del model, diff, net
        torch.cuda.empty_cache()


def _z(X):
    return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)


def model_spaces():
    from sklearn.decomposition import PCA
    E = [np.load(REFS / f"ref_{j}" / "pool_embedding.npz") for j in range(N_REFS)]
    x_in = np.hstack([_z(e["x0"]) for e in E])
    hid = np.hstack([_z(e["hidden"][i]) for e in E for i in range(len(T_GRID))])
    x_hid = PCA(64, random_state=0).fit_transform(hid)
    x_loss = _z(np.mean([np.log(e["loss"]) for e in E], 0))
    return {"ddpm_input": x_in, "ddpm_hidden": x_hid, "ddpm_loss": x_loss}


def analyze(k=10):
    import analysis as A
    import spectral_core as sc
    import tabddpm_experiment as TE
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    from tabddpm_pool_analysis import graph_rows, load_releases

    R, pid, runs, S, M = load_releases()
    sv = np.load(TP.WORK / "scores.npz")
    assert np.array_equal(sv["runs"], runs)
    scores = {a: sv[a] for a in ("dcr", "melomia", "rawloss") if a in sv}
    cols = list(R.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(R[c]) and R[c].nunique() > 20]
    codes_all, _, ordered, sizes = TE.gower_codes(R, S[:1], cols, numeric)
    G = np.load(OUT / f"per_record_k{k}.npz")
    dists = {"gower": A.gower(codes_all[pid], codes_all[pid], ordered, sizes)}
    dists.update({name: cdist(X, X) for name, X in model_spaces().items()})

    def knn_dist(D):
        D = D.copy()
        np.fill_diagonal(D, np.inf)
        return np.sort(D, 1)[:, :k].mean(1)

    own = {name: knn_dist(D) for name, D in dists.items()}
    rows = []
    for name, D in dists.items():
        W = sc.knn_graph_precomputed(D, k=k)
        lam, U, L = sc.laplacian_eig(W)
        shared = W.multiply(sc.knn_graph_precomputed(dists["gower"], k=k)).sum() / W.sum()
        dens = np.column_stack([G["pool_knn"], G["pop_knn"], own[name]])
        print(f"graph {name} k={k}: {int(W.sum() / 2)} edges, {int((lam < 1e-8).sum())} component(s), "
              f"{shared:.0%} of edges shared with gower; spearman(own kNN dist, gower pop kNN)="
              f"{spearmanr(own[name], G['pop_knn'])[0]:+.2f}", flush=True)
        r_, _ = graph_rows(scores, M, runs, L, U, lam, dens, k, graph=name)
        rows += r_
    pd.DataFrame(rows).to_csv(OUT / f"summary_modelgraph_k{k}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one_ref(int(sys.argv[2]), sys.argv[3])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["refs", "embed", "analyze"])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    {"refs": lambda: refs(a.gpu), "embed": lambda: embed(a.gpu), "analyze": lambda: analyze(a.k)}[a.stage]()
