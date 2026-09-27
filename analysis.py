"""Steps 1-4 of spectral_memorization_experiment_plan.md on the mode-S pool runs.

  python analysis.py                       # every (sdg, eps) with runs on disk
  python analysis.py --metric gower --k 20 # robustness settings

Writes results/summary_{metric}_k{k}.csv and figures to results/figures/.
"""
import argparse
import json
import os
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
warnings.filterwarnings("ignore", category=RuntimeWarning)
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import spectral_core as sc

HERE = Path(__file__).resolve().parent
POOL = HERE / "workspace" / "results" / "pool"
POOL_IDS = HERE / "workspace" / "pool_ids.npy"
OUT = HERE / "results"
SNAKE = Path("/home/golobs/SyntheticData_MIA/SNAKE")

N_PERM = 1000
N_BOOT = 200
BANDS = ["low", "mid", "high"]


# --------------------------------------------------------------------------- data

META = json.load(open(SNAKE / "meta.json"))


def encode(df):
    """Integer codes per column; the representation list gives the value order (ordinal for finite/ordered)."""
    codes = pd.DataFrame(index=df.index)
    for m in META:
        lookup = {str(v): i for i, v in enumerate(m["representation"])}
        codes[m["name"]] = df[m["name"]].astype(str).map(lookup)
    if codes.isna().any().any():
        bad = codes.columns[codes.isna().any()].tolist()
        raise ValueError(f"unmapped values in {bad}")
    return codes.astype(int)


def load_pool_records():
    aux = pd.read_parquet(SNAKE / "base.parquet")
    aux.index = range(aux.shape[0])
    cols = [m["name"] for m in META]
    ordered = np.array([m["type"] == "finite/ordered" for m in META])
    sizes = np.array([len(m["representation"]) for m in META])
    return encode(aux), cols, ordered, sizes


def gower(A, B, ordered, sizes):
    """Gower distance: range-normalized |diff| on ordered columns, mismatch on nominal ones."""
    rng = np.maximum(sizes - 1, 1).astype(float)
    D = np.zeros((A.shape[0], B.shape[0]))
    for j in range(A.shape[1]):
        a, b = A[:, j][:, None], B[:, j][None, :]
        D += np.abs(a - b) / rng[j] if ordered[j] else (a != b)
    return D / A.shape[1]


def hamming(A, B):
    return (A[:, None, :] != B[None, :, :]).mean(-1)


def dcr_raw(synth_file, P, ref, ordered, sizes):
    """Distance-to-closest-record attack (attack-independent of marginals):
    score = d(target, reference aux sample) - d(target, synth), nearest-neighbour Gower distances."""
    S = encode(pd.read_parquet(synth_file)).to_numpy()
    return gower(P, ref, ordered, sizes).min(1) - gower(P, S, ordered, sizes).min(1)


def load_runs(sdg, eps_tag, dcr=None):
    files = sorted((POOL / f"{sdg}_e{eps_tag}").glob("run_???.npz"))
    runs = [np.load(f) for f in files]
    ridx = np.array([int(re.search(r"run_(\d+)", f.name).group(1)) for f in files])
    M = np.stack([r["membership"] for r in runs]).astype(bool)
    out = {"ridx": ridx, "M": M}
    for att in ("mm", "kde"):
        raw = np.stack([r[f"{att}_raw"] for r in runs]).astype(float)
        out[f"{att}_raw"] = raw
        out[f"{att}_auc"] = np.array([float(r[f"{att}_auc"]) for r in runs])
    synths = [f.with_name(f.stem + "_synth.parquet") for f in files]
    if dcr is not None and all(f.exists() for f in synths):
        from sklearn.metrics import roc_auc_score
        out["dcr_raw"] = np.stack([dcr_raw(f, *dcr) for f in synths])
        out["dcr_auc"] = np.array([roc_auc_score(m, s) for m, s in zip(M, out["dcr_raw"])])
    return out


def per_run_z(raw, log=True):
    """z-scored (log) score within each run: the input to activate_3, minus run-level shifts."""
    x = np.log(np.clip(raw, 1e-300, None)) if log else raw
    ok = np.isfinite(x).all(1)
    z = np.full_like(x, np.nan)
    z[ok] = stats.zscore(x[ok], axis=1)
    return z


# --------------------------------------------------------------------------- Step 1

def signals(Z, M, runs_mask):
    """mu_out, mu_in, v, v_std per record from the given subset of runs."""
    Z, M = Z[runs_mask], M[runs_mask]
    keep = np.isfinite(Z).all(1)
    Z, M = Z[keep], M[keep]
    zin = np.where(M, Z, np.nan)
    zout = np.where(~M, Z, np.nan)
    mu_in, mu_out = np.nanmean(zin, 0), np.nanmean(zout, 0)
    sd = np.sqrt((np.nanvar(zin, 0, ddof=1) + np.nanvar(zout, 0, ddof=1)) / 2)
    v = mu_in - mu_out
    return {"mu_out": mu_out, "mu_in": mu_in, "v": v, "v_std": v / sd}


def split_signals(Z, M, ridx, rng=None):
    """Signals from even runs (a) and odd runs (b); with rng, bootstrap-resample runs within each half."""
    halves = []
    for parity in (0, 1):
        idx = np.where(ridx % 2 == parity)[0]
        if rng is not None:
            idx = rng.choice(idx, len(idx), replace=True)
        halves.append(signals(Z[idx], M[idx], np.ones(len(idx), bool)))
    return halves


# --------------------------------------------------------------------------- Steps 2-3

def spectral_stats(va, vb, U, lam, L, perm_rng=None):
    # a record never in (or never out) within a half has no estimate: set it to the mean (no energy)
    ok = np.isfinite(va) & np.isfinite(vb)
    va = np.where(ok, va, np.nanmean(va))
    vb = np.where(ok, vb, np.nanmean(vb))
    res = {
        "n_missing": int((~ok).sum()),
        "splithalf_r": float(np.corrcoef(va, vb)[0, 1]),
        "R_naive": sc.roughness((va + vb) / 2, L),
        "R_cross": sc.cross_roughness(va, vb, L),
        "cross_total": float(np.dot(va - va.mean(), vb - vb.mean())),
    }
    e = sc.band_cross_energy(va, vb, U, lam)
    for b, x in zip(BANDS, e):
        res[f"E_{b}"] = float(x)
        res[f"share_{b}"] = float(x / e.sum())
    if perm_rng is not None:
        n = len(va)
        null_R, null_low, null_high = [], [], []
        for _ in range(N_PERM):
            p = perm_rng.permutation(n)
            null_R.append(sc.cross_roughness(va[p], vb[p], L))
            en = sc.band_cross_energy(va[p], vb[p], U, lam)
            null_low.append(en[0] / en.sum())
            null_high.append(en[2] / en.sum())
        null_R = np.array(null_R)
        res["R_cross_null_mean"] = float(null_R.mean())
        res["R_cross_null_sd"] = float(null_R.std())
        res["p_smoother_than_null"] = float((1 + (null_R <= res["R_cross"]).sum()) / (1 + N_PERM))
        res["p_low_share_gt_null"] = float((1 + (np.array(null_low) >= res["share_low"]).sum()) / (1 + N_PERM))
        res["p_high_share_gt_null"] = float((1 + (np.array(null_high) >= res["share_high"]).sum()) / (1 + N_PERM))
    return res


def residualize(y, X):
    """OLS residual of y on X (with intercept), fitted on finite y; non-finite y stay NaN."""
    X1 = np.column_stack([np.ones(len(y)), X])
    ok = np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X1[ok], y[ok], rcond=None)
    r = y - X1 @ beta
    r2 = 1 - np.nanvar(r) / np.nanvar(y)
    return r, r2


def cross_R(va, vb, L):
    ok = np.isfinite(va) & np.isfinite(vb)
    return sc.cross_roughness(np.where(ok, va, np.nanmean(va)), np.where(ok, vb, np.nanmean(vb)), L)


def build_graph(metric, k, P, ordered, sizes):
    D = gower(P, P, ordered, sizes) if metric == "gower" else hamming(P, P)
    W = sc.knn_graph_precomputed(D, k=k)
    lam, U, L = sc.laplacian_eig(W)
    n_comp = int((lam < 1e-8).sum())
    return D, W, lam, U, L, n_comp


def density_features(P, pop, ordered, sizes, metric, k=10):
    """Mean distance to the k nearest records: within the pool, and within a population sample."""
    dist = (lambda A, B: gower(A, B, ordered, sizes)) if metric == "gower" else hamming
    Dp = dist(P, P)
    np.fill_diagonal(Dp, np.inf)
    pool_knn = np.sort(Dp, 1)[:, :k].mean(1)
    pop_knn = np.concatenate([np.sort(dist(P[i:i + 50], pop), 1)[:, :k].mean(1) for i in range(0, len(P), 50)])
    return pool_knn, pop_knn


# --------------------------------------------------------------------------- driver

def analyze(metric, k, n_pop=20000):
    codes, cols, ordered, sizes = load_pool_records()
    pool_ids = np.load(POOL_IDS)
    P = codes.loc[pool_ids].to_numpy()
    pop_rng = np.random.default_rng(1)
    rest = np.setdiff1d(codes.index.values, pool_ids)
    pop = codes.loc[pop_rng.choice(rest, n_pop, replace=False)].to_numpy()
    ref = pop[:1000]   # DCR reference sample: same size as synth, disjoint from the pool

    D, W, lam, U, L, n_comp = build_graph(metric, k, P, ordered, sizes)
    pool_knn, pop_knn = density_features(P, pop, ordered, sizes, metric)
    dens = np.column_stack([pool_knn, pop_knn])
    print(f"graph {metric} k={k}: {int(W.sum() / 2)} edges, {n_comp} component(s), lam2={lam[n_comp]:.3f}")

    rows, per_record = [], {}
    for d in sorted(POOL.glob("*_e*")):
        sdg, eps_tag = d.name.split("_e")
        R = load_runs(sdg, eps_tag, dcr=(P, ref, ordered, sizes))
        if len(R["ridx"]) < 4:
            continue
        for att in ("mm", "kde", "dcr"):
            if f"{att}_raw" not in R:
                continue
            Z = per_run_z(R[f"{att}_raw"], log=(att != "dcr"))
            if np.isnan(Z).all():
                continue
            a, b = split_signals(Z, R["M"], R["ridx"])
            full = signals(Z, R["M"], np.ones(len(R["ridx"]), bool))
            per_record[(sdg, eps_tag, att)] = full
            base = dict(sdg=sdg, eps=float(eps_tag), attack=att, n_runs=len(R["ridx"]),
                        auc_mean=float(np.nanmean(R[f"{att}_auc"])), metric=metric, k=k)
            prng = np.random.default_rng(0)
            for sig in ("mu_out", "v", "v_std"):
                rows.append({**base, "signal": sig, **spectral_stats(a[sig], b[sig], U, lam, L, prng)})
            # H0: remove what local density explains, then look again
            ra, r2a = residualize(a["v"], dens)
            rb, _ = residualize(b["v"], dens)
            _, r2 = residualize(full["v"], dens)
            rows.append({**base, "signal": "v_resid_density", "density_R2": r2,
                         **spectral_stats(ra, rb, U, lam, L, prng)})
            # H1 paired contrast with a run bootstrap: R_cross(v) - R_cross(mu_out)
            brng = np.random.default_rng(1)
            diffs = []
            for _ in range(N_BOOT):
                ba, bb = split_signals(Z, R["M"], R["ridx"], brng)
                diffs.append(cross_R(ba["v"], bb["v"], L) - cross_R(ba["mu_out"], bb["mu_out"], L))
            v_row = next(r for r in rows[::-1] if r["signal"] == "v")
            v_row["H1_dR_boot_lo"], v_row["H1_dR_boot_hi"] = np.nanpercentile(diffs, [2.5, 97.5])
            print(f"  {sdg:5s} e{eps_tag:>7s} {att:3s} runs={len(R['ridx']):3d} AUC={base['auc_mean']:.3f} "
                  f"r(v)={rows[-3]['splithalf_r']:+.2f} Rx(v)={rows[-3]['R_cross']:.2f} "
                  f"Rx(mu_out)={rows[-4]['R_cross']:.2f} shares(v)={[round(rows[-3][f'share_{x}'], 2) for x in BANDS]} "
                  f"dens R2={r2:.2f}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"summary_{metric}_k{k}.csv", index=False)
    np.savez(OUT / f"graph_{metric}_k{k}.npz", lam=lam, U=U, pool_knn=pool_knn, pop_knn=pop_knn)
    return df, per_record, dict(lam=lam, U=U, L=L, pool_knn=pool_knn, pop_knn=pop_knn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="hamming", choices=["hamming", "gower"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    df, per_record, G = analyze(a.metric, a.k)
    if a.figures:
        import figures
        figures.make_all(df, per_record, G, tag=f"{a.metric}_k{a.k}")
