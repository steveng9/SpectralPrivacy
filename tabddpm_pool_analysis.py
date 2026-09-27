"""Spectral analysis of the fixed-pool TabDDPM releases (tabddpm_pool.py).

Attacks, all scored for the 1,000 pool records under every release:
  dcr      model-free: -(nearest Gower distance from the record to the release)
  melomia  MeLoMIA-ND: diffusion probe on each release -> denoising-loss trajectories -> XGB meta-classifier trained on
           OTHER releases (labels known), cross-fitted over run folds x record folds so no (run, record) scores itself
  rawloss  meta-free MeLoMIA feature: -(mean small-t denoising loss)
Per-record signals (per-run z-scored) and the split-half cross-spectrum use even vs odd runs, exactly as the MST
grid did: each half has its own in-releases, so release-to-release synthesis noise cancels as well as attack noise.

  python tabddpm_pool_analysis.py features --gpu 1     # MeLoMIA probes (GPU), resumable
  python tabddpm_pool_analysis.py analyze [--k 10]
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd

import tabddpm_experiment as TE
import tabddpm_pool as TP

WORK = TP.WORK
OUT = TE.HERE / "results" / "tabddpm_pool"
SEED = 11


def load_releases():
    R = TP.records()
    pid = np.load(WORK / "pool_ids.npy")
    runs = sorted(int(p.name[4:]) for p in WORK.glob("run_*") if (p / "synth.csv").exists())
    S = [pd.read_csv(WORK / f"run_{r:03d}" / "synth.csv")[list(R.columns)] for r in runs]
    M = np.stack([np.load(WORK / f"run_{r:03d}" / "membership.npy") for r in runs]).astype(bool)
    return R, pid, np.array(runs), S, M


def features(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(TE.CAMDA))
    import torch
    from mia.generators.nd import NDGenerator
    from mia.attacks.melomia.backends import NDBackend
    from mia.attacks.melomia import features as F

    R, pid, runs, S, _ = load_releases()
    cols = list(R.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(R[c]) and R[c].nunique() > 20]
    enc = TE.Encoder(S, R, cols, numeric)
    X_pool = enc(R.iloc[pid])
    be = NDBackend(dataset="adult", device="cuda", seed=SEED, n_noise_vectors=TE.N_NOISE)
    fdir = WORK / "melomia_features"
    fdir.mkdir(exist_ok=True)
    for r, s in zip(runs, S):
        out = fdir / f"run_{r:03d}.npz"
        if out.exists():
            continue
        t0 = time.time()
        X_syn = enc(s)
        gen = NDGenerator(input_dim=X_pool.shape[1], hidden_dims=(1024, 1024), epochs=300, batch_size=32,
                          dp_noise_multiplier=0.0, smote_upsample_to=None, unconditional=True, norm_method="none",
                          early_stopping=True, patience=30, device="cuda", seed=SEED * 1000 + r, verbose=False)
        gen.fit(X_syn, np.zeros(len(X_syn), np.int64), 1)
        losses, _ = be.extract(gen, X_pool)
        np.savez(out, feats=F.summarize(losses), mean_loss=losses.mean(2), timesteps=np.array(be.sweep_points))
        del gen
        torch.cuda.empty_cache()
        print(f"probe run {r}: {time.time() - t0:.0f}s", flush=True)


def melomia_scores(runs, M, n_run_folds=4, n_rec_folds=5):
    from xgboost import XGBClassifier
    Fs = np.stack([TE._zcols(np.load(WORK / "melomia_features" / f"run_{r:03d}.npz")["feats"]) for r in runs])
    n_runs, n = M.shape
    rng = np.random.default_rng(0)
    run_fold = rng.permutation(n_runs) % n_run_folds
    rec_fold = rng.permutation(n) % n_rec_folds
    scores = np.zeros((n_runs, n))
    for g in range(n_run_folds):
        for f in range(n_rec_folds):
            tr_runs, te_runs = run_fold != g, run_fold == g
            tr_rec, te_rec = rec_fold != f, rec_fold == f
            Xtr = Fs[tr_runs][:, tr_rec].reshape(-1, Fs.shape[2])
            ytr = M[tr_runs][:, tr_rec].reshape(-1).astype(int)
            clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, n_jobs=8, eval_metric="logloss")
            clf.fit(Xtr, ytr)
            Xte = Fs[te_runs][:, te_rec]
            p = np.clip(clf.predict_proba(Xte.reshape(-1, Fs.shape[2]))[:, 1], 1e-6, 1 - 1e-6)
            scores[np.ix_(te_runs, te_rec)] = np.log(p / (1 - p)).reshape(te_runs.sum(), te_rec.sum())
    return scores


def rawloss_scores(runs, t_max=20):
    out = []
    for r in runs:
        d = np.load(WORK / "melomia_features" / f"run_{r:03d}.npz")
        out.append(-d["mean_loss"][:, d["timesteps"] <= t_max].mean(1))
    return np.stack(out)


def analyze(k=10):
    from sklearn.metrics import roc_auc_score
    import analysis as A
    import spectral_core as sc

    R, pid, runs, S, M = load_releases()
    cols = list(R.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(R[c]) and R[c].nunique() > 20]
    codes_all, codes_syn, ordered, sizes = TE.gower_codes(R, S, cols, numeric)
    P = codes_all[pid]
    pop = np.delete(codes_all, pid, axis=0)
    print(f"{len(runs)} releases, pool {len(pid)}, runs per record in/out: "
          f"{M.sum(0).min()}-{M.sum(0).max()} / {(~M).sum(0).min()}-{(~M).sum(0).max()}", flush=True)

    scores = {"dcr": np.stack([-A.gower(P, Cs, ordered, sizes).min(1) for Cs in codes_syn])}
    if all((WORK / "melomia_features" / f"run_{r:03d}.npz").exists() for r in runs):
        scores["melomia"] = melomia_scores(runs, M)
        scores["rawloss"] = rawloss_scores(runs)
    np.savez(WORK / "scores.npz", runs=runs, M=M, **scores)

    D = A.gower(P, P, ordered, sizes)
    W = sc.knn_graph_precomputed(D, k=k)
    lam, U, L = sc.laplacian_eig(W)
    np.fill_diagonal(D, np.inf)
    pool_knn = np.sort(D, 1)[:, :k].mean(1)
    pop_knn = np.concatenate([np.sort(A.gower(P[i:i + 100], pop, ordered, sizes), 1)[:, :k].mean(1)
                              for i in range(0, len(P), 100)])
    dens = np.column_stack([pool_knn, pop_knn])
    ref = {}
    for j, c in enumerate(cols):
        x = P[:, j].astype(float) if ordered[j] else (P[:, j] == np.bincount(P[:, j]).argmax()).astype(float)
        if x.std() > 0:
            ref[c] = round(sc.roughness(x, L), 3)
    print(f"graph gower k={k}: {int(W.sum() / 2)} edges, {int((lam < 1e-8).sum())} component(s); "
          f"attribute roughness {min(ref.values()):.2f}-{max(ref.values()):.2f}", flush=True)

    rows, per_record = [], {}
    for att, sc_ in scores.items():
        Z = (sc_ - sc_.mean(1, keepdims=True)) / sc_.std(1, keepdims=True)
        aucs = [roc_auc_score(m, z) for m, z in zip(M, Z)]
        a, b = A.split_signals(Z, M, runs)
        full = A.signals(Z, M, np.ones(len(runs), bool))
        per_record[att] = full
        prng = np.random.default_rng(0)
        base = dict(attack=att, k=k, n_runs=len(runs), auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)))
        for sig in ("mu_out", "v", "v_std"):
            rows.append({**base, "signal": sig, **A.spectral_stats(a[sig], b[sig], U, lam, L, prng)})
        ra, _ = A.residualize(a["v"], dens)
        rb, _ = A.residualize(b["v"], dens)
        _, r2 = A.residualize(full["v"], dens)
        rows.append({**base, "signal": "v_resid_density", "density_R2": r2, **A.spectral_stats(ra, rb, U, lam, L, prng)})
        # H1 contrast with a run bootstrap: R_cross(v) - R_cross(mu_out); and the same for the density residual
        brng = np.random.default_rng(1)
        d_v, d_res = [], []
        for _ in range(A.N_BOOT):
            ba, bb = A.split_signals(Z, M, runs, brng)
            d_v.append(A.cross_R(ba["v"], bb["v"], L) - A.cross_R(ba["mu_out"], bb["mu_out"], L))
            d_res.append(A.cross_R(A.residualize(ba["v"], dens)[0], A.residualize(bb["v"], dens)[0], L) - 1.0)
        for r_ in rows[-4:]:
            if r_["signal"] == "v":
                r_["H1_dR_boot_lo"], r_["H1_dR_boot_hi"] = np.nanpercentile(d_v, [2.5, 97.5])
            if r_["signal"] == "v_resid_density":
                r_["Rresid_minus1_boot_lo"], r_["Rresid_minus1_boot_hi"] = np.nanpercentile(d_res, [2.5, 97.5])
        vr = next(r_ for r_ in rows[::-1] if r_["signal"] == "v")
        mr = next(r_ for r_ in rows[::-1] if r_["signal"] == "mu_out")
        rr = rows[-1]
        print(f"  {att:8s} AUC={base['auc_mean']:.3f}±{base['auc_sd']:.3f}  r½(v)={vr['splithalf_r']:+.2f} "
              f"Rx(v)={vr['R_cross']:.2f} [Δ vs μ_out CI {vr['H1_dR_boot_lo']:+.2f},{vr['H1_dR_boot_hi']:+.2f}] "
              f"Rx(μ_out)={mr['R_cross']:.2f}  shares(v)={[round(vr[f'share_{x}'], 2) for x in A.BANDS]}  "
              f"dens R²={r2:.2f}  resid: r½={rr['splithalf_r']:+.2f} Rx={rr['R_cross']:.2f} "
              f"[Rx-1 CI {rr['Rresid_minus1_boot_lo']:+.2f},{rr['Rresid_minus1_boot_hi']:+.2f}] "
              f"high={rr['share_high']:.2f} (p={rr['p_high_share_gt_null']:.3f})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"summary_gower_k{k}.csv", index=False)
    np.savez(OUT / f"per_record_k{k}.npz", lam=lam, U=U[:, :10], pool_knn=pool_knn, pop_knn=pop_knn,
             **{f"{a}_{s}": v for a, d in per_record.items() for s, v in d.items()})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "analyze"])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    if a.stage == "features":
        features(a.gpu)
    else:
        analyze(a.k)
