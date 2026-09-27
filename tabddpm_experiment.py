"""Spectral analysis of TabDDPM leakage, scored with MeLoMIA-ND, on the existing reconstruction-project releases.

Data: ../data/reconstruction_data/{dataset}/size_1000/sample_0{0..4}/{train.csv, TabDDPM/synth.csv}.
Five DISJOINT training samples, one TabDDPM release each, so every one of the 5,000 records is a member of
exactly one release and a non-member of the other four.

Attack (MeLoMIA-ND, reused from ../MIA_on_bulkRNAseq_CAMDA2026 -- its unconditional ND probe, frozen-epsilon
denoising-loss extraction and feature summariser, unchanged):
  * one probe per release, trained on that release's synthetic data (the attacker's proxy);
  * loss trajectories for all 5,000 records under every probe;
  * the meta-classifier's "shadows" are the OTHER releases, whose membership we know -- real TabDDPM fits on real
    splits, so no synth-shadow layer is needed.  Cross-fitted: release j, record fold f is scored by a model trained
    on releases != j and records not in f, so no record ever scores itself.
  * two independently seeded probes per release (seed a, b) -> two independent attack-noise realisations.
Also a model-free DCR score (-nearest Gower distance to the release) as an attack-independent check.

Per-record signals (per-release z-scored scores):
  s_in(x)   = score of x under its own release
  mu_out(x) = mean score of x under the 4 releases it was not in       (typicality)
  v(x)      = s_in(x) - mu_out(x)                                      (membership signal)
Split for the cross-spectrum: a = (probe seed a, out-releases {first two}), b = (seed b, other two).  This cancels
attack noise and out-release synthesis noise; the one own-release synthesis draw is shared by a and b and cannot be
cancelled with one release per record -- that needs the fixed-pool design.

  python tabddpm_experiment.py features --dataset adult      # GPU
  python tabddpm_experiment.py analyze  --dataset adult --k 10
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path("/home/golobs/data/reconstruction_data")
CAMDA = Path("/home/golobs/MIA_on_bulkRNAseq_CAMDA2026")
WORK = HERE / "workspace" / "tabddpm"
OUT = HERE / "results" / "tabddpm"
N_REL = 5
SEEDS = (11, 22)
N_NOISE = 200
N_FOLDS = 5


# --------------------------------------------------------------------------- data

def load(dataset, size="size_1000"):
    root = DATA / dataset / size
    T = [pd.read_csv(root / f"sample_{j:02d}" / "train.csv") for j in range(N_REL)]
    S = [pd.read_csv(root / f"sample_{j:02d}" / "TabDDPM" / "synth.csv") for j in range(N_REL)]
    records = pd.concat(T, ignore_index=True)
    own = np.repeat(np.arange(N_REL), [len(t) for t in T])
    cols = list(records.columns)
    S = [s[cols] for s in S]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(records[c]) and records[c].nunique() > 20]
    return records, S, own, cols, numeric


class Encoder:
    """Continuous encoding for the diffusion probe: quantile-normal numerics, one-hot categoricals.

    Fitted on the released synthetic data only (what the adversary holds), categories from the public schema.
    """

    def __init__(self, S, records, cols, numeric):
        from sklearn.preprocessing import QuantileTransformer
        self.cols, self.numeric = cols, numeric
        self.cat = [c for c in cols if c not in numeric]
        pooled = pd.concat(S, ignore_index=True)
        self.qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000).fit(pooled[numeric].astype(float))
        self.levels = {c: sorted(set(records[c].astype(str)) | set(pooled[c].astype(str))) for c in self.cat}

    def __call__(self, df):
        parts = [self.qt.transform(df[self.numeric].astype(float))] if self.numeric else []
        for c in self.cat:
            idx = {v: i for i, v in enumerate(self.levels[c])}
            oh = np.zeros((len(df), len(idx)), np.float32)
            oh[np.arange(len(df)), df[c].astype(str).map(idx).to_numpy()] = 1.0
            parts.append(oh * 2 - 1)          # +-1, same scale as the normal-scored numerics
        return np.concatenate(parts, 1).astype(np.float32)


def gower_codes(records, S, cols, numeric):
    """Integer codes for the Gower graph: dense ranks for numerics (ordered), labels for categoricals."""
    pooled = pd.concat([records] + list(S), ignore_index=True)
    codes, ordered, sizes = [], [], []
    for c in cols:
        if c in numeric:
            vals = np.sort(pooled[c].astype(float).unique())
            codes.append(np.searchsorted(vals, pooled[c].astype(float)))
            ordered.append(True)
        else:
            vals = sorted(pooled[c].astype(str).unique())
            codes.append(np.searchsorted(vals, pooled[c].astype(str)))
            ordered.append(False)
        sizes.append(len(vals))
    C = np.stack(codes, 1)
    n = len(records)
    Cs = [C[n + j * len(S[0]): n + (j + 1) * len(S[0])] for j in range(len(S))]
    return C[:n], Cs, np.array(ordered), np.array(sizes)


# --------------------------------------------------------------------------- stage 1: GPU features

def features(dataset, device="cuda"):
    sys.path.insert(0, str(CAMDA))
    import torch
    from mia.generators.nd import NDGenerator
    from mia.attacks.melomia.backends import NDBackend
    from mia.attacks.melomia import features as F

    records, S, own, cols, numeric = load(dataset)
    enc = Encoder(S, records, cols, numeric)
    X_rec = enc(records)
    d = X_rec.shape[1]
    out_dir = WORK / dataset / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{dataset}] {len(records)} records, {d} encoded dims, numeric={numeric}", flush=True)

    for seed in SEEDS:
        be = NDBackend(dataset=dataset, device=device, seed=seed, n_noise_vectors=N_NOISE)
        for j in range(N_REL):
            out = out_dir / f"rel{j}_seed{seed}.npz"
            if out.exists():
                continue
            t0 = time.time()
            X_syn = enc(S[j])
            # MeLoMIA's probe: unconditional, no DP noise, free to overfit; sized for ~100-dim tabular input
            gen = NDGenerator(input_dim=d, hidden_dims=(1024, 1024), epochs=300, batch_size=32,
                              dp_noise_multiplier=0.0, smote_upsample_to=None, unconditional=True,
                              norm_method="none", early_stopping=True, patience=30,
                              device=device, seed=seed * 100 + j, verbose=False)
            gen.fit(X_syn, np.zeros(len(X_syn), np.int64), 1)
            losses, _ = be.extract(gen, X_rec)                  # (n, n_timesteps, n_noise)
            np.savez(out, feats=F.summarize(losses), mean_loss=losses.mean(2),
                     timesteps=np.array(be.sweep_points))
            del gen
            torch.cuda.empty_cache()
            print(f"  probe rel{j} seed{seed}: {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- stage 2: scores

def _zcols(F_):
    return (F_ - F_.mean(0)) / (F_.std(0) + 1e-8)


def melomia_scores(dataset, seed, own):
    """Cross-fitted meta-classifier scores, shape (N_REL, n_records)."""
    from xgboost import XGBClassifier
    fdir = WORK / dataset / "features"
    Fs = [_zcols(np.load(fdir / f"rel{j}_seed{seed}.npz")["feats"]) for j in range(N_REL)]   # per-probe scale removed
    n = len(own)
    rng = np.random.default_rng(seed)
    fold = np.empty(n, int)
    for j in range(N_REL):                                  # folds stratified by own release
        idx = np.where(own == j)[0]
        fold[idx] = rng.permutation(len(idx)) % N_FOLDS
    scores = np.zeros((N_REL, n))
    for j in range(N_REL):
        for f in range(N_FOLDS):
            tr_rel = [k for k in range(N_REL) if k != j]
            keep = fold != f
            Xtr = np.concatenate([Fs[k][keep] for k in tr_rel])
            ytr = np.concatenate([(own[keep] == k).astype(int) for k in tr_rel])
            clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, n_jobs=8, eval_metric="logloss",
                                scale_pos_weight=(ytr == 0).sum() / max(1, (ytr == 1).sum()))
            clf.fit(Xtr, ytr)
            p = np.clip(clf.predict_proba(Fs[j][fold == f])[:, 1], 1e-6, 1 - 1e-6)
            scores[j, fold == f] = np.log(p / (1 - p))
    return scores


def raw_loss_scores(dataset, seed, t_max=20):
    """Meta-free MeLoMIA feature: minus the mean denoising loss at small timesteps (low loss = member-like)."""
    fdir = WORK / dataset / "features"
    out = []
    for j in range(N_REL):
        d = np.load(fdir / f"rel{j}_seed{seed}.npz")
        cols = d["timesteps"] <= t_max
        out.append(-d["mean_loss"][:, cols].mean(1))
    return np.stack(out)


def dcr_scores(codes, codes_syn, ordered, sizes):
    import analysis as A
    return np.stack([-np.concatenate([A.gower(codes[i:i + 500], Cs, ordered, sizes).min(1)
                                      for i in range(0, len(codes), 500)]) for Cs in codes_syn])


# --------------------------------------------------------------------------- stage 3: spectral analysis

def split_out_sets(own):
    """Per record, split its 4 out-releases 2/2 into halves a and b (fixed, balanced)."""
    halves = {}
    for j in range(N_REL):
        others = [k for k in range(N_REL) if k != j]
        halves[j] = (others[:2], others[2:])
    return halves


def signals_split(Za, Zb, own):
    """Signals from two independent attack runs, each paired with disjoint out-release halves."""
    n = len(own)
    halves = split_out_sets(own)
    res = {"a": {}, "b": {}}
    for tag, Z, h in (("a", Za, 0), ("b", Zb, 1)):
        s_in = Z[own, np.arange(n)]
        mu_out = np.array([Z[halves[own[i]][h], i].mean() for i in range(n)])
        res[tag] = {"s_in": s_in, "mu_out": mu_out, "v": s_in - mu_out}
    return res["a"], res["b"]


def zrel(S):
    return (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)


def analyze(dataset, k=10):
    from sklearn.metrics import roc_auc_score
    import analysis as A
    import spectral_core as sc

    records, S, own, cols, numeric = load(dataset)
    codes, codes_syn, ordered, sizes = gower_codes(records, S, cols, numeric)
    cache = WORK / dataset / "scores.npz"
    if cache.exists():
        sc_all = dict(np.load(cache))
    else:
        sc_all = {}
        for s in SEEDS:
            sc_all[f"melomia_{s}"] = melomia_scores(dataset, s, own)
            sc_all[f"rawloss_{s}"] = raw_loss_scores(dataset, s)
        sc_all["dcr"] = dcr_scores(codes, codes_syn, ordered, sizes)
        np.savez(cache, **sc_all)

    D = A.gower(codes, codes, ordered, sizes)
    W = sc.knn_graph_precomputed(D, k=k)
    lam, U, L = sc.laplacian_eig(W)
    np.fill_diagonal(D, np.inf)
    dens = np.sort(D, 1)[:, :k].mean(1)[:, None]
    n_comp = int((lam < 1e-8).sum())
    print(f"[{dataset}] graph gower k={k}: {int(W.sum() / 2)} edges, {n_comp} component(s)", flush=True)

    # attribute reference: how rough is each single column on this graph?
    ref = {}
    for j, c in enumerate(cols):
        x = codes[:, j].astype(float) if ordered[j] else (codes[:, j] == np.bincount(codes[:, j]).argmax()).astype(float)
        if x.std() > 0:
            ref[c] = sc.roughness(x, L)

    rows = []
    attacks = {"melomia": ("melomia_%d", True), "rawloss": ("rawloss_%d", True), "dcr": ("dcr", False)}
    for att, (key, seeded) in attacks.items():
        Za = zrel(sc_all[key % SEEDS[0]] if seeded else sc_all[key])
        Zb = zrel(sc_all[key % SEEDS[1]] if seeded else sc_all[key])
        aucs = [roc_auc_score(own == j, Za[j]) for j in range(N_REL)]
        a, b = signals_split(Za, Zb, own)
        prng = np.random.default_rng(0)
        base = dict(dataset=dataset, attack=att, k=k, auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)))
        for sig in ("mu_out", "v", "s_in"):
            rows.append({**base, "signal": sig, **A.spectral_stats(a[sig], b[sig], U, lam, L, prng)})
        ra, r2 = A.residualize(a["v"], dens)
        rb, _ = A.residualize(b["v"], dens)
        rows.append({**base, "signal": "v_resid_density", "density_R2": r2,
                     **A.spectral_stats(ra, rb, U, lam, L, prng)})
        vr = next(r for r in rows[::-1] if r["signal"] == "v")
        mr = next(r for r in rows[::-1] if r["signal"] == "mu_out")
        print(f"  {att:8s} AUC={base['auc_mean']:.3f}±{base['auc_sd']:.3f}  r(v)={vr['splithalf_r']:+.2f} "
              f"Rx(v)={vr['R_cross']:.2f} Rx(mu_out)={mr['R_cross']:.2f} null={vr['R_cross_null_mean']:.2f}±{vr['R_cross_null_sd']:.3f} "
              f"shares(v)={[round(vr[f'share_{x}'], 2) for x in A.BANDS]} dens R2={r2:.2f}", flush=True)
        np.savez(WORK / dataset / f"signals_{att}.npz", own=own,
                 **{f"{h}_{s}": d[s] for h, d in (("a", a), ("b", b)) for s in d})

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"summary_{dataset}_gower_k{k}.csv", index=False)
    (OUT / f"attribute_roughness_{dataset}_k{k}.json").write_text(json.dumps({c: round(v, 3) for c, v in ref.items()}, indent=1))
    print(f"  attribute roughness range: {min(ref.values()):.2f}-{max(ref.values()):.2f}")
    np.savez(WORK / dataset / f"graph_gower_k{k}.npz", lam=lam, U=U[:, :10], dens=dens)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "analyze"])
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.stage == "features":
        features(a.dataset, a.device)
    else:
        analyze(a.dataset, a.k)
