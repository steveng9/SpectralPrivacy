"""Mode S: LiRA-style fixed-pool MAMA-MIA experiment on SNAKE (see spectral_memorization_experiment_plan.md).

Two phases, both resumable (finished units are skipped):
  shadow : focal-point shadow modelling, n_FP_shadowruns=15 shadow SDG fits per (sdg, eps),
           written as FP4_{sdg}_e{eps}_n1000_snake in the same {tuple: count} format as the old code.
  pool   : R runs per (sdg, eps). Each run: half the fixed 500-record pool are members, plus 750
           non-pool aux records -> train (n=1000) -> synthesize -> MAMA-MIA + KDE (DOMIAS) scores
           for all 500 pool records. One .npz per run.

Usage:
  python pool_experiment.py pool_ids
  python pool_experiment.py run --sdgs mst priv --eps 1 1000 --runs 16 --workers 16
"""
import argparse
import os
import pickle
import time
import traceback
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "workspace" / "results" / "pool"
POOL_IDS = HERE / "workspace" / "pool_ids.npy"

EPSILONS = [0.1, 0.32, 1, 3.16, 10, 31.62, 100, 1000]
POOL_SIZE = 500
N_MEMBERS = 250
N_FP_SHADOWRUNS = 15
POOL_SEED = 20260925
MASK_SEED = 7


def fo(eps):
    return "{0:.2f}".format(eps)


def run_dir(sdg, eps):
    return RESULTS / f"{sdg}_e{fo(eps)}"


def membership_mask(r):
    """Run r's mask, shared across all (sdg, eps) so configs are paired."""
    rng = np.random.default_rng([MASK_SEED, r])
    m = np.zeros(POOL_SIZE, dtype=np.int8)
    m[rng.choice(POOL_SIZE, N_MEMBERS, replace=False)] = 1
    return m


def task_seed(*parts):
    return int(np.random.SeedSequence([zlib.crc32(str(p).encode()) for p in parts]).generate_state(1)[0])


# --------------------------------------------------------------------------- workers

_W = {}


def _init_worker():
    import mamamia_bridge as B
    cfg, aux, columns, meta = B.load_snake()
    _W.update(B=B, cfg=cfg, aux=aux, columns=columns, meta=meta)


def _cast(synth):
    return synth.astype({"age": "int", "ownchild": "int", "hoursut": "int"})


def _synthesize(sdg, train, eps):
    B, meta, columns = _W["B"], _W["meta"], _W["columns"]
    if sdg == "mst":
        gen = B.mst.MST(dataset=train[columns], metadata=meta, size=len(train), epsilon=eps)
    elif sdg == "priv":
        gen = B.privbayes.PRIVBAYES(dataset=train[columns], metadata=meta, size=len(train), epsilon=eps)
    else:
        raise ValueError(sdg)
    gen.run()
    return gen


def shadow_task(sdg, eps, i):
    """One shadow fit on a random aux sample; returns its focal points (cliques / conditionals)."""
    np.random.seed(task_seed("shadow", sdg, eps, i))
    aux, columns = _W["aux"], _W["columns"]
    for attempt in range(5):
        try:
            gen = _synthesize(sdg, aux[columns].sample(n=_W["B"].N_TRAIN), eps)
            break
        except ValueError:
            if attempt == 4:
                raise
    fps = []
    if sdg == "mst":
        for fp in gen.cliques:
            fps.append(tuple(sorted(np.array(fp).tolist())))
    else:
        for fp in gen.conditionals:
            attrs = np.array(fp).tolist()
            fps.append(tuple([attrs[0]] + sorted(attrs[1:])))
    return fps


def _kde_raw(synth, targets, pool_ids, seed):
    """DOMIAS-KDE ratio p_synth / p_aux at each target (as kde_get_ma in conduct_attacks.py)."""
    from scipy import stats
    B, cfg, aux = _W["B"], _W["cfg"], _W["aux"]
    aux_sample = aux[~aux.index.isin(pool_ids)].sample(n=synth.shape[0], random_state=seed)
    enc = lambda d: B.encode_data_all_numeric(cfg, d, keep_indeces=False).to_numpy().T
    es, et, ea = enc(synth), enc(targets), enc(aux_sample)
    return stats.gaussian_kde(es).evaluate(et) / (stats.gaussian_kde(ea).evaluate(et) + 1e-20)


def _mm_raw(sdg, eps, synth, targets, fps):
    """MAMA-MIA raw score A / num_queries_used, before activate_3 (same math as custom_*_attack)."""
    import pandas as pd
    B, cfg, aux = _W["B"], _W["cfg"], _W["aux"]
    A = np.zeros(len(targets))
    n_used = np.zeros(len(targets))
    threshold = B.determine_weight_threshold(cfg, eps, fps)
    default_val = 1e-10
    for fp, weight in fps.items():
        if weight < threshold:
            continue
        cols = list(fp)
        if sdg == "mst":
            D_synth = synth[cols].value_counts(normalize=True)
            D_aux = aux[cols].value_counts(normalize=True)
            A += np.array([weight * D_synth.get(tuple(v), default=default_val) / D_aux.get(tuple(v), default=default_val)
                           for v in targets[cols].values])
        else:
            child, parents = cols[0], cols[1:]
            D_synth = synth.groupby(parents if parents else (lambda _: True))[child].value_counts(normalize=True)
            D_aux = aux.groupby(parents if parents else (lambda _: True))[child].value_counts(normalize=True)
            empty = pd.Series(dtype=np.float64)
            r = []
            for child_val, *pv in targets[[child] + parents].values:
                pv = tuple(pv)
                s = D_synth.get(pv or True, default=empty).get(child_val, default=default_val)
                a = max(D_aux.get(pv or True, default=empty).get(child_val, default=default_val), default_val)
                r.append(s / a)
            A += np.array(r)
        n_used += 1
    return A / np.maximum(1, n_used)


def pool_task(sdg, eps, r, pool_ids, fps):
    out = run_dir(sdg, eps) / f"run_{r:03d}.npz"
    if out.exists():
        return str(out), None
    import pandas as pd
    B, aux, columns = _W["B"], _W["aux"], _W["columns"]
    seed = task_seed("pool", sdg, eps, r)
    np.random.seed(seed)
    t0 = time.time()

    m = membership_mask(r)
    targets = aux.loc[pool_ids]                      # pool_ids sorted -> aligned with m
    members = targets[m == 1]
    filler = aux[~aux.index.isin(pool_ids)].sample(n=B.N_TRAIN - N_MEMBERS, random_state=seed % (2**31))
    train = pd.concat([filler, members]).sample(frac=1, random_state=seed % (2**31))

    for attempt in range(5):
        try:
            synth = _cast(_synthesize(sdg, train, eps).output)
            break
        except ValueError:
            if attempt == 4:
                raise

    mm_raw = _mm_raw(sdg, eps, synth, targets, fps)
    try:
        kde_raw = _kde_raw(synth, targets, pool_ids, seed % (2**31))
    except np.linalg.LinAlgError:
        kde_raw = np.full(len(targets), np.nan)
    mm_act = B.activate_3(mm_raw)
    kde_act = B.activate_3(kde_raw) if np.all(np.isfinite(kde_raw)) and np.all(kde_raw > 0) else np.full_like(kde_raw, np.nan)
    auc = lambda s: B.area_under_curve(m, s) if np.all(np.isfinite(s)) else np.nan
    out.parent.mkdir(parents=True, exist_ok=True)
    # keep the synthetic data so other attacks (e.g. distance-to-closest-record) can be scored offline
    synth[columns].reset_index(drop=True).to_parquet(out.with_name(f"run_{r:03d}_synth.parquet"))
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, membership=m, mm_raw=mm_raw, mm_act=mm_act, kde_raw=kde_raw, kde_act=kde_act,
             mm_auc=auc(mm_act), kde_auc=auc(kde_act), seed=seed, seconds=time.time() - t0)
    os.replace(tmp, out)
    return str(out), (auc(mm_act), auc(kde_act))


# --------------------------------------------------------------------------- driver

def make_pool_ids():
    import mamamia_bridge as B
    _, aux, _, _ = B.load_snake()
    rng = np.random.default_rng(POOL_SEED)
    ids = np.sort(rng.choice(aux.index.values, POOL_SIZE, replace=False))
    POOL_IDS.parent.mkdir(parents=True, exist_ok=True)
    np.save(POOL_IDS, ids)
    print(f"saved {len(ids)} pool ids -> {POOL_IDS}")


def _fp_file(sdg, eps):
    import mamamia_bridge as B
    return B.fp_path(sdg, eps)


def run(args):
    import mamamia_bridge  # noqa: F401  (creates workspace, fits encoders once before forking)
    mamamia_bridge.load_snake()
    pool_ids = np.load(POOL_IDS)
    configs = [(s, e) for s in args.sdgs for e in args.eps]

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        # phase 1: shadow modelling for configs missing focal points
        need = [(s, e) for s, e in configs if not _fp_file(s, e).exists()]
        if need:
            futs = {ex.submit(shadow_task, s, e, i): (s, e) for s, e in need for i in range(N_FP_SHADOWRUNS)}
            counts = {c: Counter() for c in need}
            for f in as_completed(futs):
                counts[futs[f]].update(f.result())
            for (s, e), c in counts.items():
                fps = dict(sorted(c.items(), key=lambda x: -x[1]))
                with open(_fp_file(s, e), "wb") as fh:
                    pickle.dump(fps, fh)
                print(f"[shadow] {s} e{fo(e)}: {len(fps)} focal points, top weight {max(fps.values())}", flush=True)

        # phase 2: pool runs
        futs = {}
        for s, e in configs:
            with open(_fp_file(s, e), "rb") as fh:
                fps = pickle.load(fh)
            for r in range(args.runs):
                if not (run_dir(s, e) / f"run_{r:03d}.npz").exists():
                    futs[ex.submit(pool_task, s, e, r, pool_ids, fps)] = (s, e, r)
        print(f"[pool] {len(futs)} runs queued on {args.workers} workers", flush=True)
        done = 0
        for f in as_completed(futs):
            s, e, r = futs[f]
            done += 1
            try:
                _, aucs = f.result()
                print(f"[pool {done}/{len(futs)}] {s} e{fo(e)} r{r}: AUC mm={aucs[0]:.3f} kde={aucs[1]:.3f}", flush=True)
            except Exception:
                print(f"[pool {done}/{len(futs)}] {s} e{fo(e)} r{r} FAILED\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool_ids")
    q = sub.add_parser("run")
    q.add_argument("--sdgs", nargs="+", default=["mst", "priv"])
    q.add_argument("--eps", nargs="+", type=float, default=EPSILONS)
    q.add_argument("--runs", type=int, default=64)
    q.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.cmd == "pool_ids":
        make_pool_ids()
    else:
        run(a)
