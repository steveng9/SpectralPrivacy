# Is memorization high-frequency? First experiment on MAMA-MIA

*Plan, 2026-09-25. Code base: `PycharmProjects/SyntheticData_MIA` (commit 4213618). Runs go on the server, where the focal-point artifacts and results live.*

## The question

Build a graph over the records, where each record is linked to its most similar records. Write each record's measured privacy leakage on its node. Is that leakage **smooth** over the graph (similar records leak alike, which is population structure) or **rough** (neighbours leak very differently, which is individual memorization)? And does DP remove the rough part first?

- **H1.** A record's *membership signal* (the attack score when it's in the training data minus the score when it's out) is rougher on the graph than its *typicality signal* (the score when it's out).
- **H2.** As ε shrinks, the high-frequency part of the membership signal falls first and the low-frequency part stays. That residue would be the SoK's plateau.
- **H0, the baseline to beat.** Leakage is just local density: outliers leak, and that's all. The spectral view has to say something that the "kNN distance → risk" story (DOMIAS, the risk-equalized DP synthetic data work in 2602.10232) doesn't.

## Why the saved results aren't enough

`attack_experiment_A` saves `MM_ROC = (membership, activated_scores)` for each run, but not `target_ids`. Scores can't be matched back to records. Also, each run draws only 32 fresh targets (n = 1000, 16 of them members), so no record is seen both in and out often enough to estimate its own leakage. We need a new mode.

## Step 0: new experiment mode `S` (the only code change)

Use a LiRA-style fixed pool:

1. Draw a **pool of T = 500 records** from SNAKE aux once. Save their indices (`pool_ids.npy`).
2. For each run r = 1..R (R = 64):
   - Draw a membership mask m_r: exactly half the pool (250) are members.
   - Train set = the 250 members + 750 records from aux not in the pool (n = 1000 as in Exp. A).
   - Synthesize, then attack **all 500 pool records** as targets.
   - Save `m_r`, the **raw** score `A / num_queries_used` (before `activate_3`), and the activated score. Save them for both the MAMA-MIA score and the KDE score (`kde_get_ma`), which gives us an attack-independent check.
3. Each record is then a member in ~32 runs and a non-member in ~32 runs.

Sweep: SDG ∈ {mst, priv, gsd} × ε ∈ {0.1, 0.32, 1, 3.16, 10, 31.62, 100, 1000}. Reuse the existing focal-point files `FP4_{sdg}_e{ε}_n1000_snake_*`, so there's no new shadow modelling.

Change `sample_experimental_data` → add `sample_pool_experiment(cfg, aux, pool_ids, rng)`. `score_attack` already works on any target set; just return the raw `A` as well.

**Note:** half of each training set now comes from the pool, versus 1.6% in Exp. A. That's standard for per-record estimation, but say so in the paper.

**Pilot first:** MST only, ε ∈ {1, 1000}, R = 16. Check that per-record estimates aren't pure noise (the split-half correlation below should be > 0) before running the grid.

## Step 1: per-record signals

For record i, over the runs:

- `mu_out[i]` = mean score when i is a non-member. This is the **typicality** signal: how "in-distribution" the record looks.
- `mu_in[i]` = mean score when i is a member.
- `v[i] = mu_in[i] − mu_out[i]`. This is the **membership signal**: this record's own leakage. Also compute the standardized version, `(mu_in − mu_out) / pooled sd`.
- Compute every signal twice, once from even-numbered runs and once from odd (`v_a`, `v_b`). This is needed for Step 3.

## Step 2: graph and frequencies

- **Encode the pool.** Use ordinal codes for `finite/ordered` columns and one-hot for `finite` columns (types are in `SNAKE/meta.json`, 15 columns).
- **Graph.** Symmetric kNN, k = 10, Hamming distance. Robustness: k ∈ {5, 20}, and Gower distance.
- **Laplacian.** Normalized, L = I − D^{-1/2} W D^{-1/2}, full eigendecomposition (500 × 500 is instant). Small eigenvalues = slow/smooth; large = fast/rough.
- **Graph Fourier transform.** `v_hat = Uᵀ(v − mean(v))`, which measures how much of v lines up with each eigenvector.
- **One-number summary: roughness** `R(v) = vᵀLv / vᵀv`. It's about 0 for a signal that's constant within clusters and about 1 for white noise (normalized L).

The tested code for all of this is in `spectral_core.py` (appendix below). On a toy example, a smooth signal gives R = 0.06, white noise gives 1.03, and the split-half cross-energy recovers the smooth signal's low band through heavy noise.

## Step 3: controls (these make or break it)

1. **Estimation-noise trap (most important).** Per-record estimates from finite runs carry noise, and noise is white. White noise looks high-frequency, so a naive spectrum of v would "confirm" H1 for free. The fix: use the **cross-spectrum** `Σ_band v_hat_a · v_hat_b` from the two halves of the runs. Independent noise cancels in expectation; only real structure survives.
2. **Permutation null.** Shuffle v over the nodes 1000× and recompute R and the band energies. H1 needs v to differ from the shuffles *and* from mu_out.
3. **Density baseline (H0).** Compute local density (mean distance to the k nearest neighbours). Regress v on density, then run Steps 2–4 on the **residual**. If the residual has no band structure, the spectral story adds nothing beyond "outliers leak".
4. **Attack-independence.** Repeat with the KDE attack's scores. MAMA-MIA scores are built from marginals, which may make them smooth over attributes by construction. If both attacks show the same spectral pattern, it belongs to the synthesizer, not the attack.

## Step 4: figures

- **F1 (H1).** Band energy (low/mid/high, by eigenvalue tertile; cross-spectrum) for mu_out vs v at ε = 1000, per SDG. Prediction: mu_out concentrates low; v spreads high.
- **F2 (H2).** Band energy of v vs ε, one line per band, per SDG. Also plot roughness R(v) vs ε next to AUC vs ε. Prediction: the high band drops first, and the low band persists where AUC plateaus.
- **F3 (H0).** v vs local density scatter, with the band energies of the residual inset.
- **F4 (intuition).** An eigenmap: place each pool record at (u₂, u₃), colour by v. Is the leakage in patches or salt-and-pepper?

## Kill criteria

Any one of these ends it, and all can be seen in the pilot or the first week of the grid:

- The split-half cross-spectrum of v is flat across bands, the same as the permutation null.
- v is almost fully explained by density, and the residual has no structure.
- The band shares of v don't change with ε.

## If it works: where the theory goes

- **Roughness as a Rayleigh quotient.** Roughness R(v) is a Rayleigh quotient, the same object behind Weyl/Courant–Fischer. The natural next claim: a DP mechanism with noise scale σ can only suppress leakage components above some frequency, which sets a floor on what stays attackable.
- **Mechanism.** It leads to a mechanism idea (idea #2): spend the noise on the rough part only.
- **Follow-ups.** The same pipeline runs on scMAMA-MIA with donors as nodes (the cell graph is already standard in that field). The fixed pool also gives per-record risk scores for free, which connects to the effective-resistance question in the research-vision note.

## Appendix: `spectral_core.py`

```python
import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from sklearn.neighbors import NearestNeighbors

def knn_graph(X, k=10):
    """Symmetric, unweighted kNN graph over rows of X (already encoded)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="hamming").fit(X)
    _, idx = nn.kneighbors(X)
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    W = csr_matrix((np.ones_like(rows, dtype=float), (rows, cols)), shape=(n, n))
    return ((W + W.T) > 0).astype(float)

def laplacian_eig(W):
    """Normalized Laplacian L = I - D^-1/2 W D^-1/2; eigenvalues ascending = slow -> fast."""
    d = np.asarray(W.sum(1)).ravel()
    Dm = diags(1 / np.sqrt(d))
    L = (identity(W.shape[0]) - Dm @ W @ Dm).toarray()
    lam, U = np.linalg.eigh(L)
    return lam, U, L

def gft(v, U):
    """Graph Fourier transform: how much of signal v lines up with each eigenvector."""
    return U.T @ (v - v.mean())

def roughness(v, L):
    """Rayleigh quotient v'Lv / v'v: 0 = perfectly smooth, higher = more high-frequency."""
    v = v - v.mean()
    return float(v @ L @ v / (v @ v))

def band_cross_energy(va, vb, U, lam, n_bands=3):
    """Energy per band from two independent estimates of v; white estimation noise cancels."""
    a, b = gft(va, U), gft(vb, U)
    edges = np.quantile(lam, np.linspace(0, 1, n_bands + 1))
    band = np.clip(np.searchsorted(edges, lam, side="right") - 1, 0, n_bands - 1)
    return np.array([np.sum(a[band == j] * b[band == j]) for j in range(n_bands)])
```
