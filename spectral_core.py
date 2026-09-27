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

def knn_graph_precomputed(D, k=10):
    """Symmetric, unweighted kNN graph from a precomputed distance matrix."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(D)
    _, idx = nn.kneighbors(D)
    n = D.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    W = csr_matrix((np.ones_like(rows, dtype=float), (rows, cols)), shape=(n, n))
    return ((W + W.T) > 0).astype(float)

def cross_roughness(va, vb, L):
    """Roughness from two independent estimates: va'L vb / va'vb. White estimation noise cancels in expectation."""
    va, vb = va - va.mean(), vb - vb.mean()
    return float(va @ L @ vb / (va @ vb))

def band_index(lam, n_bands=3):
    edges = np.quantile(lam, np.linspace(0, 1, n_bands + 1))
    return np.clip(np.searchsorted(edges, lam, side="right") - 1, 0, n_bands - 1)

def band_cross_shares(va, vb, U, lam, n_bands=3):
    """band_cross_energy normalized to sum to 1 (share of the signal's reliable energy per band)."""
    e = band_cross_energy(va, vb, U, lam, n_bands)
    return e / e.sum()

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # toy: 400 records, 15 categorical columns, two latent clusters
    z = rng.integers(0, 2, 400)
    X = np.where(rng.random((400, 15)) < 0.8, z[:, None] * 3 + rng.integers(0, 3, (400, 15)) % 3, rng.integers(0, 6, (400, 15)))
    W = knn_graph(X, k=10)
    lam, U, L = laplacian_eig(W)
    smooth = z + 0.1 * rng.standard_normal(400)          # "population" signal
    rough = rng.standard_normal(400)                     # white "memorization-like" signal
    print("eigenvalue range", lam[:3].round(3), lam[-1].round(3))
    print("roughness smooth:", round(roughness(smooth, L), 3), " rough:", round(roughness(rough, L), 3))
    va, vb = smooth + rng.standard_normal(400), smooth + rng.standard_normal(400)
    print("cross-band energy (low, mid, high):", band_cross_energy(va, vb, U, lam).round(1))
