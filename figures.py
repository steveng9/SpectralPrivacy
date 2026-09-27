"""Figures F1-F4 of the plan. Called from analysis.py --figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "figures"
BANDS = ["low", "mid", "high"]
RELIABLE_R = 0.15   # split-half correlation of v below which per-record estimates are treated as noise
BAND_COLORS = ["#86b6ef", "#2a78d6", "#104281"]          # ordinal blue ramp: slow -> fast
SIG_COLORS = {"mu_out": "#2a78d6", "v": "#eb6834", "v_resid_density": "#1baf7a"}
SIG_LABELS = {"mu_out": "typicality  μ_out", "v": "membership  v", "v_resid_density": "v | density residual"}
ATT_LABELS = {"mm": "MAMA-MIA", "dcr": "DCR", "kde": "KDE (DOMIAS)"}
SDG_LABELS = {"mst": "MST", "priv": "PrivBayes", "gsd": "Private-GSD"}
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#f0efec", "#b02a2a"])
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, name, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}_{tag}.{ext}")
    plt.close(fig)


def f1_band_energy(df, tag, eps=1000.0):
    """F1 (H1): cross-spectrum band shares of mu_out vs v at high eps, per SDG x attack."""
    d = df[(df.eps == eps) & df.signal.isin(["mu_out", "v"])]
    sdgs = [s for s in SDG_LABELS if s in d.sdg.unique()]
    atts = [a for a in ATT_LABELS if a in d.attack.unique()]
    fig, axes = plt.subplots(len(atts), len(sdgs), figsize=(3.2 * len(sdgs), 2.4 * len(atts)), squeeze=False, sharey=True)
    x = np.arange(3)
    for i, a in enumerate(atts):
        for j, s in enumerate(sdgs):
            ax = axes[i, j]
            for o, sig in zip((-0.2, 0.2), ("mu_out", "v")):
                row = d[(d.sdg == s) & (d.attack == a) & (d.signal == sig)]
                if row.empty:
                    continue
                r = row.iloc[0]
                ax.bar(x + o, [r[f"share_{b}"] for b in BANDS], 0.38, color=SIG_COLORS[sig],
                       label=f"{SIG_LABELS[sig]} (r½={r.splithalf_r:.2f})", edgecolor="white", linewidth=1)
            ax.axhline(1 / 3, color=MUTED, ls="--", lw=1)
            ax.set_xticks(x, BANDS)
            ax.set_title(f"{SDG_LABELS[s]} · {ATT_LABELS[a]} · ε={eps:g}", color=INK)
            if j == 0:
                ax.set_ylabel("share of cross-spectral energy")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("F1  Where does reliable signal live on the record graph? (dashed = white noise, 1/3)", color=INK, x=0.01, ha="left")
    _save(fig, "F1_band_energy", tag)


def f2_vs_eps(df, tag, attack="mm"):
    """F2 (H2): band shares of v vs eps; R_cross(v) vs eps; AUC vs eps (separate panels, no dual axis)."""
    d = df[(df.attack == attack) & (df.signal == "v")].sort_values("eps").copy()
    # cross-spectral ratios are noise/noise where v has no reliable per-record signal: hide those points
    unreliable = d.splithalf_r < RELIABLE_R
    d.loc[unreliable, [f"share_{b}" for b in BANDS] + ["R_cross"]] = np.nan
    sdgs = [s for s in SDG_LABELS if s in d.sdg.unique()]
    fig, axes = plt.subplots(3, len(sdgs), figsize=(3.4 * len(sdgs), 7), squeeze=False, sharex=True)
    for j, s in enumerate(sdgs):
        q = d[d.sdg == s]
        ax = axes[0, j]
        for b, c in zip(BANDS, BAND_COLORS):
            ax.plot(q.eps, q[f"share_{b}"], "-o", color=c, lw=2, ms=4, label=f"{b} band")
        ax.axhline(1 / 3, color=MUTED, ls="--", lw=1)
        ax.set_title(f"{SDG_LABELS[s]} · {ATT_LABELS[attack]}", color=INK)
        ax.set_ylabel("share of v's cross energy") if j == 0 else None
        ax.legend(fontsize=7)
        ax = axes[1, j]
        ax.plot(q.eps, q.R_cross, "-o", color=SIG_COLORS["v"], lw=2, ms=4, label="v")
        m = df[(df.attack == attack) & (df.signal == "mu_out") & (df.sdg == s)].sort_values("eps")
        ax.set_ylim(0.6, 1.2)
        ax.plot(m.eps, m.R_cross, "-o", color=SIG_COLORS["mu_out"], lw=2, ms=4, label="μ_out")
        ax.fill_between(q.eps, q.R_cross_null_mean - 2 * q.R_cross_null_sd, q.R_cross_null_mean + 2 * q.R_cross_null_sd,
                        color="#e6e5e1", label="permutation null ±2sd")
        ax.set_ylabel("cross-roughness R×") if j == 0 else None
        ax.legend(fontsize=7)
        ax = axes[2, j]
        ax.plot(q.eps, q.auc_mean, "-o", color=INK, lw=2, ms=4)
        ax.axhline(0.5, color=MUTED, ls="--", lw=1)
        ax.set_ylabel("attack AUC") if j == 0 else None
        ax.set_xscale("log")
        ax.set_xlabel("ε")
    fig.suptitle(f"F2  Does DP remove the rough part of leakage first?  (points hidden where split-half r(v) < {RELIABLE_R})", color=INK, x=0.01, ha="left")
    _save(fig, f"F2_vs_eps_{attack}", tag)


def f3_density(df, per_record, G, tag, eps=1000.0, attack="mm"):
    """F3 (H0): v vs population kNN distance, and band shares of v before/after removing density."""
    eps_tag = "{0:.2f}".format(eps)
    sdgs = [s for s in SDG_LABELS if (s, eps_tag, attack) in per_record]
    fig, axes = plt.subplots(2, len(sdgs), figsize=(3.4 * len(sdgs), 5.2), squeeze=False)
    for j, s in enumerate(sdgs):
        v = per_record[(s, eps_tag, attack)]["v"]
        ax = axes[0, j]
        ax.scatter(G["pop_knn"], v, s=9, color=SIG_COLORS["v"], alpha=0.6, edgecolor="none")
        rho = np.corrcoef(G["pop_knn"][np.isfinite(v)], v[np.isfinite(v)])[0, 1]
        ax.set_title(f"{SDG_LABELS[s]} · ε={eps:g}  (r={rho:.2f})", color=INK)
        ax.set_xlabel("mean distance to 10 nearest population records")
        ax.set_ylabel("membership signal v") if j == 0 else None
        ax = axes[1, j]
        x = np.arange(3)
        for o, sig in zip((-0.2, 0.2), ("v", "v_resid_density")):
            r = df[(df.sdg == s) & (df.eps == eps) & (df.attack == attack) & (df.signal == sig)].iloc[0]
            ax.bar(x + o, [r[f"share_{b}"] for b in BANDS], 0.38, color=SIG_COLORS[sig], label=SIG_LABELS[sig],
                   edgecolor="white", linewidth=1)
        ax.axhline(1 / 3, color=MUTED, ls="--", lw=1)
        ax.set_xticks(x, BANDS)
        ax.set_ylabel("band share") if j == 0 else None
        ax.legend(fontsize=7)
    fig.suptitle(f"F3  Is leakage just local density? ({ATT_LABELS[attack]})", color=INK, x=0.01, ha="left")
    _save(fig, f"F3_density_{attack}", tag)


def f4_eigenmap(per_record, G, tag, eps=1000.0, attack="mm"):
    """F4: records at (u2, u3) coloured by v. Patches = smooth leakage; salt-and-pepper = rough."""
    eps_tag = "{0:.2f}".format(eps)
    sdgs = [s for s in SDG_LABELS if (s, eps_tag, attack) in per_record]
    U = G["U"]
    k0 = int((G["lam"] < 1e-8).sum())
    fig, axes = plt.subplots(1, len(sdgs) + 1, figsize=(3.3 * (len(sdgs) + 1), 3.1), squeeze=False)
    first = per_record[(sdgs[0], eps_tag, attack)]
    panels = [("typicality μ_out", first["mu_out"], SDG_LABELS[sdgs[0]])] + \
             [("membership v", per_record[(s, eps_tag, attack)]["v"], SDG_LABELS[s]) for s in sdgs]
    for ax, (name, sig, lab) in zip(axes[0], panels):
        z = (sig - np.nanmean(sig)) / np.nanstd(sig)
        lim = np.nanpercentile(np.abs(z), 95)
        sc_ = ax.scatter(U[:, k0], U[:, k0 + 1], c=z, cmap=DIVERGING, vmin=-lim, vmax=lim, s=10, edgecolor="none")
        ax.set_title(f"{lab}: {name}", color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel("u₂"), ax.set_ylabel("u₃")
    fig.colorbar(sc_, ax=axes[0].tolist(), shrink=0.8, label="z-score")
    fig.suptitle(f"F4  Laplacian eigenmap of the pool, ε={eps:g}, {ATT_LABELS[attack]}", color=INK, x=0.01, ha="left", y=1.06)
    _save(fig, f"F4_eigenmap_{attack}", tag)


def make_all(df, per_record, G, tag):
    f1_band_energy(df, tag)
    for att in ("mm", "dcr"):
        if att in df.attack.unique():
            f2_vs_eps(df, tag, att)
            f3_density(df, per_record, G, tag, attack=att)
            f4_eigenmap(per_record, G, tag, attack=att)
    print(f"figures -> {OUT}")
