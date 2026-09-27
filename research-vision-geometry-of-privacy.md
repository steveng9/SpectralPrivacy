---
name: research-vision-geometry-of-privacy
description: "Steven's core final-year PhD research pull — \"a single ε handed to a shaped object buys unequal protection\"; the geometry of privacy leakage (spectral, curvature, effective resistance, manifolds, donor-invariant single-cell structure). Read before any brainstorming or research-direction conversation."
metadata:
  node_type: memory
  type: project
  originSessionId: 3bfe3228-9f4f-45a7-8b07-5ebb7162eb3d
  modified: 2026-09-25T20:17:35.789Z
---

# The thing Steven is circling (as of 2026-09-25)

## The one-sentence thesis
**ε is one scalar handed to an entire object. But the object has shape, and shape means one scalar buys wildly unequal protection depending on where you stand in it.** On a bridge you're exposed; in a dense cluster you're hidden. Off the manifold you're exposed; in a crowded fold you're hidden. In the high-frequency band you're exposed; in the low-frequency bulk you're anonymous. Privacy is not uniform over a geometric object, and the single-ε convention pretends it is.

Steven arrived at this intuition himself before having words for it: he "enjoys thinking about what applying a single epsilon to a topology means (a single polygon in 2d, etc.)" but lacked tools to formalize it. When I named it back to him he lit up. This is THE thread. He has effectively been circling it for four years: his SoK's "risk concentrates on atypical records," scMAMA-MIA's "leakage increases as donors decrease," and his own DP-limit note (which re-derived the Kairouz–Oh–Viswanath privacy region / f-DP trade-off curve without knowing the name) are all measurements of it.

He loves the image of **the crinkled/crumpled sheet of paper** — high-dimensional data lying on a thin, curved, low-dimensional manifold. He called it beautiful. Use that image; it's his.

## What he explicitly wants (and doesn't)
- Wants: ambitious, novel, *exciting*, geometric, mathematically beautiful. Not "another attack on a new target."
- Rejected as too adjacent / not exciting enough (though he liked them as practical fallbacks): graph reconstruction/imputation attacks, graph SDG SoK, MPC-on-GNNs with a cross-hospital synthesis narrative, running his RA on LLM-generated synthetic demographic tables, DP on genomic data. Spectral privacy was the one from that list that genuinely excited him.
- Found my first round of starred ideas (population-baseline trade-off functions, capacity-bound plateau theorem, show-people-their-reconstruction survey, node-DP vs inferential privacy) *not practical/exciting*. Don't re-pitch them unprompted.
- His three original target areas: (1) a real DP proof (he wants to understand Rényi DP / zCDP; "math is beautiful"), (2) graphs / GNNs / novel network applications, (3) social-good social-science study via surveys or AI-agent simulation. The geometry thread fuses (1) and (2).

## The live ideas, ranked by his interest
1. **Donor-invariant manifold in single-cell data** (his strongest pull). Reframe from "privatize a 20,000-dim distribution" to "how much of cell state is donor-invariant, and can I release that part almost free?" Expression = shared program/cell-type structure (population biology, not private) + donor-specific deviation (eQTL-flavored; what re-identifies donors, what scMAMA-MIA catches) + noise. DP budget only needs to cover the donor deviation. He asked: *how does one find the minimal way to express such a manifold?*
   - My answer so far: the minimal description of a manifold is an **atlas** — (#charts) × (local dimension per chart) + transition maps. scDesign2 is *already* an atlas: one chart per cell type, Gaussian copula = flat local coordinates. What's missing: charts are hand-picked (Leiden) rather than description-length-minimal, and local dim is unconstrained (~200 latent) when truth is probably ~10–15. Tools: local intrinsic dimension (Levina–Bickel MLE, TwoNN/Facco), Laplacian eigengap for #charts, diffusion maps / Laplacian eigenmaps for global coordinates, low-rank Σ ≈ LLᵀ + D. Theorem shape: DP cost scales with local dimension, not ambient.
   - Local dimension varies across the manifold (dense folds vs rare cell-type branches) — which is the ε thesis again, in single-cell.
2. **Effective resistance as per-instance privacy loss.** Established: effective resistance of an edge ≡ leverage score of that row of the incidence matrix (L = BᵀB; verified numerically in-session, triangle edges 0.667, bridge 1.000, sums to n−1 by Foster). Wang's per-instance DP ties pDP loss to leverage in linear regression (**from search summaries only — NOT yet verified by reading the paper**). My conjecture (unproven): per-instance privacy loss ≈ effective resistance in the data's geometry. Defensible narrow version: holds for *spectral* queries (Laplacian/diffusion/eigenvector-based), probably not for arbitrary nonlinear GNNs. The manifold extension (→ local density / intrinsic dim) is my weakest, overstated link — flag it as such. Known threat: von Luxburg–Radl–Hein — in large graphs resistance degenerates to 1/deg(u)+1/deg(v), which would collapse the story into "degree."
3. **Memorization is high-frequency.** Laplacian eigenbasis over a record-similarity graph; conjecture population structure lives in low modes, memorization in high ones; white DP noise overpays; shape noise to the leakage spectrum. Would explain his SoK's plateau past ε≈1.
4. **Curvature predicts where privacy leaks.** Negative Ollivier-Ricci curvature = bridges = over-squashing in GNNs. Conjecture: same bottlenecks are where membership/edges are recoverable. Curvature+GNN is crowded; curvature+privacy is empty.
5. **Diffusion-time privacy** — an *accounting* change (not a mechanism): replace k-hop group privacy (kε, vacuous past 2 hops) with heat-kernel mass e^(−tL); t slides continuously from edge-DP to node-DP. He initially misread this as "diffuse noise across the network" (a real but different idea: graph-shaped correlated noise) — I corrected it.
6. **Persistence diagrams as a DP release primitive** (high risk, high beauty). Bottleneck stability gives sensitivity for free under point *perturbation*; the real work is add/remove-a-record neighbors, which can create features.

All of 1–5 share one toolkit: the graph Laplacian, its eigenvalues, the heat kernel, effective resistance (Laplacian pseudoinverse). That's linear algebra he has. Persistent homology needs more.

## The genomic-DP wall, diagnosed
His prior attempt at formal DP for scRNA-seq (scDesign2 Gaussian copula, donor-level) stalled on (a) donor-level composition and (b) a huge parameter count (~200-dim covariance + thousands of gene marginals). My diagnosis:
- Composition wall is an artifact: release sufficient statistics, not parameters. Step 1 DP per-gene marginals; step 2 PIT with public marginals, one noisy second-moment matrix (Analyze-Gauss, Dwork–Talwar–Thakurta–Zhou). Two Gaussian mechanisms, ρ₁+ρ₂ under zCDP; everything downstream is post-processing.
- Donor-level is a clipping constant: per-donor summary, clip to norm C, average → sensitivity C/P. Same move as DP-SGD clipping, one level up.
- The *real* wall: P donors ≈ 20–50 records. No proof rescues estimating ~30k params from 20 samples. scMAMA-MIA already found this from the attack side.

## The cheap first experiment (proposed, not run)
Per-donor effective resistance (or local density / local intrinsic dimension) in the cell–cell kNN graph vs per-donor scMAMA-MIA attack success. Correlation → empirical footing for an unwritten theory. No correlation → killed a bad idea in a week. Also test the von Luxburg degeneration on his real graph.

## Open next steps I offered (he hasn't chosen yet)
- (b) Actually read Wang's per-instance DP paper and verify claim 3; harder literature check of the effective-resistance conjecture. I recommended this first.
- (a) Tutor him through rung 1: clipped Gaussian mean query, per-instance ε as a function of ‖x − μ‖ — five lines he can own.
- Offered a visual explainer page (curvature, heat kernel diagrams).
- He wants to pursue the donor-invariant manifold / minimal-manifold question further.

## Key refs surfaced
Kulynych et al. NeurIPS 2025 (arXiv 2507.06969) unified DP→risk bounds · Dong–Roth–Su f-DP · Wang per-instance DP (JPC) · Feldman–Zrnic Rényi filter · Topping et al./Nguyen et al. Ollivier-Ricci & over-squashing (arXiv 2211.15779) · Spielman–Srivastava sparsification · Belkin–Niyogi Laplacian→Laplace–Beltrami · von Luxburg–Radl–Hein resistance degeneration · DP manifold denoising (arXiv 2604.00942) · Diffusive topology-preserving manifold distances (PNAS 2404860121) · Kasiviswanathan et al. node-DP.

Related: [[steven-profile]], [[how-to-explain-to-steven]], [[prior-work-and-assets]]
