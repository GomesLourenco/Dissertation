# Magnetic advection–diffusion eigenvalue problems in Firedrake

Two self-contained studies of the magnetic advection–diffusion eigenvalue problem

$$\varepsilon\,\mathrm{d}^{k-1}\delta^{k}u \;+\; \mathrm{d}^{k-1}\iota^{k}_\beta u \;=\; \lambda\,u,
\qquad \varepsilon = R_m^{-1},$$

on $\Omega \subset \mathbb{R}^2$, one for **1-forms** and one for **top forms**. They share
a solver architecture (SLEPc shift-and-invert with a MUMPS direct solve) and differ in
every function space and weak form.

| File | $k$ | What it is |
|:--|:--|:--|
| `baseline.ipynb` | $1$ | **Baseline** on the square: analytic-spectrum validation, $h$- and $p$-convergence tables with computed rates, eigenfunction convergence, and $\varepsilon$-pseudospectra for the $W^{1,\infty}$ winds $\beta_2$, $\beta_4$. |
| `MHD_Spectral_Tutorial.ipynb` | $1$ | Tutorial: three formulations (`A(CG)`, `B(N1)`, `B(N2)`) on a square and an L-shape, with discontinuous winds, convergence studies and left eigenvectors. |
| `TopForm_Spectral_Benchmark.ipynb` | $n=2$ | Benchmark: six discretisations at $k=n$, including the Boffi–Brezzi–Gastaldi $P_1$–div$(P_1)$ trap on a criss-cross mesh. |
| `Lshape_Pseudospectra_Partial_Schur.ipynb` | $1$ | The partial-Schur-projection pseudospectrum method, on the L-shape. |
| `beta_regularity_1form.ipynb` | $1$ | **$\beta$-regularity**: what degrading the smoothness of the velocity field does to spectra, convergence rates and pseudospectra. |
| `beta_regularity_topform.ipynb` | $n=2$ | The same programme at $k=n$, with the deliberately spurious element as a paired control. |
| `pseudospectra_partial_schur.py` | — | The $\varepsilon$-pseudospectrum engine, shared by every notebook. Takes assembled `PETSc.Mat` objects and knows nothing about the formulation. |
| `spectral_common.py` | — | Meshes, the shift-and-invert solve, the conditioning estimates, the numerical-analysis utilities and the table formatter. Form-degree agnostic. |
| `baseline.py`, `mhd_spectral_tutorial.py`, `topform_spectral_benchmark.py` | | Runnable scripts, auto-exported from the notebooks. Each reproduces every figure. |
| `figures/` | | PDF output (20 baseline, 38 top-form benchmark, 51 $\beta$-regularity, 6 tutorial). |

---

## $\beta$-regularity — `beta_regularity_1form.ipynb`, `beta_regularity_topform.ipynb`

Both on $\Omega=(0,\pi)^2$, both run every experiment in three (or four) formulations that
discretise the same operator, so consistency between them serves as a diagnostic where no
analytic answer exists. Fields: `b6`, `b7`, `b8` in $L^\infty\setminus W^{1,\infty}$, and the
vortex `b9`$(\alpha)$ in $L^p$ with $\alpha$ dialling the singularity.

**What survives the loss of regularity**

* $O(h^{2r})$ on the first five eigenvalues, for every field down to the $L^p$ vortex, in
  every formulation — and across two decades of $R_m$.
* For $\beta = (f(x),0)$ or $(f(y),0)$, however rough $f$, the $1$-form problem keeps the
  **exact** eigenvalues $\lambda = n^2/R_m$; recovered to $10^{-9}$, validating the
  reference values independently of the cross-validation.
* The well-posedness shift $\nu$ stays spectrally inert to $10^{-12}$ for $L^\infty$ fields.

**What degrades, and in what order**

1. **$p$-refinement first.** Exponential convergence goes as soon as $\beta$ leaves
   $W^{1,\infty}$, and is weakest for the point singularity. For rough data, refine $h$
   before raising $r$.
2. **Cross-formulation agreement next**, monotonically: 7–11 digits for the $L^\infty$
   fields (worst for `b8`, the only one neither a gradient nor solenoidal), falling from 8
   to 5 as $\alpha$ goes $1 \to 1.75$. A usable proxy for regularity where no estimate exists.
3. **The rate itself last**, and only where a strong singularity meets a high $R_m$.

**Pollution is a property of the space pair, not of the data** (top-form notebook). The
shadow spectrum of $\sigma(P_1$–div$P_1)$ moves by 12% across every field while the genuine
spectrum moves by 45%, and by under 0.1% across the whole range of $\alpha$. Roughening
$\beta$ neither creates pollution in a sound discretisation nor removes it from a broken
one. The $3\varepsilon(m^2+n^2)$ rule of the diffusive limit does **not** survive advection:
the diffusive part of a spurious eigenvalue is tripled, the advective part is not — a
complex spurious mode carries an imaginary part within 2% of the genuine mode it shadows.

Conditioning is reported with every table and every figure in both notebooks, and is flat in
$\alpha$ and in the roughness of $\beta$: nothing that degrades is the linear algebra.

---

## Two conventions in the study notebooks

**One figure per cell, one panel per file.** No subplot grids: every figure is drawn in
its own cell, with no plotting helper and no loop over panels, and saved to its own PDF.
Composition into rows and grids is left to the document that uses them.

**Pseudospectra come from a module.** `pseudospectra_partial_schur.py` implements the
six-step partial-Schur projection — shift-and-invert, column-pivoted QR, QZ, randomised
$\sigma_{\min}$ sampling, interpolation — behind `pseudospectrum(A, M, tau, ...)` and
`plot_pseudospectrum(result)`. The notebooks assemble a pencil and hand it over.

---

## Baseline — `baseline.ipynb`

$\Omega = (0,\pi)^2$, $k = 1$, criss-cross meshes. The control against which every later
experiment is read: convex domain, analytic spectrum, Lipschitz winds.

* **Spectrum.** All three formulations reproduce the Maxwell values $m^2+n^2$
  ($m,n\ge0$, not both zero) to five or six digits at $p=1$, $N=64$; the spectrum is real
  to round-off, as a self-adjoint operator requires.
* **Convergence, as tables of computed rates and as figures.** $h$-refinement gives exactly
  $O(h^{2p})$ — observed $2.00$ at $p=1$ and $4.00$ at $p=2$ — in all three formulations;
  $p$-refinement is exponential, reaching $10^{-13}$ by $p=5$. A 1×2 panel is produced
  separately for each formulation so any one can carry the main text.
* **Eigenfunctions.** Measured as the **gap between eigenspaces** — the sine of the largest
  principal angle in $L^2$ — because $\lambda = 1$ and $\lambda = 4$ are double and a
  computed eigenvector is then defined only up to a rotation within the eigenspace. The
  same field $B$ is compared for all three formulations. Here they differ: `B(N2)` converges at $O(h^2)$ (the full space contains all of
  $\mathcal{P}_1$), while `B(N1)` and `A(CG)` converge at $O(h)$ — the latter because the
  field is $\nabla\times A_h$ and differentiating costs an order. `B(N1)` therefore holds
  the best eigenvalues and among the worst fields.
* **Pseudospectra** for $\beta_2 = (x,-y)$ and $\beta_4 = (-y,x)$ at $R_m = 1$, via
  `pseudospectra_partial_schur.py`, with the eigenvalues overlaid. Spectrum and
  pseudospectrum agree: tight, near-circular level sets around each eigenvalue. $\beta_2$
  is a gradient field and its spectrum is real to round-off (King's Theorem 7); $\beta_4$
  is not, and its spectrum leaves the axis in conjugate pairs.
* **Conditioning** is reported with every table. `A(CG)` grows at the classical
  $O(h^{-2})$; the mixed pencils grow closer to $O(h^{-4})$, because the saddle-point
  blocks are assembled unscaled. Worst over 60 solves: $4.5\times10^7$, `INFOG(1) = 0`
  throughout.
* **Deliberately omitted:** the $R_m$ sweep and the low-regularity fields
  ($L^\infty\setminus W^{1,\infty}$ shears, $L^p$ vortex), which are the subject of the
  following sections.

---

## 1-forms — `MHD_Spectral_Tutorial.ipynb`

$\mathcal{L}B = \varepsilon\,\nabla\times\nabla\times B - \nabla\times(u\times B) = \lambda B$
with $\nabla\!\cdot B=0$, on the square $(0,\pi)^2$ and an L-shape with a $3\pi/2$
re-entrant corner.

* **Formulations** — `A(CG)` ($\mathrm{CG}_r$ + a real gauge multiplier, applied as a
  rank-one border), `B(N1)` ($\mathrm{N1curl}_r\times\mathrm{CG}_r$), `B(N2)`
  ($\mathrm{N2curl}_r\times\mathrm{CG}_{r+1}$).
* **Reference data** — $\lambda=(m^2+n^2)/R_m$ on the square; the Dauge Maxwell benchmark
  rescaled by $(2/\pi)^2$ on the L-shape; and, for any shear $u=(f(y),0)$, the exact
  family $\lambda = n^2/R_m$ with $A=\cos(ny)$.
* **Results** — $O(h^2)$ on the convex domain, $O(h^{4/3})$ for the corner-singular mode;
  left eigenvectors from the transposed pencil; $M$-norm condition numbers that are
  mesh-independent and exactly 1 on simple self-adjoint eigenvalues.

## Top forms — `TopForm_Spectral_Benchmark.ipynb`

At $k=n=2$ the closedness constraint is vacuous and the multiplier disappears. In the
rotated ($H(\operatorname{div})$) complex, $\iota^n_\beta u \leftrightarrow u\beta$ and
$\delta^n \leftrightarrow -\nabla$, so the problem is

$$-\varepsilon\Delta u + \nabla\!\cdot(\beta u) = \lambda u, \qquad u|_{\partial\Omega}=0,$$

posed in the total flux $\sigma := \varepsilon\delta^n u + \iota^n_\beta u \in H(\operatorname{div})$.

* **Discretisations** — `A(CG)` (primal $H^1_0$); `B(RT)`/`B(BDM)` (mixed,
  $\mathrm{RT}_r\times\mathrm{DG}_{r-1}$ and $\mathrm{BDM}_r\times\mathrm{DG}_{r-1}$);
  `s(RT)`/`s(BDM)` (the shifted $\sigma$-formulation, $\nu$ from Lemma 4); and
  `s(P1-divP1)` with $\Sigma_h=[P_1]^2$, $V_h=\operatorname{div}\Sigma_h$ on a criss-cross
  mesh.
* **Reference data** — $\lambda=\varepsilon(m^2+n^2)$, $m,n\ge1$; for constant $\beta$ the
  exact rigid shift $+|\beta|^2/4\varepsilon$; King's and Zeldovich's theorems as
  structural checks.
* **Conditioning** — every solve estimates $\kappa(A-\sigma M)$ for the matrix MUMPS
  actually factorised, and every results table carries it. Two estimators, calibrated
  against exact dense values on small meshes: the normwise Hager–Higham $\kappa_1$
  (sparse LU + `onenormest`) and MUMPS' own `COND1` (`ICNTL(11)=1` → `RINFOG(10)`, read
  through `Mat.getMumpsRinfog`). Over the 83 solves in the notebook the worst
  $\kappa_1$ is $3.3\times10^5$ and `INFOG(1) = 0` throughout — about five digits lost
  of sixteen.
* **Main findings**
  * The five FEEC discretisations are clean over all 100 computed eigenvalues; the
    Lemma-3 shift $\nu$ is spectrally inert to $10^{-12}$.
  * On the criss-cross mesh $\operatorname{div}[P_1]^2$ has codimension $N^2$; the cokernel
    is exactly the checkerboard mode, constructed in closed form and verified to machine
    precision.
  * `s(P1-divP1)` produces a **complete shadow spectrum at $3\varepsilon(m^2+n^2)$** with
    the true multiplicities, converging at the full $O(h^2)$ rate to those wrong values.
    The eigenmodes are the checkerboard times a genuine envelope.
  * Under advection the spurious modes are camouflaged, not exposed: their imaginary parts
    track the genuine ones to within 2%, and by $R_m=20$ the first sits within 7% of
    $\lambda_1$.
  * The genuine modes of `s(P1-divP1)` converge at the full $O(h^2)$ in both eigenvalue
    and eigenfunction — once the indexing is repaired by matching against the exact list,
    since the spurious value shifts every position after it. Rate is not the failure; the
    **count** is.
  * The resolvent does not detect the pollution either: $\sigma_{\min}(zM-A)$ collapses at
    the spurious eigenvalue exactly as at a genuine one, because it *is* an eigenvalue of
    the discrete pencil.
  * Two reliable diagnostics: matching against a FEEC pair on the same mesh, and the
    rigid-shift test for a constant potential field (no reference computation needed).
  * Conditioning is **blind to the pollution** — the pathological pair is the
    best-conditioned of the six. A healthy $\kappa$ certifies the linear algebra, not the
    spectrum.

---

## Running

Needs Firedrake with SLEPc, MUMPS and (for the L-shape) Netgen, plus
numpy/scipy/pandas/matplotlib.

```bash
python mhd_spectral_tutorial.py
```

```bash
python topform_spectral_benchmark.py
```

or open either notebook with the Firedrake kernel. Each takes about a minute in serial.

## Caveat carried by both

Nothing is stabilised, deliberately — SUPG and its relatives perturb the operator whose
spectrum is the object of study. Unstabilised Galerkin advection–diffusion is dependable
only while the cell Péclet number $\mathrm{Pe}_h = h\|\beta\|_\infty R_m/2 \lesssim 1$,
i.e. $N \gtrsim \pi R_m/2$ on $(0,\pi)^2$. Every figure respects that bound; running past
it produces eigenvalues with negative real part that look like dynamo growth and are not.
