r"""$\varepsilon$-pseudospectra of finite element operators by partial Schur projection.

A mixed finite element discretisation produces a pencil $(A, M)$ with $M$ singular,
but the object of interest is not the pencil: it is the discrete operator

.. math::

    L_h = M^{-1}A : V_h \to V_h ,

and the norm in which a perturbation of it is measured has to be the norm of the
*function space*, not the Euclidean norm of the coefficient vector.  Fixing a
symmetric positive (semi-)definite metric $W$ -- a mass or a stiffness matrix --
gives the inner product $\langle x, y\rangle_W = y^*Wx$ that reproduces the
continuous one, and the $\varepsilon$-pseudospectrum in the sense of
Trefethen--Embree is

.. math::

    \sigma_\varepsilon(L_h)
      = \{\, z \in \mathbb{C} : \|(z - L_h)^{-1}\|_W \ge \varepsilon^{-1} \,\}
      = \{\, z \in \mathbb{C} : \mathcal{R}_W(z) \le \varepsilon \,\},
    \qquad
    \mathcal{R}_W(z) = \|(z - L_h)^{-1}\|_W^{-1},

so the level sets of $z \mapsto \mathcal{R}_W(z)$ draw every $\varepsilon$-pseudospectrum
at once.  For an operator that is normal *in the $W$ inner product* they are exact discs
of radius $\varepsilon$ about the eigenvalues; the amount by which they bulge beyond
those discs measures the non-normality.

Why the weight is not optional
------------------------------
With $W = I$ the quantity degenerates to $\sigma_{\min}(zM - A)$ in the Euclidean norm,
which measures *coefficient vectors*.  Those inherit the $O(h^{d})$ scaling of the mass
matrix and the dimension of the space, so the same physical perturbation registers as a
different number in two discretisations of the same operator -- a formulation with fewer
degrees of freedom reports systematically smaller $\sigma_{\min}$, and panels drawn side
by side separate into contour bands that are an artefact of the mesh rather than of the
physics.  Whitening against $W$ removes that scaling entirely: $\mathcal{R}_W$ has the
units of $z$, is mesh-independent as $h \to 0$, and is comparable across formulations.

Choosing $W$
------------
$W$ must be the Gram matrix of the norm in which the *physical field* is measured.

* **Mixed formulations** that carry the field itself (the $B$ formulations of the
  $1$-form study, the total-flux formulations of the top-form study) already assemble
  that Gram matrix: it is $M$.  Its zero block on the multiplier costs nothing, because
  the constraint forces the multiplier to vanish on the eigenvectors, so $M$ restricted
  to the computed subspace is exactly the $L^2$ metric.  This is the default,
  ``metric="mass"``, and it needs no extra assembly.
* **Potential formulations** compute $A$ with $B = \operatorname{curl}A$, so
  $\|A\|_{L^2}$ is not the quantity the $B$ formulations measure.  In two dimensions
  $|\operatorname{curl}A| = |\nabla A|$ pointwise, so the stiffness matrix -- the $H^1$
  seminorm -- makes the two agree: $\nabla^{\perp}$ is an isometry from
  $(\{\int A = 0\}, |\cdot|_{H^1})$ onto $(\{\operatorname{div}B = 0\}, \|\cdot\|_{L^2})$
  and it intertwines the two operators exactly, so the two pseudospectra coincide in the
  continuum and differ only by discretisation error.  Pass that stiffness matrix as
  ``metric=``.

Reducing it to something computable
-----------------------------------
Evaluating $\mathcal{R}_W$ directly is a dense SVD of an $N \times N$ matrix at every
point, hopeless for $N \sim 10^4$.  The steps below reduce it to an SVD of an
$m \times m$ *triangular* matrix with $m \sim 10^2$:

===== ======================================================= ==========================
Step  Object                                                  Tool
===== ======================================================= ==========================
1     saddle-point pencil $(A, M)$, $M$ singular              caller (e.g. Firedrake)
2     reciprocal shift-and-invert about $\tau$                SLEPc ``EPS`` + ``ST``
3     orthonormal basis $Q$; $A_m = Q^*AQ$, $M_m = Q^*MQ$     column-pivoted QR
3b    $G = Q^*WQ = R^*R$; whiten $\to A_w, M_w$               Cholesky (``eigh`` fallback)
4     generalised Schur $A_w = USV^*$, $M_w = UTV^*$          ``scipy.linalg.qz``
4b    $C = T^{-1}S$, the operator in triangular form          triangular solve
5     $\mathcal{R}_W(z) = \sigma_{\min}(C - zI)$, random $z$  ``scipy.linalg.svdvals``
6     scatter $\to$ grid, contour by decade                   ``scipy.interpolate``
===== ======================================================= ==========================

The order of steps 3 and 3b is the whole trick.  Eigenvectors of a strongly non-normal
operator are close to linearly dependent, and a weighted Gram matrix built directly on
them is numerically indefinite: the Cholesky factorisation fails outright.  Taking the
column-pivoted QR *first* extracts a stable rank-truncated basis, and $G = Q^*WQ$ is then
a small, well-conditioned $m \times m$ matrix that factorises cleanly.  Cholesky is used
where it succeeds and a truncated symmetric eigendecomposition where it does not, so a
badly conditioned metric degrades the rank rather than raising.

With $X$ the whitening factor ($X^*GX = I$) and $Z = QX$ the resulting $W$-orthonormal
basis, $C = T^{-1}S$ is unitarily similar to $X^{-1}(M_m^{-1}A_m)X$, the matrix of $L_h$
restricted to the computed invariant subspace written in the $Z$ basis.  Hence
$\sigma_{\min}(C - zI) = \mathcal{R}_W(z)$ for the restriction, exactly.

The module is deliberately **finite-element agnostic**: it takes assembled ``PETSc.Mat``
objects and knows nothing about the formulation that produced them, so the same code
serves the $1$-form and the top-form studies.

Typical use::

    from pseudospectra_partial_schur import pseudospectrum, plot_pseudospectrum

    # mixed formulation: its own mass matrix is the L^2 metric
    result = pseudospectrum(A, M, tau=0.9, nev=50, label=r"$\beta_4$")

    # potential formulation: measure |A|_{H^1} = ||curl A||_{L^2} instead
    K = assemble(inner(grad(u), grad(v)) * dx).petscmat
    result = pseudospectrum(A, M, tau=0.9, nev=50, metric=K)

    plot_pseudospectrum(result)

Method and implementation follow ``Lshape_Pseudospectra_Partial_Schur.ipynb``.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from petsc4py import PETSc
from slepc4py import SLEPc

__all__ = ["pseudospectrum", "plot_pseudospectrum", "diagnostics_frame",
           "partial_schur_basis", "metric_gram", "whitening_factor",
           "qz_pencil", "operator_form", "resolvent_samples", "resolvent_at",
           "bulge", "interpolate_to_grid", "bounding_box", "decade_levels",
           "resolve_levels", "shared_scale", "COMPLEX_PETSC"]

#: PETSc built with real scalars returns a complex eigenvector as a pair of real
#: vectors; every routine here handles both builds.
COMPLEX_PETSC = np.issubdtype(PETSc.ScalarType, np.complexfloating)

#: One options-database prefix per eigensolve; see :func:`partial_schur_basis`.
_prefix = itertools.count()

#: ``metric`` values that mean "use the pencil's own mass matrix".
_MASS = (None, "mass", "M")

#: ``metric`` values that mean "no weighting at all" -- the Euclidean norm of the
#: coefficient vector.  Kept only to reproduce a pre-weighting figure; the
#: resulting levels are not comparable between formulations.
_NONE = (False, "none", "euclidean")


# --------------------------------------------------------------------- step 2-3
def partial_schur_basis(A, M, tau, nev, tol=1e-10, max_it=5000,
                        finite_tol=1e8, rank_rtol=1e-12, icntl14=800):
    r"""Shift-and-invert solve, then an orthonormal basis of the invariant subspace.

    Parameters
    ----------
    A, M : PETSc.Mat
        The pencil.  ``M`` may be singular -- that is the usual case here, and is
        why the transformation is reciprocal.
    tau : float or complex
        Target.  The subspace is that of the ``nev`` eigenvalues nearest ``tau``,
        so ``tau`` must sit where the plotting window will be.
    nev : int
        Number of eigenpairs requested.
    rank_rtol : float
        Relative threshold on the pivoted-QR diagonal below which a column is
        dropped as linearly dependent on the ones before it.
    icntl14 : int
        MUMPS working-space headroom, as a percentage; see the note in the body.

    Returns
    -------
    Q : ndarray, shape (N, m)
        Orthonormal columns spanning the invariant subspace.
    lam : ndarray, shape (k,)
        The converged finite eigenvalues.

    Notes
    -----
    ``EPSGetInvariantSubspace`` cannot be used: in a real PETSc build ``EPSSolve``
    reorders complex-conjugate pairs at the end, leaving the solver in the
    ``EPS_STATE_EIGENVECTORS`` state, after which that call raises.  The basis is
    rebuilt from the converged eigenvectors instead, which is not an
    approximation -- :math:`\{\operatorname{Re}v_i, \operatorname{Im}v_i\}` spans
    exactly the same subspace, and the pseudospectrum is invariant under the
    change of basis between the two.

    The column-pivoted QR is essential rather than cosmetic: eigenvectors of a
    strongly non-normal operator are close to linearly dependent, the rank
    truncation is what stops that near-dependence contaminating ``Q``, and it is
    what makes the weighted Gram matrix of :func:`metric_gram` factorisable.
    """
    solver = SLEPc.EPS().create(comm=A.getComm())
    # A private prefix per solve, so that the one option set below cannot leak
    # into any other eigensolver in the session.
    prefix = f"psps{next(_prefix)}_"
    solver.setOptionsPrefix(prefix)
    solver.setOperators(A, M)
    # GNHEP: even with no advection the saddle-point pencil is not a definite
    # pair, because M is singular on the multiplier block.
    solver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    solver.setDimensions(nev=nev)
    solver.setTolerances(tol=tol, max_it=max_it)

    st = solver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(tau)
    ksp = st.getKSP()
    ksp.setType("preonly")           # the LU is the solve
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")  # A - tau M is indefinite

    # setTarget as well as the ST shift: without it EPS would sort by largest
    # magnitude rather than nearest tau.
    solver.setTarget(tau)
    solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

    # ICNTL(14) is percentage headroom in the MUMPS working space.  The default
    # 20% is not enough for these saddle points once the mesh is refined -- the
    # factorisation stops with INFOG(1) = -9 -- and 800 is what every other solve
    # in this project uses.  It has to go through the options database: the
    # factor Mat it applies to does not exist until STSetUp, which is also when
    # the factorisation happens, so there is no moment in between at which
    # ``setMumpsIcntl`` could be called on it.
    db = PETSc.Options()
    db[prefix + "st_mat_mumps_icntl_14"] = icntl14
    solver.setFromOptions()
    solver.solve()
    db.delValue(prefix + "st_mat_mumps_icntl_14")

    lam, cols = [], []
    vr, vi = A.createVecRight(), A.createVecRight()
    for i in range(solver.getConverged()):
        value = complex(solver.getEigenvalue(i))
        # Guard against an eigenvalue at infinity: with sinvert this should never
        # fire, and it is here so a failure would be loud rather than silent.
        if not np.isfinite(value) or abs(value) > finite_tol:
            continue
        lam.append(value)
        if COMPLEX_PETSC:
            solver.getEigenvector(i, vr)
            cols.append(np.asarray(vr.getArray()).copy())
        else:
            # A conjugate pair is stored as (vr, vi); both are needed, since
            # together they span the 2D real invariant subspace of the pair.
            solver.getEigenvector(i, vr, vi)
            cols.append(np.asarray(vr.getArray()).copy())
            imag = np.asarray(vi.getArray())
            if np.linalg.norm(imag) > 0.0:
                cols.append(imag.copy())

    if not lam:
        raise RuntimeError(f"no finite eigenpairs converged near target {tau}")

    Q, R, _ = sla.qr(np.column_stack(cols), mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    return Q[:, diag > rank_rtol * diag[0]], np.array(lam, dtype=complex)


def _apply(mat, dense):
    """``mat @ dense`` for a sparse PETSc Mat and a dense block, column by column.

    PETSc exposes no matrix-times-dense-block product here; with ``m ~ 100``
    columns this is negligible beside the LU done in the eigensolve.
    """
    x, y = mat.createVecRight(), mat.createVecLeft()
    out = np.empty((mat.getSize()[0], dense.shape[1]), dtype=dense.dtype)
    for j in range(dense.shape[1]):
        x.setArray(np.ascontiguousarray(dense[:, j]))
        mat.mult(x, y)
        out[:, j] = y.getArray()
    return out


# --------------------------------------------------------------------- step 3b
def metric_gram(metric, Q):
    r"""The Gram matrix :math:`G = Q^*WQ` of the physical metric on the subspace.

    Parameters
    ----------
    metric : PETSc.Mat
        Symmetric positive semi-definite :math:`W`.  It may be *smaller* than the
        pencil, in which case it is applied to the leading block of ``Q`` and the
        trailing rows carry no weight.  That is exactly what a bordered
        multiplier needs: a potential formulation gauged by a rank-one border can
        hand over the plain stiffness matrix of the potential space, with no
        padding, because the border direction contributes nothing to the physical
        norm anyway.
    Q : ndarray, shape (N, m)
        The orthonormal basis from :func:`partial_schur_basis`.

    Returns
    -------
    ndarray, shape (m, m)
        Explicitly symmetrised, since only the symmetric part is meaningful and
        the asymmetry left by the two matrix products is pure round-off.
    """
    n = metric.getSize()[0]
    if n > Q.shape[0]:
        raise ValueError(f"metric is {n}x{n}, larger than the pencil "
                         f"({Q.shape[0]}); it must match or be a leading block")
    block = Q if n == Q.shape[0] else np.ascontiguousarray(Q[:n])
    G = block.conj().T @ _apply(metric, block)
    return 0.5 * (G + G.conj().T)


def whitening_factor(G, rtol=1e-12):
    r"""``X`` with :math:`X^*GX = I`: the change of basis to a $W$-orthonormal frame.

    Cholesky where the Gram matrix is safely definite, and a truncated symmetric
    eigendecomposition where it is not.

    The fallback matters because $W$ is singular on every formulation here: it
    weights the physical field and nothing else, so a multiplier, a gauge border
    or the flux block of a mixed system contributes zero to it.  ``G`` is
    nonetheless positive definite in exact arithmetic -- a vector of the subspace
    on which it vanished would be an eigenvector carrying no field at all, which
    none of these pencils admits -- but the margin can be thin, and a poorly
    converged eigenpair leaves enough unweighted content behind to tip it.
    Dropping those directions costs a little of the subspace and keeps the rest
    exact, which is much better than failing.

    Returns
    -------
    X : ndarray, shape (m, k), ``k <= m``
    info : dict
        ``metric cond`` (of ``G`` as handed in), ``whitening`` (which branch ran)
        and ``metric rank`` -- all carried into the result and reported by
        :func:`diagnostics_frame`, so a silent truncation is impossible.
    """
    w = np.linalg.eigvalsh(G)
    cond = float(w[-1] / w[0]) if w[0] > 0 else np.inf
    if w[0] > rtol * w[-1]:
        R = sla.cholesky(G, lower=False)                      # G = R^* R
        X = sla.solve_triangular(R, np.eye(G.shape[0], dtype=G.dtype), lower=False)
        return X, {"metric cond": cond, "whitening": "cholesky",
                   "metric rank": G.shape[0]}

    w, U = np.linalg.eigh(G)
    keep = w > rtol * w[-1]
    if not keep.any():
        raise RuntimeError("the metric vanishes on the computed subspace: "
                           "either W is the wrong matrix or the eigenvectors "
                           "live entirely in its null space")
    X = U[:, keep] / np.sqrt(w[keep])
    return X, {"metric cond": cond, "whitening": "eigh",
               "metric rank": int(keep.sum())}


# ----------------------------------------------------------------------- step 4
def qz_pencil(A_m, M_m, inf_rtol=1e-10):
    r"""Generalised Schur factorisation of the projected pencil.

    Returns ``(S, T, ritz, residual)`` with :math:`A_m = USV^*`,
    :math:`M_m = UTV^*`, both triangular, and the finite Ritz values
    :math:`S_{ii}/T_{ii}`.  ``output="complex"`` forces a true triangular form
    rather than the real quasi-triangular one with 2x2 blocks.
    """
    S, T, U, V = sla.qz(A_m, M_m, output="complex")
    residual = max(np.abs(U @ S @ V.conj().T - A_m).max(),
                   np.abs(U @ T @ V.conj().T - M_m).max())
    dS, dT = np.diag(S), np.diag(T)
    # T_ii ~ 0 marks an eigenvalue at infinity; with sinvert none should survive
    # into the subspace, but the test is free and makes it visible.
    finite = np.abs(dT) > inf_rtol * np.abs(dT).max()
    return S, T, dS[finite] / dT[finite], residual


# ---------------------------------------------------------------------- step 4b
def operator_form(S, T, rcond=1e-10):
    r"""Turn the triangular pencil into the triangular *operator* :math:`C = T^{-1}S`.

    The pencil and the operator do not have the same pseudospectrum unless
    :math:`M_w = I`.  What the physics asks for is the resolvent of
    :math:`L_h = M^{-1}A`, so the mass has to be divided out:
    :math:`(z - L_h)^{-1} = (zM - A)^{-1}M`, whence
    :math:`\mathcal{R}_W(z) = \sigma_{\min}(zI - T^{-1}S)`.  When the metric *is*
    the mass matrix, $T$ comes back as the identity up to round-off and this step
    is a no-op -- which is the sense in which the mixed formulations get the fix
    for free.

    ``T`` is triangular, so the solve is $O(m^3/3)$ once and the product stays
    triangular; every later sample is still one small triangular SVD.
    """
    d = np.abs(np.diag(T))
    if d.min() <= rcond * d.max():
        raise RuntimeError("the whitened mass matrix is singular on the computed "
                           "subspace, so the operator M^{-1}A does not exist "
                           "there; check that the metric matches the pencil")
    C = np.triu(sla.solve_triangular(T, S, lower=False))
    return C, np.eye(C.shape[0], dtype=C.dtype), float(d.max() / d.min())


# --------------------------------------------------------------------- step 5-6
def resolvent_samples(S, T, real_range, imag_range, n_points=2000, seed=0):
    r""":math:`\mathcal{R}(z) = \sigma_{\min}(S - zT)` at random ``z`` in the box.

    Random rather than lattice sampling avoids aligning the samples with the
    roughly circular level sets, and costs nothing extra: ``S - zT`` is
    triangular, so each evaluation is a small SVD.  ``svdvals`` returns singular
    values in descending order, hence ``[-1]``.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(*real_range, n_points) + 1j * rng.uniform(*imag_range, n_points)
    return z, np.array([sla.svdvals(S - zk * T)[-1] for zk in z])


def resolvent_at(result, z):
    r"""$\mathcal{R}_W$ at an arbitrary point, from an already-computed result.

    The triangular pair is kept in the result, so probing extra points -- at a
    Ritz value, at a distance from one -- costs one small SVD each and needs no
    re-solve.  Accepts a scalar or an array and returns the same shape.
    """
    z = np.asarray(z, dtype=complex)
    S, T = result["S"], result["T"]
    out = np.array([float(sla.svdvals(S - zk * T)[-1]) for zk in z.ravel()])
    return out.reshape(z.shape) if z.ndim else float(out[0])


def interpolate_to_grid(z, sigma, real_range, imag_range, res=260):
    r"""Scattered :math:`(z, \sigma)` onto a regular grid, interpolating the log.

    Interpolating :math:`\log_{10}\sigma` keeps the result positive and weights
    every decade equally; a linear interpolant of :math:`\sigma` itself would be
    dominated by the largest values and could undershoot below zero.  Points
    outside the convex hull of the samples are filled from the nearest sample.
    """
    X, Y = np.meshgrid(np.linspace(*real_range, res), np.linspace(*imag_range, res))
    points = np.column_stack([z.real, z.imag])
    values = np.log10(np.maximum(sigma, np.finfo(float).tiny))
    grid = griddata(points, values, (X, Y), method="cubic")
    holes = ~np.isfinite(grid)
    if holes.any():
        grid[holes] = griddata(points, values, (X, Y), method="nearest")[holes]
    return X, Y, 10.0 ** grid, holes.mean()


def bounding_box(ritz, tau, n_box=8, pad_frac=0.15):
    """A window around the ``n_box`` Ritz values nearest ``tau``.

    Keeping ``n_box`` well below the subspace dimension keeps the window inside
    the part of the spectrum the subspace actually resolves, where the inner
    approximation is tight.  The window is symmetric in :math:`\\operatorname{Im}z`
    because a real pencil has a conjugation-closed spectrum, so an asymmetric one
    would only record which member of a pair happened to converge.
    """
    sel = ritz[np.argsort(np.abs(ritz - tau))][:min(n_box, ritz.size)]
    lo, hi = float(sel.real.min()), float(sel.real.max())
    span = max(hi - lo, 1e-3)
    pad = pad_frac * span
    half = max(float(np.abs(sel.imag).max()) * 1.4, 0.30 * span) + pad
    return (lo - pad, hi + pad), (-half, half)


# ------------------------------------------------------------------ the driver
def _resolve_metric(metric, M):
    """``(W, name)`` from the user-facing ``metric`` argument."""
    if isinstance(metric, PETSc.Mat):
        return metric, "custom"
    if isinstance(metric, np.ndarray):
        raise TypeError("metric must be an assembled PETSc.Mat, not a dense array")
    if metric in _MASS:
        return M, "mass"
    if metric in _NONE:
        return None, "none"
    raise ValueError(f"unknown metric {metric!r}; expected a PETSc.Mat, "
                     f"one of {_MASS} or one of {_NONE}")


def pseudospectrum(A, M, tau, nev=50, n_points=2000, res=260, n_box=8,
                   window=None, seed=0, label=None, metric="mass",
                   rank_rtol=1e-12, metric_rtol=1e-12, **meta):
    r"""Steps 1--6 for one assembled pencil.  This is the entry point.

    Parameters
    ----------
    A, M : PETSc.Mat
        The pencil, already assembled by whatever formulation is being studied.
    tau : float or complex
        Shift-and-invert target; also the centre of the default window.
    nev : int
        Eigenpairs requested.  Larger gives a tighter inner approximation of
        :math:`\mathcal{R}_W` but a looser one-sided projection at the outer
        edge -- see ``win_err`` in the result.
    metric : PETSc.Mat or {"mass", "none"}, default "mass"
        The Gram matrix $W$ of the norm in which a perturbation is measured; see
        the module docstring for how to choose it.  ``"mass"`` uses the pencil's
        own $M$, which is the right answer whenever the pencil's primary unknown
        *is* the physical field -- every mixed formulation here.  A potential
        formulation should pass its stiffness matrix instead, so that the
        perturbation is measured as $|A|_{H^1} = \|\operatorname{curl}A\|_{L^2}$
        and the levels line up with the mixed formulations'.  The matrix may be a
        leading block of the pencil, which is what lets a gauged formulation hand
        over the unbordered stiffness matrix directly.  ``None`` is an alias for
        ``"mass"``, so a per-formulation helper can return ``None`` for the
        formulations that need nothing special.  ``"none"`` disables the
        weighting and recovers the raw Euclidean $\sigma_{\min}(zM-A)$; it is
        kept only for reproducing an unweighted figure, and its levels are *not*
        comparable between formulations.
    window : ((lo, hi), (lo, hi)), optional
        Explicit plotting window; by default one is chosen around the ``n_box``
        Ritz values nearest ``tau``.
    label : str, optional
        Free-form label carried through to the plot title.
    rank_rtol, metric_rtol : float
        Relative truncation tolerances for the pivoted QR of the eigenvectors and
        for the whitening of the metric Gram matrix.
    **meta
        Any further keys to carry into the result dict (formulation name, wind,
        :math:`R_m`, ...); they are ignored here and available downstream.

    Returns
    -------
    dict
        Everything needed to plot and to audit: ``eigenvalues`` and ``eigs_in``
        (the pencil's own spectrum, all of it and the part inside the window --
        this is what :func:`plot_pseudospectrum` marks), ``ritz`` and
        ``ritz_in`` (the projected system's counterparts, retained for the
        ``win_err`` comparison), ``field`` on the grid ``X, Y``, the triangular pair
        ``S, T`` that the samples were taken from -- ``(C, I)`` when weighted,
        so that ``svdvals(S - z T)`` reads the same either way -- and the
        diagnostics ``qz_resid``, ``win_err``, ``hole``, ``metric cond``,
        ``whitening``, ``metric rank`` and ``pencil cond``.

    Notes
    -----
    The projection is a **one-sided inner** approximation:
    $\mathcal{R}_W$ computed on the subspace is $\ge$ the true one pointwise,
    with equality as $m \to N$.  Contours therefore move inward as ``nev`` grows,
    and a feature that matters should be checked for convergence in ``nev``.
    """
    W, metric_name = _resolve_metric(metric, M)

    Q, lam = partial_schur_basis(A, M, tau, nev, rank_rtol=rank_rtol)
    A_m = Q.conj().T @ _apply(A, Q)
    M_m = Q.conj().T @ _apply(M, Q)

    if W is None:
        S, T, ritz, qz_resid = qz_pencil(A_m, M_m)
        info = {"metric cond": np.nan, "whitening": "none",
                "metric rank": Q.shape[1], "pencil cond": np.nan}
    else:
        # Orthogonalise first, whiten second: G is built on the QR basis, never
        # on the raw eigenvectors, which is what keeps it factorisable.
        X, info = whitening_factor(metric_gram(W, Q), rtol=metric_rtol)
        S, T, ritz, qz_resid = qz_pencil(X.conj().T @ A_m @ X,
                                         X.conj().T @ M_m @ X)
        S, T, pencil_cond = operator_form(S, T)
        info["pencil cond"] = pencil_cond

    real_range, imag_range = (bounding_box(ritz, tau, n_box=n_box)
                              if window is None else window)
    z, sigma = resolvent_samples(S, T, real_range, imag_range, n_points, seed)
    X_grid, Y_grid, field, hole = interpolate_to_grid(
        z, sigma, real_range, imag_range, res)

    def _inside(v):
        return (real_range[0] <= v.real <= real_range[1]
                and imag_range[0] <= v.imag <= imag_range[1])

    # Fidelity of the compression, judged only on what is drawn: the outermost
    # few Ritz values of a one-sided projection are always the worst, and they
    # lie outside the window.
    inbox = [v for v in lam if _inside(v)]
    win_err = max((float(np.min(np.abs(ritz - v))) for v in inbox), default=0.0)
    ritz_in = np.array([v for v in ritz if _inside(v)], dtype=complex)
    # What a figure marks: the eigenvalues of the pencil itself, restricted to
    # the drawn window.  `ritz_in` is the projected system's counterpart and is
    # kept beside it, because `win_err` -- the distance between the two -- is the
    # number that says whether the compression is faithful where it is plotted.
    eigs_in = np.array(inbox, dtype=complex)

    return dict(ritz=ritz, ritz_in=ritz_in, eigenvalues=lam, eigs_in=eigs_in,
                S=S, T=T,
                X=X_grid, Y=Y_grid, field=field, z=z, sigma=sigma,
                real_range=real_range, imag_range=imag_range,
                m=S.shape[0], m_qr=Q.shape[1], N=A.getSize()[0],
                tau=tau, nev=nev, label=label, metric=metric_name,
                qz_resid=qz_resid, win_err=win_err, n_in=len(inbox), hole=hole,
                **info, **meta)


# ------------------------------------------------------------- non-normality
def bulge(result, eps=None, q=2.0):
    r"""How far the $\varepsilon$-pseudospectrum reaches beyond an $\varepsilon$-disc.

    ``max{ dist(z, spectrum) : R(z) <= eps } - eps``, which is zero for an
    operator that is normal in the metric and grows with the departure from
    normality.

    ``eps`` should be given explicitly and held fixed across the results being
    compared: now that the level is a physical quantity rather than a
    coefficient-vector norm, one $\varepsilon$ means the same perturbation in
    every formulation, and a bulge computed at a common $\varepsilon$ is a
    like-for-like number.  Left out, it falls back to the ``q``-th percentile of
    this result's own samples, which is self-normalising and therefore only
    comparable with itself.

    Returns ``(bulge, eps)``.
    """
    eps_level = float(np.percentile(result["sigma"], q)) if eps is None else float(eps)
    inside = result["sigma"] <= eps_level
    if not inside.any() or result["ritz"].size == 0:
        return np.nan, eps_level
    d = np.min(np.abs(result["z"][inside][:, None] - result["ritz"][None, :]), axis=1)
    return float(d.max() - eps_level), eps_level


# -------------------------------------------------------------------- plotting
def decade_levels(field, per_decade=1):
    """Contour levels spanning ``field``, by default one per decade.

    Always at least two levels, so the logarithmic colour scale has a
    non-degenerate range even when the field spans less than one decade.
    ``per_decade`` > 1 subdivides each decade logarithmically, which is useful
    when a pseudospectrum is nearly flat over the window.
    """
    lo = int(np.floor(np.log10(field.min())))
    hi = int(np.ceil(np.log10(field.max())))
    if hi <= lo:
        lo, hi = lo - 1, lo + 1
    n = max(2, int(round((hi - lo) * per_decade)) + 1)
    return np.logspace(lo, hi, n)


def resolve_levels(field, levels=None, per_decade=1):
    """Turn the ``levels`` argument of :func:`plot_pseudospectrum` into an array.

    ``None`` gives whole decades; an integer gives that many log-spaced levels
    across the range of ``field``; anything else is taken as explicit levels.
    """
    if levels is None:
        return decade_levels(field, per_decade=per_decade)
    if np.isscalar(levels):
        return np.logspace(np.log10(field.min()), np.log10(field.max()), int(levels))
    return np.asarray(levels, dtype=float)


def shared_scale(results, pad_decades=0.0):
    r"""A common ``(vmin, vmax)`` for a group of results, for comparable panels.

    Once the projection is whitened against a physical metric the level is an
    $\varepsilon$ in the units of $z$, mesh-independent and free of the
    degree-of-freedom count, so a shared scale is legitimate **across
    formulations** as well as across a physical parameter -- and comparing the
    levels is then the point of the figure rather than a hazard.  Different
    formulations may well need *different* metric matrices to measure the same
    physical field, ``"mass"`` for one and a stiffness matrix for another, and
    mixing those is exactly what this is for.

    Two things are still required: the results must be drawn on the same window,
    and every one of them must be weighted.  The second is checked here, because
    an unweighted result carries a coefficient-vector norm that no physical level
    can be compared against.
    """
    items = list(results.values() if isinstance(results, dict) else results)
    if any(r.get("metric", "none") == "none" for r in items):
        raise ValueError("unweighted results have no comparable level; recompute "
                         "with a metric before sharing a colour scale")
    lo = min(float(np.min(r["field"])) for r in items)
    hi = max(float(np.max(r["field"])) for r in items)
    lo = 10.0 ** (np.floor(np.log10(max(lo, np.finfo(float).tiny))) - pad_decades)
    hi = 10.0 ** (np.ceil(np.log10(hi)) + pad_decades)
    return lo, hi


def _power_of_ten(value, _pos=None):
    """Render a log10 value as a power of ten.

    Sub-decade levels (``per_decade`` > 1) must not be rounded to the nearest
    whole power, or two adjacent contours end up carrying the same label.
    """
    nearest = round(value)
    if abs(value - nearest) < 1e-6:
        return rf"$10^{{{int(nearest)}}}$"
    return rf"$10^{{{value:.1f}}}$"


#: Colour-bar labels.  Every weighted result gets the same name, whichever matrix
#: supplied the metric: the point of choosing $W$ per formulation is that they all
#: end up measuring the same physical norm, so panels laid side by side must not
#: be labelled as though they showed different quantities.  The unweighted
#: quantity genuinely is a different object and is named differently.
_CBAR_LABEL = {"mass": r"$\sigma^{W}_{\min}(z - L_h)$",
               "custom": r"$\sigma^{W}_{\min}(z - L_h)$",
               "none": r"$\sigma_{\min}(zM-A)$"}


def plot_pseudospectrum(
        result, ax=None, *, title=None, figsize=(4.6, 3.6),
        # --- colour field -----------------------------------------------------
        cmap="viridis_r", vmin=None, vmax=None, norm=None,
        shading="gouraud", rasterized=True, field_alpha=None,
        # --- contours ---------------------------------------------------------
        contours=True, levels=None, per_decade=1, contour_kw=None,
        clabel=True, clabel_kw=None,
        # --- eigenvalues ------------------------------------------------------
        show_eigenvalues=True, eig_kw=None, eig_label="eigenvalues",
        in_window_only=False,
        # --- sample points ----------------------------------------------------
        show_samples=False, sample_kw=None,
        # --- axes -------------------------------------------------------------
        xlim=None, ylim=None, xlabel=r"$\mathrm{Re}\,z$",
        ylabel=r"$\mathrm{Im}\,z$", aspect=None, grid=False, grid_kw=None,
        # --- colour bar -------------------------------------------------------
        colorbar=True, colorbar_label="auto", colorbar_kw=None,
        # --- legend -----------------------------------------------------------
        legend=False, legend_kw=None):
    r"""Draw one pseudospectrum: continuous log field, contours, eigenvalues.

    The field is Gouraud-shaded on a logarithmic colour scale so it shows no
    banding; only the overlaid lines are discrete, and they are labelled with the
    :math:`\varepsilon` they bound.

    Everything about the appearance can be overridden.  The parameters below are
    keyword-only, so adding to them never breaks an existing call.

    Parameters
    ----------
    result : dict
        As returned by :func:`pseudospectrum`.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes; a new figure is made if omitted.
    title : str, optional
        Axes title.  Defaults to the formulation and label carried in ``result``;
        pass ``""`` for no title.
    cmap, shading, rasterized, field_alpha
        Passed to ``pcolormesh``.
    vmin, vmax : float, optional
        Colour-scale limits.  Default to the first and last contour level.  Fix
        them across a group of panels -- see :func:`shared_scale` -- when the
        panels are meant to be compared.
    norm : matplotlib.colors.Normalize, optional
        Overrides the default ``LogNorm(vmin, vmax)`` entirely; use for a linear
        or symmetric-log scale.
    contours : bool
        Draw the contour lines at all.
    levels : None, int or array-like
        ``None`` gives whole decades (see ``per_decade``); an integer gives that
        many log-spaced levels; an array is used verbatim.
    per_decade : int
        Sub-levels per decade when ``levels is None``.
    contour_kw, clabel_kw : dict, optional
        Merged over the defaults for ``contour`` and ``clabel``.
    clabel : bool
        Label the contour lines with the value of :math:`\varepsilon`.
    show_eigenvalues : bool
        Mark the spectrum.  The markers are the eigenvalues of the full pencil
        from the SLEPc solve, *not* the Ritz values of the projected system that
        the contours are computed from; the two agree to ``win_err``, which the
        result reports and :func:`diagnostics_frame` tabulates as ``window
        fidelity``.
    in_window_only : bool
        Mark only the eigenvalues inside the drawn window rather than all the
        converged ones.  Cosmetically equivalent -- the axes clip the rest --
        but it keeps the legend count honest.
    eig_kw : dict, optional
        Merged over the default red-dot styling.
    show_samples : bool
        Overlay the randomised evaluation points, to make the method visible.
    xlim, ylim : tuple, optional
        Axis limits.  Default to the window the result was computed on; pass a
        narrower pair to zoom without recomputing.
    aspect : {'equal', 'auto'} or float, optional
    grid : bool
    colorbar : bool
    colorbar_label : str, optional
        ``"auto"`` names the quantity according to the metric the result was
        computed with; ``None`` leaves the bar unlabelled.
    colorbar_kw : dict, optional
        Passed to ``figure.colorbar``.
    legend : bool
    legend_kw : dict, optional

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.  The artists are also attached to it as the
        dictionary ``ax.ps_artists`` with keys ``field``, ``contours``,
        ``eigenvalues``, ``samples`` and ``colorbar`` (``None`` where not drawn),
        so that any of them can be restyled afterwards.
    """
    data = np.maximum(result["field"], np.finfo(float).tiny)
    lv = resolve_levels(data, levels, per_decade)
    vmin = lv[0] if vmin is None else vmin
    vmax = lv[-1] if vmax is None else vmax

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.get_figure()

    art = ax.pcolormesh(result["X"], result["Y"], data,
                        norm=norm if norm is not None else LogNorm(vmin=vmin, vmax=vmax),
                        cmap=cmap, shading=shading, rasterized=rasterized,
                        alpha=field_alpha)

    lines = None
    if contours:
        kw = dict(colors="k", linewidths=0.5, alpha=0.6)
        kw.update(contour_kw or {})
        # Contour on log10 of the field so the levels are evenly weighted.
        lines = ax.contour(result["X"], result["Y"], np.log10(data),
                           levels=np.log10(lv), **kw)
        if clabel:
            ckw = dict(inline=True, fontsize=6, fmt=_power_of_ten)
            ckw.update(clabel_kw or {})
            ax.clabel(lines, **ckw)

    samples = None
    if show_samples:
        skw = dict(color="w", ms=0.5, alpha=0.30, zorder=3, linestyle="none",
                   marker=".")
        skw.update(sample_kw or {})
        samples, = ax.plot(result["z"].real, result["z"].imag, **skw)

    dots = None
    if show_eigenvalues:
        # The markers are eigenvalues of the *full* pencil, as returned by the
        # SLEPc solve -- not the Ritz values of the projected system.  The
        # contours are a property of the compression and are computed exactly as
        # before; the spectrum is a property of the discretisation, and a figure
        # meant for publication should not blur the two.  `win_err` in the result
        # records how far the two sets sit apart inside the window.
        # `.get` keeps result dicts built before this change plottable.
        ev = (result.get("eigs_in", result["ritz_in"]) if in_window_only
              else result.get("eigenvalues", result["ritz"]))
        ekw = dict(color="r", marker=".", linestyle="none", markersize=6,
                   zorder=4, label=eig_label)
        ekw.update(eig_kw or {})
        dots, = ax.plot(ev.real, ev.imag, **ekw)

    cbar = None
    if colorbar:
        cbar = fig.colorbar(art, ax=ax, **(colorbar_kw or {}))
        if colorbar_label == "auto":
            colorbar_label = _CBAR_LABEL[result.get("metric", "none")]
        if colorbar_label:
            cbar.set_label(colorbar_label)

    ax.set_xlim(result["real_range"] if xlim is None else xlim)
    ax.set_ylim(result["imag_range"] if ylim is None else ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if aspect is not None:
        ax.set_aspect(aspect)
    if grid:
        ax.grid(**(grid_kw or dict(color="0.85", lw=0.4, alpha=0.6)))
    if legend:
        ax.legend(**(legend_kw or dict(fontsize=7, frameon=False)))

    if title is None:
        bits = [str(result.get("form", "")), str(result.get("label") or "")]
        title = ",  ".join(b for b in bits if b)
    if title:
        ax.set_title(title)

    ax.ps_artists = {"field": art, "contours": lines, "eigenvalues": dots,
                     "samples": samples, "colorbar": cbar}
    return ax


def diagnostics_frame(results, keys=("form", "wind")):
    """Audit table for a dict or list of results: is the compression trustworthy?

    ``QZ residual`` checks the unitary factorisation, ``window fidelity`` is
    :math:`\\max|\\lambda_{\\rm SLEPc} - \\mathrm{Ritz}|` over the eigenvalues that
    are actually plotted, and the two imaginary-part columns separate the
    spectrum inside the window from the poorly converged outermost Ritz values.

    ``metric cond`` is the condition number of :math:`Q^*WQ`, the small matrix
    the whitening factorises, and ``whitening`` records which branch of
    :func:`whitening_factor` ran: ``cholesky`` everywhere means no direction of
    the subspace was dropped, and ``m`` equalling the pre-whitening rank
    confirms it.
    """
    items = results.values() if isinstance(results, dict) else results
    rows = []
    for r in items:
        row = {k: r.get(k) for k in keys if r.get(k) is not None}
        row.update({
            "N": r["N"], "m": r["m"],
            "metric": r.get("metric", "none"),
            "metric cond": r.get("metric cond", np.nan),
            "whitening": r.get("whitening", "none"),
            "QZ residual": r["qz_resid"],
            "max |Im| in window": (np.abs(r["ritz_in"].imag).max()
                                   if r["ritz_in"].size else 0.0),
            "max |Im| all ritz": np.abs(r["ritz"].imag).max(),
            "window fidelity": r["win_err"],
            "eigs in window": r["n_in"],
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    index = [k for k in keys if k in df.columns]
    return df.set_index(index) if index else df
