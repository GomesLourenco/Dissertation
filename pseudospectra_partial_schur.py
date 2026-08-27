r"""$\varepsilon$-pseudospectra of generalised matrix pencils by partial Schur projection.

For a pencil $(A, M)$ coming from a mixed finite element discretisation the
$\varepsilon$-pseudospectrum is taken in the van Dorsselaer / Frayssé sense,

.. math::

    \sigma_\varepsilon(A, M)
      = \{\, z \in \mathbb{C} : \|(zM - A)^{-1}\|_2 \ge \varepsilon^{-1} \,\}
      = \{\, z \in \mathbb{C} : \sigma_{\min}(zM - A) \le \varepsilon \,\},

so the level sets of :math:`z \mapsto \sigma_{\min}(zM-A)` draw every
:math:`\varepsilon`-pseudospectrum at once.  For a *normal* operator they are exact discs
of radius :math:`\varepsilon` about the eigenvalues; the amount by which they bulge beyond
those discs measures the non-normality.

Evaluating :math:`\sigma_{\min}` directly is a dense SVD of an :math:`N \times N` matrix at
every point, hopeless for :math:`N \sim 10^4`.  The six steps below reduce it to an SVD of
an :math:`m \times m` *triangular* pencil with :math:`m \sim 10^2`:

===== ====================================================== ==========================
Step  Object                                                 Tool
===== ====================================================== ==========================
1     saddle-point pencil :math:`(A, M)`, :math:`M` singular  caller (e.g. Firedrake)
2     reciprocal shift-and-invert about :math:`\tau`          SLEPc ``EPS`` + ``ST``
3     orthonormal basis :math:`Q`; :math:`A_m = Q^*AQ`        column-pivoted QR
4     generalised Schur :math:`A_m = WSV^*`, :math:`M_m=WTV^*` ``scipy.linalg.qz``
5     :math:`\mathcal{R}(z) = \sigma_{\min}(S - zT)`, random z ``scipy.linalg.svdvals``
6     scatter :math:`\to` grid, contour by decade             ``scipy.interpolate``
===== ====================================================== ==========================

Because :math:`W` and :math:`V` are unitary,
:math:`\sigma_{\min}(S - zT) = \sigma_{\min}(A_m - zM_m)` exactly; the QZ factorisation is
computed once and each sample costs one small triangular SVD.

The module is deliberately **finite-element agnostic**: it takes assembled
``PETSc.Mat`` objects and knows nothing about the formulation that produced them, so the
same code serves the 1-form and the top-form studies.

Typical use::

    from pseudospectra_partial_schur import pseudospectrum, plot_pseudospectrum

    result = pseudospectrum(A, M, tau=0.9, nev=50, label=r"$\beta_4$")
    plot_pseudospectrum(result)

Method and implementation follow ``Lshape_Pseudospectra_Partial_Schur.ipynb``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from petsc4py import PETSc
from slepc4py import SLEPc

__all__ = ["pseudospectrum", "plot_pseudospectrum", "diagnostics_frame",
           "partial_schur_basis", "qz_pencil", "resolvent_samples",
           "interpolate_to_grid", "bounding_box", "decade_levels",
           "COMPLEX_PETSC"]

#: PETSc built with real scalars returns a complex eigenvector as a pair of real
#: vectors; every routine here handles both builds.
COMPLEX_PETSC = np.issubdtype(PETSc.ScalarType, np.complexfloating)


# --------------------------------------------------------------------- step 2-3
def partial_schur_basis(A, M, tau, nev, tol=1e-10, max_it=5000,
                        finite_tol=1e8, rank_rtol=1e-12):
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
    exactly the same subspace, and :math:`\sigma_{\min}` is invariant under the
    unitary change of basis between the two.

    The column-pivoted QR is essential rather than cosmetic: eigenvectors of a
    strongly non-normal operator are close to linearly dependent, and the rank
    truncation is what stops that near-dependence contaminating ``Q``.
    """
    solver = SLEPc.EPS().create(comm=A.getComm())
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
    solver.solve()

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


# ----------------------------------------------------------------------- step 4
def qz_pencil(A_m, M_m, inf_rtol=1e-10):
    r"""Generalised Schur factorisation of the projected pencil.

    Returns ``(S, T, ritz, residual)`` with :math:`A_m = WSV^*`,
    :math:`M_m = WTV^*`, both triangular, and the finite Ritz values
    :math:`S_{ii}/T_{ii}`.  ``output="complex"`` forces a true triangular form
    rather than the real quasi-triangular one with 2x2 blocks.
    """
    S, T, W, V = sla.qz(A_m, M_m, output="complex")
    residual = max(np.abs(W @ S @ V.conj().T - A_m).max(),
                   np.abs(W @ T @ V.conj().T - M_m).max())
    dS, dT = np.diag(S), np.diag(T)
    # T_ii ~ 0 marks an eigenvalue at infinity; with sinvert none should survive
    # into the subspace, but the test is free and makes it visible.
    finite = np.abs(dT) > inf_rtol * np.abs(dT).max()
    return S, T, dS[finite] / dT[finite], residual


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
def pseudospectrum(A, M, tau, nev=50, n_points=2000, res=260, n_box=8,
                   window=None, seed=0, label=None, **meta):
    r"""Steps 1--6 for one assembled pencil.  This is the entry point.

    Parameters
    ----------
    A, M : PETSc.Mat
        The pencil, already assembled by whatever formulation is being studied.
    tau : float or complex
        Shift-and-invert target; also the centre of the default window.
    nev : int
        Eigenpairs requested.  Larger gives a tighter inner approximation of
        :math:`\sigma_{\min}` but a looser one-sided projection at the outer
        edge -- see ``win_err`` in the result.
    window : ((lo, hi), (lo, hi)), optional
        Explicit plotting window; by default one is chosen around the ``n_box``
        Ritz values nearest ``tau``.
    label : str, optional
        Free-form label carried through to the plot title.
    **meta
        Any further keys to carry into the result dict (formulation name, wind,
        :math:`R_m`, ...); they are ignored here and available downstream.

    Returns
    -------
    dict
        Everything needed to plot and to audit: ``ritz``, ``ritz_in`` (those
        inside the window), ``field`` on the grid ``X, Y``, the triangular pair
        ``S, T``, and the diagnostics ``qz_resid``, ``win_err``, ``hole``.

    Notes
    -----
    The projection is a **one-sided inner** approximation:
    :math:`\sigma_{\min}(S-zT) \ge \sigma_{\min}(zM-A)` pointwise, with equality
    as :math:`m \to N`.  Contours therefore move inward as ``nev`` grows, and a
    feature that matters should be checked for convergence in ``nev``.
    """
    Q, lam = partial_schur_basis(A, M, tau, nev)
    A_m = Q.conj().T @ _apply(A, Q)
    M_m = Q.conj().T @ _apply(M, Q)
    S, T, ritz, qz_resid = qz_pencil(A_m, M_m)

    real_range, imag_range = (bounding_box(ritz, tau, n_box=n_box)
                              if window is None else window)
    z, sigma = resolvent_samples(S, T, real_range, imag_range, n_points, seed)
    X, Y, field, hole = interpolate_to_grid(z, sigma, real_range, imag_range, res)

    def _inside(v):
        return (real_range[0] <= v.real <= real_range[1]
                and imag_range[0] <= v.imag <= imag_range[1])

    # Fidelity of the compression, judged only on what is drawn: the outermost
    # few Ritz values of a one-sided projection are always the worst, and they
    # lie outside the window.
    inbox = [v for v in lam if _inside(v)]
    win_err = max((float(np.min(np.abs(ritz - v))) for v in inbox), default=0.0)
    ritz_in = np.array([v for v in ritz if _inside(v)], dtype=complex)

    return dict(ritz=ritz, ritz_in=ritz_in, eigenvalues=lam, S=S, T=T,
                X=X, Y=Y, field=field, z=z, sigma=sigma,
                real_range=real_range, imag_range=imag_range,
                m=Q.shape[1], N=A.getSize()[0], tau=tau, nev=nev, label=label,
                qz_resid=qz_resid, win_err=win_err, n_in=len(inbox), hole=hole,
                **meta)


# -------------------------------------------------------------------- plotting
def decade_levels(field):
    """The powers of ten spanned by ``field``, i.e. eps = 1e-1, 1e-2, ...

    Always at least two levels, so the logarithmic colour scale has a
    non-degenerate range even when the field spans less than one decade.
    """
    lo = int(np.floor(np.log10(field.min())))
    hi = int(np.ceil(np.log10(field.max())))
    if hi <= lo:
        lo, hi = lo - 1, lo + 1
    return 10.0 ** np.arange(lo, hi + 1)


def _power_of_ten(value, _pos=None):
    return rf"$10^{{{int(round(value))}}}$"


def plot_pseudospectrum(result, ax=None, title=None, colorbar=True,
                        show_samples=False, figsize=(4.6, 3.6), cmap="viridis_r"):
    """Draw one pseudospectrum: continuous log field, decade contours, eigenvalues.

    The field is Gouraud-shaded on a logarithmic colour scale so it shows no
    banding; only the overlaid black lines are discrete, and they are labelled
    with the :math:`\\varepsilon` they bound.  Red dots are the eigenvalues.
    """
    field = np.maximum(result["field"], np.finfo(float).tiny)
    levels = decade_levels(field)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.get_figure()

    art = ax.pcolormesh(result["X"], result["Y"], field,
                        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
                        cmap=cmap, shading="gouraud", rasterized=True)
    lines = ax.contour(result["X"], result["Y"], np.log10(field),
                       levels=np.log10(levels), colors="k", linewidths=0.5, alpha=0.6)
    ax.clabel(lines, inline=True, fontsize=6, fmt=_power_of_ten)

    if show_samples:
        ax.plot(result["z"].real, result["z"].imag, ".", color="w", ms=0.5,
                alpha=0.30, zorder=3)

    ev = result["ritz"]
    ax.plot(ev.real, ev.imag, "r.", markersize=6, zorder=4, label="eigenvalues")

    if colorbar:
        fig.colorbar(art, ax=ax).set_label(r"$\sigma_{\min}(zM-A)$")

    ax.set_xlim(result["real_range"])
    ax.set_ylim(result["imag_range"])
    ax.set_xlabel(r"$\mathrm{Re}\,z$")
    ax.set_ylabel(r"$\mathrm{Im}\,z$")
    if title is None:
        bits = [str(result.get("form", "")), str(result.get("label") or "")]
        title = ",  ".join(b for b in bits if b)
    ax.set_title(title)
    return ax


def diagnostics_frame(results, keys=("form", "wind")):
    """Audit table for a dict or list of results: is the compression trustworthy?

    ``QZ residual`` checks the unitary factorisation, ``window fidelity`` is
    :math:`\\max|\\lambda_{\\rm SLEPc} - \\mathrm{Ritz}|` over the eigenvalues that
    are actually plotted, and the two imaginary-part columns separate the
    spectrum inside the window from the poorly converged outermost Ritz values.
    """
    items = results.values() if isinstance(results, dict) else results
    rows = []
    for r in items:
        row = {k: r.get(k) for k in keys if r.get(k) is not None}
        row.update({
            "N": r["N"], "m": r["m"],
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
