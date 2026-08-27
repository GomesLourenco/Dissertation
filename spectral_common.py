r"""Shared machinery for the magnetic advection--diffusion spectral studies.

Everything here is independent of the differential form degree: meshes, the
shift-and-invert eigensolve, the conditioning estimates that accompany every
result, the small numerical-analysis utilities (observed orders, one-to-one
eigenvalue assignment, agreeing digits) and the table formatter.

What is *not* here is any finite element formulation.  Those differ between the
$1$-form and top-form studies and are the subject of the notebooks, so each
notebook defines its own spaces and weak forms and hands the assembled pencil to
:func:`solve_pencil`.

Companion module: ``pseudospectra_partial_schur`` for $\varepsilon$-pseudospectra.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from firedrake import SquareMesh
from petsc4py import PETSc
from slepc4py import SLEPc

__all__ = [
    "L_DOMAIN", "crisscross_mesh", "EPS_OPTS", "KAPPA_LOG",
    "to_scipy", "to_petsc", "border_gauge",
    "shifted_matrix", "condition_estimates",
    "Result", "solve_pencil",
    "rates", "assign", "agreeing_digits", "spread",
    "fmt_table",
]

L_DOMAIN = np.pi


# --------------------------------------------------------------------- meshes
def crisscross_mesh(N, L=L_DOMAIN):
    r"""Criss-cross (Union-Jack) triangulation of $(0,L)^2$: $4N^2$ cells, $h=L/N$.

    Each grid square is split by *both* diagonals about an added centre vertex,
    so the mesh is symmetric under all eight symmetries of the square and cannot
    bias one coordinate direction over the other when a shear wind is applied.
    """
    return SquareMesh(N, N, L, quadrilateral=False, diagonal="crossed")


# ------------------------------------------------------- sparse matrix bridges
def to_scipy(mat):
    """PETSc AIJ -> scipy CSR (serial)."""
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())


def to_petsc(S):
    """scipy sparse -> PETSc AIJ; PETSc requires sorted column indices."""
    S = sp.csr_matrix(S)
    S.sort_indices()
    return PETSc.Mat().createAIJWithArrays(
        S.shape, (S.indptr.astype(PETSc.IntType),
                  S.indices.astype(PETSc.IntType), S.data))


def border_gauge(K, M, g):
    r"""Attach the gauge constraint $g^{\mathsf T}a = 0$ as a rank-one border.

    Returns the bordered pair

    .. math::

        A = \begin{pmatrix} K & -g \\ -g^{\mathsf T} & 0\end{pmatrix},
        \qquad
        M = \begin{pmatrix} M & 0 \\ 0 & 0 \end{pmatrix},

    which is how the real multiplier of a potential formulation is imposed here.
    The natural Firedrake spelling would be a mixed space ``CG * R``, but that
    assembles only as a ``MATNEST``, which cannot be converted to AIJ and cannot
    be factorised by MUMPS -- and shift-and-invert needs exactly that.  The sign
    is negative on both off-diagonal blocks so the pencil stays symmetric when
    there is no advection.
    """
    col = sp.csr_matrix(-np.asarray(g).reshape(-1, 1))
    A_b = sp.bmat([[to_scipy(K), col], [col.T, None]], format="csr")
    M_b = sp.bmat([[to_scipy(M), None], [None, sp.csr_matrix((1, 1))]], format="csr")
    return to_petsc(A_b), to_petsc(M_b)


# -------------------------------------------------------------- conditioning
def shifted_matrix(A, M, sigma):
    r"""$S = A - \sigma M$: exactly the matrix the spectral transformation factorises."""
    S = A.copy()
    # A and M generally have different sparsity -- constrained rows of A carry a
    # diagonal that M does not -- so the pattern must be declared unequal.
    S.axpy(-sigma, M, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    return S


def _mumps_cond1(S, icntl14=800):
    """MUMPS' own condition estimate, RINFOG(10), via a throw-away LU and solve.

    ``setFactorSetUpSolverType()`` builds the factor Mat *before* the numeric
    factorisation, which is the only moment at which ICNTL(11) can still be set;
    the error analysis itself runs during the solve, hence the dummy right-hand
    side.
    """
    ksp = PETSc.KSP().create(comm=S.getComm())
    ksp.setOperators(S)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    F = pc.getFactorMatrix()
    F.setMumpsIcntl(11, 1)          # 1 = full error analysis -> RINFOG(10)
    F.setMumpsIcntl(14, icntl14)
    ksp.setUp()
    b, x = S.createVecLeft(), S.createVecRight()
    b.set(1.0)
    ksp.solve(b, x)
    out = (F.getMumpsRinfog(10), F.getMumpsInfog(1))
    ksp.destroy()
    return out


def _hager_cond1(S):
    r"""Normwise $\kappa_1 = \|S\|_1\|S^{-1}\|_1$, both factors by ``onenormest``.

    The same one-norm power iteration LAPACK's ``gecon`` uses, driven through a
    sparse LU.
    """
    Ssp = to_scipy(S).tocsc()
    lu = spla.splu(Ssp)
    n = Ssp.shape[0]
    inverse = spla.LinearOperator(
        (n, n), matvec=lu.solve, rmatvec=lambda v: lu.solve(v, trans="T"))
    return spla.onenormest(Ssp) * spla.onenormest(inverse)


def condition_estimates(A, M, sigma):
    r"""Two independent estimates of $\kappa(A-\sigma M)$.

    ``Hager kappa_1`` is the normwise condition number; MUMPS' ``COND1`` is the
    componentwise Arioli--Demmel--Duff quantity computed by the factorisation
    itself, which sits an order of magnitude lower and is carried as an
    independent check.  ``INFOG(1) = 0`` confirms the factorisation succeeded.
    """
    S = shifted_matrix(A, M, sigma)
    cond, infog1 = _mumps_cond1(S)
    return {r"Hager $\kappa_1$": _hager_cond1(S),
            "MUMPS COND1": cond, "INFOG(1)": infog1}


#: Every condition estimate produced in a session, for an end-of-notebook tally.
KAPPA_LOG = []


# -------------------------------------------------------------- the eigensolve
EPS_OPTS = {
    "eps_gen_non_hermitian": None,   # generalised, non-Hermitian pencil (GNHEP)
    "eps_type": "krylovschur",
    "eps_target_magnitude": None,    # order by |lambda - tau|
    "eps_tol": 1e-12,
    "eps_max_it": 5000,
    "st_type": "sinvert",            # (A - tau M)^{-1} M
    "st_ksp_type": "preonly",        # the LU *is* the solve
    "st_pc_type": "lu",
    "st_pc_factor_mat_solver_type": "mumps",
    "st_mat_mumps_icntl_14": 800,    # working-space headroom for pivoting
}
_prefix = itertools.count()


class Result:
    r"""One solve: eigenvalues, the solver that produced them, and $\kappa(A-\tau M)$."""

    __slots__ = ("values", "index", "solver", "A", "space", "kappa", "meta")

    def __init__(self, values, index, solver, A, space, kappa, meta):
        self.values, self.index = values, index
        self.solver, self.A, self.space = solver, A, space
        self.kappa, self.meta = kappa, meta

    @property
    def real(self):
        return self.values.real

    @property
    def kappa1(self):
        return None if self.kappa is None else self.kappa[r"Hager $\kappa_1$"]

    @property
    def size(self):
        return self.A.getSize()[0]

    def __repr__(self):
        return (f"Result({self.meta.get('form', '?')}, "
                f"n={len(self.values)}, kappa1={self.kappa1:.2e})")


def solve_pencil(A, M, tau, nev=25, n_eigs=None, space=None, cond=True,
                 inf_tol=1e6, zero_tol=1e-9, **meta):
    r"""Reciprocal shift-and-invert solve of $Ax = \lambda Mx$, with conditioning.

    $M$ is singular in every formulation used here -- the multiplier block
    carries no mass -- so the transformation $(A-\tau M)^{-1}M$ is what makes the
    problem tractable: it never needs $M^{-1}$, and it sends the infinite
    eigenvalues to $\theta = 0$ where Krylov--Schur never looks.  The shifted
    matrix is an indefinite saddle point, so MUMPS factorises it directly.

    Unless ``cond=False`` the condition of the matrix actually factorised is
    estimated and attached, so that no table can quietly omit it.

    ``inf_tol`` discards the multiplier block's modes, which an ill-conditioned
    coarse pencil returns as large finite numbers rather than as ``inf``.
    """
    opts = dict(EPS_OPTS, eps_target=tau,
                eps_ncv=min(2 * nev + 30, A.getSize()[0]))
    prefix = f"sc{next(_prefix)}_"
    db = PETSc.Options()
    for k, v in opts.items():
        db[prefix + k] = v

    solver = SLEPc.EPS().create(comm=A.getComm())
    solver.setOptionsPrefix(prefix)
    solver.setOperators(A, M)
    solver.setDimensions(nev=nev)
    solver.setFromOptions()
    solver.solve()
    for k in opts:
        db.delValue(prefix + k)

    found = []
    for i in range(solver.getConverged()):
        z = complex(solver.getEigenvalue(i))
        if np.isfinite(z) and zero_tol < abs(z) < inf_tol:
            found.append((i, z))
    found.sort(key=lambda t: (t[1].real, abs(t[1].imag)))
    if n_eigs is not None:
        found = found[:n_eigs]

    kappa = condition_estimates(A, M, tau) if cond else None
    if kappa is not None:
        KAPPA_LOG.append(dict(tau=tau, n=A.getSize()[0], **meta, **kappa))

    meta = dict(meta, tau=tau, nconv=solver.getConverged(),
                reason=solver.getConvergedReason())
    return Result(np.array([z for _, z in found]),
                  np.array([i for i, _ in found]), solver, A, space, kappa, meta)


# ------------------------------------------------------- numerical utilities
def rates(h, e):
    r"""Observed order between consecutive levels, $\log(e_{i-1}/e_i)/\log(h_{i-1}/h_i)$."""
    h, e = np.asarray(h, float), np.asarray(e, float)
    r = np.full(e.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])
    return r


def assign(exact, computed):
    """Greedy one-to-one: the computed value nearest each reference eigenvalue.

    Positional comparison fails as soon as a discretisation inserts a spurious
    value, because every index after it is then shifted; matching by value
    repairs the indexing.
    """
    pool = list(computed)
    out = []
    for value in exact:
        j = int(np.argmin([abs(c - value) for c in pool]))
        out.append(pool.pop(j))
    return np.array(out)


def spread(values):
    """Max absolute difference between formulations, elementwise over a stack."""
    arr = np.asarray(values)
    return np.abs(arr.max(axis=0) - arr.min(axis=0))


def agreeing_digits(values, cap=16):
    """Leading significant digits common to a set of independently computed values.

    ``-log10(spread / |mean|)``, floored at zero and capped at the width of a
    double.  Reported alongside the raw spread because it is the quantity one
    actually quotes when saying two computations agree.
    """
    values = np.asarray(values, dtype=complex)
    lo, hi = np.abs(values).min(), np.abs(values).max()
    gap = np.abs(values.max() - values.min()) if values.size > 1 else 0.0
    scale = np.abs(np.mean(values))
    if gap == 0.0 or scale == 0.0:
        return cap
    return int(min(cap, max(0.0, -np.log10(gap / scale))))


# --------------------------------------------------------------- table output
def _cell(value, spec):
    if value is None:
        return "--"
    if isinstance(value, complex):
        return (f"{value.real:.5f}{value.imag:+.5f}i" if abs(value.imag) > 1e-10
                else spec.format(value.real))
    try:
        return "--" if not np.isfinite(value) else spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def fmt_table(df, caption=None, default="{:.6f}", rules=(), row_rules=(),
              display_fn=None):
    """Format a DataFrame for display without pandas' jinja2-dependent ``.style``.

    ``rules`` match column names and ``row_rules`` index labels, the latter
    taking precedence; in both, an exact match wins over a substring match so a
    short key cannot capture a longer name that happens to contain it.
    """
    def spec_for(key, table):
        name = " ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        return next((spec for k, spec in table if k == name),
                    next((spec for k, spec in table if k in name), None))

    out = pd.DataFrame(
        {col: [_cell(v, spec_for(idx, row_rules) or spec_for(col, rules) or default)
               for idx, v in zip(df.index, df[col])]
         for col in df.columns}, index=df.index)
    if caption and display_fn is not None:
        display_fn(caption)
    return out
