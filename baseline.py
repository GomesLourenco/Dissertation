"""
Spectral approximation of the MHD induction operator in Firedrake.

Script form of ``MHD_Spectral_Tutorial.ipynb`` -- the same code, in the same
order, with the narrative reduced to section banners.  Read the notebook for
the derivations, the tables and the figures; import this module (or run it
with ``python mhd_spectral_tutorial.py``) to reuse the machinery.
"""


# --------------------------------------------------------------------------
# 0. Setup
# --------------------------------------------------------------------------

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from firedrake import *
from firedrake.function import PointEvaluator
from firedrake.pyplot import triplot, tripcolor

from petsc4py import PETSc
from slepc4py import SLEPc

# The epsilon-pseudospectrum machinery lives in its own module: it works on
# assembled PETSc matrices and knows nothing about the formulation, so the same
# code serves this notebook and the top-form benchmark.
import pseudospectra_partial_schur as ps

# `from firedrake import *` exports a `logging` of its own, so the standard
# library module has to be re-imported under another name afterwards.
import logging as stdlogging
stdlogging.getLogger("firedrake").setLevel(stdlogging.ERROR)
stdlogging.getLogger("tsfc").setLevel(stdlogging.ERROR)

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)
L_DOMAIN = np.pi

# PETSc here is built with REAL scalars: SLEPc returns a complex eigenvector as a
# pair (real part, imaginary part) of separate real vectors.
COMPLEX_PETSC = ps.COMPLEX_PETSC

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
})

# Publication style for the pseudospectrum panels, matching the L-shape notebook.
PS_RC = {
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "ps.fonttype": 42,
}


# --- table rendering, without pandas' jinja2-dependent .style ----------------
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


def fmt_table(df, caption=None, default="{:.6f}", rules=(), row_rules=()):
    """Format a DataFrame for display; `rules` match column names, `row_rules`
    index labels and take precedence."""
    def spec_for(key, table):
        name = " ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        # Exact matches win over substring matches: a rule keyed on "m" must not
        # capture a column called "max |Im ritz|".
        return next((spec for k, spec in table if k == name),
                    next((spec for k, spec in table if k in name), None))

    out = pd.DataFrame(
        {col: [_cell(v, spec_for(idx, row_rules) or spec_for(col, rules) or default)
               for idx, v in zip(df.index, df[col])]
         for col in df.columns}, index=df.index)
    if caption:
        display(Markdown(f"**{caption}**"))
    return out


print(f"PETSc scalars: {PETSc.ScalarType.__name__} "
      f"({'complex' if COMPLEX_PETSC else 'real'} build)")


# --------------------------------------------------------------------------
# 2. Discretisation and solver
# --------------------------------------------------------------------------

def crisscross_mesh(N, L=L_DOMAIN):
    r"""Criss-cross (Union-Jack) triangulation of $(0,L)^2$: $4N^2$ cells, $h = L/N$."""
    return SquareMesh(N, N, L, quadrilateral=False, diagonal="crossed")


def cross2d(a, b):
    """Scalar 2D cross product $(a\\times b)_z = a_1b_2 - a_2b_1$; antisymmetric,
    so the ordering in the induction term (`cross2d(v, u)`) is load-bearing."""
    return a[0] * b[1] - a[1] * b[0]


def space(form, mesh, p):
    if form == "A(CG)":
        return FunctionSpace(mesh, "CG", p)
    if form == "B(N1)":
        return FunctionSpace(mesh, "N1curl", p) * FunctionSpace(mesh, "CG", p)
    return FunctionSpace(mesh, "N2curl", p) * FunctionSpace(mesh, "CG", p + 1)


FORMS = ["A(CG)", "B(N1)", "B(N2)"]
STYLE = {"A(CG)": ("C2", "^"), "B(N1)": ("C0", "o"), "B(N2)": ("C1", "s")}

def to_scipy(mat):
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())


def to_petsc(S):
    S = sp.csr_matrix(S); S.sort_indices()
    return PETSc.Mat().createAIJWithArrays(
        S.shape, (S.indptr.astype(PETSc.IntType),
                  S.indices.astype(PETSc.IntType), S.data))


def build_pencil(form, mesh, p=1, beta=None, eps=Constant(1.0)):
    r"""Assemble $(A, M)$ for one formulation. $M$ is singular in every case."""
    if form == "A(CG)":
        V = space(form, mesh, p)
        a_, z = TrialFunction(V), TestFunction(V)
        a = eps * inner(grad(a_), grad(z)) * dx
        if beta is not None:
            a += inner(dot(beta, grad(a_)), z) * dx
        K = assemble(a).petscmat
        M = assemble(inner(a_, z) * dx).petscmat
        # g_i = \int phi_i : the gauge constraint \int A = 0 as a rank-one border,
        #   [[K, -g], [-g^T, 0]] x = lambda [[M, 0], [0, 0]] x.
        g = np.asarray(assemble(TestFunction(V) * dx).dat.data_ro).copy()
        col = sp.csr_matrix(-g.reshape(-1, 1))
        A_b = sp.bmat([[to_scipy(K), col], [col.T, None]], format="csr")
        M_b = sp.bmat([[to_scipy(M), None], [None, sp.csr_matrix((1, 1))]],
                      format="csr")
        return to_petsc(A_b), to_petsc(M_b), V

    W = space(form, mesh, p)
    (v, q) = TrialFunctions(W)
    (w, r) = TestFunctions(W)
    a = eps * inner(curl(v), curl(w)) * dx
    if beta is not None:
        a += inner(cross2d(v, beta), curl(w)) * dx
    a += (-inner(grad(q), w) + inner(v, grad(r))) * dx
    # DirichletBC on an H(curl) space constrains the TANGENTIAL trace, i.e. the
    # perfectly-conducting condition; p = 0 puts the multiplier in CG \cap H^1_0.
    bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary"),
           DirichletBC(W.sub(1), Constant(0.0), "on_boundary")]
    # weight=0.0 gives the constrained rows a zero mass diagonal, so the boundary
    # modes go to lambda = infinity rather than to a spurious lambda = 1.
    return (assemble(a, bcs=bcs).petscmat,
            assemble(inner(v, w) * dx, bcs=bcs, weight=0.0).petscmat, W)


# --------------------------------------------------------------------------
# 2.2 Shift-and-invert, and the conditioning of what is factorised
# --------------------------------------------------------------------------

EPS_OPTS = {
    "eps_gen_non_hermitian": None,
    "eps_type": "krylovschur",
    "eps_target_magnitude": None,
    "eps_tol": 1e-12,
    "eps_max_it": 5000,
    "st_type": "sinvert",
    "st_ksp_type": "preonly",
    "st_pc_type": "lu",
    "st_pc_factor_mat_solver_type": "mumps",
    "st_mat_mumps_icntl_14": 800,
}
_prefix = itertools.count()
KAPPA_LOG = []


def shifted_matrix(A, M, sigma):
    S = A.copy()
    S.axpy(-sigma, M, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    return S


def condition_estimates(A, M, sigma):
    r"""Estimates of $\kappa(A-\sigma M)$: normwise $\kappa_1$, and MUMPS' COND1."""
    S = shifted_matrix(A, M, sigma)

    ksp = PETSc.KSP().create(comm=S.getComm())
    ksp.setOperators(S); ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu"); pc.setFactorSolverType("mumps")
    # The factor Mat has to exist before the numeric factorisation for ICNTL(11)
    # to take effect; the error analysis itself runs during the solve.
    pc.setFactorSetUpSolverType()
    F = pc.getFactorMatrix()
    F.setMumpsIcntl(11, 1)
    F.setMumpsIcntl(14, 800)
    ksp.setUp()
    b, xv = S.createVecLeft(), S.createVecRight()
    b.set(1.0)
    ksp.solve(b, xv)
    cond1, infog1 = F.getMumpsRinfog(10), F.getMumpsInfog(1)
    ksp.destroy()

    Ssp = to_scipy(S).tocsc()
    lu = spla.splu(Ssp)
    n = Ssp.shape[0]
    inverse = spla.LinearOperator(
        (n, n), matvec=lu.solve, rmatvec=lambda v: lu.solve(v, trans="T"))
    kappa1 = spla.onenormest(Ssp) * spla.onenormest(inverse)
    return {r"Hager $\kappa_1$": kappa1, "MUMPS COND1": cond1, "INFOG(1)": infog1}


@dataclass
class Result:
    values: np.ndarray
    index: np.ndarray
    solver: object
    A: object
    space: object
    form: str
    ndof: int
    kappa: dict

    @property
    def real(self):
        return self.values.real

    @property
    def kappa1(self):
        return self.kappa[r"Hager $\kappa_1$"]


#: Eigenvalues above this are the multiplier block's modes at infinity, returned
#: as large finite numbers by an ill-conditioned coarse pencil.  The resolved
#: spectrum here is O(10^2), so the gap is wide and the threshold uncritical.
INF_TOL = 1e6


def solve_eigs(form, mesh, p=1, beta=None, Rm=1.0, tau=None, nev=25, cond=True):
    r"""Shift-and-invert solve for one configuration, with $\kappa(A-\tau M)$ attached.

    `tau` defaults to $0.9\varepsilon$, just below the first Maxwell eigenvalue
    $\lambda_1 = \varepsilon$; that keeps the wanted end of the spectrum nearest
    the shift without ever sitting on it.
    """
    eps = Constant(1.0 / Rm)
    tau = 0.9 / Rm if tau is None else tau
    A, M, W = build_pencil(form, mesh, p, beta=beta, eps=eps)

    opts = dict(EPS_OPTS, eps_target=tau,
                eps_ncv=min(2 * nev + 30, A.getSize()[0]))
    prefix = f"bl{next(_prefix)}_"
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
        if np.isfinite(z) and 1e-9 < abs(z) < INF_TOL:
            found.append((i, z))
    found.sort(key=lambda t: (t[1].real, abs(t[1].imag)))

    kappa = condition_estimates(A, M, tau) if cond else None
    if kappa is not None:
        KAPPA_LOG.append(dict(form=form, p=p, N=mesh.num_cells(), Rm=Rm,
                              tau=tau, **kappa))
    return Result(np.array([z for _, z in found]),
                  np.array([i for i, _ in found]), solver, A, W, form,
                  A.getSize()[0], kappa)


def eigenfunction(res, k):
    """The k-th eigenmode: the H(curl) field for B(...), the potential for A(CG)."""
    vr, vi = res.A.createVecRight(), res.A.createVecRight()
    res.solver.getEigenvector(int(res.index[k]), vr, vi)
    arr = np.asarray(vr.getArray())
    if res.form == "A(CG)":
        f = Function(res.space)
        f.dat.data[:] = arr[:res.space.dim()]
        return f
    f = Function(res.space)
    with f.dat.vec_wo as fv:
        fv.setArray(arr[:fv.getLocalSize()])
    return f.subfunctions[0]


def exact_spectrum(n, Rm=1.0, kmax=30):
    r"""$\varepsilon(m^2+n^2)$, $m,n\ge0$, $(m,n)\ne(0,0)$, with multiplicity."""
    vals = sorted(m * m + k * k for m in range(kmax) for k in range(kmax)
                  if (m, k) != (0, 0))
    return np.array(vals[:n], dtype=float) / Rm


#: The exact eigenvalue *clusters* covering the first five eigenvalues, each with
#: the index pairs $(m,n)$ whose potential $\cos(mx)\cos(ny)$ spans its eigenspace.
#: Clusters rather than individual modes because $\lambda = 1$ and $\lambda = 4$
#: are double: on a degenerate eigenspace the computed eigenvector is only defined
#: up to a rotation within the space, so a vector-by-vector error is meaningless.
CLUSTERS = [(1.0, [(1, 0), (0, 1)]),
            (2.0, [(1, 1)]),
            (4.0, [(2, 0), (0, 2)])]
N_TRACK = 5          # eigenvalues followed individually in the convergence study


def exact_field(mesh, m, n):
    r"""$B = \nabla\times\cos(mx)\cos(ny) = (-n\cos mx\,\sin ny,\ m\sin mx\,\cos ny)$."""
    x, y = SpatialCoordinate(mesh)
    return as_vector([-n * cos(m * x) * sin(n * y),
                      m * sin(m * x) * cos(n * y)])


# --------------------------------------------------------------------------
# 3. Baseline spectrum, $\beta_0 = 0$
# --------------------------------------------------------------------------

N_FINE, P_BASE = 64, 1
mesh_fine = crisscross_mesh(N_FINE)

runs0 = {f: solve_eigs(f, mesh_fine, p=P_BASE, nev=25) for f in FORMS}

ex5 = exact_spectrum(5)
tbl = pd.DataFrame({"exact": ex5} | {f: runs0[f].real[:5] for f in FORMS},
                   index=[rf"$\lambda_{{{i+1}}}$" for i in range(5)])
for f in FORMS:
    tbl[f"rel err {f}"] = np.abs(tbl[f] - ex5) / ex5
# Conditioning of the matrix each column was produced from, and the size of it.
tbl.loc[r"$\kappa_1(A-\tau M)$"] = [np.nan] + [runs0[f].kappa1 for f in FORMS] + [np.nan] * 3
tbl.loc["pencil size"] = [np.nan] + [runs0[f].ndof for f in FORMS] + [np.nan] * 3

fmt_table(tbl, rules=[("rel err", "{:.2e}")],
          row_rules=[(r"\kappa_1", "{:.2e}"), ("pencil", "{:.0f}")],
          caption=rf"First five eigenvalues, $\beta_0 = 0$, $p = {P_BASE}$, criss-cross "
                  rf"$N = {N_FINE}$ ($h = \pi/{N_FINE}$, {mesh_fine.num_cells()} cells)")

print("self-adjointness check (u = 0 must give a real spectrum):")
for f in FORMS:
    v = runs0[f].values[:20]
    print(f"  {f:7s} max |Im lambda| = {np.abs(v.imag).max():.3e}"
          f"    MUMPS COND1 = {runs0[f].kappa['MUMPS COND1']:.2e}"
          f"    INFOG(1) = {runs0[f].kappa['INFOG(1)']}")


# --------------------------------------------------------------------------
# 4. Convergence
# --------------------------------------------------------------------------

# The exact modes are trigonometric, so error integrals use a fixed high-order
# rule rather than UFL's degree estimate.
DX_HI = dx(metadata={"quadrature_degree": 10})


def rates(h, e):
    """Observed order between consecutive refinement levels."""
    h, e = np.asarray(h, float), np.asarray(e, float)
    r = np.full(e.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])
    return r


def as_field(form, f):
    r"""The magnetic field of an eigenmode: $\nabla\times A_h$, or $v_h$ itself."""
    return as_vector([f.dx(1), -f.dx(0)]) if form == "A(CG)" else f


def _whiten(G):
    """W with W^T G W = I, dropping numerically null directions.

    An eigendecomposition rather than a Cholesky: the computed basis of a cluster
    can be close to rank deficient on coarse meshes, where a Cholesky simply
    fails.
    """
    w, V = np.linalg.eigh(G)
    keep = w > w.max() * 1e-13
    return V[:, keep] / np.sqrt(w[keep])


def subspace_error(computed, exact):
    r"""$\sin\theta_{\max}$ between two $L^2$ subspaces (the gap).

    With orthonormalised bases the singular values of the cross-Gram matrix are
    the cosines of the principal angles, so the largest angle is set by the
    smallest singular value.  Equals the usual normalised $L^2$ error, up to
    sign, when both subspaces are one-dimensional.
    """
    gram = lambda P, Q: np.array([[assemble(inner(a, b) * DX_HI) for b in Q]
                                  for a in P])
    Wc, We = _whiten(gram(computed, computed)), _whiten(gram(exact, exact))
    sv = np.linalg.svd(Wc.T @ gram(computed, exact) @ We, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - min(sv.min(), 1.0) ** 2)))


def h_study(form, p, levels, with_fields=False):
    """Eigenvalue errors for the first `N_TRACK` modes, and optionally the
    eigenspace gaps, over a sequence of meshes."""
    exact = exact_spectrum(N_TRACK)
    rows_e, rows_f = [], []
    for N in levels:
        mesh = crisscross_mesh(N)
        res = solve_eigs(form, mesh, p=p, nev=25)
        lam = res.real
        rows_e.append(dict(N=N, h=np.pi / N, dof=res.ndof,
                           **{rf"$\lambda_{{{j+1}}}$": abs(lam[j] - exact[j]) / exact[j]
                              for j in range(N_TRACK)},
                           **{r"$\kappa_1$": res.kappa1}))
        if with_fields:
            row = dict(N=N, h=np.pi / N)
            for value, modes in CLUSTERS:
                # Selected by proximity to the exact cluster, not by position:
                # robust to any reordering of the computed spectrum.
                idx = np.argsort(np.abs(lam - value))[:len(modes)]
                row[rf"$\lambda={value:g}$"] = subspace_error(
                    [as_field(form, eigenfunction(res, int(i))) for i in idx],
                    [exact_field(mesh, m, n) for m, n in modes])
            row[r"$\kappa_1$"] = res.kappa1
            rows_f.append(row)

    df_e = pd.DataFrame(rows_e).set_index("N")
    df_f = pd.DataFrame(rows_f).set_index("N") if with_fields else None
    return df_e, df_f


H_LEVELS = {1: (4, 8, 16, 32, 64), 2: (4, 8, 16, 32)}
h_eig, h_fun = {}, {}
for p in (1, 2):
    for f in FORMS:
        h_eig[(f, p)], fields = h_study(f, p, H_LEVELS[p], with_fields=(p == 1))
        if fields is not None:
            h_fun[f] = fields

def rate_frame(df, cols):
    """Interleave each error column with its observed rate."""
    out = pd.DataFrame(index=df.index)
    out["h"] = df["h"]
    for c in cols:
        out[c] = df[c]
        out[f"rate {c}"] = rates(df["h"], df[c])
    out[r"$\kappa_1$"] = df[r"$\kappa_1$"]
    return out


EIG_COLS = [rf"$\lambda_{{{j+1}}}$" for j in range(N_TRACK)]
for p in (1, 2):
    for f in FORMS:
        display(fmt_table(
            rate_frame(h_eig[(f, p)], EIG_COLS), default="{:.3e}",
            rules=[("rate", "{:.2f}"), ("h", "{:.5f}"), (r"$\kappa_1$", "{:.2e}")],
            caption=rf"$h$-refinement, **{f}**, $p = {p}$: relative error in the first "
                    rf"{N_TRACK} eigenvalues, expected rate $2p = {2*p}$"))

FUN_COLS = [rf"$\lambda={v:g}$" for v, _ in CLUSTERS]
for f in FORMS:
    display(fmt_table(
        rate_frame(h_fun[f], FUN_COLS), default="{:.3e}",
        rules=[("rate", "{:.2f}"), ("h", "{:.5f}"), (r"$\kappa_1$", "{:.2e}")],
        caption=rf"Eigenfunction convergence, **{f}**, $p = 1$: gap between the computed "
                r"and exact eigenspaces of $B$, per cluster"))


# --------------------------------------------------------------------------
# 4.2 $p$-refinement: spectral convergence
# --------------------------------------------------------------------------

P_LEVELS = (1, 2, 3, 4, 5)
N_PREF = 6


def p_study(form, levels=P_LEVELS, N=N_PREF):
    exact = exact_spectrum(N_TRACK)
    rows = []
    for p in levels:
        res = solve_eigs(form, crisscross_mesh(N), p=p, nev=25)
        lam = res.real
        rows.append(dict(p=p, dof=res.ndof,
                         **{rf"$\lambda_{{{j+1}}}$": abs(lam[j] - exact[j]) / exact[j]
                            for j in range(N_TRACK)},
                         **{r"$\kappa_1$": res.kappa1}))
    return pd.DataFrame(rows).set_index("p")


p_tables = {f: p_study(f) for f in FORMS}
for f in FORMS:
    display(fmt_table(
        p_tables[f], default="{:.3e}",
        rules=[("dof", "{:.0f}"), (r"$\kappa_1$", "{:.2e}")],
        caption=rf"$p$-refinement, **{f}**, fixed $N = {N_PREF}$: relative error in the "
                rf"first {N_TRACK} eigenvalues"))


# --------------------------------------------------------------------------
# 4.3 The figures
# --------------------------------------------------------------------------

df = h_eig[("A(CG)", 1)]
h = df["h"].to_numpy()
e1 = df[r"$\lambda_{1}$"].to_numpy()
e2 = df[r"$\lambda_{2}$"].to_numpy()
e3 = df[r"$\lambda_{3}$"].to_numpy()
e4 = df[r"$\lambda_{4}$"].to_numpy()
e5 = df[r"$\lambda_{5}$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color="C2", ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color="C4", ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^{2})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"A(CG): $h$-refinement, $p = 1$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_h_p1_ACG.pdf")
plt.show()

df = h_eig[("B(N1)", 1)]
h = df["h"].to_numpy()
e1 = df[r"$\lambda_{1}$"].to_numpy()
e2 = df[r"$\lambda_{2}$"].to_numpy()
e3 = df[r"$\lambda_{3}$"].to_numpy()
e4 = df[r"$\lambda_{4}$"].to_numpy()
e5 = df[r"$\lambda_{5}$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color="C2", ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color="C4", ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^{2})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"B(N1): $h$-refinement, $p = 1$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_h_p1_BN1.pdf")
plt.show()

df = h_eig[("B(N2)", 1)]
h = df["h"].to_numpy()
e1 = df[r"$\lambda_{1}$"].to_numpy()
e2 = df[r"$\lambda_{2}$"].to_numpy()
e3 = df[r"$\lambda_{3}$"].to_numpy()
e4 = df[r"$\lambda_{4}$"].to_numpy()
e5 = df[r"$\lambda_{5}$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color="C2", ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color="C4", ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^{2})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"B(N2): $h$-refinement, $p = 1$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_h_p1_BN2.pdf")
plt.show()

df = h_eig[("A(CG)", 2)]
h = df["h"].to_numpy()
e1 = df[r"$\lambda_{1}$"].to_numpy()
e2 = df[r"$\lambda_{2}$"].to_numpy()
e3 = df[r"$\lambda_{3}$"].to_numpy()
e4 = df[r"$\lambda_{4}$"].to_numpy()
e5 = df[r"$\lambda_{5}$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color="C2", ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color="C4", ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 4, "k--", lw=1.1, label=r"$O(h^{4})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"A(CG): $h$-refinement, $p = 2$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_h_p2_ACG.pdf")
plt.show()

df = h_eig[("B(N1)", 2)]
h = df["h"].to_numpy()
e1 = df[r"$\lambda_{1}$"].to_numpy()
e2 = df[r"$\lambda_{2}$"].to_numpy()
e3 = df[r"$\lambda_{3}$"].to_numpy()
e4 = df[r"$\lambda_{4}$"].to_numpy()
e5 = df[r"$\lambda_{5}$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color="C2", ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color="C4", ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 4, "k--", lw=1.1, label=r"$O(h^{4})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"B(N1): $h$-refinement, $p = 2$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_h_p2_BN1.pdf")
plt.show()

df = h_eig[("B(N2)", 2)]
h = df["h"].to_numpy()
e1 = df[r"$\lambda_{1}$"].to_numpy()
e2 = df[r"$\lambda_{2}$"].to_numpy()
e3 = df[r"$\lambda_{3}$"].to_numpy()
e4 = df[r"$\lambda_{4}$"].to_numpy()
e5 = df[r"$\lambda_{5}$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color="C2", ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color="C4", ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 4, "k--", lw=1.1, label=r"$O(h^{4})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"B(N2): $h$-refinement, $p = 2$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_h_p2_BN2.pdf")
plt.show()

df = p_tables["A(CG)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[r"$\lambda_{1}$"], "o-", color="C0", ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[r"$\lambda_{2}$"], "o-", color="C1", ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[r"$\lambda_{3}$"], "o-", color="C2", ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[r"$\lambda_{4}$"], "o-", color="C3", ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[r"$\lambda_{5}$"], "o-", color="C4", ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $p$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"A(CG): $p$-refinement, $N = {}$ (spectral)".format(N_PREF))
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_pref_ACG.pdf")
plt.show()

df = p_tables["B(N1)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[r"$\lambda_{1}$"], "o-", color="C0", ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[r"$\lambda_{2}$"], "o-", color="C1", ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[r"$\lambda_{3}$"], "o-", color="C2", ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[r"$\lambda_{4}$"], "o-", color="C3", ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[r"$\lambda_{5}$"], "o-", color="C4", ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $p$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"B(N1): $p$-refinement, $N = {}$ (spectral)".format(N_PREF))
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_pref_BN1.pdf")
plt.show()

df = p_tables["B(N2)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[r"$\lambda_{1}$"], "o-", color="C0", ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[r"$\lambda_{2}$"], "o-", color="C1", ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[r"$\lambda_{3}$"], "o-", color="C2", ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[r"$\lambda_{4}$"], "o-", color="C3", ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[r"$\lambda_{5}$"], "o-", color="C4", ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $p$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"B(N2): $p$-refinement, $N = {}$ (spectral)".format(N_PREF))
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "baseline_pref_BN2.pdf")
plt.show()

df = h_fun["A(CG)"]
h = df["h"].to_numpy()
g1 = df[r"$\lambda=1$"].to_numpy()
g2 = df[r"$\lambda=2$"].to_numpy()
g3 = df[r"$\lambda=4$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, g1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda = 1$  ({np.nanmean(rates(h, g1)[1:]):.2f})")
ax.loglog(h, g2, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda = 2$  ({np.nanmean(rates(h, g2)[1:]):.2f})")
ax.loglog(h, g3, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda = 4$  ({np.nanmean(rates(h, g3)[1:]):.2f})")

ref = 0.30 * min(g1[0], g2[0], g3[0])
ax.loglog(h, ref * (h / h[0]) ** 1, "k--", lw=1.1, label=r"$O(h^{1})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\sin\theta_{\max}$ between eigenspaces of $B$")
ax.set_title(r"A(CG): eigenfunction convergence, $p = 1$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "baseline_eigfun_ACG.pdf")
plt.show()

df = h_fun["B(N1)"]
h = df["h"].to_numpy()
g1 = df[r"$\lambda=1$"].to_numpy()
g2 = df[r"$\lambda=2$"].to_numpy()
g3 = df[r"$\lambda=4$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, g1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda = 1$  ({np.nanmean(rates(h, g1)[1:]):.2f})")
ax.loglog(h, g2, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda = 2$  ({np.nanmean(rates(h, g2)[1:]):.2f})")
ax.loglog(h, g3, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda = 4$  ({np.nanmean(rates(h, g3)[1:]):.2f})")

ref = 0.30 * min(g1[0], g2[0], g3[0])
ax.loglog(h, ref * (h / h[0]) ** 1, "k--", lw=1.1, label=r"$O(h^{1})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\sin\theta_{\max}$ between eigenspaces of $B$")
ax.set_title(r"B(N1): eigenfunction convergence, $p = 1$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "baseline_eigfun_BN1.pdf")
plt.show()

df = h_fun["B(N2)"]
h = df["h"].to_numpy()
g1 = df[r"$\lambda=1$"].to_numpy()
g2 = df[r"$\lambda=2$"].to_numpy()
g3 = df[r"$\lambda=4$"].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, g1, "o-", color="C0", ms=4.5, lw=1.2,
          label=rf"$\lambda = 1$  ({np.nanmean(rates(h, g1)[1:]):.2f})")
ax.loglog(h, g2, "o-", color="C3", ms=4.5, lw=1.2,
          label=rf"$\lambda = 2$  ({np.nanmean(rates(h, g2)[1:]):.2f})")
ax.loglog(h, g3, "o-", color="C1", ms=4.5, lw=1.2,
          label=rf"$\lambda = 4$  ({np.nanmean(rates(h, g3)[1:]):.2f})")

ref = 0.30 * min(g1[0], g2[0], g3[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^{2})$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\sin\theta_{\max}$ between eigenspaces of $B$")
ax.set_title(r"B(N2): eigenfunction convergence, $p = 1$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "baseline_eigfun_BN2.pdf")
plt.show()


# --------------------------------------------------------------------------
# 5. Pseudospectra for $W^{1,\infty}$ winds
# --------------------------------------------------------------------------

def wind_b2(mesh):
    """beta_2 = (x, -y): linear, solenoidal, and a gradient field."""
    x, y = SpatialCoordinate(mesh)
    return as_vector([x, -y])


def wind_b4(mesh):
    """beta_4 = (-y, x): rigid rotation about the origin; solenoidal, not a gradient."""
    x, y = SpatialCoordinate(mesh)
    return as_vector([-y, x])


PS_WINDS = {"b2": (r"$\beta_2=(x,\,-y)$", wind_b2, "gradient field"),
            "b4": (r"$\beta_4=(-y,\,x)$", wind_b4, "rigid rotation")}

RM_PS = 1.0       # see Section 6: R_m is swept later, and not here
N_PS = 24
TAU_PS = 0.9 / RM_PS
NEV_PS = 50

# Cell Peclet number: with ||beta||_inf = pi*sqrt(2) on this domain the
# unstabilised Galerkin discretisation needs h ||beta|| Rm / 2 <~ 1.
beta_sup = np.pi * np.sqrt(2)
print(f"cell Peclet = {(np.pi / N_PS) * beta_sup * RM_PS / 2:.3f}   "
      f"(must stay <~ 1; no stabilisation is used anywhere)")

mesh_w = crisscross_mesh(12)
beta_w = wind_b2(mesh_w)
field_w = Function(VectorFunctionSpace(mesh_w, "CG", 1)).interpolate(beta_w)
speed_w = Function(FunctionSpace(mesh_w, "CG", 1)).interpolate(sqrt(dot(beta_w, beta_w)))

# Arrow positions on a regular grid.  PointEvaluator is the supported replacement
# for the deprecated Function.at; the tight tolerance stops points being snapped
# into neighbouring cells.
grid_w = np.linspace(0.12, np.pi - 0.12, 13)
Xw, Yw = np.meshgrid(grid_w, grid_w)
vals_w = np.asarray(PointEvaluator(mesh_w, np.column_stack([Xw.ravel(), Yw.ravel()]),
                                   missing_points_behaviour="ignore",
                                   tolerance=1e-6).evaluate(field_w))
Uw = vals_w[:, 0].reshape(Xw.shape)
Vw = vals_w[:, 1].reshape(Xw.shape)
step_w = float(grid_w[1] - grid_w[0])

fig, ax = plt.subplots(figsize=(4.8, 4.0))
art = tripcolor(speed_w, axes=ax, cmap="viridis")
ax.quiver(Xw, Yw, Uw, Vw, color="w", angles="xy", scale_units="xy",
          scale=np.hypot(Uw, Vw).max() / (0.85 * step_w), width=0.006)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"$\beta_2 = (x,\,-y)$   (gradient field)")
fig.colorbar(art, ax=ax, label=r"$|\beta|$")
fig.savefig(FIGDIR / "baseline_wind_b2.pdf")
plt.show()

mesh_w = crisscross_mesh(12)
beta_w = wind_b4(mesh_w)
field_w = Function(VectorFunctionSpace(mesh_w, "CG", 1)).interpolate(beta_w)
speed_w = Function(FunctionSpace(mesh_w, "CG", 1)).interpolate(sqrt(dot(beta_w, beta_w)))

# Arrow positions on a regular grid.  PointEvaluator is the supported replacement
# for the deprecated Function.at; the tight tolerance stops points being snapped
# into neighbouring cells.
grid_w = np.linspace(0.12, np.pi - 0.12, 13)
Xw, Yw = np.meshgrid(grid_w, grid_w)
vals_w = np.asarray(PointEvaluator(mesh_w, np.column_stack([Xw.ravel(), Yw.ravel()]),
                                   missing_points_behaviour="ignore",
                                   tolerance=1e-6).evaluate(field_w))
Uw = vals_w[:, 0].reshape(Xw.shape)
Vw = vals_w[:, 1].reshape(Xw.shape)
step_w = float(grid_w[1] - grid_w[0])

fig, ax = plt.subplots(figsize=(4.8, 4.0))
art = tripcolor(speed_w, axes=ax, cmap="viridis")
ax.quiver(Xw, Yw, Uw, Vw, color="w", angles="xy", scale_units="xy",
          scale=np.hypot(Uw, Vw).max() / (0.85 * step_w), width=0.006)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"$\beta_4 = (-y,\,x)$   (rigid rotation)")
fig.colorbar(art, ax=ax, label=r"$|\beta|$")
fig.savefig(FIGDIR / "baseline_wind_b4.pdf")
plt.show()


# --------------------------------------------------------------------------
# 5.3 Running it
# --------------------------------------------------------------------------

def run_pseudospectrum(form, mesh, wind_key, Rm=RM_PS, tau=TAU_PS, nev=NEV_PS,
                       p=1, **kw):
    """Assemble the pencil for one configuration and project it."""
    label, wind_fn, note = PS_WINDS[wind_key]
    A, M, _ = build_pencil(form, mesh, p, beta=wind_fn(mesh), eps=Constant(1.0 / Rm))
    # Everything after this point is formulation-agnostic and lives in the module.
    return ps.pseudospectrum(A, M, tau, nev=nev, label=label,
                             form=form, wind=wind_key, Rm=Rm, **kw)


mesh_ps = crisscross_mesh(N_PS)
results = {(f, w): run_pseudospectrum(f, mesh_ps, w) for w in PS_WINDS for f in FORMS}

fmt_table(ps.diagnostics_frame(results), default="{:.3e}",
          rules=[("N", "{:.0f}"), ("m", "{:.0f}"), ("eigs in window", "{:.0f}"),
                 ("max |Im| in window", "{:.2e}"), ("max |Im| all ritz", "{:.2e}")],
          caption=r"Projection diagnostics. `QZ residual` checks the unitary "
                  r"factorisation; `window fidelity` is $\max|\lambda_{\rm SLEPc} - "
                  r"\mathrm{Ritz}|$ over the eigenvalues actually plotted.")

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ps.plot_pseudospectrum(results[("A(CG)", "b2")], ax=ax,
                           title=r"A(CG),  " + PS_WINDS["b2"][0])
    fig.savefig(FIGDIR / "baseline_pseudo_b2_ACG.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ps.plot_pseudospectrum(results[("B(N1)", "b2")], ax=ax,
                           title=r"B(N1),  " + PS_WINDS["b2"][0])
    fig.savefig(FIGDIR / "baseline_pseudo_b2_BN1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ps.plot_pseudospectrum(results[("B(N2)", "b2")], ax=ax,
                           title=r"B(N2),  " + PS_WINDS["b2"][0])
    fig.savefig(FIGDIR / "baseline_pseudo_b2_BN2.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ps.plot_pseudospectrum(results[("A(CG)", "b4")], ax=ax,
                           title=r"A(CG),  " + PS_WINDS["b4"][0])
    fig.savefig(FIGDIR / "baseline_pseudo_b4_ACG.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ps.plot_pseudospectrum(results[("B(N1)", "b4")], ax=ax,
                           title=r"B(N1),  " + PS_WINDS["b4"][0])
    fig.savefig(FIGDIR / "baseline_pseudo_b4_BN1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ps.plot_pseudospectrum(results[("B(N2)", "b4")], ax=ax,
                           title=r"B(N2),  " + PS_WINDS["b4"][0])
    fig.savefig(FIGDIR / "baseline_pseudo_b4_BN2.pdf")
    plt.show()


# --------------------------------------------------------------------------
# 6. What is deliberately not here
# --------------------------------------------------------------------------

klog = pd.DataFrame(KAPPA_LOG)
worst = klog.loc[klog[r"Hager $\kappa_1$"].idxmax()]
print(f"{len(klog)} eigensolves, each with kappa(A - tau M) estimated.\n")
print(f"  worst normwise kappa_1 : {worst[r'Hager $\kappa_1$']:.3e}"
      f"   ({worst['form']}, p={worst['p']:.0f}, {worst['N']:.0f} cells)")
print(f"  median normwise kappa_1: {klog[r'Hager $\kappa_1$'].median():.3e}")
print(f"  worst MUMPS COND1      : {klog['MUMPS COND1'].max():.3e}")
print(f"  MUMPS INFOG(1) != 0    : {int((klog['INFOG(1)'] != 0).sum())} of {len(klog)}")
print(f"\nfigures written to {FIGDIR.resolve()}")
for q in sorted(FIGDIR.glob("baseline_*.pdf")):
    print("  ", q.name)

