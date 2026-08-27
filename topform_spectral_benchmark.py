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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
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
# code serves this benchmark and the 1-form baseline.
import pseudospectra_partial_schur as ps

# `from firedrake import *` shadows the standard-library `logging`; re-import it
# under another name and silence the one warning `triplot` provokes on every call.
import logging as stdlogging
stdlogging.getLogger("firedrake").addFilter(
    lambda record: "is empty. This is likely an error" not in record.getMessage())

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
})

L_DOMAIN = np.pi          # Omega = (0, L)^2


# --- table rendering, without the jinja2 dependency of pandas' .style --------
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
    """Format a DataFrame for display.

    `rules` is a list of (column-name-substring, format spec); `row_rules` the
    same for index labels, and takes precedence -- tables here sometimes carry a
    condition-number *row* rather than a column.
    """
    def spec_for(key, table):
        name = " ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        # Exact matches win over substring matches, so a rule keyed on a short
        # name cannot capture a longer column that happens to contain it.
        return next((spec for k, spec in table if k == name),
                    next((spec for k, spec in table if k in name), None))

    out = pd.DataFrame(
        {col: [_cell(v, spec_for(idx, row_rules) or spec_for(col, rules) or default)
               for idx, v in zip(df.index, df[col])]
         for col in df.columns}, index=df.index)
    if caption:
        display(Markdown(f"**{caption}**"))
    return out


print(f"PETSc scalars: {PETSc.ScalarType.__name__}")


# --------------------------------------------------------------------------
# 2. Meshes: the criss-cross grid
# --------------------------------------------------------------------------

def crisscross_mesh(N, L=L_DOMAIN):
    r"""Criss-cross triangulation of $(0,L)^2$: $4N^2$ cells, $h = L/N$.

    `diagonal="crossed"` is Firedrake's spelling of the Union-Jack pattern -- it
    inserts a centre vertex in every grid square and cuts both diagonals.  The
    resulting mesh is symmetric under all eight symmetries of the square, which
    matters here: the spurious mode of Section 4.4 is a checkerboard pattern
    living on exactly this structure.
    """
    return SquareMesh(N, N, L, quadrilateral=False, diagonal="crossed")

mesh = crisscross_mesh(2)
markers = mesh.exterior_facets.unique_markers

fig, ax = plt.subplots(figsize=(4.2, 4.2))
# `interior_kw` reaches a PolyCollection, so it wants `edgecolors` (passing
# `color` would fill the cells); `boundary_kw["colors"]` is zipped against the
# marker list, so it needs one entry per marker.
triplot(mesh, axes=ax,
        interior_kw={"linewidths": 0.7, "edgecolors": "0.45"},
        boundary_kw={"linewidths": 1.6, "colors": ["k"] * len(markers)})
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(rf"$N = 2$:  {mesh.num_cells()} cells,  $h = \pi/2$")
fig.savefig(FIGDIR / "topform_mesh_N2.pdf")
plt.show()

mesh = crisscross_mesh(4)
markers = mesh.exterior_facets.unique_markers

fig, ax = plt.subplots(figsize=(4.2, 4.2))
# `interior_kw` reaches a PolyCollection, so it wants `edgecolors` (passing
# `color` would fill the cells); `boundary_kw["colors"]` is zipped against the
# marker list, so it needs one entry per marker.
triplot(mesh, axes=ax,
        interior_kw={"linewidths": 0.7, "edgecolors": "0.45"},
        boundary_kw={"linewidths": 1.6, "colors": ["k"] * len(markers)})
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(rf"$N = 4$:  {mesh.num_cells()} cells,  $h = \pi/4$")
fig.savefig(FIGDIR / "topform_mesh_N4.pdf")
plt.show()

mesh = crisscross_mesh(8)
markers = mesh.exterior_facets.unique_markers

fig, ax = plt.subplots(figsize=(4.2, 4.2))
# `interior_kw` reaches a PolyCollection, so it wants `edgecolors` (passing
# `color` would fill the cells); `boundary_kw["colors"]` is zipped against the
# marker list, so it needs one entry per marker.
triplot(mesh, axes=ax,
        interior_kw={"linewidths": 0.7, "edgecolors": "0.45"},
        boundary_kw={"linewidths": 1.6, "colors": ["k"] * len(markers)})
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(rf"$N = 8$:  {mesh.num_cells()} cells,  $h = \pi/8$")
fig.savefig(FIGDIR / "topform_mesh_N8.pdf")
plt.show()


# --------------------------------------------------------------------------
# 3. Velocity fields
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Wind:
    """A named background field, built lazily on a given mesh."""
    key: str
    tex: str
    note: str
    potential: bool           # beta = grad(phi)?  -> real spectrum (King)
    sup: float                # ||beta||_Linf, used for the shift nu of Lemma 4
    _build: Callable

    def __call__(self, mesh, L=L_DOMAIN):
        return self._build(mesh, L)


def _centre(mesh, L):
    x, y = SpatialCoordinate(mesh)
    return x - Constant(L / 2), y - Constant(L / 2)


WINDS = {
    "b0": Wind("b0", r"$\beta=(0,0)$", "no wind", True, 0.0,
               lambda mesh, L: None),
    "b1": Wind("b1", r"$\beta=(1,1)$", "uniform potential field", True, np.sqrt(2),
               lambda mesh, L: as_vector([Constant(1.0), Constant(1.0)])),
    "b4": Wind("b4", r"$\beta=(-\hat y,\ \hat x)$", "rigid rotation", False,
               np.pi / np.sqrt(2),
               lambda mesh, L: as_vector([-_centre(mesh, L)[1], _centre(mesh, L)[0]])),
    # NB: matplotlib's mathtext has no \tfrac, and these labels go into figure
    # titles as well as into markdown -- keep them to the common subset.
    "b7": Wind("b7", r"$\beta=(\mathrm{sign}(y-\frac{\pi}{2}),\ 0)$",
               "Heaviside shear", False, 1.0,
               lambda mesh, L: as_vector(
                   [conditional(gt(SpatialCoordinate(mesh)[1], Constant(L / 2)),
                                1.0, -1.0), Constant(0.0)])),
}

def sample_grid(f, mesh, n=15, pad=0.06):
    """Evaluate a vector `Function` on a regular n x n grid over the mesh bbox."""
    coords = mesh.coordinates.dat.data_ro
    (x0, y0), (x1, y1) = coords.min(axis=0), coords.max(axis=0)
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    X, Y = np.meshgrid(np.linspace(x0 + dx, x1 - dx, n),
                       np.linspace(y0 + dy, y1 - dy, n))
    pts = np.column_stack([X.ravel(), Y.ravel()])
    # A tight tolerance: the default is inherited from the mesh (0.5 in reference
    # coordinates) and is loose enough to snap exterior points into boundary cells.
    vals = np.asarray(PointEvaluator(mesh, pts, missing_points_behaviour="ignore",
                                     tolerance=1e-6).evaluate(f))
    return X, Y, vals[:, 0].reshape(X.shape), vals[:, 1].reshape(X.shape)

mesh_w = crisscross_mesh(12)
markers = mesh_w.exterior_facets.unique_markers

fig, ax = plt.subplots(figsize=(4.2, 4.2))
triplot(mesh_w, axes=ax, interior_kw={"linewidths": 0.3, "edgecolors": "0.8"},
        boundary_kw={"linewidths": 1.2, "colors": ["0.3"] * len(markers)})
ax.text(0.5, 0.5, r"$\beta\equiv 0$", ha="center", va="center",
        transform=ax.transAxes, fontsize=15)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("b0: " + WINDS["b0"].tex + "\n" + WINDS["b0"].note, fontsize=10)
fig.savefig(FIGDIR / "topform_wind_b0.pdf")
plt.show()

mesh_w = crisscross_mesh(12)
beta_w = WINDS["b1"](mesh_w)

# DG0 keeps the jump in b7 a jump; the quadrature sees the exact sign().
bx = Function(FunctionSpace(mesh_w, "DG", 0)).interpolate(beta_w[0])
lim = float(np.abs(bx.dat.data_ro).max()) or 1.0

bf = Function(VectorFunctionSpace(mesh_w, "CG", 1)).project(beta_w)
X, Y, U, V = sample_grid(bf, mesh_w)
speed = np.nanmax(np.hypot(U, V)) or 1.0
step = float(X[0, 1] - X[0, 0])

fig, ax = plt.subplots(figsize=(4.4, 4.2))
art = tripcolor(bx, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.quiver(X, Y, U, V, color="k", angles="xy", scale_units="xy",
          scale=speed / (0.8 * step), width=0.006, alpha=0.85)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("b1: " + WINDS["b1"].tex + "\n" + WINDS["b1"].note, fontsize=10)
fig.colorbar(art, ax=ax, label=r"$\beta_x$")
fig.savefig(FIGDIR / "topform_wind_b1.pdf")
plt.show()

mesh_w = crisscross_mesh(12)
beta_w = WINDS["b4"](mesh_w)

# DG0 keeps the jump in b7 a jump; the quadrature sees the exact sign().
bx = Function(FunctionSpace(mesh_w, "DG", 0)).interpolate(beta_w[0])
lim = float(np.abs(bx.dat.data_ro).max()) or 1.0

bf = Function(VectorFunctionSpace(mesh_w, "CG", 1)).project(beta_w)
X, Y, U, V = sample_grid(bf, mesh_w)
speed = np.nanmax(np.hypot(U, V)) or 1.0
step = float(X[0, 1] - X[0, 0])

fig, ax = plt.subplots(figsize=(4.4, 4.2))
art = tripcolor(bx, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.quiver(X, Y, U, V, color="k", angles="xy", scale_units="xy",
          scale=speed / (0.8 * step), width=0.006, alpha=0.85)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("b4: " + WINDS["b4"].tex + "\n" + WINDS["b4"].note, fontsize=10)
fig.colorbar(art, ax=ax, label=r"$\beta_x$")
fig.savefig(FIGDIR / "topform_wind_b4.pdf")
plt.show()

mesh_w = crisscross_mesh(12)
beta_w = WINDS["b7"](mesh_w)

# DG0 keeps the jump in b7 a jump; the quadrature sees the exact sign().
bx = Function(FunctionSpace(mesh_w, "DG", 0)).interpolate(beta_w[0])
lim = float(np.abs(bx.dat.data_ro).max()) or 1.0

bf = Function(VectorFunctionSpace(mesh_w, "CG", 1)).project(beta_w)
X, Y, U, V = sample_grid(bf, mesh_w)
speed = np.nanmax(np.hypot(U, V)) or 1.0
step = float(X[0, 1] - X[0, 0])

fig, ax = plt.subplots(figsize=(4.4, 4.2))
art = tripcolor(bx, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.quiver(X, Y, U, V, color="k", angles="xy", scale_units="xy",
          scale=speed / (0.8 * step), width=0.006, alpha=0.85)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("b7: " + WINDS["b7"].tex + "\n" + WINDS["b7"].note, fontsize=10)
fig.colorbar(art, ax=ax, label=r"$\beta_x$")
fig.savefig(FIGDIR / "topform_wind_b7.pdf")
plt.show()


# --------------------------------------------------------------------------
# 4. The four discretisations
# --------------------------------------------------------------------------

def primal_pencil(mesh, degree=1, beta=None, eps=Constant(1.0), nu=0.0):
    r"""`A(CG)`: $\varepsilon(\nabla u,\nabla v) - (\beta u,\nabla v) = \lambda(u,v)$."""
    V = FunctionSpace(mesh, "CG", degree)
    u, v = TrialFunction(V), TestFunction(V)

    a = eps * inner(grad(u), grad(v)) * dx
    if beta is not None:
        a -= inner(u * beta, grad(v)) * dx        # iota^n_beta u  <->  u * beta
    if nu:
        a += Constant(nu) * inner(u, v) * dx

    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
    # `weight=0.0` puts 0 (not 1) on the constrained diagonal of M, so the
    # boundary rows contribute eigenvalues at infinity rather than at lambda = 1.
    A = assemble(a, bcs=bcs).petscmat
    M = assemble(inner(u, v) * dx, bcs=bcs, weight=0.0).petscmat
    return A, M, V, None


# --------------------------------------------------------------------------
# 4.2 `B(RT)`, `B(BDM)` — the mixed total-flux system
# --------------------------------------------------------------------------

def nu_min(beta_sup, eps, C=1.0):
    r"""The shift of Lemma 4: $\nu \ge 2C^2\|\beta\|_\infty^2/\varepsilon + 2$.

    Remark 1 gives $\|\iota^n_\beta u\|_{L^2}\le\|\beta\|_{L^\infty}\|u\|_{L^2}$ for
    top forms, so $C=1$ is admissible.
    """
    return 2.0 * C**2 * beta_sup**2 / float(eps) + 2.0


def mixed_pencil(mesh, family="RT", degree=1, beta=None, eps=Constant(1.0), nu=0.0):
    r"""`B(...)` / `σ(...)`: the total-flux saddle point on $\Sigma_h \times V_h$.

    `family` is "RT" (trimmed) or "BDM" (full); both pair with $\mathrm{DG}_{r-1}$,
    which is what makes $\mathrm{d}^{n-1}\Sigma_h = V_h$ exactly.  No boundary
    condition is applied anywhere: $u|_{\partial\Omega}=0$ is natural here.
    """
    W = FunctionSpace(mesh, family, degree) * FunctionSpace(mesh, "DG", degree - 1)
    (s, u) = TrialFunctions(W)
    (t, v) = TestFunctions(W)

    a = ((1 / eps) * inner(s, t) * dx          # (eps^-1 sigma, tau)
         - inner(u, div(t)) * dx               # -(u, d^{n-1} tau)
         + inner(div(s), v) * dx)              # +(d^{n-1} sigma, v)
    if beta is not None:
        a -= (1 / eps) * inner(u * beta, t) * dx   # -(eps^-1 iota_beta u, tau)
    if nu:
        a += Constant(nu) * inner(u, v) * dx       # the shift of Lemma 3

    # The mass form touches only the u block: M is singular by construction, and
    # the sigma block supplies dim(Sigma_h) eigenvalues at infinity.
    return assemble(a).petscmat, assemble(inner(u, v) * dx).petscmat, W, None


# --------------------------------------------------------------------------
# 4.4 `σ(P1–divP1)` — the trap
# --------------------------------------------------------------------------

def crisscross_blocks(mesh, N, L=L_DOMAIN):
    """Group the 4N^2 cells into N^2 macro-squares; return (blocks, north_south).

    Cells are identified by their centroids, so nothing depends on Firedrake's
    internal cell ordering.  Within a macro-square the four sub-triangle
    centroids sit at the centre plus (+-h/3, 0) and (0, +-h/3), so |dy| > |dx|
    picks out the north/south pair.
    """
    W = VectorFunctionSpace(mesh, "DG", 0)
    centroid = Function(W).interpolate(SpatialCoordinate(mesh)).dat.data_ro
    h = L / N
    idx = np.clip(np.floor(centroid / h).astype(int), 0, N - 1)
    offset = centroid - (idx + 0.5) * h
    north_south = np.abs(offset[:, 1]) > np.abs(offset[:, 0])
    key = idx[:, 0] * N + idx[:, 1]
    blocks = np.argsort(key, kind="stable").reshape(N * N, 4)
    return blocks, north_south


def checkerboard_basis(mesh, N, L=L_DOMAIN):
    r"""Return (C, Z): a basis of $\mathrm{coker}(\operatorname{div})$ and of $V_h$.

    C : (4N^2, N^2)  the checkerboard modes, +1 on N/S and -1 on E/W per square
    Z : (4N^2, 3N^2) an orthonormal basis of their orthogonal complement, i.e. of
        $V_h = \operatorname{div}[P_1]^2$, built three vectors at a time:
        $(e_N-e_S)/\sqrt2$, $(e_E-e_W)/\sqrt2$ and $(e_N+e_S+e_E+e_W)/2$.
    """
    blocks, north_south = crisscross_blocks(mesh, N, L)
    ncell = 4 * N * N
    q = 1 / np.sqrt(2)

    rows, cols, vals = [], [], []
    for k, blk in enumerate(blocks):
        rows += list(blk); cols += [k] * 4
        vals += list(np.where(north_south[blk], 0.5, -0.5))
    C = sp.csr_matrix((vals, (rows, cols)), shape=(ncell, N * N))

    rows, cols, vals = [], [], []
    for k, blk in enumerate(blocks):
        ns, ew = blk[north_south[blk]], blk[~north_south[blk]]
        for j, (cells, v) in enumerate([(tuple(ns), (q, -q)),
                                        (tuple(ew), (q, -q)),
                                        (tuple(blk), (0.5,) * 4)]):
            rows += list(cells); cols += [3 * k + j] * len(cells); vals += list(v)
    Z = sp.csr_matrix((vals, (rows, cols)), shape=(ncell, 3 * N * N))
    return C, Z


def to_scipy(mat):
    """PETSc AIJ -> scipy CSR (serial)."""
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())


def to_petsc(S):
    """scipy sparse -> PETSc AIJ; PETSc wants sorted column indices."""
    S = sp.csr_matrix(S); S.sort_indices()
    return PETSc.Mat().createAIJWithArrays(
        S.shape, (S.indptr.astype(PETSc.IntType),
                  S.indices.astype(PETSc.IntType), S.data))


def p1divp1_pencil(mesh, N, beta=None, eps=Constant(1.0), nu=0.0, degree=1):
    r"""`σ(P1–divP1)`: the same form on $\Sigma_h=[P_1]^2$, $V_h=\operatorname{div}\Sigma_h$.

    The blocks are assembled on $\Sigma_h\times\mathrm{DG}_0$ -- Firedrake happily
    assembles a form whose two arguments live in different spaces -- and the
    $\mathrm{DG}_0$ side is then restricted to $V_h$ by the orthonormal `Z`.
    Restricting rather than working in the whole of $\mathrm{DG}_0$ matters for
    the eigensolver: the cokernel would otherwise contribute $N^2$ eigenvalues at
    $\lambda = 0$, sitting the same distance from the shift as the physical ones.

    `degree` is accepted only so that the registry can call every builder the same
    way; this pair is defined at lowest order and has no degree to raise.
    """
    S = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    s, t = TrialFunction(S), TestFunction(S)
    u, v = TrialFunction(Q), TestFunction(Q)

    A_ss = to_scipy(assemble((1 / eps) * inner(s, t) * dx).petscmat)
    form_su = -inner(u, div(t)) * dx
    if beta is not None:
        form_su = form_su - (1 / eps) * inner(u * beta, t) * dx
    A_su = to_scipy(assemble(form_su).petscmat)             # rows Sigma, cols DG0
    A_us = to_scipy(assemble(inner(div(s), v) * dx).petscmat)
    M_uu = to_scipy(assemble(inner(u, v) * dx).petscmat)

    _, Z = checkerboard_basis(mesh, N)
    A_su, A_us = A_su @ Z, Z.T @ A_us
    A_uu = Z.T @ (nu * M_uu) @ Z
    M_uu = Z.T @ M_uu @ Z

    A = sp.bmat([[A_ss, A_su], [A_us, sp.csr_matrix(A_uu)]], format="csr")
    M = sp.bmat([[sp.csr_matrix(A_ss.shape), None],
                 [None, sp.csr_matrix(M_uu)]], format="csr")
    return to_petsc(A), to_petsc(M), (S, Q), Z

# --- verification of the construction ---------------------------------------
print(" N   cells   dim[P1]^2   rank(div)   3N^2    ||div^T C||   ||Z^T Z - I||   ||Z^T C||")
for N in (2, 3, 4, 6):
    mesh = crisscross_mesh(N)
    S = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    B = to_scipy(assemble(inner(div(TrialFunction(S)), TestFunction(Q)) * dx).petscmat)
    C, Z = checkerboard_basis(mesh, N)
    rank = np.linalg.matrix_rank(B.toarray())
    print(f"{N:2d} {Q.dim():7d} {S.dim():11d} {rank:11d} {3*N*N:7d}"
          f"   {np.abs((B.T @ C).toarray()).max():.2e}"
          f"      {np.abs((Z.T @ Z - sp.eye(Z.shape[1])).toarray()).max():.2e}"
          f"        {np.abs((Z.T @ C).toarray()).max():.2e}")

N = 6
mesh = crisscross_mesh(N)
C, _ = checkerboard_basis(mesh, N)
vec = C[:, 0].toarray().ravel()
markers = mesh.exterior_facets.unique_markers

f = Function(FunctionSpace(mesh, "DG", 0))
f.dat.data[:] = vec
lim = float(np.abs(vec).max())

fig, ax = plt.subplots(figsize=(4.4, 4.2))
tripcolor(f, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
triplot(mesh, axes=ax, interior_kw={"linewidths": 0.35, "edgecolors": "0.4"},
        boundary_kw={"linewidths": 1.2, "colors": ["k"] * len(markers)})
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"$\mathrm{coker}(\operatorname{div})$: a single macro-square mode")
fig.savefig(FIGDIR / "topform_checkerboard_single.pdf")
plt.show()

N = 6
mesh = crisscross_mesh(N)
C, _ = checkerboard_basis(mesh, N)
vec = C.sum(axis=1).A.ravel()
markers = mesh.exterior_facets.unique_markers

f = Function(FunctionSpace(mesh, "DG", 0))
f.dat.data[:] = vec
lim = float(np.abs(vec).max())

fig, ax = plt.subplots(figsize=(4.4, 4.2))
tripcolor(f, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
triplot(mesh, axes=ax, interior_kw={"linewidths": 0.35, "edgecolors": "0.4"},
        boundary_kw={"linewidths": 1.2, "colors": ["k"] * len(markers)})
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"$\mathrm{coker}(\operatorname{div})$: the whole cokernel, summed")
fig.savefig(FIGDIR / "topform_checkerboard_all.pdf")
plt.show()


# --------------------------------------------------------------------------
# 5. The eigensolver and the matching pipeline
# --------------------------------------------------------------------------

EPS_OPTS = {
    "eps_gen_non_hermitian": None,       # generalised, non-Hermitian pencil
    "eps_type": "krylovschur",
    "eps_target_magnitude": None,        # order by |lambda - tau|
    "eps_tol": 1e-11,
    "eps_max_it": 5000,
    "st_type": "sinvert",                # (A - tau M)^{-1} M
    "st_ksp_type": "preonly",            # the LU *is* the solve
    "st_pc_type": "lu",
    "st_pc_factor_mat_solver_type": "mumps",
    "st_mat_mumps_icntl_14": 800,        # working-space headroom for pivoting
}
_prefix = itertools.count()

#: registry consumed by `solve_topform`
FORMS = {
    "A(CG)":       dict(build=lambda mesh, N, **kw: primal_pencil(mesh, **kw),
                        shift=False, kind="primal"),
    "B(RT)":       dict(build=lambda mesh, N, **kw: mixed_pencil(mesh, "RT", **kw),
                        shift=False, kind="mixed"),
    "B(BDM)":      dict(build=lambda mesh, N, **kw: mixed_pencil(mesh, "BDM", **kw),
                        shift=False, kind="mixed"),
    "s(RT)":       dict(build=lambda mesh, N, **kw: mixed_pencil(mesh, "RT", **kw),
                        shift=True, kind="mixed"),
    "s(BDM)":      dict(build=lambda mesh, N, **kw: mixed_pencil(mesh, "BDM", **kw),
                        shift=True, kind="mixed"),
    "s(P1-divP1)": dict(build=lambda mesh, N, **kw: p1divp1_pencil(mesh, N, **kw),
                        shift=True, kind="restricted"),
}
FEEC_FORMS = ["A(CG)", "B(RT)", "B(BDM)", "s(RT)", "s(BDM)"]


@dataclass
class Spectrum:
    values: np.ndarray          # eigenvalues, shifted back, sorted by (Re, |Im|)
    space: object               # the FunctionSpace(s) the eigenvectors live in
    restriction: object         # Z for the restricted pencil, else None
    size: int                   # pencil dimension
    meta: dict
    eps_obj: object = None
    index: np.ndarray = None    # SLEPc indices matching `values`
    kappa: dict = None          # condition estimates for the matrix ST factorised

    @property
    def real(self):
        return self.values.real

    @property
    def kappa1(self):
        r"""Normwise $\kappa_1(A-\sigma M)$ of the solve that produced this spectrum."""
        return None if self.kappa is None else self.kappa[r"Hager $\kappa_1$"]


#: every condition estimate produced in this notebook, for the tally in Section 8
KAPPA_LOG = []


def solve_topform(form="B(RT)", wind="b0", N=32, degree=1, Rm=1.0, n_eigs=100,
                  nev=None, tau=None, mesh=None, L=L_DOMAIN, cond=True):
    r"""Solve the top-form eigenproblem for one (formulation, wind, $R_m$, mesh).

    `tau` defaults to $1.9\varepsilon$, just below $\lambda_1 \ge 2\varepsilon$.  The
    shifted formulations carry $\nu$ from Lemma 4 and hand back $\tilde\lambda-\nu$,
    so the returned eigenvalues are directly comparable across all six rows of
    `FORMS`.

    Unless `cond=False`, the condition of the matrix that was actually factorised,
    $A - \sigma M$ with $\sigma = \tau + \nu$, is estimated and attached to the
    result.  It is cheap next to the eigensolve, and computing it here rather than
    on request means no table in this notebook can quietly omit it.
    """
    mesh = crisscross_mesh(N, L) if mesh is None else mesh
    eps = Constant(1.0 / Rm)
    w = WINDS[wind]
    beta = w(mesh, L)

    spec = FORMS[form]
    nu = nu_min(w.sup, 1.0 / Rm) if spec["shift"] else 0.0
    A, M, space, Z = spec["build"](mesh, N, degree=degree, beta=beta, eps=eps, nu=nu)

    tau = (1.9 / Rm) if tau is None else tau
    nev = nev or min(n_eigs + 20, A.getSize()[0] - 1)

    opts = dict(EPS_OPTS, eps_target=tau + nu,
                eps_ncv=min(2 * nev + 30, A.getSize()[0]))
    prefix = f"tf{next(_prefix)}_"
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
        if np.isfinite(z) and abs(z) < 1e10:
            found.append((i, z - nu))          # undo the Lemma 4 shift
    found = [p for p in found if abs(p[1]) > 1e-9]
    found.sort(key=lambda p: (p[1].real, abs(p[1].imag)))
    found = found[:n_eigs]

    kappa = condition_estimates(A, M, tau + nu) if cond else None
    if kappa is not None:
        KAPPA_LOG.append(dict(form=form, wind=wind, N=N, Rm=Rm, sigma=tau + nu,
                              **kappa))

    return Spectrum(np.array([z for _, z in found]), space, Z, A.getSize()[0],
                    dict(form=form, wind=wind, N=N, degree=degree, Rm=Rm,
                         eps=1.0 / Rm, tau=tau, nu=nu, kind=spec["kind"],
                         nconv=solver.getConverged(),
                         reason=solver.getConvergedReason()),
                    eps_obj=solver, index=np.array([i for i, _ in found]),
                    kappa=kappa)


def eigenfunction(spectrum, k):
    """The scalar field $u$ of the k-th eigenmode, as a Firedrake Function."""
    solver = spectrum.eps_obj
    A, _ = solver.getOperators()
    vr, vi = A.createVecRight(), A.createVecRight()
    solver.getEigenvector(int(spectrum.index[k]), vr, vi)
    arr = np.asarray(vr.getArray())
    kind, space = spectrum.meta["kind"], spectrum.space

    if kind == "restricted":                   # P1-divP1: lift V_h back to DG0
        S, Q = space
        f = Function(Q)
        f.dat.data[:] = spectrum.restriction @ arr[S.dim():]
        return f

    f = Function(space)                        # primal CG, or mixed RT/BDM x DG
    with f.dat.vec_wo as fv:
        fv.setArray(arr[:fv.getLocalSize()])
    return f.subfunctions[1] if kind == "mixed" else f

def exact_spectrum(n, Rm=1.0, beta_sq=0.0, kmax=80):
    r"""$\varepsilon(m^2+n^2) + |\beta|^2/4\varepsilon$, $m,n\ge1$, with multiplicity."""
    eps = 1.0 / Rm
    vals = sorted(m * m + k * k for m in range(1, kmax) for k in range(1, kmax))
    return np.array(vals[:n], dtype=float) * eps + beta_sq / (4.0 * eps)


def match(test, reference, rtol=3e-3):
    """Greedy one-to-one matching of `test` against `reference`.

    Returns (matched, extra): `matched[i]` is the reference partner of `test[i]`
    (NaN where none), `extra` collects the test values left over.  Only the range
    covered by both is judged, so a shorter reference window cannot manufacture
    spurious values at the top end.
    """
    ref = list(reference)
    hi = max(abs(r) for r in ref) if ref else np.inf
    matched, extra = [], []
    for v in test:
        if abs(v) > hi:
            matched.append(np.nan); continue
        if not ref:
            extra.append(v); matched.append(np.nan); continue
        j = int(np.argmin([abs(r - v) for r in ref]))
        if abs(ref[j] - v) <= rtol * max(1.0, abs(v)):
            matched.append(ref.pop(j))
        else:
            extra.append(v); matched.append(np.nan)
    return np.array(matched), np.array(extra)


# --------------------------------------------------------------------------
# 5.1 Is the shifted matrix well conditioned?
# --------------------------------------------------------------------------

def shifted_matrix(A, M, sigma):
    r"""$S = A - \sigma M$: exactly the matrix the ST hands to MUMPS."""
    S = A.copy()
    # A and M generally have different sparsity (the constrained rows of A(CG)
    # carry a diagonal that M does not), so the pattern must be declared unequal.
    S.axpy(-sigma, M, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    return S


def mumps_cond1(S, icntl14=800):
    """MUMPS' own condition estimate, RINFOG(10), via a throw-away LU + solve.

    `setFactorSetUpSolverType()` builds the factor Mat *before* the numeric
    factorisation, which is the only moment at which ICNTL(11) can still be set.
    The error analysis itself is performed during the solve, so a dummy solve is
    required to populate RINFOG.
    """
    ksp = PETSc.KSP().create(comm=S.getComm())
    ksp.setOperators(S)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    F = pc.getFactorMatrix()
    F.setMumpsIcntl(11, 1)            # 1 = full error analysis -> RINFOG(10), (11)
    F.setMumpsIcntl(14, icntl14)
    ksp.setUp()
    b, x = S.createVecLeft(), S.createVecRight()
    b.set(1.0)
    ksp.solve(b, x)
    cond, infog1 = F.getMumpsRinfog(10), F.getMumpsInfog(1)
    ksp.destroy()
    return cond, infog1


def hager_cond1(S):
    r"""Normwise $\kappa_1 = \|S\|_1\|S^{-1}\|_1$, both factors by `onenormest`."""
    Ssp = to_scipy(S).tocsc()
    lu = spla.splu(Ssp)
    n = Ssp.shape[0]
    inverse = spla.LinearOperator(
        (n, n), matvec=lu.solve, rmatvec=lambda v: lu.solve(v, trans="T"))
    return spla.onenormest(Ssp) * spla.onenormest(inverse)


def condition_estimates(A, M, sigma, exact=False):
    """Both estimates for $A-\\sigma M$ (plus dense truth when `exact`)."""
    S = shifted_matrix(A, M, sigma)
    cond, infog1 = mumps_cond1(S)
    out = {"MUMPS COND1": cond, r"Hager $\kappa_1$": hager_cond1(S),
           "INFOG(1)": infog1}
    if exact:
        dense = to_scipy(S).toarray()
        out[r"exact $\kappa_2$"] = np.linalg.cond(dense, 2)
        out[r"exact $\kappa_1$"] = np.linalg.cond(dense, 1)
    return out


def condition_of(spectrum, exact=False):
    """Condition estimates for the matrix that *this* solve actually factorised."""
    A, M = spectrum.eps_obj.getOperators()
    return condition_estimates(A, M, spectrum.meta["tau"] + spectrum.meta["nu"],
                               exact=exact)

rows = []
for N in (4, 6, 8):
    for form in ("B(RT)", "s(P1-divP1)"):
        s = solve_topform(form=form, wind="b0", N=N, n_eigs=5)
        rows.append(dict(N=N, form=form, n=s.size, **condition_of(s, exact=True)))
df_cal = pd.DataFrame(rows).set_index(["N", "form"])
df_cal = df_cal[[r"exact $\kappa_2$", r"exact $\kappa_1$", r"Hager $\kappa_1$",
                 "MUMPS COND1", "n", "INFOG(1)"]]
fmt_table(df_cal, default="{:.3e}", rules=[("n", "{:.0f}"), ("INFOG", "{:.0f}")],
          caption=r"Calibration of the two sparse estimators against dense truth, "
                  r"$\beta=0$, $\sigma = 1.9$")

mesh_c = crisscross_mesh(16)
A_c, M_c, _, _ = mixed_pencil(mesh_c, "RT", 1)
sigmas = np.concatenate([np.linspace(0.0, 1.6, 9),
                         2.0 + np.array([-0.3, -0.1, -0.02, 0.02, 0.1, 0.3, 0.8]),
                         5.0 + np.array([-0.5, -0.1, -0.02, 0.02, 0.1, 0.5, 1.5, 3.0])])
sweep = [dict(sigma=sg, **{k: v for k, v in
                           condition_estimates(A_c, M_c, sg).items() if k != "INFOG(1)"})
         for sg in sigmas]
df_sweep = pd.DataFrame(sweep)

fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.semilogy(df_sweep["sigma"], df_sweep[r"Hager $\kappa_1$"], "o-", ms=3.5,
            color="C0", label=r"Hager $\kappa_1$")
ax.semilogy(df_sweep["sigma"], df_sweep["MUMPS COND1"], "s-", ms=3.5,
            color="C1", label="MUMPS COND1")
ax.axvline(2.0, color="C3", lw=0.9, ls="--")
ax.axvline(5.0, color="C3", lw=0.9, ls="--")
ax.text(2.0, ax.get_ylim()[1], r"  $\lambda=2$", color="C3", fontsize=8, va="top")
ax.text(5.0, ax.get_ylim()[1], r"  $\lambda=5$", color="C3", fontsize=8, va="top")
ax.axvline(1.9, color="0.4", lw=1.0, ls=":")
ax.annotate(r"working shift $\tau$", xy=(1.9, 4e2), xytext=(0.15, 3e3),
            fontsize=8, color="0.3",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
ax.set_xlabel(r"shift $\sigma$")
ax.set_ylabel(r"$\kappa(A-\sigma M)$")
ax.set_title(r"B(RT), $N=16$, $\beta=0$: conditioning vs the shift")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_conditioning_shift.pdf")
plt.show()

levels = (8, 12, 16, 24, 32, 40)
h_c = np.pi / np.array(levels, dtype=float)
k_acg = [condition_of(solve_topform(form="A(CG)", wind="b0", N=N, n_eigs=3))[r"Hager $\kappa_1$"] for N in levels]
k_rt = [condition_of(solve_topform(form="B(RT)", wind="b0", N=N, n_eigs=3))[r"Hager $\kappa_1$"] for N in levels]
k_bdm = [condition_of(solve_topform(form="B(BDM)", wind="b0", N=N, n_eigs=3))[r"Hager $\kappa_1$"] for N in levels]
k_p1 = [condition_of(solve_topform(form="s(P1-divP1)", wind="b0", N=N, n_eigs=3))[r"Hager $\kappa_1$"] for N in levels]

fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.loglog(h_c, k_acg, "^-", color="C2", ms=4, label="A(CG)")
ax.loglog(h_c, k_rt, "o-", color="C0", ms=4, label="B(RT)")
ax.loglog(h_c, k_bdm, "s-", color="C1", ms=4, label="B(BDM)")
ax.loglog(h_c, k_p1, "x-", color="C3", ms=4, label=r"$\sigma(P_1$-div$P_1)$")
ax.loglog(h_c, 3e2 * (h_c / h_c[0]) ** -2, "k--", lw=0.8, label=r"$O(h^{-2})$")
ax.set_xlabel(r"$h$")
ax.set_ylabel(r"Hager $\kappa_1(A-\tau M)$")
ax.set_title(r"conditioning vs mesh size at $\tau = 1.9\varepsilon$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_conditioning_mesh.pdf")
plt.show()


# --------------------------------------------------------------------------
# 6. Experiment I — the diffusive limit $\beta = 0$
# --------------------------------------------------------------------------

N_BASE, N_EIGS = 32, 100
mesh = crisscross_mesh(N_BASE)
ALL_FORMS = list(FORMS)

runs0 = {f: solve_topform(form=f, wind="b0", N=N_BASE, mesh=mesh, n_eigs=N_EIGS)
         for f in ALL_FORMS}

diag = pd.DataFrame([
    dict(formulation=f, pencil=s.size, nu=s.meta["nu"],
         sigma=s.meta["tau"] + s.meta["nu"], converged=s.meta["nconv"],
         reason=s.meta["reason"], **{r"$\lambda_{\max}$": s.real.max()},
         **{k: v for k, v in s.kappa.items() if k != "INFOG(1)"})
    for f, s in runs0.items()]).set_index("formulation")
fmt_table(diag, default="{:.3e}",
          rules=[("pencil", "{:.0f}"), ("converged", "{:.0f}"), ("reason", "{:.0f}"),
                 ("nu", "{:.1f}"), ("sigma", "{:.2f}"), ("lambda", "{:.2f}")],
          caption=rf"Solver diagnostics at $N={N_BASE}$, $\beta=0$. $\sigma=\tau+\nu$ is "
                  r"the shift actually handed to MUMPS; `reason` $=1$ is SLEPc's "
                  r"converged flag.")

ex10 = exact_spectrum(10)
tbl = pd.DataFrame({"exact": ex10} |
                   {f: runs0[f].real[:10] for f in ALL_FORMS},
                   index=[f"$\\lambda_{{{i+1}}}$" for i in range(10)])
# every results table in this notebook closes with the conditioning of the matrix
# that produced it
tbl.loc[r"$\kappa_1(A-\sigma M)$"] = [np.nan] + [runs0[f].kappa1 for f in ALL_FORMS]
fmt_table(tbl, row_rules=[(r"\kappa_1", "{:.2e}")],
          caption=r"First ten eigenvalues, $\beta = 0$, $R_m = 1$, "
                       rf"criss-cross $N = {N_BASE}$, lowest order")

for a, b in [("B(RT)", "s(RT)"), ("B(BDM)", "s(BDM)")]:
    d = np.abs(runs0[a].values[:60] - runs0[b].values[:60]).max()
    print(f"  max |{a} - {b}| over the first 60 eigenvalues:  {d:.3e}"
          f"   (nu = {runs0[b].meta['nu']:.1f})")

ex = exact_spectrum(N_EIGS + 40)
STYLE = {"A(CG)": ("C2", "^"), "B(RT)": ("C0", "o"), "B(BDM)": ("C1", "s"),
         "s(RT)": ("C0", "o"), "s(BDM)": ("C1", "s"), "s(P1-divP1)": ("C3", "x")}
show = ["A(CG)", "B(RT)", "B(BDM)", "s(P1-divP1)"]
idx = np.arange(1, N_EIGS + 1)

fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.plot(idx, ex[:N_EIGS], "k-", lw=1.0, label=r"exact $m^2+n^2$")
ax.plot(idx, runs0["A(CG)"].real[:N_EIGS], "^", ms=3.2, color="C2", mfc="none", label="A(CG)")
ax.plot(idx, runs0["B(RT)"].real[:N_EIGS], "o", ms=3.2, color="C0", mfc="none", label="B(RT)")
ax.plot(idx, runs0["B(BDM)"].real[:N_EIGS], "s", ms=3.2, color="C1", mfc="none", label="B(BDM)")
ax.plot(idx, runs0["s(P1-divP1)"].real[:N_EIGS], "x", ms=3.2, color="C3", label=r"$\sigma(P_1$-div$P_1)$")
ax.set_xlabel(r"index $j$")
ax.set_ylabel(r"$\lambda_j$")
ax.set_title(rf"the spectral staircase, $\beta=0$, $N={N_BASE}$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_staircase.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.semilogy(idx, np.abs(runs0["A(CG)"].real[:N_EIGS] - ex[:N_EIGS]) / ex[:N_EIGS],
            "^", ms=3.2, color="C2", mfc="none", label="A(CG)")
ax.semilogy(idx, np.abs(runs0["B(RT)"].real[:N_EIGS] - ex[:N_EIGS]) / ex[:N_EIGS],
            "o", ms=3.2, color="C0", mfc="none", label="B(RT)")
ax.semilogy(idx, np.abs(runs0["B(BDM)"].real[:N_EIGS] - ex[:N_EIGS]) / ex[:N_EIGS],
            "s", ms=3.2, color="C1", mfc="none", label="B(BDM)")
ax.semilogy(idx, np.abs(runs0["s(P1-divP1)"].real[:N_EIGS] - ex[:N_EIGS]) / ex[:N_EIGS],
            "x", ms=3.2, color="C3", label=r"$\sigma(P_1$-div$P_1)$")
ax.set_xlabel(r"index $j$")
ax.set_ylabel(r"$|\lambda_j - \lambda_j^{\rm exact}|/\lambda_j^{\rm exact}$")
ax.set_title(r"relative error against the $j$-th exact eigenvalue")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_staircase_error.pdf")
plt.show()


# --------------------------------------------------------------------------
# 6.1 Convergence of the genuine spectrum
# --------------------------------------------------------------------------

DX_HI = dx(metadata={"quadrature_degree": 10})
N_TRACK = 5           # eigenvalues followed individually

#: Exact eigenvalue clusters of the Dirichlet Laplacian on (0,pi)^2 covering the
#: first five eigenvalues, with the index pairs whose sin(mx)sin(ny) span each
#: eigenspace.  The lambda = 10 pair is carried whole rather than cut in half.
CLUSTERS = [(2.0, [(1, 1)]),
            (5.0, [(1, 2), (2, 1)]),
            (8.0, [(2, 2)]),
            (10.0, [(1, 3), (3, 1)])]


def exact_mode(mesh, m, n):
    """The Dirichlet eigenfunction u_mn = sin(mx) sin(ny)."""
    x, y = SpatialCoordinate(mesh)
    return sin(m * x) * sin(n * y)


def rates(h, e):
    """Observed order between consecutive refinement levels."""
    h, e = np.asarray(h, float), np.asarray(e, float)
    r = np.full(e.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])
    return r


def assign(exact, computed):
    """Greedy one-to-one: the computed value nearest each exact eigenvalue.

    Needed because `s(P1-divP1)` interleaves spurious values with the genuine
    ones, so the j-th computed eigenvalue is not the approximation to the j-th
    exact one.
    """
    pool = list(computed)
    out = []
    for value in exact:
        j = int(np.argmin([abs(c - value) for c in pool]))
        out.append(pool.pop(j))
    return np.array(out)


def _whiten(G):
    """W with W^T G W = I, dropping numerically null directions.

    An eigendecomposition rather than a Cholesky: on coarse meshes the computed
    basis of a cluster can be close to rank deficient, where a Cholesky fails.
    """
    w, V = np.linalg.eigh(G)
    keep = w > w.max() * 1e-13
    return V[:, keep] / np.sqrt(w[keep])


def subspace_error(computed, exact):
    """Sine of the largest principal angle between two L2 subspaces (the gap).

    With orthonormalised bases the singular values of the cross-Gram matrix are
    the cosines of the principal angles, so the largest angle is set by the
    smallest singular value.
    """
    gram = lambda P, Q: np.array([[assemble(inner(a, b) * DX_HI) for b in Q]
                                  for a in P])
    Wc, We = _whiten(gram(computed, computed)), _whiten(gram(exact, exact))
    sv = np.linalg.svd(Wc.T @ gram(computed, exact) @ We, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - min(sv.min(), 1.0) ** 2)))


def convergence_study(form, levels, degree=1):
    """Eigenvalue errors for the first N_TRACK modes, and the eigenspace gaps."""
    exact = exact_spectrum(N_TRACK)
    rows_e, rows_f = [], []
    for N in levels:
        mesh = crisscross_mesh(N)
        res = solve_topform(form=form, wind="b0", N=N, mesh=mesh, degree=degree,
                            n_eigs=25, nev=40)
        lam = res.real
        got = assign(exact, lam)
        rows_e.append(dict(N=N, h=np.pi / N, dof=res.size,
                           **{f"$\\lambda_{{{j+1}}}$": abs(got[j] - exact[j]) / exact[j]
                              for j in range(N_TRACK)},
                           **{r"$\kappa_1$": res.kappa1}))

        row = dict(N=N, h=np.pi / N)
        for value, modes in CLUSTERS:
            idx = np.argsort(np.abs(lam - value))[:len(modes)]
            row[f"$\\lambda={value:g}$"] = subspace_error(
                [eigenfunction(res, int(i)) for i in idx],
                [exact_mode(mesh, m, n) for m, n in modes])
        row[r"$\kappa_1$"] = res.kappa1
        rows_f.append(row)
    return (pd.DataFrame(rows_e).set_index("N"),
            pd.DataFrame(rows_f).set_index("N"))


CONV_LEVELS = (4, 8, 16, 24, 32)
CONV_FORMS = ["A(CG)", "B(RT)", "B(BDM)", "s(P1-divP1)"]
conv_eig, conv_fun = {}, {}
for f in CONV_FORMS:
    conv_eig[f], conv_fun[f] = convergence_study(f, CONV_LEVELS)

def rate_frame(df, cols):
    """Interleave each error column with its observed rate."""
    out = pd.DataFrame(index=df.index)
    out["h"] = df["h"]
    for c in cols:
        out[c] = df[c]
        out[f"rate {c}"] = rates(df["h"], df[c])
    out[r"$\kappa_1$"] = df[r"$\kappa_1$"]
    return out


EIG_COLS = [f"$\\lambda_{{{j+1}}}$" for j in range(N_TRACK)]
FUN_COLS = [f"$\\lambda={v:g}$" for v, _ in CLUSTERS]

for f in CONV_FORMS:
    display(fmt_table(
        rate_frame(conv_eig[f], EIG_COLS), default="{:.3e}",
        rules=[("rate", "{:.2f}"), ("h", "{:.5f}"), (r"$\kappa_1$", "{:.2e}")],
        caption=f"$h$-refinement, **{f}**, $\\beta=0$: relative error in the first "
                f"{N_TRACK} genuine eigenvalues (expected $O(h^2)$)"))

for f in CONV_FORMS:
    display(fmt_table(
        rate_frame(conv_fun[f], FUN_COLS), default="{:.3e}",
        rules=[("rate", "{:.2f}"), ("h", "{:.5f}"), (r"$\kappa_1$", "{:.2e}")],
        caption=f"Eigenfunction convergence, **{f}**, $\\beta=0$: gap between the "
                r"computed and exact eigenspaces of $u$"))

EIG_COLOURS = ["C0", "C1", "C2", "C3", "C4"]
FUN_COLOURS = ["C0", "C3", "C1", "C2"]

df = conv_eig["A(CG)"]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"A(CG): eigenvalue convergence, $\beta = 0$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "topform_conv_eig_A_CG.pdf")
plt.show()

df = conv_eig["B(RT)"]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"B(RT): eigenvalue convergence, $\beta = 0$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "topform_conv_eig_B_RT.pdf")
plt.show()

df = conv_eig["B(BDM)"]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"B(BDM): eigenvalue convergence, $\beta = 0$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "topform_conv_eig_B_BDM.pdf")
plt.show()

df = conv_eig["s(P1-divP1)"]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_{1}$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_{2}$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_{3}$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_{4}$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_{5}$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"s(P1-divP1): eigenvalue convergence, $\beta = 0$")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "topform_conv_eig_s_P1divP1.pdf")
plt.show()

df = conv_fun["A(CG)"]
h = df["h"].to_numpy()
g1 = df[FUN_COLS[0]].to_numpy()
g2 = df[FUN_COLS[1]].to_numpy()
g3 = df[FUN_COLS[2]].to_numpy()
g4 = df[FUN_COLS[3]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, g1, "o-", color=FUN_COLOURS[0], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[0]}  ({np.nanmean(rates(h, g1)[1:]):.2f})")
ax.loglog(h, g2, "o-", color=FUN_COLOURS[1], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[1]}  ({np.nanmean(rates(h, g2)[1:]):.2f})")
ax.loglog(h, g3, "o-", color=FUN_COLOURS[2], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[2]}  ({np.nanmean(rates(h, g3)[1:]):.2f})")
ax.loglog(h, g4, "o-", color=FUN_COLOURS[3], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[3]}  ({np.nanmean(rates(h, g4)[1:]):.2f})")

ref = 0.30 * min(g1[0], g2[0], g3[0], g4[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\sin\theta_{\max}$ between eigenspaces of $u$")
ax.set_title(r"A(CG): eigenfunction convergence, $\beta = 0$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_conv_fun_A_CG.pdf")
plt.show()

df = conv_fun["B(RT)"]
h = df["h"].to_numpy()
g1 = df[FUN_COLS[0]].to_numpy()
g2 = df[FUN_COLS[1]].to_numpy()
g3 = df[FUN_COLS[2]].to_numpy()
g4 = df[FUN_COLS[3]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, g1, "o-", color=FUN_COLOURS[0], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[0]}  ({np.nanmean(rates(h, g1)[1:]):.2f})")
ax.loglog(h, g2, "o-", color=FUN_COLOURS[1], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[1]}  ({np.nanmean(rates(h, g2)[1:]):.2f})")
ax.loglog(h, g3, "o-", color=FUN_COLOURS[2], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[2]}  ({np.nanmean(rates(h, g3)[1:]):.2f})")
ax.loglog(h, g4, "o-", color=FUN_COLOURS[3], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[3]}  ({np.nanmean(rates(h, g4)[1:]):.2f})")

ref = 0.30 * min(g1[0], g2[0], g3[0], g4[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\sin\theta_{\max}$ between eigenspaces of $u$")
ax.set_title(r"B(RT): eigenfunction convergence, $\beta = 0$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_conv_fun_B_RT.pdf")
plt.show()

df = conv_fun["B(BDM)"]
h = df["h"].to_numpy()
g1 = df[FUN_COLS[0]].to_numpy()
g2 = df[FUN_COLS[1]].to_numpy()
g3 = df[FUN_COLS[2]].to_numpy()
g4 = df[FUN_COLS[3]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, g1, "o-", color=FUN_COLOURS[0], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[0]}  ({np.nanmean(rates(h, g1)[1:]):.2f})")
ax.loglog(h, g2, "o-", color=FUN_COLOURS[1], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[1]}  ({np.nanmean(rates(h, g2)[1:]):.2f})")
ax.loglog(h, g3, "o-", color=FUN_COLOURS[2], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[2]}  ({np.nanmean(rates(h, g3)[1:]):.2f})")
ax.loglog(h, g4, "o-", color=FUN_COLOURS[3], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[3]}  ({np.nanmean(rates(h, g4)[1:]):.2f})")

ref = 0.30 * min(g1[0], g2[0], g3[0], g4[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\sin\theta_{\max}$ between eigenspaces of $u$")
ax.set_title(r"B(BDM): eigenfunction convergence, $\beta = 0$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_conv_fun_B_BDM.pdf")
plt.show()

df = conv_fun["s(P1-divP1)"]
h = df["h"].to_numpy()
g1 = df[FUN_COLS[0]].to_numpy()
g2 = df[FUN_COLS[1]].to_numpy()
g3 = df[FUN_COLS[2]].to_numpy()
g4 = df[FUN_COLS[3]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, g1, "o-", color=FUN_COLOURS[0], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[0]}  ({np.nanmean(rates(h, g1)[1:]):.2f})")
ax.loglog(h, g2, "o-", color=FUN_COLOURS[1], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[1]}  ({np.nanmean(rates(h, g2)[1:]):.2f})")
ax.loglog(h, g3, "o-", color=FUN_COLOURS[2], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[2]}  ({np.nanmean(rates(h, g3)[1:]):.2f})")
ax.loglog(h, g4, "o-", color=FUN_COLOURS[3], ms=4.5, lw=1.2,
          label=f"{FUN_COLS[3]}  ({np.nanmean(rates(h, g4)[1:]):.2f})")

ref = 0.30 * min(g1[0], g2[0], g3[0], g4[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\sin\theta_{\max}$ between eigenspaces of $u$")
ax.set_title(r"s(P1-divP1): eigenfunction convergence, $\beta = 0$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_conv_fun_s_P1divP1.pdf")
plt.show()


# --------------------------------------------------------------------------
# 6.2 Isolating the spurious modes
# --------------------------------------------------------------------------

_, extra = match(runs0["s(P1-divP1)"].real[:60], runs0["B(BDM)"].real, rtol=3e-3)
print("values in s(P1-divP1) with no counterpart in B(BDM) on the same mesh:")
print("  ", np.round(extra, 5))
print("\nnearest exact eigenvalues:      ", exact_spectrum(24).astype(int))


# --------------------------------------------------------------------------
# 6.3 Do the spurious values converge?
# --------------------------------------------------------------------------

LEVELS = (8, 12, 16, 24, 32, 40, 48)
rows = []
for N in LEVELS:
    s = solve_topform(form="s(P1-divP1)", wind="b0", N=N, n_eigs=40)
    v = s.real
    rows.append(dict(N=N, h=np.pi / N,
                     **{r"$\lambda^{sp}_1$": v[np.argmin(np.abs(v - 6.0))],
                        r"$\lambda^{sp}_2$": v[np.argmin(np.abs(v - 15.0))],
                        r"$\kappa_1$": s.kappa1}))
df_sp = pd.DataFrame(rows).set_index("N")
df_sp[r"err $\lambda^{sp}_1$"] = np.abs(df_sp[r"$\lambda^{sp}_1$"] - 6.0)
df_sp[r"err $\lambda^{sp}_2$"] = np.abs(df_sp[r"$\lambda^{sp}_2$"] - 15.0)
fmt_table(df_sp, default="{:.6f}",
          rules=[("err", "{:.3e}"), ("h", "{:.4f}"), (r"$\kappa_1$", "{:.2e}")],
          caption=r"The two lowest spurious eigenvalues converge to $6$ and $15$, "
                  r"neither of which is of the form $m^2+n^2$")

h = df_sp["h"].to_numpy()

fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.loglog(h, np.abs(df_sp[r"$\lambda^{sp}_1$"] - 6.0), "o-", color="C3", ms=4,
          label=r"$|\lambda^{sp} - 6|$")
ax.loglog(h, np.abs(df_sp[r"$\lambda^{sp}_2$"] - 15.0), "s-", color="C4", ms=4,
          label=r"$|\lambda^{sp} - 15|$")
ax.loglog(h, 0.5 * (h / h[0]) ** 2, "k--", lw=0.8, label=r"$O(h^2)$")
ax.set_xlabel(r"$h$")
ax.set_ylabel("distance to the spurious limit")
ax.set_title("the spurious values converge — to 6 and 15")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_spurious_convergence.pdf")
plt.show()

h = df_sp["h"].to_numpy()

fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.semilogx(h, df_sp[r"$\lambda^{sp}_1$"], "o-", color="C3", ms=4,
            label=r"$\lambda^{sp}_1$")
ax.semilogx(h, df_sp[r"$\lambda^{sp}_2$"], "s-", color="C4", ms=4,
            label=r"$\lambda^{sp}_2$")
for y in (5, 6, 8, 13, 15, 17):
    ax.axhline(y, color="0.85", lw=0.7, zorder=-1)
    ax.text(h[0] * 1.05, y, f"{y}", fontsize=7, va="center", color="0.5")
ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\lambda^{sp}$")
ax.set_title("relative to the exact spectrum (grey lines)")
ax.legend(fontsize=8, frameon=False)
fig.savefig(FIGDIR / "topform_spurious_levels.pdf")
plt.show()


# --------------------------------------------------------------------------
# 6.4 What do the spurious modes look like?
# --------------------------------------------------------------------------

def plot_mode(ax, f, title, cmap="RdBu_r"):
    """Draw one scalar eigenmode with a symmetric colour scale."""
    d = f.dat.data_ro
    lim = float(np.abs(d).max()) or 1.0
    if d[int(np.argmax(np.abs(d)))] < 0:            # fix an arbitrary global sign
        f.dat.data[:] = -d
    tripcolor(f, axes=ax, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


s_p1 = solve_topform(form="s(P1-divP1)", wind="b0", N=24, n_eigs=20)
s_rt = solve_topform(form="B(BDM)", wind="b0", N=24, n_eigs=20)
_, extra24 = match(s_p1.real[:20], s_rt.real, rtol=3e-3)
sp_idx = [int(np.argmin(np.abs(s_p1.real - v))) for v in extra24[:2]]

fig, ax = plt.subplots(figsize=(4.2, 4.2))
plot_mode(ax, eigenfunction(s_rt, 0),
          rf"B(BDM), $\lambda_1={s_rt.real[0]:.4f}$ (genuine)")
fig.savefig(FIGDIR / "topform_mode_bdm_genuine.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(4.2, 4.2))
plot_mode(ax, eigenfunction(s_p1, 0),
          rf"$P_1$-div$P_1$, $\lambda_1={s_p1.real[0]:.4f}$ (genuine)")
fig.savefig(FIGDIR / "topform_mode_p1_genuine.pdf")
plt.show()

k = sp_idx[0]

fig, ax = plt.subplots(figsize=(4.2, 4.2))
plot_mode(ax, eigenfunction(s_p1, k),
          rf"$P_1$-div$P_1$, $\lambda={s_p1.real[k]:.4f}$ (SPURIOUS)")
fig.savefig(FIGDIR / "topform_mode_p1_spurious1.pdf")
plt.show()

k = sp_idx[1]

fig, ax = plt.subplots(figsize=(4.2, 4.2))
plot_mode(ax, eigenfunction(s_p1, k),
          rf"$P_1$-div$P_1$, $\lambda={s_p1.real[k]:.4f}$ (SPURIOUS)")
fig.savefig(FIGDIR / "topform_mode_p1_spurious2.pdf")
plt.show()

rows = []
for N in (32, 48, 64):
    ref = solve_topform(form="B(BDM)", wind="b0", N=N, n_eigs=90)
    bad = solve_topform(form="s(P1-divP1)", wind="b0", N=N, n_eigs=90)
    _, extra = match(bad.real, ref.real, rtol=3e-3)
    predicted = 3.0 * exact_spectrum(len(extra))
    k = min(8, len(extra))
    rows.append(dict(N=N, h=np.pi / N,
                     **{f"$\\lambda^{{sp}}_{{{j+1}}}$": extra[j] for j in range(6)},
                     **{"max rel dev": np.max(np.abs(extra[:k] - predicted[:k]) / predicted[:k]),
                        r"$\kappa_1$": bad.kappa1}))
df_branch = pd.DataFrame(rows).set_index("N")
df_branch.loc["$3(m^2+n^2)$"] = [np.nan] + list(3.0 * exact_spectrum(6)) + [np.nan, np.nan]
fmt_table(df_branch, default="{:.4f}",
          rules=[("max rel dev", "{:.2e}"), ("h", "{:.4f}"), (r"$\kappa_1$", "{:.2e}")],
          caption=r"The spurious branch of $\sigma(P_1$–div$P_1)$ against the prediction "
                  r"$3\varepsilon(m^2+n^2)$, with the same multiplicities "
                  r"(simple, double, double, simple, double, double, ...)")


# --------------------------------------------------------------------------
# 7. Experiment II — the advective regime $\beta \neq 0$
# --------------------------------------------------------------------------

N_ADV = 40
mesh_adv = crisscross_mesh(N_ADV)
beta_sq = 2.0                                    # |(1,1)|^2

runs_b1 = {f: solve_topform(form=f, wind="b1", N=N_ADV, mesh=mesh_adv, n_eigs=40,
                            tau=0.9 * (2.0 + beta_sq / 4.0))
           for f in ALL_FORMS}

ex_b1 = exact_spectrum(8, beta_sq=beta_sq)
tbl = pd.DataFrame({"exact": ex_b1} | {f: runs_b1[f].real[:8] for f in ALL_FORMS},
                   index=[f"$\\lambda_{{{i+1}}}$" for i in range(8)])
tbl.loc[r"$\kappa_1(A-\sigma M)$"] = [np.nan] + [runs_b1[f].kappa1 for f in ALL_FORMS]
fmt_table(tbl, row_rules=[(r"\kappa_1", "{:.2e}")],
          caption=r"$\beta = (1,1)$, $R_m = 1$: exact spectrum is "
                       r"$m^2+n^2+\tfrac{1}{2}$; " rf"criss-cross $N={N_ADV}$")

print("King's Theorem 7: a potential field gives a real, non-negative spectrum.")
print("Zeldovich (Theorem 4): a solenoidal field in 2D gives Re(lambda) > 0.\n")
print(f"{'form':13s} {'max |Im lambda|':>16s} {'min Re lambda':>15s}")
for f in ALL_FORMS:
    v = runs_b1[f].values[:40]
    print(f"{f:13s} {np.abs(v.imag).max():16.3e} {v.real.min():15.6f}")

runs0_adv = {f: solve_topform(form=f, wind="b0", N=N_ADV, mesh=mesh_adv, n_eigs=40)
             for f in ("B(BDM)", "s(P1-divP1)")}
offset = beta_sq / 4.0

rows = []
for f in ("B(BDM)", "s(P1-divP1)"):
    quiet, windy = runs0_adv[f].real, runs_b1[f].real
    for j, v in enumerate(quiet[:10]):
        predicted = v + offset
        actual = windy[int(np.argmin(np.abs(windy - predicted)))]
        rows.append(dict(form=f, j=j + 1, **{r"$\lambda(\beta=0)$": v,
                                             "predicted": predicted,
                                             "computed": actual,
                                             "discrepancy": abs(actual - predicted),
                                             r"$\kappa_1$": runs_b1[f].kappa1}))
df_shift = pd.DataFrame(rows).set_index(["form", "j"])
fmt_table(df_shift, rules=[("discrepancy", "{:.2e}"), (r"$\kappa_1$", "{:.2e}")],
          caption=r"Rigid-shift test: with constant $\beta$ every genuine eigenvalue must "
                  r"move by exactly $|\beta|^2/4\varepsilon = 0.5$")


# --------------------------------------------------------------------------
# 7.2 Non-potential fields: the spectrum leaves the real axis
# --------------------------------------------------------------------------

ref = solve_topform(form="B(BDM)", wind="b4", N=N_ADV, mesh=mesh_adv, n_eigs=45)
bad = solve_topform(form="s(P1-divP1)", wind="b4", N=N_ADV, mesh=mesh_adv, n_eigs=45)
_, extra = match(bad.values[:30], ref.values, rtol=3e-3)

fig, ax = plt.subplots(figsize=(5.8, 4.6))
ax.scatter(ref.values.real, ref.values.imag, s=46, marker="o", facecolors="none",
           edgecolors="C1", linewidths=1.2, label="B(BDM)  (FEEC)")
ax.scatter(bad.values.real, bad.values.imag, s=20, marker="x", color="C0",
           linewidths=1.0, label=r"$\sigma(P_1$-div$P_1)$")
if len(extra):
    ax.scatter(extra.real, extra.imag, s=150, marker="o", facecolors="none",
               edgecolors="C3", linewidths=1.6, label="spurious")
ax.axhline(0, color="0.88", lw=0.6, zorder=-2)
ax.set_xlim(0, 30)
ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax.set_title(f"b4: " + WINDS["b4"].tex + f"   ({WINDS['b4'].note}),  "
             rf"$R_m = 1$, $N = {N_ADV}$", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.savefig(FIGDIR / "topform_complex_plane_b4.pdf")
plt.show()

ref = solve_topform(form="B(BDM)", wind="b7", N=N_ADV, mesh=mesh_adv, n_eigs=45)
bad = solve_topform(form="s(P1-divP1)", wind="b7", N=N_ADV, mesh=mesh_adv, n_eigs=45)
_, extra = match(bad.values[:30], ref.values, rtol=3e-3)

fig, ax = plt.subplots(figsize=(5.8, 4.6))
ax.scatter(ref.values.real, ref.values.imag, s=46, marker="o", facecolors="none",
           edgecolors="C1", linewidths=1.2, label="B(BDM)  (FEEC)")
ax.scatter(bad.values.real, bad.values.imag, s=20, marker="x", color="C0",
           linewidths=1.0, label=r"$\sigma(P_1$-div$P_1)$")
if len(extra):
    ax.scatter(extra.real, extra.imag, s=150, marker="o", facecolors="none",
               edgecolors="C3", linewidths=1.6, label="spurious")
ax.axhline(0, color="0.88", lw=0.6, zorder=-2)
ax.set_xlim(0, 30)
ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax.set_title(f"b7: " + WINDS["b7"].tex + f"   ({WINDS['b7'].note}),  "
             rf"$R_m = 1$, $N = {N_ADV}$", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.savefig(FIGDIR / "topform_complex_plane_b7.pdf")
plt.show()


# --------------------------------------------------------------------------
# 7.3 Sweeping the magnetic Reynolds number
# --------------------------------------------------------------------------

RMS = (1.0, 2.0, 5.0, 10.0, 20.0)
rows = []
for Rm in RMS:
    ref = solve_topform(form="B(BDM)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=45)
    bad = solve_topform(form="s(P1-divP1)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=45)
    _, extra = match(bad.values[:25], ref.values, rtol=3e-3)
    rows.append(dict(Rm=Rm, Pe=np.pi / N_ADV * Rm / 2,
                     **{r"genuine $\lambda_1$": ref.values[0],
                        r"first spurious": extra[0] if len(extra) else None,
                        "ratio": (extra[0].real / ref.values[0].real) if len(extra) else None,
                        "n spurious in 25": len(extra),
                        r"$\kappa_1$ B(BDM)": ref.kappa1,
                        r"$\kappa_1$ P1": bad.kappa1}))
df_rm = pd.DataFrame(rows).set_index("Rm")
fmt_table(df_rm, default="{:.5f}", rules=[("Pe", "{:.3f}"), ("ratio", "{:.3f}"),
                                          ("n spurious", "{:.0f}"), (r"$\kappa_1$", "{:.2e}")],
          caption=rf"Heaviside shear `b7`, criss-cross $N={N_ADV}$. "
                  r"$\mathrm{Pe}_h = h\|\beta\|R_m/2$ stays below 1 throughout.")

Rm = 1.0
ref = solve_topform(form="B(BDM)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
bad = solve_topform(form="s(P1-divP1)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
_, extra = match(bad.values[:22], ref.values, rtol=3e-3)

fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.scatter(ref.values.real, ref.values.imag, s=34, marker="o", facecolors="none",
           edgecolors="C1", linewidths=1.0, label="B(BDM)")
ax.scatter(bad.values.real, bad.values.imag, s=14, marker="x", color="C0",
           linewidths=0.9, label=r"$P_1$-div$P_1$")
if len(extra):
    ax.scatter(extra.real, extra.imag, s=110, marker="o", facecolors="none",
               edgecolors="C3", linewidths=1.5, label="spurious")
ax.set_xlim(0, 8.0 / Rm + 2.0)
ax.axhline(0, color="0.9", lw=0.6, zorder=-2)
ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax.set_title(rf"Heaviside shear, $R_m = {Rm:g}$")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.savefig(FIGDIR / "topform_rm_1.pdf")
plt.show()

Rm = 2.0
ref = solve_topform(form="B(BDM)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
bad = solve_topform(form="s(P1-divP1)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
_, extra = match(bad.values[:22], ref.values, rtol=3e-3)

fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.scatter(ref.values.real, ref.values.imag, s=34, marker="o", facecolors="none",
           edgecolors="C1", linewidths=1.0, label="B(BDM)")
ax.scatter(bad.values.real, bad.values.imag, s=14, marker="x", color="C0",
           linewidths=0.9, label=r"$P_1$-div$P_1$")
if len(extra):
    ax.scatter(extra.real, extra.imag, s=110, marker="o", facecolors="none",
               edgecolors="C3", linewidths=1.5, label="spurious")
ax.set_xlim(0, 8.0 / Rm + 2.0)
ax.axhline(0, color="0.9", lw=0.6, zorder=-2)
ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax.set_title(rf"Heaviside shear, $R_m = {Rm:g}$")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.savefig(FIGDIR / "topform_rm_2.pdf")
plt.show()

Rm = 5.0
ref = solve_topform(form="B(BDM)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
bad = solve_topform(form="s(P1-divP1)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
_, extra = match(bad.values[:22], ref.values, rtol=3e-3)

fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.scatter(ref.values.real, ref.values.imag, s=34, marker="o", facecolors="none",
           edgecolors="C1", linewidths=1.0, label="B(BDM)")
ax.scatter(bad.values.real, bad.values.imag, s=14, marker="x", color="C0",
           linewidths=0.9, label=r"$P_1$-div$P_1$")
if len(extra):
    ax.scatter(extra.real, extra.imag, s=110, marker="o", facecolors="none",
               edgecolors="C3", linewidths=1.5, label="spurious")
ax.set_xlim(0, 8.0 / Rm + 2.0)
ax.axhline(0, color="0.9", lw=0.6, zorder=-2)
ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax.set_title(rf"Heaviside shear, $R_m = {Rm:g}$")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.savefig(FIGDIR / "topform_rm_5.pdf")
plt.show()

Rm = 10.0
ref = solve_topform(form="B(BDM)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
bad = solve_topform(form="s(P1-divP1)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
_, extra = match(bad.values[:22], ref.values, rtol=3e-3)

fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.scatter(ref.values.real, ref.values.imag, s=34, marker="o", facecolors="none",
           edgecolors="C1", linewidths=1.0, label="B(BDM)")
ax.scatter(bad.values.real, bad.values.imag, s=14, marker="x", color="C0",
           linewidths=0.9, label=r"$P_1$-div$P_1$")
if len(extra):
    ax.scatter(extra.real, extra.imag, s=110, marker="o", facecolors="none",
               edgecolors="C3", linewidths=1.5, label="spurious")
ax.set_xlim(0, 8.0 / Rm + 2.0)
ax.axhline(0, color="0.9", lw=0.6, zorder=-2)
ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax.set_title(rf"Heaviside shear, $R_m = {Rm:g}$")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.savefig(FIGDIR / "topform_rm_10.pdf")
plt.show()

Rm = 20.0
ref = solve_topform(form="B(BDM)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
bad = solve_topform(form="s(P1-divP1)", wind="b7", N=N_ADV, mesh=mesh_adv, Rm=Rm, n_eigs=40)
_, extra = match(bad.values[:22], ref.values, rtol=3e-3)

fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.scatter(ref.values.real, ref.values.imag, s=34, marker="o", facecolors="none",
           edgecolors="C1", linewidths=1.0, label="B(BDM)")
ax.scatter(bad.values.real, bad.values.imag, s=14, marker="x", color="C0",
           linewidths=0.9, label=r"$P_1$-div$P_1$")
if len(extra):
    ax.scatter(extra.real, extra.imag, s=110, marker="o", facecolors="none",
               edgecolors="C3", linewidths=1.5, label="spurious")
ax.set_xlim(0, 8.0 / Rm + 2.0)
ax.axhline(0, color="0.9", lw=0.6, zorder=-2)
ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax.set_title(rf"Heaviside shear, $R_m = {Rm:g}$")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.savefig(FIGDIR / "topform_rm_20.pdf")
plt.show()


# --------------------------------------------------------------------------
# 7.4 Does the resolvent see the pollution?
# --------------------------------------------------------------------------

PS_N, PS_TAU, PS_NEV = 24, 1.9, 60
PS_WINDOW = ((0.5, 11.5), (-3.0, 3.0))     # contains lambda = 2, 5, 8, 10 and 6
PS_FORMS = ["B(BDM)", "s(P1-divP1)"]
PS_WINDS = ["b0", "b7"]

mesh_ps = crisscross_mesh(PS_N)
ps_results = {}
for wind in PS_WINDS:
    beta = WINDS[wind](mesh_ps)
    for form in PS_FORMS:
        # nu = 0 here, so the Ritz values are lambda itself rather than lambda + nu;
        # the shift is spectrally inert (Section 6) and only clutters the axis.
        A, M, _, _ = FORMS[form]["build"](mesh_ps, PS_N, degree=1, beta=beta,
                                          eps=Constant(1.0), nu=0.0)
        ps_results[(form, wind)] = ps.pseudospectrum(
            A, M, PS_TAU, nev=PS_NEV, window=PS_WINDOW,
            form=form, wind=wind, label=WINDS[wind].tex)

fmt_table(ps.diagnostics_frame(ps_results), default="{:.3e}",
          rules=[("N", "{:.0f}"), ("m", "{:.0f}"), ("eigs in window", "{:.0f}"),
                 ("max |Im| in window", "{:.2e}"), ("max |Im| all ritz", "{:.2e}")],
          caption=r"Projection diagnostics for the pseudospectrum panels.")

with matplotlib.rc_context({"font.size": 9, "axes.titlesize": 9,
                            "axes.labelsize": 9, "font.family": "serif",
                            "mathtext.fontset": "cm"}):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ps.plot_pseudospectrum(ps_results[("B(BDM)", "b0")], ax=ax,
                           title="B(BDM),  " + WINDS["b0"].tex)
    fig.savefig(FIGDIR / "topform_pseudo_b0_B_BDM.pdf")
    plt.show()

with matplotlib.rc_context({"font.size": 9, "axes.titlesize": 9,
                            "axes.labelsize": 9, "font.family": "serif",
                            "mathtext.fontset": "cm"}):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ps.plot_pseudospectrum(ps_results[("s(P1-divP1)", "b0")], ax=ax,
                           title="s(P1-divP1),  " + WINDS["b0"].tex)
    # Mark where the shadow spectrum 3(m^2+n^2) falls.
    ax.axvline(6.0, color="w", lw=0.8, ls=":", alpha=0.8)
    ax.text(6.0, PS_WINDOW[1][1] * 0.82, r" spurious", color="w", fontsize=7,
            rotation=90, va="top")
    fig.savefig(FIGDIR / "topform_pseudo_b0_s_P1divP1.pdf")
    plt.show()

with matplotlib.rc_context({"font.size": 9, "axes.titlesize": 9,
                            "axes.labelsize": 9, "font.family": "serif",
                            "mathtext.fontset": "cm"}):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ps.plot_pseudospectrum(ps_results[("B(BDM)", "b7")], ax=ax,
                           title="B(BDM),  " + WINDS["b7"].tex)
    fig.savefig(FIGDIR / "topform_pseudo_b7_B_BDM.pdf")
    plt.show()

with matplotlib.rc_context({"font.size": 9, "axes.titlesize": 9,
                            "axes.labelsize": 9, "font.family": "serif",
                            "mathtext.fontset": "cm"}):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ps.plot_pseudospectrum(ps_results[("s(P1-divP1)", "b7")], ax=ax,
                           title="s(P1-divP1),  " + WINDS["b7"].tex)
    # Mark where the shadow spectrum 3(m^2+n^2) falls.
    ax.axvline(6.0, color="w", lw=0.8, ls=":", alpha=0.8)
    ax.text(6.0, PS_WINDOW[1][1] * 0.82, r" spurious", color="w", fontsize=7,
            rotation=90, va="top")
    fig.savefig(FIGDIR / "topform_pseudo_b7_s_P1divP1.pdf")
    plt.show()

# sigma_min at a genuine eigenvalue and at the spurious one, side by side.
rows = []
for form in PS_FORMS:
    r = ps_results[(form, "b0")]
    for name, target in [("genuine (near 2)", 2.0), ("genuine (near 8)", 8.0),
                         ("spurious (near 6)", 6.0)]:
        nearest = r["ritz"][int(np.argmin(np.abs(r["ritz"] - target)))]
        rows.append(dict(form=form, probe=name,
                         **{"nearest Ritz": nearest,
                            "distance to probe": abs(nearest - target),
                            r"$\sigma_{\min}$ at that Ritz value":
                                float(sla.svdvals(r["S"] - nearest * r["T"])[-1]),
                            r"$\sigma_{\min}$ one unit away":
                                float(sla.svdvals(r["S"] - (nearest + 1.0) * r["T"])[-1])}))
fmt_table(pd.DataFrame(rows).set_index(["form", "probe"]), default="{:.3e}",
          caption=r"$\sigma_{\min}$ evaluated at the Ritz value nearest each probe, "
                  r"and one unit away from it, for $\beta = 0$.")


# --------------------------------------------------------------------------
# 8. Summary
# --------------------------------------------------------------------------

# Counting pollution needs a tolerance, and the honest way to set one is to
# measure the discretisation error first.  Over the first 20 eigenvalues at
# N = 32 every FEEC row sits within ~1% of the exact spectrum, so 5% is a safe
# threshold: comfortably above the discretisation error, far below the ~20% gap
# that separates a spurious value from its nearest genuine neighbour.
TOL, NCOUNT = 0.05, 20
ex_count = exact_spectrum(400)

summary = []
for f in ALL_FORMS:
    q = runs0[f]
    v = q.real[:NCOUNT]
    rel = np.array([np.min(np.abs(ex_count - x)) / x for x in v])
    summary.append(dict(
        formulation=f, kind=FORMS[f]["kind"],
        shifted="yes" if FORMS[f]["shift"] else "no",
        **{"pencil": q.size,
           r"$\lambda_1$": q.real[0],
           r"rel err $\lambda_1$": abs(q.real[0] - 2.0) / 2.0,
           r"$\kappa_1(A-\sigma M)$": q.kappa1,
           "max rel dist (20)": rel.max(),
           "spurious (20)": int((rel > TOL).sum())}))
df_sum = pd.DataFrame(summary).set_index("formulation")
fmt_table(df_sum, default="{:.6f}",
          rules=[("rel err", "{:.2e}"), ("max rel dist", "{:.4f}"),
                 ("pencil", "{:.0f}"), ("spurious", "{:.0f}"), ("kappa", "{:.2e}"),
                 (r"$\kappa_1$", "{:.2e}")],
          caption=rf"$\beta=0$, $R_m=1$, criss-cross $N={N_BASE}$. A computed value counts "
                  rf"as spurious when it lies more than {TOL:.0%} from every exact "
                  r"eigenvalue $m^2+n^2$.")

# Every solve in this notebook logged its conditioning; this is the tally.
klog = pd.DataFrame(KAPPA_LOG)
worst = klog.loc[klog[r"Hager $\kappa_1$"].idxmax()]
print(f"{len(klog)} solves, each with its shifted matrix A - sigma*M estimated.\n")
print(f"  worst normwise kappa_1 : {worst[r'Hager $\kappa_1$']:.3e}"
      f"   ({worst['form']}, {worst['wind']}, N={worst['N']:.0f}, Rm={worst['Rm']:g},"
      f" sigma={worst['sigma']:.3f})")
print(f"  median normwise kappa_1: {klog[r'Hager $\kappa_1$'].median():.3e}")
print(f"  worst MUMPS COND1      : {klog['MUMPS COND1'].max():.3e}")
print(f"  MUMPS INFOG(1) != 0    : {int((klog['INFOG(1)'] != 0).sum())} of {len(klog)}"
      "   (0 means every factorisation succeeded)")
print(f"\n  digits lost to the factorisation, worst case: "
      f"{np.log10(worst[r'Hager $\kappa_1$']):.1f} of ~16")

print("figures written to", FIGDIR.resolve())
for p in sorted(FIGDIR.glob("topform_*.pdf")):
    print("  ", p.name)

