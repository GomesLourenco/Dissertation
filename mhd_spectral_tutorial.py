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
from typing import Callable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# --- Firedrake ------------------------------------------------------------
# `from firedrake import *` pulls in both the UFL symbols used to write weak
# forms (inner, grad, curl, dx, ...) and the Firedrake API (Mesh, FunctionSpace,
# assemble, DirichletBC, ...).  This star-import is the house style of the
# project's own demos and is what keeps variational forms readable.
from firedrake import *
from firedrake.function import PointEvaluator          # grid sampling of Functions
from firedrake.pyplot import triplot, tripcolor        # matplotlib bindings

# --- PETSc / SLEPc --------------------------------------------------------
# petsc4py gives us the assembled sparse matrices; slepc4py gives us the
# generalised eigensolver (EPS) and its spectral transformation (ST).
from petsc4py import PETSc
from slepc4py import SLEPc

# --- Netgen ---------------------------------------------------------------
# Firedrake can consume a Netgen/NGSolve mesh directly, which is by far the
# shortest route to a non-tensor-product domain such as the L-shape.
from netgen.occ import WorkPlane, OCCGeometry

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
})

# Note the alias: `from firedrake import *` exports a module of its own called
# `logging`, so the standard-library one has to be re-imported under a different
# name *after* the star-import.  This is the one real cost of the star-import
# style, and it is worth knowing about before it bites.
import logging as stdlogging

# `triplot` asks the mesh for the *interior* facets carrying each boundary
# marker; those sets are legitimately empty, and Firedrake warns about it every
# time.  Silence that one message so the notebook output stays readable.
stdlogging.getLogger("firedrake").addFilter(
    lambda record: "is empty. This is likely an error" not in record.getMessage())


# --- table rendering ------------------------------------------------------
# pandas' `.style` accessor needs jinja2, which a plain Firedrake environment
# does not ship.  This helper does the same job -- per-column number formats and
# a caption -- by formatting to strings, so the notebook has no extra
# dependencies.
def _cell(value, spec):
    if value is None:
        return "--"
    if isinstance(value, complex):
        return (f"{value.real:.5f}{value.imag:+.5f}i" if abs(value.imag) > 1e-12
                else spec.format(value.real))
    try:
        return "--" if not np.isfinite(value) else spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def fmt_table(df, caption=None, default="{:.6f}", rules=()):
    """Format a DataFrame for display.  `rules` is a sequence of
    (substring-of-column-name, format-spec) pairs, first match wins."""
    def spec_for(col):
        name = " ".join(map(str, col)) if isinstance(col, tuple) else str(col)
        return next((spec for key, spec in rules if key in name), default)

    out = pd.DataFrame({col: [_cell(v, spec_for(col)) for v in df[col]]
                        for col in df.columns}, index=df.index)
    if caption:
        display(Markdown(f"**{caption}**"))
    return out


print(f"PETSc scalar type : {PETSc.ScalarType.__name__}")
print(f"SLEPc EPS types   : krylovschur (default)")


# --------------------------------------------------------------------------
# 2. Meshes and visualisation
# --------------------------------------------------------------------------

L_DOMAIN = np.pi          # side of the bounding box for both domains


def square_mesh(N, L=L_DOMAIN):
    r"""Criss-cross ("Union Jack") triangulation of $(0,L)^2$ with $h = L/N$.

    `diagonal="crossed"` splits every grid square with *both* diagonals, adding a
    centre vertex: 4N^2 cells.  The crossed mesh is chosen deliberately -- it is
    symmetric under the reflections of the square, so it cannot bias one
    coordinate direction over the other when we introduce a shear wind.
    """
    return SquareMesh(N, N, L, quadrilateral=False, diagonal="crossed")


# The L-shape boundary, traversed counter-clockwise from the origin as fractions
# of L.  The re-entrant corner is the second vertex, (L/2, L/2).
_LSHAPE_PATH = ((0.5, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))


def lshape_mesh(N, L=L_DOMAIN):
    r"""Unstructured triangulation of the L-shaped domain, via Netgen/OCC.

    Firedrake's `Mesh` accepts a Netgen mesh object directly, so the whole domain
    is described by a polyline.  Netgen numbers the boundary segments 1..6 in the
    order they are traced, which is exactly the order of `_LSHAPE_PATH`:

        1: y = 0,    0 <= x <= L/2        4: x = L
        2: x = L/2   (re-entrant edge)    5: y = L
        3: y = L/2   (re-entrant edge)    6: x = 0

    The `triplot` legend below colours the segments in this numbering, which is
    the quickest way to confirm it.
    """
    wp = WorkPlane().MoveTo(0.0, 0.0)
    for px, py in _LSHAPE_PATH:
        wp.LineTo(L * px, L * py)
    ngmesh = OCCGeometry(wp.Face(), dim=2).GenerateMesh(maxh=L / N)
    return Mesh(ngmesh)


#: Registry consumed by the execution loop in Section 7.
DOMAINS = {"square": square_mesh, "lshape": lshape_mesh}

fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))

for row, (name, builder) in enumerate(DOMAINS.items()):
    for col, N in enumerate((4, 8, 16)):
        ax = axes[row, col]
        mesh = builder(N)
        # `interior_kw` is forwarded to a matplotlib PolyCollection (one polygon
        # per cell), so it wants `edgecolors`, not `color`: passing `color` sets
        # the *face* colour too and paints the domain solid.  `boundary_kw` goes
        # to a LineCollection, one per boundary marker.
        triplot(mesh, axes=ax,
                interior_kw={"linewidths": 0.35, "edgecolors": "0.55"},
                boundary_kw={"linewidths": 1.6})
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}, N = {N}  ({mesh.num_cells()} cells)", fontsize=10)

# One legend for the whole figure: it maps colours to boundary markers.
axes[0, 2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, title="markers", title_fontsize=8, frameon=False)
axes[1, 2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, title="markers", title_fontsize=8, frameon=False)
fig.suptitle("Refinement sequences: convex square (top), re-entrant L-shape (bottom)")
fig.savefig(FIGDIR / "meshes.pdf")
plt.show()


# --------------------------------------------------------------------------
# 3. Velocity fields (winds)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Wind:
    """A named background velocity field, built lazily on a given mesh."""
    key: str
    tex: str                  # LaTeX label, for figure titles
    note: str                 # one-line description
    _build: Callable          # (mesh, L) -> UFL vector expression or None

    def __call__(self, mesh, L=L_DOMAIN):
        return self._build(mesh, L)


def _zero(mesh, L):
    # `None` is the sentinel for "no advection term at all".  Returning a zero
    # vector would work too, but assembling nothing is both faster and keeps the
    # resulting matrix exactly symmetric.
    return None


def _uniform(mesh, L):
    # Constant() wraps a number as a UFL constant so it can be changed without
    # recompiling the form.
    return as_vector([Constant(1.0), Constant(0.0)])


def _cellular(mesh, L):
    # SpatialCoordinate(mesh) is the UFL symbol for x; `sin` here is UFL's sin,
    # not numpy's, so the expression stays symbolic and differentiable.
    x, y = SpatialCoordinate(mesh)
    return as_vector([sin(y), sin(x)])


def _shear(mesh, L):
    # conditional(gt(a, b), p, q) is UFL's ternary operator; it is evaluated at
    # each quadrature point, so the jump is captured exactly at the element level
    # rather than being interpolated into a finite element space.
    x, y = SpatialCoordinate(mesh)
    return as_vector([conditional(gt(y, Constant(L / 2)), 1.0, -1.0), Constant(0.0)])


WINDS = {
    "W0": Wind("W0", r"$u=(0,0)$", "self-adjoint baseline", _zero),
    "W1": Wind("W1", r"$u=(1,0)$", "uniform translation", _uniform),
    "W2": Wind("W2", r"$u=(\sin y,\,\sin x)$", "smooth cellular", _cellular),
    "W3": Wind("W3", r"$u=(\mathrm{sign}(y-L/2),\,0)$", "Heaviside shear", _shear),
}

def sample_grid(f, mesh, n=18, pad=0.06):
    """Evaluate a vector `Function` on a regular n x n grid over the mesh bbox.

    `PointEvaluator` is the supported replacement for the deprecated
    `Function.at`.  With `missing_points_behaviour="ignore"` the points that fall
    outside the domain -- the whole notch of the L-shape -- come back as NaN,
    which matplotlib simply does not draw.
    """
    coords = mesh.coordinates.dat.data_ro
    (x0, y0), (x1, y1) = coords.min(axis=0), coords.max(axis=0)
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    X, Y = np.meshgrid(np.linspace(x0 + dx, x1 - dx, n),
                       np.linspace(y0 + dy, y1 - dy, n))
    pts = np.column_stack([X.ravel(), Y.ravel()])
    # `tolerance` is measured in *reference cell* coordinates and defaults to
    # the mesh's own value (0.5 here), which is loose enough to snap points that
    # sit well outside the domain into a nearby boundary cell -- arrows would
    # then leak across the L-shape's notch.  A tight tolerance fixes it.
    vals = np.asarray(
        PointEvaluator(mesh, pts, missing_points_behaviour="ignore",
                       tolerance=1e-6).evaluate(f))
    return X, Y, vals[:, 0].reshape(X.shape), vals[:, 1].reshape(X.shape)


def as_cg_vector(expr, mesh, degree=1):
    """Project a UFL vector expression into [CG_degree]^2, for plotting only."""
    return Function(VectorFunctionSpace(mesh, "CG", degree)).project(expr)


def plot_wind(ax, u, mesh, n=16, cmap="RdBu_r"):
    """Colour = horizontal component; arrows = direction and relative magnitude."""
    if u is None:
        # `boundary_kw["colors"]` is zipped against the list of markers, so a
        # single colour string would draw only the first marker and silently
        # drop the rest.  One entry per marker.
        markers = mesh.exterior_facets.unique_markers
        triplot(mesh, axes=ax,
                interior_kw={"linewidths": 0.25, "edgecolors": "0.8"},
                boundary_kw={"linewidths": 1.2, "colors": ["0.3"] * len(markers)})
        # (0.5, 0.5) in axes coordinates is the re-entrant corner of the
        # L-shape, so the label goes into the upper-left arm instead.
        ax.text(0.28, 0.74, r"$u\equiv 0$", ha="center", va="center",
                transform=ax.transAxes, fontsize=15)
        handle = None
    else:
        # DG0 interpolation: one constant per cell, so a jump stays a jump.
        ux = Function(FunctionSpace(mesh, "DG", 0)).interpolate(u[0])
        lim = float(np.abs(ux.dat.data_ro).max()) or 1.0
        handle = tripcolor(ux, axes=ax, cmap=cmap, vmin=-lim, vmax=lim)
        X, Y, U, V = sample_grid(as_cg_vector(u, mesh), mesh, n=n)
        # `angles="xy", scale_units="xy"` measures arrow length in *data* units,
        # so a vector of maximal magnitude spans 0.8 of the sampling spacing and
        # no arrow can spill across the L-shape's notch.
        speed = np.nanmax(np.hypot(U, V)) or 1.0
        step = float(X[0, 1] - X[0, 0])
        ax.quiver(X, Y, U, V, color="k", angles="xy", scale_units="xy",
                  scale=speed / (0.8 * step), width=0.005, alpha=0.85)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return handle


fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.2))
for col, key in enumerate(("W0", "W1", "W2", "W3")):
    for row, builder in enumerate((square_mesh, lshape_mesh)):
        mesh = builder(14)
        h = plot_wind(axes[row, col], WINDS[key](mesh), mesh)
        if row == 0:
            axes[row, col].set_title(f"{key}: {WINDS[key].tex}", fontsize=11)
fig.colorbar(h, ax=axes, fraction=0.02, pad=0.01, label=r"$u_x$")
fig.suptitle("Background winds on both domains (colour: $u_x$, arrows: direction)", y=0.95)
fig.savefig(FIGDIR / "winds.pdf")
plt.show()


# --------------------------------------------------------------------------
# 4. Function spaces and weak formulations
# --------------------------------------------------------------------------

def space_A(mesh, degree=1):
    r"""Formulation A: the potential $A$ is a discrete **0-form**, $A_h \in \mathrm{CG}_r$.

    No multiplier space appears here.  The gauge constraint $\int_\Omega A = 0$ lives
    in $\mathbb{R}$, and is attached to the assembled matrix in Section 5.3 -- see
    the note there on why an "R" block cannot be carried in the mixed space itself.
    """
    return FunctionSpace(mesh, "CG", degree)


def space_BN1(mesh, degree=1):
    r"""Formulation B, trimmed: $\mathrm{N1curl}_r \times \mathrm{CG}_r$.

    In Firedrake `V * Q` builds the MixedFunctionSpace; `TrialFunctions(W)` then
    splits a trial function into its components in one call.
    """
    return FunctionSpace(mesh, "N1curl", degree) * FunctionSpace(mesh, "CG", degree)


def space_BN2(mesh, degree=1):
    r"""Formulation B, full: $\mathrm{N2curl}_r \times \mathrm{CG}_{r+1}$.

    The multiplier degree is bumped by one so that
    $\nabla(\mathrm{CG}_{r+1}) \subseteq \mathrm{N2curl}_r$, i.e. so that the pair
    is still a de Rham subcomplex.  Using CG_r here instead would silently break
    exactness and is a classic source of spurious modes.
    """
    return FunctionSpace(mesh, "N2curl", degree) * FunctionSpace(mesh, "CG", degree + 1)


SPACES = {"A(CG)": space_A, "B(N1)": space_BN1, "B(N2)": space_BN2}


# --------------------------------------------------------------------------
# 4.2 The B formulations
# --------------------------------------------------------------------------

def cross2d(a, b):
    r"""Scalar 2D cross product $(a\times b)_z = a_0 b_1 - a_1 b_0$.

    Antisymmetric, so the argument order is load-bearing: the induction term needs
    cross2d(v, u), i.e. $v\times u$.
    """
    return a[0] * b[1] - a[1] * b[0]


def forms_B(W, u=None, eps=Constant(1.0)):
    r"""Bilinear forms of the mixed B-formulation on the mixed space `W`.

    Returns (a, m) with $a$ the operator form and $m$ the mass form.  Note that
    $m$ involves **only** the field: the multiplier block of the mass matrix is
    identically zero.  That is the structural reason the pencil is singular.
    """
    v, p = TrialFunctions(W)      # unpack the mixed trial function
    w, q = TestFunctions(W)       # ... and the mixed test function

    # 1. Diffusion.  In 2D curl(v) is scalar, so this is a scalar L2 product.
    a = eps * inner(curl(v), curl(w)) * dx

    # 2. Induction.  Skipped entirely when u is None, which keeps the W0 pencil
    #    exactly symmetric down to the last bit.
    if u is not None:
        a += inner(cross2d(v, u), curl(w)) * dx

    # 3. The saddle-point coupling.  The two off-diagonal blocks carry opposite
    #    signs so that the whole operator stays symmetric when u is None.
    a += (-inner(grad(p), w) + inner(v, grad(q))) * dx

    m = inner(v, w) * dx
    return a, m


def bcs_B(W):
    r"""Essential boundary conditions for the B-formulations.

    * `DirichletBC(W.sub(0), (0,0), "on_boundary")` on an H(curl) space constrains
      the **tangential trace** $v\times n$, not the full vector: the edge degrees
      of freedom of N1curl/N2curl *are* tangential moments, so setting them to
      zero imposes exactly $B\times n = 0$ and nothing more.  This is the discrete
      statement of the wall condition from Section 1.2.

    * `DirichletBC(W.sub(1), 0, "on_boundary")` puts the multiplier in
      $\mathrm{CG}\cap H^1_0$.  This is not a physical condition on $p$; it is what
      makes the discrete gradients $\nabla Q_h$ a subspace of $H_0(\mathrm{curl})$,
      so that the constraint $(v,\nabla q)=0$ tests against the right space.  Drop
      it and the constraint over-determines $v$ near the boundary.
    """
    return [DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary"),
            DirichletBC(W.sub(1), Constant(0.0), "on_boundary")]


# --------------------------------------------------------------------------
# 4.3 The A formulation
# --------------------------------------------------------------------------

def forms_A(V, u=None, eps=Constant(1.0)):
    r"""Bilinear forms of the potential formulation on the scalar space `V`.

    `TrialFunction` / `TestFunction` (singular) because `V` is not mixed.  The
    advection term uses `dot(u, grad(A))`, which is the convective derivative
    $u\cdot\nabla A$; it is *not* integrated by parts, so no boundary term is
    generated and the discontinuity of `u` is only ever seen by the quadrature.
    """
    A, z = TrialFunction(V), TestFunction(V)

    a = eps * inner(grad(A), grad(z)) * dx           # diffusion (symmetric)
    if u is not None:
        a += inner(dot(u, grad(A)), z) * dx          # advection (non-symmetric)

    m = inner(A, z) * dx
    return a, m


def bcs_A(V):
    """No essential conditions: see the discussion above -- the wall condition on
    this formulation is natural, and the gauge is applied to the matrix."""
    return []


FORMS = {"A(CG)": forms_A, "B(N1)": forms_B, "B(N2)": forms_B}


# --------------------------------------------------------------------------
# 5. Assembling the matrix pencil
# --------------------------------------------------------------------------

def to_scipy(mat):
    """PETSc AIJ -> scipy CSR (serial).  `getValuesCSR` hands back the raw arrays."""
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())


def to_petsc(S):
    """scipy CSR -> PETSc AIJ.  PETSc requires sorted column indices per row."""
    S = S.tocsr(); S.sort_indices()
    return PETSc.Mat().createAIJWithArrays(
        S.shape, (S.indptr.astype(PETSc.IntType),
                  S.indices.astype(PETSc.IntType), S.data))


def border_with_gauge(K, M, g):
    r"""Return the bordered pencil enforcing $g^{\mathsf T}a = 0$.

    `sp.bmat` with `None` blocks builds the zero blocks for us, so the whole
    construction is one call per matrix.
    """
    col = sp.csr_matrix(-np.asarray(g).reshape(-1, 1))          # the -g column
    A_b = sp.bmat([[to_scipy(K), col], [col.T, None]], format="csr")
    M_b = sp.bmat([[to_scipy(M), None], [None, sp.csr_matrix((1, 1))]], format="csr")
    return to_petsc(A_b), to_petsc(M_b)


@dataclass
class Pencil:
    """An assembled generalised eigenproblem $Ax = \\lambda Mx$ plus its provenance."""
    A: PETSc.Mat
    M: PETSc.Mat
    V: object            # the FunctionSpace the eigenvectors live in
    form: str
    ndof: int            # dofs of V, i.e. excluding any border row

    @property
    def size(self):
        return self.A.getSize()[0]

    def to_function(self, vec):
        """Wrap a PETSc Vec as a Firedrake Function on `V`.

        `dat.vec_wo` exposes the Function's own Vec in write-only mode; taking the
        leading `getLocalSize()` entries silently discards the gauge multiplier of
        the bordered A-pencil, which is exactly what we want to plot.
        """
        f = Function(self.V)
        with f.dat.vec_wo as fv:
            fv.setArray(vec.getArray(readonly=True)[:fv.getLocalSize()])
        return f

    def transposed(self):
        """The pencil $(A^{\\mathsf T}, M^{\\mathsf T})$ -- see Section 6.3.

        Careful: `PETSc.Mat.transpose()` called with no output argument
        transposes **in place** and hands back the same object, which would
        quietly corrupt the pencil we were given.  Copy first.
        """
        At, Mt = self.A.copy(), self.M.copy()
        At.transpose(); Mt.transpose()
        return Pencil(At, Mt, self.V, self.form + "^T", self.ndof)


def build_pencil(mesh, form="B(N1)", u=None, Rm=1.0, degree=1):
    """Assemble (A, M) for any of the three formulations."""
    eps = Constant(1.0 / Rm)

    if form == "A(CG)":
        V = space_A(mesh, degree)
        a, m = forms_A(V, u=u, eps=eps)
        # bcs_A(V) is empty: the wall condition here is natural (Section 4.3).
        K, M = assemble(a).petscmat, assemble(m).petscmat
        # g_i = \int phi_i : assembling the linear form `z*dx` gives exactly that.
        g = np.asarray(assemble(TestFunction(V) * dx).dat.data_ro).copy()
        A_b, M_b = border_with_gauge(K, M, g)
        return Pencil(A_b, M_b, V, form, V.dim())

    W = SPACES[form](mesh, degree)
    a, m = forms_B(W, u=u, eps=eps)
    bcs = bcs_B(W)
    A = assemble(a, bcs=bcs).petscmat
    M = assemble(m, bcs=bcs, weight=0.0).petscmat     # boundary modes -> infinity
    return Pencil(A, M, W, form, W.dim())


# --------------------------------------------------------------------------
# 6. The solver: shift-and-invert with SLEPc
# --------------------------------------------------------------------------

#: PETSc/SLEPc options, in the same dictionary form used by Firedrake solvers.
EPS_OPTS = {
    # --- the eigenproblem ---------------------------------------------------
    "eps_gen_non_hermitian": None,   # generalised, non-Hermitian pencil (GNHEP)
    "eps_type": "krylovschur",       # thick-restart Krylov-Schur
    "eps_target_magnitude": None,    # order by |lambda - tau|
    "eps_tol": 1e-11,
    "eps_max_it": 2000,
    # --- the spectral transformation ---------------------------------------
    "st_type": "sinvert",            # (A - tau M)^{-1} M
    # --- the inner linear solve --------------------------------------------
    "st_ksp_type": "preonly",        # no Krylov wrapper: the LU *is* the solve
    "st_pc_type": "lu",
    "st_pc_factor_mat_solver_type": "mumps",
    "st_mat_mumps_icntl_14": 500,    # working-space headroom for pivoting
}

_prefix_counter = itertools.count()


def solve_pencil(pencil, tau=0.95, nev=24, **overrides):
    """Run one SLEPc EPS solve on an assembled pencil and return the solver.

    The options above are pushed into the global PETSc options database under a
    unique prefix, which is how PETSc-style configuration reaches an object that
    Firedrake did not create for us.  The keys are removed afterwards so that
    repeated calls in a notebook never leak options into one another.
    """
    opts = dict(EPS_OPTS, eps_target=tau, **overrides)
    prefix = f"mhd{next(_prefix_counter)}_"
    db = PETSc.Options()
    for key, value in opts.items():
        db[prefix + key] = value

    eps = SLEPc.EPS().create(comm=pencil.A.getComm())
    eps.setOptionsPrefix(prefix)
    eps.setOperators(pencil.A, pencil.M)     # the generalised pair (A, M)
    eps.setDimensions(nev=nev)               # how many eigenpairs to request
    eps.setFromOptions()                     # reads everything set above
    eps.solve()

    for key in opts:
        db.delValue(prefix + key)
    return eps


def harvest(eps, inf_tol=1e8):
    """Converged **finite** eigenvalues as (slepc_index, value), sorted by Re.

    The infinite eigenvalues of Section 5.2 come back either as `inf`/`nan` or as
    very large finite numbers, depending on how the Krylov-Schur restart happened
    to hit them; one threshold catches both.
    """
    out = [(i, complex(eps.getEigenvalue(i))) for i in range(eps.getConverged())]
    out = [(i, z) for i, z in out if np.isfinite(z) and abs(z) < inf_tol]
    return sorted(out, key=lambda pair: (pair[1].real, abs(pair[1].imag)))


# --------------------------------------------------------------------------
# 6.3 Left eigenvectors and non-normality
# --------------------------------------------------------------------------

def complex_vector(eps, i, pencil):
    """The i-th eigenvector as (complex coefficient array, Re Function, Im Function).

    PETSc here is built for *real* scalars, so SLEPc returns a complex eigenvector
    as a pair of real Vecs.  We keep both views: the numpy array for algebra, the
    Firedrake Functions for plotting.
    """
    vr, vi = pencil.A.createVecRight(), pencil.A.createVecRight()
    eps.getEigenvector(i, vr, vi)
    z = np.asarray(vr.getArray()).copy() + 1j * np.asarray(vi.getArray()).copy()
    return z, pencil.to_function(vr), pencil.to_function(vi)


@dataclass
class Mode:
    """One eigenpair: value, right eigenfunction, and (if computed) its adjoint."""
    value: complex
    right: tuple                       # (Re, Im) Firedrake Functions
    left: Optional[tuple] = None
    condition: Optional[float] = None  # kappa(lambda); None when not computed

    @property
    def is_complex(self):
        return abs(self.value.imag) > 1e-10 * max(1.0, abs(self.value.real))


# --------------------------------------------------------------------------
# 7. The execution loop
# --------------------------------------------------------------------------

@dataclass
class Spectrum:
    """The result of one experiment: modes, the pencil they came from, and metadata."""
    modes: list
    pencil: Pencil
    mesh: object
    wind: Optional[object]
    meta: dict

    @property
    def values(self):
        return np.array([m.value for m in self.modes])

    def frame(self):
        """A tidy pandas view of the spectrum."""
        return pd.DataFrame({
            "n": np.arange(len(self.modes)),
            "Re lambda": [m.value.real for m in self.modes],
            "Im lambda": [m.value.imag for m in self.modes],
            "kappa": [m.condition for m in self.modes],
        })


def solve_induction(domain="square", form="B(N1)", wind="W0", *, N=16, degree=1,
                    Rm=1.0, tau=None, n_eigs=6, nev=24, L=L_DOMAIN,
                    left=None, mesh=None):
    r"""Solve the induction eigenproblem for one (domain, formulation, wind).

    Parameters
    ----------
    domain : {"square", "lshape"}
    form   : {"A(CG)", "B(N1)", "B(N2)"}
    wind   : {"W0", "W1", "W2", "W3"}
    N      : mesh level -- cells per unit length (square) or 1/maxh (L-shape)
    Rm     : magnetic Reynolds number; the diffusivity is eps = 1/Rm
    tau    : shift for the spectral transformation.  Defaults to 0.95/Rm, which
             tracks the leading decay rate as Rm varies.
    left   : compute left eigenvectors and condition numbers.  Defaults to True
             whenever the wind is non-zero, i.e. whenever they can differ from
             the right ones.
    """
    tau = 0.95 / Rm if tau is None else tau
    mesh = DOMAINS[domain](N, L) if mesh is None else mesh
    u = WINDS[wind](mesh, L)

    pencil = build_pencil(mesh, form=form, u=u, Rm=Rm, degree=degree)
    eps = solve_pencil(pencil, tau=tau, nev=nev)
    found = harvest(eps)[:n_eigs]

    want_left = (u is not None) if left is None else left
    if want_left:
        adjoint = pencil.transposed()
        adj_eps = solve_pencil(adjoint, tau=tau, nev=nev)
        adj_found = harvest(adj_eps)
        M_sp = to_scipy(pencil.M)

    modes = []
    for index, lam in found:
        x, xr, xi = complex_vector(eps, index, pencil)
        mode = Mode(lam, (xr, xi))
        if want_left and adj_found:
            # pair by eigenvalue: the adjoint run sees conj(lambda)
            j = min(adj_found, key=lambda pair: abs(pair[1] - np.conj(lam)))[0]
            y, yr, yi = complex_vector(adj_eps, j, adjoint)
            # kappa in the M-norm: mesh-independent, and exactly 1 whenever the
            # eigenvalue is simple and the operator normal.  M is only positive
            # *semi*-definite -- the multiplier block carries no mass -- so these
            # are the L2 norms of the field part alone, which is what we want.
            overlap = abs(np.vdot(y, M_sp @ x))
            x_M = np.sqrt(abs(np.vdot(x, M_sp @ x)))
            y_M = np.sqrt(abs(np.vdot(y, M_sp @ y)))
            mode.left = (yr, yi)
            mode.condition = x_M * y_M / overlap if overlap > 0 else np.inf
        modes.append(mode)

    return Spectrum(modes, pencil, mesh, u,
                    dict(domain=domain, form=form, wind=wind, N=N, degree=degree,
                         Rm=Rm, tau=tau, L=L, nconv=eps.getConverged()))

# A first run, to see the machinery turn over.
demo = solve_induction(domain="square", form="B(N2)", wind="W3", N=16, Rm=5.0)
print(f"{demo.meta['form']} on the {demo.meta['domain']}, wind {demo.meta['wind']}, "
      f"Rm = {demo.meta['Rm']}, tau = {demo.meta['tau']:.4f}")
print(f"pencil size {demo.pencil.size} ({demo.pencil.ndof} dofs), "
      f"{demo.meta['nconv']} eigenpairs converged")
demo.frame().round(6)


# --------------------------------------------------------------------------
# 8. Results
# --------------------------------------------------------------------------

def exact_square(n_wanted, Rm=1.0, kmax=12):
    """The first `n_wanted` eigenvalues (m^2+n^2)/Rm, with multiplicity."""
    vals = sorted(m * m + n * n
                  for m in range(kmax) for n in range(kmax) if (m, n) != (0, 0))
    return np.array(vals[:n_wanted], dtype=float) / Rm


#: Maxwell eigenvalues of the unit-scale L-shape (Dauge benchmark), rescaled to
#: side pi by (2/pi)^2.  lambda_1 and lambda_2 are corner-singular; lambda_3 and
#: lambda_4 equal pi^2 on the reference domain and so are exactly 4 on ours.
_DAUGE_LSHAPE = np.array([1.47562182408, 3.53403136678,
                          9.86960440109, 9.86960440109, 11.3894793979])


def exact_lshape(n_wanted, Rm=1.0):
    vals = _DAUGE_LSHAPE * (2.0 / np.pi) ** 2 / Rm
    if n_wanted > vals.size:            # only five benchmark values are tabulated
        vals = np.pad(vals, (0, n_wanted - vals.size), constant_values=np.nan)
    return vals[:n_wanted]


EXACT = {"square": exact_square, "lshape": exact_lshape}


# --------------------------------------------------------------------------
# 8.2 Baseline validation: the self-adjoint case $u \equiv 0$
# --------------------------------------------------------------------------

def compare_forms(domain, wind, Rm=1.0, N=16, n_eigs=6, degree=1, forms=None, **kw):
    """One table: computed eigenvalues per formulation, next to the reference."""
    forms = forms or ["A(CG)", "B(N1)", "B(N2)"]
    ref = EXACT[domain](n_eigs, Rm)
    table = {"exact" if wind == "W0" else "reference": ref}
    sizes = {}
    for form in forms:
        s = solve_induction(domain=domain, form=form, wind=wind, N=N, Rm=Rm,
                            degree=degree, n_eigs=n_eigs, left=False, **kw)
        vals = s.values.real
        table[form] = np.pad(vals, (0, n_eigs - len(vals)), constant_values=np.nan)
        sizes[form] = s.pencil.size
    df = pd.DataFrame(table, index=[f"$\\lambda_{i+1}$" for i in range(n_eigs)])
    for form in forms:
        df[f"err {form}"] = np.abs(df[form] - ref) / np.abs(ref)
    return df, sizes


df_sq, sizes_sq = compare_forms("square", "W0", Rm=1.0, N=16)
print("pencil sizes:", sizes_sq)
fmt_table(df_sq, rules=[("err", "{:.2e}")],
          caption="Square, $u \\equiv 0$, $R_m = 1$, $N = 16$, degree 1")

# Only five reference values are tabulated for the L-shape, so ask for five.
df_L, sizes_L = compare_forms("lshape", "W0", Rm=1.0, N=16, n_eigs=5, tau=0.5)
print("pencil sizes:", sizes_L)
fmt_table(df_L, rules=[("err", "{:.2e}")],
          caption="L-shape, $u \\equiv 0$, $R_m = 1$, $N = 16$, degree 1")


# --------------------------------------------------------------------------
# 8.3 An exact eigenpair for the shear wind
# --------------------------------------------------------------------------

def exact_shear_family(n_wanted, Rm=1.0):
    """The exact subfamily lambda = n^2/Rm, A = cos(n y), valid for any u=(f(y),0)."""
    return np.arange(1, n_wanted + 1) ** 2 / Rm


def match_family(form, Rm, N=32, n_family=3, n_eigs=16, nev=44):
    """Locate each exact family value inside the computed spectrum.

    The family members are *not* the first few eigenvalues: the wind moves the
    x-dependent modes around them, so they interleave.  Matching by proximity is
    the meaningful comparison; matching by position is not.
    """
    spectrum = solve_induction(domain="square", form=form, wind="W3", N=N,
                               Rm=Rm, n_eigs=n_eigs, nev=nev, left=False)
    computed = spectrum.values
    row = {}
    for n, exact in enumerate(exact_shear_family(n_family, Rm), start=1):
        nearest = computed[int(np.argmin(np.abs(computed - exact)))]
        row[f"$n={n}$"] = nearest.real
        row[f"err $n={n}$"] = abs(nearest.real - exact) / exact
    return row


rows = [dict(Rm=Rm, form=form, **match_family(form, Rm))
        for Rm in (1.0, 5.0, 20.0)
        for form in ("A(CG)", "B(N1)", "B(N2)")]

df_shear = pd.DataFrame(rows).set_index(["Rm", "form"])
fmt_table(df_shear, rules=[("err", "{:.2e}")],
          caption="Square, Heaviside shear W3, $N=32$: the exact family "
                  "$\\lambda_n = n^2/R_m$ located inside the computed spectrum")


# --------------------------------------------------------------------------
# 8.4 Convergence under refinement
# --------------------------------------------------------------------------

def convergence(domain, wind, Rm=1.0, k=1, levels=(8, 12, 16, 24, 32, 44),
                degree=1, **kw):
    """Relative error in the k-th eigenvalue versus mesh size, per formulation.

    `k` is 1-based.  On the square with a shear the reference is the exact family
    value nearest the computed eigenvalue; elsewhere it is the k-th reference
    eigenvalue of the domain.
    """
    target = EXACT[domain](k, Rm)[k - 1]
    out = {}
    for form in ("A(CG)", "B(N1)", "B(N2)"):
        hs, errs = [], []
        for N in levels:
            s = solve_induction(domain=domain, form=form, wind=wind, N=N, Rm=Rm,
                                degree=degree, n_eigs=k, left=False, **kw)
            hs.append(np.pi / N)
            errs.append(abs(s.values[k - 1].real - target) / abs(target))
        out[form] = (np.array(hs), np.array(errs))
    return out, target


STYLE = {"A(CG)": ("C2", "^"), "B(N1)": ("C0", "o"), "B(N2)": ("C1", "s")}

#: (domain, wind, Rm, eigenvalue index, extra solver kwargs, panel caption)
PANELS = [
    ("square", "W0", 1.0, 1, {}, "convex, no wind"),
    ("square", "W3", 5.0, 1, {}, "convex, Heaviside shear"),
    ("lshape", "W0", 1.0, 1, dict(tau=0.5), r"re-entrant, $\lambda_1$ singular"),
    ("lshape", "W0", 1.0, 3, dict(tau=0.5), r"re-entrant, $\lambda_3$ regular"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.1), sharey=True)
for ax, (domain, wind, Rm, k, kw, caption) in zip(axes, PANELS):
    curves, target = convergence(domain, wind, Rm=Rm, k=k, **kw)
    for form, (hs, errs) in curves.items():
        colour, marker = STYLE[form]
        ax.loglog(hs, np.maximum(errs, 1e-16), marker=marker, color=colour,
                  ms=4.5, lw=1.2, label=form)
    hs = curves["B(N1)"][0]
    ax.loglog(hs, 0.3 * (hs / hs[0]) ** 2, "k--", lw=0.8, label=r"$O(h^2)$")
    ax.loglog(hs, 0.3 * (hs / hs[0]) ** (4 / 3), "k:", lw=0.9, label=r"$O(h^{4/3})$")
    ax.set_xlabel(r"$h$")
    ax.set_title(f"{caption}\n$R_m={Rm:g}$,  "
                 f"$\\lambda_{{{k}}} = {target:.6f}$", fontsize=9.5)
axes[0].set_ylabel(r"relative error in $\lambda_k$")
axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("Convergence under uniform refinement", y=1.04)
fig.savefig(FIGDIR / "convergence.pdf")
plt.show()


# --------------------------------------------------------------------------
# 8.5 Eigenmodes
# --------------------------------------------------------------------------

def normalise(f):
    """Scale a Function to unit maximum modulus.

    For a scalar the sign of the largest entry is fixed too, so that repeated
    solves of the same mode always come out the same way up; for a vector only
    the magnitude is normalised, the overall sign of an eigenvector being
    arbitrary in any case.
    """
    data = f.dat.data_ro
    mag = np.abs(data) if data.ndim == 1 else np.linalg.norm(data, axis=1)
    k = int(np.argmax(mag))
    scale = data[k] if data.ndim == 1 else mag[k]
    if abs(scale) > 0:
        f.dat.data[:] = f.dat.data_ro / scale
    return f


def plot_mode(ax, mode, form, mesh, part="right", n_stream=20, cmap="RdBu_r"):
    """Draw one eigenmode.  Complex modes are shown through their real part."""
    fr, fi = (mode.right if part == "right" else mode.left)
    f = fr if not mode.is_complex else fr        # real part of the eigenfunction

    if form == "A(CG)":
        A = normalise(Function(f.function_space()).assign(f))
        # A 98th-percentile colour limit rather than the maximum: adjoint modes
        # of an inflow/outflow problem carry a thin boundary layer whose peak
        # would otherwise wash out the whole interior.
        lim = float(np.percentile(np.abs(A.dat.data_ro), 98)) or 1.0
        handle = tripcolor(A, axes=ax, cmap=cmap, vmin=-lim, vmax=lim)
        # Field lines = level sets of A.
        B = as_cg_vector(as_vector([A.dx(1), -A.dx(0)]), mesh)
    else:
        v = f.subfunctions[0]                    # the H(curl) component
        B = normalise(as_cg_vector(v, mesh))
        mag = Function(FunctionSpace(mesh, "CG", 1)).project(sqrt(dot(B, B)))
        handle = tripcolor(mag, axes=ax, cmap="viridis")

    # NaNs from `sample_grid` mark points outside the domain; matplotlib's
    # streamplot stops integrating there, which is exactly the behaviour we want
    # around the L-shape's notch.
    X, Y, U, V = sample_grid(B, mesh, n=n_stream)
    ax.streamplot(X, Y, U, V, color="k", density=0.75, linewidth=0.55,
                  arrowsize=0.55)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return handle


fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
for row, domain in enumerate(("square", "lshape")):
    kw = dict(tau=0.5) if domain == "lshape" else {}
    s = solve_induction(domain=domain, form="B(N2)", wind="W0", N=20, n_eigs=4, **kw)
    for col in range(4):
        plot_mode(axes[row, col], s.modes[col], "B(N2)", s.mesh)
        axes[row, col].set_title(rf"{domain}: $\lambda_{col+1} = "
                                 rf"{s.values[col].real:.4f}$", fontsize=10)
fig.suptitle(r"B(N2) eigenmodes, $u\equiv 0$, $R_m=1$ "
             r"(colour $|B|$, lines: magnetic field lines)", y=0.97)
fig.savefig(FIGDIR / "modes_W0.pdf")
plt.show()


# --------------------------------------------------------------------------
# 8.6 Non-normality: right versus left eigenvectors
# --------------------------------------------------------------------------

# Check 1: for a self-adjoint problem kappa must be exactly 1 -- but only where
# the eigenvalue is simple.  On the square, lambda = 1, 4, 5, ... are all double,
# and there the two solves may return different bases of the eigenspace, so the
# overlap is arbitrary.  lambda_3 = 2 is simple and is the meaningful test.
check = {form: solve_induction(domain="square", form=form, wind="W0", N=16,
                               Rm=1.0, n_eigs=4, left=True)
         for form in ("A(CG)", "B(N1)", "B(N2)")}
print("kappa for u = 0 (lambda = 1, 1, 2, 4; only lambda_3 = 2 is simple)")
for form, spectrum in check.items():
    print(f"  {form:6s}", np.round([m.condition for m in spectrum.modes], 6))

# Check 2: in the M-norm kappa is mesh-independent.  The adjoint boundary layer
# has width O(1/Rm) = 0.05 here, so A(CG) only settles once h resolves it.
print("\nkappa(lambda_1) under refinement, W3, Rm = 20")
for N in (16, 24, 32, 48, 64):
    row = {form: solve_induction(domain="square", form=form, wind="W3", N=N,
                                 Rm=20.0, n_eigs=1).modes[0].condition
           for form in ("A(CG)", "B(N2)")}
    print(f"  N = {N:3d}  h = {np.pi/N:.4f}   "
          + "   ".join(f"{f}: {k:7.3f}" for f, k in row.items()))

fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
runs = [("A(CG)", 20.0), ("B(N2)", 20.0)]

for row, (form, Rm) in enumerate(runs):
    s = solve_induction(domain="square", form=form, wind="W3", N=32, Rm=Rm, n_eigs=2)
    for k in range(2):
        for j, side in enumerate(("right", "left")):
            ax = axes[row, 2 * k + j]
            plot_mode(ax, s.modes[k], form, s.mesh, part=side)
            ax.set_title(rf"{form}  $\lambda_{k+1}={s.values[k].real:.4f}$"
                         rf"  ({side}), $\kappa={s.modes[k].condition:.1f}$",
                         fontsize=9)
fig.suptitle(r"Right vs left eigenvectors under the Heaviside shear, $R_m = 20$", y=0.97)
fig.savefig(FIGDIR / "left_right.pdf")
plt.show()

# N = 32 keeps the cell Peclet number below 1 for every Rm in this sweep
# (see Section 9); Rm = 20 is the largest value this mesh can carry honestly.
rows = []
for Rm in (1.0, 2.0, 5.0, 10.0, 20.0):
    for form in ("A(CG)", "B(N1)", "B(N2)"):
        s = solve_induction(domain="square", form=form, wind="W3", N=32,
                            Rm=Rm, n_eigs=5)
        rows.append(dict(Rm=Rm, form=form,
                         **{rf"$\kappa_{k+1}$": s.modes[k].condition
                            for k in range(5)}))

df_kappa = pd.DataFrame(rows).set_index(["Rm", "form"])
fmt_table(df_kappa, default="{:.4g}",
          caption=r"Condition numbers $\kappa(\lambda_k)$ of the first five modes, "
                  r"in the $M$-norm (square, W3, $N = 32$)")


# --------------------------------------------------------------------------
# 8.7 The spectrum in the complex plane
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
for ax, Rm in zip(axes, (1.0, 5.0, 20.0)):
    span = 0.0
    for form in ("A(CG)", "B(N1)", "B(N2)"):
        colour, marker = STYLE[form]
        s = solve_induction(domain="square", form=form, wind="W3", N=32, Rm=Rm,
                            n_eigs=14, nev=44, left=False)
        ax.scatter(s.values.real, s.values.imag, s=36, marker=marker,
                   facecolors="none", edgecolors=colour, linewidths=1.2, label=form)
        span = max(span, s.values.real.max())
    exact = exact_shear_family(6, Rm)
    exact = exact[exact <= span]        # only the family members actually resolved
    ax.scatter(exact, np.zeros_like(exact), s=95, marker="x", color="k",
               linewidths=1.0, label=r"exact $n^2/R_m$", zorder=0)
    ax.axhline(0, color="0.85", lw=0.6, zorder=-1)
    ax.set_title(rf"$R_m = {Rm:g}$"); ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    ax.set_xlim(-0.05 * span, 1.08 * span)
axes[0].set_ylabel(r"$\mathrm{Im}\,\lambda$")
axes[-1].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Computed spectra under the Heaviside shear, square domain, $N=32$", y=1.0)
fig.savefig(FIGDIR / "complex_plane.pdf")
plt.show()


# --------------------------------------------------------------------------
# 8.8 The full selection grid
# --------------------------------------------------------------------------

records = []
for domain in DOMAINS:
    for wind in ("W0", "W2", "W3"):
        for form in ("A(CG)", "B(N1)", "B(N2)"):
            kw = dict(tau=0.5) if domain == "lshape" else {}
            s = solve_induction(domain=domain, form=form, wind=wind, N=20,
                                Rm=5.0, n_eigs=3, **kw)
            records.append(dict(
                domain=domain, wind=wind, form=form, size=s.pencil.size,
                **{f"$\\lambda_{i+1}$": s.values[i] for i in range(3)},
                kappa=s.modes[0].condition))

grid = pd.DataFrame(records).set_index(["domain", "wind", "form"])
fmt_table(grid, default="{:.5f}", rules=[("size", "{:.0f}"), ("kappa", "{:.3g}")],
          caption=r"Every combination at $R_m = 5$, $N = 20$, degree 1 "
                  r"($\kappa$ is undefined for the self-adjoint case W0)")


# --------------------------------------------------------------------------
# 9. Notes, caveats and extensions
# --------------------------------------------------------------------------

print("figures written to", FIGDIR.resolve())
for p in sorted(FIGDIR.glob("*.pdf")):
    print("  ", p.name)

