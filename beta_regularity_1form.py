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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from firedrake import *
from firedrake.function import PointEvaluator
from firedrake.pyplot import triplot, tripcolor

from petsc4py import PETSc
from slepc4py import SLEPc

import spectral_common as sc
import pseudospectra_partial_schur as ps
from spectral_common import (crisscross_mesh, solve_pencil, border_gauge,
                             condition_estimates, rates, assign,
                             agreeing_digits, spread, L_DOMAIN, KAPPA_LOG)

# `from firedrake import *` exports a `logging` of its own, so the standard
# library one has to be re-imported under another name afterwards.
import logging as stdlogging
stdlogging.getLogger("firedrake").setLevel(stdlogging.ERROR)
stdlogging.getLogger("tsfc").setLevel(stdlogging.ERROR)

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
})
PS_RC = {
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}


def show(df, caption=None, **kw):
    """`sc.fmt_table` wired to this notebook's display function."""
    return sc.fmt_table(df, caption=caption,
                        display_fn=lambda c: display(Markdown(f"**{c}**")), **kw)


print(f"PETSc scalars: {PETSc.ScalarType.__name__}")


# --------------------------------------------------------------------------
# 1. The problem and the three formulations
# --------------------------------------------------------------------------

def cross2d(a, b):
    r"""Scalar 2D cross product $(a\times b)_z = a_1b_2 - a_2b_1$.

    Antisymmetric, so the ordering is load-bearing: the induction term needs
    $u\times\beta$, the proxy of the contraction $\iota^1_\beta u$.
    """
    return a[0] * b[1] - a[1] * b[0]


def space(form, mesh, r):
    """The mixed (or scalar) space of one formulation at polynomial degree r."""
    if form == "A(P)":
        return FunctionSpace(mesh, "CG", r)
    if form == "B(N1)":
        return FunctionSpace(mesh, "N1curl", r) * FunctionSpace(mesh, "CG", r)
    return FunctionSpace(mesh, "N2curl", r) * FunctionSpace(mesh, "CG", r + 1)


def build_pencil(form, mesh, r=1, beta=None, eps=Constant(1.0)):
    r"""Assemble $(A, M)$ for one formulation.  $M$ is singular in every case."""
    if form == "A(P)":
        V = space(form, mesh, r)
        a_, z = TrialFunction(V), TestFunction(V)
        a = eps * inner(grad(a_), grad(z)) * dx
        if beta is not None:
            a += inner(dot(beta, grad(a_)), z) * dx
        K, M = assemble(a).petscmat, assemble(inner(a_, z) * dx).petscmat
        # g_i = \int phi_i : the gauge (A, s) = 0 as a rank-one border.
        g = np.asarray(assemble(TestFunction(V) * dx).dat.data_ro).copy()
        A_b, M_b = border_gauge(K, M, g)
        return A_b, M_b, V

    W = space(form, mesh, r)
    (u, q) = TrialFunctions(W)
    (v, s) = TestFunctions(W)
    a = eps * inner(curl(u), curl(v)) * dx
    if beta is not None:
        a += inner(cross2d(u, beta), curl(v)) * dx
    a += (-inner(grad(q), v) + inner(u, grad(s))) * dx
    # On an H(curl) space a DirichletBC constrains the TANGENTIAL trace, which is
    # the perfectly-conducting condition; p = 0 puts the multiplier in H^1_0.
    bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary"),
           DirichletBC(W.sub(1), Constant(0.0), "on_boundary")]
    # weight=0.0 gives the constrained rows a zero mass diagonal, so the boundary
    # modes go to infinity rather than to a spurious lambda = 1.
    return (assemble(a, bcs=bcs).petscmat,
            assemble(inner(u, v) * dx, bcs=bcs, weight=0.0).petscmat, W)


FORMS = ["B(N1)", "B(N2)", "A(P)"]
TEX = {"B(N1)": r"$B(\mathcal{N}^{\mathrm{I}}_r)$",
       "B(N2)": r"$B(\mathcal{N}^{\mathrm{II}}_r)$",
       "A(P)":  r"$A(\mathcal{P}_r)$"}
TAG = {"B(N1)": "BN1", "B(N2)": "BN2", "A(P)": "AP"}
STYLE = {"B(N1)": "C0", "B(N2)": "C1", "A(P)": "C2"}


def solve_form(form, mesh, r=1, beta=None, Rm=1.0, tau=None, nev=25, n_eigs=None,
               **meta):
    r"""Assemble and solve one configuration.  ``tau`` defaults to $0.9\varepsilon$."""
    A, M, W = build_pencil(form, mesh, r, beta=beta, eps=Constant(1.0 / Rm))
    return solve_pencil(A, M, (0.9 / Rm) if tau is None else tau, nev=nev,
                        n_eigs=n_eigs, space=W, form=form, r=r, Rm=Rm, **meta)


def eigenfunction(res, k):
    """The k-th eigenmode: the H(curl) field, or the potential for A(P)."""
    vr, vi = res.A.createVecRight(), res.A.createVecRight()
    res.solver.getEigenvector(int(res.index[k]), vr, vi)
    arr = np.asarray(vr.getArray())
    if res.meta["form"] == "A(P)":
        f = Function(res.space)
        f.dat.data[:] = arr[:res.space.dim()]
        return f
    f = Function(res.space)
    with f.dat.vec_wo as fv:
        fv.setArray(arr[:fv.getLocalSize()])
    return f.subfunctions[0]


# --------------------------------------------------------------------------
# 2. The velocity fields
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Field:
    """A named velocity field together with the properties that matter here."""
    key: str
    tex: str
    regularity: str
    gradient: bool          # beta = grad(phi)?  -> real spectrum (King)
    solenoidal: bool
    sup: Optional[float]    # ||beta||_Linf, or None when unbounded
    _build: Callable

    def __call__(self, mesh, L=L_DOMAIN, **kw):
        return self._build(mesh, L, **kw)


def _b6(mesh, L):
    x, y = SpatialCoordinate(mesh)
    return as_vector([conditional(gt(x, Constant(L / 2)), 1.0, -1.0), Constant(0.0)])


def _b7(mesh, L):
    x, y = SpatialCoordinate(mesh)
    return as_vector([conditional(gt(y, Constant(L / 2)), 1.0, -1.0), Constant(0.0)])


def _b8(mesh, L):
    x, y = SpatialCoordinate(mesh)
    s = conditional(gt(x, Constant(L / 2)), 1.0, -1.0)
    return as_vector([s, s])


def _b9(mesh, L, alpha=1.5, delta=1e-12):
    r"""The $L^p$ vortex.  ``delta`` floors $r_c$ so that the expression stays
    finite if a quadrature point lands on the centre; it is far below the
    distance from the centre to the nearest quadrature point on any mesh used
    here, so it changes nothing that is measured."""
    x, y = SpatialCoordinate(mesh)
    xc = yc = Constant(L / 2)
    r = sqrt((x - xc) ** 2 + (y - yc) ** 2 + Constant(delta))
    return as_vector([-(y - yc) / r ** alpha, (x - xc) / r ** alpha])


FIELDS = {
    "b6": Field("b6", r"$(\mathrm{sign}(x-\frac{\pi}{2}),\,0)$",
                r"$L^\infty\setminus W^{1,\infty}$", True, False, 1.0, _b6),
    "b7": Field("b7", r"$(\mathrm{sign}(y-\frac{\pi}{2}),\,0)$",
                r"$L^\infty\setminus W^{1,\infty}$", False, True, 1.0, _b7),
    "b8": Field("b8", r"$(\mathrm{sign}(x-\frac{\pi}{2}),\,\mathrm{sign}(x-\frac{\pi}{2}))$",
                r"$L^\infty\setminus W^{1,\infty}$", False, False, np.sqrt(2.0), _b8),
    "b9": Field("b9", r"$(-\hat y/r^{\alpha},\ \hat x/r^{\alpha})$",
                r"$L^p$", False, True, None, _b9),
}
ROUGH = ["b6", "b7", "b8"]          # the L^inf \ W^{1,inf} family

mesh_f = crisscross_mesh(16)
beta_f = FIELDS["b6"](mesh_f)

# DG0 keeps a jump a jump; the quadrature always sees the exact expression.
bx = Function(FunctionSpace(mesh_f, "DG", 0)).interpolate(beta_f[0])
lim = float(np.percentile(np.abs(bx.dat.data_ro), 99)) or 1.0

bf = Function(VectorFunctionSpace(mesh_f, "CG", 1)).project(beta_f)
grid = np.linspace(0.12, np.pi - 0.12, 15)
Xf, Yf = np.meshgrid(grid, grid)
vals = np.asarray(PointEvaluator(mesh_f, np.column_stack([Xf.ravel(), Yf.ravel()]),
                                 missing_points_behaviour="ignore",
                                 tolerance=1e-6).evaluate(bf))
Uf, Vf = vals[:, 0].reshape(Xf.shape), vals[:, 1].reshape(Xf.shape)
step_f = float(grid[1] - grid[0])

fig, ax = plt.subplots(figsize=(4.6, 4.1))
art = tripcolor(bx, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.quiver(Xf, Yf, Uf, Vf, color="k", angles="xy", scale_units="xy",
          scale=np.nanmax(np.hypot(Uf, Vf)) / (0.8 * step_f), width=0.005, alpha=0.85)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"b6:  $\beta = (\mathrm{sign}(x-\pi/2),\,0)$")
fig.colorbar(art, ax=ax, label=r"$\beta_x$")
fig.savefig(FIGDIR / "breg1_field_b6.pdf")
plt.show()

mesh_f = crisscross_mesh(16)
beta_f = FIELDS["b7"](mesh_f)

bx = Function(FunctionSpace(mesh_f, "DG", 0)).interpolate(beta_f[0])
lim = float(np.percentile(np.abs(bx.dat.data_ro), 99)) or 1.0

bf = Function(VectorFunctionSpace(mesh_f, "CG", 1)).project(beta_f)
grid = np.linspace(0.12, np.pi - 0.12, 15)
Xf, Yf = np.meshgrid(grid, grid)
vals = np.asarray(PointEvaluator(mesh_f, np.column_stack([Xf.ravel(), Yf.ravel()]),
                                 missing_points_behaviour="ignore",
                                 tolerance=1e-6).evaluate(bf))
Uf, Vf = vals[:, 0].reshape(Xf.shape), vals[:, 1].reshape(Xf.shape)
step_f = float(grid[1] - grid[0])

fig, ax = plt.subplots(figsize=(4.6, 4.1))
art = tripcolor(bx, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.quiver(Xf, Yf, Uf, Vf, color="k", angles="xy", scale_units="xy",
          scale=np.nanmax(np.hypot(Uf, Vf)) / (0.8 * step_f), width=0.005, alpha=0.85)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"b7:  $\beta = (\mathrm{sign}(y-\pi/2),\,0)$")
fig.colorbar(art, ax=ax, label=r"$\beta_x$")
fig.savefig(FIGDIR / "breg1_field_b7.pdf")
plt.show()

mesh_f = crisscross_mesh(16)
beta_f = FIELDS["b8"](mesh_f)

bx = Function(FunctionSpace(mesh_f, "DG", 0)).interpolate(beta_f[0])
lim = float(np.percentile(np.abs(bx.dat.data_ro), 99)) or 1.0

bf = Function(VectorFunctionSpace(mesh_f, "CG", 1)).project(beta_f)
grid = np.linspace(0.12, np.pi - 0.12, 15)
Xf, Yf = np.meshgrid(grid, grid)
vals = np.asarray(PointEvaluator(mesh_f, np.column_stack([Xf.ravel(), Yf.ravel()]),
                                 missing_points_behaviour="ignore",
                                 tolerance=1e-6).evaluate(bf))
Uf, Vf = vals[:, 0].reshape(Xf.shape), vals[:, 1].reshape(Xf.shape)
step_f = float(grid[1] - grid[0])

fig, ax = plt.subplots(figsize=(4.6, 4.1))
art = tripcolor(bx, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.quiver(Xf, Yf, Uf, Vf, color="k", angles="xy", scale_units="xy",
          scale=np.nanmax(np.hypot(Uf, Vf)) / (0.8 * step_f), width=0.005, alpha=0.85)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"b8:  $\beta = (\mathrm{sign}(x-\pi/2),\,\mathrm{sign}(x-\pi/2))$")
fig.colorbar(art, ax=ax, label=r"$\beta_x$")
fig.savefig(FIGDIR / "breg1_field_b8.pdf")
plt.show()

mesh_f = crisscross_mesh(16)
beta_f = FIELDS["b9"](mesh_f, alpha=1.5)

# The vortex is unbounded at the centre, so the colour scale is clipped at the
# 99th percentile; the quadrature still sees the exact expression.
bx = Function(FunctionSpace(mesh_f, "DG", 0)).interpolate(beta_f[0])
lim = float(np.percentile(np.abs(bx.dat.data_ro), 99)) or 1.0

bf = Function(VectorFunctionSpace(mesh_f, "CG", 1)).project(beta_f)
grid = np.linspace(0.12, np.pi - 0.12, 15)
Xf, Yf = np.meshgrid(grid, grid)
vals = np.asarray(PointEvaluator(mesh_f, np.column_stack([Xf.ravel(), Yf.ravel()]),
                                 missing_points_behaviour="ignore",
                                 tolerance=1e-6).evaluate(bf))
Uf, Vf = vals[:, 0].reshape(Xf.shape), vals[:, 1].reshape(Xf.shape)
step_f = float(grid[1] - grid[0])

fig, ax = plt.subplots(figsize=(4.6, 4.1))
art = tripcolor(bx, axes=ax, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.quiver(Xf, Yf, Uf, Vf, color="k", angles="xy", scale_units="xy",
          scale=np.nanmax(np.hypot(Uf, Vf)) / (0.8 * step_f), width=0.005, alpha=0.85)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"b9:  vortex, $\alpha = 1.5$   (colour clipped at the 99th percentile)")
fig.colorbar(art, ax=ax, label=r"$\beta_x$")
fig.savefig(FIGDIR / "breg1_field_b9.pdf")
plt.show()


# --------------------------------------------------------------------------
# 3. Fields in $L^\infty\setminus W^{1,\infty}$
# --------------------------------------------------------------------------

N_REF, R_REF, NEV_REF = 24, 3, 20

mesh_ref = crisscross_mesh(N_REF)
ref_runs, REFERENCE = {}, {}
for key in ROUGH:
    beta_ref = FIELDS[key](mesh_ref)
    for form in FORMS:
        ref_runs[(key, form)] = solve_form(form, mesh_ref, R_REF, beta=beta_ref,
                                           nev=NEV_REF, n_eigs=5, field=key)
    stack = np.array([ref_runs[(key, f)].values[:5] for f in FORMS])
    # The reference is the mean of the three: they agree to far better than the
    # discretisation error of anything measured against it.
    REFERENCE[key] = stack.mean(axis=0)

rows = []
for key in ROUGH:
    stack = np.array([ref_runs[(key, f)].values[:5] for f in FORMS])
    sprd = spread(stack)
    for j in range(5):
        rows.append(dict(field=key, j=j + 1,
                         **{TEX["B(N1)"]: stack[0, j], TEX["B(N2)"]: stack[1, j],
                            TEX["A(P)"]: stack[2, j], "spread": sprd[j],
                            "digits": agreeing_digits(stack[:, j])}))
df_cv = pd.DataFrame(rows).set_index(["field", "j"])
show(df_cv, default="{:.9f}", rules=[("spread", "{:.2e}"), ("digits", "{:.0f}")],
     caption=rf"Cross-validation of the first five eigenvalues, $R_m = 1$, "
             rf"$N = {N_REF}$, $r = {R_REF}$")

print("conditioning of the reference computations\n")
for key in ROUGH:
    line = "  ".join(f"{f}: {ref_runs[(key, f)].kappa1:.2e}" for f in FORMS)
    print(f"  {key}   {line}")

# For b6 and b7 the values n^2/Rm are exact whatever the roughness of the profile,
# which validates the reference independently of the cross-validation.
print("\nexact-family check (lambda = n^2 for beta = (f(x),0) and (f(y),0))")
for key in ("b6", "b7"):
    got = REFERENCE[key].real
    for target in (1.0, 4.0):
        near = got[int(np.argmin(np.abs(got - target)))]
        print(f"  {key}: exact {target:.0f}  ->  computed {near:.12f}"
              f"   (error {abs(near - target):.2e})")


# --------------------------------------------------------------------------
# 3.2 $h$-refinement
# --------------------------------------------------------------------------

H_LEVELS = (4, 8, 16, 24, 32)
EIG_COLOURS = ["C0", "C1", "C2", "C3", "C4"]
EIG_COLS = [rf"$\lambda_{{{j+1}}}$" for j in range(5)]
REFERENCE_LP = {}          # filled in Section 4


def reference_for(field_key, alpha):
    return REFERENCE[field_key] if alpha is None else REFERENCE_LP[(field_key, alpha)]


def h_study(form, field_key, levels=H_LEVELS, r=1, Rm=1.0, alpha=None):
    """Errors of the first five eigenvalues against the reference, level by level."""
    ref = reference_for(field_key, alpha)
    rows = []
    for N in levels:
        mesh = crisscross_mesh(N)
        beta = (FIELDS[field_key](mesh) if alpha is None
                else FIELDS[field_key](mesh, alpha=alpha))
        res = solve_form(form, mesh, r, beta=beta, Rm=Rm, nev=25, field=field_key)
        # Matched by value, not by position, so that an inserted or missing mode
        # cannot masquerade as a loss of convergence.
        got = assign(ref, res.values)
        rows.append(dict(N=N, h=np.pi / N, dof=res.size,
                         **{EIG_COLS[j]: abs(got[j] - ref[j]) / abs(ref[j])
                            for j in range(5)},
                         **{r"$\kappa_1$": res.kappa1}))
    return pd.DataFrame(rows).set_index("N")


def rate_frame(df, cols):
    """Interleave each error column with its observed order."""
    out = pd.DataFrame(index=df.index)
    out["h"] = df["h"]
    for c in cols:
        out[c] = df[c]
        out[f"rate {c}"] = rates(df["h"], df[c])
    out[r"$\kappa_1$"] = df[r"$\kappa_1$"]
    return out


h_tables = {(f, "b7"): h_study(f, "b7") for f in FORMS}
for f in FORMS:
    display(show(rate_frame(h_tables[(f, "b7")], EIG_COLS), default="{:.3e}",
                 rules=[("rate", "{:.2f}"), ("h", "{:.5f}"), (r"$\kappa_1$", "{:.2e}")],
                 caption=rf"$h$-refinement, **{f}**, field `b7`, $r = 1$, $R_m = 1$"))

df = h_tables[("B(N1)", "b7")]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_1$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_2$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_3$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_4$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_5$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathcal{N}^{\mathrm{I}}_1)$, field b7, $r=1$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_h_b7_BN1.pdf")
plt.show()

df = h_tables[("B(N2)", "b7")]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_1$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_2$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_3$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_4$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_5$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathcal{N}^{\mathrm{II}}_1)$, field b7, $r=1$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_h_b7_BN2.pdf")
plt.show()

df = h_tables[("A(P)", "b7")]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_1$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_2$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_3$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_4$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_5$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$A(\mathcal{P}_1)$, field b7, $r=1$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_h_b7_AP.pdf")
plt.show()


# --------------------------------------------------------------------------
# 3.3 $p$-refinement
# --------------------------------------------------------------------------

P_LEVELS = (1, 2, 3, 4)
N_PREF = 8


def p_study(form, field_key, levels=P_LEVELS, N=N_PREF, Rm=1.0, alpha=None):
    """Errors of the first five eigenvalues as the degree rises on a fixed mesh."""
    ref = reference_for(field_key, alpha)
    mesh = crisscross_mesh(N)
    beta = (FIELDS[field_key](mesh) if alpha is None
            else FIELDS[field_key](mesh, alpha=alpha))
    rows = []
    for r in levels:
        res = solve_form(form, mesh, r, beta=beta, Rm=Rm, nev=25, field=field_key)
        got = assign(ref, res.values)
        rows.append(dict(r=r, dof=res.size,
                         **{EIG_COLS[j]: abs(got[j] - ref[j]) / abs(ref[j])
                            for j in range(5)},
                         **{r"$\kappa_1$": res.kappa1}))
    return pd.DataFrame(rows).set_index("r")


p_tables = {(f, "b7"): p_study(f, "b7") for f in FORMS}
for f in FORMS:
    display(show(p_tables[(f, "b7")], default="{:.3e}",
                 rules=[("dof", "{:.0f}"), (r"$\kappa_1$", "{:.2e}")],
                 caption=rf"$p$-refinement, **{f}**, field `b7`, $N = {N_PREF}$, $R_m = 1$"))

df = p_tables[("B(N1)", "b7")]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_1$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_2$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_3$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_4$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_5$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathcal{N}^{\mathrm{I}}_r)$, field b7, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_p_b7_BN1.pdf")
plt.show()

df = p_tables[("B(N2)", "b7")]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_1$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_2$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_3$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_4$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_5$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathcal{N}^{\mathrm{II}}_r)$, field b7, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_p_b7_BN2.pdf")
plt.show()

df = p_tables[("A(P)", "b7")]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_1$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_2$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_3$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_4$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_5$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$A(\mathcal{P}_r)$, field b7, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_p_b7_AP.pdf")
plt.show()


# --------------------------------------------------------------------------
# 3.4 The other fields
# --------------------------------------------------------------------------

rows = []
for key in ROUGH:
    for form in FORMS:
        if (form, key) not in h_tables:
            h_tables[(form, key)] = h_study(form, key)
        df = h_tables[(form, key)]
        row = {"field": key, "formulation": form}
        for col in EIG_COLS:
            row[f"rate {col}"] = rates(df["h"], df[col])[-1]
        row[r"$\kappa_1$ (finest)"] = df[r"$\kappa_1$"].iloc[-1]
        rows.append(row)
df_rates = pd.DataFrame(rows).set_index(["field", "formulation"])
show(df_rates, default="{:.2f}", rules=[(r"$\kappa_1$ (finest)", "{:.2e}")],
     caption=r"Observed $h$-refinement orders at the finest level, for all three "
             r"$L^\infty\setminus W^{1,\infty}$ fields and all three formulations "
             r"($r = 1$, $R_m = 1$)")


# --------------------------------------------------------------------------
# 3.5 An expanded spectral window
# --------------------------------------------------------------------------

PS_N, PS_R, PS_NEV = 20, 1, 90
PS_WINDOW = ((0.0, 22.0), (-5.0, 5.0))

mesh_ps = crisscross_mesh(PS_N)


def run_ps(form, field_key, Rm=1.0, alpha=None, window=PS_WINDOW, nev=PS_NEV,
           mesh=None, r=PS_R):
    """Assemble one pencil and hand it to the pseudospectrum engine."""
    mesh = mesh_ps if mesh is None else mesh
    beta = (FIELDS[field_key](mesh) if alpha is None
            else FIELDS[field_key](mesh, alpha=alpha))
    A, M, W = build_pencil(form, mesh, r, beta=beta, eps=Constant(1.0 / Rm))
    kappa = condition_estimates(A, M, 0.9 / Rm)
    # NB: `field` is already a key of the result dict (the sigma_min grid), so
    # the metadata key for which velocity field this is must be named otherwise.
    out = ps.pseudospectrum(A, M, 0.9 / Rm, nev=nev, window=window,
                            form=form, beta=field_key, Rm=Rm, alpha=alpha,
                            label=FIELDS[field_key].tex)
    out["kappa1"] = kappa[r"Hager $\kappa_1$"]
    return out


ps_b7 = {f: run_ps(f, "b7") for f in FORMS}
show(ps.diagnostics_frame(ps_b7.values(), keys=("form", "beta")), default="{:.3e}",
     rules=[("N", "{:.0f}"), ("m", "{:.0f}"), ("eigs in window", "{:.0f}")],
     caption=rf"Projection diagnostics for the wide-window scan, field `b7`, "
             rf"$N = {PS_N}$, $r = {PS_R}$, nev $= {PS_NEV}$")

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ps.plot_pseudospectrum(ps_b7["B(N1)"], ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{I}}_1)$, field b7, $R_m=1$"
                                 + rf"   ($\kappa_1 = {ps_b7['B(N1)']['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_wide_b7_BN1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ps.plot_pseudospectrum(ps_b7["B(N2)"], ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{II}}_1)$, field b7, $R_m=1$"
                                 + rf"   ($\kappa_1 = {ps_b7['B(N2)']['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_wide_b7_BN2.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ps.plot_pseudospectrum(ps_b7["A(P)"], ax=ax,
                           title=r"$A(\mathcal{P}_1)$, field b7, $R_m=1$"
                                 + rf"   ($\kappa_1 = {ps_b7['A(P)']['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_wide_b7_AP.pdf")
    plt.show()


# --------------------------------------------------------------------------
# 3.6 The same scan under increasing $R_m$
# --------------------------------------------------------------------------

PS_WINDOW_B8 = ((0.0, 12.0), (-4.0, 4.0))
# Cell Peclet: ||beta_8|| = sqrt(2), so N >~ pi*sqrt(2)*Rm/2 keeps the
# unstabilised discretisation honest.  N = 20 supports Rm up to about 9.
for Rm in (1.0, 5.0):
    print(f"Rm = {Rm:4.1f}   cell Peclet = "
          f"{(np.pi / PS_N) * FIELDS['b8'].sup * Rm / 2:.2f}")

ps_b8 = {(f, Rm): run_ps(f, "b8", Rm=Rm, window=PS_WINDOW_B8)
         for Rm in (1.0, 5.0) for f in FORMS}
show(ps.diagnostics_frame(ps_b8.values(), keys=("form", "Rm")), default="{:.3e}",
     rules=[("N", "{:.0f}"), ("m", "{:.0f}"), ("eigs in window", "{:.0f}"),
            ("Rm", "{:.1f}")],
     caption=r"Projection diagnostics for the `b8` scan at two Reynolds numbers")

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_b8[("B(N1)", 1.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{I}}_1)$, b8, $R_m=1$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b8_BN1_Rm1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_b8[("B(N2)", 1.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{II}}_1)$, b8, $R_m=1$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b8_BN2_Rm1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_b8[("A(P)", 1.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$A(\mathcal{P}_1)$, b8, $R_m=1$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b8_AP_Rm1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_b8[("B(N1)", 5.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{I}}_1)$, b8, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b8_BN1_Rm5.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_b8[("B(N2)", 5.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{II}}_1)$, b8, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b8_BN2_Rm5.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_b8[("A(P)", 5.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$A(\mathcal{P}_1)$, b8, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b8_AP_Rm5.pdf")
    plt.show()

# A scalar summary of how far each pseudospectrum bulges beyond a disc about the
# spectrum.  eps is set per pencil as a low quantile of its own sampled sigma_min,
# because the absolute level is not comparable across formulations (Section 3.5).
def bulge(result, q=2.0):
    """max{ dist(z, spectrum) : sigma_min(z) <= eps } - eps, with eps the q-th
    percentile of the sampled sigma_min.  Zero for a normal operator."""
    eps_level = float(np.percentile(result["sigma"], q))
    inside = result["sigma"] <= eps_level
    if not inside.any() or result["ritz"].size == 0:
        return np.nan, eps_level
    d = np.min(np.abs(result["z"][inside][:, None] - result["ritz"][None, :]), axis=1)
    return float(d.max() - eps_level), eps_level


rows = []
for Rm in (1.0, 5.0):
    for form in FORMS:
        b, lvl = bulge(ps_b8[(form, Rm)])
        rows.append(dict(Rm=Rm, formulation=form,
                         **{r"$\varepsilon$ (2nd pctile)": lvl, "bulge": b,
                            r"$\kappa_1$": ps_b8[(form, Rm)]["kappa1"]}))
show(pd.DataFrame(rows).set_index(["Rm", "formulation"]), default="{:.3e}",
     rules=[("bulge", "{:.3f}"), ("Rm", "{:.1f}")],
     caption=r"How far the $\varepsilon$-pseudospectrum extends beyond an "
             r"$\varepsilon$-disc about the spectrum, field `b8`")


# --------------------------------------------------------------------------
# 3.7 Convergence rates as $R_m$ varies
# --------------------------------------------------------------------------

RM_LEVELS = (16, 24, 32, 40)
RM_VALUES = (1.0, 2.0, 5.0, 10.0)

# The unstabilised discretisation needs h ||beta|| Rm / 2 <~ 1; with ||beta|| = 1
# and the coarsest level N = 16, that caps Rm at about 10.
print("cell Peclet at the coarsest level used:")
for Rm in RM_VALUES:
    print(f"  Rm = {Rm:5.1f}:  {(np.pi / RM_LEVELS[0]) * FIELDS['b7'].sup * Rm / 2:.2f}")

REFERENCE_RM = {}
for Rm in RM_VALUES:
    beta_r = FIELDS["b7"](mesh_ref)
    stack = np.array([solve_form(f, mesh_ref, R_REF, beta=beta_r, Rm=Rm,
                                 nev=NEV_REF, n_eigs=5, field="b7").values[:5]
                      for f in FORMS])
    REFERENCE_RM[Rm] = stack.mean(axis=0)
    print(f"  Rm = {Rm:5.1f}: reference lambda_1..3 = {np.round(stack.mean(axis=0)[:3], 6)}"
          f"   spread {spread(stack).max():.1e}")

rows = []
for Rm in RM_VALUES:
    ref = REFERENCE_RM[Rm]
    for form in FORMS:
        errs, kap = [], []
        for N in RM_LEVELS:
            mesh = crisscross_mesh(N)
            res = solve_form(form, mesh, 1, beta=FIELDS["b7"](mesh), Rm=Rm,
                             nev=25, field="b7")
            got = assign(ref, res.values)
            errs.append([abs(got[j] - ref[j]) / abs(ref[j]) for j in range(5)])
            kap.append(res.kappa1)
        errs = np.array(errs)
        h = np.pi / np.array(RM_LEVELS, dtype=float)
        per_mode = np.array([rates(h, errs[:, j])[-1] for j in range(5)])
        rows.append(dict(Rm=Rm, formulation=form,
                         **{r"rate $\lambda_1$": per_mode[0],
                            "mean rate": per_mode.mean(),
                            "min rate": per_mode.min(),
                            r"$\kappa_1$ (finest)": kap[-1]}))
df_rm = pd.DataFrame(rows).set_index(["Rm", "formulation"])
show(df_rm, default="{:.2f}",
     rules=[(r"$\kappa_1$ (finest)", "{:.2e}"), ("Rm", "{:.1f}")],
     caption=r"Observed $h$-refinement orders for the first five eigenvalues as $R_m$ "
             r"varies, field `b7`, $r = 1$")


# --------------------------------------------------------------------------
# 4. Fields in $L^p$
# --------------------------------------------------------------------------

ALPHAS = (1.0, 1.25, 1.5, 1.75)
ALPHA_MAIN = 1.5

# What the discretisation actually sees: the largest cell-average of |beta|.
print("effective magnitude of the vortex, by mesh and exponent")
print(f"{'N':>5s}" + "".join(f"{'a=' + format(a, '.2f'):>12s}" for a in ALPHAS))
for N in (8, 16, 32):
    mesh_e = crisscross_mesh(N)
    Q0 = FunctionSpace(mesh_e, "DG", 0)
    line = f"{N:5d}"
    for a in ALPHAS:
        b = FIELDS["b9"](mesh_e, alpha=a)
        mag = Function(Q0).interpolate(sqrt(dot(b, b)))
        line += f"{float(np.abs(mag.dat.data_ro).max()):12.2f}"
    print(line)

mesh_ref_lp = crisscross_mesh(N_REF)
ref_runs_lp = {}
for a in ALPHAS:
    beta_a = FIELDS["b9"](mesh_ref_lp, alpha=a)
    for form in FORMS:
        ref_runs_lp[(a, form)] = solve_form(form, mesh_ref_lp, R_REF, beta=beta_a,
                                            nev=NEV_REF, n_eigs=5, field="b9",
                                            alpha=a)
    stack = np.array([ref_runs_lp[(a, f)].values[:5] for f in FORMS])
    REFERENCE_LP[("b9", a)] = stack.mean(axis=0)

rows = []
for a in ALPHAS:
    stack = np.array([ref_runs_lp[(a, f)].values[:5] for f in FORMS])
    sprd = spread(stack)
    for j in range(5):
        rows.append(dict(alpha=a, j=j + 1,
                         **{TEX["B(N1)"]: stack[0, j], TEX["B(N2)"]: stack[1, j],
                            TEX["A(P)"]: stack[2, j], "spread": sprd[j],
                            "digits": agreeing_digits(stack[:, j])}))
df_cv_lp = pd.DataFrame(rows).set_index(["alpha", "j"])
show(df_cv_lp, default="{:.8f}",
     rules=[("spread", "{:.2e}"), ("digits", "{:.0f}"), ("alpha", "{:.2f}")],
     caption=rf"Cross-validation for the $L^p$ vortex, $R_m = 1$, $N = {N_REF}$, "
             rf"$r = {R_REF}$, as the singularity strengthens")

summary = []
for a in ALPHAS:
    stack = np.array([ref_runs_lp[(a, f)].values[:5] for f in FORMS])
    summary.append(dict(alpha=a,
                        **{r"$p$ limit": (2.0 / (a - 1.0)) if a > 1.0 else np.inf,
                           "max spread": spread(stack).max(),
                           "min digits": min(agreeing_digits(stack[:, j])
                                             for j in range(5)),
                           r"$\kappa_1$ B(N1)": ref_runs_lp[(a, "B(N1)")].kappa1,
                           r"$\kappa_1$ A(P)": ref_runs_lp[(a, "A(P)")].kappa1}))
show(pd.DataFrame(summary).set_index("alpha"), default="{:.3e}",
     rules=[("alpha", "{:.2f}"), (r"$p$ limit", "{:.2f}"), ("min digits", "{:.0f}")],
     caption=r"Cross-formulation agreement degrades monotonically with the strength "
             r"of the singularity")


# --------------------------------------------------------------------------
# 4.2 $h$- and $p$-refinement
# --------------------------------------------------------------------------

h_tables_lp = {f: h_study(f, "b9", alpha=ALPHA_MAIN) for f in FORMS}
for f in FORMS:
    display(show(rate_frame(h_tables_lp[f], EIG_COLS), default="{:.3e}",
                 rules=[("rate", "{:.2f}"), ("h", "{:.5f}"), (r"$\kappa_1$", "{:.2e}")],
                 caption=rf"$h$-refinement, **{f}**, vortex $\alpha = {ALPHA_MAIN}$, "
                         rf"$r = 1$, $R_m = 1$"))

df = h_tables_lp["B(N1)"]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_1$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_2$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_3$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_4$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_5$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathcal{N}^{\mathrm{I}}_1)$, vortex $\alpha=1.5$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_h_b9_BN1.pdf")
plt.show()

df = h_tables_lp["B(N2)"]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_1$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_2$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_3$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_4$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_5$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathcal{N}^{\mathrm{II}}_1)$, vortex $\alpha=1.5$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_h_b9_BN2.pdf")
plt.show()

df = h_tables_lp["A(P)"]
h = df["h"].to_numpy()
e1 = df[EIG_COLS[0]].to_numpy()
e2 = df[EIG_COLS[1]].to_numpy()
e3 = df[EIG_COLS[2]].to_numpy()
e4 = df[EIG_COLS[3]].to_numpy()
e5 = df[EIG_COLS[4]].to_numpy()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.loglog(h, e1, "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2,
          label=rf"$\lambda_1$  ({np.nanmean(rates(h, e1)[1:]):.2f})")
ax.loglog(h, e2, "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2,
          label=rf"$\lambda_2$  ({np.nanmean(rates(h, e2)[1:]):.2f})")
ax.loglog(h, e3, "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2,
          label=rf"$\lambda_3$  ({np.nanmean(rates(h, e3)[1:]):.2f})")
ax.loglog(h, e4, "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2,
          label=rf"$\lambda_4$  ({np.nanmean(rates(h, e4)[1:]):.2f})")
ax.loglog(h, e5, "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2,
          label=rf"$\lambda_5$  ({np.nanmean(rates(h, e5)[1:]):.2f})")

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$A(\mathcal{P}_1)$, vortex $\alpha=1.5$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_h_b9_AP.pdf")
plt.show()

p_tables_lp = {f: p_study(f, "b9", alpha=ALPHA_MAIN) for f in FORMS}
for f in FORMS:
    display(show(p_tables_lp[f], default="{:.3e}",
                 rules=[("dof", "{:.0f}"), (r"$\kappa_1$", "{:.2e}")],
                 caption=rf"$p$-refinement, **{f}**, vortex $\alpha = {ALPHA_MAIN}$, "
                         rf"$N = {N_PREF}$, $R_m = 1$"))

df = p_tables_lp["B(N1)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_1$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_2$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_3$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_4$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_5$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathcal{N}^{\mathrm{I}}_r)$, vortex $\alpha=1.5$, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_p_b9_BN1.pdf")
plt.show()

df = p_tables_lp["B(N2)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_1$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_2$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_3$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_4$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_5$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathcal{N}^{\mathrm{II}}_r)$, vortex $\alpha=1.5$, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_p_b9_BN2.pdf")
plt.show()

df = p_tables_lp["A(P)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_1$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_2$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_3$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_4$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_5$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$A(\mathcal{P}_r)$, vortex $\alpha=1.5$, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "breg1_p_b9_AP.pdf")
plt.show()


# --------------------------------------------------------------------------
# 4.3 Convergence as $R_m$ varies
# --------------------------------------------------------------------------

RM_VALUES_LP = (1.0, 2.0, 5.0)
REFERENCE_RM_LP = {}
for Rm in RM_VALUES_LP:
    beta_a = FIELDS["b9"](mesh_ref_lp, alpha=ALPHA_MAIN)
    stack = np.array([solve_form(f, mesh_ref_lp, R_REF, beta=beta_a, Rm=Rm,
                                 nev=NEV_REF, n_eigs=5, field="b9").values[:5]
                      for f in FORMS])
    REFERENCE_RM_LP[Rm] = stack.mean(axis=0)

rows = []
for Rm in RM_VALUES_LP:
    ref = REFERENCE_RM_LP[Rm]
    for form in FORMS:
        errs, kap = [], []
        for N in RM_LEVELS:
            mesh = crisscross_mesh(N)
            res = solve_form(form, mesh, 1, beta=FIELDS["b9"](mesh, alpha=ALPHA_MAIN),
                             Rm=Rm, nev=25, field="b9")
            got = assign(ref, res.values)
            errs.append([abs(got[j] - ref[j]) / abs(ref[j]) for j in range(5)])
            kap.append(res.kappa1)
        errs = np.array(errs)
        h = np.pi / np.array(RM_LEVELS, dtype=float)
        per_mode = np.array([rates(h, errs[:, j])[-1] for j in range(5)])
        rows.append(dict(Rm=Rm, formulation=form,
                         **{r"rate $\lambda_1$": per_mode[0],
                            "mean rate": per_mode.mean(),
                            "min rate": per_mode.min(),
                            r"$\kappa_1$ (finest)": kap[-1]}))
show(pd.DataFrame(rows).set_index(["Rm", "formulation"]), default="{:.2f}",
     rules=[(r"$\kappa_1$ (finest)", "{:.2e}"), ("Rm", "{:.1f}")],
     caption=rf"Observed $h$-refinement orders as $R_m$ varies, vortex "
             rf"$\alpha = {ALPHA_MAIN}$, $r = 1$")


# --------------------------------------------------------------------------
# 4.4 Pseudospectra across formulations and $R_m$
# --------------------------------------------------------------------------

PS_WINDOW_LP = ((0.0, 12.0), (-6.0, 6.0))
ps_lp = {(f, Rm): run_ps(f, "b9", Rm=Rm, alpha=ALPHA_MAIN, window=PS_WINDOW_LP)
         for Rm in (1.0, 5.0) for f in FORMS}
show(ps.diagnostics_frame(ps_lp.values(), keys=("form", "Rm")), default="{:.3e}",
     rules=[("N", "{:.0f}"), ("m", "{:.0f}"), ("eigs in window", "{:.0f}"),
            ("Rm", "{:.1f}")],
     caption=rf"Projection diagnostics, vortex $\alpha = {ALPHA_MAIN}$")

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_lp[("B(N1)", 1.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{I}}_1)$, vortex $\alpha=1.5$, $R_m=1$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b9_BN1_Rm1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_lp[("B(N2)", 1.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{II}}_1)$, vortex $\alpha=1.5$, $R_m=1$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b9_BN2_Rm1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_lp[("A(P)", 1.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$A(\mathcal{P}_1)$, vortex $\alpha=1.5$, $R_m=1$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b9_AP_Rm1.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_lp[("B(N1)", 5.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{I}}_1)$, vortex $\alpha=1.5$, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b9_BN1_Rm5.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_lp[("B(N2)", 5.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathcal{N}^{\mathrm{II}}_1)$, vortex $\alpha=1.5$, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b9_BN2_Rm5.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    r_ = ps_lp[("A(P)", 5.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$A(\mathcal{P}_1)$, vortex $\alpha=1.5$, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "breg1_ps_b9_AP_Rm5.pdf")
    plt.show()


# --------------------------------------------------------------------------
# 4.5 Non-normality against the strength of the singularity
# --------------------------------------------------------------------------

RM_HIGH = 5.0
rows = []
for a in ALPHAS:
    for form in FORMS:
        r_ = run_ps(form, "b9", Rm=RM_HIGH, alpha=a, window=PS_WINDOW_LP)
        b, lvl = bulge(r_)
        rows.append(dict(alpha=a, formulation=form,
                         **{r"$\varepsilon$ (2nd pctile)": lvl, "bulge": b,
                            r"max $|\mathrm{Im}\,\lambda|$":
                                float(np.abs(r_["ritz_in"].imag).max())
                                if r_["ritz_in"].size else np.nan,
                            r"$\kappa_1$": r_["kappa1"]}))
show(pd.DataFrame(rows).set_index(["alpha", "formulation"]), default="{:.3e}",
     rules=[("bulge", "{:.3f}"), ("alpha", "{:.2f}"),
            (r"max $|\mathrm{Im}\,\lambda|$", "{:.3f}")],
     caption=rf"Non-normality against the strength of the singularity, at fixed "
             rf"$R_m = {RM_HIGH:g}$, $N = {PS_N}$, $r = {PS_R}$")


# --------------------------------------------------------------------------
# 5. Summary
# --------------------------------------------------------------------------

klog = pd.DataFrame(KAPPA_LOG)
worst = klog.loc[klog[r"Hager $\kappa_1$"].idxmax()]
print(f"{len(klog)} eigensolves, each with kappa(A - tau M) estimated.\n")
print(f"  worst normwise kappa_1 : {worst[r'Hager $\kappa_1$']:.3e}"
      f"   ({worst.get('form', '?')}, field {worst.get('field', '?')}, "
      f"r={worst.get('r', 0):.0f}, Rm={worst.get('Rm', 1):g}, n={worst['n']:.0f})")
print(f"  median normwise kappa_1: {klog[r'Hager $\kappa_1$'].median():.3e}")
print(f"  worst MUMPS COND1      : {klog['MUMPS COND1'].max():.3e}")
print(f"  MUMPS INFOG(1) != 0    : {int((klog['INFOG(1)'] != 0).sum())} of {len(klog)}")
print(f"\nfigures written to {FIGDIR.resolve()}")
for q in sorted(FIGDIR.glob("breg1_*.pdf")):
    print("  ", q.name)

