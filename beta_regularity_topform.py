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
import scipy.sparse as sp
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
from spectral_common import (crisscross_mesh, solve_pencil, condition_estimates,
                             to_scipy, to_petsc, rates, assign, agreeing_digits,
                             spread, L_DOMAIN, KAPPA_LOG)

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
# 1. The problem and the formulations
# --------------------------------------------------------------------------

def crisscross_blocks(mesh, N, L=L_DOMAIN):
    """Group the 4N^2 cells into N^2 macro-squares, by centroid.

    Nothing depends on Firedrake's internal cell ordering: within a macro-square
    the four sub-triangle centroids sit at the centre plus (+-h/3, 0) and
    (0, +-h/3), so |dy| > |dx| picks out the north/south pair.
    """
    W = VectorFunctionSpace(mesh, "DG", 0)
    centroid = Function(W).interpolate(SpatialCoordinate(mesh)).dat.data_ro
    h = L / N
    idx = np.clip(np.floor(centroid / h).astype(int), 0, N - 1)
    offset = centroid - (idx + 0.5) * h
    north_south = np.abs(offset[:, 1]) > np.abs(offset[:, 0])
    blocks = np.argsort(idx[:, 0] * N + idx[:, 1], kind="stable").reshape(N * N, 4)
    return blocks, north_south


def checkerboard_complement(mesh, N, L=L_DOMAIN):
    r"""Orthonormal basis of $V_h = \operatorname{div}[P_1]^2$.

    The cokernel of the divergence is spanned, one vector per macro-square, by the
    checkerboard $+1$ on north/south and $-1$ on east/west; these have disjoint
    supports and are therefore mutually orthogonal, so a basis of the complement
    can be written down three vectors at a time.
    """
    blocks, north_south = crisscross_blocks(mesh, N, L)
    ncell, q = 4 * N * N, 1 / np.sqrt(2)
    rows, cols, vals = [], [], []
    for k, blk in enumerate(blocks):
        ns, ew = blk[north_south[blk]], blk[~north_south[blk]]
        for j, (cells, v) in enumerate([(tuple(ns), (q, -q)), (tuple(ew), (q, -q)),
                                        (tuple(blk), (0.5,) * 4)]):
            rows += list(cells); cols += [3 * k + j] * len(cells); vals += list(v)
    return sp.csr_matrix((vals, (rows, cols)), shape=(ncell, 3 * N * N))


def build_pencil(form, mesh, N, r=1, beta=None, eps=Constant(1.0), nu=0.0):
    r"""Assemble $(A, M)$ for one formulation.  $M$ is singular in every case."""
    if form == "A(CG)":
        V = FunctionSpace(mesh, "CG", r)
        u, v = TrialFunction(V), TestFunction(V)
        a = eps * inner(grad(u), grad(v)) * dx
        if beta is not None:
            a -= inner(u * beta, grad(v)) * dx      # iota^n_beta u  <->  u * beta
        if nu:
            a += Constant(nu) * inner(u, v) * dx
        bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
        return (assemble(a, bcs=bcs).petscmat,
                assemble(inner(u, v) * dx, bcs=bcs, weight=0.0).petscmat, V)

    if form in ("B(RT)", "B(BDM)", "s(RT)", "s(BDM)"):
        family = "RT" if form.endswith("(RT)") else "BDM"
        W = FunctionSpace(mesh, family, r) * FunctionSpace(mesh, "DG", r - 1)
        (s_, u) = TrialFunctions(W)
        (t_, v) = TestFunctions(W)
        a = ((1 / eps) * inner(s_, t_) * dx
             - inner(u, div(t_)) * dx
             + inner(div(s_), v) * dx)
        if beta is not None:
            a -= (1 / eps) * inner(u * beta, t_) * dx
        if nu:
            a += Constant(nu) * inner(u, v) * dx
        # No boundary condition anywhere: u = 0 on the boundary is natural here.
        return assemble(a).petscmat, assemble(inner(u, v) * dx).petscmat, W

    # s(P1-divP1): assembled blockwise, then the DG0 side restricted to div(Sigma_h).
    S = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    s_, t_ = TrialFunction(S), TestFunction(S)
    u, v = TrialFunction(Q), TestFunction(Q)
    A_ss = to_scipy(assemble((1 / eps) * inner(s_, t_) * dx).petscmat)
    form_su = -inner(u, div(t_)) * dx
    if beta is not None:
        form_su = form_su - (1 / eps) * inner(u * beta, t_) * dx
    A_su = to_scipy(assemble(form_su).petscmat)
    A_us = to_scipy(assemble(inner(div(s_), v) * dx).petscmat)
    M_uu = to_scipy(assemble(inner(u, v) * dx).petscmat)

    Z = checkerboard_complement(mesh, N)
    A_su, A_us = A_su @ Z, Z.T @ A_us
    M_uu = Z.T @ M_uu @ Z
    A = sp.bmat([[A_ss, A_su], [A_us, sp.csr_matrix(nu * M_uu)]], format="csr")
    M = sp.bmat([[sp.csr_matrix(A_ss.shape), None],
                 [None, sp.csr_matrix(M_uu)]], format="csr")
    return to_petsc(A), to_petsc(M), (S, Q)


FEEC = ["A(CG)", "B(RT)", "B(BDM)"]
ALL_FORMS = FEEC + ["s(RT)", "s(BDM)", "s(P1-divP1)"]
H_FORMS = FEEC + ["s(P1-divP1)"]        # everything that exists at lowest order
TEX = {"A(CG)": r"$A(\mathcal{P}_r)$", "B(RT)": r"$B(\mathrm{RT}_r)$",
       "B(BDM)": r"$B(\mathrm{BDM}_r)$", "s(RT)": r"$\sigma(\mathrm{RT}_r)$",
       "s(BDM)": r"$\sigma(\mathrm{BDM}_r)$",
       "s(P1-divP1)": r"$\sigma(P_1\text{–div}P_1)$"}
TAG = {"A(CG)": "ACG", "B(RT)": "BRT", "B(BDM)": "BBDM",
       "s(RT)": "sRT", "s(BDM)": "sBDM", "s(P1-divP1)": "sP1"}


def nu_for(form, beta_sup, Rm):
    r"""The shift of the well-posedness argument, $\nu \ge 2\|\beta\|_\infty^2/\varepsilon + 2$.

    Only the two $\sigma$ rows carry it; everything else runs unshifted so that the
    returned eigenvalues are $\lambda$ itself.  For an unbounded field the bound is
    vacuous, and the *effective* magnitude on the mesh is used instead -- which is
    exactly the quantity the discretisation sees.
    """
    if not form.startswith("s(") or form == "s(P1-divP1)":
        return 0.0
    return 2.0 * beta_sup ** 2 * Rm + 2.0


def solve_form(form, mesh, N, r=1, beta=None, Rm=1.0, tau=None, nev=30,
               n_eigs=None, beta_sup=1.0, **meta):
    r"""Assemble and solve one configuration; ``tau`` defaults to $1.9\varepsilon$."""
    nu = nu_for(form, beta_sup, Rm)
    A, M, W = build_pencil(form, mesh, N, r, beta=beta, eps=Constant(1.0 / Rm), nu=nu)
    tau = (1.9 / Rm) if tau is None else tau
    res = solve_pencil(A, M, tau + nu, nev=nev, n_eigs=n_eigs, space=W,
                       form=form, r=r, Rm=Rm, **meta)
    if nu:
        # Undo the shift so every row of every table is directly comparable.
        res.values = res.values - nu
    return res


def exact_spectrum(n, Rm=1.0, kmax=40):
    r"""$\varepsilon(m^2+n^2)$, $m,n\ge1$: the Dirichlet Laplacian on $(0,\pi)^2$."""
    vals = sorted(m * m + k * k for m in range(1, kmax) for k in range(1, kmax))
    return np.array(vals[:n], dtype=float) / Rm


# --------------------------------------------------------------------------
# 2. The velocity fields
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Field:
    key: str
    tex: str
    regularity: str
    solenoidal: bool
    sup: Optional[float]
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
    """The L^p vortex; `delta` floors r_c so the expression stays finite if a
    quadrature point lands on the centre."""
    x, y = SpatialCoordinate(mesh)
    xc = yc = Constant(L / 2)
    r = sqrt((x - xc) ** 2 + (y - yc) ** 2 + Constant(delta))
    return as_vector([-(y - yc) / r ** alpha, (x - xc) / r ** alpha])


FIELDS = {
    "b6": Field("b6", r"$(\mathrm{sign}(x-\frac{\pi}{2}),\,0)$",
                r"$L^\infty\setminus W^{1,\infty}$", False, 1.0, _b6),
    "b7": Field("b7", r"$(\mathrm{sign}(y-\frac{\pi}{2}),\,0)$",
                r"$L^\infty\setminus W^{1,\infty}$", True, 1.0, _b7),
    "b8": Field("b8", r"$(\mathrm{sign}(x-\frac{\pi}{2}),\,\mathrm{sign}(x-\frac{\pi}{2}))$",
                r"$L^\infty\setminus W^{1,\infty}$", False, np.sqrt(2.0), _b8),
    "b9": Field("b9", r"$(-\hat y/r^{\alpha},\ \hat x/r^{\alpha})$", r"$L^p$",
                True, None, _b9),
}
ROUGH = ["b6", "b7", "b8"]


def effective_sup(mesh, beta):
    """The largest cell-average of |beta|: what the discretisation actually sees."""
    mag = Function(FunctionSpace(mesh, "DG", 0)).interpolate(sqrt(dot(beta, beta)))
    return float(np.abs(mag.dat.data_ro).max())


# --------------------------------------------------------------------------
# 3. Fields in $L^\infty\setminus W^{1,\infty}$
# --------------------------------------------------------------------------

N_REF, R_REF, NEV_REF = 24, 3, 25

mesh_ref = crisscross_mesh(N_REF)
ref_runs, REFERENCE = {}, {}
for key in ROUGH:
    beta_ref = FIELDS[key](mesh_ref)
    for form in FEEC + ["s(RT)", "s(BDM)"]:
        ref_runs[(key, form)] = solve_form(form, mesh_ref, N_REF, R_REF,
                                           beta=beta_ref, nev=NEV_REF, n_eigs=5,
                                           beta_sup=FIELDS[key].sup, field=key)
    stack = np.array([ref_runs[(key, f)].values[:5] for f in FEEC])
    REFERENCE[key] = stack.mean(axis=0)

rows = []
for key in ROUGH:
    stack = np.array([ref_runs[(key, f)].values[:5] for f in FEEC])
    sprd = spread(stack)
    for j in range(5):
        rows.append(dict(field=key, j=j + 1,
                         **{TEX["A(CG)"]: stack[0, j], TEX["B(RT)"]: stack[1, j],
                            TEX["B(BDM)"]: stack[2, j], "spread": sprd[j],
                            "digits": agreeing_digits(stack[:, j])}))
show(pd.DataFrame(rows).set_index(["field", "j"]), default="{:.9f}",
     rules=[("spread", "{:.2e}"), ("digits", "{:.0f}")],
     caption=rf"Cross-validation of the first five eigenvalues over the FEEC "
             rf"formulations, $R_m = 1$, $N = {N_REF}$, $r = {R_REF}$")

print("is the well-posedness shift still spectrally inert for a rough field?\n")
for key in ROUGH:
    nu = nu_for("s(RT)", FIELDS[key].sup, 1.0)
    d_rt = np.abs(ref_runs[(key, "B(RT)")].values[:5]
                  - ref_runs[(key, "s(RT)")].values[:5]).max()
    d_bdm = np.abs(ref_runs[(key, "B(BDM)")].values[:5]
                   - ref_runs[(key, "s(BDM)")].values[:5]).max()
    print(f"  {key}: nu = {nu:5.1f}   max |B(RT) - s(RT)| = {d_rt:.2e}"
          f"   max |B(BDM) - s(BDM)| = {d_bdm:.2e}")

print("\nconditioning of the reference computations")
for key in ROUGH:
    print(f"  {key}   " + "  ".join(f"{f}: {ref_runs[(key, f)].kappa1:.1e}" for f in FEEC))


# --------------------------------------------------------------------------
# 3.2 $h$-refinement
# --------------------------------------------------------------------------

H_LEVELS = (4, 8, 16, 24, 32)
EIG_COLOURS = ["C0", "C1", "C2", "C3", "C4"]
EIG_COLS = [rf"$\lambda_{{{j+1}}}$" for j in range(5)]
REFERENCE_LP = {}


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
        sup = FIELDS[field_key].sup or effective_sup(mesh, beta)
        res = solve_form(form, mesh, N, r, beta=beta, Rm=Rm, nev=30,
                         beta_sup=sup, field=field_key)
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


h_tables = {(f, "b7"): h_study(f, "b7") for f in H_FORMS}
for f in H_FORMS:
    display(show(rate_frame(h_tables[(f, "b7")], EIG_COLS), default="{:.3e}",
                 rules=[("rate", "{:.2f}"), ("h", "{:.5f}"), (r"$\kappa_1$", "{:.2e}")],
                 caption=rf"$h$-refinement, **{f}**, field `b7`, $r = 1$, $R_m = 1$"))

df = h_tables[("A(CG)", "b7")]
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

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$A(\mathcal{P}_1)$, field b7, $r=1$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_h_b7_ACG.pdf")
plt.show()

df = h_tables[("B(RT)", "b7")]
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

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathrm{RT}_1)$, field b7, $r=1$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_h_b7_BRT.pdf")
plt.show()

df = h_tables[("B(BDM)", "b7")]
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

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathrm{BDM}_1)$, field b7, $r=1$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_h_b7_BBDM.pdf")
plt.show()

df = h_tables[("s(P1-divP1)", "b7")]
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

# Reference slope, anchored a factor 0.30 below the lowest curve so that it sits
# beside the data rather than under it.
ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$\sigma(P_1\text{–div}P_1)$, field b7, $r=1$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_h_b7_sP1.pdf")
plt.show()


# --------------------------------------------------------------------------
# 3.3 $p$-refinement
# --------------------------------------------------------------------------

P_LEVELS = (1, 2, 3, 4)
N_PREF = 8


def p_study(form, field_key, levels=P_LEVELS, N=N_PREF, Rm=1.0, alpha=None):
    ref = reference_for(field_key, alpha)
    mesh = crisscross_mesh(N)
    beta = (FIELDS[field_key](mesh) if alpha is None
            else FIELDS[field_key](mesh, alpha=alpha))
    sup = FIELDS[field_key].sup or effective_sup(mesh, beta)
    rows = []
    for r in levels:
        res = solve_form(form, mesh, N, r, beta=beta, Rm=Rm, nev=30,
                         beta_sup=sup, field=field_key)
        got = assign(ref, res.values)
        rows.append(dict(r=r, dof=res.size,
                         **{EIG_COLS[j]: abs(got[j] - ref[j]) / abs(ref[j])
                            for j in range(5)},
                         **{r"$\kappa_1$": res.kappa1}))
    return pd.DataFrame(rows).set_index("r")


p_tables = {(f, "b7"): p_study(f, "b7") for f in FEEC}
for f in FEEC:
    display(show(p_tables[(f, "b7")], default="{:.3e}",
                 rules=[("dof", "{:.0f}"), (r"$\kappa_1$", "{:.2e}")],
                 caption=rf"$p$-refinement, **{f}**, field `b7`, $N = {N_PREF}$, $R_m = 1$"))

df = p_tables[("A(CG)", "b7")]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$A(\mathcal{P}_r)$, field b7, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_p_b7_ACG.pdf")
plt.show()

df = p_tables[("B(RT)", "b7")]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathrm{RT}_r)$, field b7, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_p_b7_BRT.pdf")
plt.show()

df = p_tables[("B(BDM)", "b7")]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathrm{BDM}_r)$, field b7, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_p_b7_BBDM.pdf")
plt.show()


# --------------------------------------------------------------------------
# 3.4 The other fields
# --------------------------------------------------------------------------

rows = []
for key in ROUGH:
    for form in H_FORMS:
        if (form, key) not in h_tables:
            h_tables[(form, key)] = h_study(form, key)
        df = h_tables[(form, key)]
        row = {"field": key, "formulation": form}
        for col in EIG_COLS:
            row[f"rate {col}"] = rates(df["h"], df[col])[-1]
        row[r"$\kappa_1$ (finest)"] = df[r"$\kappa_1$"].iloc[-1]
        rows.append(row)
show(pd.DataFrame(rows).set_index(["field", "formulation"]), default="{:.2f}",
     rules=[(r"$\kappa_1$ (finest)", "{:.2e}")],
     caption=r"Observed $h$-refinement orders at the finest level, all three "
             r"$L^\infty\setminus W^{1,\infty}$ fields and all four lowest-order "
             r"discretisations ($r = 1$, $R_m = 1$)")


# --------------------------------------------------------------------------
# 3.5 Does the spurious branch survive the loss of regularity?
# --------------------------------------------------------------------------

N_SP = 24
mesh_sp = crisscross_mesh(N_SP)


def spurious_set(field_key, Rm=1.0, alpha=None, n_test=25, rtol=3e-3):
    """Values in the polluting spectrum with no counterpart in B(BDM) on the same mesh."""
    beta = (None if field_key == "b0" else
            (FIELDS[field_key](mesh_sp) if alpha is None
             else FIELDS[field_key](mesh_sp, alpha=alpha)))
    sup = 1.0 if beta is None else (FIELDS[field_key].sup
                                    or effective_sup(mesh_sp, beta))
    ref = solve_form("B(BDM)", mesh_sp, N_SP, 1, beta=beta, Rm=Rm, nev=45,
                     beta_sup=sup, field=field_key)
    bad = solve_form("s(P1-divP1)", mesh_sp, N_SP, 1, beta=beta, Rm=Rm, nev=45,
                     beta_sup=sup, field=field_key)
    extra = np.array([z for z in bad.values[:n_test]
                      if np.min(np.abs(ref.values - z)) > rtol * max(1.0, abs(z))])
    return ref, bad, extra


rows = []
for key in ["b0"] + ROUGH + ["b9"]:
    alpha = 1.5 if key == "b9" else None
    ref, bad, extra = spurious_set(key, alpha=alpha)
    rows.append(dict(
        field=key, **{
            r"genuine $\lambda_1$": ref.values[0],
            r"genuine $\lambda_2$": ref.values[1],
            r"1st spurious": extra[0] if len(extra) else None,
            r"2nd spurious": extra[1] if len(extra) > 1 else None,
            r"$3\lambda_1$": 3 * ref.values[0],
            "n spurious in 25": len(extra),
            r"$\kappa_1$": bad.kappa1}))
show(pd.DataFrame(rows).set_index("field"), default="{:.4f}",
     rules=[("n spurious", "{:.0f}"), (r"$\kappa_1$", "{:.2e}")],
     caption=rf"The shadow spectrum under each field, $R_m = 1$, $N = {N_SP}$, $r = 1$ "
             rf"(`b0` is the diffusive limit, `b9` has $\alpha = 1.5$)")

print("imaginary parts: does a spurious mode inherit the genuine one's?\n")
for key in ROUGH + ["b9"]:
    alpha = 1.5 if key == "b9" else None
    ref, bad, extra = spurious_set(key, alpha=alpha)
    cx = [z for z in extra if abs(z.imag) > 1e-8]
    if not cx:
        print(f"  {key}: no complex spurious value in the window")
        continue
    z = cx[0]
    partner = ref.values[int(np.argmin(np.abs(ref.values.imag - z.imag)))]
    print(f"  {key}: spurious {z.real:8.4f}{z.imag:+8.4f}i   "
          f"nearest genuine Im: {partner.real:8.4f}{partner.imag:+8.4f}i   "
          f"|Im ratio| = {abs(z.imag / partner.imag):.3f}")


# --------------------------------------------------------------------------
# 3.6 An expanded spectral window
# --------------------------------------------------------------------------

PS_N, PS_NEV = 20, 80
PS_WINDOW = ((0.0, 32.0), (-6.0, 6.0))
mesh_ps = crisscross_mesh(PS_N)


def run_ps(form, field_key, Rm=1.0, alpha=None, window=PS_WINDOW, nev=PS_NEV, r=1):
    """Assemble one pencil and hand it to the pseudospectrum engine."""
    beta = (None if field_key == "b0" else
            (FIELDS[field_key](mesh_ps) if alpha is None
             else FIELDS[field_key](mesh_ps, alpha=alpha)))
    sup = 1.0 if beta is None else (FIELDS[field_key].sup
                                    or effective_sup(mesh_ps, beta))
    # nu = 0 here so the Ritz values are lambda itself; the shift is inert (3.1).
    A, M, W = build_pencil(form, mesh_ps, PS_N, r, beta=beta,
                           eps=Constant(1.0 / Rm), nu=0.0)
    kap = condition_estimates(A, M, 1.9 / Rm)
    out = ps.pseudospectrum(A, M, 1.9 / Rm, nev=nev, window=window,
                            form=form, beta=field_key, Rm=Rm, alpha=alpha,
                            label=FIELDS[field_key].tex if field_key != "b0" else r"$\beta=0$")
    out["kappa1"] = kap[r"Hager $\kappa_1$"]
    return out


ps_wide = {f: run_ps(f, "b7") for f in ["B(BDM)", "s(P1-divP1)"]}
show(ps.diagnostics_frame(ps_wide.values(), keys=("form", "beta")), default="{:.3e}",
     rules=[("N", "{:.0f}"), ("m", "{:.0f}"), ("eigs in window", "{:.0f}")],
     caption=rf"Projection diagnostics for the wide-window scan, field `b7`, "
             rf"$N = {PS_N}$, nev $= {PS_NEV}$")

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    r_ = ps_wide["B(BDM)"]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathrm{BDM}_1)$, field b7, $R_m=1$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "bregT_ps_wide_b7_BBDM.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    r_ = ps_wide["s(P1-divP1)"]
    ax = ps.plot_pseudospectrum(r_, ax=ax,
                                title=r"$\sigma(P_1$–div$P_1)$, field b7, $R_m=1$"
                                      + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    for value in (6.0, 15.0, 24.0):
        ax.axvline(value, color="w", lw=0.8, ls=":", alpha=0.8)
    ax.text(6.0, 5.0, r" shadow spectrum", color="w", fontsize=7, va="top")
    fig.savefig(FIGDIR / "bregT_ps_wide_b7_sP1.pdf")
    plt.show()


# --------------------------------------------------------------------------
# 3.7 Convergence rates as $R_m$ varies
# --------------------------------------------------------------------------

RM_LEVELS = (16, 24, 32, 40)
RM_VALUES = (1.0, 2.0, 5.0, 10.0)

print("cell Peclet at the coarsest level used:")
for Rm in RM_VALUES:
    print(f"  Rm = {Rm:5.1f}:  {(np.pi / RM_LEVELS[0]) * FIELDS['b7'].sup * Rm / 2:.2f}")

REFERENCE_RM = {}
for Rm in RM_VALUES:
    beta_r = FIELDS["b7"](mesh_ref)
    stack = np.array([solve_form(f, mesh_ref, N_REF, R_REF, beta=beta_r, Rm=Rm,
                                 nev=NEV_REF, n_eigs=5, beta_sup=1.0,
                                 field="b7").values[:5] for f in FEEC])
    REFERENCE_RM[Rm] = stack.mean(axis=0)
    print(f"  Rm = {Rm:5.1f}: reference lambda_1..3 = {np.round(stack.mean(axis=0)[:3], 6)}"
          f"   spread {spread(stack).max():.1e}")

rows = []
for Rm in RM_VALUES:
    ref = REFERENCE_RM[Rm]
    for form in H_FORMS:
        errs, kap = [], []
        for N in RM_LEVELS:
            mesh = crisscross_mesh(N)
            res = solve_form(form, mesh, N, 1, beta=FIELDS["b7"](mesh), Rm=Rm,
                             nev=30, beta_sup=1.0, field="b7")
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
     caption=r"Observed $h$-refinement orders for the first five eigenvalues as $R_m$ "
             r"varies, field `b7`, $r = 1$")


# --------------------------------------------------------------------------
# 4. Fields in $L^p$
# --------------------------------------------------------------------------

ALPHAS = (1.0, 1.25, 1.5, 1.75)
ALPHA_MAIN = 1.5

print("effective magnitude of the vortex, by mesh and exponent")
print(f"{'N':>5s}" + "".join(f"{'a=' + format(a, '.2f'):>12s}" for a in ALPHAS))
for N in (8, 16, 32):
    mesh_e = crisscross_mesh(N)
    line = f"{N:5d}"
    for a in ALPHAS:
        line += f"{effective_sup(mesh_e, FIELDS['b9'](mesh_e, alpha=a)):12.2f}"
    print(line)

mesh_ref_lp = crisscross_mesh(N_REF)
ref_runs_lp = {}
for a in ALPHAS:
    beta_a = FIELDS["b9"](mesh_ref_lp, alpha=a)
    sup_a = effective_sup(mesh_ref_lp, beta_a)
    for form in FEEC:
        ref_runs_lp[(a, form)] = solve_form(form, mesh_ref_lp, N_REF, R_REF,
                                            beta=beta_a, nev=NEV_REF, n_eigs=5,
                                            beta_sup=sup_a, field="b9", alpha=a)
    stack = np.array([ref_runs_lp[(a, f)].values[:5] for f in FEEC])
    REFERENCE_LP[("b9", a)] = stack.mean(axis=0)

rows = []
for a in ALPHAS:
    stack = np.array([ref_runs_lp[(a, f)].values[:5] for f in FEEC])
    rows.append(dict(alpha=a,
                     **{r"$p$ limit": (2.0 / (a - 1.0)) if a > 1.0 else np.inf,
                        r"$\lambda_1$": stack.mean(axis=0)[0],
                        r"$\lambda_2$": stack.mean(axis=0)[1],
                        "max spread": spread(stack).max(),
                        "min digits": min(agreeing_digits(stack[:, j])
                                          for j in range(5)),
                        r"$\kappa_1$ B(RT)": ref_runs_lp[(a, "B(RT)")].kappa1}))
show(pd.DataFrame(rows).set_index("alpha"), default="{:.6f}",
     rules=[("alpha", "{:.2f}"), (r"$p$ limit", "{:.2f}"), ("max spread", "{:.2e}"),
            ("min digits", "{:.0f}"), (r"$\kappa_1$ B(RT)", "{:.2e}")],
     caption=rf"Cross-validation for the $L^p$ vortex over the FEEC formulations, "
             rf"$R_m = 1$, $N = {N_REF}$, $r = {R_REF}$")


# --------------------------------------------------------------------------
# 4.2 The shadow spectrum against the singularity
# --------------------------------------------------------------------------

rows = []
for a in ALPHAS:
    ref, bad, extra = spurious_set("b9", alpha=a)
    rows.append(dict(alpha=a,
                     **{r"genuine $\lambda_1$": ref.values[0],
                        r"1st spurious": extra[0] if len(extra) else None,
                        r"2nd spurious": extra[1] if len(extra) > 1 else None,
                        r"$3\lambda_1$": 3 * ref.values[0],
                        "n spurious in 25": len(extra),
                        r"$\kappa_1$": bad.kappa1}))
show(pd.DataFrame(rows).set_index("alpha"), default="{:.4f}",
     rules=[("alpha", "{:.2f}"), ("n spurious", "{:.0f}"), (r"$\kappa_1$", "{:.2e}")],
     caption=rf"The shadow spectrum against the strength of the singularity, "
             rf"$R_m = 1$, $N = {N_SP}$, $r = 1$")


# --------------------------------------------------------------------------
# 4.3 $h$- and $p$-refinement
# --------------------------------------------------------------------------

h_tables_lp = {f: h_study(f, "b9", alpha=ALPHA_MAIN) for f in H_FORMS}
for f in H_FORMS:
    display(show(rate_frame(h_tables_lp[f], EIG_COLS), default="{:.3e}",
                 rules=[("rate", "{:.2f}"), ("h", "{:.5f}"), (r"$\kappa_1$", "{:.2e}")],
                 caption=rf"$h$-refinement, **{f}**, vortex $\alpha = {ALPHA_MAIN}$, "
                         rf"$r = 1$, $R_m = 1$"))

df = h_tables_lp["A(CG)"]
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

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$A(\mathcal{P}_1)$, vortex $\alpha=1.5$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_h_b9_ACG.pdf")
plt.show()

df = h_tables_lp["B(RT)"]
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

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathrm{RT}_1)$, vortex $\alpha=1.5$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_h_b9_BRT.pdf")
plt.show()

df = h_tables_lp["B(BDM)"]
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

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathrm{BDM}_1)$, vortex $\alpha=1.5$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_h_b9_BBDM.pdf")
plt.show()

df = h_tables_lp["s(P1-divP1)"]
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

ref = 0.30 * min(e1[0], e2[0], e3[0], e4[0], e5[0])
ax.loglog(h, ref * (h / h[0]) ** 2, "k--", lw=1.1, label=r"$O(h^2)$")

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$\sigma(P_1\text{–div}P_1)$, vortex $\alpha=1.5$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_h_b9_sP1.pdf")
plt.show()

p_tables_lp = {f: p_study(f, "b9", alpha=ALPHA_MAIN) for f in FEEC}
for f in FEEC:
    display(show(p_tables_lp[f], default="{:.3e}",
                 rules=[("dof", "{:.0f}"), (r"$\kappa_1$", "{:.2e}")],
                 caption=rf"$p$-refinement, **{f}**, vortex $\alpha = {ALPHA_MAIN}$, "
                         rf"$N = {N_PREF}$, $R_m = 1$"))

df = p_tables_lp["A(CG)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$A(\mathcal{P}_r)$, vortex $\alpha=1.5$, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_p_b9_ACG.pdf")
plt.show()

df = p_tables_lp["B(RT)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathrm{RT}_r)$, vortex $\alpha=1.5$, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_p_b9_BRT.pdf")
plt.show()

df = p_tables_lp["B(BDM)"]

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax.semilogy(df.index, df[EIG_COLS[0]], "o-", color=EIG_COLOURS[0], ms=4.5, lw=1.2, label=r"$\lambda_{1}$")
ax.semilogy(df.index, df[EIG_COLS[1]], "o-", color=EIG_COLOURS[1], ms=4.5, lw=1.2, label=r"$\lambda_{2}$")
ax.semilogy(df.index, df[EIG_COLS[2]], "o-", color=EIG_COLOURS[2], ms=4.5, lw=1.2, label=r"$\lambda_{3}$")
ax.semilogy(df.index, df[EIG_COLS[3]], "o-", color=EIG_COLOURS[3], ms=4.5, lw=1.2, label=r"$\lambda_{4}$")
ax.semilogy(df.index, df[EIG_COLS[4]], "o-", color=EIG_COLOURS[4], ms=4.5, lw=1.2, label=r"$\lambda_{5}$")
ax.set_xlabel(r"polynomial degree $r$")
ax.set_xticks(list(P_LEVELS))
ax.set_ylabel(r"relative error in $\lambda_j$")
ax.set_title(r"$B(\mathrm{BDM}_r)$, vortex $\alpha=1.5$, $N=8$"
             + rf"   ($\kappa_1 \leq {df[r'$\kappa_1$'].max():.1e}$)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.savefig(FIGDIR / "bregT_p_b9_BBDM.pdf")
plt.show()


# --------------------------------------------------------------------------
# 4.4 Convergence as $R_m$ varies
# --------------------------------------------------------------------------

RM_VALUES_LP = (1.0, 2.0, 5.0)
REFERENCE_RM_LP = {}
for Rm in RM_VALUES_LP:
    beta_a = FIELDS["b9"](mesh_ref_lp, alpha=ALPHA_MAIN)
    sup_a = effective_sup(mesh_ref_lp, beta_a)
    stack = np.array([solve_form(f, mesh_ref_lp, N_REF, R_REF, beta=beta_a, Rm=Rm,
                                 nev=NEV_REF, n_eigs=5, beta_sup=sup_a,
                                 field="b9").values[:5] for f in FEEC])
    REFERENCE_RM_LP[Rm] = stack.mean(axis=0)

rows = []
for Rm in RM_VALUES_LP:
    ref = REFERENCE_RM_LP[Rm]
    for form in H_FORMS:
        errs, kap = [], []
        for N in RM_LEVELS:
            mesh = crisscross_mesh(N)
            beta = FIELDS["b9"](mesh, alpha=ALPHA_MAIN)
            res = solve_form(form, mesh, N, 1, beta=beta, Rm=Rm, nev=30,
                             beta_sup=effective_sup(mesh, beta), field="b9")
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
# 4.5 Pseudospectra, and non-normality against the singularity
# --------------------------------------------------------------------------

PS_WINDOW_LP = ((0.0, 16.0), (-6.0, 6.0))


def bulge(result, q=2.0):
    """max{ dist(z, spectrum) : sigma_min(z) <= eps } - eps, eps the q-th percentile."""
    eps_level = float(np.percentile(result["sigma"], q))
    inside = result["sigma"] <= eps_level
    if not inside.any() or result["ritz"].size == 0:
        return np.nan, eps_level
    d = np.min(np.abs(result["z"][inside][:, None] - result["ritz"][None, :]), axis=1)
    return float(d.max() - eps_level), eps_level


RM_HIGH = 5.0
rows = []
ps_alpha = {}
for a in ALPHAS:
    for form in ["A(CG)", "B(RT)", "B(BDM)", "s(P1-divP1)"]:
        r_ = run_ps(form, "b9", Rm=RM_HIGH, alpha=a, window=PS_WINDOW_LP)
        ps_alpha[(form, a)] = r_
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
             rf"$R_m = {RM_HIGH:g}$, $N = {PS_N}$, $r = 1$")

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    r_ = ps_alpha[("B(BDM)", 1.0)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathrm{BDM}_1)$, vortex $\alpha=1.0$, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "bregT_ps_b9_BBDM_a100.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    r_ = ps_alpha[("B(BDM)", 1.75)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$B(\mathrm{BDM}_1)$, vortex $\alpha=1.75$, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "bregT_ps_b9_BBDM_a175.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    r_ = ps_alpha[("s(P1-divP1)", 1.75)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$\sigma(P_1$–div$P_1)$, vortex $\alpha=1.75$, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "bregT_ps_b9_sP1_a175.pdf")
    plt.show()

with matplotlib.rc_context(PS_RC):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    r_ = ps_alpha[("A(CG)", 1.75)]
    ps.plot_pseudospectrum(r_, ax=ax,
                           title=r"$A(\mathcal{P}_1)$, vortex $\alpha=1.75$, $R_m=5$"
                                 + rf"   ($\kappa_1 = {r_['kappa1']:.1e}$)")
    fig.savefig(FIGDIR / "bregT_ps_b9_ACG_a175.pdf")
    plt.show()


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
for q in sorted(FIGDIR.glob("bregT_*.pdf")):
    print("  ", q.name)

