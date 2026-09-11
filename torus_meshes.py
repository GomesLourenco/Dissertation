r"""Geometry for the full-Hodge study: a flat torus and a curved one.

Two closed surfaces with **the same topology** ($b_0=1$, $b_1=2$, $b_2=1$; no
boundary, so every trace condition in the formulation is vacuous) and different
**geometry**:

* :func:`flat_torus` -- $(\mathbb{R}/L\mathbb{Z})^2$, intrinsically flat, Gauss
  curvature $K \equiv 0$, and with a spectrum known in closed form;
* :func:`curved_torus` -- the torus of revolution of radii $R > r$ embedded in
  $\mathbb{R}^3$, with $K$ of both signs, carried isoparametrically so that the
  geometric error does not masquerade as a curvature effect.

Because the geometry is the only thing that differs, anything the two disagree
about is geometric by construction.  Three closed-form quantities make that
measurable without a reference computation, and they are what
:func:`gauss_curvature`, :data:`HARMONIC_SHARE` and :data:`HARMONIC_RATIO` are
for; see their docstrings.

Companion modules: ``spectral_common`` (solve, conditioning, tables) and
``pseudospectra_partial_schur``.
"""
from __future__ import annotations

import numpy as np

from firedrake import (PeriodicRectangleMesh, TorusMesh, Mesh, Function,
                       VectorFunctionSpace, FunctionSpace, SpatialCoordinate,
                       CellNormal, Constant, as_vector, cross, sqrt)

__all__ = [
    "L_FLAT", "R_MAJOR", "R_MINOR",
    "flat_torus", "curved_torus", "snap_to_torus",
    "perp_of", "gauss_curvature", "toroidal", "poloidal",
    "flat_hodge_spectrum", "harmonic_share", "harmonic_ratio",
    "betti_numbers",
]

#: Side of the flat torus.  $2\pi$ makes the wavenumbers integers.
L_FLAT = 2.0 * np.pi

#: Radii of the torus of revolution.  $R > r > 0$ keeps it embedded.
R_MAJOR, R_MINOR = 2.0, 1.0

#: Betti numbers, the same for both surfaces.  $b_1$ is the one that matters:
#: it is the dimension of the space of harmonic $1$-forms, so it is the exact
#: multiplicity of $\lambda = 0$ in the full-Hodge problem.
betti_numbers = (1, 2, 1)


# ------------------------------------------------------------------- meshes
def flat_torus(n, L=L_FLAT):
    r"""$(\mathbb{R}/L\mathbb{Z})^2$ as a periodic criss-cross triangulation.

    Firedrake builds this with a discontinuous coordinate field, so the mesh is
    the flat square with its sides identified: geometrically flat, topologically
    a torus.  Criss-cross for the same reason as everywhere else in this study --
    the mesh is then symmetric under all eight symmetries of the square and
    cannot distinguish a direction that the wind is about to.
    """
    return PeriodicRectangleMesh(n, n, L, L, direction="both",
                                 quadrilateral=False, diagonal="crossed")


def snap_to_torus(X, R=R_MAJOR, r=R_MINOR):
    """Nearest point on the torus of revolution, for an array of points.

    The nearest point on the core circle is $R(x,y,0)/\\rho$ with
    $\\rho=\\sqrt{x^2+y^2}$; the surface point is then that plus $r$ times the
    unit offset.  Used to place the nodes of a higher-degree coordinate field
    exactly on the surface.
    """
    X = np.asarray(X, float)
    rho = np.linalg.norm(X[:, :2], axis=1)
    core = np.zeros_like(X)
    core[:, 0] = R * X[:, 0] / rho
    core[:, 1] = R * X[:, 1] / rho
    off = X - core
    return core + r * off / np.linalg.norm(off, axis=1)[:, None]


def curved_torus(nR, nr, degree=2, R=R_MAJOR, r=R_MINOR):
    r"""The torus of revolution, immersed in $\mathbb{R}^3$, rim carried to $h^{p+1}$.

    A mesh generator returns a polyhedron with its *vertices* on the surface and
    its faces flat, so the metric it carries is wrong by $\mathcal{O}(h^2)$ --
    the same order as the $P_1$ discretisation error, and of a fixed sign.  On a
    problem whose whole subject is the difference geometry makes, that is not a
    detail: at ``degree=1`` the meshed area is $1.8\%$ short and
    $\int_\Omega K\,\mathrm{d}A$, which Gauss--Bonnet says is exactly zero, comes
    out at $-0.11$.  Raising the coordinate field to degree $2$ and putting all
    of its nodes on the surface takes both to $10^{-4}$.  Pass ``degree=1`` to
    reproduce the artefact.

    ``init_cell_orientations`` is required before any form involving
    :func:`~firedrake.ufl_expr.CellNormal` can be assembled; the expression
    passed is the outward direction, away from the core circle.
    """
    mesh = TorusMesh(nR, nr, R, r, quadrilateral=False)
    if degree > 1:
        Vc = VectorFunctionSpace(mesh, "CG", degree, dim=3)
        coords = Function(Vc).interpolate(mesh.coordinates)
        coords.dat.data[:] = snap_to_torus(coords.dat.data_ro, R, r)
        mesh = Mesh(coords)
    X = SpatialCoordinate(mesh)
    rho = sqrt(X[0] ** 2 + X[1] ** 2)
    mesh.init_cell_orientations(
        as_vector([X[0] * (1 - R / rho), X[1] * (1 - R / rho), X[2]]))
    return mesh


# --------------------------------------------------------------- geometry
def perp_of(mesh):
    r"""Rotation by $+\pi/2$ in the tangent plane, as a callable.

    This is the one place the two geometries need different code.  In the plane
    it is $(a_1,a_2)\mapsto(-a_2,a_1)$; on a surface it is $a \mapsto n\times a$
    with $n$ the cell normal, which is the same rotation and reduces to it when
    $n = e_3$.  Everything else in the study is written once and runs on both.
    """
    if mesh.geometric_dimension == 3:
        normal = CellNormal(mesh)
        return lambda a: cross(normal, a)
    return lambda a: as_vector([-a[1], a[0]])


def gauss_curvature(mesh, R=R_MAJOR, r=R_MINOR):
    r"""$K$ as a UFL expression in the ambient coordinates.

    On the torus of revolution $K = \cos\varphi / (r(R + r\cos\varphi))$, and
    since $\rho = \sqrt{x^2+y^2} = R + r\cos\varphi$ this is

    .. math:: K = \frac{\rho - R}{r^2\,\rho},

    positive on the outer half, negative on the inner half, with
    $K \in [-1/(r(R-r)),\ 1/(r(R+r))]$ and $\int_\Omega K\,\mathrm{d}A = 0$ by
    Gauss--Bonnet, $\chi(T^2) = 0$.  That integral is a free check on the mesh
    geometry, and it is the one :func:`curved_torus` is graded against.

    On the flat torus it returns the constant $0$: not an approximation, the
    exact answer, which is what makes the pair a controlled comparison.
    """
    if mesh.geometric_dimension == 2:
        return Constant(0.0)
    X = SpatialCoordinate(mesh)
    rho = sqrt(X[0] ** 2 + X[1] ** 2)
    return (rho - Constant(R)) / (Constant(r ** 2) * rho)


def toroidal(mesh):
    r"""$\partial_\theta$: the wind that goes the long way round.

    On **both** surfaces this generates a one-parameter group of isometries -- a
    translation on the flat torus, a rotation about the axis on the curved one --
    so it is a **Killing field** on both.  For a Killing $\beta$ the Lie
    derivative is skew-adjoint and commutes with the Hodge Laplacian, so
    $\varepsilon\Delta + \mathcal{L}_\beta$ is normal and the advection can move
    an eigenvalue only along the imaginary axis.  That is the control.
    """
    if mesh.geometric_dimension == 2:
        return as_vector([Constant(1.0), Constant(0.0)])
    X = SpatialCoordinate(mesh)
    return as_vector([-X[1], X[0], Constant(0.0)])


def poloidal(mesh, R=R_MAJOR):
    r"""$\partial_\varphi$: the wind that goes the short way round the tube.

    Written in the *same* parametrisation as :func:`toroidal` and of constant
    length $r$ on the curved surface.  On the flat torus it is again a
    translation, hence Killing.  On the curved one it is **not**: its flow
    carries a $\theta$-circle of circumference $2\pi(R+r\cos\varphi)$ onto one of
    a different circumference, so it does not preserve the metric.

    The pair (:func:`toroidal`, :func:`poloidal`) is therefore the experiment:
    the same two winds, the same topology, and the Killing property of one of
    them is destroyed by curvature alone.
    """
    if mesh.geometric_dimension == 2:
        return as_vector([Constant(0.0), Constant(1.0)])
    X = SpatialCoordinate(mesh)
    rho = sqrt(X[0] ** 2 + X[1] ** 2)
    return as_vector([-X[2] * X[0] / rho, -X[2] * X[1] / rho, rho - Constant(R)])


# ------------------------------------------------------- exact references
def flat_hodge_spectrum(n_eigs=20, L=L_FLAT, eps=1.0, beta=(0.0, 0.0), kmax=8):
    r"""The exact full-Hodge spectrum of the flat torus, with advection.

    On $(\mathbb{R}/L\mathbb{Z})^2$ the frame $\mathrm{d}x,\mathrm{d}y$ is
    parallel, so the Hodge Laplacian on $1$-forms acts componentwise as the
    scalar Laplacian, and for a **constant** $\beta$ the Lie derivative is the
    directional derivative.  Each lattice mode $\mathrm{e}^{i(m x + n y)}$ with
    $(m,n) \in (2\pi/L)\mathbb{Z}^2$ therefore contributes

    .. math:: \lambda = \varepsilon(m^2+n^2) + i(m\beta_1 + n\beta_2),

    **twice** -- once for each of the two constant $1$-forms.  Note that
    $(m,n)=(0,0)$ is included and gives $\lambda = 0$ with multiplicity $2$,
    which is $b_1$: the harmonic forms fall out of the same formula rather than
    being a separate case.

    Returns the ``n_eigs`` smallest by $(\mathrm{Re}, \mathrm{Im})$, with
    multiplicity.
    """
    k = 2.0 * np.pi / L
    vals = []
    for m in range(-kmax, kmax + 1):
        for j in range(-kmax, kmax + 1):
            lam = (eps * ((k * m) ** 2 + (k * j) ** 2)
                   + 1j * (k * m * beta[0] + k * j * beta[1]))
            vals.extend([lam, lam])
    vals = np.array(vals)
    return vals[np.lexsort((vals.imag, vals.real))][:n_eigs]


def harmonic_share(R=R_MAJOR, r=R_MINOR):
    r"""$\int K|h|^2 / \|h\|^2$ for **every** harmonic $1$-form: $-1/(R^2-r^2)$.

    The Weitzenböck identity on a surface reads $\Delta = \nabla^*\nabla + K$,
    the curvature term for $1$-forms being multiplication by the Gauss
    curvature.  A harmonic form has $\Delta h = 0$, so

    .. math:: 0 = \|\nabla h\|^2 + \int_\Omega K|h|^2 ,

    and the curvature share is *minus* the rough-Laplacian energy: strictly
    negative unless $h$ is parallel, which on a closed surface forces $K\equiv0$.
    On the torus of revolution $h = \mathrm{d}\theta$ is harmonic with
    $|h|^2 = \rho^{-2}$, and the two integrals evaluate in closed form to give
    $-1/(R^2-r^2)$ exactly.

    The value is basis-independent: $\star$ is a pointwise isometry with
    $h \perp \star h$, so every unit vector of the two-dimensional harmonic space
    has the same $|h|^2$ profile and the same share.  That matters in practice,
    because an eigensolver returns an arbitrary basis of a degenerate eigenspace.
    """
    return -1.0 / (R ** 2 - r ** 2)


def harmonic_ratio(R=R_MAJOR, r=R_MINOR):
    r"""$\max|h| / \min|h| = (R+r)/(R-r)$ for a harmonic $1$-form.

    From $|h|^2 = \rho^{-2}$ with $\rho \in [R-r, R+r]$: the harmonic form is
    largest on the **inner** rim, where the curvature is most negative, and its
    non-constancy is exactly its failure to be parallel.  On the flat torus the
    harmonic forms are the constant ones and the ratio is $1$.
    """
    return (R + r) / (R - r)
