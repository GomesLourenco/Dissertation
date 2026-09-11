r"""Geometry for the domain-regularity study: the L-shape and the re-entrant sector.

Where ``spectral_common`` is form-degree agnostic and knows nothing about the
domain, this module is domain-specific and knows nothing about the formulation.
It provides

* :func:`lshape_mesh` -- structured criss-cross triangulations of
  $\Omega_L = (-1,1)^2\setminus([0,1]\times(-1,0])$, uniform or **graded**
  towards the re-entrant corner by a single parameter $\gamma$;
* :func:`sector_hierarchy` -- unstructured meshes of the re-entrant sector
  $\Omega_p(\delta)$, with the rim carried isoparametrically so that the
  geometric error does not contaminate a corner-limited convergence rate;
* :func:`sector_maxwell_eigs` -- the **exact** spectrum of the sector, in closed
  form, which is what makes the $\delta$-sweep a measurement rather than a
  comparison between two computations;
* the corner arithmetic (:func:`corner_rates`) and Dauge's L-shape benchmark
  (:data:`DAUGE_LSHAPE`).

The structured L-shape construction and the Gmsh sector follow
``../functions.py``; they are reproduced here so that ``final/`` stands alone.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from firedrake import (Mesh, Function, FunctionSpace, VectorFunctionSpace,
                       DirichletBC, Constant, TestFunction, assemble, dx,
                       COMM_WORLD, MeshHierarchy)
from firedrake.cython import dmcommon
from firedrake.mesh import plex_from_cell_list

__all__ = [
    "DAUGE_LSHAPE", "OMEGA_LSHAPE", "corner_rates",
    "lshape_mesh", "grade_towards_corner", "min_cell_volume",
    "sector_mesh", "sector_hierarchy", "graded_sector_mesh",
    "curve_rim", "snap_rim",
    "sector_maxwell_eigs",
]


# --------------------------------------------------------------- the corner
#: Interior angle at the re-entrant corner of the L-shape.
OMEGA_LSHAPE = 1.5 * np.pi

#: Dauge's benchmark values for the Maxwell eigenvalue problem on
#: $(-1,1)^2\setminus([0,1]\times(-1,0])$ -- the first five, $\lambda_3$ and
#: $\lambda_4$ being the double eigenvalue $\pi^2$.
DAUGE_LSHAPE = np.array([1.47562182408, 3.53403136678, 9.86960440109,
                         9.86960440109, 11.38947939790])


def corner_rates(omega, p=1):
    r"""What a re-entrant corner of interior angle $\omega$ costs.

    The leading singular function is $r^{\alpha}\sin(\alpha\theta)$ with
    $\alpha = \pi/\omega$, so the eigenfunction is in $H^{\alpha+1-\epsilon}$
    and no more.  On a **quasi-uniform** mesh of degree $p$ that caps the
    eigenvalue at $\mathcal{O}(h^{2\min(p,\alpha)})$ -- i.e. $h^{2\alpha}$ for
    any $\alpha<p$ -- and the $p$-version at the algebraic $\mathcal{O}(p^{-4\alpha})$
    rather than exponentially.  On a mesh **graded** with parameter $\mu$ the
    cap moves to $\mathcal{O}(h^{2\min(p,\,\alpha/\mu)})$, which is the
    prediction :func:`grade_towards_corner` is there to test.
    """
    alpha = np.pi / omega
    return dict(omega=omega, alpha=alpha,
                h_rate=2.0 * min(p, alpha), p_rate=4.0 * alpha)


# ------------------------------------------------------------------ L-shape
_LSHAPE_MARKERS = """1 : x = -1   2 : x = 1   3 : y = -1
4 : y = 1    5 : x = 0 (re-entrant)   6 : y = 0 (re-entrant)"""


def _mark_lshape_boundaries(plex, tol):
    """Tag the six straight edges of the L-shape by the midpoint of each face."""
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    if plex.getStratumSize("boundary_faces", 1) == 0:
        return
    coords, sec = plex.getCoordinates(), plex.getCoordinateSection()
    for face in plex.getStratumIS("boundary_faces", 1).getIndices():
        x, y = plex.vecGetClosure(sec, coords, face).reshape(-1, 2).mean(axis=0)
        if abs(x + 1.0) < tol:
            marker = 1
        elif abs(x - 1.0) < tol:
            marker = 2
        elif abs(y + 1.0) < tol:
            marker = 3
        elif abs(y - 1.0) < tol:
            marker = 4
        elif abs(x) < tol:
            marker = 5
        elif abs(y) < tol:
            marker = 6
        else:
            raise RuntimeError(f"unclassifiable L-shape face at ({x}, {y})")
        plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, marker)
    plex.removeLabel("boundary_faces")


def grade_towards_corner(coords, gamma):
    r"""Radial grading towards the origin in the $\ell^\infty$ metric.

    Each point is moved along its own ray by

    .. math::

        p \;\longmapsto\; p\,\rho(p)^{1/\gamma-1},
        \qquad \rho(p)=\max(|x|,|y|),

    so that $\rho \mapsto \rho^{1/\gamma}$.  Two properties make this the right
    map for $\Omega_L$ and not merely a plausible one: $\rho$ is **constant on
    the outer boundary** of $(-1,1)^2$, so the outer boundary is fixed
    pointwise; and $\Omega_L$ is star-shaped about the re-entrant corner, so
    every ray stays inside it.  The domain is therefore mapped exactly onto
    itself -- no geometric error is introduced, which matters because a
    geometric error would be indistinguishable from the effect being measured.

    The element size becomes $h_K \sim h\,\rho_K^{\,1-\gamma}$ away from the
    corner and $h_K \sim h^{1/\gamma}$ at it, which is the classical grading of
    parameter $\mu=\gamma$.  $\gamma = 1$ is the identity.
    """
    if gamma == 1.0:
        return coords
    rho = np.max(np.abs(coords), axis=1)
    scale = np.ones(len(coords))
    nz = rho > 0.0
    scale[nz] = rho[nz] ** (1.0 / gamma - 1.0)
    return coords * scale[:, None]


def lshape_mesh(n, gamma=1.0, diagonal="crossed", comm=COMM_WORLD,
                name="L_shape"):
    r"""Structured triangulation of $\Omega_L=(-1,1)^2\setminus([0,1]\times(-1,0])$.

    ``n`` cells per unit edge, so $12n^2$ criss-cross cells and $h=1/n$ away
    from the corner.  ``gamma`` < 1 grades towards the re-entrant corner
    (:func:`grade_towards_corner`); ``gamma = 1`` is the uniform mesh on which
    every rate in the study is capped.

    Criss-cross rather than a single diagonal, for the same reason as on the
    square: the mesh is then symmetric under all eight symmetries of the
    ambient square, so it cannot itself distinguish a direction that the wind
    is about to.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if diagonal not in ("crossed", "left", "right"):
        raise ValueError(f"unknown diagonal {diagonal!r}")

    h, N = 1.0 / n, 2 * n
    grid = -1.0 + np.arange(N + 1) * h
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    coords = np.column_stack([gx.ravel(), gy.ravel()])

    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    keep = ~((i >= n) & (j < n))          # drop the bottom-right quadrant
    i, j = i[keep], j[keep]

    v00 = i * (N + 1) + j
    v10, v01 = v00 + (N + 1), v00 + 1
    v11 = v10 + 1
    if diagonal == "right":
        cells = np.vstack([np.column_stack([v00, v10, v11]),
                           np.column_stack([v00, v11, v01])])
    elif diagonal == "left":
        cells = np.vstack([np.column_stack([v00, v10, v01]),
                           np.column_stack([v10, v11, v01])])
    else:
        centre = len(coords) + np.arange(len(i))
        coords = np.vstack([coords,
                            np.column_stack([-1.0 + (i + 0.5) * h,
                                             -1.0 + (j + 0.5) * h])])
        cells = np.vstack([np.column_stack([v00, v10, centre]),
                           np.column_stack([v10, v11, centre]),
                           np.column_stack([v11, v01, centre]),
                           np.column_stack([v01, v00, centre])])

    used, inverse = np.unique(cells, return_inverse=True)
    cells, coords = inverse.reshape(-1, 3), coords[used]
    coords = grade_towards_corner(coords, gamma)

    plex = plex_from_cell_list(2, cells, coords, comm)
    # The grading map contracts towards the corner, so the face-classification
    # tolerance has to be measured on the *graded* mesh: an outer face is still
    # h wide, but a face next to the corner is only h**(1/gamma).
    _mark_lshape_boundaries(plex, tol=0.25 * h)
    return Mesh(plex, name=name, comm=comm)


def min_cell_volume(mesh):
    """Smallest cell volume; negative would mean the grading inverted an element."""
    v = assemble(TestFunction(FunctionSpace(mesh, "DG", 0)) * dx)
    return float(np.min(v.dat.data_ro))


# ----------------------------------------------------------- the sector
def sector_mesh(delta_deg, maxh, radius=1.0, maxh_corner=None,
                comm=COMM_WORLD, name="sector"):
    r"""Unstructured mesh of the unit disc less a wedge of opening $\delta$.

    The re-entrant corner sits at the origin with interior angle
    $\omega = 2\pi-\delta$, and the wedge is removed symmetrically about the
    positive $x$-axis.  Boundary markers: ``1`` the rim, ``2`` the upper lip,
    ``3`` the lower lip.

    Unstructured deliberately.  A polar grid would refine towards the corner on
    its own, which is exactly the effect being measured, and its innermost cells
    would be anisotropic by a factor of $n$.

    ``maxh_corner`` sets a *separate* target size at the origin, from which Gmsh
    grows the elements out to ``maxh``.  Left at ``None`` the mesh is
    quasi-uniform, which is what every rate in the domain-regularity study is
    measured on; set small it grades towards the re-entrant corner and is what
    :func:`graded_sector_mesh` uses to build a reference far more accurate than
    uniform refinement can reach at the same cost.
    """
    import gmsh

    full = 360.0
    if not 0.0 < delta_deg < full:
        raise ValueError(f"delta must lie in (0, 360), got {delta_deg}")
    alpha = np.pi * delta_deg / full            # half the mouth, in radians
    omega = 2.0 * np.pi - 2.0 * alpha

    running = gmsh.isInitialized()
    if not running:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        h_corner = maxh if maxh_corner is None else maxh_corner
        origin = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h_corner)
        # Gmsh circle arcs must span less than half a turn.
        n_arcs = max(2, int(np.ceil(omega / (0.5 * np.pi))))
        angles = alpha + np.arange(n_arcs + 1) * omega / n_arcs
        rim = [gmsh.model.geo.addPoint(radius * np.cos(a), radius * np.sin(a),
                                       0.0, maxh) for a in angles]
        arcs = [gmsh.model.geo.addCircleArc(rim[k], origin, rim[k + 1])
                for k in range(n_arcs)]
        upper = gmsh.model.geo.addLine(origin, rim[0])
        lower = gmsh.model.geo.addLine(rim[-1], origin)
        loop = gmsh.model.geo.addCurveLoop([upper, *arcs, lower])
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, arcs, 1)
        gmsh.model.addPhysicalGroup(1, [upper], 2)
        gmsh.model.addPhysicalGroup(1, [lower], 3)
        gmsh.model.addPhysicalGroup(2, [surface], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.model.mesh.generate(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sector.msh")
            gmsh.write(path)
            return Mesh(path, name=name, comm=comm)
    finally:
        if running:
            gmsh.model.remove()
        else:
            gmsh.finalize()


def snap_rim(mesh, radius, rim_marker=1):
    """Push the P1 rim vertices back onto the exact circle, in place.

    Uniform refinement inserts edge midpoints, which sit strictly inside the
    disc; without this the meshed area creeps away from the true one as the
    hierarchy is refined.
    """
    V = mesh.coordinates.function_space()
    nodes = DirichletBC(V, Constant((0.0, 0.0)), rim_marker).nodes
    x = mesh.coordinates.dat.data
    r = np.linalg.norm(x[nodes], axis=1)
    keep = r > 1e-12
    x[nodes[keep]] *= (radius / r[keep])[:, None]


def curve_rim(mesh, radius, degree=2, rim_marker=1):
    r"""Isoparametric mesh whose rim is represented to $\mathcal{O}(h^{p+1})$.

    Snapping vertices leaves the edges as chords, so the meshed area is still
    too small by $\mathcal{O}(h^2)$: a geometric error independent of the finite
    element space, which biases every eigenvalue the same way.

    That would be harmless if it were small, and here it is not.  The corner
    limits the discretisation error to $\mathcal{O}(h^{2\alpha})$ with
    $2\alpha \approx 1.3$, which decays *more slowly* than the geometric
    $\mathcal{O}(h^2)$, and the two carry opposite signs; at attainable
    resolutions they are comparable, the total error passes through zero
    between levels and the observed rate comes out negative.  Raising the
    coordinate field to degree 2 and snapping all of its rim nodes -- vertices
    *and* edge nodes -- drops the geometric error to $\mathcal{O}(h^3)$ and
    restores a single-signed, monotone error sequence.  Pass ``degree=1`` to
    reproduce the chord polygon and see the artefact.
    """
    if degree < 2:
        return mesh
    V = VectorFunctionSpace(mesh, "CG", degree)
    coords = Function(V).interpolate(mesh.coordinates)
    nodes = DirichletBC(V, Constant((0.0, 0.0)), rim_marker).nodes
    x = coords.dat.data
    r = np.linalg.norm(x[nodes], axis=1)
    keep = r > 1e-12
    x[nodes[keep]] *= (radius / r[keep])[:, None]
    return Mesh(coords)


_SECTOR_CACHE = {}


def sector_hierarchy(delta_deg, nlevels=3, radius=1.0, maxh=0.35,
                     geom_degree=2):
    r"""Uniformly refined meshes of $\Omega_p(\delta)$, rim carried to $h^3$.

    Returns ``nlevels + 1`` meshes.  Cached, because the same $\delta$ is used
    by several studies and Gmsh is the slow step.
    """
    key = (delta_deg, nlevels, radius, maxh, geom_degree)
    if key not in _SECTOR_CACHE:
        base = sector_mesh(delta_deg, maxh=maxh, radius=radius)
        hierarchy = MeshHierarchy(base, nlevels)
        for m in hierarchy:
            snap_rim(m, radius)
        _SECTOR_CACHE[key] = [curve_rim(m, radius, degree=geom_degree)
                              for m in hierarchy]
    return _SECTOR_CACHE[key]


_GRADED_CACHE = {}


def graded_sector_mesh(delta_deg, maxh, maxh_corner, radius=1.0, geom_degree=2):
    r"""One sector mesh graded towards the re-entrant corner, rim carried to $h^3$.

    The reference the advected studies are read against has to be better than
    every mesh in those studies by enough that the comparison measures the
    discretisation and not the reference.  Uniform refinement cannot deliver
    that here: the corner caps the eigenvalue error at $\mathcal{O}(h^{2\alpha})$
    with $2\alpha \to 1$ as the mouth closes, so the finest uniform mesh that
    fits in memory is only a factor of two or three better than the coarser ones
    it is supposed to referee.

    Grading breaks the cap.  Concentrating elements at the origin -- ``maxh`` at
    the rim, ``maxh_corner`` at the corner, Gmsh interpolating between -- buys
    four decades on the fundamental at $\delta = 5^\circ$ over a uniform mesh of
    twice the size, because the error it removes is the corner's and not the far
    field's.  This is the same idea as :func:`grade_towards_corner` on the
    L-shape, expressed through the mesh generator rather than through a map of
    the coordinates.

    Cached on its arguments; Gmsh is the slow step and the same reference is
    wanted by several studies.
    """
    key = (delta_deg, maxh, maxh_corner, radius, geom_degree)
    if key not in _GRADED_CACHE:
        m = sector_mesh(delta_deg, maxh=maxh, radius=radius,
                        maxh_corner=maxh_corner, name="sector_graded")
        snap_rim(m, radius)
        _GRADED_CACHE[key] = curve_rim(m, radius, degree=geom_degree)
    return _GRADED_CACHE[key]


def sector_maxwell_eigs(omega, radius=1.0, n_eigs=10, kmax=40, mmax=12,
                        x_max=80.0, n_scan=40000, with_index=False):
    r"""Exact non-zero Maxwell eigenvalues of a circular sector, in closed form.

    On a sector the problem separates: with $\nu = k\pi/\omega$ the solutions
    are $J_\nu(\sqrt\lambda\,r)\cos(\nu\theta)$, which satisfy the natural
    condition on both straight lips for every integer $k$, and the rim
    condition gives $J'_\nu(\sqrt\lambda\,R) = 0$.  On a simply connected
    planar domain the non-zero Maxwell eigenvalues coincide with the non-zero
    Neumann Laplacian ones, so this is exact for $\Omega_p(\delta)$ with
    $\omega = 2\pi-\delta$.

    ``with_index=True`` additionally returns the angular index $k$ of each
    eigenvalue.  That index is what decides the convergence rate: the family
    behaves like $r^{k\pi/\omega}$ at the corner, so it is singular only when
    $k\pi/\omega < 1$ and the predicted eigenvalue rate is
    $2\min(p,\,k\pi/\omega)$ -- a *per-mode* prediction, sharper than "the
    fundamental is capped and the rest are not", and one the $\delta$-sweep can
    check mode by mode.
    """
    from scipy.special import jvp
    from scipy.optimize import brentq

    xs = np.linspace(1e-6, x_max, n_scan)
    values, angular = [], []
    for k in range(kmax + 1):
        nu = k * np.pi / omega
        f = jvp(nu, xs, 1)
        brackets = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        for idx in brackets[:mmax]:
            root = brentq(lambda x: jvp(nu, x, 1), xs[idx], xs[idx + 1])
            if root > 1e-8:
                values.append((root / radius) ** 2)
                angular.append(k)
    values, angular = np.asarray(values), np.asarray(angular)
    order = np.argsort(values)
    values, angular = values[order], angular[order]
    keep = values > 1e-8
    values, angular = values[keep][:n_eigs], angular[keep][:n_eigs]
    return (values, angular) if with_index else values
