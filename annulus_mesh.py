"""
Annulus meshing utilities.

Extracted verbatim from Annulus.ipynb so that several notebooks can share one
definition.  The annulus  inner_radius < |x| < outer_radius  has first Betti
number b_1 = 1, which is what makes it the right test domain for the harmonic /
zero-eigenvalue count.

Boundary markers
    1 : outer rim,  r = outer_radius
    2 : inner rim,  r = inner_radius
"""

import os
import tempfile

import numpy as np
from firedrake import *
from firedrake.cython import dmcommon
from firedrake.mesh import plex_from_cell_list

def make_annulus_mesh(
    n=10,
    inner_radius=0.5,
    outer_radius=1.0,
    diagonal="crossed",
    structured=None,
    maxh=None,
    n_angular=None,
    comm=COMM_WORLD,
    name="annulus",
    **kwargs,
):
    """
    Mesh the annulus  inner_radius < |x| < outer_radius.

    Boundary markers
        1 : outer rim,  r = outer_radius
        2 : inner rim,  r = inner_radius

    Unlike the Pac-Man sector there is no degenerate apex, so the structured
    polar branch is genuinely shape-regular: the worst aspect ratio is about
    outer_radius/inner_radius and is independent of n. Both branches are
    therefore usable for convergence studies.

    Parameters mirror make_pacman_mesh. ``n`` is cells per unit length: a
    structured mesh has round(n*(b-a)) cells radially and round(2*pi*n*r_mid)
    angularly, so cells are roughly square at the mean radius.
    """
    a, b = float(inner_radius), float(outer_radius)
    if not 0.0 < a < b:
        raise ValueError("need 0 < inner_radius < outer_radius")
    if structured is None:
        structured = maxh is None
    if structured:
        if maxh is not None:
            raise ValueError("maxh only applies to unstructured meshes; "
                             "pass structured=False")
        return _annulus_structured(n, a, b, diagonal, n_angular,
                                   comm, name, **kwargs)
    return _annulus_unstructured(maxh if maxh is not None else 1.0 / n,
                                 a, b, comm, name, **kwargs)


def _annulus_structured(n, a, b, diagonal, n_angular, comm, name, **kwargs):
    """Polar grid, periodic in theta. Every cell is a quadrilateral."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    if diagonal not in ("crossed", "left", "right"):
        raise ValueError(f"Unknown diagonal '{diagonal}'")

    nr = max(1, int(round(n * (b - a))))
    if n_angular is None:
        nt = max(3, int(round(n * 2.0 * np.pi * 0.5 * (a + b))))
    else:
        nt = int(n_angular)
        if nt < 3:
            raise ValueError("n_angular must be at least 3")

    # Node (i, j) = ring i = 0..nr, spoke j = 0..nt-1, index i*nt + j.
    # Periodic in j: spoke nt is spoke 0, so no duplicated seam.
    ring_r = a + np.arange(nr + 1) * (b - a) / nr
    theta = np.arange(nt) * 2.0 * np.pi / nt
    R, TH = np.meshgrid(ring_r, theta, indexing="ij")
    coords = np.column_stack([(R * np.cos(TH)).ravel(),
                              (R * np.sin(TH)).ravel()])

    i, j = np.meshgrid(np.arange(nr), np.arange(nt), indexing="ij")
    i, j = i.ravel(), j.ravel()
    jp = (j + 1) % nt
    v00 = i * nt + j            # inner, clockwise corner
    v10 = (i + 1) * nt + j      # outer, clockwise
    v11 = (i + 1) * nt + jp     # outer, anticlockwise
    v01 = i * nt + jp           # inner, anticlockwise

    if diagonal == "right":
        cells = [np.column_stack([v00, v10, v11]),
                 np.column_stack([v00, v11, v01])]
    elif diagonal == "left":
        cells = [np.column_stack([v00, v10, v01]),
                 np.column_stack([v10, v11, v01])]
    else:
        quad = np.column_stack([v00, v10, v11, v01])
        centre = len(coords) + np.arange(len(quad))
        coords = np.vstack([coords, coords[quad].mean(axis=1)])
        cells = [np.column_stack([v00, v10, centre]),
                 np.column_stack([v10, v11, centre]),
                 np.column_stack([v11, v01, centre]),
                 np.column_stack([v01, v00, centre])]

    plex = plex_from_cell_list(2, np.vstack(cells), coords, comm)
    _mark_annulus_boundaries(plex, a, b)
    return Mesh(plex, name=name, comm=comm, **kwargs)


def _mark_annulus_boundaries(plex, a, b):
    """1 = outer rim, 2 = inner rim, classified by edge-midpoint radius."""
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    if plex.getStratumSize("boundary_faces", 1) > 0:
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        mid = 0.5 * (a + b)
        for face in plex.getStratumIS("boundary_faces", 1).getIndices():
            x, y = (plex.vecGetClosure(coord_sec, coords, face)
                    .reshape(-1, 2).mean(axis=0))
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face,
                               1 if np.hypot(x, y) > mid else 2)
    plex.removeLabel("boundary_faces")


def _annulus_unstructured(maxh, a, b, comm, name, **kwargs):
    """Gmsh annulus: outer curve loop with the inner loop as a hole."""
    import gmsh
    already_running = gmsh.isInitialized()
    if not already_running:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        geo = gmsh.model.geo
        origin = geo.addPoint(0.0, 0.0, 0.0, maxh)

        def circle(radius):
            # Four quarter-turn arcs; gmsh arcs must span less than half a turn.
            angles = np.arange(4) * 0.5 * np.pi
            pts = [geo.addPoint(radius * np.cos(t), radius * np.sin(t),
                                0.0, maxh) for t in angles]
            return [geo.addCircleArc(pts[k], origin, pts[(k + 1) % 4])
                    for k in range(4)]

        outer, inner = circle(b), circle(a)
        surface = geo.addPlaneSurface([geo.addCurveLoop(outer),
                                       geo.addCurveLoop(inner)])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(1, outer, 1)
        gmsh.model.addPhysicalGroup(1, inner, 2)
        gmsh.model.addPhysicalGroup(2, [surface], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
        gmsh.model.mesh.generate(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            msh_file = os.path.join(tmpdir, "annulus.msh")
            gmsh.write(msh_file)
            return Mesh(msh_file, name=name, comm=comm, **kwargs)
    finally:
        if already_running:
            gmsh.model.remove()
        else:
            gmsh.finalize()


def snap_annulus(mesh, inner_radius, outer_radius):
    """Push both rims back onto their circles after refinement."""
    V = mesh.coordinates.function_space()
    x = mesh.coordinates.dat.data
    for marker, radius in ((1, outer_radius), (2, inner_radius)):
        nodes = DirichletBC(V, Constant((0.0, 0.0)), marker).nodes
        r = np.linalg.norm(x[nodes], axis=1)
        x[nodes] *= (radius / r)[:, None]


def curve_annulus(mesh, inner_radius, outer_radius, degree=2):
    r"""Isoparametric annulus: both rims carried to $\mathcal{O}(h^{p+1})$.

    Snapping vertices onto the two circles leaves the edges as chords, so the
    meshed area is short by $\mathcal{O}(h^2)$ -- a geometric error that is
    independent of the finite element space and of fixed sign, and therefore
    indistinguishable from a discretisation error of the same order.  On the
    annulus at ``n = 32`` the chord polygon misses the area by $7\times10^{-4}$
    and its eigenvalue error is *smaller* than the isoparametric mesh's, because
    the two errors carry opposite signs and partly cancel; the cancellation is
    not a virtue, it is a rate that cannot be trusted.  Raising the coordinate
    field to degree $2$ and snapping all of its rim nodes -- vertices *and* edge
    nodes -- takes the area error to $10^{-8}$ and restores a single-signed,
    monotone error sequence.  Pass ``degree=1`` to reproduce the artefact.

    The same construction as ``domain_meshes.curve_rim``, for two rims rather
    than one; the markers are the module's own, 1 = outer and 2 = inner.
    """
    if degree < 2:
        return mesh
    V = VectorFunctionSpace(mesh, "CG", degree)
    coords = Function(V).interpolate(mesh.coordinates)
    x = coords.dat.data
    for marker, radius in ((1, float(outer_radius)), (2, float(inner_radius))):
        nodes = DirichletBC(V, Constant((0.0, 0.0)), marker).nodes
        r = np.linalg.norm(x[nodes], axis=1)
        keep = r > 1e-12
        x[nodes[keep]] *= (radius / r[keep])[:, None]
    return Mesh(coords)


def annulus_hierarchy(nlevels=4, inner_radius=0.5, outer_radius=1.0,
                      maxh=0.15, **kwargs):
    base = make_annulus_mesh(inner_radius=inner_radius,
                             outer_radius=outer_radius,
                             structured=False, maxh=maxh, **kwargs)
    mh = MeshHierarchy(base, nlevels)
    for m in mh:
        snap_annulus(m, inner_radius, outer_radius)
    return mh
