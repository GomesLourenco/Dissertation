import os
import tempfile

import numpy as np
from firedrake import *
from firedrake.cython import dmcommon
from firedrake.mesh import plex_from_cell_list
from netgen.occ import *
from ngsolve.webgui import Draw


# Corners of the L-shaped domain
#
#     [-1, 1]^2 \ ([0, 1] x [-1, 0])
#
# listed counter-clockwise from the re-entrant corner (0, 0).
#
#              (-1,1) -------- (1,1)
#                 |              |
#                 |              |
#                 |              |
#                 |      (0,0)--(1,0)
#                 |        |
#                 |        |
#              (-1,-1)--(0,-1)
#
# The re-entrant corner is at (0, 0).

_LSHAPE_CORNERS = (
    (0, 0),
    (1, 0),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (0, -1),
)


# Boundary markers:
#
#   1: x = -1
#   2: x =  1
#   3: y = -1
#   4: y =  1
#   5: x =  0   (re-entrant edge)
#   6: y =  0   (re-entrant edge)
#
# Listed in the same order as the edges of _LSHAPE_CORNERS:
#
#   (0,0)   -> (1,0)    : y =  0  -> 6
#   (1,0)   -> (1,1)    : x =  1  -> 2
#   (1,1)   -> (-1,1)   : y =  1  -> 4
#   (-1,1)  -> (-1,-1)  : x = -1  -> 1
#   (-1,-1) -> (0,-1)   : y = -1  -> 3
#   (0,-1)  -> (0,0)    : x =  0  -> 5

_LSHAPE_EDGE_MARKERS = (6, 2, 4, 1, 3, 5)


def make_Lshape_mesh(
    n=10,
    diagonal="crossed",
    structured=None,
    maxh=None,
    comm=COMM_WORLD,
    name="L_shape",
    **kwargs,
):
    """
    Mesh the L-shaped domain

        [-1, 1]^2 \\ ([0, 1] x [-1, 0]),

    with the re-entrant corner at (0, 0).

    Parameters
    ----------
    n : int
        Cells per unit length. A structured mesh has n cells along each
        unit edge; an unstructured mesh uses maxh = 1/n unless maxh is given.

    diagonal : {"crossed", "left", "right"}
        Structured meshes only. How each square of the grid is split into
        triangles:

        "right"
            Cut from bottom-left to top-right.

        "left"
            Cut from top-left to bottom-right.

        "crossed"
            Insert a centre node and cut both ways, producing four triangles
            per square.

        Same meaning as the ``diagonal`` argument of RectangleMesh.

    structured : bool, optional
        Defaults to True unless maxh is given.

    maxh : float, optional
        Target element size for the unstructured (Gmsh) mesh.

    comm : MPI communicator
        MPI communicator used to construct the mesh.

    name : str
        Name of the resulting mesh.

    **kwargs
        Forwarded to firedrake.Mesh, e.g. ``reorder`` or
        ``distribution_parameters``.

    Returns
    -------
    Mesh
        L-shaped mesh with boundary markers

            1: x = -1
            2: x =  1
            3: y = -1
            4: y =  1
            5: x =  0  (re-entrant)
            6: y =  0  (re-entrant)

    Notes
    -----
    The unstructured branch shells out to Gmsh via a temporary .msh file and
    is intended for serial use.
    """
    if structured is None:
        structured = maxh is None

    if structured:
        if maxh is not None:
            raise ValueError(
                "maxh only applies to unstructured meshes; "
                "pass structured=False"
            )

        return _lshape_structured(
            n,
            diagonal,
            comm,
            name,
            **kwargs,
        )

    return _lshape_unstructured(
        maxh if maxh is not None else 1.0 / n,
        comm,
        name,
        **kwargs,
    )


def _lshape_structured(n, diagonal, comm, name, **kwargs):
    """
    Structured L-shaped mesh.

    Start with a uniform grid on [-1, 1]^2 and remove the bottom-right
    quadrant

        [0, 1] x [-1, 0].

    The remaining squares are then split into triangles.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    if diagonal not in ("crossed", "left", "right"):
        raise ValueError(f"Unknown diagonal '{diagonal}'")

    h = 1.0 / n

    # Number of cells along each side of the enclosing square [-1, 1]^2.
    N = 2 * n

    # Vertices of the full (N+1) x (N+1) grid.
    #
    # Vertex (i, j) has coordinates
    #
    #     x = -1 + i*h
    #     y = -1 + j*h
    #
    # and index
    #
    #     i*(N+1) + j.
    grid = -1.0 + np.arange(N + 1) * h

    gx, gy = np.meshgrid(
        grid,
        grid,
        indexing="ij",
    )

    coords = np.column_stack([
        gx.ravel(),
        gy.ravel(),
    ])

    # Cell indices of the enclosing square.
    i, j = np.meshgrid(
        np.arange(N),
        np.arange(N),
        indexing="ij",
    )

    # Remove the bottom-right quadrant:
    #
    #     x >= 0  <=> i >= n
    #     y <  0  <=> j < n
    #
    # Thus the removed cells satisfy
    #
    #     (i >= n) & (j < n).
    keep = ~((i >= n) & (j < n))

    i = i[keep]
    j = j[keep]

    # Corners of each retained square, counter-clockwise from the
    # bottom-left vertex.
    v00 = i * (N + 1) + j
    v10 = v00 + (N + 1)
    v11 = v10 + 1
    v01 = v00 + 1

    if diagonal == "right":
        cells = np.vstack([
            np.column_stack([v00, v10, v11]),
            np.column_stack([v00, v11, v01]),
        ])

    elif diagonal == "left":
        cells = np.vstack([
            np.column_stack([v00, v10, v01]),
            np.column_stack([v10, v11, v01]),
        ])

    else:
        # "crossed":
        # Add one vertex at the centre of every retained square and
        # split the square into four triangles.
        centre = len(coords) + np.arange(len(i))

        centre_coords = np.column_stack([
            -1.0 + (i + 0.5) * h,
            -1.0 + (j + 0.5) * h,
        ])

        coords = np.vstack([
            coords,
            centre_coords,
        ])

        cells = np.vstack([
            np.column_stack([v00, v10, centre]),
            np.column_stack([v10, v11, centre]),
            np.column_stack([v11, v01, centre]),
            np.column_stack([v01, v00, centre]),
        ])

    # Drop vertices that are not referenced by any retained cell, and
    # renumber the cell connectivity accordingly.
    used, inverse = np.unique(
        cells,
        return_inverse=True,
    )

    cells = inverse.reshape(-1, 3)
    coords = coords[used]

    plex = plex_from_cell_list(
        2,
        cells,
        coords,
        comm,
    )

    _mark_lshape_boundaries(
        plex,
        tol=0.25 * h,
    )

    return Mesh(
        plex,
        name=name,
        comm=comm,
        **kwargs,
    )


def _mark_lshape_boundaries(plex, tol):
    """
    Tag the six boundary edges of the L-shape.

    Markers
    -------
    1 : x = -1
    2 : x =  1
    3 : y = -1
    4 : y =  1
    5 : x =  0, -1 <= y <= 0
    6 : y =  0,  0 <= x <= 1
    """
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")

    if plex.getStratumSize("boundary_faces", 1) > 0:
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()

        for face in plex.getStratumIS(
            "boundary_faces",
            1,
        ).getIndices():

            # Compute the midpoint of the boundary edge.
            x, y = (
                plex.vecGetClosure(
                    coord_sec,
                    coords,
                    face,
                )
                .reshape(-1, 2)
                .mean(axis=0)
            )

            if abs(x + 1.0) < tol:
                # Left outer boundary: x = -1
                marker = 1

            elif abs(x - 1.0) < tol:
                # Right outer boundary: x = 1
                marker = 2

            elif abs(y + 1.0) < tol:
                # Bottom outer boundary: y = -1
                marker = 3

            elif abs(y - 1.0) < tol:
                # Top outer boundary: y = 1
                marker = 4

            elif abs(x) < tol:
                # Vertical re-entrant boundary: x = 0
                marker = 5

            elif abs(y) < tol:
                # Horizontal re-entrant boundary: y = 0
                marker = 6

            else:
                raise RuntimeError(
                    f"Could not classify L-shape boundary face "
                    f"with midpoint ({x}, {y})"
                )

            plex.setLabelValue(
                dmcommon.FACE_SETS_LABEL,
                face,
                marker,
            )

    plex.removeLabel("boundary_faces")


def _lshape_unstructured(maxh, comm, name, **kwargs):
    """
    Unstructured L-shaped mesh generated with Gmsh.

    The polygon is

        (0,0)
          -> (1,0)
          -> (1,1)
          -> (-1,1)
          -> (-1,-1)
          -> (0,-1)
          -> (0,0).
    """
    import gmsh

    already_running = gmsh.isInitialized()

    if not already_running:
        gmsh.initialize()

    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)

        points = [
            gmsh.model.geo.addPoint(
                x,
                y,
                0,
                maxh,
            )
            for x, y in _LSHAPE_CORNERS
        ]

        lines = [
            gmsh.model.geo.addLine(
                points[k],
                points[(k + 1) % len(points)],
            )
            for k in range(len(points))
        ]

        curve_loop = gmsh.model.geo.addCurveLoop(lines)

        surface = gmsh.model.geo.addPlaneSurface([
            curve_loop
        ])

        gmsh.model.geo.synchronize()

        # Apply boundary markers in the same order as the polygon edges.
        for line, marker in zip(
            lines,
            _LSHAPE_EDGE_MARKERS,
        ):
            gmsh.model.addPhysicalGroup(
                1,
                [line],
                marker,
            )

        # Physical marker for the 2D domain itself.
        gmsh.model.addPhysicalGroup(
            2,
            [surface],
            1,
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeMax",
            maxh,
        )

        gmsh.model.mesh.generate(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            msh_file = os.path.join(
                tmpdir,
                "lshape.msh",
            )

            gmsh.write(msh_file)

            return Mesh(
                msh_file,
                name=name,
                comm=comm,
                **kwargs,
            )

    finally:
        if already_running:
            gmsh.model.remove()
        else:
            gmsh.finalize()


# The Pac-Man domain is the circular sector of radius R
#
#     { (r, theta) : 0 <= r <= R,  alpha <= theta <= 2*pi - alpha },
#
# i.e. a disc with a wedge ("the mouth") of opening angle
#
#     mouth_angle = 2*alpha
#
# removed, symmetrically about the positive x-axis.
#
#                       . - ~ - .
#                   ,'           `.
#                  /               \ ______  theta = +alpha
#                 |             _ - '
#                 |         (0,0)          <-- re-entrant corner
#                 |             ` - _
#                  \               / ~~~~~~  theta = -alpha
#                   `.           ,'
#                       ` - ~ - '
#
# The re-entrant corner is at the origin and has interior angle
#
#     omega = 2*pi - mouth_angle,
#
# so the mesh is re-entrant (omega > pi) whenever mouth_angle < pi.
#
# Boundary markers:
#
#   1: the circular rim,  r = R
#   2: the upper lip,     theta = +alpha
#   3: the lower lip,     theta = -alpha


def make_pacman_mesh(
    n=10,
    mouth_angle=60.0,
    radius=1.0,
    diagonal="crossed",
    structured=None,
    maxh=None,
    n_angular=None,
    degrees=True,
    comm=COMM_WORLD,
    name="pacman",
    **kwargs,
):
    """
    Mesh the Pac-Man domain: a disc of the given radius with a wedge of
    opening angle ``mouth_angle`` removed, symmetrically about the positive
    x-axis.

    The re-entrant corner sits at the origin and has interior angle

        omega = 2*pi - mouth_angle.

    Parameters
    ----------
    n : int
        Cells per unit length. A structured mesh has ``round(n*radius)``
        cells in the radial direction and ``round(n*radius*omega)`` in the
        angular direction, so that cells near the rim are roughly square.
        An unstructured mesh uses maxh = 1/n unless maxh is given.

    mouth_angle : float
        Opening angle of the mouth, in degrees unless ``degrees=False``.
        Must lie strictly between 0 and a full turn. Small values give a
        sharp re-entrant corner and a strong corner singularity;
        ``mouth_angle`` equal to half a turn gives a half-disc, which is
        convex and has no singularity.

    radius : float
        Radius of the disc.

    diagonal : {"crossed", "left", "right"}
        Structured meshes only. How each cell of the polar grid is split
        into triangles, with the radial direction playing the role of x and
        the angular direction the role of y:

        "right"
            Cut from the inner-clockwise to the outer-anticlockwise corner.

        "left"
            Cut from the inner-anticlockwise to the outer-clockwise corner.

        "crossed"
            Insert a centre node and cut both ways, producing four triangles
            per cell.

        Same meaning as the ``diagonal`` argument of RectangleMesh.

    structured : bool, optional
        Defaults to True unless maxh is given.

    maxh : float, optional
        Target element size for the unstructured (Gmsh) mesh.

    n_angular : int, optional
        Structured meshes only. Overrides the number of cells in the angular
        direction.

    degrees : bool
        Whether ``mouth_angle`` is given in degrees (the default) or radians.

    comm : MPI communicator
        MPI communicator used to construct the mesh.

    name : str
        Name of the resulting mesh.

    **kwargs
        Forwarded to firedrake.Mesh, e.g. ``reorder`` or
        ``distribution_parameters``.

    Returns
    -------
    Mesh
        Pac-Man mesh with boundary markers

            1: the circular rim,  r = radius
            2: the upper lip,     theta = +mouth_angle/2
            3: the lower lip,     theta = -mouth_angle/2

    Notes
    -----
    Both branches approximate the rim by straight edges; the vertices lie
    exactly on the circle but the edges between them are chords, so the
    meshed area is slightly less than the exact area. This is an O(h^2)
    geometric error, which is worth keeping in mind when measuring
    convergence rates against exact eigenvalues.

    The structured branch is a polar grid, whose innermost ring is a fan of
    triangles meeting at the origin. Its cells are strongly anisotropic near
    the origin: the innermost cells have a radial-to-angular aspect ratio of
    roughly n*radius. This refinement towards the corner is often welcome,
    since that is where the singularity lives, but the anisotropy is not
    something the unstructured branch produces.

    The unstructured branch shells out to Gmsh via a temporary .msh file and
    is intended for serial use.
    """
    full_turn = 360.0 if degrees else 2.0 * np.pi

    if not 0.0 < mouth_angle < full_turn:
        raise ValueError(
            f"mouth_angle must lie strictly between 0 and {full_turn}, "
            f"got {mouth_angle}"
        )

    if radius <= 0.0:
        raise ValueError("radius must be positive")

    # Half the mouth opening, in radians. The domain then spans
    # alpha <= theta <= alpha + omega.
    alpha = np.pi * mouth_angle / full_turn
    omega = 2.0 * np.pi - 2.0 * alpha

    if structured is None:
        structured = maxh is None

    if structured:
        if maxh is not None:
            raise ValueError(
                "maxh only applies to unstructured meshes; "
                "pass structured=False"
            )

        return _pacman_structured(
            n,
            radius,
            alpha,
            omega,
            diagonal,
            n_angular,
            comm,
            name,
            **kwargs,
        )

    return _pacman_unstructured(
        maxh if maxh is not None else 1.0 / n,
        radius,
        alpha,
        omega,
        comm,
        name,
        **kwargs,
    )


def _pacman_structured(
    n,
    radius,
    alpha,
    omega,
    diagonal,
    n_angular,
    comm,
    name,
    **kwargs,
):
    """
    Structured Pac-Man mesh: a polar grid on the sector.

    The grid has nr cells in the radial direction and nt in the angular one.
    The innermost ring degenerates at the origin, so it is a fan of nt
    triangles with their apex there; the remaining (nr - 1)*nt cells are
    quadrilaterals split according to ``diagonal``.

    For "crossed" the apex triangles are split barycentrically into three,
    so that every cell of the polar grid, quadrilateral or triangular, is
    barycentrically refined.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    if diagonal not in ("crossed", "left", "right"):
        raise ValueError(f"Unknown diagonal '{diagonal}'")

    # Radial cells, and angular cells chosen so that cells near the rim are
    # roughly square: the rim has arc length radius*omega.
    nr = max(1, int(round(n * radius)))

    if n_angular is None:
        nt = max(3, int(round(n * radius * omega)))
    else:
        nt = int(n_angular)

        if nt < 3:
            raise ValueError("n_angular must be at least 3")

    # Node layout:
    #
    #   index 0            the apex, at the origin
    #   index 1 + (i-1)*(nt+1) + j    ring i = 1..nr, spoke j = 0..nt
    #
    # Ring i sits at radius i*radius/nr and spoke j at angle
    # alpha + j*omega/nt.
    theta = alpha + np.arange(nt + 1) * omega / nt
    ring_radii = np.arange(1, nr + 1) * radius / nr

    ring_r, ring_theta = np.meshgrid(
        ring_radii,
        theta,
        indexing="ij",
    )

    coords = np.vstack([
        np.zeros((1, 2)),
        np.column_stack([
            (ring_r * np.cos(ring_theta)).ravel(),
            (ring_r * np.sin(ring_theta)).ravel(),
        ]),
    ])

    crossed = diagonal == "crossed"
    cells = []

    # --- Innermost ring: a fan of triangles meeting at the apex ------------
    #
    # Counter-clockwise as apex -> spoke j -> spoke j+1, since theta
    # increases with j.
    spoke = np.arange(nt)

    fan = np.column_stack([
        np.zeros(nt, dtype=int),
        1 + spoke,
        1 + spoke + 1,
    ])

    if crossed:
        fan_centroid = len(coords) + np.arange(nt)

        coords = np.vstack([
            coords,
            coords[fan].mean(axis=1),
        ])

        cells.extend([
            np.column_stack([fan[:, 0], fan[:, 1], fan_centroid]),
            np.column_stack([fan[:, 1], fan[:, 2], fan_centroid]),
            np.column_stack([fan[:, 2], fan[:, 0], fan_centroid]),
        ])

    else:
        cells.append(fan)

    # --- Remaining rings: quadrilaterals -----------------------------------
    if nr > 1:
        i, j = np.meshgrid(
            np.arange(1, nr),
            np.arange(nt),
            indexing="ij",
        )

        i = i.ravel()
        j = j.ravel()

        # Corners of each cell, counter-clockwise from the inner-clockwise
        # one. Radial plays the role of x and angular the role of y, so this
        # matches the naming used for the L-shape.
        v00 = 1 + (i - 1) * (nt + 1) + j
        v10 = v00 + (nt + 1)
        v11 = v10 + 1
        v01 = v00 + 1

        if diagonal == "right":
            cells.extend([
                np.column_stack([v00, v10, v11]),
                np.column_stack([v00, v11, v01]),
            ])

        elif diagonal == "left":
            cells.extend([
                np.column_stack([v00, v10, v01]),
                np.column_stack([v10, v11, v01]),
            ])

        else:
            # "crossed":
            # The centre node is the average of the four corners rather than
            # the polar midpoint, so that it is guaranteed to lie inside the
            # straight-edged cell.
            quad = np.column_stack([v00, v10, v11, v01])
            centre = len(coords) + np.arange(len(quad))

            coords = np.vstack([
                coords,
                coords[quad].mean(axis=1),
            ])

            cells.extend([
                np.column_stack([v00, v10, centre]),
                np.column_stack([v10, v11, centre]),
                np.column_stack([v11, v01, centre]),
                np.column_stack([v01, v00, centre]),
            ])

    plex = plex_from_cell_list(
        2,
        np.vstack(cells),
        coords,
        comm,
    )

    _mark_pacman_boundaries(
        plex,
        alpha,
        tol=0.25 * omega / nt,
    )

    return Mesh(
        plex,
        name=name,
        comm=comm,
        **kwargs,
    )


def _mark_pacman_boundaries(plex, alpha, tol):
    """
    Tag the three boundary curves of the Pac-Man domain.

    Markers
    -------
    1 : the circular rim
    2 : the upper lip, theta = +alpha
    3 : the lower lip, theta = -alpha

    The two lips are the only boundary edges lying exactly along a ray from
    the origin, so classifying by the polar angle of the edge midpoint is
    enough; everything else is rim. ``tol`` must therefore be smaller than
    the angular width of one rim edge.
    """
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")

    if plex.getStratumSize("boundary_faces", 1) > 0:
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()

        for face in plex.getStratumIS(
            "boundary_faces",
            1,
        ).getIndices():

            # Compute the midpoint of the boundary edge.
            x, y = (
                plex.vecGetClosure(
                    coord_sec,
                    coords,
                    face,
                )
                .reshape(-1, 2)
                .mean(axis=0)
            )

            # Polar angle in [0, 2*pi). Since alpha > 0, neither lip sits at
            # the branch cut, so no wrap-around handling is needed.
            angle = np.arctan2(y, x) % (2.0 * np.pi)

            if abs(angle - alpha) < tol:
                # Upper lip: theta = +alpha
                marker = 2

            elif abs(angle - (2.0 * np.pi - alpha)) < tol:
                # Lower lip: theta = -alpha
                marker = 3

            else:
                # Everything else on the boundary is the rim.
                marker = 1

            plex.setLabelValue(
                dmcommon.FACE_SETS_LABEL,
                face,
                marker,
            )

    plex.removeLabel("boundary_faces")


def _pacman_unstructured(maxh, radius, alpha, omega, comm, name, **kwargs):
    """
    Unstructured Pac-Man mesh generated with Gmsh.

    The boundary is the closed curve

        origin
          -> rim at theta = +alpha
          -> (round the rim, anticlockwise)
          -> rim at theta = -alpha
          -> origin.

    Gmsh circle arcs must span less than half a turn, so the rim is built
    from several arcs sharing the origin as their centre.
    """
    import gmsh

    already_running = gmsh.isInitialized()

    if not already_running:
        gmsh.initialize()

    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)

        # The apex of the sector and the centre of the rim arcs are the
        # same point.
        origin = gmsh.model.geo.addPoint(
            0.0,
            0.0,
            0.0,
            maxh,
        )

        # Split the rim so that every arc spans at most a quarter turn,
        # comfortably below the half-turn limit.
        n_arcs = max(2, int(np.ceil(omega / (0.5 * np.pi))))
        arc_angles = alpha + np.arange(n_arcs + 1) * omega / n_arcs

        rim_points = [
            gmsh.model.geo.addPoint(
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0,
                maxh,
            )
            for angle in arc_angles
        ]

        arcs = [
            gmsh.model.geo.addCircleArc(
                rim_points[k],
                origin,
                rim_points[k + 1],
            )
            for k in range(n_arcs)
        ]

        upper_lip = gmsh.model.geo.addLine(origin, rim_points[0])
        lower_lip = gmsh.model.geo.addLine(rim_points[-1], origin)

        curve_loop = gmsh.model.geo.addCurveLoop([
            upper_lip,
            *arcs,
            lower_lip,
        ])

        surface = gmsh.model.geo.addPlaneSurface([
            curve_loop
        ])

        gmsh.model.geo.synchronize()

        # Boundary markers, matching those of the structured branch.
        gmsh.model.addPhysicalGroup(1, arcs, 1)
        gmsh.model.addPhysicalGroup(1, [upper_lip], 2)
        gmsh.model.addPhysicalGroup(1, [lower_lip], 3)

        # Physical marker for the 2D domain itself.
        gmsh.model.addPhysicalGroup(
            2,
            [surface],
            1,
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeMax",
            maxh,
        )

        gmsh.model.mesh.generate(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            msh_file = os.path.join(
                tmpdir,
                "pacman.msh",
            )

            gmsh.write(msh_file)

            return Mesh(
                msh_file,
                name=name,
                comm=comm,
                **kwargs,
            )

    finally:
        if already_running:
            gmsh.model.remove()
        else:
            gmsh.finalize()


# ---------------------------------------------------------------------------
# Function Spaces
# ---------------------------------------------------------------------------

def FEEC_B_N2_2d1k(mesh, degree):
    """
    Construct the full FEEC mixed function space for the B-formulation
    in 2D with k = 1.

    The corresponding discrete de Rham sequence is
        CG_{r+1}  --grad-->  N2curl_r,
    where N2curl denotes the Nedelec second-kind (full) H(curl)-conforming
    finite element space.

    Parameters:
    ----------
    mesh : Mesh, Computational mesh.
    degree : int, Polynomial degree r of the N2curl space.

    Returns:
    -------
    FunctionSpace, Mixed space W_h = N2curl_r x CG_{r+1}, for the vector field v and scalar field p, respectively.
    """
    return (
        FunctionSpace(mesh, "N2curl", degree)
        * FunctionSpace(mesh, "CG", degree + 1)
    )


def FEEC_B_N1_2d1k(mesh, degree):
    """
    Construct the trimmed FEEC mixed function space for the B-formulation
    in 2D with k = 1.

    The corresponding discrete de Rham sequence is
        CG_r  --grad-->  N1curl_r,
    where N1curl denotes the Nedelec first-kind (trimmed) H(curl)-conforming finite element space.

    Parameters:
    ----------
    mesh : Mesh, Computational mesh.
    degree : int, Polynomial degree r of the N1curl space.

    Returns:
    -------
    FunctionSpace, Mixed space W_h = N1curl_r x CG_r, for the vector field v and scalar field p, respectively.
    """
    return (
        FunctionSpace(mesh, "N1curl", degree)
        * FunctionSpace(mesh, "CG", degree)
    )


def FEEC_A_2d1k(mesh, degree):
    """
    Construct the FEEC function space for the A-formulation.
    In this formulation, A is a discrete 0-form and therefore belongs to

    Parameters:
    ----------
    mesh : Mesh, Computational mesh.
    degree : int, Polynomial degree of the CG space.

    Returns:
    -------
    FunctionSpace, H^1-conforming finite element space for the scalar field A.
    """
 

   # return FunctionSpace(mesh, 'CG',degree) * FunctionSpace(mesh, 'R', 0)

    return FunctionSpace(mesh, "CG", degree)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def cross2d(a, b):
    """
    Return the scalar 2D cross product of two planar vector fields.

    For

        a = (a_1, a_2),    b = (b_1, b_2),

    this is the z-component of the corresponding 3D cross product:

        (a x b)_z = a_1 b_2 - a_2 b_1.

    This scalar representation is the standard 2D proxy for the
    cross product of two planar vectors.

    Note that the operation is antisymmetric:

        cross2d(a, b) = -cross2d(b, a).

    This is important in the B-formulation, where the required ordering
    is cross2d(v, u), i.e. v x u.
    """
    return a[0] * b[1] - a[1] * b[0]

# ---------------------------------------------------------------------------
# Formulations
# ---------------------------------------------------------------------------

def B_formulation_2d1k(W, u=None, eps=Constant(1.0)):
    """
    Define the mixed 2D B-formulation for the generalized eigenvalue problem.

    Parameters:
    ----------
    W : FunctionSpace, Mixed function space for the vector field v and scalar field p.
    u : Function, optional, Given background/advection velocity field.
    eps : Constant, Diffusion/viscosity coefficient. This is just 1/Rm

    Returns:
    -------
    a : UFL form, Bilinear form defining the differential operator.
    m : UFL form, L2 mass bilinear form for the vector field v.
    """
    
    #v : vector-valued eigenfunction
    #p : scalar auxiliary/pressure variable
    (v, p) = TrialFunctions(W)

    # Test functions corresponding to v and p.
    (w, q) = TestFunctions(W)

    # Curl-curl (diffusion/viscous) contribution:
    a = eps * inner(curl(v), curl(w)) * dx

    # Add the advection/background-flow contribution when u is supplied:
    if u is not None:
        a += inner(cross2d(v, u), curl(w)) * dx

    # Mixed coupling between the vector field and scalar variable:
    # This provides the constraint/coupling between v and p.
    a += (-inner(grad(p), w) + inner(v, grad(q))) * dx

    # L2 mass bilinear form for the vector field.
    m = inner(v, w) * dx

    return a, m

def bcs_B_2d1k(W):
    """
    Creates standard Dirichlet boundary conditions for the mixed formulation (Formulation B).
    Assumes W.sub(0) is the vector space and W.sub(1) is the scalar multiplier space.
    """
    bc_v = DirichletBC(W.sub(0), Constant((0.0, 0.0)), 'on_boundary')
    bc_p = DirichletBC(W.sub(1), Constant(0.0), 'on_boundary')
    return [bc_v, bc_p]



def A_formulation_2d1k(W, u=None, eps=Constant(1.0)):
    """
    Constructs the bilinear forms for the 2D A-formulation eigenvalue problem (k=1).
    
    Parameters:
        W: A Firedrake mixed function space (V * R), where V is CG_r (H^1 proxy) and R is the Real space (degree 0) for the global multiplier.
        u: Optional advection velocity field.
        eps: Diffusion coefficient / scaling factor.
        
    Returns:
        a: The stiffness matrix form (LHS).
        m: The mass matrix form (RHS).
    """
    
    # # Unpack the trial and test functions from the mixed space W = CG * R.
    # # v corresponds to the scalar potential \omega in H^1(\Omega)
    # # t corresponds to the global multiplier in \mathbb{R}
    # v, t = TrialFunctions(W)
    
    # # w corresponds to the test function z in H^1(\Omega)
    # # s corresponds to the test function for the constraint
    # w, s = TestFunctions(W)
    
    # # Base stiffness matrix terms (LHS)
    # # 1. eps * inner(grad(v), grad(w)) * dx: The standard diffusion term \varepsilon(\nabla\omega, \nabla z)
    # # 2. - inner(t, w) * dx: The multiplier term -(t, z) that enters the main equation
    # # 3. - inner(v, s) * dx: Enforces the gauge constraint
    # #    A negative sign is used for the constraint to preserve matrix symmetry when u = 0.
    # a = eps * inner(grad(v), grad(w)) * dx - inner(t, w) * dx - inner(v, s) * dx
    
    # # Add advection (Lie derivative proxy) if a velocity field is provided
    # if u is not None:
    #     # inner(dot(u, grad(v)), w) * dx represents (\mathbf{u} \cdot \nabla\omega, z)
    #     a += inner(dot(u, grad(v)), w) * dx
        
    # # Mass matrix 
    # m = inner(v, w) * dx
    v = TrialFunction(W)
    w = TestFunction(W)

    a = eps * inner(grad(v), grad(w)) * dx
    if u is not None:
        a += inner(dot(u, grad(v)), w) * dx

    m = inner(v, w) * dx

    return a, m
    


def bcs_A_2d1k(W):
    """
    Creates standard Dirichlet boundary conditions for the mixed formulation (Formulation B).
    Assumes W.sub(0) is the vector space and W.sub(1) is the scalar multiplier space.
    """

    return []


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------


def make_eigensolver(a, m, bcs, n_evals, opts):
    prob = LinearEigenproblem(A=a, M=m, bcs=bcs)
    return LinearEigensolver(prob, n_evals=n_evals, solver_parameters=opts)


def solve_eigenpairs(solver, drop_harmonic=True, tol=1e-8):
    """
    Solves the eigenvalue problem and extracts the valid eigenvalues and eigenfunctions.
    
    Parameters:
        solver: The initialized eigenvalue solver object (e.g., a SLEPc wrapper).
        drop_harmonic: If True, filters out the zero eigenvalue(s) corresponding to the constant harmonic mode.
        tol: The numerical tolerance for determining if an eigenvalue is effectively zero.
        
    Returns:
        eig: A list of the computed eigenvalues.
        eigf: A list of the corresponding eigenfunctions (Firedrake Functions).
    """
    # Execute the eigensolver and get the number of converged eigenpairs
    nconv = solver.solve()
    
    eig = []
    eigf = []
    
    for i in range(nconv):
        # Extract the i-th eigenvalue
        lam = solver.eigenvalue(i)
        
        # an exact λ = 0 is removed here.
        if drop_harmonic and abs(lam) <= tol:
            continue
            
        # Store the valid eigenvalue
        eig.append(lam)
        
        # Extract and store the corresponding eigenfunction.
        eigf.append(solver.eigenfunction(i)) 
        
    return eig, eigf



# ---------------------------------------------------------------------------
# Winds
# ---------------------------------------------------------------------------

def get_wind(n, mesh=None, y0=None, L=np.pi):
    """
    W0  (0, 0)                  -- no wind
    W1  (1, 1)                  -- uniform, C^inf
    W3  (sign(y - y0), 0)       -- Heaviside shear, L^inf but not W^{1,inf}
    """
    if n == 0:
        return None
    elif n == 1:
        return as_vector([Constant(1.0), Constant(1.0)])
    elif n == 2:
        x, y = SpatialCoordinate(mesh)
        # FIX: Replaced math.sin with UFL's sin
        return as_vector([sin(y), sin(x)])
        
    elif n == 3:                                     # W3 in the table
        x, y = SpatialCoordinate(mesh)
        y0 = Constant(L / 2) if y0 is None else Constant(y0)
        return as_vector([conditional(gt(y, y0), 1.0, -1.0), Constant(0.0)])

    elif n == 4:
        x, y = SpatialCoordinate(mesh)
        # Calculate radial distance in Cartesian coordinates
        r = sqrt(x**2 + y**2)
        
        # 1. Use abs() to prevent negative numbers
        # 2. Add a tiny epsilon to prevent exact division by zero
        eps = Constant(1e-6) 
        
        # Apply the scalar multiplier to the (-y, x) vector
        return (1.0 / (abs(r - 1.5) + eps)**1.5) * as_vector([-y, x])