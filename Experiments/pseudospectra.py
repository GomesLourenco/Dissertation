"""
epsilon-pseudospectra of the FEEC magnetic advection-diffusion matrix pencils.

For a generalised eigenvalue problem  A x = lambda B x  arising from a mixed
finite element discretisation, the mass matrix B is singular (it has a zero
block for the Lagrange multiplier).  The epsilon-pseudospectrum of the pencil
is then taken in the sense of van Dorsselaer,

    sigma_eps(A, B) = { z in C : ||(zB - A)^{-1}|| >= 1/eps }
                    = { z in C : sigma_min(zB - A) <= eps },

with sigma_min the smallest singular value.

Evaluating sigma_min(zB - A) directly on a grid is far too expensive for a
large sparse pencil, so the problem is compressed onto the invariant subspace
spanned by the eigenvectors nearest a target.  With V an orthonormal basis of
that subspace,

    A_r = V^* A V,      B_r = V^* B V,

and sigma_min(z B_r - A_r) is evaluated on the grid instead.  Because the
subspace is (approximately) invariant, this is an *inner* approximation: the
plotted pseudospectrum is contained in the true one, and it converges as `nev`
grows.  Always check convergence by re-running with a larger `nev`.

Two choices of inner product are available (see `norm`):

  norm="mass"       V is orthonormal in the mass (L2) inner product, V^* B V = I.
                    sigma_min then measures the resolvent in the L2 norm of the
                    field, which is the mesh-independent, physically meaningful
                    quantity for a PDE operator.  This is the default.

  norm="euclidean"  V is orthonormal in the Euclidean inner product on
                    coefficient vectors.  This is the literal matrix-pencil
                    definition, but sigma_min then inherits the O(h^d) scaling
                    of the mass matrix and is not comparable across meshes.

A note on the two-sided variant
-------------------------------
Compressing with distinct left and right *eigenvector* bases (A_r = W^* A V,
B_r = W^* B V) does not work, and silently returns numerical noise.  Left and
right eigenvectors of a pencil are B-biorthogonal, so W^* B V is diagonal with
entries w_i^* B v_i = 1 / kappa(lambda_i).  For a strongly non-normal operator
those are at the level of machine precision, B_r is numerically singular, and
z B_r - A_r is then singular for *every* z, giving sigma_min ~ 1e-16 everywhere.
The two-sided Krylov-Schur method of Zwaan & Hochstenbach uses the two-sided
Krylov *search spaces*, not the eigenvector spans, and is a different object.
`two_sided=True` is kept here for comparison, and reports cond(B_r) so the
breakdown is visible rather than silent.
"""

import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm

import firedrake as fd
from petsc4py import PETSc
from slepc4py import SLEPc

__all__ = ["pseudospectrum", "plot_pseudospectrum"]

_COMPLEX = np.issubdtype(PETSc.ScalarType, np.complexfloating)


# ---------------------------------------------------------------------------
# PETSc helpers
# ---------------------------------------------------------------------------



def _columns_to_array(vecs):
    """Stack a list of (vr, vi) PETSc vector pairs into a dense complex array."""
    cols = []
    for vr, vi in vecs:
        col = np.asarray(vr.getArray()).astype(complex)
        if not _COMPLEX and vi is not None:
            col = col + 1j * np.asarray(vi.getArray())
        cols.append(col)
    return np.column_stack(cols)


def _apply(mat, dense):
    """
    Computes `mat @ dense` (matrix-matrix multiplication).
    `mat` is a large sparse PETSc matrix, and `dense` is a dense complex NumPy array.
    """
    
    # createVecRight() creates an empty PETSc vector sized for the matrix input (domain)
    x = mat.createVecRight()
    # createVecLeft() creates an empty PETSc vector sized for the matrix output (range)
    y = mat.createVecLeft()
    
    # Pre-allocate an empty complex NumPy array to store our final results.
    # It has the same number of rows as the matrix, and same number of columns as `dense`.
    out = np.empty((mat.getSize()[0], dense.shape[1]), dtype=complex)
    
    # PETSc can't multiply a matrix by a block of vectors all at once, 
    # so we must loop through `dense` one column at a time.
    for j in range(dense.shape[1]):
        
        if _COMPLEX:
            # SCENARIO A: PETSc is compiled with Complex number support.
            # This is easy. We just dump the complex column into our input vector `x`...
            x.setArray(dense[:, j])
            # ... tell PETSc to multiply it (y = mat @ x) ...
            mat.mult(x, y)
            # ... and save the result into our output array.
            out[:, j] = y.getArray()
            
        else:
            # SCENARIO B: PETSc is compiled in Real mode and will crash if we give it complex numbers.
            # We must use the mathematical rule: M @ (u + iv) = (M @ u) + i(M @ v).
            
            # Step 1: Extract only the Real part of the column, and multiply it.
            x.setArray(dense[:, j].real)
            mat.mult(x, y)
            
            # We MUST use `copy=True` here. PETSc vectors share memory with NumPy, 
            # so if we don't copy it, the next step will overwrite our real result!
            re = np.array(y.getArray(), copy=True)
            
            # Step 2: Extract only the Imaginary part of the column, and multiply it.
            x.setArray(dense[:, j].imag)
            mat.mult(x, y)
            
            # Step 3: Stitch them back together into a single complex column and save it.
            out[:, j] = re + 1j * np.asarray(y.getArray())
            
    return out

def _transpose(mat):
    """A genuine transposed copy.

    petsc4py's Mat.transpose() transposes *in place* when called with no `out`
    argument and returns the same object, so `A_t = A.transpose()` silently
    overwrites A and leaves A_t aliased to it.
    """
    mat_t = mat.copy()
    mat_t.transpose()
    if _COMPLEX:
        mat_t.conjugate()
    return mat_t


# ---------------------------------------------------------------------------
# Operators and eigenpairs
# ---------------------------------------------------------------------------

def _default_parameters(target):
    return {
        "eps_gen_non_hermitian": None,
        "eps_target": target,
        "eps_target_magnitude": None,
        "st_type": "sinvert",
        "st_pc_type": "lu",
        "st_pc_factor_mat_solver_type": "mumps",
        "st_mat_mumps_icntl_14": 500,
    }


def _merge_parameters(target, solver_parameters):
    """Caller options on top of the defaults, but `target` always wins: the
    subspace has to sit where the pseudospectrum window is."""
    params = dict(_default_parameters(target))
    if solver_parameters:
        params.update(solver_parameters)
    for key in ("eps_smallest_magnitude", "eps_largest_magnitude",
                "eps_smallest_real", "eps_largest_real"):
        params.pop(key, None)
    params["eps_target"] = target
    params["eps_target_magnitude"] = None
    return params


def _build_solver(A, M, bcs, nev, target, solver_parameters):
    """Assemble the pencil through Firedrake so the boundary conditions are
    applied exactly as they are in the rest of the eigenvalue study."""
    problem = fd.LinearEigenproblem(A=A, M=M, bcs=bcs)
    return fd.LinearEigensolver(
        problem, n_evals=nev,
        solver_parameters=_merge_parameters(target, solver_parameters),
    )
def _resolve(A, M, bcs, nev, target, solver_parameters):
    """
    Standardize the various Firedrake/PETSc input formats into raw PETSc matrices.

    Inputs:
    -------
    A                 : The primary input. Can be a Firedrake LinearEigensolver, 
                        LinearEigenproblem, assembled Matrix, raw PETSc.Mat, or UFL form.
    M                 : The mass matrix or mass form (required if A is a matrix or form).
    bcs               : Boundary conditions (only allowed if A is a raw UFL form).
    nev               : Number of eigenvalues to compute.
    target            : The target shift (the exact center of the pseudospectrum plot).
    solver_parameters : Optional dictionary of PETSc/SLEPc solver options.

    Outputs:
    --------
    A_mat : The raw, sparse PETSc stiffness matrix (C-level object).
    B_mat : The raw, sparse PETSc mass matrix (C-level object).
    eps   : The SLEPc Eigenvalue Problem Solver object. 
            Returns `None` if we only extracted matrices (forcing the main script 
            to build a new solver), or returns the actual solved `eps` object if we 
            just built and solved one perfectly centered on the target.
    """
    
    # 1. An already-built firedrake LinearEigensolver.
    # If the input A has the attribute 'es', it means the user passed a Firedrake LinearEigensolver
    if hasattr(A, "es"): 
        try:
            # In that case we try to extract the underlying PETSc matrices A and B
            A_mat, B_mat = A.es.getOperators() 
        except PETSc.Error: 
            # However, if the user hasn't called .solve() yet, the matrices aren't built in memory.
            # So in this case we force the solver to build them by calling solve...
            A.solve() 
            # ...and then extract the matrices. 
            A_mat, B_mat = A.es.getOperators() 
            
        # We return None for `eps` because the user's solver was likely configured to search
        # for eigenvalues near a different target. We throw away their eigenpairs.
        return A_mat, B_mat, None 

    # 2. Assembled firedrake matrices or raw PETSc matrices
    # If A has a 'petscmat' attribute (Firedrake matrix) or is a raw C-level PETSc matrix
    if hasattr(A, "petscmat") or isinstance(A, PETSc.Mat):
        if M is None: 
            # If A is just a matrix, the user must explicitly supply the mass matrix M as well
            raise ValueError("a mass matrix M must be supplied alongside the stiffness matrix A")
        if bcs: 
            # If boundary conditions are passed here, we MUST raise an error. Boundary conditions 
            #  alter the matrix during assembly; they cannot be applied after the fact.
            raise ValueError(
                "boundary conditions cannot be applied to already-assembled matrices; "
                "pass the bilinear forms (or a LinearEigenproblem) together with bcs"
            )
        # Strip away any Firedrake Python wrappers to get the raw C-level PETSc matrices
        A_mat = A.petscmat if hasattr(A, "petscmat") else A
        B_mat = M.petscmat if hasattr(M, "petscmat") else M
        
        # We only have matrices, no solver yet, so return None for `eps`
        return A_mat, B_mat, None

    # 3. A firedrake LinearEigenproblem
    # If A has 'M' (mass form) and 'bcs' (boundary conditions) attributes, it is a problem container object
    if hasattr(A, "M") and hasattr(A, "bcs"):
        # Build a brand new eigensolver around this problem. 
        # Crucially, we merge the parameters to ensure this solver searches exactly at our `target`
        solver = fd.LinearEigensolver(
            A, n_evals=nev,
            solver_parameters=_merge_parameters(target, solver_parameters),
        )
        # Force the solver to assemble the matrices and compute the eigenpairs
        solver.solve()
        
        # Unpack the matrices, AND return the SLEPc solver (`solver.es`). 
        # Because we built this solver specifically for our target, its eigenpairs can be reused!
        return (*solver.es.getOperators(), solver.es)

    # 4. Raw UFL Bilinear forms
    # If A has 'arguments' (test and trial functions), it is raw, un-assembled UFL math
    if hasattr(A, "arguments"):
        if M is None:
            # We strictly need the mass form to build the generalized eigenvalue problem
            raise ValueError("a mass form M must be supplied alongside the stiffness form A")
            
        # Use our helper function to package the forms and BCs, and build a targeted eigensolver
        solver = _build_solver(A, M, bcs, nev, target, solver_parameters)
        # Assemble matrices and compute eigenpairs at the target
        solver.solve()
        
        # Return matrices and the active solver so we can reuse the computed eigenpairs
        return (*solver.es.getOperators(), solver.es)

    # 5. Fallback
    # If the user passed a string, a float, or some other object we don't recognize, fail safely.
    raise TypeError(f"cannot interpret {type(A).__name__} as a stiffness operator")



    
def _solve_eps(A_mat, B_mat, nev, target, solver_parameters, desc):
    """Shift-and-invert solve for the eigenpairs nearest `target`."""
    eps = SLEPc.EPS().create(comm=A_mat.getComm())
    prefix = f"pseudospectrum_{id(eps):x}_"
    eps.setOptionsPrefix(prefix)
    eps.setOperators(A_mat, B_mat)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(nev=nev)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    opts = PETSc.Options()
    params = _merge_parameters(target, solver_parameters)
    for key, value in params.items():
        opts[prefix + key] = value
    eps.setFromOptions()
    for key in params:
        del opts[prefix + key]

    # Set after setFromOptions so the requested target is authoritative.
    # Selecting eigenvalues near the shift needs setTarget together with
    # TARGET_MAGNITUDE; setting only the ST shift leaves EPS sorting by
    # LARGEST_MAGNITUDE, which is what the original code did.
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError(
            f"no {desc} eigenpairs converged near target {target}; "
            "try a different target or a larger nev"
        )
    return eps, nconv


def _extract(eps, mat, nconv, finite_tol):
    """
    Extracts eigenvalues and eigenvectors from the solver, removing infinite/spurious 
    pairs caused by singular mass matrices.
    """
    values, vecs = [], []
    
    # Loop over every eigenvalue the solver successfully found
    for i in range(nconv):
        # Ask the solver for the i-th eigenvalue
        lam = eps.getEigenvalue(i)
        
        # Check if the eigenvalue is mathematically infinite (NaN, Inf, or bigger than our tolerance).
        if not np.isfinite(lam) or abs(lam) > finite_tol:
            continue          # Skip this eigenpair entirely and move to the next one
            
        # Allocate an empty PETSc vector with the exact same memory layout as our matrices
        # to hold the eigenvector.
        vr = mat.createVecRight()
        
        # Handle how PETSc stores complex numbers depending on how the user installed it:
        if _COMPLEX:
            # If PETSc was compiled with complex numbers, it stores the entire 
            # complex eigenvector inside a single vector (`vr`).
            vi = None
            eps.getEigenvector(i, vr)
        else:
            # If PETSc was compiled with real numbers, we must allocate a second 
            # empty vector to hold the imaginary part of the eigenvector.
            vi = mat.createVecRight()
            # SLEPc will put the real part in `vr` and the imaginary part in `vi`.
            eps.getEigenvector(i, vr, vi)
            
        # Save the valid eigenvalue, forcing it to be a Python complex type just to be safe
        values.append(complex(lam))
        
        # Save the eigenvector as a tuple of (real_vector, imaginary_vector)
        vecs.append((vr, vi))
        
    # If the solver found eigenvalues, but our filter threw EVERY single one of them away...
    if not values:
        raise RuntimeError("every converged eigenvalue was infinite or spurious")
        
    # Return the clean list of eigenvalues as a NumPy array, and the list of eigenvector tuples
    return np.array(values), vecs


# ---------------------------------------------------------------------------
# Subspace compression
# ---------------------------------------------------------------------------
def _orthonormalise(X, gram=None, rtol=1e-10):
    """
    Creates an orthonormal basis from the raw eigenvectors (X), while discarding 
    directions that are too numerically similar (rank truncation).
    """
    
    # ---------------------------------------------------------
    # PATH 1: Standard Euclidean Norm (gram is None)
    # We want a basis U such that U^* U = I
    # ---------------------------------------------------------
    if gram is None:
        # Compute the Singular Value Decomposition: X = U * S * V^*
        # 'U' contains the orthonormal basis vectors. 
        # 's' contains the singular values (importance of each direction).
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        
        # TRUNCATION: Keep only the directions where the singular value is 
        # significantly larger than numerical noise (relative to the largest one, s[0]).
        keep = s > s[0] * rtol
        
        # Return the filtered basis and the new dimension size
        return U[:, keep], int(keep.sum())

    # ---------------------------------------------------------
    # PATH 2: Mass (L2) Norm (gram = B @ X)
    # We want a basis V such that V^* B V = I
    # ---------------------------------------------------------
    
    # Step 1: Compute the Gram matrix: G = X^* B X
    # (Since `gram` was precomputed as B @ X, we just do X^* @ gram)
    G = X.conj().T @ gram
    
    # Step 2: Force perfect symmetry. 
    # Floating point math can leave tiny asymmetric errors (like 1e-17). 
    # The eigenvalue solver below requires a perfectly Hermitian (symmetric) matrix.
    G = 0.5 * (G + G.conj().T) #this should be very similar to the original G
    
    # Step 3: Compute the eigenvalues (w) and eigenvectors (U) of the Gram matrix.
    w, U = np.linalg.eigh(G)
    
    # `eigh` returns them in ascending order. We want the most important 
    # (largest) directions first, so we reverse the sort order.
    order = np.argsort(w)[::-1]          
    w, U = w[order], U[:, order]         
    
    # Step 4: TRUNCATION
    # Identify the significant eigenvalues of the Gram matrix. 
    # If an eigenvalue is tiny, the original eigenvectors were practically parallel.
    keep = w > w[0] * rtol
    
    # Step 5: Construct the new B-orthonormal basis.
    # To make V^* B V = I, we rotate the raw vectors X using U, 
    # and then scale them down by the square root of the eigenvalues (w).
    # New Basis = X @ (U * w^{-1/2})
    final_basis = X @ (U[:, keep] / np.sqrt(w[keep]))
    
    return final_basis, int(keep.sum())

def _sigma_min_grid(A_r, B_r, Z, chunk_elems=4_000_000):
    """
    Computes the smallest singular value sigma_min(z B_r - A_r) for every point z 
    in the complex grid Z, processing them in memory-safe batches.
    """
    # Flatten the 2D grid of complex numbers (e.g., 200x200) into a 1D list (40,000)
    # so we can easily slice it into batches.
    z = Z.ravel()
    
    # Pre-allocate an empty 1D array to store our final 40,000 singular values.
    out = np.empty(z.size)
    
    # Get the dimension of our tiny compressed matrices (e.g., m = 60).
    m = A_r.shape[0]
    
    # A single matrix takes up (m * m) elements. We want our batch tensor to have 
    # at most `chunk_elems` (default 4 million) elements so we don't run out of RAM.
    # `step` is the maximum number of grid points `z` we can process at one time.
    step = max(1, int(chunk_elems // max(m * m, 1)))
    
    # Loop over the 40,000 points, jumping forward by `step` each time.
    for k in range(0, z.size, step):
        
        # Grab the current batch of complex numbers (e.g., 1,000 points at a time).
        blk = z[k:k + step]
        
        # We need to build the matrix (z * B_r - A_r) for every 'z' in our batch.
        # Instead of a loop, we use NumPy broadcasting (expanding dimensions):
        #   - blk[:, None, None]  turns our 1D list of z's into a 3D column (batch_size, 1, 1)
        #   - B_r[None, :, :]     turns our 2D matrix into a 3D layer (1, m, m)
        # Multiplying them stacks copies of B_r, each multiplied by a different z.
        # Finally, we subtract A_r from every layer in the stack.
        # The result `Mz` is a 3D tensor of shape (batch_size, m, m).
        Mz = blk[:, None, None] * B_r[None, :, :] - A_r[None, :, :]
        
        # Compute the Singular Value Decomposition (SVD) for the entire batch at once.
        #   - `compute_uv=False`: Tells NumPy we only want the singular values, 
        #      saving massive computation time by not calculating the U and V vectors.
        #   - `[:, -1]`: NumPy sorts singular values from largest to smallest. 
        #      This extracts the LAST value (-1), which is sigma_min, for every matrix.
        out[k:k + step] = np.linalg.svd(Mz, compute_uv=False)[:, -1]
        
    # Reshape the flat 1D array of 40,000 answers back into the 
    # original 2D grid shape (200x200) so matplotlib can plot it as an image.
    return out.reshape(Z.shape)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pseudospectrum(A, M=None, target=0.0, real_range=(-1.0, 1.0),
                   imag_range=(-1.0, 1.0), *, bcs=None, nev=60, res=200,
                   norm="mass", rtol=1e-10,
                   finite_tol=1e10, solver_parameters=None, verbose=True):
    """Compute the pseudospectrum of the pencil (A, B) over a window.

    Parameters
    ----------
    A, M : bilinear forms, assembled matrices, a LinearEigenproblem, or a
        LinearEigensolver.  Passing the forms together with `bcs` (or passing a
        problem/solver directly) is what keeps the pencil consistent with the
        eigenvalues computed elsewhere in the study.
    target : centre of the shift-and-invert transformation.
    real_range, imag_range : window in the complex plane.
    bcs : boundary conditions, applied when forms are given.
    nev : number of eigenpairs defining the reduced subspace.  Increase it
        until the contours stop moving.
    res : grid points per direction.
    norm : "mass" (L2, mesh-independent) or "euclidean" (literal pencil form).
    two_sided : use left and right eigenvector bases.  Kept for comparison;
        see the module docstring for why it degenerates.

    Returns
    -------
    dict with keys X, Y, sigma, eigenvalues, ritz, dim, cond_Br, eig_error.
    """
    if norm not in ("mass", "euclidean"):
        raise ValueError("norm must be 'mass' or 'euclidean'")

    A_mat, B_mat, eps = _resolve(A, M, bcs, nev, target, solver_parameters) #this the raw, assembled PETSc stiffness and mass matrices
    # it also returns thee SLEPc Eigenvalue Problem Solver object if _resolve had to build a new one centered on your target, otherwise it 
    # returns None.

    if verbose:
        print(f"pencil size {A_mat.getSize()[0]}, solving for {nev} eigenpairs "
              f"near {target} ...") 

    if eps is None or eps.getConverged() == 0: 
    # Check if we need to find the eigenvalues.
    if eps is None or eps.getConverged() == 0: 
        # Build a fresh SLEPc solver using Shift-and-Invert Krylov-Schur.
        eps, nconv = _solve_eps(A_mat, B_mat, nev, target, solver_parameters, "right")
        
    else:
        # If we reach here, `_resolve` already handed us a perfectly good solver 
        # that successfully found eigenvalues exactly at our target window.
        nconv = eps.getConverged()

    values, vecs = _extract(eps, A_mat, nconv, finite_tol) #if needed to get the eigenvaleus and eigenvectors 
    V_raw = _columns_to_array(vecs)  # just make a  dense complex array of eigenvectors
    
    if verbose:
        print(f"  {nconv} converged, {len(values)} finite eigenvalues retained")


    # If the user requested the mesh-independent PDE "mass" norm, we must make 
    # the vectors orthogonal with respect to the mass matrix (V^* B V = I).
    # To prep for this, we use `_apply` to compute the matrix-matrix product 
    # (B_mat @ V_raw). If using the standard Euclidean norm, we just pass None.
    gram = _apply(B_mat, V_raw) if norm == "mass" else None

    # Pass the raw, highly collinear eigenvectors (and the B-matrix action, if any)
    # into our linear algebra filter. This function will:
    #   1. Rotate and scale the vectors so they are perfectly perpendicular.
    #   2. Throw away (truncate) redundant directions that are pure numerical noise.
    # It returns the clean, mathematically stable basis (V) and its new, 
    # potentially smaller dimension size (dim_v).
    V, dim_v = _orthonormalise(V_raw, gram=gram, rtol=rtol)


    A_r = V.conj().T @ _apply(A_mat, V)
    B_r = V.conj().T @ _apply(B_mat, V)

    # Consistency check: the reduced pencil must reproduce the eigenvalues it
    # was built from.  If it does not, the compression has broken down and the
    # sigma_min field is meaningless.
    cond_Br = np.linalg.cond(B_r)


    # We ask Scipy to solve the generalized eigenvalue problem for our TINY matrices.
    # The resulting eigenvalues are called "Ritz values".
    ritz = sla.eig(A_r, B_r, right=False)
    ritz = ritz[np.isfinite(ritz)]

    # We don't expect the tiny matrices to perfectly reproduce all original eigenvalues.
    # The ones on the edges of the subspace will be inaccurate. 
    # But the ones closest to our exact `target` should match.
    # Here, we sort the original SLEPc eigenvalues by distance to the target, 
    # and take the closest 33% of them as our "reference" set.
    ref = values[np.argsort(np.abs(values - target))][:max(1, dim // 3)]

    # For every reference eigenvalue, we find the closest Ritz value, compute the 
    # relative error between them, and take the worst-case (maximum) error.
    eig_error = max(np.min(np.abs(ritz - l)) / max(abs(l), 1e-30) for l in ref) if ritz.size else np.inf

    if verbose:
        print(f"  reduced dimension {dim}, cond(B_r) = {cond_Br:.2e}, "
              f"eigenvalue reproduction error = {eig_error:.2e}")

    # If the worst-case error is worse than 1e-6, or the matrices are highly unstable,
    # the code prints a massive warning telling you not to trust the resulting plot.
    if eig_error > 1e-6 or cond_Br > 1e12:
        print("  WARNING: the reduced pencil does not faithfully represent the "
              "original one (ill-conditioned B_r or eigenvalues not reproduced). "
              "The pseudospectrum below should not be trusted."
              + (" Try two_sided=False." if two_sided else ""))

    x = np.linspace(real_range[0], real_range[1], res)
    y = np.linspace(imag_range[0], imag_range[1], res)
    X, Y = np.meshgrid(x, y)
    if verbose:
        print(f"  evaluating sigma_min on a {res}x{res} grid ...")
    sigma = _sigma_min_grid(A_r, B_r, X + 1j * Y)

    return {"X": X, "Y": Y, "sigma": sigma, "eigenvalues": values,
            "ritz": ritz, "dim": dim, "cond_Br": cond_Br,
            "eig_error": eig_error, "norm": norm}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _decade_levels(log_sigma, n_levels):
    """Contour levels on a round grid of powers of ten.

    Prefers whole decades; subdivides only when the data span too few decades
    to give roughly `n_levels` contours.  Returns log10 values.
    """
    lo, hi = float(np.min(log_sigma)), float(np.max(log_sigma))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.array([lo - 1.0, lo, lo + 1.0])
    steps = (1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.025, 0.02, 0.01,
             0.005, 0.0025, 0.002, 0.001)
    for step in steps:
        levels = np.arange(np.floor(lo / step) * step,
                           np.ceil(hi / step) * step + 0.5 * step, step)
        if len(levels) >= n_levels:
            return levels
    # Too narrow a range to reach n_levels on any round step: keep the round
    # labels and accept fewer contours rather than falling back to a linspace
    # with unreadable exponents.
    return levels


def _line_levels(log_sigma, n_levels):
    """log10 levels for the overlaid contour lines.

    Whole decades whenever the field spans enough of them, so that a labelled
    line reads 10^-1, 10^0, ...; a round subdivision otherwise.
    """
    lo_dec = int(np.floor(np.min(log_sigma)))
    hi_dec = int(np.ceil(np.max(log_sigma)))
    decades = np.arange(lo_dec, hi_dec + 1, dtype=float)
    if len(decades) >= 4:
        step = max(1, int(np.ceil(len(decades) / max(n_levels, 1))))
        return decades[::step]
    return _decade_levels(log_sigma, n_levels)


def _power_of_ten(value, _pos=None):
    """Render a log10 value as a power of ten: -1.0 -> $10^{-1}$."""
    return rf"$10^{{{value:g}}}$"


def plot_pseudospectrum(A, M=None, target=0.0, real_range=(-1.0, 1.0),
                        imag_range=(-1.0, 1.0), res=200, nev=60, *,
                        bcs=None, norm="mass", two_sided=False,
                        n_levels=13, levels=None, ax=None, title=None,
                        solver_parameters=None, verbose=True, **kwargs):
    """Compute and plot the pseudospectrum.  Returns (result, ax)."""
    kwargs.pop("interp_factor", None)   # removed: sigma_min is now evaluated
                                        # directly on the plotting grid
    if kwargs:
        raise TypeError(f"unexpected arguments: {sorted(kwargs)}")

    result = pseudospectrum(
        A, M, target, real_range, imag_range, bcs=bcs, nev=nev, res=res,
        norm=norm, two_sided=two_sided, solver_parameters=solver_parameters,
        verbose=verbose,
    )

    sigma = np.maximum(result["sigma"], np.finfo(float).tiny)
    log_sigma = np.log10(sigma)

    # Snap the colour range to whole decades so every colourbar tick is an
    # integer power of ten.
    lo_dec = int(np.floor(log_sigma.min()))
    hi_dec = int(np.ceil(log_sigma.max()))
    colour_norm = LogNorm(vmin=10.0 ** lo_dec, vmax=10.0 ** hi_dec)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    # The field itself is drawn continuously (Gouraud-shaded, no banding);
    # only the overlaid lines are discrete.
    field = ax.pcolormesh(result["X"], result["Y"], sigma, norm=colour_norm,
                          cmap="viridis_r", shading="gouraud", rasterized=True)

    log_levels = (_line_levels(log_sigma, n_levels) if levels is None
                  else np.log10(np.asarray(levels, dtype=float)))
    lines = ax.contour(result["X"], result["Y"], log_sigma, levels=log_levels,
                       colors="k", linewidths=0.5, alpha=0.6)
    ax.clabel(lines, inline=True, fontsize=7, fmt=_power_of_ten)

    # Ticks are epsilon itself at integer powers of ten, not log10(epsilon).
    cbar = plt.colorbar(
        field, ax=ax,
        ticks=mticker.LogLocator(base=10.0, subs=(1.0,),
                                 numticks=max(hi_dec - lo_dec + 1, 2)),
    )
    cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    cbar.set_label(r"$\varepsilon$")

    ev = result["eigenvalues"]
    ax.plot(ev.real, ev.imag, "r.", markersize=7, label="eigenvalues")

    norm_label = "$L^2$" if result["norm"] == "mass" else "Euclidean"
    ax.set_title(title or
                 rf"$\varepsilon$-pseudospectrum ({norm_label} norm), "
                 rf"reduced dimension {result['dim']}")
    ax.set_xlabel(r"$\mathrm{Re}\,z$")
    ax.set_ylabel(r"$\mathrm{Im}\,z$")
    ax.set_xlim(real_range)
    ax.set_ylim(imag_range)
    ax.legend(loc="upper right")
    return result, ax
