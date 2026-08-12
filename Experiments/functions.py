import numpy as np
from firedrake import *
from netgen.occ import *
from ngsolve.webgui import Draw



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