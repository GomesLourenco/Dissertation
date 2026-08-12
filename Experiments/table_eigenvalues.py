import numpy as np
import pandas as pd


def table_eigenvalues(eig_list, N_list, dof_list, exact_eigs=None,
                      drop_zero=True, zero_tol=1e-8, dim=2):
    """
    Generate a table of eigenvalues with inline convergence rates.

    Rates are measured against the mesh size h, NOT the cell count.  Under
    uniform refinement in `dim` dimensions the cell count grows by 2**dim
    while h halves, so dividing by log(N_ratio) reports the true rate divided
    by dim -- an O(h^2) method looks first order.  Here h ~ num_cells**(-1/dim).

    Two modes of operation:

      * exact_eigs supplied -> the rate comes from the true error |lam_h - lam|.

      * exact_eigs is None  -> nothing is treated as exact.  The rate is
        estimated from three successive levels using successive differences,

            p = log( |lam_1 - lam_2| / |lam_2 - lam_3| ) / log(h_ratio),

        and an Aitken extrapolation of the three finest levels is reported as
        the estimated limit lam*.  This is what you want when the answer is
        unknown; using the finest mesh as "exact" biases the last rate badly,
        because that reference carries error of the same order as the level
        just below it.

    The lambda = 0 harmonic mode of the A-formulation is removed by default.
    It is exact to machine precision, so its error ratio is 0/0, and leaving
    it in shifts every later row out of alignment with exact_eigs.

    Parameters
    ----------
    eig_list : list of lists   Eigenvalues computed at each mesh level.
    N_list   : list of int     Cell count at each level (mesh.num_cells()).
    dof_list : list of int     Degrees of freedom at each level.
    exact_eigs : list, optional  Exact eigenvalues, if known.
    drop_zero : bool           Remove the zero (harmonic) modes.
    zero_tol : float           Magnitude below which an eigenvalue counts as zero.
    dim : int                  Spatial dimension, for the h <-> num_cells relation.
    """
    assert len(eig_list) == len(N_list) == len(dof_list), "Input lists must match in length."

    # ---- normalise every run: strip zero modes, order consistently -------
    def clean(run):
        vals = [complex(v) for v in run]
        if drop_zero:
            vals = [v for v in vals if abs(v) > zero_tol]
        # sort by real part, then by -imag so conjugate pairs stay adjacent
        # with the +i member first.  This makes index k mean the same mode on
        # every mesh, which raw solver output does not.
        return sorted(vals, key=lambda z: (z.real, -z.imag))

    runs = [clean(r) for r in eig_list]
    n_lev = len(runs)
    max_eigs = min(10, max(len(r) for r in runs))

    def fmt(v):
        v = complex(v)
        if abs(v.imag) > zero_tol:
            return f"{v.real:.4f} + {v.imag:.4f}i"
        return f"{v.real:.4f}"

    # ---- reference values ------------------------------------------------
    if exact_eigs is not None:
        ref = clean(exact_eigs)
        ref_label = "Exact"
    else:
        ref = []
        for k in range(max_eigs):
            seq = [runs[j][k] for j in range(n_lev) if k < len(runs[j])]
            ref.append(_aitken(seq))
        ref_label = "λ* (est)"

    # ---- mesh size ratios between consecutive levels ---------------------
    h_ratio = [None] + [(N_list[j] / N_list[j - 1]) ** (1.0 / dim)
                        for j in range(1, n_lev)]

    data = {ref_label: [fmt(v) if v is not None else "-" for v in ref[:max_eigs]]}
    data[ref_label] += ["-"] * (max_eigs - len(data[ref_label]))

    for j in range(n_lev):
        col = []
        for k in range(max_eigs):
            if k >= len(runs[j]):
                col.append("-")
                continue

            val = runs[j][k]
            rate = None

            if exact_eigs is not None:
                # true error against a known answer
                if j >= 1 and k < len(runs[j - 1]) and k < len(ref) and h_ratio[j] > 1:
                    rate = _rate(abs(runs[j - 1][k] - ref[k]),
                                 abs(val - ref[k]), h_ratio[j], ref[k])
            else:
                # three-level estimate, needs no reference at all
                if j >= 2 and k < len(runs[j - 1]) and k < len(runs[j - 2]) and h_ratio[j] > 1:
                    rate = _rate(abs(runs[j - 2][k] - runs[j - 1][k]),
                                 abs(runs[j - 1][k] - val), h_ratio[j], val)

            col.append(fmt(val) + (f" ({rate:.1f})" if rate is not None else ""))

        data[f"N = {N_list[j]}"] = col

    df = pd.DataFrame(data)
    df.index = [f"{i + 1}" for i in range(max_eigs)]
    df.loc["DOF"] = ["-"] + [str(d) for d in dof_list]
    return df


def _rate(err_prev, err_curr, h_ratio, scale):
    """log-ratio convergence rate, or None if the numbers cannot support one."""
    floor = 1e-12 * max(1.0, abs(scale))
    if err_prev <= floor or err_curr <= floor:
        return None                      # converged to round-off: rate is meaningless
    return np.log(err_prev / err_curr) / np.log(h_ratio)


def _aitken(seq):
    """Aitken/Richardson limit from the last three entries of seq."""
    if len(seq) < 3:
        return seq[-1] if seq else None
    a, b, c = seq[-3], seq[-2], seq[-1]
    d1, d2 = b - a, c - b
    den = d2 - d1
    if abs(den) < 1e-14 * max(1.0, abs(c)):
        return c
    return c - d2 * d2 / den
