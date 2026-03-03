import numpy as np
import casadi as ca

from simple_osqp import SimpleOSQP, OSQPSettings


def build_problem():
    # P = diag([2, 2, 2, 1])
    P = np.array([
        [2.0, 0,   0,   0],
        [0,   2.0, 0,   0],
        [0,   0,   2.0, 0],
        [0,   0,   0,   1.0],
    ])
    q = np.array([1.0, -2.0, 1.0, 2.0])

    # A_eq x = b_eq
    A_eq = np.array([
        [1.0, 1.0,  0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
    ])
    b_eq = np.array([8.0, -7.0])

    # G_ineq x <= h_ineq
    G = np.array([
        [ 1.0, 0.0, 0.0, 0.0],  #  x1 <= 15.5
        [-1.0, 0.0, 0.0, 0.0],  # -x1 <= -15  => x1 >= 15
        [ 0.0, 1.0, 0.0, 0.0],  #  x2 <= 15
    ])
    h = np.array([15.5, -15.0, 15.0])

    # l <= A x <= u
    A = np.vstack([A_eq, G])
    l = np.concatenate([b_eq, np.full(h.shape, -np.inf)])
    u = np.concatenate([b_eq, h])

    return P, q, A, l, u


def solve_with_simple_osqp(P, q, A, l, u, settings: OSQPSettings):
    solver = SimpleOSQP(P, q, A, l, u, settings)
    solver.setup()
    x, z, y, info = solver.solve()
    return x, z, y, info


def solve_with_casadi_ipopt(P, q, A, l, u):
    nx = P.shape[0]
    x = ca.SX.sym("x", nx)

    obj = 0.5 * ca.mtimes([x.T, P, x]) + ca.dot(q, x)
    g = ca.mtimes(A, x)

    prob = {"x": x, "f": obj, "g": g}
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-12,
        "ipopt.acceptable_tol": 1e-12,
    }
    solver = ca.nlpsol("solver", "ipopt", prob, opts)
    res = solver(lbg=l, ubg=u)
    return np.array(res["x"]).reshape(-1)


def benchmark_once(verbose_iterations: bool = True):
    P, q, A, l, u = build_problem()

    settings = OSQPSettings(
        rho=1.0,
        sigma=1e-6,
        alpha=1.78,
        max_iter=5000,
        eps_abs=1e-8,
        verbose=verbose_iterations,
        log_every=1,
        store_history=True,
    )

    x_osqp, _, _, info = solve_with_simple_osqp(P, q, A, l, u, settings)
    x_ipopt = solve_with_casadi_ipopt(P, q, A, l, u)

    diff = np.linalg.norm(x_osqp - x_ipopt)

    print("\nResult:")
    print("  status        :", info["status"])
    print("  iterations    :", info["iter"])
    print("  ||r_p||_inf   :", f"{info['prim_res_inf']:.3e}")
    print("  ||r_d||_inf   :", f"{info['dual_res_inf']:.3e}")
    print("  x_osqp        :", x_osqp)
    print("  x_ipopt       :", x_ipopt)
    print("  ||x_osqp-x*|| :", f"{diff:.3e}")


def alpha_sweep():
    P, q, A, l, u = build_problem()

    alphas = np.round(np.arange(1.5, 1.9, 0.01), 2)

    print("\nalpha sweep: iterations to converge")
    print(" alpha      iters     status     ||r_p||_inf    ||r_d||_inf")
    for a in alphas:
        settings = OSQPSettings(
            rho=1.0,
            sigma=1e-6,
            alpha=float(a),
            max_iter=5000,
            eps_abs=1e-8,
            verbose=False,
            store_history=False,
        )

        _, _, _, info = solve_with_simple_osqp(P, q, A, l, u, settings)
        print(
            f"{a:5.2f}   {info['iter']:8d}   {info['status']:>8s}   "
            f"{info['prim_res_inf']:11.3e}   {info['dual_res_inf']:11.3e}"
        )


if __name__ == "__main__":
    benchmark_once(verbose_iterations=True)
    alpha_sweep()
