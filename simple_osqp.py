import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class OSQPSettings:
    rho: float = 1.0
    sigma: float = 1e-6
    alpha: float = 1.6                # 1 < alpha < 2
    max_iter: int = 10000
    eps_abs: float = 1e-12
    eps_rel: float = 1e-4             # not used), for future extension
    verbose: bool = False
    log_every: int = 1               
    store_history: bool = True       


class SimpleOSQP:
    """
    Minimal OSQP-style ADMM splitting for:
        minimize    0.5 x^T P x + q^T x
        subject to  l <= A x <= u

    Variables:
        x : primal
        z : auxiliary, z = A x (projected into C := [l, u])
        y : dual (nu in your notes)

    Iteration structure matches OSQP (14)-(19):
        (x~, gamma) from KKT solve
        z~ recovery
        over-relaxation: x_hat, z_hat
        z := Pi_C(z_hat + rho^{-1} y)
        y := y + rho (z_hat - z)
    """

    def __init__(self, P, q, A, l, u, settings: Optional[OSQPSettings] = None):
        self.P = spa.csc_matrix(P)
        self.q = np.asarray(q, dtype=float).reshape(-1)

        self.A = spa.csc_matrix(A)
        self.l = np.asarray(l, dtype=float).reshape(-1)
        self.u = np.asarray(u, dtype=float).reshape(-1)

        self.s = settings if settings is not None else OSQPSettings()

        self.n = self.P.shape[0]
        self.m = self.A.shape[0]

        # Cold start
        self.x = np.zeros(self.n)
        self.z = np.zeros(self.m)
        self.y = np.zeros(self.m)

        # Linear solver cache
        self._kkt_solver: Optional[spla.SuperLU] = None

        # log
        self.history: List[Dict[str, Any]] = []

    def setup(self) -> None:
        """
        Build and factorize the (quasi-definite) KKT matrix (direct method):
            [ P + sigma I    A^T      ]
            [ A             -rho^{-1}I]
        """
        rho = float(self.s.rho)
        sigma = float(self.s.sigma)

        P_sigma = self.P + sigma * spa.eye(self.n, format="csc")
        minus_rho_inv_I = -(1.0 / rho) * spa.eye(self.m, format="csc")

        KKT = spa.bmat(
            [[P_sigma, self.A.T],
             [self.A,   minus_rho_inv_I]],
            format="csc",
        )

        self._kkt_solver = spla.splu(KKT)

    def _project_C(self, z_in: np.ndarray) -> np.ndarray:
        """Projection onto C = { z | l <= z <= u }."""
        return np.clip(z_in, self.l, self.u)


    def _residuals_inf(self) -> (float, float):
        """
        Keep your residual definitions:
            r_prim = A x - z
            r_dual = P x + q + A^T y
        (Both measured in infinity norm.)
        """
        r_prim = self.A @ self.x - self.z
        r_dual = self.P @ self.x + self.q + self.A.T @ self.y
        return float(np.linalg.norm(r_prim, np.inf)), float(np.linalg.norm(r_dual, np.inf))

    def _objective(self) -> float:
        """0.5 x^T P x + q^T x (for logging only)."""
        Px = self.P @ self.x
        return 0.5 * float(self.x @ Px) + float(self.q @ self.x)

    # main solve

    def solve(self):
        """
        Returns:
            x, z, y, info

        info includes:
            status: "solved" or "max_iter"
            iter:   number of iterations executed
            prim_res_inf, dual_res_inf
            history: list of per-iteration logs (if store_history=True)
        """
        if self._kkt_solver is None:
            self.setup()

        rho = float(self.s.rho)
        sigma = float(self.s.sigma)
        alpha = float(self.s.alpha)

        if self.s.verbose:
            print(" k        ||r_p||_inf      ||r_d||_inf          obj")

        status = "max_iter"
        k_final = self.s.max_iter - 1

        for k in range(self.s.max_iter):
            # --- (1) KKT solve for (x_tilde, gamma) ---
            # RHS:
            #   [ sigma x^k - q ]
            #   [ z^k - rho^{-1} y^k ]
            rhs = np.concatenate([
                sigma * self.x - self.q,
                self.z - (1.0 / rho) * self.y
            ])

            sol = self._kkt_solver.solve(rhs)
            x_tilde = sol[:self.n]
            gamma = sol[self.n:]  # multiplier of A x_tilde = z_tilde

            # Recover z_tilde:
            #   z_tilde = z^k + rho^{-1} (gamma - y^k)
            z_tilde = self.z + (1.0 / rho) * (gamma - self.y)

            # --- (2) over-relaxation ---
            x_hat = alpha * x_tilde + (1.0 - alpha) * self.x
            z_hat = alpha * z_tilde + (1.0 - alpha) * self.z

            # --- (3) primal updates ---
            # x^{k+1} = x_hat  (w term omitted; in OSQP paper it stays zero)
            self.x = x_hat

            # z^{k+1} = Pi_C(z_hat + rho^{-1} y^k)
            self.z = self._project_C(z_hat + (1.0 / rho) * self.y)

            # --- (4) dual update ---
            # y^{k+1} = y^k + rho (z_hat - z^{k+1})
            self.y = self.y + rho * (z_hat - self.z)

            prim_res, dual_res = self._residuals_inf()
            obj = self._objective()

            if self.s.store_history:
                self.history.append({
                    "k": k,
                    "prim_res_inf": prim_res,
                    "dual_res_inf": dual_res,
                    "obj": obj,
                })

            if self.s.verbose and (k % max(1, self.s.log_every) == 0):
                print(f"{k:4d}   {prim_res:14.6e}  {dual_res:14.6e}  {obj:14.6e}")

            if prim_res < self.s.eps_abs and dual_res < self.s.eps_abs:
                status = "solved"
                k_final = k
                break

        prim_res, dual_res = self._residuals_inf()
        info = {
            "status": status,
            "iter": int(k_final + 1),
            "prim_res_inf": prim_res,
            "dual_res_inf": dual_res,
            "history": self.history if self.s.store_history else None,
        }
        return self.x, self.z, self.y, info
