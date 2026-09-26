"""Rotinas de integração numérica."""

import numpy as np
from scipy.integrate import solve_ivp

from .model import ReactorParams, reactor_rhs


def simulate(p: ReactorParams, t_final: float = 2000.0, n_points: int = 1200):
    """Integra o modelo e retorna um objeto ``OdeResult`` do SciPy."""
    if t_final <= 0:
        raise ValueError("t_final deve ser positivo.")
    if n_points < 2:
        raise ValueError("n_points deve ser pelo menos 2.")

    y0 = np.array([p.CA0, p.CB0, p.CC0, p.CI0, p.CP0, p.T0], dtype=float)
    t_eval = np.linspace(0.0, t_final, n_points)
    sol = solve_ivp(
        reactor_rhs,
        (0.0, t_final),
        y0,
        args=(p,),
        method="BDF",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Falha na integração: {sol.message}")
    return sol
