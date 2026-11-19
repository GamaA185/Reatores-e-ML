"""Rotinas de integração numérica."""

import numpy as np
from scipy.integrate import solve_ivp

from .model import ReactorParams, reactor_rhs


def _maximum_event(index):
    def event(t, y, p):
        return reactor_rhs(t, y, p)[index]

    event.direction = -1
    return event


def simulate(p: ReactorParams, t_final: float = 2000.0, n_points: int = 1200):
    """Integra até t_final [s]. n_points controla a malha de saída.

    O resultado inclui interpolação contínua em sol e eventos de máximo local
    de CI e T, nessa ordem. O integrador detecta mudanças de sinal da derivada
    entre passos internos. Extremos múltiplos em um mesmo passo podem escapar.
    """
    if not np.isfinite(t_final) or t_final <= 0:
        raise ValueError("t_final deve ser finito e positivo.")
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points deve ser um inteiro de pelo menos 2.")

    y0 = np.array([p.CA0, p.CB0, p.CC0, p.CI0, p.CP0, p.T0], dtype=float)
    t_eval = np.linspace(0.0, t_final, n_points)
    sol = solve_ivp(
        reactor_rhs,
        (0.0, t_final),
        y0,
        args=(p,),
        method="BDF",
        t_eval=t_eval,
        dense_output=True,
        events=(_maximum_event(3), _maximum_event(5)),
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Falha na integração: {sol.message}")
    return sol
