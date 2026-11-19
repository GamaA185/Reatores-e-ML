"""Verificação integral dos inventários do tanque ao longo da simulação."""

import numpy as np

from .model import ReactorParams


def _inventories(state):
    ca, cb, cc, ci, cp, _ = state
    products = cc + ci + cp
    return np.array([ca + products, cb + products])


def _enthalpy_density(state, p):
    """Entalpia sensível e química relativa [J/L], com A e B como referências."""
    _, _, cc, ci, cp, temperature = state
    return (p.rho_cp * (temperature - p.T_ref)
            + p.dH1 * (cc + ci + cp) + p.dH2 * (ci + cp) + p.dH3 * cp)


def audit_balances(sol, p: ReactorParams):
    """Máximos absolutos dos resíduos de inventário [mol] e de energia [J].

    Compara os inventários de A e B com a solução analítica da diluição.
    Compara a variação de entalpia total com as trocas pelas correntes e pela
    camisa, integradas sobre cada passo aceito do BDF. Requer a saída densa
    de simulate e os mesmos parâmetros usados na integração.
    """
    if getattr(sol, "sol", None) is None:
        raise ValueError("A auditoria requer uma solução com interpolação contínua.")
    time = sol.sol.ts
    state = sol.sol(time)
    initial = (p.CA0, p.CB0, p.CC0, p.CI0, p.CP0, p.T0)
    feed = (p.CA_in, p.CB_in, p.CC_in, p.CI_in, p.CP_in, p.T_in)
    flow = p.F if p.mode.upper() == "CSTR" else 0.0
    inlet = _inventories(feed)[:, None]
    start = _inventories(initial)[:, None]
    expected = start + (inlet - start) * (-np.expm1(-flow / p.V * (time - time[0])))
    inventory_error = p.V * np.max(np.abs(_inventories(state) - expected), axis=1)

    # Fluxo de entalpia linear no estado. Três nós de Gauss integram até
    # grau cinco, a ordem máxima da interpolação do BDF em cada passo.
    nodes, weights = np.polynomial.legendre.leggauss(3)
    half_width = np.diff(time) / 2
    midpoint = time[:-1] + half_width
    quad_times = midpoint[:, None] + half_width[:, None] * nodes
    quad_state = sol.sol(quad_times.ravel())
    stream_power = flow * (_enthalpy_density(feed, p) - _enthalpy_density(quad_state, p))
    jacket_power = p.UA * (p.T_coolant - quad_state[-1])
    power = (stream_power + jacket_power).reshape(quad_times.shape)
    transferred = np.r_[0.0, np.cumsum(half_width * (power @ weights))]
    stored = p.V * (_enthalpy_density(state, p) - _enthalpy_density(initial, p))
    return {
        "max_inventory_A_residual_mol": float(inventory_error[0]),
        "max_inventory_B_residual_mol": float(inventory_error[1]),
        "max_enthalpy_residual_J": float(np.max(np.abs(stored - transferred))),
    }
