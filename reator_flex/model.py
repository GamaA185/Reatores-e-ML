"""Parâmetros e equações do modelo do reator.

Rede de reações: A + B <-> C -> I -> P.
"""

from dataclasses import dataclass

import numpy as np

R = 8.314462618  # J/(mol K)


@dataclass(frozen=True)
class ReactorParams:
    mode: str = "CSTR"

    # Operação
    V: float = 100.0
    F: float = 1.0
    rho_cp: float = 4180.0
    UA: float = 250.0
    T_coolant: float = 300.0

    # Alimentação do CSTR
    CA_in: float = 1.0
    CB_in: float = 1.0
    CC_in: float = 0.0
    CI_in: float = 0.0
    CP_in: float = 0.0
    T_in: float = 330.0

    # Condições iniciais
    CA0: float = 1.0
    CB0: float = 1.0
    CC0: float = 0.0
    CI0: float = 0.0
    CP0: float = 0.0
    T0: float = 330.0

    # Cinética na temperatura de referência
    T_ref: float = 330.0
    k1f_ref: float = 0.020
    k1r_ref: float = 0.006
    k2_ref: float = 0.012
    k3_ref: float = 0.004

    # Energias de ativação
    E1f: float = 45_000.0
    E1r: float = 40_000.0
    E2: float = 55_000.0
    E3: float = 50_000.0

    # Entalpias na direção escrita (J/mol)
    dH1: float = -50_000.0
    dH2: float = -30_000.0
    dH3: float = -20_000.0

    def __post_init__(self):
        if self.mode.upper() not in {"BATCH", "CSTR"}:
            raise ValueError("mode deve ser 'BATCH' ou 'CSTR'.")
        if self.V <= 0:
            raise ValueError("V deve ser positivo.")
        if self.F < 0:
            raise ValueError("F deve ser maior ou igual a zero.")
        if self.rho_cp <= 0:
            raise ValueError("rho_cp deve ser positivo.")
        if self.UA < 0:
            raise ValueError("UA deve ser maior ou igual a zero.")
        if self.T_ref <= 0 or self.T0 <= 0 or self.T_in <= 0 or self.T_coolant <= 0:
            raise ValueError("As temperaturas devem ser positivas.")

        concentrations = (
            self.CA_in, self.CB_in, self.CC_in, self.CI_in, self.CP_in,
            self.CA0, self.CB0, self.CC0, self.CI0, self.CP0,
        )
        if any(value < 0 for value in concentrations):
            raise ValueError("As concentrações não podem ser negativas.")
        if any(value <= 0 for value in (self.k1f_ref, self.k1r_ref, self.k2_ref, self.k3_ref)):
            raise ValueError("As constantes cinéticas de referência devem ser positivas.")


def arrhenius(k_ref: float, E: float, T: float, T_ref: float) -> float:
    """Calcula k(T) pela forma relativa da equação de Arrhenius."""
    if k_ref < 0 or E < 0 or T_ref <= 0:
        raise ValueError("k_ref e E devem ser não negativos e T_ref positiva.")
    T_safe = max(float(T), 1.0)
    return float(k_ref * np.exp((-E / R) * (1.0 / T_safe - 1.0 / T_ref)))


def kinetic_constants(T: float, p: ReactorParams) -> dict[str, float]:
    return {
        "k1f": arrhenius(p.k1f_ref, p.E1f, T, p.T_ref),
        "k1r": arrhenius(p.k1r_ref, p.E1r, T, p.T_ref),
        "k2": arrhenius(p.k2_ref, p.E2, T, p.T_ref),
        "k3": arrhenius(p.k3_ref, p.E3, T, p.T_ref),
    }


def _safe_state(y):
    state = np.asarray(y, dtype=float)
    if state.shape != (6,):
        raise ValueError("O estado deve conter [CA, CB, CC, CI, CP, T].")
    concentrations = np.maximum(state[:5], 0.0)
    return *concentrations, max(float(state[5]), 1.0)


def reaction_rates(y, p: ReactorParams):
    CA, CB, CC, CI, _CP, T = _safe_state(y)
    k = kinetic_constants(T, p)
    r1 = k["k1f"] * CA * CB - k["k1r"] * CC
    r2 = k["k2"] * CC
    r3 = k["k3"] * CI
    return r1, r2, r3, k


def reactor_rhs(_t, y, p: ReactorParams):
    CA, CB, CC, CI, CP, T = _safe_state(y)
    r1, r2, r3, _ = reaction_rates(y, p)

    derivatives = np.array([-r1, -r1, r1 - r2, r2 - r3, r3, 0.0])
    reaction_heat = -(p.dH1 * r1 + p.dH2 * r2 + p.dH3 * r3) / p.rho_cp
    exchange_heat = p.UA * (p.T_coolant - T) / (p.rho_cp * p.V)
    derivatives[5] = reaction_heat + exchange_heat

    if p.mode.upper() == "CSTR":
        dilution = p.F / p.V
        feed = np.array([p.CA_in, p.CB_in, p.CC_in, p.CI_in, p.CP_in, p.T_in])
        derivatives += dilution * (feed - np.array([CA, CB, CC, CI, CP, T]))

    return derivatives
