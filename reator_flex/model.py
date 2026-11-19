"""Reator líquido ideal e perfeitamente misturado: A + B ⇌ C → I → P.

Volume, densidade, capacidade térmica volumétrica e entalpias de reação
são constantes. A alimentação tem as mesmas propriedades térmicas do tanque.
No CSTR as vazões volumétricas de entrada e saída são iguais. Na batelada
ambas são nulas. A camisa tem temperatura prescrita e UA constante. Desprezam-se
trabalho de eixo, calor de mistura e variações de energia cinética e potencial.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import exp, isclose, isfinite

import numpy as np

R = 8.314462618  # J/(mol K)


@dataclass(frozen=True)
class ReactorParams:
    mode: str = "CSTR"

    # V [L], F [L/s], rho_cp [J/(L K)], UA [J/(s K)], temperaturas [K].
    V: float = 100.0
    F: float = 1.0
    rho_cp: float = 4180.0
    UA: float = 250.0
    T_coolant: float = 300.0

    # Concentrações de alimentação e iniciais [mol/L].
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

    # k1f_ref [L/(mol s)], constantes de primeira ordem [1/s].
    T_ref: float = 330.0
    k1f_ref: float = 0.020
    k1r_ref: float = 0.006
    k2_ref: float = 0.012
    k3_ref: float = 0.004

    # Energias de ativação [J/mol]. A reversa é determinada por E1f e dH1.
    E1f: float = 45_000.0
    E1r: float | None = None
    E2: float = 55_000.0
    E3: float = 50_000.0

    # Entalpias na direção escrita (J/mol)
    dH1: float = -50_000.0
    dH2: float = -30_000.0
    dH3: float = -20_000.0

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode.upper() not in {"BATCH", "CSTR"}:
            raise ValueError("mode deve ser 'BATCH' ou 'CSTR'.")
        for field in fields(self):
            if field.name == "mode" or (field.name == "E1r" and self.E1r is None):
                continue
            if not isfinite(getattr(self, field.name)):
                raise ValueError(f"{field.name} deve ser finito.")
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
        if self.k1f_ref <= 0 or self.k1r_ref <= 0:
            raise ValueError("A etapa reversível requer k1f_ref e k1r_ref positivos.")
        if self.k2_ref < 0 or self.k3_ref < 0:
            raise ValueError("k2_ref e k3_ref devem ser não negativos.")
        if min(self.E1f, self.E2, self.E3, self.E1f - self.dH1) < 0:
            raise ValueError("As energias de ativação devem ser não negativas.")
        if self.E1r is not None and not isclose(self.E1r, self.E1f - self.dH1, abs_tol=1e-8):
            raise ValueError("E1r deve ser igual a E1f - dH1 para o equilíbrio da etapa 1.")


def arrhenius(k_ref: float, E: float, T: float, T_ref: float) -> float:
    """Calcula k(T) pela forma relativa da equação de Arrhenius."""
    if not all(map(isfinite, (k_ref, E, T, T_ref))):
        raise ValueError("Os parâmetros de Arrhenius devem ser finitos.")
    if k_ref < 0 or E < 0 or T <= 0 or T_ref <= 0:
        raise ValueError("k_ref e E devem ser não negativos e T e T_ref positivas (K).")
    if k_ref == 0:
        return 0.0
    return k_ref * exp((E / R) * (1.0 / T_ref - 1.0 / T))


def kinetic_constants(T: float, p: ReactorParams) -> dict[str, float]:
    """O quociente k1f/k1r [L/mol] obedece a van 't Hoff com dH1 constante."""
    return {
        "k1f": arrhenius(p.k1f_ref, p.E1f, T, p.T_ref),
        "k1r": arrhenius(p.k1r_ref, p.E1f - p.dH1, T, p.T_ref),
        "k2": arrhenius(p.k2_ref, p.E2, T, p.T_ref),
        "k3": arrhenius(p.k3_ref, p.E3, T, p.T_ref),
    }


def reaction_rates(y, p: ReactorParams):
    """Velocidades por volume de líquido [mol/(L s)]; r1 é líquida."""
    state = np.asarray(y, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("O estado deve conter seis valores finitos [CA, CB, CC, CI, CP, T].")
    CA, CB, CC, CI, _CP, T = state
    k = kinetic_constants(T, p)
    r1 = k["k1f"] * CA * CB - k["k1r"] * CC
    r2 = k["k2"] * CC
    r3 = k["k3"] * CI
    return r1, r2, r3, k


def reactor_rhs(_t, y, p: ReactorParams):
    """Derivadas das cinco concentrações [mol/(L s)] e da temperatura [K/s]."""
    state = np.asarray(y, dtype=float)
    r1, r2, r3, _ = reaction_rates(state, p)
    T = state[5]

    derivatives = np.array([-r1, -r1, r1 - r2, r2 - r3, r3, 0.0])
    reaction_heat = -(p.dH1 * r1 + p.dH2 * r2 + p.dH3 * r3) / p.rho_cp
    exchange_heat = p.UA * (p.T_coolant - T) / (p.rho_cp * p.V)
    derivatives[5] = reaction_heat + exchange_heat

    if p.mode.upper() == "CSTR":
        dilution = p.F / p.V
        feed = np.array([p.CA_in, p.CB_in, p.CC_in, p.CI_in, p.CP_in, p.T_in])
        derivatives += dilution * (feed - state)

    return derivatives
