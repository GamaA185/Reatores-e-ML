"""
Exercicio proposto
------------------
Simular um reator liquido ideal, operando como Batch ou CSTR, para a rede:

    A + B <-> C -> I -> P

A primeira etapa e elementar reversivel e de segunda ordem na direcao direta.
As etapas seguintes sao elementares irreversiveis e de primeira ordem.
O modelo inclui dependencia da temperatura por Arrhenius e balanco de energia
com calor de reacao e troca termica com uma camisa.

Unidades principais:
    concentracao: mol/L
    tempo: s
    volume: L
    temperatura: K
    energia: J
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


R = 8.314462618  # J/(mol K)


@dataclass(frozen=True)
class ReactorParams:
    mode: str = "CSTR"  # "BATCH" ou "CSTR"

    # Operacao
    V: float = 100.0  # L
    F: float = 1.0  # L/s, usado apenas no CSTR
    rho_cp: float = 4180.0  # J/(L K), aproximacao para mistura aquosa
    UA: float = 250.0  # J/(s K)
    T_coolant: float = 300.0  # K

    # Alimentacao do CSTR
    CA_in: float = 1.0  # mol/L
    CB_in: float = 1.0  # mol/L
    CC_in: float = 0.0  # mol/L
    CI_in: float = 0.0  # mol/L
    CP_in: float = 0.0  # mol/L
    T_in: float = 330.0  # K

    # Condicoes iniciais
    CA0: float = 1.0  # mol/L
    CB0: float = 1.0  # mol/L
    CC0: float = 0.0  # mol/L
    CI0: float = 0.0  # mol/L
    CP0: float = 0.0  # mol/L
    T0: float = 330.0  # K

    # Cinetica em T_ref
    T_ref: float = 330.0  # K
    k1f_ref: float = 0.020  # L/(mol s), A + B -> C
    k1r_ref: float = 0.006  # 1/s, C -> A + B
    k2_ref: float = 0.012  # 1/s, C -> I
    k3_ref: float = 0.004  # 1/s, I -> P

    # Energias de ativacao
    E1f: float = 45_000.0  # J/mol
    E1r: float = 40_000.0  # J/mol
    E2: float = 55_000.0  # J/mol
    E3: float = 50_000.0  # J/mol

    # Entalpias de reacao, na direcao escrita
    dH1: float = -50_000.0  # J/mol, A + B -> C
    dH2: float = -30_000.0  # J/mol, C -> I
    dH3: float = -20_000.0  # J/mol, I -> P


def arrhenius(k_ref, E, T, T_ref):
    """Constante cinetica em T usando forma relativa da equacao de Arrhenius."""
    T = max(float(T), 1.0)
    return k_ref * np.exp((-E / R) * (1.0 / T - 1.0 / T_ref))


def kinetic_constants(T, p):
    return {
        "k1f": arrhenius(p.k1f_ref, p.E1f, T, p.T_ref),
        "k1r": arrhenius(p.k1r_ref, p.E1r, T, p.T_ref),
        "k2": arrhenius(p.k2_ref, p.E2, T, p.T_ref),
        "k3": arrhenius(p.k3_ref, p.E3, T, p.T_ref),
    }


def reaction_rates(y, p):
    CA, CB, CC, CI, CP, T = np.maximum(y, 0.0)
    k = kinetic_constants(T, p)

    r1f = k["k1f"] * CA * CB
    r1r = k["k1r"] * CC
    r1 = r1f - r1r
    r2 = k["k2"] * CC
    r3 = k["k3"] * CI

    return r1, r2, r3, k


def reactor_rhs(t, y, p):
    CA, CB, CC, CI, CP, T = np.maximum(y, 0.0)
    r1, r2, r3, _ = reaction_rates(y, p)

    dCA = -r1
    dCB = -r1
    dCC = r1 - r2
    dCI = r2 - r3
    dCP = r3

    heat_reaction = -(p.dH1 * r1 + p.dH2 * r2 + p.dH3 * r3) / p.rho_cp
    heat_exchange = p.UA * (p.T_coolant - T) / (p.rho_cp * p.V)
    dT = heat_reaction + heat_exchange

    if p.mode.upper() == "CSTR":
        dilution = p.F / p.V
        dCA += dilution * (p.CA_in - CA)
        dCB += dilution * (p.CB_in - CB)
        dCC += dilution * (p.CC_in - CC)
        dCI += dilution * (p.CI_in - CI)
        dCP += dilution * (p.CP_in - CP)
        dT += dilution * (p.T_in - T)
    elif p.mode.upper() != "BATCH":
        raise ValueError("mode deve ser 'BATCH' ou 'CSTR'.")

    return np.array([dCA, dCB, dCC, dCI, dCP, dT])


def simulate(p, t_final=2000.0, n_points=1200):
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
        raise RuntimeError(f"Falha na integracao: {sol.message}")

    return sol


def summarize(sol, p):
    names = ["A", "B", "C", "I", "P", "T"]
    final = sol.y[:, -1]
    df_final = pd.DataFrame({"variavel": names, "valor_final": final})

    k_final = kinetic_constants(final[-1], p)
    df_k = pd.DataFrame(
        {
            "constante": list(k_final.keys()),
            "valor": list(k_final.values()),
            "unidade": ["L/(mol s)", "1/s", "1/s", "1/s"],
        }
    )

    CA_ref = p.CA_in if p.mode.upper() == "CSTR" else p.CA0
    conversion_A = (CA_ref - final[0]) / CA_ref
    max_temperature = np.max(sol.y[-1])
    max_I = np.max(sol.y[3])

    print("\n=== Exercicio: A + B <-> C -> I -> P com balanco de energia ===")
    print(f"Modo de operacao: {p.mode.upper()}")
    print(f"Tempo final: {sol.t[-1]:.1f} s")
    print(f"Conversao aparente de A: {conversion_A:.4f}")
    print(f"Temperatura maxima: {max_temperature:.2f} K")
    print(f"Maior acumulo de intermediario I: {max_I:.4e} mol/L")

    print("\nConstantes cineticas na temperatura final:")
    print(df_k.to_string(index=False))

    print("\nEstado final:")
    print(df_final.to_string(index=False))


def plot_results(sol, p):
    t = sol.t
    CA, CB, CC, CI, CP, T = sol.y

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(t, CA, label="A")
    axes[0].plot(t, CB, label="B")
    axes[0].plot(t, CC, label="C")
    axes[0].plot(t, CI, label="I")
    axes[0].plot(t, CP, label="P")
    axes[0].set_xlabel("Tempo (s)")
    axes[0].set_ylabel("Concentracao (mol/L)")
    axes[0].set_title(f"Especies no reator ({p.mode.upper()})")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, T, color="tab:red", label="T")
    axes[1].axhline(p.T_coolant, color="tab:blue", linestyle="--", label="T camisa")
    if p.mode.upper() == "CSTR":
        axes[1].axhline(p.T_in, color="0.3", linestyle=":", label="T entrada")
    axes[1].set_xlabel("Tempo (s)")
    axes[1].set_ylabel("Temperatura (K)")
    axes[1].set_title("Balanco de energia")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    params = ReactorParams(mode="CSTR")
    sol = simulate(params)
    summarize(sol, params)
    plot_results(sol, params)


if __name__ == "__main__":
    main()
