"""Relatórios numéricos dos resultados."""

import numpy as np
import pandas as pd

from .model import ReactorParams, kinetic_constants, reactor_rhs


def _peak(sol, species, event):
    # Compatibilidade com resultados antigos, que só guardavam a malha de saída.
    if getattr(sol, "y_events", None) is None:
        i = np.argmax(sol.y[species])
        return float(sol.y[species, i]), float(sol.t[i])

    times = sol.t_events[event]
    heights = sol.y_events[event][:, species] if times.size else []
    values = np.r_[sol.y[species, 0], heights, sol.y[species, -1]]
    times = np.r_[sol.t[0], times, sol.t[-1]]
    i = np.argmax(values)
    return float(values[i]), float(times[i])


def _steady_state(sol, p, residence_time, concentration_rate_tol, temperature_rate_tol):
    assessment = {
        "steady_state": None,
        "steady_state_window_s": residence_time,
        "max_concentration_rate_mol_L_s": None,
        "max_temperature_rate_K_s": None,
        "concentration_rate_tol_mol_L_s": concentration_rate_tol,
        "temperature_rate_tol_K_s": temperature_rate_tol,
    }
    if residence_time is None or sol.t[-1] - sol.t[0] < residence_time:
        return assessment
    if getattr(sol, "sol", None) is None:
        return assessment

    times = np.linspace(sol.t[-1] - residence_time, sol.t[-1], 51)
    states = sol.sol(times).T
    rates = np.array([reactor_rhs(t, state, p) for t, state in zip(times, states)])
    concentration_rate = float(np.max(np.abs(rates[:, :5])))
    temperature_rate = float(np.max(np.abs(rates[:, 5])))
    assessment.update(
        steady_state=bool(concentration_rate <= concentration_rate_tol
                          and temperature_rate <= temperature_rate_tol),
        max_concentration_rate_mol_L_s=concentration_rate,
        max_temperature_rate_K_s=temperature_rate,
    )
    return assessment


def summarize(sol, p: ReactorParams, display: bool = True, *,
              concentration_rate_tol: float = 1e-8, temperature_rate_tol: float = 1e-6):
    """Resume a trajetória de simulate.

    Os tempos são em s. O diagnóstico de regime permanente do CSTR compara
    |dC/dt| e |dT/dt| em 51 instantes no último tempo de residência com os
    limites em mol/(L s) e K/s. Retorna None se o critério for inaplicável
    ou faltar histórico. product_outlet_mol_s é a vazão molar bruta de P
    na saída no instante final (zero em operação fechada).
    """
    tolerances = (concentration_rate_tol, temperature_rate_tol)
    if any(not np.isfinite(v) or v < 0 for v in tolerances):
        raise ValueError("As tolerâncias das derivadas devem ser finitas e não negativas.")
    names = ["A", "B", "C", "I", "P", "T"]
    final = sol.y[:, -1]
    final_df = pd.DataFrame({"variavel": names, "valor_final": final,
                             "unidade": ["mol/L"] * 5 + ["K"]})

    constants = kinetic_constants(final[-1], p)
    constants_df = pd.DataFrame({
        "constante": list(constants),
        "valor": list(constants.values()),
        "unidade": ["L/(mol s)", "1/s", "1/s", "1/s"],
    })

    flowing = p.mode.upper() == "CSTR" and p.F > 0
    reference = p.CA_in if flowing else p.CA0
    conversion = np.nan if reference == 0 else (reference - final[0]) / reference
    residence_time = p.V / p.F if flowing else None
    max_I, time_max_I = _peak(sol, 3, 0)
    max_temperature, time_max_temperature = _peak(sol, 5, 1)
    metrics = {
        "mode": p.mode.upper(),
        "time_final": float(sol.t[-1]),
        "conversion_A": float(conversion),
        "max_temperature": max_temperature,
        "time_max_temperature": time_max_temperature,
        "max_I": max_I,
        "time_max_I": time_max_I,
        "peak_method": "events" if getattr(sol, "y_events", None) is not None else "samples",
        "residence_time_s": residence_time,
        "product_outlet_mol_s": float(p.F * final[4]) if flowing else 0.0,
    }
    metrics.update(_steady_state(sol, p, residence_time, *tolerances))

    if display:
        print("\n=== A + B <-> C -> I -> P com balanço de energia ===")
        print(f"Modo de operação: {metrics['mode']}")
        print(f"Tempo final: {metrics['time_final']:.1f} s")
        print(f"Conversão aparente de A: {metrics['conversion_A']:.4f}")
        print(f"Temperatura máxima: {max_temperature:.2f} K em {time_max_temperature:.2f} s")
        print(f"Maior acúmulo de I: {max_I:.4e} mol/L em {time_max_I:.2f} s")
        if flowing:
            print(f"Tempo de residência: {residence_time:.2f} s")
            print(f"Vazão molar final de P na saída: {metrics['product_outlet_mol_s']:.4e} mol/s")
            status = {True: "atingido", False: "em transiente", None: "histórico insuficiente"}
            print(f"Regime permanente pelo critério de derivadas: {status[metrics['steady_state']]}")
        print("\nConstantes cinéticas na temperatura final:")
        print(constants_df.to_string(index=False))
        print("\nEstado final:")
        print(final_df.to_string(index=False))

    return {"metrics": metrics, "final": final_df, "kinetic_constants": constants_df}
