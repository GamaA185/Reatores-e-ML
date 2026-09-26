"""Relatórios numéricos dos resultados."""

import numpy as np
import pandas as pd

from .model import ReactorParams, kinetic_constants


def summarize(sol, p: ReactorParams, display: bool = True):
    names = ["A", "B", "C", "I", "P", "T"]
    final = sol.y[:, -1]
    final_df = pd.DataFrame({"variavel": names, "valor_final": final})

    constants = kinetic_constants(final[-1], p)
    constants_df = pd.DataFrame({
        "constante": list(constants),
        "valor": list(constants.values()),
        "unidade": ["L/(mol s)", "1/s", "1/s", "1/s"],
    })

    reference = p.CA_in if p.mode.upper() == "CSTR" else p.CA0
    conversion = np.nan if reference == 0 else (reference - final[0]) / reference
    metrics = {
        "mode": p.mode.upper(),
        "time_final": float(sol.t[-1]),
        "conversion_A": float(conversion),
        "max_temperature": float(np.max(sol.y[5])),
        "max_I": float(np.max(sol.y[3])),
    }

    if display:
        print("\n=== A + B <-> C -> I -> P com balanço de energia ===")
        print(f"Modo de operação: {metrics['mode']}")
        print(f"Tempo final: {metrics['time_final']:.1f} s")
        print(f"Conversão aparente de A: {metrics['conversion_A']:.4f}")
        print(f"Temperatura máxima: {metrics['max_temperature']:.2f} K")
        print(f"Maior acúmulo de I: {metrics['max_I']:.4e} mol/L")
        print("\nConstantes cinéticas na temperatura final:")
        print(constants_df.to_string(index=False))
        print("\nEstado final:")
        print(final_df.to_string(index=False))

    return {"metrics": metrics, "final": final_df, "kinetic_constants": constants_df}
