"""Visualização dos resultados."""

import matplotlib.pyplot as plt

from .model import ReactorParams


def plot_results(sol, p: ReactorParams, show: bool = True):
    t = sol.t
    CA, CB, CC, CI, CP, temperature = sol.y
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for concentration, label in zip((CA, CB, CC, CI, CP), ("A", "B", "C", "I", "P")):
        axes[0].plot(t, concentration, label=label)
    axes[0].set(xlabel="Tempo (s)", ylabel="Concentração (mol/L)", title=f"Espécies no reator ({p.mode.upper()})")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, temperature, color="tab:red", label="T")
    axes[1].axhline(p.T_coolant, color="tab:blue", linestyle="--", label="T camisa")
    if p.mode.upper() == "CSTR":
        axes[1].axhline(p.T_in, color="0.3", linestyle=":", label="T entrada")
    axes[1].set(xlabel="Tempo (s)", ylabel="Temperatura (K)", title="Balanço de energia")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
