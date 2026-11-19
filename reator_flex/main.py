"""Executa um caso padrão do reator.

Uso, a partir da raiz do repositório:
    python -m reator_flex.main
"""

from .model import ReactorParams
from .balances import audit_balances
from .plotting import plot_results
from .reporting import summarize
from .simulation import simulate


def main():
    params = ReactorParams(mode="CSTR")
    solution = simulate(params)
    summarize(solution, params)
    errors = audit_balances(solution, params)
    print("\nResíduos máximos de fechamento nos passos da integração")
    print(f"Inventário de A: {errors['max_inventory_A_residual_mol']:.3e} mol")
    print(f"Inventário de B: {errors['max_inventory_B_residual_mol']:.3e} mol")
    print(f"Entalpia total: {errors['max_enthalpy_residual_J']:.3e} J")
    plot_results(solution, params)
    return solution


if __name__ == "__main__":
    main()
