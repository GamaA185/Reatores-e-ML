"""Executa um caso padrão do reator.

Uso, a partir da raiz do repositório:
    python -m reator_flex.main
"""

from .model import ReactorParams
from .plotting import plot_results
from .reporting import summarize
from .simulation import simulate


def main():
    params = ReactorParams(mode="CSTR")
    solution = simulate(params)
    summarize(solution, params)
    plot_results(solution, params)
    return solution


if __name__ == "__main__":
    main()
