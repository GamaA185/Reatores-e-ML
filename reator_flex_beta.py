"""Entrada histórica para o modelo líquido Batch/CSTR do pacote reator_flex."""

from reator_flex import (
    ReactorParams, arrhenius, kinetic_constants, plot_results, reaction_rates,
    reactor_rhs, simulate, summarize,
)
from reator_flex.main import main
from reator_flex.model import R

if __name__ == "__main__":
    main()
