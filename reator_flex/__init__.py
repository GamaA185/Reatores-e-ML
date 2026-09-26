"""Modelo de reator flexível Batch/CSTR."""

from .model import ReactorParams, arrhenius, kinetic_constants, reaction_rates, reactor_rhs
from .simulation import simulate
from .reporting import summarize
from .plotting import plot_results

__all__ = [
    "ReactorParams",
    "arrhenius",
    "kinetic_constants",
    "reaction_rates",
    "reactor_rhs",
    "simulate",
    "summarize",
    "plot_results",
]
