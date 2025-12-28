"""Module d'optimisation et strategies d'execution.

Ce module contient les implementations des strategies d'execution
TWAP, VWAP et optimale (Almgren-Chriss).
"""

from src.optimization.strategies import (
    ExecutionParams,
    ImpactParams,
    ExecutionStrategy,
    TWAP,
    VWAP,
    OptimalAC,
    OptimalACVariable,
    compare_strategies,
    plot_strategy_comparison,
    print_comparison_table,
    generate_volume_profile,
    compute_volume_profile_from_data,
    plot_volume_profile,
    compare_profiles,
)

__all__ = [
    "ExecutionParams",
    "ImpactParams",
    "ExecutionStrategy",
    "TWAP",
    "VWAP",
    "OptimalAC",
    "OptimalACVariable",
    "compare_strategies",
    "plot_strategy_comparison",
    "print_comparison_table",
    "generate_volume_profile",
    "compute_volume_profile_from_data",
    "plot_volume_profile",
    "compare_profiles",
]
