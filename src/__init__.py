"""Market Impact Models - Modeles d'impact de marche pour l'execution optimale.

Ce package implemente le modele Almgren-Chriss pour l'execution optimale d'ordres,
avec calibration des parametres a partir de donnees d'orderbook.

Modules:
    models: Modele Almgren-Chriss et parametres
    optimization: Strategies TWAP, VWAP, Optimale
    calibration: Calibration des parametres d'impact
    data: Collecte de donnees orderbook
    storage: Gestion du stockage S3
"""

__version__ = "0.1.0"

from src.models import (
    AlmgrenChrissModel,
    MarketParameters,
    ImpactParameters,
)
from src.optimization import (
    TWAP,
    VWAP,
    OptimalAC,
    ExecutionParams,
    ImpactParams,
)
from src.calibration import MarketImpactCalibrator
from src.data import BinanceOrderBookCollector

__all__ = [
    # Models
    "AlmgrenChrissModel",
    "MarketParameters",
    "ImpactParameters",
    # Optimization
    "TWAP",
    "VWAP",
    "OptimalAC",
    "ExecutionParams",
    "ImpactParams",
    # Calibration
    "MarketImpactCalibrator",
    # Data
    "BinanceOrderBookCollector",
]
