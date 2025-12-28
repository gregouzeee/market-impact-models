"""Module de calibration des parametres de market impact.

Ce module contient les outils pour calibrer les parametres du modele
Almgren-Chriss a partir des donnees d'orderbook.
"""

from src.calibration.calibrator import MarketImpactCalibrator

__all__ = [
    "MarketImpactCalibrator",
]
