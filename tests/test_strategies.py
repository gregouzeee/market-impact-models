"""Tests unitaires pour les stratégies d'exécution.

Ces tests vérifient le bon fonctionnement des stratégies TWAP, VWAP et Optimale.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ajouter le répertoire racine au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.strategies import (
    ExecutionParams,
    ImpactParams,
    TWAP,
    VWAP,
    OptimalAC,
    compare_strategies,
)


class TestExecutionParams:
    """Tests pour les paramètres d'exécution."""

    def test_creation(self):
        """Teste la création de paramètres d'exécution."""
        params = ExecutionParams(
            X=1000,
            T=1.0,
            n_slices=10,
            sigma=0.02,
            S0=100.0,
        )

        assert params.X == 1000
        assert params.T == 1.0
        assert params.n_slices == 10
        assert params.sigma == 0.02

    def test_tau_calculation(self):
        """Teste le calcul de tau (durée d'une tranche)."""
        params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)

        assert params.tau == 0.1

    def test_time_grid(self):
        """Teste la génération de la grille temporelle."""
        params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)

        time_grid = params.time_grid

        assert len(time_grid) == 11  # n_slices + 1
        assert time_grid[0] == 0
        assert time_grid[-1] == 1.0


class TestImpactParams:
    """Tests pour les paramètres d'impact."""

    def test_creation(self):
        """Teste la création de paramètres d'impact."""
        params = ImpactParams(
            psi=0.0005,
            eta=0.01,
            phi=0.5,
            k=0.001,
            daily_volume=10000,
        )

        assert params.psi == 0.0005
        assert params.eta == 0.01
        assert params.phi == 0.5
        assert params.k == 0.001

    def test_from_calibration(self):
        """Teste la création depuis un résultat de calibration."""
        calibration = {
            "psi": 0.0005,
            "eta": 0.01,
            "phi": 0.5,
            "k": 0.001,
        }

        params = ImpactParams.from_calibration(calibration, daily_volume=10000)

        assert params.psi == 0.0005
        assert params.daily_volume == 10000


class TestTWAP:
    """Tests pour la stratégie TWAP."""

    @pytest.fixture
    def twap_strategy(self):
        """Crée une stratégie TWAP pour les tests."""
        exec_params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )
        return TWAP(exec_params, impact_params)

    def test_schedule_uniform(self, twap_strategy):
        """Teste que TWAP génère un schedule uniforme."""
        schedule = twap_strategy.compute_schedule()

        # Toutes les tranches doivent être égales
        assert np.allclose(schedule, schedule[0])

        # La somme doit égaler X
        assert np.isclose(schedule.sum(), twap_strategy.exec.X)

    def test_trajectory_linear(self, twap_strategy):
        """Teste que la trajectoire TWAP est linéaire."""
        trajectory = twap_strategy.compute_trajectory()

        # Vérifier la linéarité
        expected = twap_strategy.exec.X * (
            1 - np.linspace(0, 1, twap_strategy.exec.n_slices + 1)
        )
        assert np.allclose(trajectory, expected)

    def test_trajectory_bounds(self, twap_strategy):
        """Teste les bornes de la trajectoire."""
        trajectory = twap_strategy.compute_trajectory()

        assert trajectory[0] == twap_strategy.exec.X
        assert np.isclose(trajectory[-1], 0)

    def test_costs_positive(self, twap_strategy):
        """Teste que les coûts sont positifs."""
        costs = twap_strategy.compute_costs()

        assert costs["spread_cost"] >= 0
        assert costs["temporary_cost"] >= 0
        assert costs["permanent_cost"] >= 0
        assert costs["total_cost"] > 0


class TestVWAP:
    """Tests pour la stratégie VWAP."""

    @pytest.fixture
    def vwap_strategy(self):
        """Crée une stratégie VWAP pour les tests."""
        exec_params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )
        return VWAP(exec_params, impact_params)

    def test_schedule_sum(self, vwap_strategy):
        """Teste que la somme du schedule VWAP égale X."""
        schedule = vwap_strategy.compute_schedule()

        assert np.isclose(schedule.sum(), vwap_strategy.exec.X)

    def test_schedule_positive(self, vwap_strategy):
        """Teste que toutes les tranches sont positives."""
        schedule = vwap_strategy.compute_schedule()

        assert np.all(schedule >= 0)

    def test_volume_profile_normalized(self, vwap_strategy):
        """Teste que le profil de volume est normalisé."""
        assert np.isclose(vwap_strategy.volume_profile.sum(), 1.0)

    def test_custom_volume_profile(self):
        """Teste VWAP avec un profil de volume personnalisé."""
        exec_params = ExecutionParams(X=1000, T=1.0, n_slices=5, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )

        # Profil personnalisé: plus de volume au début
        custom_profile = np.array([3, 2, 2, 2, 1])

        vwap = VWAP(exec_params, impact_params, volume_profile=custom_profile)

        schedule = vwap.compute_schedule()

        # Vérifier que le profil est respecté
        assert schedule[0] > schedule[-1]  # Plus au début qu'à la fin


class TestOptimalAC:
    """Tests pour la stratégie optimale Almgren-Chriss."""

    @pytest.fixture
    def optimal_strategy(self):
        """Crée une stratégie optimale pour les tests."""
        exec_params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )
        return OptimalAC(exec_params, impact_params, risk_aversion=1e-5)

    def test_schedule_sum(self, optimal_strategy):
        """Teste que la somme du schedule optimal égale X."""
        schedule = optimal_strategy.compute_schedule()

        assert np.isclose(schedule.sum(), optimal_strategy.exec.X, rtol=1e-3)

    def test_trajectory_monotonic(self, optimal_strategy):
        """Teste que la trajectoire est monotone décroissante."""
        trajectory = optimal_strategy.compute_trajectory()

        for i in range(len(trajectory) - 1):
            assert trajectory[i] >= trajectory[i + 1]

    def test_high_risk_aversion(self):
        """Teste qu'une haute aversion au risque accélère l'exécution."""
        exec_params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )

        # Faible aversion au risque
        low_lambda = OptimalAC(exec_params, impact_params, risk_aversion=1e-8)
        # Haute aversion au risque
        high_lambda = OptimalAC(exec_params, impact_params, risk_aversion=1e-3)

        schedule_low = low_lambda.compute_schedule()
        schedule_high = high_lambda.compute_schedule()

        # Avec haute aversion, on exécute plus au début
        # (pour réduire l'exposition au risque)
        assert schedule_high[0] >= schedule_low[0]


class TestCompareStrategies:
    """Tests pour la comparaison des stratégies."""

    def test_compare_returns_all_strategies(self):
        """Teste que compare_strategies retourne toutes les stratégies."""
        exec_params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )

        results = compare_strategies(exec_params, impact_params)

        assert "TWAP" in results
        assert "VWAP" in results
        assert "Optimal" in results

    def test_compare_structure(self):
        """Teste la structure des résultats de comparaison."""
        exec_params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )

        results = compare_strategies(exec_params, impact_params)

        for strategy_name, data in results.items():
            assert "schedule" in data
            assert "trajectory" in data
            assert "costs" in data
            assert "risk" in data
            assert "risk_adjusted_cost" in data


class TestNumericalStability:
    """Tests de stabilité numérique."""

    def test_small_quantities(self):
        """Teste avec de petites quantités."""
        exec_params = ExecutionParams(X=0.001, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )

        twap = TWAP(exec_params, impact_params)
        schedule = twap.compute_schedule()

        assert not np.any(np.isnan(schedule))
        assert not np.any(np.isinf(schedule))

    def test_large_quantities(self):
        """Teste avec de grandes quantités."""
        exec_params = ExecutionParams(X=1e9, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=1e12
        )

        twap = TWAP(exec_params, impact_params)
        schedule = twap.compute_schedule()

        assert not np.any(np.isnan(schedule))
        assert not np.any(np.isinf(schedule))

    def test_extreme_risk_aversion(self):
        """Teste avec des valeurs extrêmes d'aversion au risque."""
        exec_params = ExecutionParams(X=1000, T=1.0, n_slices=10, sigma=0.02)
        impact_params = ImpactParams(
            psi=0.0005, eta=0.01, phi=0.5, k=0.001, daily_volume=10000
        )

        # Très faible aversion
        opt_low = OptimalAC(exec_params, impact_params, risk_aversion=1e-12)
        schedule_low = opt_low.compute_schedule()
        assert not np.any(np.isnan(schedule_low))

        # Très haute aversion
        opt_high = OptimalAC(exec_params, impact_params, risk_aversion=1e-1)
        schedule_high = opt_high.compute_schedule()
        assert not np.any(np.isnan(schedule_high))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
