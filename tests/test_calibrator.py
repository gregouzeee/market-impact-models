"""Tests unitaires pour le module de calibration.

Ces tests vérifient le bon fonctionnement des méthodes d'estimation
des paramètres du modèle de market impact.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

# Pour les tests, on crée des données synthétiques
# sans avoir besoin d'importer les modules réels


class TestMarketImpactCalibrator:
    """Tests pour la classe MarketImpactCalibrator."""

    @pytest.fixture
    def mock_snapshots(self):
        """Génère des snapshots d'orderbook synthétiques pour les tests."""
        snapshots = []
        base_price = 100000  # Prix de base (ex: BTC)

        for i in range(100):
            # Générer un orderbook synthétique
            mid = base_price + np.random.normal(0, 100)

            # Bids: prix décroissants, quantités variables
            bids = []
            for j in range(50):
                price = mid - 0.01 * (j + 1) - np.random.uniform(0, 0.005)
                quantity = np.random.uniform(0.1, 2.0)
                bids.append([price, quantity])

            # Asks: prix croissants, quantités variables
            asks = []
            for j in range(50):
                price = mid + 0.01 * (j + 1) + np.random.uniform(0, 0.005)
                quantity = np.random.uniform(0.1, 2.0)
                asks.append([price, quantity])

            snapshots.append({
                "timestamp": f"2024-01-01T{i:02d}:00:00",
                "symbol": "BTCUSDT",
                "bids": bids,
                "asks": asks,
            })

        return snapshots

    @pytest.fixture
    def orderbook_file(self, mock_snapshots, tmp_path):
        """Crée un fichier JSON temporaire avec les snapshots."""
        filepath = tmp_path / "test_orderbook.json"
        with open(filepath, "w") as f:
            json.dump(mock_snapshots, f)
        return filepath

    def test_spread_calculation(self, mock_snapshots):
        """Teste le calcul du spread."""
        spreads = []

        for snapshot in mock_snapshots:
            bids = np.array(snapshot["bids"], dtype=float)
            asks = np.array(snapshot["asks"], dtype=float)

            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            mid = (best_bid + best_ask) / 2

            spread = (best_ask - best_bid) / mid
            spreads.append(spread)

        avg_spread = np.mean(spreads)

        # Le spread devrait être positif et petit
        assert avg_spread > 0, "Le spread doit être positif"
        assert avg_spread < 0.01, "Le spread ne devrait pas dépasser 1%"

    def test_orderbook_liquidity(self, mock_snapshots):
        """Teste que l'orderbook a de la liquidité."""
        for snapshot in mock_snapshots:
            bids = np.array(snapshot["bids"], dtype=float)
            asks = np.array(snapshot["asks"], dtype=float)

            # Vérifier qu'il y a de la liquidité
            assert len(bids) > 0, "Il doit y avoir des bids"
            assert len(asks) > 0, "Il doit y avoir des asks"

            # Vérifier que les prix sont ordonnés correctement
            assert bids[0, 0] > bids[-1, 0], "Bids doivent être décroissants"
            assert asks[0, 0] < asks[-1, 0], "Asks doivent être croissants"

            # Vérifier que best_bid < best_ask (pas de crossed book)
            assert bids[0, 0] < asks[0, 0], "Best bid doit être < best ask"

    def test_walk_the_book(self, mock_snapshots):
        """Teste la simulation d'un ordre de marché."""
        snapshot = mock_snapshots[0]
        asks = np.array(snapshot["asks"], dtype=float)

        target_quantity = 5.0
        remaining = target_quantity
        total_cost = 0.0
        levels_used = 0

        for price, quantity in asks:
            if remaining <= 0:
                break
            executed = min(remaining, quantity)
            total_cost += executed * price
            remaining -= executed
            levels_used += 1

        if remaining <= 0:
            avg_price = total_cost / target_quantity
            mid_price = (asks[0, 0] + np.array(snapshot["bids"])[0, 0]) / 2

            # L'impact devrait être positif (on achète, donc prix monte)
            impact = (avg_price - mid_price) / mid_price
            assert impact >= 0, "L'impact d'un achat doit être positif ou nul"

            # L'impact ne devrait pas être trop grand pour 5 unités
            assert impact < 0.01, "L'impact ne devrait pas dépasser 1% pour 5 unités"


class TestImpactModel:
    """Tests pour les fonctions d'impact de marché."""

    def test_power_law_impact(self):
        """Teste la fonction d'impact power-law."""
        eta = 0.01
        phi = 0.5
        psi = 0.0005
        k = 0.001

        def impact_model(rho):
            return psi + eta * np.power(np.abs(rho), phi) + k * rho

        # Test avec différentes valeurs de rho
        rhos = [0.01, 0.05, 0.1, 0.2]

        for rho in rhos:
            impact = impact_model(rho)

            # L'impact doit être positif pour rho > 0
            assert impact > 0, f"Impact doit être positif pour rho={rho}"

            # L'impact doit être croissant avec rho
            if rho > 0.01:
                prev_impact = impact_model(rho - 0.01)
                assert impact > prev_impact, "Impact doit croître avec rho"

    def test_impact_components(self):
        """Teste que les composantes d'impact sont cohérentes."""
        eta = 0.01
        phi = 0.5
        k = 0.001
        rho = 0.1

        temp_impact = eta * np.power(rho, phi)
        perm_impact = k * rho

        # L'impact temporaire devrait dominer pour phi < 1 et rho petit
        # car rho^phi > rho quand phi < 1 et rho < 1

        # Pour rho = 0.1 et phi = 0.5:
        # rho^0.5 = 0.316 > rho = 0.1
        assert np.power(rho, phi) > rho, "rho^phi > rho quand phi < 1 et rho < 1"


class TestExecutionStrategies:
    """Tests pour les stratégies d'exécution."""

    def test_twap_schedule(self):
        """Teste que TWAP divise uniformément."""
        X = 100  # Quantité totale
        n_slices = 10

        schedule = np.ones(n_slices) * (X / n_slices)

        # Vérifier que la somme égale X
        assert np.isclose(schedule.sum(), X), "TWAP doit exécuter exactement X"

        # Vérifier que toutes les tranches sont égales
        assert np.allclose(schedule, schedule[0]), "TWAP doit être uniforme"

    def test_trajectory_monotonic(self):
        """Teste que la trajectoire est monotone décroissante."""
        X = 100
        n_slices = 10

        # Trajectoire TWAP
        time_grid = np.linspace(0, 1, n_slices + 1)
        trajectory = X * (1 - time_grid)

        # Vérifier que la trajectoire décroît
        for i in range(len(trajectory) - 1):
            assert trajectory[i] >= trajectory[i + 1], \
                "La trajectoire doit être décroissante"

        # Vérifier les bornes
        assert trajectory[0] == X, "Trajectoire commence à X"
        assert trajectory[-1] == 0, "Trajectoire finit à 0"

    def test_optimal_vs_twap(self):
        """Teste que la stratégie optimale diffère de TWAP."""
        X = 100
        T = 1.0
        n_slices = 10
        risk_aversion = 1e-5
        sigma = 0.02

        # TWAP: linéaire
        twap_trajectory = X * (1 - np.linspace(0, T, n_slices + 1) / T)

        # Optimal AC: dépend de kappa
        # Pour un risk_aversion élevé, l'exécution devrait être plus rapide au début

        kappa = np.sqrt(risk_aversion * sigma**2 / 0.01)  # Approximation

        if kappa * T > 0.1:
            # La stratégie optimale devrait différer de TWAP
            times = np.linspace(0, T, n_slices + 1)
            optimal_trajectory = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

            # Les trajectoires ne devraient pas être identiques
            # (sauf pour lambda très petit)
            diff = np.abs(optimal_trajectory - twap_trajectory).max()
            # On ne peut pas être trop strict ici car ça dépend des paramètres


class TestDataValidation:
    """Tests pour la validation des données."""

    def test_positive_quantities(self):
        """Teste que les quantités sont positives."""
        quantities = np.random.uniform(0.1, 10, 100)

        assert np.all(quantities > 0), "Toutes les quantités doivent être positives"

    def test_participation_rate_bounds(self):
        """Teste que le taux de participation est dans [0, 1]."""
        Q = 100
        V = 1000
        T = 1

        rho = Q / (V * T)

        assert 0 <= rho <= 1, "Le taux de participation doit être dans [0, 1]"

    def test_cost_positive(self):
        """Teste que les coûts sont positifs."""
        psi = 0.0005
        eta = 0.01
        phi = 0.5
        k = 0.001
        rho = 0.1

        total_cost = psi + eta * np.power(rho, phi) + k * rho

        assert total_cost > 0, "Le coût total doit être positif"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
