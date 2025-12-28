"""Stratégies d'exécution optimale pour le modèle Almgren-Chriss.

Ce module implémente différentes stratégies d'exécution d'ordres:
- TWAP (Time-Weighted Average Price): exécution uniforme dans le temps
- VWAP (Volume-Weighted Average Price): exécution proportionnelle au volume
- Optimal (Almgren-Chriss): minimise le coût total incluant l'impact de marché

Références:
    - Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions
    - Almgren (2003): Optimal Execution with Nonlinear Impact Functions
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class ExecutionParams:
    """Paramètres d'exécution d'un ordre.

    Attributes:
        X: Quantité totale à exécuter (en unités de l'actif)
        T: Horizon d'exécution (en jours)
        n_slices: Nombre de tranches d'exécution
        sigma: Volatilité journalière de l'actif
        S0: Prix initial de l'actif
    """
    X: float           # Quantité totale à exécuter
    T: float           # Horizon d'exécution (jours)
    n_slices: int      # Nombre de tranches
    sigma: float       # Volatilité journalière
    S0: float = 1.0    # Prix initial (normalisé)

    @property
    def tau(self) -> float:
        """Durée d'une tranche."""
        return self.T / self.n_slices

    @property
    def time_grid(self) -> np.ndarray:
        """Grille temporelle."""
        return np.linspace(0, self.T, self.n_slices + 1)


@dataclass
class ImpactParams:
    """Paramètres du modèle d'impact de marché.

    Modèle: C(ρ) = ψ + η·ρ^φ + k·ρ

    Attributes:
        psi: Coûts proportionnels (spread + fees)
        eta: Coefficient d'impact temporaire
        phi: Exposant power-law (typiquement 0.5)
        k: Coefficient d'impact permanent
        daily_volume: Volume journalier moyen
    """
    psi: float          # Spread + fees
    eta: float          # Coefficient temporaire
    phi: float          # Exposant power-law
    k: float            # Coefficient permanent
    daily_volume: float # Volume quotidien

    @classmethod
    def from_calibration(cls, calibration_result: Dict[str, Any], daily_volume: float) -> "ImpactParams":
        """Crée des ImpactParams depuis un résultat de calibration."""
        return cls(
            psi=calibration_result["psi"],
            eta=calibration_result["eta"],
            phi=calibration_result["phi"],
            k=calibration_result["k"],
            daily_volume=daily_volume,
        )


class ExecutionStrategy:
    """Classe de base pour les stratégies d'exécution."""

    def __init__(self, exec_params: ExecutionParams, impact_params: ImpactParams):
        """Initialise la stratégie.

        Args:
            exec_params: Paramètres d'exécution
            impact_params: Paramètres d'impact de marché
        """
        self.exec = exec_params
        self.impact = impact_params

    def compute_schedule(self) -> np.ndarray:
        """Calcule le planning d'exécution.

        Returns:
            Array de taille n_slices avec la quantité à exécuter par tranche
        """
        raise NotImplementedError

    def compute_trajectory(self) -> np.ndarray:
        """Calcule la trajectoire de position restante.

        Returns:
            Array de taille n_slices+1 avec la position restante à chaque instant
        """
        schedule = self.compute_schedule()
        trajectory = np.zeros(self.exec.n_slices + 1)
        trajectory[0] = self.exec.X
        for i in range(self.exec.n_slices):
            trajectory[i + 1] = trajectory[i] - schedule[i]
        return trajectory

    def participation_rate(self, quantity: float) -> float:
        """Calcule le taux de participation pour une quantité donnée."""
        return quantity / (self.impact.daily_volume * self.exec.tau)

    def temporary_impact(self, quantity: float) -> float:
        """Calcule l'impact temporaire pour une quantité donnée.

        Impact = η·ρ^φ où ρ = participation rate
        """
        rho = self.participation_rate(quantity)
        return self.impact.eta * np.power(np.abs(rho), self.impact.phi) * np.sign(rho)

    def permanent_impact(self, quantity: float) -> float:
        """Calcule l'impact permanent pour une quantité donnée.

        Impact = k·ρ où ρ = participation rate
        """
        rho = self.participation_rate(quantity)
        return self.impact.k * rho

    def compute_costs(self) -> Dict[str, float]:
        """Calcule les différentes composantes du coût d'exécution.

        Returns:
            Dictionnaire avec:
                - spread_cost: coût du spread
                - temporary_cost: coût d'impact temporaire
                - permanent_cost: coût d'impact permanent
                - total_cost: coût total
                - cost_bps: coût total en points de base
        """
        schedule = self.compute_schedule()
        trajectory = self.compute_trajectory()

        spread_cost = 0.0
        temporary_cost = 0.0
        permanent_cost = 0.0

        cumulative_permanent = 0.0

        for i in range(self.exec.n_slices):
            q = schedule[i]
            if q == 0:
                continue

            # Coût du spread (proportionnel à la quantité)
            spread_cost += self.impact.psi * q * self.exec.S0

            # Coût d'impact temporaire
            temp_impact = self.temporary_impact(q)
            temporary_cost += temp_impact * q * self.exec.S0

            # Coût d'impact permanent (appliqué à la position restante)
            perm_impact = self.permanent_impact(q)
            cumulative_permanent += perm_impact
            permanent_cost += cumulative_permanent * trajectory[i + 1] * self.exec.S0

        total_cost = spread_cost + temporary_cost + permanent_cost

        # Normaliser par le notionnel
        notional = self.exec.X * self.exec.S0
        cost_bps = (total_cost / notional) * 10000 if notional > 0 else 0

        return {
            "spread_cost": spread_cost,
            "temporary_cost": temporary_cost,
            "permanent_cost": permanent_cost,
            "total_cost": total_cost,
            "cost_bps": cost_bps,
        }

    def compute_risk(self) -> float:
        """Calcule le risque (variance) de l'exécution.

        Risque = σ² × Σ(x_k² × τ) où x_k est la position restante
        """
        trajectory = self.compute_trajectory()
        variance = 0.0

        for i in range(self.exec.n_slices):
            variance += trajectory[i] ** 2 * self.exec.tau

        return self.exec.sigma ** 2 * variance


class TWAP(ExecutionStrategy):
    """Time-Weighted Average Price: exécution uniforme dans le temps.

    La stratégie TWAP divise la quantité totale en tranches égales
    exécutées à intervalles réguliers.

    Avantages:
        - Simple à implémenter
        - Prévisible
        - Réduit le risque de timing

    Inconvénients:
        - Ne prend pas en compte l'impact de marché
        - Sous-optimal en termes de coûts
    """

    def compute_schedule(self) -> np.ndarray:
        """Calcule le planning TWAP (quantités égales)."""
        return np.ones(self.exec.n_slices) * (self.exec.X / self.exec.n_slices)


class VWAP(ExecutionStrategy):
    """Volume-Weighted Average Price: exécution proportionnelle au volume.

    La stratégie VWAP exécute proportionnellement au volume de marché
    attendu à chaque période.

    Attributes:
        volume_profile: Profil de volume intraday (normalisé à 1)
    """

    def __init__(
        self,
        exec_params: ExecutionParams,
        impact_params: ImpactParams,
        volume_profile: Optional[np.ndarray] = None,
    ):
        """Initialise la stratégie VWAP.

        Args:
            exec_params: Paramètres d'exécution
            impact_params: Paramètres d'impact
            volume_profile: Profil de volume par tranche (si None, utilise U-shape)
        """
        super().__init__(exec_params, impact_params)

        if volume_profile is None:
            # Profil U-shape typique (plus de volume en début et fin de journée)
            self.volume_profile = self._generate_ushape_profile()
        else:
            self.volume_profile = volume_profile / volume_profile.sum()

    def _generate_ushape_profile(self) -> np.ndarray:
        """Génère un profil de volume en U typique."""
        n = self.exec.n_slices
        x = np.linspace(0, 1, n)
        # Forme en U: plus de volume en début et fin
        profile = 1.5 - np.abs(x - 0.5) * 2 + 0.5 * np.cos(np.pi * x) ** 2
        return profile / profile.sum()

    def compute_schedule(self) -> np.ndarray:
        """Calcule le planning VWAP (proportionnel au volume)."""
        return self.volume_profile * self.exec.X


class OptimalAC(ExecutionStrategy):
    """Stratégie optimale Almgren-Chriss.

    Cette stratégie minimise le coût d'exécution total (impact + risque)
    en utilisant le modèle Almgren-Chriss avec impact power-law.

    Le problème d'optimisation est:
        min E[C] + λ·Var[C]

    où λ est le paramètre d'aversion au risque.

    Attributes:
        risk_aversion: Paramètre d'aversion au risque λ
    """

    def __init__(
        self,
        exec_params: ExecutionParams,
        impact_params: ImpactParams,
        risk_aversion: float = 1e-6,
    ):
        """Initialise la stratégie optimale.

        Args:
            exec_params: Paramètres d'exécution
            impact_params: Paramètres d'impact
            risk_aversion: Paramètre d'aversion au risque (λ)
        """
        super().__init__(exec_params, impact_params)
        self.risk_aversion = risk_aversion

    def _compute_kappa(self) -> float:
        """Calcule le paramètre κ de la solution AC.

        κ = sqrt(λσ² / (η/τ^φ))

        où τ est la durée d'une tranche.
        """
        tau = self.exec.tau
        # Pour le modèle power-law, on linéarise autour du point moyen
        avg_trade = self.exec.X / self.exec.n_slices
        rho_avg = avg_trade / (self.impact.daily_volume * tau)

        # Dérivée de l'impact temporaire: d(η·ρ^φ)/dρ = η·φ·ρ^(φ-1)
        # On utilise une approximation linéaire locale
        if rho_avg > 0:
            eta_eff = self.impact.eta * self.impact.phi * np.power(rho_avg, self.impact.phi - 1)
        else:
            eta_eff = self.impact.eta

        # Ajouter l'impact permanent
        eta_eff += self.impact.k

        if eta_eff <= 0:
            return 0.01  # Valeur par défaut si problème

        kappa_sq = self.risk_aversion * self.exec.sigma ** 2 / eta_eff
        return np.sqrt(max(kappa_sq, 1e-10))

    def compute_schedule(self) -> np.ndarray:
        """Calcule le planning optimal Almgren-Chriss.

        La solution optimale pour le modèle linéaire est:
            x_k = X · sinh(κ(T-t_k)) / sinh(κT)

        où κ dépend de l'aversion au risque et des paramètres d'impact.
        """
        kappa = self._compute_kappa()
        T = self.exec.T
        times = self.exec.time_grid[:-1]  # Temps de début de chaque tranche

        # Éviter les problèmes numériques
        if kappa * T > 100:
            # Approximation pour grand κT: exécution rapide au début
            trajectory = self.exec.X * np.exp(-kappa * times)
        elif kappa * T < 0.01:
            # Approximation pour petit κT: proche de TWAP
            trajectory = self.exec.X * (1 - times / T)
        else:
            # Solution exacte
            trajectory = self.exec.X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

        # Calculer les quantités à exécuter (différences)
        schedule = np.zeros(self.exec.n_slices)
        for i in range(self.exec.n_slices):
            if i < self.exec.n_slices - 1:
                t_next = self.exec.time_grid[i + 1]
                if kappa * T > 100:
                    next_pos = self.exec.X * np.exp(-kappa * t_next)
                elif kappa * T < 0.01:
                    next_pos = self.exec.X * (1 - t_next / T)
                else:
                    next_pos = self.exec.X * np.sinh(kappa * (T - t_next)) / np.sinh(kappa * T)
                schedule[i] = trajectory[i] - next_pos
            else:
                schedule[i] = trajectory[i]  # Dernière tranche: exécuter le reste

        return schedule


class OptimalACVariable(ExecutionStrategy):
    """Stratégie optimale Almgren-Chriss avec volume VARIABLE.

    Extension du modèle AC pour prendre en compte un profil de volume
    intraday V(t) au lieu d'un volume constant V.

    Le taux de participation devient:
        ρ(t) = v(t) / V(t)

    où V(t) est le volume instantané du marché.

    La solution n'a plus de forme analytique et nécessite une résolution
    numérique du problème aux limites (BVP).

    Attributes:
        risk_aversion: Paramètre d'aversion au risque λ
        volume_profile: Profil de volume normalisé (somme = 1)
    """

    def __init__(
        self,
        exec_params: ExecutionParams,
        impact_params: ImpactParams,
        risk_aversion: float = 1e-6,
        volume_profile: Optional[np.ndarray] = None,
    ):
        """Initialise la stratégie optimale avec volume variable.

        Args:
            exec_params: Paramètres d'exécution
            impact_params: Paramètres d'impact
            risk_aversion: Paramètre d'aversion au risque (λ)
            volume_profile: Profil de volume par tranche (normalisé).
                           Si None, utilise un profil constant (équivalent à OptimalAC)
        """
        super().__init__(exec_params, impact_params)
        self.risk_aversion = risk_aversion

        if volume_profile is None:
            # Profil constant = modèle AC classique
            self.volume_profile = np.ones(exec_params.n_slices) / exec_params.n_slices
        else:
            # Normaliser le profil
            self.volume_profile = volume_profile / volume_profile.sum()

        # Volume instantané par tranche (en unités/jour)
        # volume_profile donne la FRACTION du volume quotidien par tranche
        # V(t) = daily_volume * volume_profile[t] / tau
        self.volume_per_slice = self.impact.daily_volume * self.volume_profile

    def _volume_at_slice(self, i: int) -> float:
        """Retourne le volume du marché à la tranche i (en unités/jour)."""
        # Ajuster pour que le volume total sur la journée = daily_volume
        return self.volume_per_slice[i] / self.exec.tau

    def participation_rate_variable(self, quantity: float, slice_idx: int) -> float:
        """Calcule le taux de participation pour une tranche donnée.

        ρ_i = n_i / (V_i * τ)

        où V_i est le volume à la tranche i.
        """
        V_i = self._volume_at_slice(slice_idx)
        if V_i <= 0:
            return 0.0
        return quantity / (V_i * self.exec.tau)

    def _compute_cost_gradient(self, schedule: np.ndarray) -> np.ndarray:
        """Calcule le gradient du coût par rapport au schedule.

        Utilisé pour l'optimisation numérique.
        """
        n = len(schedule)
        gradient = np.zeros(n)
        tau = self.exec.tau

        for i in range(n):
            V_i = self._volume_at_slice(i)
            rho_i = schedule[i] / (V_i * tau) if V_i > 0 else 0

            # Dérivée du coût temporaire: d/dn[η·ρ^φ] = η·φ·ρ^(φ-1) / (V·τ)
            if rho_i > 0:
                d_temp = self.impact.eta * self.impact.phi * np.power(rho_i, self.impact.phi - 1)
                d_temp /= (V_i * tau)
            else:
                d_temp = 0

            # Dérivée du coût permanent: d/dn[k·ρ] = k / (V·τ)
            d_perm = self.impact.k / (V_i * tau) if V_i > 0 else 0

            # Dérivée du spread: constante
            d_spread = self.impact.psi / (V_i * tau) if V_i > 0 else 0

            gradient[i] = d_temp + d_perm + d_spread

        # Ajouter le gradient du risque
        trajectory = np.zeros(n + 1)
        trajectory[0] = self.exec.X
        for i in range(n):
            trajectory[i + 1] = trajectory[i] - schedule[i]

        for i in range(n):
            # Position moyenne pendant la tranche i
            q_avg = (trajectory[i] + trajectory[i + 1]) / 2
            # Contribution au risque: λ·σ²·q·τ
            gradient[i] -= self.risk_aversion * self.exec.sigma ** 2 * q_avg * tau

        return gradient

    def compute_schedule(self) -> np.ndarray:
        """Calcule le planning optimal avec volume variable.

        Utilise une optimisation numérique pour minimiser:
            E[C] + λ·Var[C]

        sous la contrainte que sum(schedule) = X.
        """
        from scipy.optimize import minimize

        n = self.exec.n_slices
        X = self.exec.X

        def objective(schedule):
            """Fonction objectif: coût total + risque."""
            # Reconstruire la trajectoire
            trajectory = np.zeros(n + 1)
            trajectory[0] = X
            for i in range(n):
                trajectory[i + 1] = trajectory[i] - schedule[i]

            # Coût d'exécution
            total_cost = 0.0
            tau = self.exec.tau

            for i in range(n):
                V_i = self._volume_at_slice(i)
                if V_i <= 0:
                    continue

                rho_i = schedule[i] / (V_i * tau)

                # Spread
                total_cost += self.impact.psi * abs(rho_i) * tau

                # Impact temporaire
                total_cost += self.impact.eta * np.power(abs(rho_i), self.impact.phi) * tau

                # Impact permanent (approximation)
                total_cost += self.impact.k * abs(rho_i) * tau

            # Risque (variance)
            risk = 0.0
            for i in range(n):
                q_avg = (trajectory[i] + trajectory[i + 1]) / 2
                risk += self.exec.sigma ** 2 * q_avg ** 2 * tau

            return total_cost + self.risk_aversion * risk

        # Contrainte: somme des quantités = X
        constraints = {'type': 'eq', 'fun': lambda s: np.sum(s) - X}

        # Bornes: quantités positives
        bounds = [(0, X) for _ in range(n)]

        # Point de départ: VWAP (proportionnel au volume)
        x0 = self.volume_profile * X

        # Optimisation
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if result.success:
            return result.x
        else:
            # Fallback sur VWAP si l'optimisation échoue
            logger.warning(f"Optimisation AC variable: {result.message}")
            logger.warning("Fallback sur VWAP")
            return self.volume_profile * X

    def compute_costs(self) -> Dict[str, float]:
        """Calcule les coûts avec volume variable.

        Override pour utiliser le volume variable par tranche.
        """
        schedule = self.compute_schedule()
        trajectory = self.compute_trajectory()
        tau = self.exec.tau

        spread_cost = 0.0
        temporary_cost = 0.0
        permanent_cost = 0.0

        for i in range(self.exec.n_slices):
            V_i = self._volume_at_slice(i)
            if V_i <= 0:
                continue

            n_i = schedule[i]
            rho_i = n_i / (V_i * tau)

            # Spread
            spread_cost += self.impact.psi * abs(rho_i) * tau

            # Impact temporaire
            temporary_cost += self.impact.eta * np.power(abs(rho_i), self.impact.phi) * tau

            # Impact permanent
            permanent_cost += self.impact.k * abs(rho_i) * tau

        total_cost = spread_cost + temporary_cost + permanent_cost

        # Coût en bps
        notional = abs(self.exec.X) * self.exec.S0
        cost_bps = (total_cost * self.exec.S0 / notional) * 10000 if notional > 0 else 0

        return {
            "spread_cost": spread_cost,
            "temporary_cost": temporary_cost,
            "permanent_cost": permanent_cost,
            "total_cost": total_cost,
            "cost_bps": cost_bps,
        }


def compare_strategies(
    exec_params: ExecutionParams,
    impact_params: ImpactParams,
    risk_aversion: float = 1e-6,
    volume_profile: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare les différentes stratégies d'exécution.

    Args:
        exec_params: Paramètres d'exécution
        impact_params: Paramètres d'impact
        risk_aversion: Aversion au risque pour la stratégie optimale
        volume_profile: Profil de volume pour VWAP (optionnel)

    Returns:
        Dictionnaire avec les résultats pour chaque stratégie
    """
    strategies = {
        "TWAP": TWAP(exec_params, impact_params),
        "VWAP": VWAP(exec_params, impact_params, volume_profile),
        "Optimal": OptimalAC(exec_params, impact_params, risk_aversion),
    }

    results = {}

    for name, strategy in strategies.items():
        schedule = strategy.compute_schedule()
        trajectory = strategy.compute_trajectory()
        costs = strategy.compute_costs()
        risk = strategy.compute_risk()

        results[name] = {
            "schedule": schedule,
            "trajectory": trajectory,
            "costs": costs,
            "risk": risk,
            "risk_adjusted_cost": costs["total_cost"] + risk_aversion * risk,
        }

    return results


def plot_strategy_comparison(
    exec_params: ExecutionParams,
    impact_params: ImpactParams,
    risk_aversion: float = 1e-6,
    save_path: Optional[str] = None,
) -> None:
    """Trace une comparaison visuelle des stratégies.

    Args:
        exec_params: Paramètres d'exécution
        impact_params: Paramètres d'impact
        risk_aversion: Aversion au risque
        save_path: Chemin de sauvegarde (optionnel)
    """
    results = compare_strategies(exec_params, impact_params, risk_aversion)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"TWAP": "blue", "VWAP": "orange", "Optimal": "green"}
    time_grid = exec_params.time_grid

    # 1. Trajectoires de position
    ax = axes[0, 0]
    for name, data in results.items():
        ax.plot(time_grid, data["trajectory"], "-o", label=name,
                color=colors[name], linewidth=2, markersize=4)
    ax.set_xlabel("Temps (jours)")
    ax.set_ylabel("Position restante")
    ax.set_title("Trajectoire d'Exécution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Planning d'exécution
    ax = axes[0, 1]
    width = 0.25
    x = np.arange(exec_params.n_slices)
    for i, (name, data) in enumerate(results.items()):
        ax.bar(x + i * width, data["schedule"], width, label=name, color=colors[name], alpha=0.7)
    ax.set_xlabel("Tranche")
    ax.set_ylabel("Quantité exécutée")
    ax.set_title("Planning d'Exécution par Tranche")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Décomposition des coûts
    ax = axes[1, 0]
    strategy_names = list(results.keys())
    x = np.arange(len(strategy_names))
    width = 0.6

    spread_costs = [results[s]["costs"]["spread_cost"] for s in strategy_names]
    temp_costs = [results[s]["costs"]["temporary_cost"] for s in strategy_names]
    perm_costs = [results[s]["costs"]["permanent_cost"] for s in strategy_names]

    ax.bar(x, spread_costs, width, label="Spread", alpha=0.7)
    ax.bar(x, temp_costs, width, bottom=spread_costs, label="Temporaire", alpha=0.7)
    ax.bar(x, perm_costs, width, bottom=np.array(spread_costs) + np.array(temp_costs),
           label="Permanent", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names)
    ax.set_ylabel("Coût")
    ax.set_title("Décomposition des Coûts")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Résumé coût vs risque
    ax = axes[1, 1]
    for name, data in results.items():
        ax.scatter(np.sqrt(data["risk"]), data["costs"]["cost_bps"],
                   s=150, label=name, color=colors[name])
    ax.set_xlabel("Risque (écart-type)")
    ax.set_ylabel("Coût (bps)")
    ax.set_title("Frontière Coût-Risque")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")

    plt.show()


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Affiche un tableau de comparaison des stratégies.

    Args:
        results: Résultats de compare_strategies()
    """
    print("\n" + "=" * 70)
    print("COMPARAISON DES STRATÉGIES D'EXÉCUTION")
    print("=" * 70)
    print(f"{'Stratégie':<12} {'Coût Total':>12} {'Coût (bps)':>12} {'Risque':>12} {'Coût Ajusté':>14}")
    print("-" * 70)

    for name, data in results.items():
        print(f"{name:<12} {data['costs']['total_cost']:>12.6f} "
              f"{data['costs']['cost_bps']:>12.2f} "
              f"{np.sqrt(data['risk']):>12.6f} "
              f"{data['risk_adjusted_cost']:>14.6f}")

    print("=" * 70)

    # Trouver la meilleure stratégie
    best = min(results.keys(), key=lambda x: results[x]["risk_adjusted_cost"])
    print(f"\nStratégie optimale (coût ajusté au risque): {best}")


# =============================================================================
# PROFILS DE VOLUME INTRADAY
# =============================================================================

def generate_volume_profile(
    n_slices: int,
    profile_type: str = "u_shape",
    **kwargs
) -> np.ndarray:
    """Génère un profil de volume intraday normalisé.

    Args:
        n_slices: Nombre de tranches temporelles
        profile_type: Type de profil parmi:
            - "flat": Volume constant (comme TWAP)
            - "u_shape": Forme en U (actions US typique)
            - "u_shape_strong": U très prononcé (haute liquidité open/close)
            - "crypto_24h": Profil crypto sur 24h (pics asiatiques/européens/US)
            - "morning_heavy": Plus de volume le matin
            - "afternoon_heavy": Plus de volume l'après-midi
        **kwargs: Paramètres spécifiques au profil

    Returns:
        Array normalisé (somme = 1) de taille n_slices

    Example:
        >>> profile = generate_volume_profile(390, "u_shape")
        >>> vwap = VWAP(exec_params, impact_params, volume_profile=profile)
    """
    x = np.linspace(0, 1, n_slices)

    if profile_type == "flat":
        profile = np.ones(n_slices)

    elif profile_type == "u_shape":
        # U-shape classique pour actions US (9h30-16h)
        # Plus de volume à l'ouverture et à la clôture
        alpha = kwargs.get("alpha", 0.3)  # Intensité du U
        profile = 1 + alpha * (4 * (x - 0.5) ** 2)

    elif profile_type == "u_shape_strong":
        # U très prononcé (ex: annonces importantes)
        profile = 1 + 0.8 * (4 * (x - 0.5) ** 2)
        # Spike à l'ouverture
        profile[:int(n_slices * 0.05)] *= 1.5
        # Spike à la clôture
        profile[-int(n_slices * 0.1):] *= 1.3

    elif profile_type == "crypto_24h":
        # Profil 24h pour crypto avec 3 pics régionaux
        # Asie (0-8h UTC), Europe (8-16h UTC), US (16-24h UTC)
        asia_peak = 0.125     # ~3h UTC
        europe_peak = 0.458   # ~11h UTC
        us_peak = 0.792       # ~19h UTC

        profile = (
            0.3 * np.exp(-((x - asia_peak) ** 2) / 0.01) +
            0.35 * np.exp(-((x - europe_peak) ** 2) / 0.01) +
            0.35 * np.exp(-((x - us_peak) ** 2) / 0.01) +
            0.3  # Volume de base
        )

    elif profile_type == "morning_heavy":
        # Plus de volume le matin (ex: news overnight)
        profile = 1.5 - x

    elif profile_type == "afternoon_heavy":
        # Plus de volume l'après-midi
        profile = 0.5 + x

    else:
        raise ValueError(f"Profil inconnu: {profile_type}. "
                        f"Options: flat, u_shape, u_shape_strong, crypto_24h, "
                        f"morning_heavy, afternoon_heavy")

    # Normaliser pour que la somme = 1
    return profile / profile.sum()


def compute_volume_profile_from_data(
    df: "pd.DataFrame",
    time_col: str = "timestamp",
    volume_col: str = "volume",
    n_slices: int = None,
    trading_hours: Tuple[int, int] = (0, 24),
) -> np.ndarray:
    """Calcule un profil de volume empirique à partir de données historiques.

    Args:
        df: DataFrame avec colonnes timestamp et volume
        time_col: Nom de la colonne timestamp
        volume_col: Nom de la colonne volume
        n_slices: Nombre de tranches (si None, 1 par minute sur les heures de trading)
        trading_hours: Tuple (heure_debut, heure_fin) en heures locales

    Returns:
        Profil de volume normalisé

    Example:
        >>> df = pd.read_parquet('data/processed/crypto/BTCUSDT_1m.parquet')
        >>> profile = compute_volume_profile_from_data(df, n_slices=48)  # 30min slices
        >>> vwap = VWAP(exec_params, impact_params, volume_profile=profile)
    """
    import pandas as pd

    df = df.copy()

    # Convertir timestamp si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Extraire l'heure du jour
    df["hour"] = df[time_col].dt.hour
    df["minute"] = df[time_col].dt.minute
    df["time_of_day"] = df["hour"] + df["minute"] / 60

    # Filtrer les heures de trading
    start_hour, end_hour = trading_hours
    if start_hour < end_hour:
        df = df[(df["hour"] >= start_hour) & (df["hour"] < end_hour)]
    else:  # Pour crypto 24h, pas de filtre
        pass

    # Déterminer le nombre de slices
    if n_slices is None:
        # Par défaut: 1 slice par minute de trading
        minutes_per_day = (end_hour - start_hour) * 60 if end_hour > start_hour else 24 * 60
        n_slices = minutes_per_day

    # Calculer les bins
    if end_hour > start_hour:
        df["bin"] = pd.cut(
            df["time_of_day"],
            bins=np.linspace(start_hour, end_hour, n_slices + 1),
            labels=range(n_slices),
            include_lowest=True
        )
    else:
        # 24h trading
        df["bin"] = pd.cut(
            df["time_of_day"],
            bins=np.linspace(0, 24, n_slices + 1),
            labels=range(n_slices),
            include_lowest=True
        )

    # Calculer le volume moyen par bin
    volume_by_bin = df.groupby("bin", observed=True)[volume_col].mean()

    # Créer le profil
    profile = np.zeros(n_slices)
    for i, val in volume_by_bin.items():
        if i is not None and not np.isnan(val):
            profile[int(i)] = val

    # Remplacer les zéros par la moyenne (pour les bins sans données)
    if profile.sum() > 0:
        mean_vol = profile[profile > 0].mean()
        profile[profile == 0] = mean_vol * 0.5
    else:
        profile = np.ones(n_slices)

    return profile / profile.sum()


def plot_volume_profile(
    profile: np.ndarray,
    title: str = "Profil de Volume Intraday",
    trading_hours: Tuple[int, int] = (9.5, 16),
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Visualise un profil de volume.

    Args:
        profile: Profil de volume normalisé
        title: Titre du graphique
        trading_hours: Heures de trading (pour l'axe X)
        figsize: Taille de la figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = len(profile)
    start_h, end_h = trading_hours

    # Créer l'axe temporel
    if end_h > start_h:
        hours = np.linspace(start_h, end_h, n)
        xlabel = "Heure (locale)"
    else:
        hours = np.linspace(0, 24, n)
        xlabel = "Heure (UTC)"

    # Barres de volume
    ax.bar(hours, profile * 100, width=(end_h - start_h) / n if end_h > start_h else 24/n,
           alpha=0.7, color='steelblue', edgecolor='navy')

    # Ligne de moyenne
    avg = 100 / n
    ax.axhline(avg, color='red', linestyle='--', linewidth=2,
               label=f'Moyenne ({avg:.2f}%)')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("% du Volume Journalier", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_profiles(profiles: Dict[str, np.ndarray], n_slices: int = 100) -> None:
    """Compare plusieurs profils de volume visuellement.

    Args:
        profiles: Dictionnaire {nom: profil}
        n_slices: Nombre de slices pour normalisation
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.linspace(0, 100, n_slices)

    for name, profile in profiles.items():
        # Interpoler si nécessaire
        if len(profile) != n_slices:
            from scipy.interpolate import interp1d
            x_orig = np.linspace(0, 100, len(profile))
            f = interp1d(x_orig, profile, kind='linear')
            profile = f(x)

        ax.plot(x, profile * 100, linewidth=2, label=name)

    ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Flat (TWAP)')

    ax.set_xlabel("Progression dans la journée (%)", fontsize=12)
    ax.set_ylabel("Volume relatif (%)", fontsize=12)
    ax.set_title("Comparaison des Profils de Volume", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
