"""Modèle Almgren-Chriss pour l'exécution optimale d'ordres.

Ce module implémente le modèle théorique d'Almgren-Chriss (2000) pour
l'exécution optimale de transactions en présence d'impact de marché.

Le modèle décompose l'impact de marché en:
- Impact permanent: changement durable du prix (proportionnel au volume)
- Impact temporaire: coût d'exécution immédiat (fonction power-law du taux de participation)

Références:
    - Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions.
      Journal of Risk, 3, 5-40.
    - Almgren, R. (2003). Optimal execution with nonlinear impact functions and trading
      enhanced risk. Applied Mathematical Finance, 10(1), 1-18.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import matplotlib.pyplot as plt


@dataclass
class MarketParameters:
    """Paramètres de marché pour le modèle Almgren-Chriss.

    Attributes:
        S0: Prix initial de l'actif
        sigma: Volatilité journalière (écart-type des rendements)
        daily_volume: Volume journalier moyen (ADV)
    """
    S0: float
    sigma: float
    daily_volume: float

    def __post_init__(self):
        if self.S0 <= 0:
            raise ValueError("Le prix initial doit être positif")
        if self.sigma <= 0:
            raise ValueError("La volatilité doit être positive")
        if self.daily_volume <= 0:
            raise ValueError("Le volume quotidien doit être positif")


@dataclass
class ImpactParameters:
    """Paramètres d'impact de marché.

    Modèle d'impact total:
        C(ρ) = ψ + η·ρ^φ + k·ρ

    où ρ = Q/(V·T) est le taux de participation.

    Attributes:
        psi: Coûts proportionnels fixes (spread + frais)
        eta: Coefficient d'impact temporaire
        phi: Exposant power-law (typiquement 0.5 selon la théorie)
        k: Coefficient d'impact permanent (Kyle's lambda)
    """
    psi: float
    eta: float
    phi: float
    k: float

    def __post_init__(self):
        if self.psi < 0:
            raise ValueError("psi doit être >= 0")
        if self.eta < 0:
            raise ValueError("eta doit être >= 0")
        if not 0 < self.phi < 2:
            raise ValueError("phi doit être dans (0, 2)")
        if self.k < 0:
            raise ValueError("k doit être >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ImpactParameters":
        """Crée des paramètres depuis un dictionnaire."""
        return cls(
            psi=d["psi"],
            eta=d["eta"],
            phi=d["phi"],
            k=d["k"],
        )

    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire."""
        return {
            "psi": self.psi,
            "eta": self.eta,
            "phi": self.phi,
            "k": self.k,
        }


@dataclass
class ExecutionPlan:
    """Plan d'exécution résultant de l'optimisation.

    Attributes:
        time_grid: Points temporels [0, t1, t2, ..., T]
        trajectory: Position restante à chaque instant
        schedule: Quantité à exécuter dans chaque intervalle
        expected_cost: Coût d'exécution espéré
        variance: Variance du coût
    """
    time_grid: np.ndarray
    trajectory: np.ndarray
    schedule: np.ndarray
    expected_cost: float
    variance: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)


class AlmgrenChrissModel:
    """Modèle Almgren-Chriss pour l'exécution optimale.

    Ce modèle résout le problème d'optimisation:
        min E[C] + λ·Var[C]

    où:
        - C est le coût d'exécution
        - λ est le paramètre d'aversion au risque
        - E[C] inclut l'impact permanent et temporaire
        - Var[C] est la variance due à la volatilité du prix
    """

    def __init__(
        self,
        market_params: MarketParameters,
        impact_params: ImpactParameters,
        risk_aversion: float = 1e-6,
    ):
        """Initialise le modèle.

        Args:
            market_params: Paramètres de marché
            impact_params: Paramètres d'impact
            risk_aversion: Coefficient d'aversion au risque λ
        """
        self.market = market_params
        self.impact = impact_params
        self.risk_aversion = risk_aversion

    def participation_rate(self, quantity: float, duration: float) -> float:
        """Calcule le taux de participation.

        Args:
            quantity: Quantité à exécuter
            duration: Durée d'exécution (jours)

        Returns:
            Taux de participation ρ = Q/(V·T)
        """
        return quantity / (self.market.daily_volume * duration)

    def temporary_impact(self, quantity: float, duration: float) -> float:
        """Calcule l'impact temporaire (en proportion du prix).

        L'impact temporaire suit une loi de puissance:
            g(ρ) = η·|ρ|^φ·sign(ρ)

        Args:
            quantity: Quantité exécutée
            duration: Durée d'exécution

        Returns:
            Impact temporaire relatif
        """
        rho = self.participation_rate(quantity, duration)
        return self.impact.eta * np.power(np.abs(rho), self.impact.phi) * np.sign(rho)

    def permanent_impact(self, quantity: float, duration: float) -> float:
        """Calcule l'impact permanent (en proportion du prix).

        L'impact permanent est linéaire:
            h(ρ) = k·ρ

        Args:
            quantity: Quantité exécutée
            duration: Durée d'exécution

        Returns:
            Impact permanent relatif
        """
        rho = self.participation_rate(quantity, duration)
        return self.impact.k * rho

    def total_impact(self, quantity: float, duration: float) -> float:
        """Calcule l'impact total (temporaire + permanent + spread).

        C(ρ) = ψ + η·ρ^φ + k·ρ

        Args:
            quantity: Quantité exécutée
            duration: Durée d'exécution

        Returns:
            Impact total relatif
        """
        return (
            self.impact.psi
            + self.temporary_impact(quantity, duration)
            + self.permanent_impact(quantity, duration)
        )

    def compute_kappa(self, avg_trade_rate: float) -> float:
        """Calcule le paramètre κ pour la solution optimale.

        κ² = λσ² / η_eff

        où η_eff est le coefficient d'impact effectif linéarisé.

        Args:
            avg_trade_rate: Taux de trading moyen par période

        Returns:
            Paramètre κ
        """
        # Linéarisation de l'impact autour du point moyen
        if avg_trade_rate > 0:
            eta_eff = (
                self.impact.eta
                * self.impact.phi
                * np.power(avg_trade_rate, self.impact.phi - 1)
            )
        else:
            eta_eff = self.impact.eta

        # Ajouter l'impact permanent
        eta_eff += self.impact.k

        if eta_eff <= 0:
            return 0.01

        kappa_sq = self.risk_aversion * self.market.sigma ** 2 / eta_eff
        return np.sqrt(max(kappa_sq, 1e-10))

    def optimal_trajectory(
        self,
        X: float,
        T: float,
        n_steps: int,
    ) -> ExecutionPlan:
        """Calcule la trajectoire d'exécution optimale.

        La solution optimale du modèle AC linéaire est:
            x(t) = X · sinh(κ(T-t)) / sinh(κT)

        Pour le modèle power-law, on utilise une approximation par linéarisation.

        Args:
            X: Quantité totale à exécuter
            T: Horizon d'exécution (jours)
            n_steps: Nombre de pas de temps

        Returns:
            ExecutionPlan avec la trajectoire optimale
        """
        tau = T / n_steps  # Durée d'un pas
        time_grid = np.linspace(0, T, n_steps + 1)

        # Estimation du taux moyen pour linéarisation
        avg_trade = X / n_steps
        avg_rate = self.participation_rate(avg_trade, tau)

        # Calcul de κ
        kappa = self.compute_kappa(avg_rate)

        # Calcul de la trajectoire
        trajectory = np.zeros(n_steps + 1)
        trajectory[0] = X

        for i, t in enumerate(time_grid[1:], 1):
            if kappa * T > 100:
                # Grand κ: exécution rapide
                trajectory[i] = X * np.exp(-kappa * t)
            elif kappa * T < 0.01:
                # Petit κ: proche de TWAP
                trajectory[i] = X * (1 - t / T)
            else:
                # Solution exacte
                trajectory[i] = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

        # Assurer que la dernière position est 0
        trajectory[-1] = 0

        # Calcul du schedule (quantités à exécuter)
        schedule = -np.diff(trajectory)

        # Calcul des coûts
        expected_cost, variance, breakdown = self._compute_execution_costs(
            trajectory, schedule, tau
        )

        return ExecutionPlan(
            time_grid=time_grid,
            trajectory=trajectory,
            schedule=schedule,
            expected_cost=expected_cost,
            variance=variance,
            cost_breakdown=breakdown,
        )

    def twap_trajectory(self, X: float, T: float, n_steps: int) -> ExecutionPlan:
        """Calcule la trajectoire TWAP (benchmark).

        Args:
            X: Quantité totale
            T: Horizon
            n_steps: Nombre de pas

        Returns:
            ExecutionPlan pour TWAP
        """
        tau = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)

        # TWAP: décroissance linéaire
        trajectory = X * (1 - time_grid / T)
        schedule = np.ones(n_steps) * (X / n_steps)

        expected_cost, variance, breakdown = self._compute_execution_costs(
            trajectory, schedule, tau
        )

        return ExecutionPlan(
            time_grid=time_grid,
            trajectory=trajectory,
            schedule=schedule,
            expected_cost=expected_cost,
            variance=variance,
            cost_breakdown=breakdown,
        )

    def _compute_execution_costs(
        self,
        trajectory: np.ndarray,
        schedule: np.ndarray,
        tau: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        """Calcule les coûts d'exécution pour une trajectoire donnée.

        Args:
            trajectory: Position restante à chaque instant
            schedule: Quantités exécutées par période
            tau: Durée d'une période

        Returns:
            (coût espéré, variance, décomposition)
        """
        S0 = self.market.S0
        n_steps = len(schedule)

        spread_cost = 0.0
        temp_cost = 0.0
        perm_cost = 0.0

        cumulative_perm_impact = 0.0

        for i in range(n_steps):
            q = schedule[i]
            if q == 0:
                continue

            # Coût du spread
            spread_cost += self.impact.psi * q * S0

            # Impact temporaire
            temp_impact = self.temporary_impact(q, tau)
            temp_cost += temp_impact * q * S0

            # Impact permanent (affecte les transactions futures)
            perm_impact = self.permanent_impact(q, tau)
            cumulative_perm_impact += perm_impact
            # Le coût permanent est payé sur la position restante
            perm_cost += cumulative_perm_impact * trajectory[i + 1] * S0

        expected_cost = spread_cost + temp_cost + perm_cost

        # Variance due à la volatilité
        variance = 0.0
        for i in range(n_steps):
            variance += trajectory[i] ** 2 * tau

        variance *= self.market.sigma ** 2

        breakdown = {
            "spread_cost": spread_cost,
            "temporary_cost": temp_cost,
            "permanent_cost": perm_cost,
            "total_cost": expected_cost,
            "cost_bps": expected_cost / (trajectory[0] * S0) * 10000 if trajectory[0] > 0 else 0,
        }

        return expected_cost, variance, breakdown

    def efficient_frontier(
        self,
        X: float,
        T: float,
        n_steps: int,
        risk_aversions: Optional[List[float]] = None,
    ) -> List[Tuple[float, float, float]]:
        """Calcule la frontière efficiente coût-risque.

        Args:
            X: Quantité à exécuter
            T: Horizon
            n_steps: Nombre de pas
            risk_aversions: Liste de valeurs de λ à tester

        Returns:
            Liste de (λ, coût espéré, écart-type)
        """
        if risk_aversions is None:
            risk_aversions = np.logspace(-8, -3, 20)

        frontier = []

        for lam in risk_aversions:
            # Temporairement changer l'aversion au risque
            original_lambda = self.risk_aversion
            self.risk_aversion = lam

            plan = self.optimal_trajectory(X, T, n_steps)

            frontier.append((
                lam,
                plan.expected_cost,
                np.sqrt(plan.variance),
            ))

            self.risk_aversion = original_lambda

        return frontier

    def plot_trajectory_comparison(
        self,
        X: float,
        T: float,
        n_steps: int,
        save_path: Optional[str] = None,
    ) -> None:
        """Compare visuellement la trajectoire optimale vs TWAP.

        Args:
            X: Quantité à exécuter
            T: Horizon
            n_steps: Nombre de pas
            save_path: Chemin de sauvegarde (optionnel)
        """
        optimal = self.optimal_trajectory(X, T, n_steps)
        twap = self.twap_trajectory(X, T, n_steps)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Trajectoires
        ax = axes[0, 0]
        ax.plot(optimal.time_grid, optimal.trajectory, "-o",
                label="Optimal AC", linewidth=2, markersize=4)
        ax.plot(twap.time_grid, twap.trajectory, "-s",
                label="TWAP", linewidth=2, markersize=4)
        ax.set_xlabel("Temps (jours)")
        ax.set_ylabel("Position restante")
        ax.set_title("Trajectoires d'Exécution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Schedules
        ax = axes[0, 1]
        width = 0.35
        x = np.arange(n_steps)
        ax.bar(x - width/2, optimal.schedule, width, label="Optimal AC", alpha=0.7)
        ax.bar(x + width/2, twap.schedule, width, label="TWAP", alpha=0.7)
        ax.set_xlabel("Période")
        ax.set_ylabel("Quantité exécutée")
        ax.set_title("Planning d'Exécution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Décomposition des coûts
        ax = axes[1, 0]
        strategies = ["Optimal AC", "TWAP"]
        costs_data = {
            "Spread": [optimal.cost_breakdown["spread_cost"],
                      twap.cost_breakdown["spread_cost"]],
            "Temporaire": [optimal.cost_breakdown["temporary_cost"],
                          twap.cost_breakdown["temporary_cost"]],
            "Permanent": [optimal.cost_breakdown["permanent_cost"],
                         twap.cost_breakdown["permanent_cost"]],
        }

        x = np.arange(len(strategies))
        width = 0.6
        bottom = np.zeros(len(strategies))

        for cost_type, values in costs_data.items():
            ax.bar(x, values, width, label=cost_type, bottom=bottom, alpha=0.7)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.set_ylabel("Coût")
        ax.set_title("Décomposition des Coûts")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Résumé
        ax = axes[1, 1]
        ax.axis('off')

        summary = f"""
        RÉSUMÉ DE L'EXÉCUTION
        {'='*40}

        Quantité totale: {X:,.0f}
        Horizon: {T:.2f} jours
        Nombre de tranches: {n_steps}
        Aversion au risque (λ): {self.risk_aversion:.2e}

        OPTIMAL AC:
          Coût total: {optimal.expected_cost:.4f}
          Coût (bps): {optimal.cost_breakdown['cost_bps']:.2f}
          Risque (σ): {np.sqrt(optimal.variance):.4f}

        TWAP:
          Coût total: {twap.expected_cost:.4f}
          Coût (bps): {twap.cost_breakdown['cost_bps']:.2f}
          Risque (σ): {np.sqrt(twap.variance):.4f}

        ÉCONOMIE (Optimal vs TWAP):
          {(twap.expected_cost - optimal.expected_cost) / twap.expected_cost * 100:.2f}%
        """

        ax.text(0.1, 0.9, summary, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graphique sauvegardé: {save_path}")

        plt.show()


def create_model_from_calibration(
    calibration_result: Dict[str, Any],
    S0: float,
    sigma: float,
    risk_aversion: float = 1e-6,
) -> AlmgrenChrissModel:
    """Crée un modèle AC depuis des résultats de calibration.

    Args:
        calibration_result: Résultat de MarketImpactCalibrator.calibrate_all_parameters()
        S0: Prix initial
        sigma: Volatilité journalière
        risk_aversion: Aversion au risque

    Returns:
        Instance de AlmgrenChrissModel
    """
    market = MarketParameters(
        S0=S0,
        sigma=sigma,
        daily_volume=calibration_result["daily_volume"],
    )

    impact = ImpactParameters(
        psi=calibration_result["psi"],
        eta=calibration_result["eta"],
        phi=calibration_result["phi"],
        k=calibration_result["k"],
    )

    return AlmgrenChrissModel(market, impact, risk_aversion)
