"""Calibrateur de paramètres pour le modèle de market impact.

Ce module contient la classe principale pour calibrer les paramètres du modèle
Almgren-Chriss à partir des données d'orderbook.

Références:
    - Almgren et al. (2005): Direct Estimation of Equity Market Impact
    - Kyle (1985): Continuous Auctions and Insider Trading
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress

from src.config.settings import (
    CALIBRATION_PARAMS,
    RESULTS_DIR,
)
from src.config.logging_config import setup_calibration_logger
from src.storage.s3_manager import save_calibrated_parameters

logger = setup_calibration_logger()


class MarketImpactCalibrator:
    """Calibrateur de paramètres de market impact depuis des données d'orderbook."""

    def __init__(self, orderbook_file: Path, daily_volume: float):
        """Initialise le calibrateur.

        Args:
            orderbook_file: Chemin du fichier JSON avec les snapshots d'orderbook
            daily_volume: Volume de trading quotidien moyen (pour calcul du participation rate)

        Raises:
            FileNotFoundError: Si le fichier orderbook n'existe pas
            ValueError: Si daily_volume <= 0
        """
        self.orderbook_file = Path(orderbook_file)

        if not self.orderbook_file.exists():
            raise FileNotFoundError(f"Fichier orderbook non trouve: {orderbook_file}")

        if daily_volume <= 0:
            raise ValueError(f"daily_volume doit etre > 0, recu: {daily_volume}")

        self.daily_volume = daily_volume
        self.snapshots = self._load_snapshots()

        logger.info(
            f"Calibrateur initialisé: {len(self.snapshots)} snapshots, "
            f"volume quotidien={daily_volume:,.0f}"
        )

    def _load_snapshots(self) -> List[Dict[str, Any]]:
        """Charge les snapshots d'orderbook depuis JSON.

        Returns:
            Liste des snapshots
        """
        with open(self.orderbook_file, "r") as f:
            snapshots = json.load(f)

        logger.info(f"Charge {len(snapshots)} snapshots d'orderbook")
        return snapshots

    def estimate_spread_cost(self, reference_size: float = 5.0) -> float:
        """Estime les coûts proportionnels (ψ) depuis le spread bid-ask EFFECTIF.

        ψ représente le spread + frais de trading.

        Note: On utilise le spread EFFECTIF (coût aller-retour d'un ordre de référence)
        plutôt que le spread quoté (best bid - best ask) car:
        - Le spread quoté peut être au minimum du tick (0.01 USD pour BTCUSDT)
        - Le spread effectif capture les coûts réels d'exécution en parcourant le book

        Args:
            reference_size: Taille de l'ordre de référence pour le calcul du spread

        Returns:
            ψ (spread effectif proportionnel)
        """
        quoted_spreads = []
        effective_spreads = []

        for snapshot in self.snapshots:
            bids = np.array(snapshot["bids"], dtype=float)
            asks = np.array(snapshot["asks"], dtype=float)

            if len(bids) == 0 or len(asks) == 0:
                continue

            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            mid_price = (best_bid + best_ask) / 2

            # Spread quoté (peut être trop petit)
            quoted_spread_prop = (best_ask - best_bid) / mid_price
            quoted_spreads.append(quoted_spread_prop)

            # Spread effectif: simuler un ordre aller-retour
            # 1. Simuler ordre BUY (monter les asks)
            remaining = reference_size
            total_cost = 0.0
            for price, quantity in asks:
                if remaining <= 0:
                    break
                executed = min(remaining, quantity)
                total_cost += executed * price
                remaining -= executed

            if remaining > 0:
                # Liquidité insuffisante pour l'achat
                continue

            avg_buy_price = total_cost / reference_size

            # 2. Simuler ordre SELL (descendre les bids)
            remaining = reference_size
            total_revenue = 0.0
            for price, quantity in bids:
                if remaining <= 0:
                    break
                executed = min(remaining, quantity)
                total_revenue += executed * price
                remaining -= executed

            if remaining > 0:
                # Liquidité insuffisante pour la vente
                continue

            avg_sell_price = total_revenue / reference_size

            # Spread effectif = moitié du coût aller-retour
            effective_spread_prop = (avg_buy_price - avg_sell_price) / 2 / mid_price
            effective_spreads.append(effective_spread_prop)

        psi_quoted = np.median(quoted_spreads) if quoted_spreads else 0
        psi_effective = np.median(effective_spreads) if effective_spreads else 0

        logger.info(
            f"Coûts proportionnels (ψ): "
            f"quoté={psi_quoted*10000:.2f} bps, "
            f"effectif={psi_effective*10000:.2f} bps (utilisé)"
        )

        return psi_effective

    def estimate_kyle_lambda(
        self, max_trade_size: float = 100
    ) -> Tuple[float, pd.DataFrame]:
        """Estime le lambda de Kyle (paramètre d'impact permanent k).

        Le lambda de Kyle mesure l'impact sur le prix par unité de volume:
        ΔP = λ × Q

        Args:
            max_trade_size: Taille maximale d'ordre à tester

        Returns:
            Tuple (lambda_kyle, données_groupées)
        """
        trade_sizes = np.linspace(1, max_trade_size, 20)
        price_impacts = []

        logger.info(f"Estimation du lambda de Kyle (impact permanent)...")

        # Utiliser les 100 premiers snapshots pour l'estimation
        for snapshot in self.snapshots[: min(100, len(self.snapshots))]:
            bids = np.array(snapshot["bids"], dtype=float)
            asks = np.array(snapshot["asks"], dtype=float)

            if len(bids) == 0 or len(asks) == 0:
                continue

            mid_price = (bids[0, 0] + asks[0, 0]) / 2

            # Simuler des ordres d'achat (monter le côté ask)
            for size in trade_sizes:
                remaining = size
                total_cost = 0.0

                for price, quantity in asks:
                    if remaining <= 0:
                        break
                    executed = min(remaining, quantity)
                    total_cost += executed * price
                    remaining -= executed

                if remaining <= 0:  # Liquidité suffisante
                    avg_price = total_cost / size
                    impact = (avg_price - mid_price) / mid_price
                    price_impacts.append({"size": size, "impact": impact})

        # Régression: impact = λ × size
        df = pd.DataFrame(price_impacts)
        df_grouped = df.groupby("size")["impact"].mean().reset_index()

        slope, intercept, r_value, p_value, std_err = linregress(
            df_grouped["size"], df_grouped["impact"]
        )

        lambda_kyle = slope

        logger.info(
            f"Lambda de Kyle (λ): {lambda_kyle:.8f}, "
            f"R²={r_value**2:.4f}, "
            f"Impact pour 100 unités≈{lambda_kyle * 100 * 10000:.2f} bps"
        )

        return lambda_kyle, df_grouped

    def estimate_temporary_impact(
        self, test_sizes: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Estime la fonction d'impact temporaire: L(ρ) = η|ρ|^(1+φ) + ψ|ρ|

        Args:
            test_sizes: Tailles d'ordres à tester

        Returns:
            Dictionnaire avec les paramètres calibrés (η, φ, ψ) ou None si échec
        """
        if test_sizes is None:
            test_sizes = [1, 2, 5, 10, 20, 50, 100]

        logger.info(f"Estimation de la fonction d'impact temporaire...")
        logger.info(f"Test de {len(test_sizes)} tailles d'ordres différentes")

        impact_data = []

        for snapshot in self.snapshots:
            bids = np.array(snapshot["bids"], dtype=float)
            asks = np.array(snapshot["asks"], dtype=float)

            if len(bids) == 0 or len(asks) == 0:
                continue

            mid_price = (bids[0, 0] + asks[0, 0]) / 2

            for size in test_sizes:
                # Participation rate
                rho = size / self.daily_volume

                # Simuler ordre d'achat au marché
                remaining = size
                total_cost = 0.0

                for price, quantity in asks:
                    if remaining <= 0:
                        break
                    executed = min(remaining, quantity)
                    total_cost += executed * price
                    remaining -= executed

                if remaining <= 0:  # Liquidité suffisante
                    avg_price = total_cost / size
                    # Impact temporaire (slippage total)
                    impact = (avg_price - mid_price) / mid_price

                    impact_data.append({"size": size, "rho": rho, "impact": impact})

        df = pd.DataFrame(impact_data)
        df_grouped = df.groupby(["size", "rho"])["impact"].mean().reset_index()

        logger.info(
            f"Collecté {len(df)} mesures d'impact, "
            f"moyenne de {len(df) / len(test_sizes):.0f} échantillons par taille"
        )

        # Régression non-linéaire: impact = η × ρ^(1+φ) + ψ × ρ
        def power_law_impact(rho, eta, phi, psi):
            """Fonction d'impact power-law."""
            return eta * np.abs(rho) ** (1 + phi) + psi * np.abs(rho)

        # Estimation initiale depuis config
        initial_guess = CALIBRATION_PARAMS["temporary_impact"]["initial_guess"]
        bounds_config = CALIBRATION_PARAMS["temporary_impact"]["bounds"]
        bounds = (
            [bounds_config["eta"][0], bounds_config["phi"][0], bounds_config["baseline"][0]],
            [bounds_config["eta"][1], bounds_config["phi"][1], bounds_config["baseline"][1]],
        )

        try:
            # Ajuster le modèle
            params, covariance = curve_fit(
                power_law_impact,
                df_grouped["rho"],
                df_grouped["impact"],
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )

            eta, phi, psi_fitted = params
            perr = np.sqrt(np.diag(covariance))

            # Qualité de l'ajustement
            y_pred = power_law_impact(df_grouped["rho"], *params)
            ss_res = np.sum((df_grouped["impact"] - y_pred) ** 2)
            ss_tot = np.sum((df_grouped["impact"] - df_grouped["impact"].mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            logger.info(
                f"Calibration réussie! η={eta:.6f}±{perr[0]:.6f}, "
                f"φ={phi:.4f}±{perr[1]:.4f}, "
                f"ψ={psi_fitted:.6f}±{perr[2]:.6f} ({psi_fitted*10000:.2f} bps), "
                f"R²={r_squared:.4f}"
            )

            return {
                "eta": eta,
                "phi": phi,
                "psi": psi_fitted,
                "eta_std": perr[0],
                "phi_std": perr[1],
                "psi_std": perr[2],
                "r_squared": r_squared,
                "data": df_grouped,
            }

        except RuntimeError as e:
            logger.error(f"Échec de la calibration: {e}")
            return None

    def estimate_joint_parameters(
        self,
        test_sizes: Optional[List[float]] = None,
        fix_phi: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Estimation JOINTE de tous les paramètres (ψ, η, φ, k) par optimisation non-linéaire.

        Cette méthode corrige le biais de l'estimation séquentielle en estimant
        tous les paramètres simultanément. Le modèle complet est:

            C_total(ρ) = ψ + η·ρ^φ + k·ρ

        où:
            - ψ: coûts proportionnels (spread + fees)
            - η: coefficient d'impact temporaire
            - φ: exposant power-law (typiquement 0.4-0.7, théoriquement ~0.5)
            - k: coefficient d'impact permanent

        Args:
            test_sizes: Tailles d'ordres à tester (défaut: [1, 2, 5, 10, 20, 50, 100])
            fix_phi: Si spécifié, fixe φ à cette valeur pour réduire la collinéarité
                     (recommandé: 0.5 pour la loi racine carrée)

        Returns:
            Dictionnaire avec les paramètres calibrés ou None si échec

        Note:
            Cette méthode est préférée à estimate_temporary_impact() car elle évite
            le biais introduit par l'estimation séquentielle qui ignore le terme k·ρ.
        """
        if test_sizes is None:
            test_sizes = [1, 2, 5, 10, 20, 50, 100]

        logger.info("=" * 70)
        logger.info("ESTIMATION JOINTE des paramètres (ψ, η, φ, k)")
        if fix_phi is not None:
            logger.info(f"φ fixé à {fix_phi} (loi racine carrée)")
        logger.info("=" * 70)

        # Collecter les données d'impact
        impact_data = []

        for snapshot in self.snapshots:
            bids = np.array(snapshot["bids"], dtype=float)
            asks = np.array(snapshot["asks"], dtype=float)

            if len(bids) == 0 or len(asks) == 0:
                continue

            mid_price = (bids[0, 0] + asks[0, 0]) / 2

            for size in test_sizes:
                rho = size / self.daily_volume

                # Simuler ordre d'achat au marché
                remaining = size
                total_cost = 0.0

                for price, quantity in asks:
                    if remaining <= 0:
                        break
                    executed = min(remaining, quantity)
                    total_cost += executed * price
                    remaining -= executed

                if remaining <= 0:
                    avg_price = total_cost / size
                    impact = (avg_price - mid_price) / mid_price
                    impact_data.append({"size": size, "rho": rho, "impact": impact})

        df = pd.DataFrame(impact_data)
        df_grouped = df.groupby(["size", "rho"])["impact"].mean().reset_index()

        rho_data = df_grouped["rho"].values
        impact_observed = df_grouped["impact"].values

        logger.info(f"Données collectées: {len(df)} mesures, {len(df_grouped)} points agrégés")

        # Définir le modèle complet
        def full_impact_model(rho, psi, eta, phi, k):
            """Modèle d'impact complet: C = ψ + η·ρ^φ + k·ρ"""
            return psi + eta * np.power(rho, phi) + k * rho

        def full_impact_model_fixed_phi(rho, psi, eta, k):
            """Modèle avec φ fixé"""
            return psi + eta * np.power(rho, fix_phi) + k * rho

        # Estimation initiale
        psi_init = self.estimate_spread_cost(reference_size=5.0)

        if fix_phi is not None:
            # Optimisation avec φ fixé (3 paramètres: ψ, η, k)
            try:
                popt, pcov = curve_fit(
                    full_impact_model_fixed_phi,
                    rho_data,
                    impact_observed,
                    p0=[psi_init, 0.01, 0.001],
                    bounds=(
                        [0, 1e-6, 0],           # min: psi>=0, eta>0, k>=0
                        [0.01, 1.0, 0.1]        # max
                    ),
                    maxfev=10000,
                )
                psi_est, eta_est, k_est = popt
                phi_est = fix_phi
                perr = np.sqrt(np.diag(pcov))
                psi_std, eta_std, k_std = perr
                phi_std = 0.0  # fixé

                y_pred = full_impact_model_fixed_phi(rho_data, *popt)

            except RuntimeError as e:
                logger.error(f"Échec curve_fit avec φ fixé: {e}")
                return None

        else:
            # Optimisation complète (4 paramètres: ψ, η, φ, k)
            try:
                popt, pcov = curve_fit(
                    full_impact_model,
                    rho_data,
                    impact_observed,
                    p0=[psi_init, 0.01, 0.5, 0.001],
                    bounds=(
                        [0, 1e-6, 0.1, 0],      # min
                        [0.01, 1.0, 1.0, 0.1]   # max
                    ),
                    maxfev=10000,
                )
                psi_est, eta_est, phi_est, k_est = popt
                perr = np.sqrt(np.diag(pcov))
                psi_std, eta_std, phi_std, k_std = perr

                y_pred = full_impact_model(rho_data, *popt)

            except RuntimeError as e:
                logger.error(f"Échec curve_fit complet: {e}")
                return None

        # Calcul du R²
        ss_res = np.sum((impact_observed - y_pred) ** 2)
        ss_tot = np.sum((impact_observed - np.mean(impact_observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calcul du RMSE
        rmse = np.sqrt(np.mean((impact_observed - y_pred) ** 2))

        # Vérifier la qualité de l'estimation
        relative_errors = {
            "psi": psi_std / psi_est if psi_est > 0 else float('inf'),
            "eta": eta_std / eta_est if eta_est > 0 else float('inf'),
            "phi": phi_std / phi_est if phi_est > 0 else float('inf'),
            "k": k_std / k_est if k_est > 0 else float('inf'),
        }

        logger.info("\nRésultats de l'estimation jointe:")
        logger.info(f"  ψ (spread)     = {psi_est:.6f} ± {psi_std:.6f} ({psi_est*10000:.2f} bps)")
        logger.info(f"  η (temp coef)  = {eta_est:.6f} ± {eta_std:.6f}")
        logger.info(f"  φ (exponent)   = {phi_est:.4f} ± {phi_std:.4f}")
        logger.info(f"  k (perm coef)  = {k_est:.6f} ± {k_std:.6f}")
        logger.info(f"\nQualité de l'ajustement:")
        logger.info(f"  R²   = {r_squared:.4f}")
        logger.info(f"  RMSE = {rmse:.6f} ({rmse*10000:.2f} bps)")

        # Avertissement si les erreurs relatives sont trop grandes
        if any(e > 1.0 for e in relative_errors.values() if e != float('inf')):
            logger.warning("Certains parametres ont une incertitude > 100%")
            logger.warning("   Considérez fixer φ=0.5 pour réduire la collinéarité")

        # Comparaison avec la littérature
        if 0.3 <= phi_est <= 0.7:
            logger.info(f"phi={phi_est:.2f} est dans le range attendu [0.3, 0.7]")
        else:
            logger.warning(f"phi={phi_est:.2f} est hors du range typique [0.3, 0.7]")

        return {
            "psi": float(psi_est),
            "eta": float(eta_est),
            "phi": float(phi_est),
            "k": float(k_est),
            "psi_std": float(psi_std),
            "eta_std": float(eta_std),
            "phi_std": float(phi_std),
            "k_std": float(k_std),
            "r_squared": float(r_squared),
            "rmse": float(rmse),
            "relative_errors": relative_errors,
            "data": df_grouped,
            "method": "joint_estimation",
            "phi_fixed": fix_phi is not None,
        }

    def plot_joint_calibration(
        self,
        calibration_result: Dict[str, Any],
        output_file: Optional[Path] = None,
    ) -> None:
        """Trace les résultats de la calibration jointe avec diagnostic.

        Args:
            calibration_result: Résultats de estimate_joint_parameters()
            output_file: Chemin du fichier de sortie
        """
        if calibration_result is None:
            logger.warning("Pas de résultat de calibration à tracer")
            return

        df = calibration_result["data"]
        psi = calibration_result["psi"]
        eta = calibration_result["eta"]
        phi = calibration_result["phi"]
        k = calibration_result["k"]

        rho_data = df["rho"].values
        impact_observed = df["impact"].values

        # Générer courbe lisse
        rho_smooth = np.linspace(rho_data.min(), rho_data.max(), 100)
        impact_smooth = psi + eta * np.power(rho_smooth, phi) + k * rho_smooth

        # Décomposition des coûts
        cost_spread = psi * np.ones_like(rho_smooth)
        cost_temp = eta * np.power(rho_smooth, phi)
        cost_perm = k * rho_smooth

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Impact total vs participation rate
        ax = axes[0, 0]
        ax.scatter(rho_data * 100, impact_observed * 10000, alpha=0.6,
                   label="Observé", s=50)
        ax.plot(rho_smooth * 100, impact_smooth * 10000, "r-", linewidth=2,
                label=f"Modèle: ψ+η·ρ^{phi:.2f}+k·ρ")
        ax.set_xlabel("Participation rate ρ (%)")
        ax.set_ylabel("Impact total (bps)")
        ax.set_title("Impact de Marché vs Taux de Participation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Décomposition des coûts
        ax = axes[0, 1]
        ax.fill_between(rho_smooth * 100, 0, cost_spread * 10000,
                        alpha=0.7, label=f"Spread ψ={psi*10000:.1f}bps")
        ax.fill_between(rho_smooth * 100, cost_spread * 10000,
                        (cost_spread + cost_temp) * 10000,
                        alpha=0.7, label=f"Temporaire η·ρ^{phi:.2f}")
        ax.fill_between(rho_smooth * 100, (cost_spread + cost_temp) * 10000,
                        impact_smooth * 10000,
                        alpha=0.7, label=f"Permanent k·ρ")
        ax.scatter(rho_data * 100, impact_observed * 10000,
                   color='black', alpha=0.5, s=20, label="Observé")
        ax.set_xlabel("Participation rate ρ (%)")
        ax.set_ylabel("Coût (bps)")
        ax.set_title("Décomposition des Coûts d'Exécution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Graphique log-log
        ax = axes[1, 0]
        # Soustraire ψ pour voir la power-law
        impact_adj = impact_observed - psi
        mask = impact_adj > 0
        ax.scatter(np.log(rho_data[mask]), np.log(impact_adj[mask]),
                   alpha=0.6, s=50, label="Observé (C-ψ)")

        impact_smooth_adj = eta * np.power(rho_smooth, phi) + k * rho_smooth
        ax.plot(np.log(rho_smooth), np.log(impact_smooth_adj), "r-",
                linewidth=2, label=f"Modèle η·ρ^{phi:.2f}+k·ρ")
        ax.set_xlabel("log(ρ)")
        ax.set_ylabel("log(C - ψ)")
        ax.set_title("Graphique Log-Log (vérification power-law)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Résidus
        ax = axes[1, 1]
        y_pred = psi + eta * np.power(rho_data, phi) + k * rho_data
        residuals = (impact_observed - y_pred) * 10000
        ax.scatter(rho_data * 100, residuals, alpha=0.6, s=50)
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Participation rate ρ (%)")
        ax.set_ylabel("Résidus (bps)")
        ax.set_title(f"Analyse des Résidus (R²={calibration_result['r_squared']:.4f})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder
        if output_file is None:
            output_file = RESULTS_DIR / "joint_calibration.png"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150)
        logger.info(f"Graphique de calibration jointe sauvegardé: {output_file}")
        plt.close()

    def plot_impact_curve(
        self, calibration_result: Dict[str, Any], output_file: Optional[Path] = None
    ) -> None:
        """Trace la courbe d'impact calibrée.

        Args:
            calibration_result: Résultats de la calibration
            output_file: Chemin du fichier de sortie (défaut: RESULTS_DIR/impact_calibration.png)
        """
        if calibration_result is None:
            logger.warning("Pas de résultat de calibration à tracer")
            return

        df = calibration_result["data"]
        eta = calibration_result["eta"]
        phi = calibration_result["phi"]
        psi = calibration_result["psi"]

        # Générer une courbe lisse
        rho_smooth = np.linspace(df["rho"].min(), df["rho"].max(), 100)
        impact_smooth = eta * rho_smooth ** (1 + phi) + psi * rho_smooth

        # Créer le graphique
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Impact vs participation rate
        ax = axes[0]
        ax.scatter(df["rho"] * 100, df["impact"] * 10000, alpha=0.6, label="Observé")
        ax.plot(
            rho_smooth * 100,
            impact_smooth * 10000,
            "r-",
            linewidth=2,
            label=f"Ajusté: η={eta:.4f}, φ={phi:.2f}",
        )
        ax.set_xlabel("Participation rate (%)")
        ax.set_ylabel("Market impact (bps)")
        ax.set_title("Market Impact vs Participation Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Graphique log-log (pour voir la power-law)
        ax = axes[1]
        ax.scatter(
            np.log(df["rho"]), np.log(df["impact"]), alpha=0.6, label="Observé"
        )
        ax.plot(
            np.log(rho_smooth),
            np.log(impact_smooth),
            "r-",
            linewidth=2,
            label=f"Power-law: φ={phi:.2f}",
        )
        ax.set_xlabel("log(ρ)")
        ax.set_ylabel("log(impact)")
        ax.set_title("Graphique Log-Log (pente ≈ 1+φ)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder
        if output_file is None:
            output_file = RESULTS_DIR / "impact_calibration.png"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150)
        logger.info(f"Graphique sauvegardé: {output_file}")
        plt.close()

    def calibrate_all_parameters(self) -> Dict[str, Any]:
        """Calibre tous les paramètres du modèle.

        Returns:
            Dictionnaire avec tous les paramètres calibrés

        Paramètres calibrés:
            - psi: Coûts proportionnels
            - k: Impact permanent (Kyle's lambda)
            - eta: Coefficient d'exécution
            - phi: Exposant power-law
            - r_squared: Qualité de l'ajustement
        """
        logger.info("=" * 70)
        logger.info("Début de la calibration des paramètres")
        logger.info("=" * 70)

        # 1. Estimer le spread (ψ)
        psi = self.estimate_spread_cost()

        # 2. Estimer le lambda de Kyle (k)
        lambda_kyle, kyle_data = self.estimate_kyle_lambda(max_trade_size=100)

        # 3. Estimer l'impact temporaire (η, φ)
        calibration = self.estimate_temporary_impact()

        if calibration:
            # Tracer les résultats
            self.plot_impact_curve(calibration)

            # Créer le dictionnaire de résultats
            results = {
                "symbol": self.snapshots[0].get("symbol", "UNKNOWN"),
                "psi": float(psi),
                "k": float(lambda_kyle),
                "eta": float(calibration["eta"]),
                "phi": float(calibration["phi"]),
                "r_squared": float(calibration["r_squared"]),
                "daily_volume": float(self.daily_volume),
                "timestamp": pd.Timestamp.now().isoformat(),
                "num_snapshots": len(self.snapshots),
            }

            # Sauvegarder les paramètres
            save_calibrated_parameters(results)

            logger.info("=" * 70)
            logger.info("RÉSUMÉ - Paramètres Calibrés:")
            logger.info("=" * 70)
            logger.info(f"  k  (impact permanent)   = {lambda_kyle:.8f}")
            logger.info(f"  η  (coût d'exécution)   = {calibration['eta']:.6f}")
            logger.info(f"  φ  (exposant power-law) = {calibration['phi']:.4f}")
            logger.info(f"  ψ  (coûts proportionnels) = {psi:.6f} ({psi*10000:.2f} bps)")
            logger.info(f"  R² (qualité ajustement) = {calibration['r_squared']:.4f}")
            logger.info("=" * 70)

            return results
        else:
            logger.error("Échec de la calibration de l'impact temporaire")
            return None
