"""Collecteurs de données pour l'analyse de market impact.

Ce module contient les classes pour collecter les données d'orderbook
depuis différentes sources (Binance, etc.) pour la calibration des modèles.
"""

import requests
import numpy as np
from datetime import datetime
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.config.settings import (
    BINANCE_API_BASE,
    DEFAULT_SYMBOL,
    DEFAULT_LIMIT,
    ORDERBOOK_COLLECTION,
    ORDERBOOK_DATA_DIR,
    S3_ORDERBOOK_PREFIX,
    is_ssp_cloud,
)
from src.config.logging_config import setup_data_collector_logger
from src.storage.s3_manager import get_s3_manager

logger = setup_data_collector_logger()


class BinanceOrderBookCollector:
    """Collecteur de données d'orderbook depuis l'API publique Binance."""

    def __init__(
        self,
        symbol: str = None,
        limit: int = None,
        base_url: str = None,
    ):
        """Initialise le collecteur d'orderbook.

        Args:
            symbol: Paire de trading (ex: 'BTCUSDT')
            limit: Nombre de niveaux d'orderbook à récupérer (max 5000)
            base_url: URL de base de l'API Binance
        """
        self.symbol = symbol or DEFAULT_SYMBOL
        self.limit = limit or DEFAULT_LIMIT
        self.base_url = base_url or BINANCE_API_BASE

        logger.info(
            f"Collecteur d'orderbook initialisé: {self.symbol} (limit={self.limit})"
        )

    def get_24h_volume(self) -> Optional[Dict[str, float]]:
        """Récupère le volume de trading sur 24h depuis l'API Binance.

        Returns:
            Dictionnaire avec:
                - volume: Volume en unités de base (ex: BTC pour BTCUSDT)
                - quoteVolume: Volume en unités de cotation (ex: USDT)
                - count: Nombre de trades
            ou None en cas d'erreur

        Example:
            >>> collector = BinanceOrderBookCollector('BTCUSDT')
            >>> vol = collector.get_24h_volume()
            >>> print(f"Volume 24h: {vol['volume']:,.0f} BTC")
        """
        endpoint = f"{self.base_url}/ticker/24hr"
        params = {"symbol": self.symbol}

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            volume_info = {
                "volume": float(data["volume"]),           # Volume en base asset
                "quoteVolume": float(data["quoteVolume"]), # Volume en quote asset
                "count": int(data["count"]),               # Nombre de trades
                "lastPrice": float(data["lastPrice"]),     # Dernier prix
                "priceChange": float(data["priceChange"]), # Changement de prix
                "priceChangePercent": float(data["priceChangePercent"]),
            }

            logger.info(
                f"Volume 24h {self.symbol}: {volume_info['volume']:,.2f} "
                f"(≈ {volume_info['quoteVolume']:,.0f} USDT)"
            )

            return volume_info

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération du volume 24h: {e}")
            return None

    def get_daily_volume_estimate(self) -> float:
        """Estime le volume quotidien moyen en unités de l'actif de base.

        Cette méthode récupère le volume des dernières 24h et l'utilise
        comme estimation du volume quotidien moyen (ADV).

        Returns:
            Volume quotidien estimé en unités de base

        Note:
            Pour une estimation plus précise, il faudrait moyenner sur
            plusieurs jours. Cette méthode utilise les 24h les plus récentes.
        """
        vol_info = self.get_24h_volume()

        if vol_info is None:
            logger.warning(
                f"Impossible de récupérer le volume, utilisation d'une valeur par défaut"
            )
            # Valeurs par défaut raisonnables pour les principales paires
            defaults = {
                "BTCUSDT": 30000,   # ~30k BTC/jour sur Binance spot
                "ETHUSDT": 200000, # ~200k ETH/jour
                "SOLUSDT": 5000000, # ~5M SOL/jour
            }
            return defaults.get(self.symbol, 10000)

        return vol_info["volume"]

    def get_orderbook_snapshot(self) -> Optional[Dict[str, Any]]:
        """Récupère un snapshot de l'orderbook.

        Returns:
            Dictionnaire contenant le snapshot ou None en cas d'erreur

        Example:
            >>> collector = BinanceOrderBookCollector('BTCUSDT')
            >>> snapshot = collector.get_orderbook_snapshot()
            >>> print(snapshot['timestamp'])
        """
        endpoint = f"{self.base_url}/depth"
        params = {"symbol": self.symbol, "limit": self.limit}

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "bids": data["bids"],  # [[price, quantity], ...]
                "asks": data["asks"],
                "lastUpdateId": data["lastUpdateId"],
            }

            logger.debug(
                f"Snapshot récupéré: {len(data['bids'])} bids, {len(data['asks'])} asks"
            )
            return snapshot

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération de l'orderbook: {e}")
            return None

    def collect_snapshots(
        self,
        duration_minutes: Optional[int] = None,
        interval_seconds: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Collecte des snapshots d'orderbook sur une période.

        Args:
            duration_minutes: Durée totale de collection (en minutes)
            interval_seconds: Intervalle entre les snapshots (en secondes)

        Returns:
            Liste de snapshots d'orderbook

        Example:
            >>> collector = BinanceOrderBookCollector('BTCUSDT')
            >>> snapshots = collector.collect_snapshots(duration_minutes=60, interval_seconds=10)
            >>> print(f"Collecté {len(snapshots)} snapshots")
        """
        duration_minutes = duration_minutes or ORDERBOOK_COLLECTION["duration_minutes"]
        interval_seconds = interval_seconds or ORDERBOOK_COLLECTION["interval_seconds"]

        snapshots = []
        num_snapshots = (duration_minutes * 60) // interval_seconds

        logger.info(
            f"Début de la collection de {num_snapshots} snapshots pour {self.symbol}"
        )
        logger.info(f"Durée: {duration_minutes} min, Intervalle: {interval_seconds}s")

        start_time = time.time()

        for i in range(num_snapshots):
            snapshot = self.get_orderbook_snapshot()

            if snapshot:
                snapshots.append(snapshot)

                if (i + 1) % 10 == 0:
                    logger.info(f"Collecté {i + 1}/{num_snapshots} snapshots")
            else:
                logger.warning(f"Échec de la récupération du snapshot {i + 1}")

            time.sleep(interval_seconds)

        elapsed = time.time() - start_time
        logger.info(
            f"Collection terminée: {len(snapshots)} snapshots en {elapsed:.1f}s"
        )

        return snapshots

    def save_snapshots(
        self,
        snapshots: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        upload_to_s3: bool = True,
    ) -> Path:
        """Sauvegarde les snapshots dans un fichier JSON.

        Sur SSP Cloud: sauvegarde localement ET upload sur S3
        En local: sauvegarde uniquement localement

        Args:
            snapshots: Liste des snapshots à sauvegarder
            output_dir: Répertoire de sortie (défaut: depuis config)
            upload_to_s3: Si True, upload aussi sur S3

        Returns:
            Chemin du fichier sauvegardé
        """
        output_dir = Path(output_dir) if output_dir else ORDERBOOK_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{self.symbol}_orderbook_{timestamp}.json"

        # Sauvegarder localement
        with open(filename, "w") as f:
            json.dump(snapshots, f, indent=2)

        logger.info(f"Snapshots sauvegardés localement: {filename}")

        # Upload sur S3 si sur SSP Cloud et demandé
        if upload_to_s3 and is_ssp_cloud():
            logger.info("Exécution sur SSP Cloud - upload vers S3...")
            s3 = get_s3_manager()
            if s3:
                s3_key = f"{S3_ORDERBOOK_PREFIX}/{self.symbol}_orderbook_{timestamp}.json"
                try:
                    metadata = {
                        "symbol": self.symbol,
                        "timestamp": timestamp,
                        "num_snapshots": str(len(snapshots)),
                    }
                    s3.upload_file(filename, s3_key, metadata=metadata)
                except Exception as e:
                    logger.warning(f"Échec de l'upload S3 (continuation): {e}")

        return filename


def calculate_orderbook_metrics(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calcule les métriques clés depuis un snapshot d'orderbook.

    Args:
        snapshot: Snapshot d'orderbook

    Returns:
        Dictionnaire de métriques ou None si erreur

    Métriques calculées:
        - mid_price: Prix médian
        - spread: Spread bid-ask
        - spread_bps: Spread en points de base
        - bid/ask_liquidity_10: Liquidité sur les 10 premiers niveaux
        - bid/ask_liquidity_100: Liquidité sur les 100 premiers niveaux
        - imbalance: Déséquilibre de l'orderbook
    """
    try:
        # Convertir en arrays
        bids = np.array(snapshot["bids"], dtype=float)
        asks = np.array(snapshot["asks"], dtype=float)

        # Meilleurs bid/ask
        best_bid = bids[0, 0] if len(bids) > 0 else None
        best_ask = asks[0, 0] if len(asks) > 0 else None

        if best_bid is None or best_ask is None:
            logger.warning("Orderbook incomplet (pas de bid ou ask)")
            return None

        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000

        # Métriques de liquidité
        bid_liquidity_10 = bids[:10, 1].sum()  # Top 10 niveaux
        ask_liquidity_10 = asks[:10, 1].sum()

        bid_liquidity_100 = bids[:100, 1].sum()  # Top 100 niveaux
        ask_liquidity_100 = asks[:100, 1].sum()

        # Déséquilibre de l'orderbook
        total_bid_volume = bids[:, 1].sum()
        total_ask_volume = asks[:, 1].sum()
        imbalance = (total_bid_volume - total_ask_volume) / (
            total_bid_volume + total_ask_volume
        )

        return {
            "timestamp": snapshot["timestamp"],
            "mid_price": mid_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_bps": spread_bps,
            "bid_liquidity_10": bid_liquidity_10,
            "ask_liquidity_10": ask_liquidity_10,
            "bid_liquidity_100": bid_liquidity_100,
            "ask_liquidity_100": ask_liquidity_100,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "imbalance": imbalance,
        }

    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques: {e}")
        return None


def walk_the_book(
    orderbook_side: np.ndarray, target_quantity: float
) -> Dict[str, Any]:
    """Simule l'exécution d'un ordre de marché en parcourant l'orderbook.

    Cette fonction calcule le prix moyen d'exécution et l'impact sur le marché
    pour un ordre de taille donnée.

    Args:
        orderbook_side: Côté de l'orderbook (bids ou asks) [[price, quantity], ...]
        target_quantity: Quantité totale à exécuter

    Returns:
        Dictionnaire avec les détails de l'exécution:
            - target_quantity: Quantité ciblée
            - executed_quantity: Quantité réellement exécutée
            - avg_price: Prix moyen d'exécution
            - levels_used: Nombre de niveaux utilisés
            - sufficient_liquidity: Si la liquidité était suffisante

    Example:
        >>> asks = np.array([[100, 1], [101, 2], [102, 3]])
        >>> result = walk_the_book(asks, 2.5)
        >>> print(f"Prix moyen: {result['avg_price']}")
    """
    orders = np.array(orderbook_side, dtype=float)

    remaining = target_quantity
    total_cost = 0.0
    levels_used = 0

    for price, quantity in orders:
        if remaining <= 0:
            break

        executed = min(remaining, quantity)
        total_cost += executed * price
        remaining -= executed
        levels_used += 1

    if remaining > 0:
        # Liquidité insuffisante dans l'orderbook
        avg_price = None
        logger.warning(
            f"Liquidité insuffisante: {remaining:.4f} unités non exécutées"
        )
    else:
        avg_price = total_cost / target_quantity

    return {
        "target_quantity": target_quantity,
        "executed_quantity": target_quantity - remaining,
        "avg_price": avg_price,
        "levels_used": levels_used,
        "sufficient_liquidity": remaining <= 0,
    }


def load_orderbook_snapshots(filepath: Path) -> List[Dict[str, Any]]:
    """Charge les snapshots d'orderbook depuis un fichier JSON.

    Args:
        filepath: Chemin du fichier JSON

    Returns:
        Liste des snapshots

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        json.JSONDecodeError: Si le fichier n'est pas un JSON valide
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")

    with open(filepath, "r") as f:
        snapshots = json.load(f)

    logger.info(f"Chargé {len(snapshots)} snapshots depuis {filepath}")
    return snapshots
