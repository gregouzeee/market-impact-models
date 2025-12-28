"""Gestionnaire de stockage S3 compatible avec SSP Cloud MinIO.

Ce module fournit une interface unifiée pour le stockage de données sur S3/MinIO,
avec fallback automatique vers le stockage local si S3 n'est pas disponible.

Usage:
    >>> from src.storage.s3_manager import S3Manager
    >>> s3 = S3Manager()
    >>> s3.upload_file('data/orderbook/BTCUSDT.json', 'orderbook/BTCUSDT.json')
    >>> s3.download_file('orderbook/BTCUSDT.json', 'data/orderbook/BTCUSDT.json')
"""

import boto3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from src.config.settings import (
    S3_ENDPOINT_URL,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET_NAME,
    AWS_DEFAULT_REGION,
    S3_ORDERBOOK_PREFIX,
    S3_RESULTS_PREFIX,
    S3_PROCESSED_PREFIX,
    is_ssp_cloud,
)
from src.config.logging_config import setup_s3_logger

logger = setup_s3_logger()


class S3Manager:
    """Gestionnaire pour les opérations S3 compatible avec SSP Cloud MinIO."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Initialise le client S3 avec les credentials.

        Args:
            endpoint_url: URL du endpoint S3 (défaut: depuis config)
            access_key: Clé d'accès AWS (défaut: depuis config)
            secret_key: Clé secrète AWS (défaut: depuis config)
            bucket_name: Nom du bucket S3 (défaut: depuis config)
            region: Région AWS (défaut: depuis config)

        Raises:
            ValueError: Si les credentials S3 sont manquantes
        """
        self.endpoint_url = endpoint_url or S3_ENDPOINT_URL
        self.access_key = access_key or AWS_ACCESS_KEY_ID
        self.secret_key = secret_key or AWS_SECRET_ACCESS_KEY
        self.bucket_name = bucket_name or S3_BUCKET_NAME
        self.region = region or AWS_DEFAULT_REGION

        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            missing = []
            if not self.endpoint_url:
                missing.append("S3_ENDPOINT_URL")
            if not self.access_key:
                missing.append("AWS_ACCESS_KEY_ID")
            if not self.secret_key:
                missing.append("AWS_SECRET_ACCESS_KEY")
            if not self.bucket_name:
                missing.append("S3_BUCKET_NAME")

            raise ValueError(
                f"Configuration S3 incomplète. Variables manquantes: {', '.join(missing)}\n"
                "Veuillez définir ces variables dans votre fichier .env\n"
                "Voir .env.example pour plus de détails."
            )

        # Initialiser le client S3
        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )

        logger.info(f"Client S3 initialisé (bucket: {self.bucket_name})")

    def upload_file(
        self,
        local_path: Union[str, Path],
        s3_key: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload un fichier vers S3.

        Args:
            local_path: Chemin du fichier local
            s3_key: Clé S3 (chemin dans le bucket). Si None, utilise le nom du fichier
            metadata: Métadonnées à attacher au fichier

        Returns:
            Clé S3 du fichier uploadé

        Raises:
            FileNotFoundError: Si le fichier local n'existe pas
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Fichier local non trouvé: {local_path}")

        if s3_key is None:
            s3_key = local_path.name

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            self.s3.upload_file(str(local_path), self.bucket_name, s3_key, ExtraArgs=extra_args)
            logger.info(f"Uploadé {local_path} → s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except Exception as e:
            logger.error(f"Échec de l'upload: {e}")
            raise

    def download_file(
        self, s3_key: str, local_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Download un fichier depuis S3.

        Args:
            s3_key: Clé S3 (chemin dans le bucket)
            local_path: Chemin local où sauvegarder. Si None, utilise s3_key comme nom

        Returns:
            Chemin du fichier téléchargé

        Raises:
            Exception: Si le téléchargement échoue
        """
        if local_path is None:
            local_path = Path(s3_key)
        else:
            local_path = Path(local_path)

        # Créer le répertoire parent si nécessaire
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.s3.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Téléchargé s3://{self.bucket_name}/{s3_key} → {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Échec du téléchargement: {e}")
            raise

    def file_exists(self, s3_key: str) -> bool:
        """Vérifie si un fichier existe dans S3.

        Args:
            s3_key: Clé S3 à vérifier

        Returns:
            True si le fichier existe, False sinon
        """
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception:
            return False

    def list_files(self, prefix: str = "", max_keys: int = 1000) -> List[str]:
        """Liste les fichiers dans le bucket S3 avec un préfixe optionnel.

        Args:
            prefix: Filtre les fichiers par préfixe (chemin de dossier)
            max_keys: Nombre maximum de clés à retourner

        Returns:
            Liste des clés S3 correspondant au préfixe
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix, MaxKeys=max_keys
            )

            if "Contents" not in response:
                return []

            files = [obj["Key"] for obj in response["Contents"]]
            logger.debug(f"Trouvé {len(files)} fichiers avec le préfixe '{prefix}'")
            return files

        except Exception as e:
            logger.error(f"Échec du listage: {e}")
            raise

    def upload_json(
        self, data: Union[Dict, List], s3_key: str, metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload des données JSON directement vers S3.

        Args:
            data: Données à sérialiser en JSON
            s3_key: Clé S3
            metadata: Métadonnées à attacher

        Returns:
            Clé S3 du fichier uploadé
        """
        try:
            json_str = json.dumps(data, indent=2, default=str)

            extra_args = {"ContentType": "application/json"}
            if metadata:
                extra_args["Metadata"] = metadata

            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str.encode("utf-8"),
                **extra_args,
            )
            logger.info(f"Uploadé JSON → s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except Exception as e:
            logger.error(f"Échec de l'upload JSON: {e}")
            raise

    def download_json(self, s3_key: str) -> Union[Dict, List]:
        """Download et parse des données JSON depuis S3.

        Args:
            s3_key: Clé S3

        Returns:
            Données JSON parsées
        """
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            logger.info(f"Téléchargé JSON depuis s3://{self.bucket_name}/{s3_key}")
            return data

        except Exception as e:
            logger.error(f"Échec du téléchargement JSON: {e}")
            raise

    def get_file_metadata(self, s3_key: str) -> Dict[str, Any]:
        """Récupère les métadonnées d'un fichier S3.

        Args:
            s3_key: Clé S3

        Returns:
            Dictionnaire contenant les métadonnées
        """
        try:
            response = self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                "content_length": response.get("ContentLength"),
                "last_modified": response.get("LastModified"),
                "content_type": response.get("ContentType"),
                "metadata": response.get("Metadata", {}),
            }
        except Exception as e:
            logger.error(f"Échec de la récupération des métadonnées: {e}")
            raise

    def list_calibrations(
        self, symbol: Optional[str] = None, date_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Liste toutes les calibrations disponibles pour un symbole.

        Args:
            symbol: Symbole à filtrer (ex: "BTCUSDT")
            date_range: Tuple (date_début, date_fin) pour filtrer

        Returns:
            Liste de dictionnaires contenant les informations de calibration
        """
        prefix = f"{S3_RESULTS_PREFIX}/calibrations/"
        if symbol:
            prefix += f"{symbol}_"

        files = self.list_files(prefix)

        calibrations = []
        for file_key in files:
            if file_key.endswith(".json"):
                try:
                    metadata = self.get_file_metadata(file_key)
                    calibrations.append(
                        {
                            "s3_key": file_key,
                            "last_modified": metadata["last_modified"],
                            "size": metadata["content_length"],
                        }
                    )
                except Exception as e:
                    logger.warning(f"Impossible de récupérer les métadonnées de {file_key}: {e}")

        return sorted(calibrations, key=lambda x: x["last_modified"], reverse=True)

    def get_latest_calibration(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Récupère la calibration la plus récente pour un symbole.

        Args:
            symbol: Symbole (ex: "BTCUSDT")

        Returns:
            Données de calibration ou None si non trouvé
        """
        calibrations = self.list_calibrations(symbol)
        if not calibrations:
            logger.warning(f"Aucune calibration trouvée pour {symbol}")
            return None

        latest = calibrations[0]
        logger.info(f"Calibration la plus récente pour {symbol}: {latest['s3_key']}")
        return self.download_json(latest["s3_key"])


def get_s3_manager() -> Optional[S3Manager]:
    """Obtient une instance de S3Manager avec gestion d'erreur.

    Returns:
        S3Manager si les credentials sont disponibles, None sinon
    """
    try:
        return S3Manager()
    except ValueError as e:
        logger.warning(f"S3 non configuré: {e}")
        return None


def load_calibrated_parameters(
    local_path: Union[str, Path] = "data/results/calibrated_parameters.json",
    s3_key: str = f"{S3_RESULTS_PREFIX}/calibrated_parameters.json",
) -> Optional[Dict[str, Any]]:
    """Charge les paramètres calibrés avec fallback automatique S3/local.

    Sur SSP Cloud: essaie S3 d'abord, puis fallback vers local
    En local: utilise seulement le fichier local

    Args:
        local_path: Chemin du fichier local
        s3_key: Clé S3 du fichier

    Returns:
        Paramètres calibrés si trouvés, None sinon
    """
    local_path = Path(local_path)

    # Essayer S3 d'abord si sur SSP Cloud
    if is_ssp_cloud():
        logger.info("Exécution sur SSP Cloud - vérification S3 pour les paramètres calibrés...")
        s3 = get_s3_manager()
        if s3 and s3.file_exists(s3_key):
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(s3_key, local_path)
                with open(local_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Échec du chargement depuis S3: {e}")

    # Fallback vers fichier local
    if local_path.exists():
        logger.info(f"Chargement des paramètres calibrés depuis le fichier local: {local_path}")
        with open(local_path, "r") as f:
            return json.load(f)

    logger.warning(f"Aucun paramètre calibré trouvé (vérifié: {local_path})")
    return None


def save_calibrated_parameters(
    parameters: Dict[str, Any],
    local_path: Union[str, Path] = "data/results/calibrated_parameters.json",
    s3_key: Optional[str] = None,
    upload_to_s3: bool = True,
) -> Path:
    """Sauvegarde les paramètres calibrés localement et optionnellement sur S3.

    Args:
        parameters: Dictionnaire des paramètres calibrés
        local_path: Chemin du fichier local
        s3_key: Clé S3 (si None, utilise la config par défaut)
        upload_to_s3: Si True, upload aussi sur S3

    Returns:
        Chemin du fichier local
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder localement
    with open(local_path, "w") as f:
        json.dump(parameters, f, indent=2, default=str)
    logger.info(f"Paramètres calibrés sauvegardés: {local_path}")

    # Upload sur S3 si demandé et disponible
    if upload_to_s3:
        s3 = get_s3_manager()
        if s3:
            if s3_key is None:
                s3_key = f"{S3_RESULTS_PREFIX}/calibrated_parameters.json"

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "symbol": parameters.get("symbol", "unknown"),
            }
            s3.upload_file(local_path, s3_key, metadata=metadata)

    return local_path
