"""Configuration du système de logging pour le projet.

Ce module fournit une configuration unifiée pour le logging
à travers tous les modules du projet.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config.settings import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, BASE_DIR


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure et retourne un logger.

    Args:
        name: Nom du logger (généralement __name__)
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin du fichier de log (optionnel)
        console: Si True, affiche aussi les logs dans la console

    Returns:
        Logger configuré

    Example:
        >>> from config.logging_config import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("Message de log")
    """
    logger = logging.getLogger(name)

    # Éviter les duplications si déjà configuré
    if logger.handlers:
        return logger

    # Définir le niveau
    log_level = getattr(logging, (level or LOG_LEVEL).upper())
    logger.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_default_log_file(module_name: str) -> Path:
    """Retourne le chemin du fichier de log par défaut pour un module.

    Args:
        module_name: Nom du module (ex: "calibration", "collection")

    Returns:
        Chemin du fichier de log
    """
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir / f"{module_name}.log"


# Logger par défaut pour le projet
project_logger = setup_logger("market_impact")


# Fonctions de logging rapides
def info(msg: str, logger_name: str = "market_impact") -> None:
    """Log un message INFO."""
    logging.getLogger(logger_name).info(msg)


def warning(msg: str, logger_name: str = "market_impact") -> None:
    """Log un message WARNING."""
    logging.getLogger(logger_name).warning(msg)


def error(msg: str, logger_name: str = "market_impact") -> None:
    """Log un message ERROR."""
    logging.getLogger(logger_name).error(msg)


def debug(msg: str, logger_name: str = "market_impact") -> None:
    """Log un message DEBUG."""
    logging.getLogger(logger_name).debug(msg)


# Configuration spécifique pour les différents modules
def setup_data_collector_logger() -> logging.Logger:
    """Configure le logger pour le collecteur de données."""
    return setup_logger(
        "market_impact.data.collector",
        log_file=get_default_log_file("data_collection"),
    )


def setup_calibration_logger() -> logging.Logger:
    """Configure le logger pour la calibration."""
    return setup_logger(
        "market_impact.calibration",
        log_file=get_default_log_file("calibration"),
    )


def setup_optimization_logger() -> logging.Logger:
    """Configure le logger pour l'optimisation."""
    return setup_logger(
        "market_impact.optimization",
        log_file=get_default_log_file("optimization"),
    )


def setup_s3_logger() -> logging.Logger:
    """Configure le logger pour les opérations S3."""
    return setup_logger(
        "market_impact.storage.s3",
        log_file=get_default_log_file("s3_operations"),
    )
