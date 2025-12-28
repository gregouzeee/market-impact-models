"""Module de stockage S3 pour SSP Cloud.

Ce module fournit une interface pour stocker et recuperer
des donnees sur S3/MinIO.
"""

from src.storage.s3_manager import (
    S3Manager,
    get_s3_manager,
    load_calibrated_parameters,
    save_calibrated_parameters,
)

__all__ = [
    "S3Manager",
    "get_s3_manager",
    "load_calibrated_parameters",
    "save_calibrated_parameters",
]
