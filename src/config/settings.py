"""Configuration centralisée pour le projet Market Impact Models.

Ce module contient tous les paramètres configurables du projet,
incluant les paramètres de calibration, les chemins de fichiers,
et les configurations S3.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Chemins de base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ORDERBOOK_DATA_DIR = DATA_DIR / "orderbook"
RESULTS_DIR = DATA_DIR / "results"

# Configuration S3/MinIO
S3_ENABLED = os.getenv("S3_ENABLED", "auto")  # "auto", "true", "false"
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://minio.lab.sspcloud.fr")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Préfixes S3 (les données sont dans le sous-dossier "diffusion" du bucket public)
S3_PREFIX = "diffusion/market-impact-data"
S3_ORDERBOOK_PREFIX = f"{S3_PREFIX}/orderbook"
S3_RESULTS_PREFIX = f"{S3_PREFIX}/results"
S3_PROCESSED_PREFIX = f"{S3_PREFIX}/processed"

# Configuration Binance API
BINANCE_API_BASE = "https://api.binance.com/api/v3"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LIMIT = 1000  # Nombre de lignes d'orderbook

# Paramètres de collection d'orderbook
ORDERBOOK_COLLECTION = {
    "interval_seconds": 10,  # Intervalle entre chaque snapshot
    "duration_minutes": 60,  # Durée totale de collection
    "default_symbol": DEFAULT_SYMBOL,
}

# Paramètres de calibration par défaut
CALIBRATION_PARAMS = {
    # Estimation du spread cost (ψ)
    "spread_cost": {
        "method": "median",  # "median" ou "mean"
    },

    # Estimation de Kyle's lambda (k - permanent impact)
    "kyle_lambda": {
        "method": "regression",  # "regression" ou "ratio"
        "lag": 1,  # Nombre de périodes pour le lag
    },

    # Estimation de temporary impact (η, φ)
    "temporary_impact": {
        "initial_guess": [0.01, 0.5, 0.0001],  # [η, φ, baseline]
        "bounds": {
            "eta": (1e-6, 1.0),      # η ∈ [10^-6, 1]
            "phi": (0.1, 2.0),       # φ ∈ [0.1, 2]
            "baseline": (0, 0.1),    # baseline ∈ [0, 0.1]
        },
        "power_law_exponent": 0.5,   # Exposant par défaut pour power law
    },

    # Paramètres par défaut si estimation échoue
    "fallback": {
        "spread_cost_psi": 0.0005,   # 5 bps
        "permanent_impact_k": 0.1,
        "temporary_impact_eta": 0.01,
        "temporary_impact_phi": 0.5,
    },

    # Volume quotidien par défaut (en nombre d'unités)
    "default_daily_volume": 25000,
}

# Paramètres d'optimisation Almgren-Chriss
OPTIMIZATION_PARAMS = {
    "risk_aversion": 1e-6,           # λ (risk aversion parameter)
    "time_horizon": 1.0,             # T (trading horizon en jours)
    "n_intervals": 10,               # Nombre d'intervalles de trading
    "volatility": 0.02,              # σ (volatilité quotidienne par défaut)
}

# Configuration des stratégies d'exécution
EXECUTION_STRATEGIES = {
    "twap": {
        "name": "Time-Weighted Average Price",
        "description": "Exécution uniforme dans le temps",
    },
    "vwap": {
        "name": "Volume-Weighted Average Price",
        "description": "Exécution pondérée par le volume",
    },
    "optimal": {
        "name": "Almgren-Chriss Optimal",
        "description": "Stratégie optimale minimisant coût + risque",
    },
}

# Configuration de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Validation des configurations critiques
def validate_s3_config():
    """Valide la configuration S3."""
    if S3_ENABLED in ["true", "auto"]:
        missing = []
        if not AWS_ACCESS_KEY_ID:
            missing.append("AWS_ACCESS_KEY_ID")
        if not AWS_SECRET_ACCESS_KEY:
            missing.append("AWS_SECRET_ACCESS_KEY")
        if not S3_BUCKET_NAME:
            missing.append("S3_BUCKET_NAME")

        if missing and S3_ENABLED == "true":
            raise ValueError(
                f"Configuration S3 incomplète. Variables manquantes: {', '.join(missing)}"
            )
        return len(missing) == 0
    return False

# Détection automatique SSP Cloud
def is_ssp_cloud():
    """Détecte si on est sur SSP Cloud."""
    return "AWS_S3_ENDPOINT" in os.environ or "SSP_CLOUD" in os.environ
