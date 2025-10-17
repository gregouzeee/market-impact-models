# Données du projet

## 📁 Structure

- `raw/` : Données brutes téléchargées (non versionnées si > 100KB)
- `processed/` : Données nettoyées et transformées

## 📊 Sources de données

### 1. Yahoo Finance (yfinance)
- **Contenu** : Prix OHLCV historiques et intraday
- **Actifs** : AAPL, MSFT, GOOGL, AMZN, JPM, BAC, GS, etc.
- **Période** : 01/01/2023 - 31/12/2024
- **Fréquence** : 1 minute (intraday) + daily
- **Script** : `notebooks/01_data_collection.ipynb`

### 2. Alpha Vantage (optionnel)
- **Contenu** : Données de spread bid-ask
- **API Key** : Stockée dans `.env` (non versionnée)
- **Limite** : 5 requêtes/minute, 500/jour (tier gratuit)

## 🔄 Reproductibilité

Les données brutes ne sont pas versionnées sur GitHub (trop volumineuses).

**Option 1 : Re-téléchargement**
```bash
jupyter notebook notebooks/01_data_collection.ipynb
# Exécuter toutes les cellules
```

**Option 2 : Stockage S3 (SSPCloud)**
Les données sont également disponibles sur MinIO S3 :
```
s3://[votre-bucket]/market-impact-models/data/
```

Voir `notebooks/00_setup_s3.ipynb` pour instructions.

## 📝 Description des fichiers

| Fichier | Description | Taille | Source |
|---------|-------------|--------|--------|
| `raw/prices_intraday.parquet` | Prix 1min pour 10 actifs | ~500MB | yfinance |
| `raw/prices_daily.csv` | Prix journaliers 2023-2024 | ~1MB | yfinance |
| `processed/trades_with_slippage.parquet` | Ordres simulés + slippage observé | ~200MB | Calculé |

## ⚠️ Notes importantes

- Les données intraday de yfinance ont une rétention de ~7 jours en temps réel
- Pour des données plus anciennes, utiliser `period="max"` et `interval="1d"`
- Le spread bid-ask n'est pas toujours disponible (utiliser approximation)
