# Market Impact Models - Almgren-Chriss

**Auteurs** : Grégoire Marguier & Pierre Robin-Schnepf

**ENSAE Paris** - Python pour la Data Science (2025-2026)

---

## Problématique

> **Comment modéliser et optimiser les coûts de transaction liés à l'impact de marché lors de l'exécution d'ordres importants ?**

Lorsqu'un investisseur institutionnel exécute un ordre de grande taille, il fait face à un dilemme : exécuter rapidement entraîne un impact de marché important, tandis qu'exécuter lentement expose à un risque de volatilité.

Ce projet implémente le **modèle d'Almgren-Chriss (2001)** pour résoudre ce problème d'exécution optimale, avec :
- Collecte de données multi-sources (Databento, Binance API)
- Calibration empirique des paramètres du modèle
- Comparaison de stratégies d'exécution (TWAP vs Optimal)

---

## Installation rapide (SSPCloud / Onyxia)

```bash
# 1. Cloner le projet
git clone https://github.com/gregouzeee/market-impact-models.git
cd market-impact-models

# 2. Installer les dépendances
pip install -r requirements.txt
```

Les notebooks sont prêts à être exécutés.

---

## Structure du projet

```
market-impact-models/
├── notebooks/
│   ├── 00_rapport_final.ipynb        # Rapport de synthèse (FR)
│   ├── 00_final_report.ipynb         # Final report (EN)
│   ├── 01_data_collection.ipynb      # Collecte données historiques
│   ├── 02_orderbook_collection.ipynb # Collecte orderbook temps réel
│   ├── 03_calibration.ipynb          # Calibration des paramètres
│   ├── 04_almgren_chriss_quadratic.ipynb  # Modèle AC quadratique
│   └── 05_almgren_chriss_powerlaw.ipynb   # Modèle AC power-law
├── src/                              # Modules Python
├── data/                             # Données (générées)
├── results/                          # Résultats et figures
└── requirements.txt                  # Dépendances
```

---

## Notebooks

### Rapport final
| Notebook | Description |
|----------|-------------|
| `00_rapport_final.ipynb` | **Notebook de synthèse** - À consulter en priorité |
| `00_final_report.ipynb` | Version anglaise |

### Pipeline complet (optionnel)
| # | Notebook | Description | Durée |
|---|----------|-------------|-------|
| 1 | `01_data_collection.ipynb` | Données OHLCV (stocks + crypto) | ~5 min |
| 2 | `02_orderbook_collection.ipynb` | Snapshots orderbook Binance | ~60 min* |
| 3 | `03_calibration.ipynb` | Calibration η, k, ψ | ~2 min |
| 4 | `04_almgren_chriss_quadratic.ipynb` | Modèle AC (solution analytique) | ~1 min |
| 5 | `05_almgren_chriss_powerlaw.ipynb` | Modèle AC power-law | ~1 min |

*Configurable via `DURATION_MINUTES`

---

## Données

| Source | Actifs | Fréquence | Période |
|--------|--------|-----------|---------|
| Databento (S3) | AAPL, MSFT, GOOG | 1 min | Jan-Juin 2025 |
| Binance API | BTC, ETH, SOL | 1 min | Novembre 2025 |
| Binance API | Order book | Temps réel | À la demande |

**Note** : Les données stocks (Databento) ne sont accessibles que depuis SSPCloud. Les données crypto fonctionnent partout.

**Langue** : Les notebooks techniques (01-05) sont rédigés en anglais. Le notebook de synthèse et le README sont disponibles en français et en anglais.

---

## Résultats clés

- **Calibration** : R² = 0.95 (modèle quadratique)
- **Gain d'optimisation** : 10-15% vs TWAP en régime d'urgence moyenne (κT ~ 1-3)
- **Insight** : Sur Binance, les frais de transaction (10 bps) dominent largement le spread (~0 bps)

---

## Références

1. Almgren, R., & Chriss, N. (2001). *Optimal execution of portfolio transactions*. Journal of Risk.
2. Kyle, A. S. (1985). *Continuous auctions and insider trading*. Econometrica.
3. Gatheral, J. (2010). *No-dynamic-arbitrage and market impact*. Quantitative Finance.

---

*ENSAE Paris - Python pour la Data Science (2025-2026)*
