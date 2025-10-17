# Market Impact Models - Almgren-Chriss & Slippage

**Auteurs** : Grégoire Marguier - Pierre Robin-Schnepf                                    
**Formation** : ENSAE Paris - Python pour la Data Science (2025-2026)  
**Encadrant** : Mr Couralet

## 📌 Problématique

Comment modéliser et optimiser les coûts de transaction liés à l'impact de marché pour différentes stratégies d'exécution d'ordres ?

## 🎯 Objectifs

1. Modéliser empiriquement le slippage sur données réelles
2. Calibrer et implémenter le modèle d'Almgren-Chriss
3. Comparer différentes stratégies d'exécution (TWAP, VWAP, optimal)
4. Analyser l'impact de la spécification du modèle sur les coûts

## 📊 Données

- **Source principale** : Yahoo Finance (yfinance)
- **Actifs** : Actions liquides du S&P 500 (AAPL, MSFT, GOOGL, JPM, etc.)
- **Période** : 2023-2024 (données intraday + historiques)
- **Compléments** : Données de spread bid-ask via Alpha Vantage API

## 🛠️ Installation

### Prérequis
- Python 3.10+
- Packages listés dans `requirements.txt`

### Setup rapide
```bash
git clone https://github.com/[votre-username]/market-impact-models.git
cd market-impact-models
pip install -r requirements.txt
```

### Utilisation sur SSPCloud
```bash
# Instructions spécifiques SSPCloud (à compléter)
```

## 📁 Structure du projet
```
market-impact-models/
├── data/                  # Données (gitignored si > 100MB)
│   ├── raw/              # Données brutes
│   ├── processed/        # Données nettoyées
│   └── README.md         # Description des données
├── notebooks/            # Notebooks Jupyter
│   ├── 01_data_collection.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_slippage_models.ipynb
│   └── 04_almgren_chriss.ipynb
├── src/                  # Code source Python
│   ├── __init__.py
│   ├── data_loader.py   # Fonctions de téléchargement
│   ├── slippage.py      # Modèles de slippage
│   ├── almgren_chriss.py # Implémentation A-C
│   └── utils.py         # Fonctions utilitaires
├── tests/                # Tests unitaires (optionnel)
├── docs/                 # Documentation
├── .gitignore
├── requirements.txt      # Dépendances Python
├── README.md
└── environment.yml       # Environnement conda (optionnel)
```

## 📈 Méthodologie

1. **Collecte & nettoyage** : Récupération via API, gestion valeurs manquantes
2. **Analyse descriptive** : Statistiques, visualisations interactives (Plotly)
3. **Modélisation slippage** : Linéaire, racine carrée, ML (Random Forest)
4. **Almgren-Chriss** : Calibration, optimisation, backtest

## 🔗 Ressources

- [Almgren & Chriss (2000)](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [Documentation yfinance](https://pypi.org/project/yfinance/)
- [Cours Python ENSAE](https://pythonds.linogaliana.fr/)

## 📅 Avancement

- [x] Initialisation du projet
- [x] Validation du sujet avec le chargé de TD
- [ ] Collecte des données
- [ ] Analyse exploratoire
- [ ] Modélisation
- [ ] Rapport final

---

**Note** : Ce projet est réalisé dans le cadre du cours Python pour la Data Science à l'ENSAE Paris.
