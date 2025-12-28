# Market Impact Models - Almgren-Chriss

**Auteurs** : Gregoire Marguier - Pierre Robin-Schnepf
**Formation** : ENSAE Paris - Python pour la Data Science (2024-2025)
**Encadrant** : Mr Couralet

---

## Problematique

Comment modeliser et optimiser les couts de transaction lies a l'impact de marche lors de l'execution d'ordres importants ?

Ce projet implemente le **modele d'Almgren-Chriss** pour l'execution optimale d'ordres sur Bitcoin (BTCUSDT), avec :
- Collecte de donnees d'orderbook en temps reel (Binance API)
- Calibration des parametres du modele
- Comparaison de strategies d'execution (TWAP, VWAP, Optimal)

---

## Installation

```bash
# Cloner le projet
git clone https://github.com/[votre-repo]/market-impact-models.git
cd market-impact-models

# Creer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dependances
pip install -r requirements.txt
```

---

## Execution des notebooks

**Les notebooks doivent etre executes dans l'ordre suivant :**

### Etape 1 : Collecte des donnees historiques
```
notebooks/01_data_collection.ipynb
```
- Telecharge les donnees OHLCV depuis Binance (1 mois de donnees minute)
- Calcule les parametres de marche (volatilite, volume)
- **Sortie** : `data/processed/crypto/*.parquet`, `data/processed/market_parameters.parquet`
- **Duree** : ~5 minutes

### Etape 2 : Collecte de l'orderbook
```
notebooks/02_orderbook_collection.ipynb
```
- Collecte des snapshots d'orderbook en temps reel
- **Sortie** : `data/orderbook/BTCUSDT_orderbook_*.json`
- **Duree** : ~60 minutes (configurable via `DURATION_MINUTES`)

> **Note** : Vous pouvez reduire `DURATION_MINUTES` a 5-10 pour un test rapide.

### Etape 3 : Calibration des parametres
```
notebooks/03_calibration.ipynb
```
- Estime les parametres du modele (k, eta, phi, psi) depuis l'orderbook
- **Sortie** : `data/results/calibrated_parameters.json`
- **Prerequis** : Notebooks 01 et 02 executes

### Etape 4 : Modele Almgren-Chriss quadratique
```
notebooks/04_almgren_chriss_quadratic.ipynb
```
- Implemente le modele AC avec couts quadratiques
- Solution analytique (sinh/cosh)
- Comparaison TWAP vs Optimal
- **Prerequis** : Notebook 03 execute

### Etape 5 : Modele Almgren-Chriss power-law
```
notebooks/05_almgren_chriss_powerlaw.ipynb
```
- Implemente le modele AC avec couts power-law
- Solution numerique (BVP solver)
- Comparaison complete des strategies
- **Prerequis** : Notebook 03 execute

---

## Structure du projet

```
market-impact-models/
│
├── notebooks/                        # Notebooks (workflow principal)
│   ├── 01_data_collection.ipynb          # Collecte donnees historiques
│   ├── 02_orderbook_collection.ipynb     # Collecte orderbook temps reel
│   ├── 03_calibration.ipynb              # Calibration parametres
│   ├── 04_almgren_chriss_quadratic.ipynb # Modele AC quadratique
│   └── 05_almgren_chriss_powerlaw.ipynb  # Modele AC power-law
│
├── src/                              # Modules Python
│   ├── models/almgren_chriss.py          # Implementation du modele AC
│   ├── optimization/strategies.py        # Strategies TWAP, VWAP, Optimal
│   ├── calibration/calibrator.py         # Calibration des parametres
│   └── data/collectors.py                # Collecteurs Binance
│
├── data/                             # Donnees (generees par les notebooks)
│   ├── processed/                        # Donnees historiques
│   ├── orderbook/                        # Snapshots orderbook
│   └── results/                          # Parametres calibres
│
├── tests/                            # Tests unitaires
├── config/                           # Configuration
└── docs/                             # Documentation
```

---

## Theorie : Modele d'Almgren-Chriss

### Objectif
Minimiser le cout d'execution d'un ordre de taille X sur un horizon T :

```
min  E[Cout] + lambda * Var[Cout]
```

### Composantes du cout

| Composante | Formule | Description |
|------------|---------|-------------|
| Impact permanent | `k * rho` | Deplacement definitif du prix |
| Impact temporaire | `eta * rho^(1+phi)` | Cout d'execution instantane |
| Spread | `psi * rho` | Cout du bid-ask spread |

Ou `rho = v/V` est le taux de participation.

### Parametres calibres (exemple BTCUSDT)

| Parametre | Description | Valeur typique |
|-----------|-------------|----------------|
| k | Impact permanent | ~10^-5 |
| eta | Coefficient impact temporaire | 0.05 - 0.20 |
| phi | Exposant power-law | 0.5 |
| psi | Spread | 0.5 - 5 bps |

---

## Strategies d'execution

| Strategie | Description |
|-----------|-------------|
| **TWAP** | Execution uniforme dans le temps |
| **VWAP** | Proportionnel au volume de marche |
| **Optimal AC** | Minimise cout + risque (Almgren-Chriss) |

---

## Tests

```bash
pytest tests/ -v
# Resultat attendu: 32 tests passed
```

---

## References

- Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions
- Almgren et al. (2005) - Direct Estimation of Equity Market Impact
- Kyle (1985) - Continuous Auctions and Insider Trading

---

## Notes pour le correcteur

1. **Execution sur SSP Cloud** : Le projet est prevu pour etre execute sur SSP Cloud ou les donnees S3 sont disponibles. En local, les donnees stocks ne seront pas chargees (seules les donnees crypto Binance fonctionnent).

2. **Temps d'execution** : Le notebook 02 (orderbook) prend ~1h. Vous pouvez reduire `DURATION_MINUTES` pour un test rapide.

3. **Donnees pre-existantes** : Si les donnees dans `data/` existent deja, vous pouvez sauter les etapes 1-2 et commencer directement a l'etape 3.

---

**ENSAE Paris - Python pour la Data Science (2024-2025)**
