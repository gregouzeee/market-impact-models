# Market Impact Parameter Calibration

Scripts pour calibrer les paramètres du modèle Almgren-Chriss à partir de données d'order book Binance.

## 🎯 Objectif

Calibrer empiriquement les paramètres du modèle d'impact de marché power-law :
- **k** : Impact permanent (Kyle's lambda)
- **η** : Coefficient de coût d'exécution temporaire
- **φ** : Exposant de la loi de puissance (0.3-0.7, typiquement 0.5)
- **ψ** : Coûts proportionnels (spread + frais)

## 📊 Données Nécessaires

### Option 1: Order Book en temps réel (RECOMMANDÉ)
Collecter des snapshots de l'order book Binance à intervalles réguliers.

**Avantages:**
- Gratuit (API publique Binance)
- Permet de calibrer tous les paramètres
- Donne une vue précise de la liquidité

**Inconvénients:**
- Nécessite 1-2 heures de collecte minimum
- Données statiques (snapshot à un instant t)

### Option 2: Données historiques d'exécution réelles
Si tu passes des ordres réels sur Binance, enregistrer :
- Taille de l'ordre
- Prix d'exécution moyen
- Slippage observé

## 🚀 Workflow Complet

### Étape 1: Collecter les données d'order book

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Collecter 1 heure de données toutes les 10 secondes
python scripts/collect_orderbook.py
```

**Paramètres à ajuster dans le script:**
- `duration_minutes=60` : Durée de collecte (60 min = 1h)
- `interval_seconds=10` : Intervalle entre snapshots
- `symbol='BTCUSDT'` : Paire de trading
- `limit=1000` : Nombre de niveaux d'order book (max 5000)

**Sortie:**
- Fichier JSON dans `data/orderbook/BTCUSDT_orderbook_YYYYMMDD_HHMMSS.json`
- ~360 snapshots pour 1h à 10s d'intervalle

### Étape 2: Calibrer les paramètres

```bash
python scripts/calibrate_impact_parameters.py
```

Le script va :
1. **Estimer ψ** (spread) : Médiane du spread bid-ask
2. **Estimer k** (Kyle's lambda) : Régression de l'impact sur le volume
3. **Estimer η et φ** : Ajustement non-linéaire de `L(ρ) = η|ρ|^(1+φ) + ψ|ρ|`

**Sortie:**
- Graphiques dans `results/impact_calibration.png`
- Paramètres dans `results/calibrated_parameters.json`
- Statistiques dans la console

### Étape 3: Utiliser les paramètres calibrés

Une fois les paramètres calibrés, les utiliser dans tes notebooks :

```python
import json

# Charger les paramètres calibrés
with open('results/calibrated_parameters.json', 'r') as f:
    params = json.load(f)

# Utiliser dans le modèle
k_calibrated = params['k']
eta_calibrated = params['eta']
phi_calibrated = params['phi']
psi_calibrated = params['psi']

print(f"Paramètres calibrés:")
print(f"  k   = {k_calibrated:.8f}")
print(f"  η   = {eta_calibrated:.6f}")
print(f"  φ   = {phi_calibrated:.4f}")
print(f"  ψ   = {psi_calibrated:.6f}")
```

## 📈 Interprétation des Résultats

### Kyle's Lambda (k)
- Mesure l'impact **permanent** d'un trade sur le prix
- Unité: ΔP/Q (changement de prix par unité tradée)
- Typique: 10⁻⁶ à 10⁻⁸ pour BTC
- Plus k est grand, plus le marché est **illiquide**

### Power-law exponent (φ)
- Décrit la **concavité** de la fonction d'impact
- φ = 0 : Impact linéaire
- φ = 0.5 : "Square root law" (Almgren et al. 2005)
- φ = 1 : Impact quadratique (modèle classique)
- Littérature empirique: φ ∈ [0.3, 0.7]

### Execution cost coefficient (η)
- Échelle de l'impact **temporaire**
- Dépend du marché et de l'unité de volume
- Doit être calibré empiriquement (pas de valeur théorique)

### Proportional costs (ψ)
- Coûts fixes par unité tradée
- ψ = spread/2 + trading fees
- Typique: 5-30 bps pour crypto

## 🔬 Validation de la Calibration

### Vérifications à faire:

1. **R² > 0.8** : Le modèle explique bien les données
2. **φ ∈ [0.3, 0.7]** : Cohérent avec la littérature
3. **ψ ≈ spread/2** : Cohérent avec le spread observé
4. **Impact croissant avec la taille** : Plus on trade, plus l'impact est grand

### Tests de robustesse:

```bash
# Collecter plusieurs sessions à différents moments
python scripts/collect_orderbook.py  # Session 1 (matin)
# Attendre quelques heures
python scripts/collect_orderbook.py  # Session 2 (après-midi)
# Attendre
python scripts/collect_orderbook.py  # Session 3 (soir)

# Calibrer sur chaque session et comparer
```

Si les paramètres varient beaucoup (> 50%), cela peut indiquer :
- Régimes de marché différents (haute/basse volatilité)
- Changements de liquidité intraday
- Événements de marché (news, manipulation)

## 🎓 Références

### Articles académiques:
1. **Almgren et al. (2005)** - "Direct Estimation of Equity Market Impact"
   - Première estimation empirique de φ ≈ 0.6
   - Méthodologie de calibration

2. **Kyle (1985)** - "Continuous Auctions and Insider Trading"
   - Définition de Kyle's lambda
   - Modèle théorique d'impact permanent

3. **Bacry et al. (2015)** - "Market Impact and Trading Profile of Hidden Orders"
   - Impact temporaire vs permanent
   - Decay functions

4. **Guéant (2016)** - "The Financial Mathematics of Market Liquidity"
   - Chapitre 3: Almgren-Chriss généralisé
   - Caractérisation Hamiltonienne

### Ressources techniques:
- [Binance API Documentation](https://binance-docs.github.io/apidocs/spot/en/)
- [Order Book Data Structures](https://www.binance.com/en/support/faq/what-is-an-order-book-360042991692)

## ⚠️ Limitations

### 1. Order book snapshots ≠ Trade data
- Les snapshots montrent la liquidité **disponible**
- Pas les trades **exécutés** réellement
- Peut surestimer la liquidité (orders fantômes, HFT)

### 2. Impact à court terme seulement
- Ces paramètres capturent l'impact **immédiat**
- Ne capturent pas le decay (réversion à long terme)
- Pour modéliser le decay: ajouter une fonction temporelle

### 3. Régime de marché
- Paramètres valables pour le régime actuel
- Peuvent changer avec volatilité/liquidité
- Re-calibrer régulièrement (hebdomadaire/mensuel)

## 💡 Prochaines Étapes

1. **Collecter plus de données** : 24h de snapshots pour voir l'intraday
2. **Tester sur différentes cryptos** : ETH, BNB, SOL...
3. **Implémenter le decay** : Modèle d'impact résilient
4. **Validation backtesting** : Comparer prédictions vs réalité

## 🆘 Troubleshooting

### Erreur: "Insufficient liquidity"
- Augmenter `limit` dans `collect_orderbook.py`
- Réduire les `test_sizes` dans la calibration

### Erreur: "Calibration failed"
- Pas assez de données : collecter plus longtemps
- Mauvaise initial guess : ajuster `p0` dans `curve_fit`
- Contraintes trop strictes : élargir `bounds`

### R² très faible (< 0.5)
- Données bruitées : augmenter durée de collecte
- Mauvais modèle : essayer φ fixe (0.5) et calibrer seulement η
- Outliers : filtrer les snapshots avec spread anormal