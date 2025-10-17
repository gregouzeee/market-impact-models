#!/bin/bash

# Script d'initialisation pour SSPCloud
# Ce script sera exécuté au lancement du service Jupyter

echo "🚀 Initialisation du projet Market Impact Models..."

# Se placer dans le répertoire de travail
cd /home/onyxia/work

# Cloner le repo (ou le mettre à jour s'il existe déjà)
if [ -d "market-impact-models" ]; then
    echo "📂 Projet existant trouvé, mise à jour..."
    cd market-impact-models
    git pull origin main
else
    echo "📥 Clonage du repository..."
    git clone https://github.com/gregouzeee/market-impact-models.git
    cd market-impact-models
fi

# Installer les dépendances Python
echo "📦 Installation des dépendances..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Configuration Git (optionnel, à personnaliser)
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@ensae.fr"

# Créer les dossiers data s'ils n'existent pas
mkdir -p data/raw data/processed

echo "✅ Initialisation terminée avec succès !"
echo "📁 Projet disponible dans : /home/onyxia/work/market-impact-models"
echo ""
echo "🎯 Prochaines étapes :"
echo "   1. Ouvrir le notebook notebooks/00_project_setup.ipynb"
echo "   2. Configurer votre fichier .env avec vos credentials"
echo "   3. Commencer la collecte de données !"
