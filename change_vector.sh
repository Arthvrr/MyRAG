#!/bin/bash

if [ -z "$1" ]; then
    echo "❌ Erreur : Tu dois fournir un chemin vers un dossier ou fichier."
    echo "👉 Utilisation : ./change_vector.sh /chemin/vers/ton/fichier"
    exit 1
fi

TARGET_PATH="$1"

# CHANGEMENT ICI : -e (Exists) au lieu de -d (Directory)
if [ ! -e "$TARGET_PATH" ]; then
    echo "❌ Erreur : Le fichier ou dossier '$TARGET_PATH' n'existe pas."
    exit 1
fi

echo "======================================================"
echo "🔄 Changement de la source de données vers :"
echo "📂 $TARGET_PATH"
echo "======================================================"

if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

echo "🗑️ Suppression de l'ancienne base vectorielle..."
rm -rf chroma_db/

echo "⚙️ Création de la nouvelle base vectorielle..."
echo "$TARGET_PATH" > .current_path.txt

python3 vector.py

echo "======================================================"
echo "✅ Opération terminée !"