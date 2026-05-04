#!/bin/bash

if [ -z "$1" ]; then
    echo "❌ Erreur : Tu dois fournir un chemin vers un dossier."
    echo "👉 Utilisation : ./change_vector.sh /chemin/vers/ton/dossier"
    exit 1
fi

TARGET_PATH="$1"

if [ ! -d "$TARGET_PATH" ]; then
    echo "❌ Erreur : Le dossier '$TARGET_PATH' n'existe pas."
    exit 1
fi

echo "🔄 Changement de la source de données vers :"
echo "📂 $TARGET_PATH"

if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

echo "🗑️ Suppression de l'ancienne base vectorielle..."
rm -rf chroma_db/

echo "$TARGET_PATH" > .current_path.txt

echo "⚙️ Création de la nouvelle base vectorielle..."
python3 vector.py

echo "✅ Opération terminée !"