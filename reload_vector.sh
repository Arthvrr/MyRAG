#!/bin/bash

echo "🔄 RECHARGEMENT DE LA BASE PAR DÉFAUT (./data)"

if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

echo "🗑️ Suppression de l'ancienne base vectorielle (chroma_db/)..."
rm -rf chroma_db/

echo "./data" > .current_path.txt

echo "⚙️ Recréation de la base..."
python3 vector.py

echo "✅ Opération terminée !"