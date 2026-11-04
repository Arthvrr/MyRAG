#!/bin/bash

echo "🦙 Vérification et lancement d'Ollama..."
# Lance l'application macOS Ollama en arrière-plan (sans bloquer le terminal)
open -a Ollama

if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

echo "🚀 Démarrage de MyRAG CLI..."

python3 main.py