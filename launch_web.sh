#!/bin/bash

echo "🦙 Vérification et lancement d'Ollama..."
# Lance l'application macOS Ollama en arrière-plan (sans bloquer le terminal)
open -a Ollama

if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

echo "🚀 Démarrage du serveur web MyRAG..."

(sleep 2 && open -a Safari http://127.0.0.1:8000) &

echo "🌐 Ouverture automatique de Safari sur : http://127.0.0.1:8000"

uvicorn api:app --reload