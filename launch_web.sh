#!/bin/bash

if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
    echo "✅ Ollama est déjà en cours d'exécution."
else
    echo "🦙 Démarrage d'Ollama..."
    open -a Ollama
fi

if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

echo "🚀 Démarrage de MyRAG Studio..."

(sleep 2 && open -a Safari http://127.0.0.1:8000) &

echo "🌐 Ouverture automatique de Safari sur : http://127.0.0.1:8000"

uvicorn api:app --reload