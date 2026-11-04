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

echo "🚀 Démarrage de MyRAG CLI..."

python3 main.py