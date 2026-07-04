#!/bin/bash

if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

echo "🚀 Démarrage du serveur web MyRAG..."
echo "🌐 Ouvre Safari et va sur : http://127.0.0.1:8000"

# Lance Uvicorn sur le fichier api.py, app=nom de l'application FastAPI
uvicorn api:app --reload