# 🧠 MyRAG - Ton Assistant Personnel Local & Intelligent

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-⚡-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-black.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)

**MyRAG** est un système de Retrieval-Augmented Generation (RAG) 100% local, conçu pour interagir avec tes documents personnels en toute confidentialité. Pose des questions sur tes PDF ou fichiers textes, et l'IA te répondra de manière claire, directe et sourcée, sans jamais envoyer tes données sur le cloud !

## ✨ Fonctionnalités Principales

*   🔒 **100% Privé & Local :** Propulsé par Ollama (Llama 3 et nomic-embed-text), aucune donnée ne quitte ton Mac.
*   📂 **Ingestion Dynamique (Multi-Dossiers) :** Tu n'es pas limité à un seul dossier ! Un script bash te permet de cibler n'importe quel dossier de ton ordinateur à la volée pour changer le cerveau de ton IA.
*   🛡️ **Architecture "Bulletproof" :** Gestion automatique de l'environnement virtuel (`venv`), filets de sécurité anti-crash (dossiers vides, sécurité macOS), et persistance du chemin de la base vectorielle.
*   🎨 **Interface Terminal Améliorée :** Des couleurs, des emojis, un affichage clair du temps de traitement, et surtout, la liste des sources exactes utilisées pour chaque réponse.
*   🐛 **Mode Debug Intégré :** Une simple variable à changer pour voir exactement quels morceaux de texte l'IA est en train de lire sous le capot.

## 🛠️ Stack Technique

*   **LLM :** `llama3` (via Ollama)
*   **Embeddings :** `nomic-embed-text` (via Ollama)
*   **Vector Store :** ChromaDB
*   **Framework :** LangChain
*   **Loaders :** PyMuPDF (pour les `.pdf`) et TextLoader (pour les `.txt`)

## 🎮 Comment l'utiliser ?

Le projet est piloté par 3 scripts Bash hyper simples d'utilisation (qui gèrent eux-mêmes l'activation du `venv` !).