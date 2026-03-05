# 🤖 MyRAG : Assistant IA Personnel & Local

MyRAG est un outil de **Retrieval-Augmented Generation (RAG)** conçu pour indexer et interroger tes documents personnels (PDF, TXT, MD) en toute confidentialité. Contrairement aux solutions cloud, MyRAG fonctionne **100% localement sur ton Mac**, garantissant que tes données ne quittent jamais ta machine.

---

## 💡 Pourquoi MyRAG ?

- **Confidentialité totale** : Utilisation de modèles open-source locaux.  
- **Zéro Coût** : Pas de clés API payantes (OpenAI, Anthropic, etc.).  
- **Intelligence Contextuelle** : L'IA répond en se basant sur tes cours, tes notes et tes documents de vie.  

---

## 🛠️ La "Stack" Technique

Le projet repose sur la "Sainte Trinité" de l'IA locale en 2026 :

| Composant       | Outil utilisé       | Rôle                                                      |
|-----------------|------------------|-----------------------------------------------------------|
| LLM (Cerveau)   | Ollama (Llama 3)  | Génère les réponses de manière naturelle et concise.     |
| Embeddings      | mxbai-embed-large | Transforme le texte en vecteurs mathématiques (compréhension sémantique). |
| Vector Store    | ChromaDB          | Base de données qui stocke et recherche les extraits les plus pertinents. |
| Orchestrateur   | LangChain         | Fait le pont entre tes fichiers, la base de données et l'IA. |

---