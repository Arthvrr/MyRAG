import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Configurer les composants
print("🧠 Initialisation du cerveau local...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="llama3", temperature=0.2)

# 2. Charger la base de données
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. La boucle de discussion simplifiée
print("\n🤖 MyRAG est prêt ! (Tape 'exit' pour quitter)")

while True:
    query = input("\n👤 Arthur : ")
    if query.lower() in ['exit', 'quit']:
        break
    
    print("🔍 Recherche dans tes documents...")
    docs = vector_db.similarity_search(query, k=5)
    # Ajoute ceci pour voir ce que MyRAG "lit" réellement
    print(f"DEBUG: J'ai trouvé {len(docs)} morceaux de texte.")
    for i, doc in enumerate(docs):
        print(f"--- Morceau {i+1} (Source: {doc.metadata.get('source')}) ---\n{doc.page_content[:200]}...\n")
    
    # On assemble le contexte et on affiche les sources pour débugger
    context = "\n\n".join([f"Extrait de {doc.metadata.get('source', 'Inconnu')}:\n{doc.page_content}" for doc in docs])
    
    # C'est ici qu'on verrouille l'IA (Le System Prompt)
    messages = [
        SystemMessage(content=(
            "Tu es l'assistant personnel d'Arthur. RÉGLES CRITIQUES :\n"
            "1. Utilise UNIQUEMENT les extraits fournis ci-dessous pour répondre.\n"
            "2. Si la réponse n'est pas dans le contexte, dis : 'Désolé Arthur, je ne trouve pas cette info dans tes fichiers.'\n"
            "3. Ne mentionne jamais quelque chose si elle n'est pas citée dans le contexte.\n"
            f"\n--- CONTEXTE ---\n{context}"
        )),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    print(f"\n🤖 MyRAG : {response.content}")