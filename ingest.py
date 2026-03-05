import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Nouvelle version plus stable
from langchain_ollama import OllamaEmbeddings # Version à jour

DOCS_PATH = os.path.expanduser("./data")
print(f"🚀 Début de l'indexation dans : {DOCS_PATH}")

# On garde tes loaders, mais on ajoute un filtre pour ignorer les fichiers trop bizarres
loaders = [
    DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
    DirectoryLoader(DOCS_PATH, glob="**/*.md", loader_cls=TextLoader),
    DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
]

docs = []
for loader in loaders:
    try:
        docs.extend(loader.load())
    except Exception as e:
        print(f"⚠️ Fichier ignoré (erreur de lecture) : {e}")

# IMPORTANT : On réduit le chunk_size à 500 pour être sûr que ça passe dans le contexte
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    add_start_index=True # Aide à retrouver l'endroit exact dans le fichier
)
chunks = text_splitter.split_documents(docs)
print(f"✂️  Texte découpé en {len(chunks)} morceaux sécurisés.")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

print("🧠 Génération des embeddings... (C'est le moment de prendre un café ☕️)")
# On ajoute un try/except ici aussi pour voir quel morceau pose problème si ça plante encore
try:
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ MyRAG est à jour !")
except Exception as e:
    print(f"❌ Erreur critique lors de la vectorisation : {e}")