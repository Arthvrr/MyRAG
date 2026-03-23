import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader # CHANGEMENT ICI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration
USER_DOCS_PATH = "./data"
VECTOR_STORE_DIR = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

if not os.path.exists(VECTOR_STORE_DIR):
    print("Création de la base vectorielle en cours...")
    
    # CHANGEMENT ICI : On utilise PyMuPDFLoader pour les .pdf
    loaders = {".pdf": PyMuPDFLoader, ".txt": TextLoader}
    documents = []
    for ext, loader_cls in loaders.items():
        if loader_cls == TextLoader:
             loader = DirectoryLoader(USER_DOCS_PATH, glob=f"**/*{ext}", loader_cls=loader_cls, loader_kwargs={'encoding': 'utf-8'})
        else:
            loader = DirectoryLoader(USER_DOCS_PATH, glob=f"**/*{ext}", loader_cls=loader_cls)
        documents.extend(loader.load())
    
    # Découper le texte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)
    
    # Sauvegarder dans Chroma
    vector_store = Chroma.from_documents(
        documents=all_splits, 
        embedding=embeddings, 
        persist_directory=VECTOR_STORE_DIR
    )
    print("Base vectorielle créée avec succès !")
else:
    print("Chargement de la base vectorielle existante...")
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIR, 
        embedding_function=embeddings
    )

# 3. Créer le retriever (l'outil de recherche) qu'on va exporter
retriever = vector_store.as_retriever(search_kwargs={"k": 10})