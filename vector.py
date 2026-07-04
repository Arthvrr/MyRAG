import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PATH_FILE = ".current_path.txt"

if os.path.exists(PATH_FILE):
    with open(PATH_FILE, "r") as f:
        USER_DOCS_PATH = f.read().strip()
else:
    USER_DOCS_PATH = "./data"

VECTOR_STORE_DIR = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

if not os.path.exists(VECTOR_STORE_DIR):
    print("Création de la base vectorielle en cours...")
    documents = []
    
    # --- NOUVEAU : GESTION D'UN FICHIER UNIQUE ---
    if os.path.isfile(USER_DOCS_PATH):
        print(f"📄 Traitement d'un fichier unique : {USER_DOCS_PATH}")
        ext = os.path.splitext(USER_DOCS_PATH)[1].lower()
        if ext == '.pdf':
            loader = PyMuPDFLoader(USER_DOCS_PATH)
            documents.extend(loader.load())
        elif ext in ['.txt', '.csv', '.md']:
            loader = TextLoader(USER_DOCS_PATH, encoding='utf-8')
            documents.extend(loader.load())
        else:
            print(f"❌ Format non supporté : {ext}")

    # --- ANCIEN CODE : GESTION D'UN DOSSIER COMPLET ---
    elif os.path.isdir(USER_DOCS_PATH):
        print(f"📁 Traitement d'un dossier : {USER_DOCS_PATH}")
        loaders = {".pdf": PyMuPDFLoader, ".txt": TextLoader}
        for ext, loader_cls in loaders.items():
            if loader_cls == TextLoader:
                 loader = DirectoryLoader(USER_DOCS_PATH, glob=f"**/*{ext}", loader_cls=loader_cls, loader_kwargs={'encoding': 'utf-8'})
            else:
                loader = DirectoryLoader(USER_DOCS_PATH, glob=f"**/*{ext}", loader_cls=loader_cls)
            documents.extend(loader.load())
    
    # Découper le texte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)
    
    if not all_splits:
        print(f"❌ Aucun texte n'a pu être extrait de : {USER_DOCS_PATH}")
    else:
        vector_store = Chroma.from_documents(
            documents=all_splits, 
            embedding=embeddings, 
            persist_directory=VECTOR_STORE_DIR
        )
        print(f"\nBase vectorielle créée avec succès ! ({len(all_splits)} morceaux générés)")
        print("📂 Sources ingérées dans la base :")
        sources_uniques = set([doc.metadata.get('source', 'Source inconnue') for doc in all_splits])
        for source in sorted(sources_uniques):
            print(f"  ✅ {source}")
else:
    print(f"🔄 Chargement de la base vectorielle existante (Source : {USER_DOCS_PATH})...")
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIR, 
        embedding_function=embeddings
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 10})