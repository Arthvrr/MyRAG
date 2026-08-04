from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import time
import subprocess
import os
import gc
import json
import chromadb
import re

# Import initial
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from vector import retriever as initial_retriever, USER_DOCS_PATH as initial_path, VECTOR_STORE_DIR

app = FastAPI(title="MyRAG Streaming API")

model = OllamaLLM(model="llama3") 

# NOUVEAU : On ajoute {chat_history} au prompt !
template = """Tu es MyRAG, l'assistant personnel d'Arthur.

Voici l'historique de votre conversation récente (mémoire) :
{chat_history}

Voici le contexte extrait des documents personnels d'Arthur : 
{context}

Voici la nouvelle question d'Arthur :
{question}

INSTRUCTIONS STRICTES :
1. Utilise l'historique pour comprendre le contexte si Arthur fait référence à quelque chose dont vous venez de parler (ex: "il", "ça", "cette personne").
2. Réponds de manière CLAIRE, DIRECTE et CONCISE. Va droit au but.
3. Ne justifie pas ta réponse en racontant comment tu as trouvé l'information.
4. Ne cite pas les phrases du contexte en entier, extrais uniquement l'information demandée.
5. Si la réponse n'est pas dans le contexte ou l'historique, dis-le poliment.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

current_retriever = initial_retriever
current_path = initial_path

# NOUVEAU : On ajoute "history" à la requête attendue
class ChatRequest(BaseModel):
    question: str
    history: list = [] 

class SourceRequest(BaseModel):
    path: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/pick_folder")
def pick_folder():
    try:
        script = """
        tell application (path to frontmost application as text)
            set folderPath to choose folder with prompt "Sélectionne le dossier source pour MyRAG :"
            return POSIX path of folderPath
        end tell
        """
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=True)
        path = result.stdout.strip()
        if path:
            return {"path": path}
        return {"path": ""}
    except subprocess.CalledProcessError:
        return {"path": ""}

@app.get("/pick_file")
def pick_file():
    try:
        script = """
        tell application (path to frontmost application as text)
            set filePath to choose file with prompt "Sélectionne le FICHIER pour MyRAG :"
            return POSIX path of filePath
        end tell
        """
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=True)
        path = result.stdout.strip()
        if path: return {"path": path}
        return {"path": ""}
    except subprocess.CalledProcessError: return {"path": ""}

@app.post("/chat")
def chat(request: ChatRequest):
    global current_retriever, current_path

    def event_generator():
        try:
            start_time = time.time()
            
            formatted_history = ""
            for msg in request.history:
                role = "Arthur" if msg.get("role") == "user" else "MyRAG"
                formatted_history += f"{role}: {msg.get('content')}\n"
            if not formatted_history:
                formatted_history = "(Début de la conversation. Aucun historique pour le moment.)"

            # 1. Aiguilleur (Excel vs ChromaDB)
            if os.path.isfile(current_path) and current_path.lower().endswith(('.csv', '.xlsx')):
                context_text = get_tabular_context(current_path)
                sources_uniques = [current_path]
            else:
                relevant_docs = current_retriever.invoke(request.question)
                context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                sources_uniques = list(set([doc.metadata.get('source', 'Source inconnue') for doc in relevant_docs]))
            
            # Variable pour stocker la réponse complète afin de pouvoir l'évaluer
            full_answer = ""
            
            # 2. Transmission en streaming
            for chunk in chain.stream({
                "context": context_text, 
                "question": request.question,
                "chat_history": formatted_history
            }):
                full_answer += chunk # On capture la réponse en direct
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            
            # ==========================================
            # 3. NOUVEAU : AUTO-ÉVALUATION (Self-Reflection)
            # ==========================================
            eval_template = """Tu es un juge IA très strict. 
            Contexte extrait : {context}
            Réponse générée : {answer}
            
            Analyse si la réponse générée est factuellement correcte et soutenue par le contexte.
            Donne un score de 0 à 100.
            Tu DOIS répondre UNIQUEMENT par un JSON valide, sans aucun texte avant ou après.
            Exemple : {{"score": 95}}
            """
            eval_prompt = ChatPromptTemplate.from_template(eval_template)
            eval_chain = eval_prompt | model
            
            try:
                # On lance l'évaluation en silence
                raw_eval = eval_chain.invoke({"context": context_text, "answer": full_answer})
                # On extrait le chiffre avec Regex pour éviter les bugs si le LLM bavarde
                match = re.search(r'["\']?score["\']?\s*:\s*(\d+)', raw_eval, re.IGNORECASE)
                confidence_self = int(match.group(1)) if match else "N/A"
            except Exception as e:
                confidence_self = "Err"
            # ==========================================

            elapsed_time = round(time.time() - start_time, 2)
            
            meta_payload = {
                'metadata': {
                    'sources': sources_uniques,
                    'time': elapsed_time,
                    'current_path': current_path,
                    'confidence_self': confidence_self # Ajout du score !
                }
            }
            yield f"data: {json.dumps(meta_payload)}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/change_source")
def change_source(request: SourceRequest):
    global current_retriever, current_path
    
    try:
        current_retriever = None
        gc.collect() 
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        if request.path == "./data":
            result = subprocess.run(["./reload_vector.sh"], check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(["./change_vector.sh", request.path], check=True, capture_output=True, text=True)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_DIR, 
            embedding_function=embeddings
        )
        current_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        current_path = request.path
        
        return {"status": "success", "message": f"Base reconstruite depuis {request.path} !"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Erreur : {e.stderr}"}