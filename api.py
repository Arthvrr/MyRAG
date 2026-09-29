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

class ChatRequest(BaseModel):
    question: str
    history: list = [] 

class SourceRequest(BaseModel):
    path: str

class ModelRequest(BaseModel):
    model: str

# --- NOUVEAU : Modèle pour la taille des fragments ---
class ChunksRequest(BaseModel):
    k: int

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

            # 1. Recherche du contexte
            relevant_docs = current_retriever.invoke(request.question)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            sources_uniques = list(set([doc.metadata.get('source', 'Source inconnue') for doc in relevant_docs]))
            
            full_answer = ""
            
            # 2. Transmission en streaming
            for chunk in chain.stream({
                "context": context_text, 
                "question": request.question,
                "chat_history": formatted_history
            }):
                full_answer += chunk 
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            
            # ==========================================
            # 3A. AUTO-ÉVALUATION (Self-Reflection)
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
                raw_eval = eval_chain.invoke({"context": context_text, "answer": full_answer})
                match = re.search(r'["\']?score["\']?\s*:\s*(\d+)', raw_eval, re.IGNORECASE)
                confidence_self = int(match.group(1)) if match else "N/A"
            except Exception:
                confidence_self = "Err"

            # ==========================================
            # 3B. ÉVALUATION RAGAS (La Triade)
            # ==========================================
            ragas_template = """Tu es un évaluateur expert de systèmes IA (RAGAS).
            Analyse cette interaction :
            QUESTION : {question}
            CONTEXTE RETROUVÉ : {context}
            RÉPONSE GÉNÉRÉE : {answer}
            
            Évalue les 3 métriques suivantes de 0 à 100 :
            1. context_relevance : Le contexte contient-il les informations pour répondre à la question ?
            2. faithfulness : La réponse est-elle strictement basée sur le contexte (sans hallucinations) ?
            3. answer_relevance : La réponse répond-elle directement à la question posée ?
            
            Tu DOIS répondre UNIQUEMENT avec un objet JSON valide. Exemple :
            {{"context_relevance": 90, "faithfulness": 100, "answer_relevance": 85}}
            """
            ragas_prompt = ChatPromptTemplate.from_template(ragas_template)
            ragas_chain = ragas_prompt | model
            
            try:
                raw_ragas = ragas_chain.invoke({"question": request.question, "context": context_text, "answer": full_answer})
                c_rel = int(re.search(r'"context_relevance"\s*:\s*(\d+)', raw_ragas, re.IGNORECASE).group(1))
                faith = int(re.search(r'"faithfulness"\s*:\s*(\d+)', raw_ragas, re.IGNORECASE).group(1))
                a_rel = int(re.search(r'"answer_relevance"\s*:\s*(\d+)', raw_ragas, re.IGNORECASE).group(1))
                ragas_scores = {"context": c_rel, "faithfulness": faith, "answer": a_rel}
            except Exception:
                ragas_scores = {"context": "Err", "faithfulness": "Err", "answer": "Err"}
            # ==========================================

            elapsed_time = round(time.time() - start_time, 2)
            
            meta_payload = {
                'metadata': {
                    'sources': sources_uniques,
                    'time': elapsed_time,
                    'current_path': current_path,
                    'confidence_self': confidence_self,
                    'ragas_scores': ragas_scores
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

@app.get("/stats")
def get_stats():
    global current_retriever, current_path, model
    chunks_count = 0
    k_current = 10
    if current_retriever:
        if hasattr(current_retriever, "vectorstore"):
            try: chunks_count = current_retriever.vectorstore._collection.count()
            except Exception: pass
        if hasattr(current_retriever, "search_kwargs"):
            k_current = current_retriever.search_kwargs.get("k", 10)
            
    return {
        "path": current_path, 
        "chunks": chunks_count, 
        "model": model.model,
        "k": k_current
    }

@app.post("/set_model")
def set_model(request: ModelRequest):
    global model, chain, prompt
    try:
        model = OllamaLLM(model=request.model)
        chain = prompt | model
        return {"status": "success", "model": request.model}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- NOUVEAU : Route pour modifier le nombre de fragments (k) ---
@app.post("/set_chunks")
def set_chunks(request: ChunksRequest):
    global current_retriever
    try:
        if current_retriever:
            if not hasattr(current_retriever, "search_kwargs") or current_retriever.search_kwargs is None:
                current_retriever.search_kwargs = {}
            current_retriever.search_kwargs["k"] = request.k
            return {"status": "success", "k": request.k}
        else:
            return {"status": "error", "message": "Aucune base chargée."}
    except Exception as e:
        return {"status": "error", "message": str(e)}