import time
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever 

# 1. Initialiser le modèle (Vérifie le nom de ton modèle, Llama3 ou llama3.2)
model = OllamaLLM(model="llama3") 

# 2. Créer le template (Comme fait Tim)
template = """Tu es un assistant expert pour m'aider à analyser mes documents personnels.
Voici les documents pertinents trouvés : 
{context}

Voici ma question :
{question}

Réponds en français en te basant uniquement sur les documents ci-dessus.
"""

prompt = ChatPromptTemplate.from_template(template)

# 3. Créer la "Chain" avec la nouvelle syntaxe LCEL (le symbole "pipe" | )
chain = prompt | model

USER_COLOR = '\033[94m'    # Bleu
BOT_COLOR = '\033[92m'     # Vert
SYS_COLOR = '\033[95m'
RESET_COLOR = '\033[0m'    # Réinitialise la couleur

print(f"\n{SYS_COLOR}======================================================{RESET_COLOR}")
print(f"{SYS_COLOR}   🧠 BIENVENUE DANS MyRAG - TON ASSISTANT LOCAL 🤖   {RESET_COLOR}")
print(f"{SYS_COLOR}======================================================{RESET_COLOR}")
print(f"{SYS_COLOR}[Système] Base vectorielle chargée. Prêt à répondre.{RESET_COLOR}")
print(f"{SYS_COLOR}[Système] Tape 'q' pour quitter.{RESET_COLOR}\n")

while True:
    print(f"{USER_COLOR}------------------------------------------------------{RESET_COLOR}")
    question = input(f"{USER_COLOR}👤 Arthur : {RESET_COLOR}")
    
    if question.lower() == 'q':
        print(f"\n{SYS_COLOR}[Système] Déconnexion de MyRAG. À bientôt ! 👋{RESET_COLOR}\n")
        break
        
    print(f"\n{SYS_COLOR}🔍 MyRAG fouille dans tes documents...{RESET_COLOR}")

    start_time = time.time()
    
    # a. Chercher les documents pertinents via vector.py
    relevant_docs = retriever.invoke(question)
    
    # b. Extraire le texte de ces documents
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # c. Poser la question au LLM avec le contexte
    result = chain.invoke({"context": context_text, "question": question})

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    
    # Affichage de la réponse
    print(f"\n{BOT_COLOR}🤖 MyRAG :{RESET_COLOR}")
    print(f"{BOT_COLOR}{result}{RESET_COLOR}\n")
    print(f"\n{SYS_COLOR}[⏱️ Temps de traitement : {elapsed_time} secondes]{RESET_COLOR}\n")