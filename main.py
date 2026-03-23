from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# On importe l'outil de recherche qu'on a créé dans vector.py
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

# 4. Boucle d'application terminal (Exactement comme dans la vidéo)
print("Pose tes questions sur tes documents ! (Tape 'q' pour quitter)")

while True:
    question = input("\nTa question : ")
    if question.lower() == 'q':
        break
        
    print("Recherche dans les documents...")
    # a. Chercher les documents pertinents via vector.py
    relevant_docs = retriever.invoke(question)
    
    # b. Extraire le texte de ces documents
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    print("\n[DEBUG] --- Ce que l'IA a trouvé dans la base et va lire : ---")
    print(context_text)
    print("----------------------------------------------------------\n")
    
    # c. Poser la question au LLM avec le contexte
    result = chain.invoke({"context": context_text, "question": question})
    
    print("\n--- Réponse de l'IA ---")
    print(result)