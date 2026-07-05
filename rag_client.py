import sys
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from config import APP_CONFIG, get_db_path

def query_subject_rag(subject_name: str, user_question: str):
    # Determine the configuration path
    target_db_dir = get_db_path(subject_name)
    
    # Strict path check to guarantee we aren't querying an empty database
    if not os.path.exists(target_db_dir) or not os.listdir(target_db_dir):
        print(f"\n[!] Error: No database shard found for subject '{subject_name}' inside '{target_db_dir}'")
        print("Please run ingestion script first.")
        sys.exit(1)

    # Initialize native stack using central config rules
    embeddings = OllamaEmbeddings(
        model=APP_CONFIG["embedding_model"], 
        base_url=APP_CONFIG["base_url"]
    )
    vector_db = Chroma(persist_directory=target_db_dir, embedding_function=embeddings)
    
    # Pull the admin-defined default model
    llm = OllamaLLM(model=APP_CONFIG["default_model"], base_url=APP_CONFIG["base_url"])

    # Vector search top 3 matched chunks
    matching_docs = vector_db.similarity_search(user_question, k=3)

    context = ""
    for doc in matching_docs:
        context += f"\n--- Source: {doc.metadata.get('source')} ---\n{doc.page_content}\n"

    final_prompt = f"""Use the following codebase snippets to accurately answer the question.
If you don't know the answer based on this context, say so.

Context:
{context}

Question: {user_question}
"""

    print(f"\n[Subject: {subject_name}] ❯ {user_question}")
    print("--- AI Response ---")
    
    for chunk in llm.stream(final_prompt):
        print(chunk, end="", flush=True)
    print("\n-------------------")

if __name__ == "__main__":
    # Check for arguments: uv run rag_client.py <subject> <question>
    if len(sys.argv) < 3:
        print("Usage: uv run rag_client.py <subject_name> \"<your question>\"")
        print("Example: uv run rag_client.py aeroflow \"What is the specification?\"")
        sys.exit(1)
        
    sub = sys.argv[1]
    question = sys.argv[2]
    query_subject_rag(sub, question)
