import os
import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

DB_DIR = "./local_chroma_db"
OLLAMA_URL = "http://localhost:11434"

# 1. Strict guard check: Verify database exists before anything else
if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    print("\n[!] Error: No database found. Please run 'uv run rag_create.py' first to index your documents.")
    sys.exit(1)

print("Loading local vector database and connecting to Ollama...")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
llm = OllamaLLM(model="qwen2.5:7b", base_url=OLLAMA_URL)

print("\n========================================================")
print(" RAG Chat Engine Active. Type your question below.")
print(" To exit the application, type: ***")
print("========================================================\n")

# 2. Interactive Loop
while True:
    query = input("\nUser Prompt ❯ ")

    # Check for exit condition
    if query.strip() == "***":
        print("Exiting chat session. Goodbye!")
        break

    if not query.strip():
        continue

    print("Searching documents...")
    # Fetch top 3 matches
    matching_docs = vector_db.similarity_search(query, k=3)

    # Package context matching chunks
    context = ""
    for doc in matching_docs:
        context += f"\n--- Source File: {doc.metadata.get('source')} ---\n{doc.page_content}\n"

    final_prompt = f"""Use the following codebase snippets to accurately answer the question.
If you don't know the answer based on this context, say so.

Context:
{context}

Question: {query}
"""

    print("Generating AI response...")
    response = llm.invoke(final_prompt)

    print("\n=== AI RESPONSE ===")
    print(response)
    print("===================")
