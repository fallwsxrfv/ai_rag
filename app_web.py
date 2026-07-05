import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from config import APP_CONFIG, get_db_path


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Local Network RAG Portal", page_icon="🤖", layout="centered")

st.title("🤖 Local Network RAG Portal")
st.write("Interact with your isolated local knowledge bases over the network.")
st.markdown("---")



# --- AUTO-DISCOVER SUBJECTS ---
# Scan the database directory to see what subjects are currently built
embedding_namespace = APP_CONFIG["embedding_model"].replace(":", "-").replace("/", "-")
db_lookup_root = os.path.join(APP_CONFIG["db_root_dir"], embedding_namespace)

available_subjects = []
if os.path.exists(db_lookup_root):
    available_subjects = [
        f for f in os.listdir(db_lookup_root) 
        if os.path.isdir(os.path.join(db_lookup_root, f))
    ]

# Error guardrail if no databases exist yet
if not available_subjects:
    st.error(f"No vector databases found in `{db_lookup_root}`. Please run your ingestion script first!")
    st.stop()

# --- SIDEBAR CONTROL PANEL (Admin Mode / Info) ---
with st.sidebar:
    st.header("⚙️ System Status")
    search_subfolders = st.checkbox("Scan subfolders in source_docs", value=True)
    st.markdown("---")
    st.info(f"**Default LLM:** `{APP_CONFIG['default_model']}`")
    st.info(f"**Embedding Math:** `{APP_CONFIG['embedding_model']}`")
    st.markdown("---")
    st.caption("Running natively on Alienware Ryzen via WSL2")

# --- MAIN UI FORM ---
# Dropdown for selecting the subject shard
selected_subject = st.selectbox("📁 Choose Subject Matter:", sorted(available_subjects))

# Chat history initialization (keeps text on screen during streaming)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
if user_query := st.chat_input(f"Ask a question about {selected_subject}..."):
    
    # Display user's question immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display assistant response box with streaming animation
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # 1. Initialize DB and model components based on selection
        target_db_dir = get_db_path(selected_subject)
        
        embeddings = OllamaEmbeddings(
            model=APP_CONFIG["embedding_model"], 
            base_url=APP_CONFIG["base_url"]
        )
        vector_db = Chroma(persist_directory=target_db_dir, embedding_function=embeddings)
        
        llm = OllamaLLM(model=APP_CONFIG["default_model"], base_url=APP_CONFIG["base_url"])


        # If the user has selected 'All_Docs', we use a higher k and a stricter approach
        # --- 2. Retrieve document context chunks ---
        if selected_subject == "All_Docs":
            # Force the retriever wrapper to explicitly break the default 3-doc cap
            retriever = vector_db.as_retriever(search_kwargs={"k": 15})
            matching_docs = retriever.invoke(user_query)
            
            # Extract keywords for re-ranking
            query_words = [w.lower() for w in user_query.split() if len(w) > 3]
            
            def score_doc(doc):
                content_lower = doc.page_content.lower()
                return sum(1 for word in query_words if word in content_lower)
            
            # Sort the 15 chunks so the literal matches jump the line
            matching_docs = sorted(matching_docs, key=score_doc, reverse=True)[:5]

        else:
            # If they chose a specific shard like 'swimming', standard search is perfectly fine
            matching_docs = vector_db.similarity_search(user_query, k=4)


        
        context = ""
        for doc in matching_docs:
            context += f"\n--- Source: {doc.metadata.get('source')} ---\n{doc.page_content}\n"

        final_prompt = f"""Use the following snippets to accurately answer the question.
If you don't know the answer based on this context, say so.

Context:
{context}

Question: {user_query}
"""

        # 3. Stream output tokens straight to web interface
        full_response = ""
        for chunk in llm.stream(final_prompt):
            full_response += chunk
            # Update the web browser dynamically as chunks arrive
            response_placeholder.markdown(full_response + "▌")
            
        # Lock in final clean text removing cursor icon
        response_placeholder.markdown(full_response)
        
    # Append final answer to persistent history state
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    if 'matching_docs' in locals() and matching_docs:
        with st.expander("📄 View Source Documents"):
            seen_files = set()
            for doc in matching_docs:
                file_path = doc.metadata.get("source", "Unknown Source")
                # Extract just the filename (e.g., swimming_tutorial.pdf)
                file_name = file_path.split("/")[-1]
                
                # Avoid listing the exact same file path multiple times if it pulled 2 chunks from it
                if file_name not in seen_files:
                    st.markdown(f"**Document:** `{file_name}`")
                    seen_files.add(file_name)

