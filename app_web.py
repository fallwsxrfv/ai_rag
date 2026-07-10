import os
import streamlit as st
import base64
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from config import APP_CONFIG, get_db_path
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredFileLoader

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Local Network RAG Portal", page_icon="🤖", layout="wide")
# --- CUSTOM CSS FOR LARGE CHAT INPUT FONT ---
st.markdown(
    """
    <style>
    /* Target the chat input textarea */
    .stChatInput textarea {
        font-size: 1.25rem !important; /* Increases font size (approx 20px) */
        line-height: 1.5 !important;
    }
    
    /* Optional: Increase the placeholder text font size as well */
    .stChatInput textarea::placeholder {
        font-size: 1.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Local Network RAG Portal")
st.write("Interact with your isolated local knowledge bases over the network.")
st.markdown("---")


# --- AUTO-DISCOVER SUBJECTS ---
embedding_namespace = APP_CONFIG["embedding_model"].replace(":", "-").replace("/", "-")
db_lookup_root = os.path.join(APP_CONFIG["db_root_dir"], embedding_namespace)

available_subjects = []
if os.path.exists(db_lookup_root):
    available_subjects = [
        f for f in os.listdir(db_lookup_root)
        if os.path.isdir(os.path.join(db_lookup_root, f))
    ]

if not available_subjects:
    st.error(f"No vector databases found in `{db_lookup_root}`. Please run your ingestion script first!")
    st.stop()


# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ System Status")
    search_subfolders = st.checkbox("Scan subfolders in source_docs", value=True)
    st.markdown("---")
    st.info(f"**Default LLM:** `{APP_CONFIG['default_model']}`")
    st.info(f"**Embedding Math:** `{APP_CONFIG['embedding_model']}`")
    st.markdown("---")
    st.caption("Running natively on Alienware Ryzen via WSL2")


# --- MAIN UI FORM ---
selected_subject = st.selectbox("📁 Choose Subject Matter:", sorted(available_subjects))

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent storage for the last active document references
if "last_matching_docs" not in st.session_state:
    st.session_state.last_matching_docs = None

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User prompt input
if user_query := st.chat_input(f"Ask a question about {selected_subject}..."):

    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # 1. Initialize DB and model components
        target_db_dir = get_db_path(selected_subject)

        embeddings = OllamaEmbeddings(
            model=APP_CONFIG["embedding_model"],
            base_url=APP_CONFIG["base_url"]
        )
        vector_db = Chroma(persist_directory=target_db_dir, embedding_function=embeddings)
        llm = OllamaLLM(model=APP_CONFIG["default_model"], base_url=APP_CONFIG["base_url"])

        # --- 2. Retrieve document context chunks ---
        if selected_subject == "All_Docs":
            retriever = vector_db.as_retriever(search_kwargs={"k": 15})
            matching_docs = retriever.invoke(user_query)

            query_words = [w.lower() for w in user_query.split() if len(w) > 3]

            def score_doc(doc):
                content_lower = doc.page_content.lower()
                return sum(1 for word in query_words if word in content_lower)

            matching_docs = sorted(matching_docs, key=score_doc, reverse=True)[:5]
        else:
            matching_docs = vector_db.similarity_search(user_query, k=8)

        # Save to session state so buttons can access them after a page refresh
        st.session_state.last_matching_docs = matching_docs

        context = ""
        for doc in matching_docs:
            context += f"\n--- Source: {doc.metadata.get('source')} ---\n{doc.page_content}\n"


        final_prompt = f"""You are a precise technical analyst evaluating athletic biomechanics.
Use the provided snippets below to comprehensively answer the question. 

If the context explicitly discusses the components of the movement, synthesize them to explain the mechanical relationships. If the snippets are entirely unrelated to the topic, state that the context is insufficient.

Context material:    
{context}

Question: {user_query}
"""

        # 3. Stream output tokens
        full_response = ""
        for chunk in llm.stream(final_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})



# --- DYNAMIC ACTION AREA (Sources & Advanced Options) ---
if st.session_state.last_matching_docs:
    with st.expander("📄 View Source Documents & Advanced Actions", expanded=True):
        
        # --- SECTION 1: UNIQUE FILES AND ABSOLUTE PATHS ---
        st.markdown("### 📁 Original Files Used (Absolute Paths):")
        seen_files = set()
        unique_file_paths = set()
        
        for doc in st.session_state.last_matching_docs:
            # Handle both object attributes and dictionary keys gracefully
            file_path = doc.get("source") if isinstance(doc, dict) else doc.metadata.get("source")
            if file_path:
                abs_path = os.path.abspath(file_path)
                unique_file_paths.add(abs_path)
                
                file_name = abs_path.split("/")[-1].split("\\")[-1]
                if file_name not in seen_files:
                    st.markdown(f"**File:** `{file_name}`")
                    st.code(abs_path, language="bash")
                    seen_files.add(file_name)
        
        st.markdown("---")
        
        # --- SECTION 2: ORIGINAL PDF DOCUMENT VIEWER & DEEP ANALYSIS ---
        col1, col2 = st.columns([4, 3])  # Wide layout column split
        
        with col1:
            st.markdown("### 🔍 Original Document Viewer")
            st.caption("Select a source file below to render the original PDF directly inside the browser.")
            
            # Create a clean selection dropdown based on the active files found by RAG
            file_options = sorted(list(unique_file_paths))
            file_names_map = {path.split("/")[-1].split("\\")[-1]: path for path in file_options}
            
            selected_file_name = st.selectbox("🎯 Select Document to View:", ["-- Choose a File --"] + list(file_names_map.keys()))
            
            if selected_file_name != "-- Choose a File --":
                target_pdf_path = file_names_map[selected_file_name]
                
                if os.path.exists(target_pdf_path) and target_pdf_path.lower().endswith(".pdf"):
                    try:
                        # Stream the file content directly into a native downloadable/scrollable data frame
                        with open(target_pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        
                        st.markdown(f"**Viewing:** `{selected_file_name}`")
                        
                        # Fix: Standardized Download button so you can instantly pop it into a browser tab or viewer
                        st.download_button(
                            label="📂 Open/Save Copy in External PDF Viewer",
                            data=pdf_bytes,
                            file_name=selected_file_name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        # Alternative browser-safe fallback method
                        import base64
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></embed>'
                        st.components.v1.html(pdf_display, height=820, scrolling=False)
                        
                    except Exception as e:
                        st.error(f"Could not render PDF inline: {e}")
                elif not target_pdf_path.lower().endswith(".pdf"):
                    st.info("Inline viewing is optimized for PDF files. For raw text files, open the path directly.")
        
        with col2:
            st.markdown("### 🧠 Document-Level Synthesis")
            st.caption("Read the entire raw files whole into memory for a comprehensive analysis.")
            
            if st.button("🚀 Run Full Document Deep Analysis", use_container_width=True):
                if not unique_file_paths:
                    st.error("Could not trace chunks back to original files on disk.")
                else:
                    full_document_context = ""
                    
                    for path in unique_file_paths:
                        file_name = path.split("/")[-1].split("\\")[-1]
                        st.write(f"📖 Reading complete file: `{file_name}`...")
                        
                        try:
                            if path.lower().endswith('.pdf'):
                                import pypdf
                                reader = pypdf.PdfReader(path)
                                file_content = ""
                                for page in reader.pages:
                                    text = page.extract_text()
                                    if text:
                                        file_content += text + "\n"
                            elif path.lower().endswith('.txt'):
                                with open(path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                            else:
                                from langchain_community.document_loaders import UnstructuredFileLoader
                                loader = UnstructuredFileLoader(path)
                                raw_docs = loader.load()
                                file_content = "\n".join([d.page_content for d in raw_docs])
                            
                            full_document_context += f"\n=== BEGIN FULL DOCUMENT: {file_name} ===\n"
                            full_document_context += file_content
                            full_document_context += f"\n=== END FULL DOCUMENT: {file_name} ===\n"
                            
                        except Exception as e:
                            st.warning(f"Failed to read raw file {file_name}: {e}")

                    st.markdown("### 🤖 Synthesizing Full-Text Knowledge...")
                    
                    analysis_prompt = f"""You are a senior analytical engineer and domain expert. 
Conduct a comprehensive, exhaustive, and systematic analysis based on the complete text of the documents provided below.

Do not rely on shallow summaries. Dive deep into the core methodology, technical frameworks, structural mechanics, and systemic relationships explicitly detailed in these texts.

---
UNABRIDGED SOURCE MATERIAL:
{full_document_context}
---

Analytical Objective: Provide a rigorous structural breakdown of the concepts and technical processes detailed across these full documents.
"""
                    
                    analysis_llm = OllamaLLM(model=APP_CONFIG["default_model"], base_url=APP_CONFIG["base_url"])
                    analysis_placeholder = st.empty()
                    analysis_response = ""
                    
                    for chunk in analysis_llm.stream(analysis_prompt):
                        analysis_response += chunk
                        analysis_placeholder.markdown(analysis_response + "▌")
                    
                    analysis_placeholder.markdown(analysis_response)