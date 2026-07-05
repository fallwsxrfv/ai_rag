import os
import sys
import zipfile
import signal
import pathlib
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import APP_CONFIG, get_db_path

# --- DEFENSIVE TIMEOUT ARCHITECTURE ---
class TimeoutException(Exception):
    """Custom exception raised when file extraction gets trapped in an infinite loop."""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Processing timed out (stuck in a heavy CPU loop).")

# Register the native Linux alarm signal handler
signal.signal(signal.SIGALRM, timeout_handler)


def index_single_subject(subject_name: str, source_folder: str):
    """Indexes any combination of PDF, TXT, DOCX, XLSX, or PPTX files into an isolated shard with progress tracking."""
    target_db_dir = get_db_path(subject_name)

    print(f"\n🚀 Initializing Ingestion for Subject: [{subject_name}]")
    print(f" Scanning Sub-folder: {source_folder}")

    # Universal file-type routing using open-source parsers
    loaders = {
	".txt": DirectoryLoader(source_folder, glob="**/*.txt", recursive=True, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8", "autodetect_encoding": False}),
	".pdf": DirectoryLoader(source_folder, glob="**/*.pdf", recursive=True, loader_cls=PyPDFLoader, silent_errors=True),
        ".docx": DirectoryLoader(source_folder, glob="**/*.docx", recursive=True, loader_cls=UnstructuredFileLoader, silent_errors=True),
        ".xlsx": DirectoryLoader(source_folder, glob="**/*.xlsx", recursive=True, loader_cls=UnstructuredFileLoader, silent_errors=True),
        ".pptx": DirectoryLoader(source_folder, glob="**/*.pptx", recursive=True, loader_cls=UnstructuredFileLoader, silent_errors=True)
    }

    documents = []

    # Loop over all extension rules, parsing with defensive error and timeout isolation
    
    for ext, loader in loaders.items():
        print(f"    Scanning and parsing matches for extension: {ext}")
        
        # Find all files matching this extension inside the source folder
        source_path = pathlib.Path(source_folder)
        file_list = list(source_path.glob(f"**/*{ext}"))
        
        for target_file in file_list:
            # Skip Windows temp/lock files safely
            if target_file.name.startswith("~$"):
                continue
                
            try:
                # Set a strict 15-second time limit per INDIVIDUAL file
                signal.alarm(15)

                if ext == ".pdf":
                    single_loader = PyPDFLoader(str(target_file), extract_images=False)
                elif ext == ".txt":
                    # This forces Python to skip bad bytes instead of throwing a mime_type exception
                    single_loader = TextLoader(str(target_file), encoding="utf-8", errors="ignore")
                else:
                    single_loader = loader.loader_cls(str(target_file), **loader.loader_kwargs)	




                loaded_docs = single_loader.load() 
                signal.alarm(0)
                
                if loaded_docs:
                    documents.extend(loaded_docs)
                    
            except TimeoutException:
                print(f"    🚨 SKIPPED (TIMED OUT): Parsing {target_file.name} took too long.")
                signal.alarm(0)
                continue
            except Exception as e:
                print(f"      SKIPPED (ERROR): Extraction error on {target_file.name}: {e}")
                signal.alarm(0)
                continue

    if not documents:
        print(f"  [~] No supported enterprise files found in '{subject_name}'. Skipping.")
        return

    print(f"  Total Ingested Chunks: {len(documents)}. Setting up mathematical embeddings...")

    embeddings = OllamaEmbeddings(
        model=APP_CONFIG["embedding_model"],
        base_url=APP_CONFIG["base_url"]
    )

    # --- ADVANCED TRACKING FOR VECTOR CALCULATIONS ---
    print(f"📥 Generating vectors and saving to database...")

    vector_db = None
    batch_size = 50  # Process chunks in small batches to keep the progress bar moving smoothly

    for i in tqdm(range(0, len(documents), batch_size), desc="Calculating Embeddings", unit="batch"):
        batch_docs = documents[i:i + batch_size]
        if vector_db is None:
            vector_db = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                persist_directory=target_db_dir
            )
        else:
            vector_db.add_documents(batch_docs)

    print(f"  [✓] Success: Subject '{subject_name}' completely built in '{target_db_dir}'\n")


def auto_discover_and_index(root_source_dir: str):
    if not os.path.exists(root_source_dir):
        print(f"[!] Error: Root source folder '{root_source_dir}' does not exist.")
        return

    # --- PHASE 1: DEEP ZIP SEARCH & UNPACK ---
    for root, dirs, files in os.walk(root_source_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                extracted_folder_name = file[:-4]
                extraction_target = os.path.join(root, extracted_folder_name)

                print(f"[⚡] Found zip archive: {zip_path}")
                print(f"    Extracting directly to: {extraction_target}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extraction_target)

                    # Delete the zip file after successful extraction to prevent recursive loops
                    os.remove(zip_path)
                except Exception as e:
                    print(f"    [!] Failed to extract zip file: {e}")

    # --- PHASE 2: PROCEED WITH OMNICHANNEL INGESTION ---
    print(f"\nStarting Omnichannel Ingestion across: {root_source_dir}\n" + "="*60)

    # Scan the top-level items in source_docs to build our subject shards
    for item in os.listdir(root_source_dir):
        item_path = os.path.join(root_source_dir, item)
        if os.path.isdir(item_path):
            index_single_subject(subject_name=item, source_folder=item_path)

    print("="*60 + "\nOmnichannel ingestion complete.")

if __name__ == "__main__":
    auto_discover_and_index("./source_docs")
