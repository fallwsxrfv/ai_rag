import os

# Centralized System Settings
APP_CONFIG = {
    "default_model": "qwen2.5-coder:32b-rag",
    "embedding_model": "nomic-embed-text",
    "base_url": "http://localhost:11434",
    "db_root_dir": "./db"
}

def get_db_path(subject_name: str) -> str:
    """
    Dynamically constructs a clean storage directory for a specific subject
    using the locked-in embedding model name as a version safety filter.
    """
    # Clean the model name string to use as a safe folder prefix
    model_safe_name = APP_CONFIG["embedding_model"].replace(":", "-").replace("/", "-")
    
    # Example output: ./db/nomic-embed-text/subject_alpha
    return os.path.join(APP_CONFIG["db_root_dir"], model_safe_name, subject_name)
