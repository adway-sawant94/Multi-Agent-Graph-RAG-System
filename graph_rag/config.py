import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

# Fallback config
USE_MOCK_FALLBACK = os.getenv("USE_MOCK_FALLBACK", "False").lower() in ("true", "1", "yes")

# Database Paths
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "indices"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Graph configs
RELATION_FREE_THRESHOLD = 50  # Max degree before relation-free flattening is triggered for speed
