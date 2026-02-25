import os
from pathlib import Path
from dotenv import load_dotenv

# env_path = r"C:\Users\User\Documents\GitHub\CXR-ChangeDetection-Backend\fastapi-ai\app\.env"
CURRENT_DIR = Path(__file__).resolve().parent 
ENV_PATH = CURRENT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

class Settings:
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: int = int(os.getenv("DB_PORT", 3306))
    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_NAME: str = os.getenv("DB_NAME")

settings = Settings()