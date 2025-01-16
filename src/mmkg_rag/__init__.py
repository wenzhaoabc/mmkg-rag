from pathlib import Path
from dotenv import load_dotenv
from .utils.log import setup_logging

ROOT_DIR = Path(__file__).parent.parent.parent

if Path(ROOT_DIR / ".env").exists():
    load_dotenv(ROOT_DIR / ".env")

setup_logging()
