# logging_config.py
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

def init_logging(level=logging.INFO):
    root = logging.getLogger()
    if root.handlers:
        # already configured
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    root.setLevel(level)
    root.addHandler(sh)
    root.addHandler(fh)