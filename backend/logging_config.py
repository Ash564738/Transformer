# logging_config.py
from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def init_logging(level=logging.INFO):
    """Configure console + file logging exactly once, regardless of prior handlers."""
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    # Console handler.
    console = None
    for handler in root.handlers:
        if getattr(handler, "_dga_console", False):
            console = handler
            break
    if console is None:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        console._dga_console = True
        root.addHandler(console)
    else:
        console.setLevel(level)
        console.setFormatter(formatter)

    # File handler.
    file_handler = None
    resolved_log = LOG_FILE.resolve()
    for handler in root.handlers:
        if getattr(handler, "_dga_file", False):
            file_handler = handler
            break
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == resolved_log:
                    file_handler = handler
                    file_handler._dga_file = True
                    break
            except Exception:
                pass

    if file_handler is None:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler._dga_file = True
        root.addHandler(file_handler)

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Make third-party logs visible but not excessively verbose.
    logging.getLogger("snorkel").setLevel(logging.INFO)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)
    logging.getLogger("catboost").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    root.info("Logging initialized | file=%s", LOG_FILE)
    return LOG_FILE