import logging
import os
import re
import json


class OneLineFormatter(logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style="%", max_len: int | None = None):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.max_len = max_len
        self._ws_re = re.compile(r"\s+")

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        msg = self._ws_re.sub(" ", msg).strip()
        if self.max_len and len(msg) > self.max_len:
            msg = msg[: self.max_len] + " …(truncated)"
        return msg


def compact_json(data) -> str:
    try:
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        return str(data)


def setup_logging() -> None:
    level_name = os.getenv("CAPTIONER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_len_env = os.getenv("CAPTIONER_LOG_MAX_LEN", "0")
    try:
        max_len = int(max_len_env) if max_len_env else 0
    except ValueError:
        max_len = 0

    root = logging.getLogger()
    # If already configured, don't add duplicate handlers
    if root.handlers:
        root.setLevel(level)
        return

    root.setLevel(level)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s [%(name)s] - %(message)s"
    handler.setFormatter(OneLineFormatter(fmt=fmt, datefmt="%H:%M:%S", max_len=max_len or None))
    root.addHandler(handler)

    # Silence noisy third-party libraries unless explicitly set
    for noisy in ("transformers", "urllib3", "PIL", "torch", "matplotlib"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, level))
