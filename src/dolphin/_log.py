from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from dolphin._types import P, PathOrStr, T

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}
__all__ = ["log_runtime", "setup_logging"]


def setup_logging(
    *,
    logger_name: str = "dolphin",
    debug: bool = False,
    filename: PathOrStr | None = None,
):
    config_file = Path(__file__).parent / Path("log-config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)

    if logger_name not in config["loggers"]:
        config["loggers"][logger_name] = {"level": "INFO", "handlers": ["stderr"]}

    if debug:
        config["loggers"][logger_name]["level"] = "DEBUG"

    if filename:
        if "file" not in config["loggers"][logger_name]["handlers"]:
            config["loggers"][logger_name]["handlers"].append("file")
        config["handlers"]["file"]["filename"] = os.fspath(filename)
        Path(filename).parent.mkdir(exist_ok=True, parents=True)

    if "filename" not in config["handlers"]["file"]:
        # We never passed in a filename: don't log to a file
        config["handlers"].pop("file")

    logging.config.dictConfig(config)

    # Temp work around for tqdm on py312
    if sys.version_info.major == 3 and sys.version_info.minor == 12:
        os.environ["TQDM_DISABLE"] = "1"


def log_runtime(f: Callable[P, T]) -> Callable[P, T]:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """
    logger = logging.getLogger("dolphin")

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        t1 = time.time()

        result = f(*args, **kwargs)

        t2 = time.time()
        elapsed_seconds = t2 - t1
        elapsed_minutes = elapsed_seconds / 60.0
        time_string = (
            f"Total elapsed time for {f.__module__}.{f.__name__} : "
            f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.2f} seconds)"
        )

        logger.info(time_string)

        return result

    return wrapper


class JSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        always_fields = {
            "timestamp": (
                datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
            ),
            "message": record.getMessage(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: (
                msg_val
                if (msg_val := always_fields.pop(val, None)) is not None
                else getattr(record, val)
            )
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message
