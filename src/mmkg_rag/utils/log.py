import sys
import logging
from datetime import datetime


def setup_logging():
    log = logging.getLogger("mgrag")
    log.setLevel(logging.DEBUG)
    log_file_ts = datetime.now().strftime("%y%m%d_%H%M%S")

    file_handler_debug = logging.FileHandler(
        filename=f"logs/{log_file_ts}-debug.log", encoding="utf-8"
    )
    file_handler_debug.setLevel(logging.DEBUG)

    file_handler_info = logging.FileHandler(
        filename=f"logs/{log_file_ts}-info.log", encoding="utf-8"
    )
    file_handler_info.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(filename)s:%(lineno)-3d - %(message)s"
    )
    file_handler_debug.setFormatter(formatter)
    file_handler_info.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    log.addHandler(file_handler_debug)
    log.addHandler(file_handler_info)
    log.addHandler(stream_handler)
