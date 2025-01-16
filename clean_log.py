# delete empty log files

import os
from pathlib import Path


def clean_log():
    path = Path(os.getcwd())
    files = os.listdir(path / "logs")
    for file in files:
        if file.endswith(".log"):
            if os.path.getsize(path / "logs" / file) == 0:
                os.remove(path / "logs" / file)


clean_log()
