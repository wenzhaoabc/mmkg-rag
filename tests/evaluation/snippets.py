import argparse
import os
import json

import dotenv

from src.mmkg_rag.storage import MemoryStorage


def save_to_neo4j():
    """
    python.exe -m tests.evaluation.snippets -i databases/pike_rag
    """
    storage = MemoryStorage(args.input)

    storage.save_to_neo4j(
        os.environ.get("NEO4J_URI"),
        os.environ.get("NEO4J_USERNAME"),
        os.environ.get("NEO4J_PASSWORD"),
    )


def wordscloud():
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    s_text = ""
    m_text = ""
    for q in data:
        for m in q["metrics"]:
            if m["winner"] == "s":
                s_text += m["explanation"]
            elif m["winner"] == "m":
                m_text += m["explanation"]

    with open("s_text.txt", "w", encoding="utf-8") as f:
        f.write(s_text)

    with open("m_text.txt", "w", encoding="utf-8") as f:
        f.write(m_text)


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Save to Neo4j")
    parser.add_argument("-i", "--input", type=str, default="./data")
    args = parser.parse_args()

    # save_to_neo4j()
    wordscloud()
