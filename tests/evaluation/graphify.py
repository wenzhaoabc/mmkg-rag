"""
Construct knowledge graph from files.
"""

import argparse
import os
import asyncio
import dotenv

from src.mmkg_rag.index.pipe import process_files


async def main():
    await process_files(args.input, args.output)


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Construct knowledge graph from files."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        default=["graphrag.pdf"],
        help="Input directory containing the files to process.",
    )
    
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/knowledge_graph",
        help="Output directory to save the processed files.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    asyncio.run(main())
