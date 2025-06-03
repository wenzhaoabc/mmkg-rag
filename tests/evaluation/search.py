"""
retrieve the ers and generate answers.
"""

import argparse
import json

from typing import Literal
import asyncio
from src.mmkg_rag.retrieval.classify import query_dismantle
from src.mmkg_rag.retrieval.generate import generate_answer
from src.mmkg_rag.retrieval.agents import agent_response
from src.mmkg_rag.retrieval.search import load_default_ers


async def s_search(q: str) -> str:
    classify = await query_dismantle(q)
    if classify["classification"] == "direct":
        return classify["response"]
    keywords = classify["keywords"]
    answer = await generate_answer(keywords, q)
    return answer["response"]


async def m_search(q: str) -> str:
    res = await agent_response(q, images=[])
    return res


async def batch_process(
    questions: list[dict], method: Literal["s", "m"], batch_size: int = 10
) -> list[dict]:
    results = []
    "batch process the questions and get res in order"
    search_method = s_search if "s" == method else m_search
    tasks = [search_method(question["question"]) for question in questions]

    results = await asyncio.gather(*tasks)

    return results


async def main():
    with open(args.questions, "r") as f:
        questions = json.load(f)
    batches = [
        questions[i : min(i + args.batch, len(questions))]
        for i in range(0, len(questions), args.batch)
    ]

    for i, batch in enumerate(batches):
        if "s" in args.method:
            try:
                s_result = await batch_process(batch, method="s", batch_size=args.batch)
                for j, question in enumerate(batch):
                    question["s_answer"] = s_result[j]
            except Exception as e:
                print(f"Error in batch {i+1} for method s: {e}")
                continue
        if "m" in args.method:
            try:
                m_result = await batch_process(batch, method="m", batch_size=args.batch)
                for j, question in enumerate(batch):
                    question["m_answer"] = m_result[j]
            except Exception as e:
                print(f"Error in batch {i+1} for method m: {e}")
                continue

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(questions, f)
        print(f"Batch {i+1}/{len(batches)} processed. Results saved to results.json")

        # break  # for testing, remove this line to process all batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute search")
    parser.add_argument("-q", "--questions", type=str, default="./questions.json")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="sm",
        choices=["s", "m", "sm"],
        help="Method to use for search, s for direct search, m for multi agents search",
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=10, help="Batch size for question processing"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="default",
        help="Input ERs to use, default for default ERs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results.json",
        help="Output file for the results",
    )
    args = parser.parse_args()

    load_default_ers(args.input)

    asyncio.run(main())
    print("Execution completed.")

"""
python.exe -m tests.evaluation.search -q tests\evaluation\questions.json -i databases/pike_rag -o tests\evaluation\search_res.json
"""
