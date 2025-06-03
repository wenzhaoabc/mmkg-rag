"""
> Original Text in paper

To evaluate the effectiveness of RAG systems for more global sensemaking tasks, we need questions that convey only a high-level understanding of dataset contents, and not the details of specific texts. We used an activity-centered approach to automate the generation of such questions: given a short description of a dataset, we asked the LLM to identify N potential users and N tasks per user, then for each (user, task) combination, we asked the LLM to generate N questions that require understanding of the entire corpus. For our evaluation, a value of N = 5 resulted in 125 test questions per dataset. Table 1 shows example questions for each of the two evaluation datasets.
"""

"""
workflow:

1. Given a short description of a dataset
2. Identify N potential users and N tasks per user
3. For each (user, task) combination, generate N questions that require understanding of the entire corpus
4. A value of N = 5 resulted in 125 test questions per dataset

example:
Dateset: Behind the Tech is a collection of transcribed podcast conversations where Microsoft CTO Kevin Scott engages with technology leaders in deep discussions. These dialogues explore a wide range of technical topics, capturing industry leaders' insights on technological innovations, development challenges, and the societal impact of technology. Through these conversations, readers can gain valuable perspectives on tech trends, learn from innovation experiences, and understand various visions for the future of technology.
User: A tech journalist looking for insights and trends in the tech industry
Task: Understanding how tech leaders view the role of policy and regulation
Questions:
1. Which episodes deal primarily with tech policy and government regulation?
2. How do guests perceive the impact of privacy laws on technology development?
3. Do any guests discuss the balance between innovation and ethical considerations?
4. What are the suggested changes to current policies mentioned by the guests?
5. Are collaborations between tech companies and governments discussed and how?

store the questions in a json file with user,task and questions key
"""

GENERATE_PERSONA_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
The text document contains a description of a corpus. You should identify potential personas who might be interested in the corpus content.

example:
Dataset: A collection of transcribed podcast conversations where Microsoft CTO Kevin Scott engages with technology leaders in deep discussions.

Potential Personas:
1. A tech journalist looking for insights and trends in the tech industry
2. A software engineer interested in learning about the latest technologies
3. A student studying the societal impact of technology
4. A startup founder seeking inspiration from industry leaders
5. A policy analyst interested in the intersection of technology and government
"""

GENERATE_TASK_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
The text document contains a description of a corpus and a persona. You should identify specific tasks that the persona might want to accomplish with the corpus.

example:
Dataset: A collection of transcribed podcast conversations where Microsoft CTO Kevin Scott engages with technology leaders in deep discussions.

Persona: A tech journalist looking for insights and trends in the tech industry

Tasks:
1. Understanding how tech leaders view the role of policy and regulation
2. Analyzing the impact of emerging technologies on the tech industry
3. Identifying the key challenges faced by technology leaders
4. Exploring the future trends in technology innovation
5. Investigating the ethical implications of tech advancements
"""

GENERATE_QUESTION_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
The text document contains a description of a corpus, a persona, and a task. You should generate questions that require understanding of the entire corpus.

example:
Dataset: A collection of transcribed podcast conversations where Microsoft CTO Kevin Scott engages with technology leaders in deep discussions.

Persona: A tech journalist looking for insights and trends in the tech industry
Task: Understanding how tech leaders view the role of policy and regulation

Questions:
1. Which episodes deal primarily with tech policy and government regulation?
2. How do guests perceive the impact of privacy laws on technology development?
3. Do any guests discuss the balance between innovation and ethical considerations?
4. What are the suggested changes to current policies mentioned by the guests?
5. Are collaborations between tech companies and governments discussed and how?
"""

import argparse
import json
from openai import OpenAI
from typing import List, Dict
import os
import re
import dotenv


def setup_openai():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )
    return client


def generate_users(client: OpenAI, dataset_desc: str, n: int = 5) -> List[str]:
    response = client.chat.completions.create(
        model="qwen-plus-latest",
        messages=[
            {
                "role": "system",
                "content": GENERATE_PERSONA_PROMPT,
            },
            {
                "role": "user",
                "content": f"Given this corpus description:\n{dataset_desc}\n\nList {n} potential users who might be interested in this corpus content. Return only the list of users, one per line.",
            },
        ],
    )
    return response.choices[0].message.content.strip().split("\n")


def generate_tasks(
    client: OpenAI, dataset_desc: str, user: str, n: int = 5
) -> List[str]:
    response = client.chat.completions.create(
        model="qwen-plus-latest",
        messages=[
            {
                "role": "system",
                "content": GENERATE_TASK_PROMPT,
            },
            {
                "role": "user",
                "content": f"Given this corpus description:\n{dataset_desc}\n\nFor this user: {user}\n\nList {n} specific tasks they might want to accomplish with this corpus. Return only the list of tasks, one per line.",
            },
        ],
    )
    return response.choices[0].message.content.strip().split("\n")


def generate_questions(
    client: OpenAI, dataset_desc: str, user: str, task: str, n: int = 5
) -> List[str]:
    response = client.chat.completions.create(
        model="qwen-plus-latest",
        messages=[
            {
                "role": "system",
                "content": GENERATE_QUESTION_PROMPT,
            },
            {
                "role": "user",
                "content": f"Given this corpus description:\n{dataset_desc}\n\nFor this user: {user}\nAnd this task: {task}\n\nGenerate {n} questions that require understanding of the entire corpus. Return only the list of questions, one per line.",
            },
        ],
    )
    return response.choices[0].message.content.strip().split("\n")


def clean_numbered_string(s: str) -> str:
    """Remove number prefixes like '1.', '2.' etc from the start of a string"""
    return re.sub(r"^\d+\.\s*", "", s)


"""
微软的研究团队提出的一篇论文，题目是PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Generation。这篇论文提出了一种名为PIKE-RAG的新型框架，通过提取、理解并应用专业知识和推理逻辑来增强检索增强生成（RAG）系统的能力，以应对工业应用场景中的复杂需求。对原始文档进行知识原子化，将原始文档细分为多个独立的知识单位，构建了包含information source layer，corpus layer，distilled knowledge layer的三层图。根据任务的难度，将任务分为四个类型，事实性问题、可链接推理问题、预测性问题和创造性问题。定义了RAG系统的四个能力级别，每个级别对应解决不同类型问题的能力。L1系统专注于提供准确的事实性回答，L2系统扩展功能以应对需要多步推理的问题，L3系统能够进行合理的预测，而L4系统则可以提出有理有据的计划或解决方案。在开放域基准和法律领域特定基准上进行了实验。选择多种基线方法进行对比，包括 Zero-Shot CoT、Naive RAG、Self-Ask 等。PIKE-RAG框架通过有效提取、理解、组织专门知识和构建连贯的推理逻辑，显著提升了 RAG 系统在工业应用中的性能。特别是在处理复杂多跳查询时，所提出的知识原子化和基于知识的任务分解方法表现出色。
"""

"""
The Microsoft research team proposed a paper titled **"PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Generation"**, introducing a novel framework called **PIKE-RAG**. This framework enhances the capabilities of Retrieval-Augmented Generation (RAG) systems by extracting, understanding, and applying specialized knowledge and reasoning logic to address complex industrial application demands. 

The approach involves **knowledge atomization**, breaking down source documents into independent knowledge units and constructing a **three-layer graph** comprising an *information source layer*, *corpus layer*, and *distilled knowledge layer*. Tasks are categorized into four types based on difficulty: **factual questions**, **linkable reasoning questions**, **predictive questions**, and **creative questions**. The paper defines **four capability levels** for RAG systems, each corresponding to solving distinct problem types: 
- **L1 systems** focus on accurate factual responses, 
- **L2 systems** handle multi-step reasoning, 
- **L3 systems** perform predictive analysis, and 
- **L4 systems** generate well-founded plans/solutions.

Experiments were conducted on **open-domain benchmarks** and a **legal domain-specific benchmark**, comparing PIKE-RAG against baselines like *Zero-Shot CoT*, *Naive RAG*, and *Self-Ask*. Results show that PIKE-RAG significantly improves RAG performance in industrial settings by effectively organizing specialized knowledge and constructing coherent reasoning chains. Notably, its **knowledge atomization** and **knowledge-based task decomposition** excel in handling complex multi-hop queries.
"""


def main():
    dataset_desc = """The Microsoft research team proposed a paper titled **"PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Generation"**, introducing a novel framework called **PIKE-RAG**. This framework enhances the capabilities of Retrieval-Augmented Generation (RAG) systems by extracting, understanding, and applying specialized knowledge and reasoning logic to address complex industrial application demands. 

The approach involves **knowledge atomization**, breaking down source documents into independent knowledge units and constructing a **three-layer graph** comprising an *information source layer*, *corpus layer*, and *distilled knowledge layer*. Tasks are categorized into four types based on difficulty: **factual questions**, **linkable reasoning questions**, **predictive questions**, and **creative questions**. The paper defines **four capability levels** for RAG systems, each corresponding to solving distinct problem types: 
- **L1 systems** focus on accurate factual responses, 
- **L2 systems** handle multi-step reasoning, 
- **L3 systems** perform predictive analysis, and 
- **L4 systems** generate well-founded plans/solutions.

Experiments were conducted on **open-domain benchmarks** and a **legal domain-specific benchmark**, comparing PIKE-RAG against baselines like *Zero-Shot CoT*, *Naive RAG*, and *Self-Ask*. Results show that PIKE-RAG significantly improves RAG performance in industrial settings by effectively organizing specialized knowledge and constructing coherent reasoning chains. Notably, its **knowledge atomization** and **knowledge-based task decomposition** excel in handling complex multi-hop queries."""
    dataset_desc = args.desc if args.desc else dataset_desc

    client = setup_openai()
    N = args.n if args.n else 5
    results = []

    users = generate_users(client, dataset_desc, N)
    for user in users:
        tasks = generate_tasks(client, dataset_desc, user, N)
        for task in tasks:
            questions = generate_questions(client, dataset_desc, user, task, N)
            for q in questions:
                results.append(
                    {
                        "user": clean_numbered_string(user),
                        "task": clean_numbered_string(task),
                        "question": clean_numbered_string(q),
                    }
                )

    output_file = args.out if args.out else "questions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions for a dataset.")
    parser.add_argument(
        "--desc",
        type=str,
        default="",
        help="Description of the dataset to generate questions for.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of users, tasks, and questions to generate.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="questions.json",
        help="Output file to save the generated questions.",
    )
    args = parser.parse_args()
    dotenv.load_dotenv()
    main()
