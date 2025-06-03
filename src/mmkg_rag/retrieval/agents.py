# Retrieval ansers by Agents

from ..utils import llm, image_base64_url

from .classify import query_dismantle
from .generate import generate_answer
from .prompts import PROMPTS
from .parser import parse_agent_defines


async def _question_decomposition(
    question: str,
    images: list[str],
) -> list[str]:
    """
    Decomposes a question into sub-questions using the agent.
    """

    images_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": image_base64_url(image),
            },
        }
        for image in images
    ]

    messages = [
        {
            "role": "system",
            "content": PROMPTS["QUERY_DECOMPOSITION"],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please design a few agents to answer the question."
                    + question,
                },
            ]
            + images_content,
        },
    ]

    res = await llm.chat_msg_sync(messages)
    agents = parse_agent_defines(res)
    if not agents:
        raise ValueError(
            "Failed to parse agent defines, please check the response from the agent."
        )
    return agents


async def _agent_generate(
    task: str,
    suggestions: list[str] | None,
) -> str:
    """
    Generates a response for the given task and suggestions using the agent.

    Args:
        task (str): The task to generate a response for.
        suggestions (list[str]): The suggestions for the agent.

    Returns:
        str: The generated response.
    """
    question_type = await query_dismantle(task)
    if question_type["classification"] == "direct":
        return question_type["response"]
    keywords = question_type["keywords"]
    answer = await generate_answer(keywords, task)
    return answer


async def agent_response(
    question: str,
    images: list[str],
) -> str:
    """
    Generates a response for the given question and images using the agent.

    Args:
        question (str): The question to generate a response for.
        images (list[str]): The images to include in the response.

    Returns:
        str: The generated response.
    """
    agents = await _question_decomposition(question, images)
    if not agents:
        raise ValueError(
            "Failed to parse agent defines, please check the response from the agent."
        )

    responses = []
    for agent in agents:
        task = agent["task"]
        suggestions = agent.get("suggestions", [])
        response = await _agent_generate(task, suggestions)
        responses.append(response)

    messages = [
        {"role": "system", "content": PROMPTS["FUSION"]},
        {
            "role": "user",
            "content": f"Question: {question}\nAnswers: \n" + "\n".join(response),
        },
    ]

    final_res = await llm.chat_msg_sync(messages=messages)

    return final_res
