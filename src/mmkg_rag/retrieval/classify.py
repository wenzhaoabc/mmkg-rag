"""
Preprocess the query
"""

import json
import logging
from pathlib import Path

from ..utils import llm, encode_image, image_base64_url

from .parser import parse_classify_response
from .prompts import PROMPTS

log = logging.getLogger("mgrag")


async def query_dismantle(
    query: str,
    images: list[str] | None = None,
    history: list | None = None,
    force_retrieval: bool = False,
):
    """
    Dismantle the query and generate solution by LLM

    Args:
        query (str): The query to dismantle
        image (str, optional): The image to include in the query. Defaults to None.
        history (list, optional): The history of the conversation. Defaults to None.

    Returns:
        dict: The classification and
            - If direct: the response
            - If retrieval: the keywords
    """
    if not query and not images:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                PROMPTS["CLASSIFY_SYSTEM"]
                if not force_retrieval
                else PROMPTS["EXTRACT_KEYWORDS"]
            ),
        }
    ]
    if history:
        messages.extend(history)

    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": PROMPTS["CLASSIFY"].format(query=query)}],
    }
    if images:
        for image in images:
            user_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64_url(image),
                    },
                }
            )

    messages.append(user_message)

    response = await llm.chat_msg_sync(messages)

    log.debug(f"Classification response: {response}")

    classification, keywords_answer = parse_classify_response(response)
    if classification == "direct":
        return {"classification": classification, "response": keywords_answer}

    assert classification == "retrieval"
    keywords = keywords_answer
    if not isinstance(keywords_answer, list):
        keywords = [keywords_answer]

    return {"classification": classification, "keywords": keywords}
