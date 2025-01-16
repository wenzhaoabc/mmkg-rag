import logging
import json
import re
from pathlib import Path

log = logging.getLogger("mgrag")


def parse_classify_response(response: str) -> tuple[str, str | list[str]]:
    """
    Parse the classify response

    Args:
        response (str): The response to parse

    Returns:
        tuple[str, str | list[str]]: The classification and the extracted keywords
    """
    json_pattern = r"\{.*\}"
    try:
        json_str = re.search(json_pattern, response, re.DOTALL)
        if not json_str:
            log.error(
                f"Failed to parse classify response: No JSON found, response: {response}"
            )
            return "retrieval", []

        json_str = json_str.group()
        response_dict = json.loads(json_str)
        log.debug(
            f"Classification response:\n{response}\nresponse_dict:\n{response_dict.__str__()}"
        )
        classification = response_dict.get("classification", "retrieval")
        if classification == "direct":
            return "direct", response_dict["response"]
        elif classification == "retrieval":
            return "retrieval", response_dict.get("keywords", [])
    except Exception as e:
        log.error(f"Failed to parse classify response: {e}")
    # Return empty list if parsing fails
    return "retrieval", []
