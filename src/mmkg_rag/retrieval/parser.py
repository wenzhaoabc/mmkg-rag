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


def parse_agent_defines(text: str) -> list[dict]:
    """
    Parse the agent defines from the text

    Args:
        text (str): The text to parse

    Returns:
        list[dict]: The agent defines
    """
    json_pattern = r"\{.*\}"
    try:
        json_str = re.search(json_pattern, text, re.DOTALL)
        if not json_str:
            log.error(f"Failed to parse agent defines: No JSON found, text: {text}")
            return []

        json_str = json_str.group()
        response_dict = json.loads(json_str)
        log.debug(f"Agent defines:\n{text}\nresponse_dict:\n{response_dict.__str__()}")
        return response_dict["agents"]
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse agent defines: JSONDecodeError: {e}")
        return []
    except Exception as e:
        log.error(f"Failed to parse agent defines: {e}")
