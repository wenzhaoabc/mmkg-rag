"""
Parse the LLM output
"""

import re
import json
import logging

from ..types import Entity, Relation

log = logging.getLogger("mgrag")


def _parse_string_list(text: str) -> list[str]:
    """Parse a string representation of list[str] with trailing commas support"""

    def clean_item(item: str) -> str:
        # Remove leading/trailing quotes and whitespace
        item = item.strip().strip("\"'")
        # # Unescape special characters
        item = item.replace('\\"', '"')
        item = item.replace("\\'", "'")
        item = item.replace("\\\\", "\\")
        return item

    # Validate basic format [...]
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("Input must be enclosed in square brackets")

    # Remove brackets
    content = text[1:-1].strip()
    if not content:
        return []

    # Split items handling escaped quotes
    pattern = r'(?:[^,"\\]|\\.)+|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''
    items = re.findall(pattern, content)

    # Clean and return
    return [clean_item(item) for item in items if item.strip(", ")]


def _parse_entity(text: str) -> list["Entity"]:
    """
    Parse entity from text
    """
    entities = []
    # Pattern to match <ENTITY, LABEL, "DESCRIPTION", [REFERENCES]>
    regex = r"<\s*([^,]*)\s*,\s*([^,]*)\s*,\s*\"(.*)\"\s*,\s*(\[.*\])\s*>"
    matches = re.finditer(regex, text, re.MULTILINE)
    for _, match in enumerate(matches, start=1):
        references = match.group(4)
        try:
            references = _parse_string_list(references)
        except json.JSONDecodeError as e:
            log.warning(
                f"Invalid JSON format for references in entity: {match.group(4)}, error: {e}"
            )
            continue
        e = Entity(
            name=match.group(1).strip(),
            label=match.group(2).strip(),
            description=match.group(3).strip(),
            references=references,
        )
        entities.append(e)

    return entities


def _parse_relation(text: str) -> list["Relation"]:
    """
    Parse relation from text
    """
    relations = []
    # Pattern to match <SOURCE, TARGET, LABEL, "DESCRIPTION", [REFERENCES]>
    regex = r"<\s*([^,]*)\s*,\s*([^,]*)\s*,\s*([^,]*)\s*,\s*\"(.*)\"\s*,\s*(\[.*\])\s*>"
    matches = re.finditer(regex, text, re.MULTILINE)
    for _, match in enumerate(matches, start=1):
        references = match.group(5)
        try:
            references = _parse_string_list(references)
        except json.JSONDecodeError as e:
            log.warning(
                f"Invalid JSON format for references in relation: {match.group(5)}, error: {e}"
            )
            continue
        r = Relation(
            source=match.group(1).strip(),
            label=match.group(2).strip(),
            target=match.group(3).strip(),
            description=match.group(4).strip(),
            references=references,
        )
        relations.append(r)

    return relations


def parse_er(rawtext: str) -> tuple[list["Entity"], list["Relation"]]:
    """
    Parse rawtext to entities and relationships
    """
    # es = _parse_entity(rawtext)
    # rs = _parse_relation(rawtext)
    # name, label, description, aliases, references
    json_es = _parse_json_object(
        rawtext, ["name", "label", "description", "aliases", "references"]
    )
    # source, label, target, description, references
    json_rs = _parse_json_object(
        rawtext, ["source", "label", "target", "description", "references"]
    )

    es = [
        Entity(
            name=e.get("name", ""),
            label=e.get("label", ""),
            description=e.get("description", ""),
            aliases=e.get("aliases", []),
            references=e.get("references", []),
        )
        for e in json_es
    ]
    rs = [
        Relation(
            source=r.get("source", ""),
            label=r.get("label", ""),
            target=r.get("target", ""),
            description=r.get("description", ""),
            references=r.get("references", []),
        )
        for r in json_rs
    ]
    return es, rs


def parse_alias(text: str) -> list[tuple[str, list[str]]]:
    """
    Parse alias from raw text

    Format: <ENTITY, ALIAS>
    Returns:
       list: list of tuples (entity, aliases)
    where aliases is a list of strings
    """
    aliases = []
    # Pattern to match <entity, alias_list>
    # Handles quoted/unquoted entities and JSON list of aliases
    alias_pattern = r'<\s*(?:"([^"]+)"|([^,]+))\s*,\s*(\[[^\]]*\])\s*>'

    matches = re.finditer(alias_pattern, text, re.MULTILINE)

    for match in matches:
        # Entity name (quoted or unquoted)
        entity = (match.group(1) or match.group(2) or "").strip()
        # Alias list as string
        alias_str = match.group(3).strip()

        try:
            alias_list = _parse_string_list(alias_str)
            if entity and isinstance(alias_list, list):
                aliases.append((entity, alias_list))
            else:
                log.warning(f"Invalid alias format for entity: {entity}")
        except (json.JSONDecodeError, TypeError) as e:
            log.warning(
                f"Invalid alias JSON for entity {entity}: {alias_str}, error: {e}"
            )
            continue

    return aliases


def parse_merged_e(text: str) -> tuple[bool, Entity | None]:
    """
    Parse merged entities from raw text, return if merged and merged entity
    """
    regex_pattern = r"\{.*\}"
    matches = re.finditer(regex_pattern, text, re.DOTALL)
    if not matches:
        log.warning("No JSON object found in text")
        return False, None
    for m_num, m in enumerate(matches, start=1):
        try:
            data = json.loads(m.group(0))
            if "same_entity" in data:
                if data["same_entity"]:
                    single_e: dict = data["entity"]
                    return True, Entity(
                        name=single_e.get("name", ""),
                        label=single_e.get("label", ""),
                        description=single_e.get("description", ""),
                        aliases=single_e.get("aliases", []),
                        references=single_e.get("references", []),
                    )
                else:
                    return False, None
            else:
                log.warning("No 'same_entity' key found in JSON")
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse JSON: {e}")
            continue
        except Exception as e:
            log.warning(f"Unexpected error parsing merged entities: {e}")
            continue

    return False, None


def parse_merged_r(text: str) -> tuple[bool, list["Relation"]]:
    """
    Parse merged relations from raw text, if merged and merged relations
    """
    relations: list["Relation"] = []
    # Pattern to match <ENTITY_1, RELATIONSHIP, ENTITY_2, DESCRIPTION, REFERENCES>
    regex_pattern = r"\{.*\}"
    matches = re.finditer(regex_pattern, text, re.DOTALL)
    if not matches:
        log.warning("No JSON object found in text")
        return False, []
    for m_num, m in enumerate(matches, start=1):
        try:
            data = json.loads(m.group(0))
            if "same_relationship" in data:
                if data["same_relationship"] and "relationship" in data:
                    single_r: dict = data["relationship"]
                    relations.append(
                        Relation(
                            source=single_r.get("source", ""),
                            label=single_r.get("label", ""),
                            target=single_r.get("target", ""),
                            description=single_r.get("description", ""),
                            references=single_r.get("references", []),
                        )
                    )
                else:
                    return False, []
            else:
                log.warning("No 'same_relation' key found in JSON")
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse JSON: {e}")
            continue
        except Exception as e:
            log.warning(f"Unexpected error parsing merged relations: {e}")
    return True, relations


def parse_image_description(text: str) -> tuple[str, list, str]:
    """
    Parse image caption, text_snippets, description from text

    Args:
        text (str): The text to parse, expected to be in JSON format

    Returns:
        tuple: (caption, text_snippets, description)
    """
    try:
        # Find JSON content - look for content between curly braces
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            log.warning("No JSON content found in text")
            return "", [], ""

        # Parse JSON
        data = json.loads(json_match.group(0))

        # Extract fields with defaults if missing
        caption = data.get("caption", "")
        text_snippets = data.get("text_snippets", [])
        description = data.get("description", "")

        return caption, text_snippets, description

    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse JSON: {e}")
        return "", [], ""
    except Exception as e:
        log.warning(f"Unexpected error parsing image description: {e}")
        return "", [], ""


def parse_json_list(text: str, fields: list[str] | None = None) -> list:
    """
    Parse a JSON list from text

    Args:
        text (str): The text to parse

    Returns:
        list: The parsed list
    """
    try:
        json_match_re = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match_re:
            log.warning("No JSON list found in text")
            return []

        json_list = json_match_re.group(0)
        data = json.loads(json_list)
        if fields:
            return [{field: item.get(field, None) for field in fields} for item in data]
        return data
    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse JSON list: {e}")
        return []
    except Exception as e:
        log.warning(f"Unexpected error parsing JSON list: {e}")
        return []


def _parse_json_object(text: str, fields: list[str]) -> list[dict]:
    """
    Parse a JSON object from text with regex

    Args:
        text (str): The text to parse

    Returns:
        dict: The parsed object
    """
    results = []
    regex_patten = r"\{(?:[^{}])*\}"
    matches = re.finditer(regex_patten, text, re.DOTALL | re.MULTILINE)
    if not matches:
        log.warning("No JSON object found in text")
        return []
    for m_num, m in enumerate(matches, start=1):
        try:
            data = json.loads(m.group(0))
            result_dict = {}
            all_field = True
            for field in fields:
                if data.get(field, None) != None:
                    result_dict[field] = data.get(field)
                else:
                    all_field = False
                    break
            if not all_field:
                continue
            results.append(result_dict)
        except json.JSONDecodeError as e:
            log.debug(f"Failed to parse JSON object: {e},\ntext: \n{m.group(0)}")
            continue
        except Exception as e:
            log.warning(f"Unexpected error parsing JSON object: {e}")
            continue
    return results
