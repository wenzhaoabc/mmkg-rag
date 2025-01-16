"""
This module contains functions for extracting entities and relationships from text
"""

import logging

from ..utils import llm
from ..types import Chunk, Entity, Relation

from .parser import parse_er, parse_alias
from .prompts import PROMPTS

log = logging.getLogger("mgrag")


async def find_alias(
    chunk: "Chunk",
    entities: list["Entity"],
    relations: list["Relation"],
) -> tuple[list["Entity"], list["Relation"]]:
    """
    Find aliases for entities and update relations accordingly
    """
    entities_str = "\n".join([f"- <{e.name}>" for e in entities])
    res = await llm.chat(
        PROMPTS["ALIAS"].format(chunk=chunk.text, entities=entities_str),
        system_prompt=PROMPTS["ALIAS_SYSTEM"],
    )
    try:
        aliases = parse_alias(res)
    except Exception as e:
        log.error(f"Error parsing response with chunk {chunk.id}: {e}")
        aliases = []

    # Store old_name -> new_name mapping
    name_mapping = {}

    # Update entities
    for entity_name, alias_list in aliases:
        entity = next((e for e in entities if e.name == entity_name), None)
        if entity:
            all_names = [entity_name] + alias_list
            full_name = max(all_names, key=len)
            other_aliases = [name for name in all_names if name != full_name]

            # Store mapping before updating
            name_mapping[entity.name] = full_name

            # Update entity
            entity.name = full_name
            entity.aliases = other_aliases

    # Update relations
    for relation in relations:
        if relation.source in name_mapping:
            relation.source = name_mapping[relation.source]
        if relation.target in name_mapping:
            relation.target = name_mapping[relation.target]

    return entities, relations


async def extract_er_from_chunk(
    chunk: "Chunk",
    loop: int = 1,
    entity_labels: list[str] = [],
    relation_labels: list[str] = [],
) -> tuple[list["Entity"], list["Relation"]]:
    """
    Extract entities and relationships from a chunk of text

    Args:
        chunk (Chunk): A chunk of text
        loop (int, optional): The number of times to loop over the text. Defaults to 1.
    """
    res_1 = await llm.chat(
        PROMPTS["INDEX"].format(chunk=chunk.text),
        system_prompt=PROMPTS["SYSTEM"].format(
            entity_labels=", ".join(entity_labels),
            relationship_labels=", ".join(relation_labels),
        ),
    )
    try:
        # parse the response to extract entities and relationships
        entities, relations = parse_er(res_1)
    except Exception as e:
        log.error(f"Error parsing response with chunk {chunk.id}: {e}")
        entities, relations = [], []

    log.debug(
        f"Chunk {chunk.id}, llm response: \n{res_1}\n {len(entities)} entities and {len(relations)} relations\n"
    )
    history_messages = [
        {"role": "user", "content": PROMPTS["INDEX"].format(chunk=chunk.text)},
        {"role": "assistant", "content": res_1},
    ]
    if loop > 1:
        for i in range(loop - 1):
            res_loop = await llm.chat(
                PROMPTS["LOOP"],
                system_prompt=PROMPTS["SYSTEM"],
                history_messages=history_messages,
            )
            try:
                entities_loop, relations_loop = parse_er(res_loop)
                if not entities_loop and not relations_loop:
                    log.warning(
                        f"No entities or relations found in chunk {chunk.id} loop {i}"
                    )
            except Exception as e:
                log.error(f"Error parsing response with chunk {chunk.id} loop {i}: {e}")
                entities_loop, relations_loop = [], []

            entities.extend(entities_loop)
            relations.extend(relations_loop)
            # update history messages
            history_messages.extend(
                [
                    {"role": "user", "content": PROMPTS["LOOP"]},
                    {"role": "assistant", "content": res_loop},
                ]
            )
            # ask llm for continuation
            res_con = await llm.chat(
                PROMPTS["IF_CONTINUE"],
                system_prompt=PROMPTS["SYSTEM"],
                history_messages=history_messages,
            )
            if "NO" in res_con.upper():
                break

    log.debug(
        f"Extract {len(entities)} entities and {len(relations)} relations from chunk {chunk.id}"
    )
    for entity in entities:
        entity.chunks = [chunk.id]
    for relation in relations:
        relation.chunks = [chunk.id]

    entities, relations = await find_alias(chunk, entities, relations)
    entities, relations = complete_reference(chunk, entities, relations)
    log.info(
        f"Extracted {len(entities)} entities and {len(relations)} relations from chunk {chunk.id}"
    )
    return entities, relations


def complete_reference(
    chunk: "Chunk",
    entities: list["Entity"],
    relations: list["Relation"],
) -> tuple[list["Entity"], list["Relation"]]:
    """
    Complete references for entities and relations by finding the complete text in chunk

    Args:
        chunk (Chunk): The chunk containing full text
        entities (list[Entity]): List of entities to update
        relations (list[Relation]): List of relations to update
    """

    def find_complete_text(ref: str, full_text: str) -> str:
        # Parse start and end portions from reference
        parts = ref.split("...")
        if len(parts) != 2:
            return ref

        start, end = parts[0].strip(), parts[1].strip()

        # Find all occurrences of text that start with start and end with end
        matches = []
        for i in range(len(full_text)):
            if full_text[i:].startswith(start):
                for j in range(i + len(start), len(full_text)):
                    if full_text[j:].startswith(end):
                        complete = full_text[i : j + len(end)]
                        if start in complete and complete.endswith(end):
                            matches.append(complete)
                        break

        # Return the shortest matching text or original if no matches
        return min(matches, key=len) if matches else ref

    # Update entity references
    for entity in entities:
        if entity.references:
            entity.references = [
                find_complete_text(ref, chunk.text) for ref in entity.references
            ]

    # Update relation references
    for relation in relations:
        if relation.references:
            relation.references = [
                find_complete_text(ref, chunk.text) for ref in relation.references
            ]

    return entities, relations
