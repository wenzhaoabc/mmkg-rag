import json
import logging
from collections import defaultdict
from functools import lru_cache

import asyncio

from rapidfuzz.fuzz import token_sort_ratio

from ..types import Entity, Relation
from ..utils import llm

from .parser import parse_merged_e, parse_merged_r
from .prompts import PROMPTS

log = logging.getLogger("mgrag")


async def deduplicate(
    entities: list[Entity], relations: list[Relation]
) -> tuple[list[Entity], list[Relation]]:
    """Deduplicate entities and relations using parallel processing"""

    # Process all groups concurrently
    entity_groups = group_by_name_alias_v2(entities, similarity=0.95)
    merged_results = await asyncio.gather(
        *[_merge_entity_group(g) for g in entity_groups]
    )
    new_entities = []
    # update relations with new entities
    for eg, (merged, ne) in zip(entity_groups, merged_results):
        if len(eg) == 1 or ne is None or not merged:
            new_entities.extend(eg)
            continue
        new_entities.append(ne)
        eg_names = [e.name for e in eg]
        for r in relations:
            if r.source in eg_names:
                r.source = ne.name
            if r.target in eg_names:
                r.target = ne.name

    # confirm every entity's aliases are not null
    for e in new_entities:
        if not e.aliases:
            e.aliases = []

    log.info(
        f"Deduplacated {len(entities)-len(new_entities)} entities in {len(entity_groups)} groups"
    )

    # Group relations by overlapping entities
    relation_groups = group_relations(relations)
    new_relations = await asyncio.gather(
        *[_merge_relation_group(g, new_entities) for g in relation_groups],
        return_exceptions=True,
    )

    # Filter out None results and return
    new_relations = [rs for rs in new_relations if isinstance(rs, list)]
    new_relations = [r for rs in new_relations for r in rs]
    log.info(
        f"Deduplacated {len(relations)-len(new_relations)} relations in {len(relation_groups)} groups"
    )
    return new_entities, new_relations


async def _merge_entity_group(entities: list[Entity]) -> tuple[bool, Entity | None]:
    """
    deduplicate similar entities
    return only one entity if merged
    """

    def ents_str(ents: list[Entity], indent=0) -> str:
        json_key = ["name", "label", "aliases", "description", "references"]
        es_dict_list = [{k: getattr(e, k) for k in json_key} for e in ents]
        return json.dumps(es_dict_list, indent=indent)

    if not entities or len(entities) == 1:
        return False, None

    res = await llm.chat(
        PROMPTS["DEDUPLICATE"].format(entities=ents_str(entities)),
        system_prompt=PROMPTS["DEDUPLICATE_SYSTEM"],
    )

    merged, merged_entity = parse_merged_e(res)
    if not merged:
        log.debug(
            f"No merged entity. entities:\n{ents_str(entities,indent=2)}\nres: \n{res}\n"
        )
        return merged, merged_entity

    assert merged_entity is not None
    log.debug(
        f"Deduplicate entities: \n{ents_str(entities)}\n===\n{ents_str([merged_entity],indent=2)}"
    )

    return merged, merged_entity


async def _merge_relation_group(
    relations: list[Relation], related_entities: list[Entity]
) -> list[Relation]:
    """
    deduplicate similar relations
    """

    def rels_str(rels: list[Relation]) -> str:
        return "\n".join([f"{r.origin_str()}" for r in rels])

    if not relations or len(relations) == 1:
        return relations
    res = await llm.chat(
        PROMPTS["DEDUPLICATE_RELATION"].format(
            entities="\n".join(["-" + e.origin_str() for e in related_entities]),
            relations="\n".join(["- " + r.origin_str() for r in relations]),
        ),
        system_prompt=PROMPTS["DEDUPLICATE_RELATION_SYSTEM"],
    )
    try:
        merged, merged_relations = parse_merged_r(res)
        if not merged:
            log.debug(
                f"No merged relations found. Relation group: \n{rels_str(relations)}\nres: \n{res}\n"
            )
            return relations
    except Exception as e:
        log.error(
            f"Error parsing merged relations: Relation group: \n{rels_str(relations)}\nres: \n{res}\nexception:{e}"
        )
        return relations

    log.debug(
        f"Deduplicate relations: \n{rels_str(relations)}\n===\n{rels_str(merged_relations)}"
    )

    return merged_relations


def group_by_name_alias_v2(
    entities: list[Entity], similarity: float = 0.9
) -> list[list[Entity]]:
    """
    Group entities by overlapping names and aliases. 
    if two entities have the same name or alias (simil), they are in the same group
    """

    @lru_cache(maxsize=None)
    def compute_similarity(s1: str, s2: str) -> float:
        return token_sort_ratio(s1.upper(), s2.upper()) / 100.0

    def same_entity(e1: Entity, e2: Entity, similarity: float = 0.9) -> bool:
        e1_strs = [e1.name] + (e1.aliases or [])
        e2_strs = [e2.name] + (e2.aliases or [])
        for s1 in e1_strs:
            for s2 in e2_strs:
                if compute_similarity(s1, s2) >= similarity:
                    return True
        return False

    if not entities:
        return []
    groups: list[list[Entity]] = [[entities[0]]]

    for entity in entities[1:]:
        matched = False
        for group in groups:
            if any(same_entity(entity, e, similarity=similarity) for e in group):
                group.append(entity)
                matched = True
                break
        if not matched:
            groups.append([entity])
    return groups


def group_by_name_alias(entities: list[Entity]) -> list[list[Entity]]:
    """
    Group entities by overlapping names and aliases.
    if two entities have the same name or alias (simil), they are in the same group

    Args:
        entities (list[Entity]): List of entities to group

    Returns:
        list[list[Entity]]: List of entity groups where entities in each group share names/aliases
    """
    # Initialize Union-Find data structure
    parent = {i: i for i in range(len(entities))}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Create name to entity index mapping
    name_to_indices = defaultdict(list)
    for i, entity in enumerate(entities):
        # Add the main name
        name_to_indices[entity.name.upper()].append(i)
        # Add aliases if they exist
        if entity.aliases:
            for alias in entity.aliases:
                name_to_indices[alias.upper()].append(i)

    # Group entities with overlapping names
    for indices in name_to_indices.values():
        for i in range(len(indices) - 1):
            union(indices[i], indices[i + 1])

    # Collect groups
    groups = defaultdict(list)
    for i in range(len(entities)):
        groups[find(i)].append(entities[i])

    return list(groups.values())


def group_relations(relations: list[Relation]) -> list[list[Relation]]:
    """
    Group relations by overlapping entities. r.source and r.target should be entity names.

    Args:
        entities (list[Entity]): List of entities
        relations (list[Relation]): List of relations

    Returns:
        list[list[Relation]]: List of relation groups where relations in each group share entities
    """

    def same_relation(r1: Relation, r2: Relation) -> bool:
        r1_st = sorted([r1.source, r1.target])
        r2_st = sorted([r2.source, r2.target])
        return (
            r1_st[0].upper() == r2_st[0].upper()
            and r1_st[1].upper() == r2_st[1].upper()
        )

    groups: list[list["Relation"]] = []

    for relation in relations:
        # Try to find matching group
        matched = False
        for group in groups:
            # Check first relation in group since all relations in group match
            sample = group[0]
            if same_relation(relation, sample):
                group.append(relation)
                matched = True
                break

        # Create new group if no match found
        if not matched:
            groups.append([relation])

    return groups
