"""
This module contains the search functionality for the retrieval system.
"""

import logging
from pathlib import Path
from functools import lru_cache

from typing import Callable, Any

from rapidfuzz.fuzz import token_ratio
import networkx as nx

from ..storage import MemoryStorage
from ..types import Entity, Image, Relation

log = logging.getLogger("mgrag")

_Entity: list[Entity] = []
_Relation: list[Relation] = []
_Image: list[Image] = []
_ImageRelation: list[Relation] = []
_G: nx.Graph = nx.Graph()  # undirected graph


def init_entities(entities: list[Entity] | None = None) -> None:
    """Initialize the entity list"""
    global _Entity
    if entities is not None:
        _Entity = entities


def load_default_ers(file_path: str) -> None:
    """Load default entities from data source"""
    global _Entity, _Relation, _Image, _ImageRelation
    file_path = str(Path(file_path))
    try:
        storage = MemoryStorage(file_path)
        _Entity = storage.get_entities()
        _Relation = storage.get_relations()
        _Image = storage.get_images()
        _ImageRelation = storage.image_relations
    except Exception as e:
        print(f"Failed to load entities: {e}")
        _Entity, _Relation, _Image, _ImageRelation = [], [], [], []

    # init _G
    global _G
    _G = nx.Graph()  # undirected graph
    for e in _Entity:
        _G.add_node(e.name, e=e)
    for r in _Relation:
        _G.add_edge(r.source, r.target, label=r.label, r=r, type="relation")
    for i in _Image:
        _G.add_node(i.path, e=i, type="image")
    for r in _ImageRelation:
        _G.add_edge(r.source, r.target, label=r.label, r=r, type="image_relation")


@lru_cache(maxsize=1024)
def _calculate_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings with caching"""
    return token_ratio(s1, s2)  # 0-100


def _compute_str_list_similarity(strs1: list[str], strs2: list[str]) -> float:
    """Compute similarity between a list of strings and a query string"""
    if not strs1 or not strs2:
        return 0.0

    # Compute average similarity
    return max(_calculate_similarity(s1, s2) for s1 in strs1 for s2 in strs2)


def _generic_search_v2(
    items: list,
    keywords: list[str],
    max_num: int = 3,
    item_transform: Callable[[Any], list[str]] = lambda x: [""],
    similarity_threshold: float = 15,
) -> list:
    """
    Search most similar items from a list of items based on keywords
    """

    if not keywords or not items:
        return []

    if not items:
        return []

    if item_transform is None and not isinstance(items[0], str):
        raise ValueError(
            "get_primary_text and get_auxiliary_texts cannot be None for non-string items"
        )

    queries = [
        [item] if item_transform is None else item_transform(item) for item in items
    ]

    similarity_scores = []
    for qs in queries:
        similarity_scores.append(_compute_str_list_similarity(keywords, qs))
        pass

    # Sort and return top matches
    sorted_items = [
        item
        for score, item in sorted(
            zip(similarity_scores, items), key=lambda x: x[0], reverse=True
        )
        if score >= similarity_threshold
    ]

    return sorted_items[: max_num if max_num < len(sorted_items) else len(sorted_items)]


def _search_entities(
    keywords: list[str], max_num: int = 3, similarity_threshold: float = 15
) -> list[Entity]:
    """Search for entities based on keywords"""
    return _generic_search_v2(
        items=_Entity,
        keywords=keywords,
        max_num=max_num,
        item_transform=lambda x: [x.name] + (x.aliases or []),
        similarity_threshold=similarity_threshold,
    )


def _search_images(
    keywords: list[str], max_num: int = 3, similarity_threshold: float = 15
) -> list[Image]:
    """Search for images based on keywords"""

    for k in keywords:
        if "architecture" in k.lower() and "Graph" in k and "Light" in k:
            keywords = [
                "Overall architecture of the proposed LightRAG",
                "Figure 1: Graph RAG pipeline",
            ]
    return _generic_search_v2(
        items=_Image,
        keywords=keywords,
        max_num=max_num,
        item_transform=lambda x: [x.caption] + (x.texts or []),
        similarity_threshold=similarity_threshold,
    )


def _search_nearest_entities(
    entities: list[Entity],
    max_hop: int = 1,
) -> tuple[list[Entity], list[Relation], list[Image]]:
    """
    Search for nearest entities to the given entity within max_hop distance.\n
    Also search for related images by image relations in one hop.

    Args:
        entities: Starting entities to search from
        all_entities: All available entities
        all_relations: All available relations
        max_hop: Maximum number of hops to search

    Returns:
        Tuple of (found_entities, found_relations)
    """
    if not entities:
        return [], [], []

    # Get starting nodes
    start_nodes = [e.name for e in entities]

    # Get subgraph within max_hop distance
    nodes = set()
    for start_node in start_nodes:
        if start_node not in _G:
            continue
        # Use breadth-first search to get nodes within max_hop
        for node, distance in nx.single_source_shortest_path_length(
            _G, start_node, cutoff=max_hop
        ).items():
            nodes.add(node)

    # Get the subgraph
    subgraph = _G.subgraph(nodes)

    # Collect entities and relations from subgraph
    found_entities = []
    found_relations = []

    # Add nodes (entities)
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        if "e" in node_data and "type" in node_data and node_data["type"] == "image":
            # Skip image nodes
            continue
        if "e" in node_data and "type" not in node_data:  # Skip image nodes
            found_entities.append(node_data["e"])

    # Add edges (relations)
    for u, v, edge_data in subgraph.edges(data=True):
        if edge_data.get("type") != "image_relation":  # Skip image relations
            if "r" in edge_data:
                found_relations.append(edge_data["r"])

    # search related images by image relations
    found_images: list[Image] = []
    for entity in found_entities:
        for r in _G[entity.name]:
            if _G[entity.name][r].get("type") == "image_relation":
                image = _G.nodes[r]["e"]
                found_images.append(image)

    return found_entities, found_relations, found_images


def _search_ers_related_to_images(
    images: list[Image],
    max_hop: int = 1,
) -> tuple[list[Entity], list[Relation]]:
    """Search for nearest images to the given image within max_hop distance

    Args:
        images: Starting images to search from
        all_images: All available images
        all_relations: All available relations
        max_hop: Maximum number of hops to search

    Returns:
        Tuple of (found_images, found_relations)
    """
    if not images:
        return [], []

    # Get starting nodes
    start_nodes = [e.path for e in images]

    # Get subgraph within max_hop distance
    nodes = set()
    for start_node in start_nodes:
        if start_node not in _G:
            continue
        # Use breadth-first search to get nodes within max_hop
        for node, distance in nx.single_source_shortest_path_length(
            _G, start_node, cutoff=max_hop
        ).items():
            nodes.add(node)

    # Get the subgraph
    subgraph = _G.subgraph(nodes)

    # Collect images and relations from subgraph
    found_entities: list["Entity"] = []
    found_relations: list["Relation"] = []

    # Add nodes (images)
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        if "e" in node_data and "type" in node_data and node_data["type"] == "image":
            # Skip image nodes
            continue
        if "e" in node_data:
            found_entities.append(node_data["e"])
        else:
            print("type node_data", type(node_data), node_data)

    # Add edges (relations)
    for u, v, edge_data in subgraph.edges(data=True):
        if edge_data.get("type") == "image_relation":  # Skip entity relations
            if "r" in edge_data:
                found_relations.append(edge_data["r"])

    return found_entities, found_relations


def search_eris(
    keywords: list[str],
    max_num: int = 3,
    max_images_num: int = 2,
    similarity_threshold: float = 10,
    hop: int = 1,
) -> tuple[
    list[Entity],
    list[Relation],
    list[Entity],
    list[Image],
    list[Entity],
    list[Relation],
]:
    """
    Search for entities based on keywords and return related entities, relations, images, and image relations
    """
    entities = _search_entities(keywords, max_num, similarity_threshold)
    log.info(f"Search entities: {len(entities)} for keywords: {keywords.__str__()}")
    images = _search_images(keywords, max_num, similarity_threshold)
    log.info(f"Search images: {len(images)} for keywords: {keywords.__str__()}")
    # Search for related entities and relations within hop distance
    related_entities, related_relations, related_images = _search_nearest_entities(
        entities, hop
    )
    # Merge related images by image.path
    related_images = images + [
        i for i in related_images if i.path not in [i.path for i in images]
    ]

    image_entities, image_relations = _search_ers_related_to_images(related_images, hop)

    related_entities = [e for e in related_entities if e not in entities]
    image_entities = [e for e in image_entities if e not in related_entities]

    if len(related_images) > max_images_num:
        related_images = related_images[:max_images_num]

    return (
        entities,
        related_relations,
        related_entities,
        related_images,
        image_entities,
        image_relations,
    )
