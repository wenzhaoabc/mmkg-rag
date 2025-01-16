"""
MultiModal Indexing: Image, Table, and ...
"""

import base64
import logging
import re
from pathlib import Path
from functools import lru_cache
from rapidfuzz.fuzz import token_sort_ratio
import asyncio
from ..utils import llm, encode_image, image_base64_url
from ..types import Entity, Relation, Image
from .parser import parse_image_description, parse_json_list
from .prompts import PROMPTS


log = logging.getLogger("mgrag")


async def mmodal_index(
    text: str, entities: list[Entity], root_path: str | None = None
) -> tuple[list[Relation], list[Image]]:
    """
    Index images in text

    Args:
        text (str): The text to index
        entities (list[Entity]): The entities to link to the images
        root_path (str, optional): The root path of the images. Defaults to None.

    Returns:
        list[Relation]: The image-entity relations
    """
    # Extract images
    images = extract_images(text)
    if not images:
        return [], []

    # Confirm images exist
    confirmed_images = []
    for path, context in images:
        if root_path:
            image_path = Path(root_path) / path
        else:
            image_path = Path(path)

        if not image_path.exists():
            log.warning(f"Image not found at {image_path}")
        elif image_path.suffix[1:] not in ["jpg", "jpeg", "png", "gif", "webp"]:
            log.warning(f"Unsupported image format at {path}")
        else:
            confirmed_images.append((str(image_path), context))

    # Describe images
    log.info("Create image descriptions...")
    img_desc_tasks = [
        image_description(path, context) for path, context in confirmed_images
    ]
    img_descs = await asyncio.gather(*img_desc_tasks)
    img_descs = [img for img in img_descs if img]
    log.info(f"Processed {len(img_descs)} images.")

    # Link images to entities
    log.info(f"Linking images to entities...")
    image_entities = [
        (img, _search_related_entities(entities, img)) for img in img_descs
    ]
    link_tasks = [
        link_image_to_entities(related_entities[: min(8, len(related_entities))], image)
        for image, related_entities in image_entities
    ]
    # Flatten results
    relations = [rels for rels in await asyncio.gather(*link_tasks) if rels]
    relations = [rel for rels in relations for rel in rels if rel]

    return relations, img_descs


def extract_images(text: str) -> list[tuple[str, str]]:
    """
    Extract images from text

    Args:
        text (str): The text (markdown format) to extract images from

    Returns:
        list[tuple[str, str]]: The extracted images with their path and context
    """
    # Find all markdown images
    image_pattern = r"!\[(?:[^\]]*)\]\(([^)]+)\)"
    images = []

    for match in re.finditer(image_pattern, text):
        path = match.group(1)
        start_pos = max(0, match.start() - 200)
        end_pos = min(len(text), match.end() + 200)

        # Get context and expand to complete sentences
        context = text[start_pos:end_pos]

        # Ensure context starts with a complete sentence
        if start_pos > 0:
            first_period = context.find(".")
            first_newline = context.find("\n")
            first_break = min(x for x in [first_period, first_newline] if x != -1)
            if first_break != -1:
                context = context[first_break + 1 :].lstrip()

        # Ensure context ends with a complete sentence
        if end_pos < len(text):
            last_period = context.rfind(".")
            last_newline = context.rfind("\n")
            last_break = max(last_period, last_newline)
            if last_break != -1:
                context = context[: last_break + 1]

        images.append((path, context.strip()))

    return images


@lru_cache()
async def image_description(path: str, context: str) -> Image | None:
    """
    Describe image by LLM

    Args:
        path (str): The path of the image
        context (str): The context of the image
        root_path (str, optional): The root path of the image. Defaults to None.
    Returns:
        Image: The described image
    """
    # Get image path
    if not Path(path).exists():
        log.error(f"Image not found at {path}")
        return None
    # Describe image
    messages = [
        {"role": "system", "content": PROMPTS["DESCRIBE_IMAGE_SYSTEM"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPTS["DESCRIBE_IMAGE"].format(context=context),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{image_base64_url(path)}"},
                },
            ],
        },
    ]
    description = await llm.chat_msg_sync(messages)
    caption, text_snippets, description = parse_image_description(description)
    return Image(
        path=path, caption=caption, texts=text_snippets, description=description
    )


def _search_related_entities(entities: list["Entity"], image: Image) -> list["Entity"]:
    """
    Search for related entities to the image and sort by relevance

    Args:
        entities (list[Entity]): The entities to search for
        image (Image): The image to search in

    Returns:
        list[Entity]: The related entities sorted by relevance score
    """

    @lru_cache(maxsize=None)
    def compute_similarity(s1: str, s2: str) -> float:
        return token_sort_ratio(s1.upper(), s2.upper()) / 100.0

    def compute_avg_similarity(list1: list[str], list2: list[str]) -> float:
        """
        Calculate average similarity between two lists of strings
        """
        if not list1 or not list2:
            return 0.0

        similarities = [compute_similarity(s1, s2) for s1 in list1 for s2 in list2]
        return sum(similarities) / len(similarities)

    def compute_entity_relevance(entity: Entity) -> float:
        """
        Calculate entity's relevance score to the image
        """
        entity_terms = [entity.name] + (entity.aliases or [])

        # Calculate similarity with image texts
        text_similarity = (
            compute_avg_similarity(entity_terms, image.texts or [])
            if image.texts
            else 0.0
        )

        # Calculate similarity with image caption
        caption_similarity = (
            compute_avg_similarity(entity_terms, [image.caption])
            if image.caption
            else 0.0
        )

        # Weight caption similarity slightly higher than text similarity
        score = caption_similarity * 0.6 + text_similarity * 0.4
        return score

    related_entities = []
    if not image.texts and not image.caption:
        return related_entities

    # Calculate relevance scores for each entity
    scored_entities = [
        (entity, compute_entity_relevance(entity)) for entity in entities
    ]

    # Filter entities with minimum relevance and sort by score
    MIN_RELEVANCE = 0.1
    related_entities = [
        entity
        for entity, score in sorted(scored_entities, key=lambda x: x[1], reverse=True)
        if score >= MIN_RELEVANCE
    ]

    return related_entities


async def link_image_to_entities(
    related_entities: list["Entity"], image: Image
) -> list[Relation]:
    """
    Link image to entities

    Args:
        related_entities (list[Entity]): The related entities
        image (Image): The image to link

    Returns:
        Relation: The image-entity relation
    """

    def _entity_json_str(entity: Entity) -> str:
        return entity.model_dump_json(
            include={"name", "aliases", "description", "references"}
        )

    def _image_json_str(image: Image) -> str:
        return image.model_dump_json(include={"caption", "description", "texts"})

    image_path = Path(image.path)
    if not image_path.exists():
        log.error(f"Image not found at {image_path}")
        return []

    messages = [
        {"role": "system", "content": PROMPTS["EI_LINK_SYSTEM"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPTS["EI_LINK"].format(
                        entities="["
                        + ",\n".join([_entity_json_str(e) for e in related_entities])
                        + "]",  # noqa
                        image=_image_json_str(image),
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{image_base64_url(str(image_path))}",
                    },
                },
            ],
        },
    ]

    res = await llm.chat_msg_sync(messages)
    log.debug(
        f'Link image to entities: entities:{",".join([e.name for e in related_entities])}, image:{image_path}, \nLLM res:\n{res}'
    )
    rels = parse_json_list(res, fields=["entity", "label", "references", "description"])
    if not rels:
        return []
    entity_image_rels = []
    for r in rels:
        entity_image_rels.append(
            Relation(
                source=r["entity"],
                target=image.path,
                label="#image" + r["label"],
                references=r["references"],
                description=r["description"],
            )
        )

    return entity_image_rels
