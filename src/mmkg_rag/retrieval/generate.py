#
import logging

from ..utils import llm, image_base64_url
from ..types import Entity, Relation, Image
from .search import search_eris
from .prompts import PROMPTS

log = logging.getLogger("mgrag")


def generate_text_prompts(
    entities: list[Entity],
    relations: list[Relation],
    related_entities: list[Entity],
) -> str:
    entities_str = "\n".join(
        [
            f"- {e.name}, {e.aliases}, {e.description}"
            for e in (entities + related_entities)
        ]
    )
    entities_str = f"Entities: every entity has a name, aliases, and a description\n{entities_str}\n"
    relations_str = "\n".join(
        [f"- {r.source}, {r.target}, {r.label}, {r.description}" for r in relations]
    )
    relations_str = f"Relations: every relation has a source, target, label, and a description\n{relations_str}\n"

    return PROMPTS["GENERATE_KNOWLEDGE"].format(
        knowledge=entities_str + "\n" + relations_str
    )


def generate_image_prompts(
    images: list[Image],
    image_relations: list[Relation],
    image_related_entities: list[Entity],
) -> str:
    images_str = "\n".join(
        [f"- {i.path}, {i.caption}, {i.description}" for i in images]
    )
    images_str = (
        f"Images: every image has a path, caption, and a description\n{images_str}\n"
    )

    images_entities_str = "\n".join(
        [f"- {e.name}, {e.aliases}, {e.description}" for e in image_related_entities]
    )
    images_entities_str = f"Entities related with the images: every entity has a name, aliases, and a description\n{images_entities_str}\n"

    image_relations_str = "\n".join(
        [
            f"- {r.source}, {r.target}, {r.label}, {r.description}"
            for r in image_relations
        ]
    )
    image_relations_str = f"Image Relations: every image relation has a source, target, label, and a description\n{image_relations_str}\n"

    return (
        "The following are the images and their related entities:\n"
        + images_str
        + images_entities_str
        + image_relations_str
    )

async def generate_answer(
    keywords: list[str],
    query: str = "",
    query_images: list[str] | None = None,
    history: list | None = None,
    max_num: int = 3,
    max_images_num: int = 2,
    similarity_threshold: float = 55,
    hop: int = 1,
) -> dict:
    """
    Generate the answer by LLM
    """

    if not keywords or not query:
        raise ValueError("Keywords and query cannot be empty")

    (
        entities,
        relations,
        related_entities,
        images,
        image_related_entities,
        image_relations,
    ) = search_eris(
        keywords,
        max_num=max_num,
        max_images_num=max_images_num,
        hop=hop,
        similarity_threshold=similarity_threshold,
    )
    log.debug(
        f'Search results for keywords: {", ".join(keywords)}'
        + "\n"
        + f"Entities {len(entities)}: {', '.join([e.name for e in entities])}\n"
        + f"Relations {len(relations)}: {', '.join([r.source+'&'+r.target for r in relations])}\n"
        + f"Related Entities {len(related_entities)}: {', '.join([e.name for e in related_entities])}\n"
        + f"Images {len(images)}: {', '.join([i.path for i in images])}\n"
        + f"Image Related Entities {len(image_related_entities)}: {', '.join([e.name for e in image_related_entities])}\n"
        + f"Image Relations {len(image_relations)}: {', '.join([r.source+'&'+r.target for r in image_relations])}\n"
    )

    messages: list[dict] = [{"role": "system", "content": PROMPTS["GENERATE_SYSTEM"]}]
    if history:
        messages.extend(history)

    knowledges = generate_text_prompts(entities, relations, related_entities)
    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": knowledges}],
    }
    images_knowledges = generate_image_prompts(
        images, image_relations, image_related_entities
    )
    user_images_content: list[dict] = [
        {
            "type": "text",
            "text": images_knowledges,
        }
    ]
    for image in images or []:
        user_images_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_base64_url(image.path)},
            }
        )

    user_message["content"].extend(user_images_content)
    messages.append(user_message)

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPTS["GENERATE"].format(query=query)}
            ],
        }
    )
    for image in query_images or []:
        messages[-1]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_base64_url(image),
                },
            }
        )
    res = await llm.chat_msg_sync(messages)

    return {
        "response": res,
        "konwledge": {
            "entities": entities,
            "relations": relations,
            "related_entities": related_entities,
            "images": images,
            "image_related_entities": image_related_entities,
            "image_relations": image_relations,
        },
    }
