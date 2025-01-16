"""
Text pipeline module
"""

import os
import logging
from pathlib import Path
import asyncio
from langchain_text_splitters import MarkdownTextSplitter

from ..types.chunk import Chunk
from ..utils.helper import extract_image_links, pdf_2_md
from ..storage import MemoryStorage
from .text import extract_er_from_chunk
from .deduplicate import deduplicate
from .mmodal import mmodal_index
from .lables import get_default_lables


log = logging.getLogger("mgrag")


def split_text(
    file_path: str, chunk_size: int = 4000, overlap: int = 200
) -> tuple[list["Chunk"], str]:
    """
    split text into chunks

    Args:
        file_path (str): path to the file
        chunk_size (int, optional): size of each chunk. Defaults to 4000.
        overlap (int, optional): overlap between chunks. Defaults to 200.
    """
    chunks: list["Chunk"] = []
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    for i, chunk in enumerate(splitter.split_text(text)):
        chunk_id = i + 1
        chunk = Chunk(id=chunk_id, text=chunk, images=extract_image_links(chunk))
        chunks.append(chunk)
    return chunks, text


async def index_graph(
    file_path: str,
    chunk_size: int = 8000,
    overlap: int = 400,
    output_path: str = "flink",  # database
    entity_labels: list[str] | None = None,
    relation_labels: list[str] | None = None,
) -> tuple:
    """
    Index the graph for a given file
    """
    log.info("Initializing storage ...")
    storage = MemoryStorage(folder=f"{output_path}")

    log.info(f"Indexing graph for {file_path}")
    chunks, text = split_text(file_path, chunk_size, overlap)
    log.info(f"Indexing {len(chunks)} chunks ...")

    entities, relations = [], []
    # Concurrent processing of chunks
    if not entity_labels:
        entity_labels, _ = get_default_lables()
    if not relation_labels:
        _, relation_labels = get_default_lables()

    tasks = [
        extract_er_from_chunk(
            chunk, entity_labels=entity_labels, relation_labels=relation_labels
        )
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks)
    for es, rs in results:
        entities.extend(es)
        relations.extend(rs)

    log.info(f"Indexed {len(entities)} entities and {len(relations)} relations")
    log.info(f"Deduplicating ...")
    entities, relations = await deduplicate(
        entities + storage.entities, relations + storage.relations
    )

    log.info(f"Final entities: {len(entities)}, relations: {len(relations)} for {file_path}")

    # Add image relations
    images_root_path = Path(file_path).parent
    image_relations, images = await mmodal_index(
        text, entities, images_root_path.as_posix()
    )
    log.info(f"Indexed {len(image_relations)} image relations")

    entities.sort(key=lambda e: e.name)
    relations.sort(key=lambda r: r.source + r.target)
    image_relations.sort(key=lambda r: r.source + r.target)

    # 更新storage
    log.info("Update memory storage ...")
    storage.entities = entities
    storage.relations = relations
    storage.add_images(images)
    storage.add_relations(image_relations, images=True)

    storage.save_to_folder()
    log.info("Saved to storage folder, %s", storage.folder)
    return entities, relations, images, image_relations


async def process_files(
    file_paths: list[str],
    database: str,
    chunks_size: int = 8000,
    overlap: int = 400,
    entity_labels: list[str] | None = None,
    relation_labels: list[str] | None = None,
) -> tuple:
    """
    Process a file and return the path to the markdown file
    database : root path of files
    """
    _root_path = "databases"
    try:
        os.makedirs(f"{_root_path}/{database}", exist_ok=True)
    except Exception as e:
        log.error(f"Error creating folder {e}")
        exit(1)
    entities, relations, images, image_relations = [], [], [], []
    for file_path in file_paths:
        file_type = file_path.split(".")[-1]
        file_name = os.path.basename(file_path).split(".")[0]
        if file_type == "pdf":
            log.info(f"Converting {file_name}.{file_type} to markdown ...")
            md_file_path = pdf_2_md(file_path, f"{_root_path}/{database}/{file_name}")
            log.info(f"Converted  {file_name}.{file_type} to {md_file_path}")
            log.info(f"Indexing graph for {file_name}.{file_type}")
            es, rs, imgs, irs = await index_graph(
                md_file_path,
                output_path=f"{_root_path}/{database}",
                chunk_size=chunks_size,
                overlap=overlap,
                entity_labels=entity_labels,
                relation_labels=relation_labels,
            )
            entities.extend(es)
            relations.extend(rs)
            images.extend(imgs)
            image_relations.extend(irs)
        elif file_type == "md" or file_type == "txt":
            log.info(f"Indexing graph for {file_name}.{file_type}")
            es, rs, imgs, irs = await index_graph(
                file_path,
                output_path=f"{_root_path}/{database}",
                chunk_size=chunks_size,
                overlap=overlap,
                entity_labels=entity_labels,
                relation_labels=relation_labels,
            )
            entities.extend(es)
            relations.extend(rs)
            images.extend(imgs)
            image_relations.extend(irs)

    log.info(
        f"Create MultiModal Graph for {database}, {len(entities)} entities, {len(relations)} relations, {len(images)} images, {len(image_relations)} image relations"
    )
    log.info("Finished processing files, stored in %s", f"{_root_path}/{database}")
    return entities, relations, images, image_relations
