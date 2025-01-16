import os
import logging
import pickle
from typing import Callable
from ..types import Entity, Relation, Image

log = logging.getLogger("mgrag")


class MemoryStorage:
    """
    Storage for the data.
    Entities, relations and images are stored in memory.
    """

    def __init__(
        self,
        folder: str,
    ):
        self.folder = folder
        # Initialize storage containers
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.images: list[Image] = []
        self.image_relations: list[Relation] = []

        if folder:
            self._load_from_folder(folder)
        else:
            log.info("No folder specified, creating empty storage")

    def _item_path_dict(self, root_folder: str) -> dict:
        return {
            "entities": os.path.join(root_folder, "entities.pkl"),
            "relations": os.path.join(root_folder, "relations.pkl"),
            "images": os.path.join(root_folder, "images.pkl"),
            "image_relations": os.path.join(root_folder, "image_relations.pkl"),
        }

    def _load_from_folder(self, folder: str):
        """Load data from pickle files in the specified folder"""
        paths = self._item_path_dict(folder)

        for key, path in paths.items():
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.__dict__[key] = data

    def save_to_folder(self, folder: str | None = None):
        """Save data to pickle files in the specified folder"""
        save_folder = folder or self.folder
        os.makedirs(save_folder, exist_ok=True)

        paths = self._item_path_dict(save_folder)
        for key, path in paths.items():
            with open(path, "wb") as f:
                pickle.dump(self.__dict__[key], f)

        with open(self.folder + "/eris.txt", "w", encoding="utf-8") as f:
            f.write("# Entities\n")
            f.write("\n".join([e.model_dump_json() for e in self.entities]))
            f.write("\n\n# Relations\n")
            f.write("\n".join([r.model_dump_json() for r in self.relations]))
            f.write("\n\n# Images\n")
            f.write("\n".join([i.model_dump_json() for i in self.images]))
            f.write("\n\n# Image Relations\n")
            f.write("\n".join([r.model_dump_json() for r in self.image_relations]))

        if os.environ.get("NEO4J_URL"):
            self.save_to_neo4j(
                os.environ.get("NEO4J_URL"),
                os.environ.get("NEO4J_USER"),
                os.environ.get("NEO4J_PASSWORD"),
            )

    def add_entities(self, entities: list[Entity]):
        if not entities:
            return
        self.entities.extend(entities)

    def add_relations(self, relations: list[Relation], images: bool = False):
        # Only add new relations
        if not relations:
            return
        if images:
            self.image_relations.extend(relations)
        else:
            self.relations.extend(relations)

    def add_images(self, images: list[Image]):
        if not images:
            return
        self.images.extend(images)

    def get_entities(
        self, wheres: Callable[[Entity], bool] = lambda x: True
    ) -> list[Entity]:
        return [e for e in self.entities if wheres(e)]

    def get_relations(
        self, wheres: Callable[[Relation], bool] = lambda x: True
    ) -> list[Relation]:
        return [r for r in self.relations if wheres(r)]

    def get_images(
        self, wheres: Callable[[Image], bool] = lambda x: True
    ) -> list[Image]:
        return [i for i in self.images if wheres(i)]

    def clear(self):
        self.entities.clear()
        self.relations.clear()
        self.images.clear()
        self.image_relations.clear()

    def get_entity_relations(self, entity_name: str) -> list[Relation]:
        """Get relations for a given entity"""
        return [
            r
            for r in self.relations
            if r.source == entity_name or r.source == entity_name
        ]

    def deduplicate(self, entities: list[Entity], merged_entity: Entity):
        """
        Deduplicate entities and merge their properties
        merge the entities into one entity and update related relations
        """
        old_entity_names = [e.name for e in entities]
        for r in self.relations:
            if r.source in old_entity_names:
                r.source = merged_entity.name
            if r.target in old_entity_names:
                r.target = merged_entity.name
        for e in entities:
            self.entities.remove(e)
        self.entities.append(merged_entity)

    def save_to_neo4j(self, url: str, user: str, password: str) -> bool:

        from neo4j import Driver, GraphDatabase

        with GraphDatabase.driver(url, auth=(user, password)) as driver:
            with driver.session() as session:
                # Clear the database
                session.run("MATCH (n) DETACH DELETE n")

                # Add entities
                for entity in self.entities:
                    session.run(
                        "CREATE (n:Entity {name: $name, label: $label, description: $description, aliases: $aliases, references: $references})",
                        name=entity.name,
                        label=entity.label,
                        description=entity.description,
                        aliases=entity.aliases,
                        references=entity.references,
                    )

                # Add relations
                for relation in self.relations:
                    session.run(
                        "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) CREATE (source)-[:RELATION {label: $label, description: $description, references: $references}]->(target)",
                        source=relation.source,
                        target=relation.target,
                        label=relation.label,
                        description=relation.description,
                        references=relation.references,
                    )

                # Add image entity
                # image properties: path, caption, description, texts
                for image in self.images:
                    session.run(
                        "CREATE (n:Image {path: $path, caption: $caption, description: $description, texts: $texts})",
                        path=image.path,
                        caption=image.caption,
                        description=image.description,
                        texts=image.texts,
                    )

                # Add image relations
                for relation in self.image_relations:
                    session.run(
                        "MATCH (source:Entity {name: $source}), (target:Image {path: $target}) CREATE (source)-[:RELATION {label: $label, description: $description, references: $references}]->(target)",
                        source=relation.source,
                        target=relation.target,
                        label=relation.label,
                        description=relation.description,
                        references=relation.references,
                    )

        return True
