from typing import Optional
from pydantic import BaseModel


class Entity(BaseModel):

    name: str
    """The unique identifier of the entity."""

    description: str
    """The text of the entity."""

    label: str
    """The type of the entity."""

    references: Optional[list[str]] = None
    """The original text references of the entity"""

    aliases: Optional[list[str]] = None
    """The aliases of the entity"""

    images: Optional[list[str]] = None
    """The images associated with the entity"""

    chunks: Optional[list[int]] = None
    """The chunks associated with the entity"""

    def __hash__(self):
        return hash(self.name + self.label + self.description)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (
            self.name == other.name
            and self.label == other.label
            and self.description == other.description
        )

    def origin_str(self):
        ref_str = (
            f'[{", ".join([f"\"{r}\"" for r in self.references])}]'
            if self.references
            else "[]"
        )
        # name, label, description, aliases, references
        return f'{{"name": "{self.name}", "label": "{self.label}", "description": "{self.description}", "aliases": {self.aliases}, "references": {ref_str}}}'
