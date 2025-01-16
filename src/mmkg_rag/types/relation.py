from typing import Optional
from pydantic import BaseModel


class Relation(BaseModel):

    source: str
    """The source entity of the relation."""

    target: str
    """The target entity of the relation."""

    label: str
    """The type of the relation."""

    references: Optional[list[str]] = None
    """The original text references of the relation"""

    images: Optional[list[str]] = None
    """The images path associated with the relation"""

    chunks: Optional[list[int]] = None
    """The chunks associated with the relation"""

    description: Optional[str] = None
    """The description of the relation"""

    def __hash__(self):
        return hash(self.source + self.target + self.label)

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.label == other.label
            and self.description == other.description
        )

    def origin_str(self):
        ref_str = (
            f'[{", ".join([f"\"{r}\"" for r in self.references])}]'
            if self.references
            else "[]"
        )
        # source, label, target, description, references
        return f'{{"source": "{self.source}", "label": "{self.label}", "target": "{self.target}", "description": "{self.description}", "references": {ref_str}}}'
