from typing import Optional
from pydantic import BaseModel


class Chunk(BaseModel):

    id: int
    """The unique identifier of the chunk."""

    text: str
    """The text of the chunk."""

    images: Optional[list[str]] = None
    """The list of images path in the chunk."""
