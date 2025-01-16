from typing import Optional
from pydantic import BaseModel


class Image(BaseModel):
    """
    Image model
    """

    path: str
    """The unique identifier of the image. The project path to the image file."""

    caption: str
    """The caption of the image."""

    description: str
    """The description of the image."""

    texts: Optional[list[str]] = None
    """The text of the image."""
