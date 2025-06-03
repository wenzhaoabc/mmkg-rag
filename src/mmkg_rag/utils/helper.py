import hashlib
import re
import os
import base64
from pathlib import Path
from typing import List
from pickle import dump


def md5(string: str) -> str:
    """Hash a string using md5."""
    return hashlib.md5(string.encode()).hexdigest()


def extract_image_links(markdown_text: str) -> List[str]:
    """
    Extract image URLs from markdown text

    Args:
        markdown_text (str): Markdown formatted text

    Returns:
        List[str]: List of image URLs found in the text
    """
    # Match markdown image syntax ![alt](url) and HTML <img src="url">
    markdown_pattern = r"!\[.*?\]\((.*?)\)"
    html_pattern = r'<img.*?src=["\'](.*?)["\'].*?>'

    # Find all matches
    markdown_images = re.findall(markdown_pattern, markdown_text)
    html_images = re.findall(html_pattern, markdown_text)

    # Combine both results and remove duplicates
    all_images = list(set(markdown_images + html_images))

    return all_images


def shorten_string(text: str, head_length: int, tail_length: int) -> str:
    """
    Shorten the string by keeping the head and tail of the string
    """
    if len(text) <= head_length + tail_length:
        return text
    return text[:head_length] + "..." + text[-tail_length:]


def encode_image(image_path: str | Path) -> str:
    """
    Encode image to base64
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_base64_url(image_path: str) -> str:
    """
    Return the base64 url of the image
    """
    image_type = image_path.split(".")[-1]
    return f"data:image/{image_type};base64,{encode_image(image_path)}"


def write_er_to_file(entities, relations, images, image_relations, save_path):
    """
    Write entities and relations to a file
    """
    file_path: Path = Path(save_path)
    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)

    with open(file_path / "er.txt", "w", encoding="utf-8") as f:
        f.write("Entities:\n")
        for e in entities:
            e_references = ", ".join([shorten_string(r, 10, 10) for r in e.references])
            e_aliases = ", ".join(e.aliases or [])
            e_str = f"{e.name},\t{e.label},\t[{e_aliases}],\t{e.description}, [{e_references}]"
            f.write(f"{e_str}\n")

        f.write("\n\nRelationships:\n")
        for r in relations:
            r_references = ", ".join([shorten_string(r, 10, 10) for r in r.references])
            r_str = f"{r.source},\t{r.target},\t{r.label},\t{r.description}, [{r_references}]"
            f.write(f"{r_str}\n")

        f.write("\n\nImages:\n")
        for i in images:
            text_snippets = ", ".join(i.texts)
            i_str = f"{i.path},\t{i.caption},\t[{text_snippets}], {i.description}"
            f.write(f"{i_str}\n")

        f.write("\n\nImage Relations:\n")
        for r in image_relations:
            r_references = ", ".join([shorten_string(r, 10, 10) for r in r.references])
            r_str = f"{r.source},\t{r.target},\t{r.label},\t{r.description}, [{r_references}]"
            f.write(f"{r_str}\n")

    # Save entities and relations to pickle file
    with open(file_path / "entities.pkl", "wb") as f:
        dump(entities, f)

    with open(file_path / "relations.pkl", "wb") as f:
        dump(relations, f)

    with open(file_path / "images.pkl", "wb") as f:
        dump(images, f)

    with open(file_path / "image_relations.pkl", "wb") as f:
        dump(image_relations, f)


_model_lst = []


def pdf_2_md(
    file_path: str,
    output_folder: str,
) -> str:
    """
    Convert a pdf file to a markdown file and return the path to the markdown file
    makdown file will be saved in output_folder/<>.md
    image files will be saved in output_folder/<>.png
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(file_path)
    full_text, _, images = text_from_rendered(rendered)

    os.makedirs(Path(output_folder), exist_ok=True)
    file_name = os.path.basename(file_path).split(".")[0]

    with open(f"{output_folder}/{file_name}.md", "w", encoding="utf-8") as f:
        f.write(full_text)

    for image_name, image in images.items():
        image.save(f"{output_folder}/{image_name}")

    return str(Path(f"{output_folder}/{file_name}.md"))


def rename_markdown_images(md_file_path):
    base_name = os.path.splitext(os.path.basename(md_file_path))[0]

    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    parent_folder = os.path.dirname(md_file_path)

    image_counter = 0

    def replace_image(match):
        nonlocal image_counter
        nonlocal parent_folder
        full_path = match.group(1)
        # Split path and file name
        folder_path, file_name = os.path.split(full_path)
        folder_path = folder_path + "/" if folder_path else ""
        # Get file extension
        file_ext = full_path.split(".")[-1]
        new_name = f"{base_name}_{image_counter}.{file_ext}"

        image_counter += 1
        # Rename file
        Path(parent_folder).joinpath(full_path).rename(
            Path(parent_folder).joinpath(folder_path).joinpath(new_name)
        )

        return f"![]({folder_path}{new_name})"

    new_content = re.sub(r"\!\[.*\]\((.+)\)", lambda m: replace_image(m), content)

    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return f"Updated {image_counter} image references in {md_file_path}"
