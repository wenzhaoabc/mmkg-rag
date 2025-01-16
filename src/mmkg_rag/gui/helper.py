import os
from pathlib import Path


_DATABASE_DIR: str = "databases"


def pdf_2_md(
    file_path: str,
    output_folder: str,
) -> str:
    """
    Convert a pdf file to a markdown file and return the path to the markdown file
    """
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

    model_lst = load_all_models()
    full_text, images, out_meta = convert_single_pdf(file_path, model_lst)

    (Path(output_folder) / "md").mkdir(parents=True, exist_ok=True)
    file_name = os.path.basename(file_path).split(".")[0]

    with open(f"{output_folder}/md/{file_name}.md", "w", encoding="utf-8") as f:
        f.write(full_text)

    for image_name, image in images.items():
        image.save(f"{output_folder}/md/{image_name}")

    return str(Path(f"{output_folder}/md/{file_name}.md").absolute())
