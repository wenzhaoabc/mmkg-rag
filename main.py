"""
MMKG-RAG: Enhancing Retrieval-Augmented Generation with Multi-Modal Knowledge Graph Integration
"""


import os
from pathlib import Path
import gradio as gr

from src.mmkg_rag.gui.index import index as index_demo
from src.mmkg_rag.gui.retrieval import retrieval as retrieval_demo

tab = gr.TabbedInterface(
    [index_demo, retrieval_demo],
    tab_names=["MMKG Construction", "MMKG-Augmented QA"],
)

with gr.Blocks(
    title="MMKG-RAG", css_paths=[Path("src/mgrag/gui/front/index.css")]
) as demo:
    gr.Markdown(
        "# MMKG-RAG\n\n Enhancing Retrieval-Augmented Generation with Multi-Modal Knowledge Graph Integration. [GitHub](https://github.com/wenzhaoabc/mmkg-rag)",
    )

    tab.render()

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[
            os.path.join(os.path.dirname(__file__), "databases"),
            str(Path("examples/rag").absolute()),
        ],
    )
