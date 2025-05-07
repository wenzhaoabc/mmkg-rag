"""
Retrieval
"""

import re
import os
import logging
from pathlib import Path
import asyncio
import gradio as gr
# from gradio_m3d_chatbot import m3d_chatbot
from ..retrieval.classify import query_dismantle
from ..retrieval.generate import generate_answer
from ..utils import image_base64_url

from .helper import _DATABASE_DIR

log = logging.getLogger("mgrag")


def direct_query_response(llm_res: str):
    return "> Direct query detected. No further retrieval needed.\n\n" + llm_res


def keyword_extraction_response(keywords: list):
    return (
        "\n> Extracted keywords from the query:\n> \n"
        + "> "
        + ", ".join(keywords)
        + "\n"
    )


def references_content(refs: dict):
    (
        entities,
        relations,
        related_entities,
        images,
        image_related_entities,
        image_relations,
    ) = (
        refs.get("entities", []),
        refs.get("relations", []),
        refs.get("related_entities", []),
        refs.get("images", []),
        refs.get("image_related_entities", []),
        refs.get("image_relations", []),
    )
    refs_str = ""
    if entities:
        refs_str += "## Entities\n\n"
        for entity in entities:
            refs_str += f"- {entity.name}, {entity.aliases}, {entity.description}\n"
    if relations:
        refs_str += "## Relations\n\n"
        for relation in relations:
            refs_str += f"- {relation.source}, {relation.target}, {relation.label}, {relation.description}\n"
    if related_entities:
        refs_str += "## Related Entities\n\n"
        for related_entity in related_entities:
            refs_str += f"- {related_entity.name}, {related_entity.aliases}, {related_entity.description}\n"
    if images:
        refs_str += "## Images\n\n"
        for image in images:
            refs_str += f"- {image.caption}, {image.description}\n"
    if image_related_entities:
        refs_str += "## Image Related Entities\n\n"
        for image_related_entity in image_related_entities:
            refs_str += f"- {image_related_entity.name}, {image_related_entity.aliases}, {image_related_entity.description}\n"
    if image_relations:
        refs_str += "## Image Relations\n\n"
        for image_relation in image_relations:
            refs_str += f"- {image_relation.source}, {image_relation.target}, {image_relation.label}, {image_relation.description}\n"
    return refs_str


def process_llm_response(llm_res: str) -> str:
    markdown_image_pattern = r"!\[.*?\]\((.*?)\)"
    html_image_pattern = r'<img src="(.*?)"'
    # replace image path with perfix gradio_api/file=
    llm_res = re.sub(markdown_image_pattern, r"![image](gradio_api/file=\1)", llm_res)
    llm_res = re.sub(html_image_pattern, r'<img src="gradio_api/file=\1"', llm_res)

    return llm_res


def process_history(history: list[dict]) -> list:
    def compress_user_ast_list(history: list[dict]) -> list[list[dict]]:
        if not history:
            return []
        compressed_list = []
        current_group = [history[0]]
        for msg in history[1:]:
            # Skip system and assistant references messages
            if msg["role"] == "system" or msg["role"] == "assistant":
                if (
                    isinstance(msg.get("metadata", None), dict)
                    and msg.get("metadata", {}).get("title", None) == "References"
                ):
                    continue
            if msg["role"] == current_group[0]["role"]:
                current_group.append(msg)
            else:
                compressed_list.append(current_group)
                current_group = [msg]
        compressed_list.append(current_group)
        return compressed_list

    def content_format(
        content: str | dict,
    ) -> str | dict:
        if isinstance(content, str):
            return {"type": "text", "text": content}
        if isinstance(content, dict):
            if content.get("path", ""):
                url = image_base64_url(content.get("path", ""))
            else:
                raise ValueError(f"Unsupported image type: {content}")
            return {
                "type": "image_url",
                "image_url": {"url": url},
            }

        return {"type": "text", "text": content.__str__()}

    # llm history
    llm_history: list[dict] = []
    history_groups = compress_user_ast_list(history)
    for msg_group in history_groups:
        current_message = dict()
        current_message["role"] = msg_group[0]["role"]

        current_message["content"] = []
        for chatmsg in msg_group:
            current_message["content"].append(content_format(chatmsg["content"]))
        llm_history.append(current_message)

    return llm_history


def llm_chatbot(
    message: dict,
    history: list[dict] | None = None,
    contain_image: bool = True,
    max_entities_num: int = 5,
    similarity_threshold: int = 20,
    max_images_num: int = 2,
    force_retrieval: bool = False,
):
    if not contain_image:
        max_images_num = 0

    history = history or []

    user_query = message["text"]
    user_query_images = message.get("files", [])
    if not user_query:
        return "", history

    history.append({"role": "user", "content": user_query})
    for file in message["files"]:
        history.append({"role": "user", "content": {"path": file, "alt_text": "image"}})

    # query answer
    history.append(dict(role="assistant", content="> Prerocessing query...\n"))
    yield "", history

    query_res = asyncio.run(query_dismantle(
            query=user_query,
            images=user_query_images,
            history=process_history(history),
            force_retrieval=force_retrieval,
        )
    )
    if not query_res:
        history[-1]["content"] = "An error occurred during query processing."
        yield "", history
        return

    if query_res["classification"] == "direct":
        response = direct_query_response(query_res["response"])
        history[-1]["content"] = response
        yield "", history
        return

    # Retrieval and Answer Generation
    response = keyword_extraction_response(query_res["keywords"])
    history[-1]["content"] = response
    yield "", history

    # Generate answer from the knowledge base
    generate_answer_res = asyncio.run(
        generate_answer(
            query_res["keywords"],
            query=user_query,
            query_images=user_query_images,
            history=process_history(history),
            max_num=max_entities_num,
            similarity_threshold=similarity_threshold,
            max_images_num=max_images_num,
        )
    )

    postprocessed_res = process_llm_response(generate_answer_res["response"])
    response += "\n\n" + postprocessed_res
    history[-1]["content"] = response
    history.append(
        {
            "role": "assistant",
            "content": references_content(generate_answer_res["konwledge"]),
            "metadata": {"title": "References"},
        }
    )
    log.debug(
        f"LLM {os.environ["LLM_MODEL"]} Answer for query: {user_query}\nLLM answer:\n{generate_answer_res["response"]}\nReferences:\n{references_content(generate_answer_res["konwledge"])}"
    )
    yield "", history
    return


def list_all_knowledgws():
    dirs = os.listdir(_DATABASE_DIR)
    dirs = [d for d in dirs if os.path.isdir(f"{_DATABASE_DIR}/{d}")]
    global _ALL_KNOWLEDGES
    _ALL_KNOWLEDGES = dirs
    return gr.Dropdown([d for d in dirs], interactive=True, label="Select Database")


def change_knowledge_graph(knowledge_graph: str):
    print(f"Change knowledge graph to {knowledge_graph}")
    from ..retrieval.search import load_default_ers

    load_default_ers(f"{_DATABASE_DIR}/" + knowledge_graph)


def change_llm_model(llm_model: str):
    os.environ["LLM_MODEL"] = llm_model


# region retrieval
with gr.Blocks(
    title="MGRAG",
    fill_height=True,
    fill_width=True,
    css_paths=[Path("src/mmkg_rag/gui/front/index.css")],
) as retrieval:
    gr.Markdown("## RAG with Knowledge Graph")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Group():
                    kg_select = gr.Dropdown(
                        [], interactive=True, label="Select Database"
                    )
                    update_kgs = gr.Button("List All Databases")
                    update_kgs.click(list_all_knowledgws, outputs=[kg_select])
                    # kg_select.focus(list_all_knowledgws)
                    kg_select.select(change_knowledge_graph, inputs=[kg_select])

            with gr.Column():
                with gr.Group():
                    contain_image = gr.Checkbox(
                        value=True,
                        label="Include Image",
                        info="Whether the answer contains images",
                    )
                    force_retrieval = gr.Checkbox(
                        value=False,
                        label="Force Retrieval",
                        info="Force retrieval even if the query is direct",
                    )
                    max_entities_num = gr.Number(
                        value=5,
                        minimum=2,
                        maximum=10,
                        label="Max Entities",
                        info="The maximum number of entities selected",
                    )
                    similarity_threshold = gr.Number(
                        value=20,
                        minimum=0,
                        maximum=100,
                        step=5,
                        label="Similarity Threshold",
                        info="The similarity threshold for the retrieval,0-100",
                    )
                    max_images_num = gr.Number(
                        value=2,
                        minimum=0,
                        maximum=10,
                        label="Max Images",
                        info="The maximum number of images to be included in the answer",
                    )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                label="Chatbot",
                show_label=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": True},
                    {"left": "\\[", "right": "\\]", "display": True},
                ],
                bubble_full_width=False,
                sanitize_html=False,
                elem_id="chatbot_id",
            )

            with gr.Row():
                with gr.Column(scale=9):
                    msg = gr.MultimodalTextbox(
                        label="Question",
                        placeholder="Type your query here",
                        max_lines=500,
                    )
                with gr.Column(scale=1):
                    llm_model = gr.Dropdown(
                        ["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet"],
                        label="LLM Model",
                        interactive=True,
                        allow_custom_value=True,
                    )
                    llm_model.change(change_llm_model, inputs=[llm_model])
                    clean_btn = gr.Button("Clear Chat")

                    clean_btn.click(lambda: chatbot.clear(), outputs=[chatbot])

    msg.submit(
        llm_chatbot,
        inputs=[
            msg,
            chatbot,
            contain_image,
            max_entities_num,
            similarity_threshold,
            max_images_num,
            force_retrieval,
        ],
        outputs=[msg, chatbot],
    )

    examples = gr.Examples(
        examples=[
            [
                {
                    "text": "Explain how TCP establishes a connection through the three-way handshake, and include diagrams to illustrate the process.",
                    "files": [],
                },
                False,
                5,
                20,
                2,
                False,
            ],
            [
                {
                    "text": "In the paper, what is the overall architecture of Graph RAG? Please provide an architecture diagram and explain it.",
                    "files": [],
                },
                True,
                5,
                20,
                2,
                False,
            ],
            [
                {
                    "text": "This is an architecture diagram of a RAG system using a multimodal knowledge graph, MMKG-RAG. Please provide an overview of MMKG-RAG and the Graph RAG and Light RAG proposed in the paper, focusing on the aspects of knowledge graph construction and retrieval-based question answering, and highlight their differences with their architecture diagrams. Thank you!",
                    "files": ["paper/images/mmkg-rag-architecture.png"],
                },
                True,
                5,
                20,
                8,
                True,
            ],
        ],
        inputs=[
            msg,
            contain_image,
            max_entities_num,
            similarity_threshold,
            max_images_num,
            force_retrieval,
        ],
    )

# endregion
