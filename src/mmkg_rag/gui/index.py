import os
import datetime
from pathlib import Path
import pickle
import asyncio
import gradio as gr
import networkx as nx
import plotly.graph_objects as go

from ..index.pipe import process_files


from .helper import _DATABASE_DIR

_CURRENT_DATABASE = None

_ALL_KNOWLEDEGS: list[str] = []


def _list_all_knowledgws():
    dirs = os.listdir(_DATABASE_DIR)
    dirs = [d for d in dirs if os.path.isdir(f"{_DATABASE_DIR}/{d}")]
    global _CURRENT_DATABASE, _ALL_KNOWLEDEGS
    _ALL_KNOWLEDEGS = dirs
    _CURRENT_DATABASE = dirs[0] if dirs else None
    return gr.Dropdown(
        choices=dirs, label="Select Database", interactive=True, allow_custom_value=True
    )


def _change_database(database: str):
    global _CURRENT_DATABASE
    _CURRENT_DATABASE = database


def index_file_graph(
    file_path: list[str],
    chunk_size: int = 8000,
    overlap: int = 400,
    base_url: str | None = None,
    api_key: str | None = None,
    llm_model: str | None = None,
    entity_labels: str = "",
    relation_labels: str = "",
):
    if base_url:
        os.environ["BASE_URL"] = base_url
    if api_key:
        os.environ["API_KEY"] = api_key
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model

    # Examples relative path
    if (
        len(file_path) == 2
        and file_path[0].endswith("graphrag.md")
        and file_path[1].endswith("lightrag.md")
    ):
        file_path = ["examples/rag/graphrag.md", "examples/rag/lightrag.md"]

    result_log = (
        "Program configs:\n"
        + f"BASE_URL: {base_url}\n\nAPI_KEY: ....\n\nLLM_MODEL: {llm_model}\n\nDatabase: {_CURRENT_DATABASE}\n"
    )
    yield result_log
    result_log += f"\nStart indexing {[Path(p).name for p in file_path].__str__()} at {datetime.datetime.now()}...\n"
    yield result_log

    es, rs, imgs, irs = asyncio.run(
        process_files(
            file_path,
            database=_CURRENT_DATABASE or "default",
            chunks_size=chunk_size,
            overlap=overlap,
            entity_labels=entity_labels.split(","),
            relation_labels=relation_labels.split(","),
        )
    )

    result_log += f"\nIndexed {len(es)} entities and {len(rs)} relations\n"
    result_log += f"\nIndexed {len(imgs)} images and {len(irs)} image relations\n"
    yield result_log
    result_log += "\nSave to database...\n"
    result_log += f"\n\nEND!"
    yield result_log
    return


def draw_interactive_graph(
    entities: list | None = None,  # Entity
    relations: list | None = None,  # Relation
    images: list | None = None,  # Image
    image_relations: list | None = None,  # Relation
):
    global _CURRENT_DATABASE
    if not entities:
        entities = pickle.load(
            open(f"{_DATABASE_DIR}/{_CURRENT_DATABASE}/entities.pkl", "rb")
        )
    if not relations:
        relations = pickle.load(
            open(f"{_DATABASE_DIR}/{_CURRENT_DATABASE}/relations.pkl", "rb")
        )
    if not images:
        images = pickle.load(
            open(f"{_DATABASE_DIR}/{_CURRENT_DATABASE}/images.pkl", "rb")
        )
    if not image_relations:
        image_relations = pickle.load(
            open(f"{_DATABASE_DIR}/{_CURRENT_DATABASE}/image_relations.pkl", "rb")
        )

    G = nx.Graph()
    for e in entities or []:
        G.add_node(e.name, type="entity", label=e.label, description=e.description)
    for r in relations or []:
        G.add_edge(r.source, r.target, label=r.label, description=r.description)
    for i in images or []:
        G.add_node(
            i.caption,
            type="image",
            label="image",
            path=i.path,
            description=i.description,
        )
    for ir in image_relations or []:
        G.add_edge(ir.source, ir.target, label=ir.label, description=ir.description)

    # Calculate layout
    pos = nx.spring_layout(G)

    # Create traces for nodes
    entity_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "entity"]
    image_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "image"]

    # Entity nodes trace
    entity_trace = go.Scatter(
        x=[pos[node][0] for node in entity_nodes],
        y=[pos[node][1] for node in entity_nodes],
        mode="markers+text",
        name="Entities",
        marker=dict(size=20, color="lightblue"),
        text=[node for node in entity_nodes],
        textposition="bottom center",
        hovertemplate=(
            "<b>Name:</b> %{text}<br>"
            "<b>Type:</b> %{customdata[0]}<br>"
            "<b>Description:</b>%{customdata[1]}<br>"
            "<extra></extra>"
        ),
        customdata=[
            [G.nodes[node]["label"], G.nodes[node]["description"]]
            for node in entity_nodes
        ],
    )

    # Image nodes trace
    image_trace = go.Scatter(
        x=[pos[node][0] for node in image_nodes],
        y=[pos[node][1] for node in image_nodes],
        mode="markers+text",
        name="Images",
        marker=dict(size=20, color="lightgreen", symbol="square"),
        text=[node.split(" ")[0] for node in image_nodes],
        textposition="bottom center",
        hovertemplate=(
            "<b>Caption:</b> %{text}<br>"
            "<b>Description:</b> %{customdata[1]}<br>"
            f"<img src='%{{customdata[0]}}' width=200><br>"
            "<extra></extra>"
        ),
        customdata=[
            [f"gradio_api/file={G.nodes[node]['path']}", G.nodes[node]["description"]]
            for node in image_nodes
        ],
    )

    # Edge trace
    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(G.edges[edge]["label"])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="text",
        text=edge_text,
        name="Relations",
    )

    # Handle empty graph case
    if not edge_x or not edge_y:
        x_range = [-1, 1]  # Default range when no edges
        y_range = [-1, 1]
    else:
        # Filter out None values before calculating min/max
        x_values = [x for x in edge_x if x is not None]
        y_values = [y for y in edge_y if y is not None]

        if x_values and y_values:
            x_range = [min(x_values) - 0.1, max(x_values) + 0.1]
            y_range = [min(y_values) - 0.1, max(y_values) + 0.1]
        else:
            x_range = [-1, 1]
            y_range = [-1, 1]

    # Create figure with updated ranges
    fig = go.Figure(
        data=[edge_trace, entity_trace, image_trace],
        layout=go.Layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, range=x_range
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, range=y_range
            ),
            dragmode="pan",  # Changed from 'drag' to 'pan'
            modebar=dict(
                add=[
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape",
                ]
            ),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        ),
    )

    # Add buttons for different interaction modes
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                x=0.1,
                y=1.1,
                xanchor="left",
                yanchor="top",
                pad={"r": 10, "t": 10},
                buttons=[
                    dict(label="Pan", method="relayout", args=[{"dragmode": "pan"}]),
                    dict(label="Zoom", method="relayout", args=[{"dragmode": "zoom"}]),
                    dict(
                        label="Select", method="relayout", args=[{"dragmode": "select"}]
                    ),
                ],
            )
        ]
    )

    return fig


with gr.Blocks() as index:
    gr.Markdown("## Construct Knowledge Graph")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Group():
                    select_database = gr.Dropdown(
                        label="Select Database",
                        interactive=True,
                        allow_custom_value=True,
                    )
                    update_database = gr.Button("List All Databases", interactive=True)
                    update_database.click(
                        fn=_list_all_knowledgws, outputs=[select_database]
                    )
                    select_database.change(_change_database, inputs=[select_database])

            files = gr.File(
                label="Index Graph",
                file_count="multiple",
                file_types=[".pdf", ".md", ".txt"],
                render=True,
                type="filepath",
            )
            input_examples = gr.Examples(
                examples=[
                    [["examples/rag/graphrag.pdf", "examples/rag/lightrag.pdf"]],
                    [["examples/rag/graphrag.md", "examples/rag/lightrag.md"]],
                ],
                inputs=[files],
            )
            gr.HTML(
                '<a href="https://arxiv.org/pdf/2410.05779" target="_blank">lightrag.pdf</a>'
                + "&nbsp;&nbsp;"
                + '<a href="https://arxiv.org/pdf/2404.16130" target="_blank">graphrag.pdf</a>'
                + "&nbsp;&nbsp;"
                + '<a href="https://github.com/wenzhaoabc/mmkg-rag/tree/main/examples/RAG" target="_blank">markdown files</a>'
            )
            with gr.Accordion(label="Index Configs", open=False):
                with gr.Row():
                    chunk_size = gr.Number(
                        value=8000,
                        label="Chunk Size",
                        minimum=1000,
                        maximum=10000,
                        step=1000,
                        interactive=True,
                    )
                    overlap = gr.Number(
                        value=400,
                        label="Overlap",
                        minimum=100,
                        maximum=1000,
                        step=100,
                        interactive=True,
                    )
                entity_labels = gr.Textbox(
                    placeholder="Expected entity labels, separated by comma",
                    label="Entity Labels",
                )
                relation_labels = gr.Textbox(
                    placeholder="Expected relation labels, separated by comma",
                    label="Relation Labels",
                )

            with gr.Accordion(label="LLM Configs", open=False):
                base_url = gr.Textbox(
                    placeholder="Type your LLM baseurl here", label="LLM Baseurl"
                )
                api_key = gr.Textbox(
                    placeholder="Type your API key here",
                    label="API Key",
                    type="password",
                )
                llm_model = gr.Dropdown(
                    ["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet"],
                    allow_custom_value=True,
                    label="LLM Model",
                    interactive=True,
                )

            button = gr.Button("Construct Graph", interactive=True)

        with gr.Column(scale=2):
            process_log = gr.Markdown("**Construct Log**\n\n")
            draw_graph = gr.Button("Visualize Graph", interactive=True)
            graph = gr.Plot(label="Graph")

    button.click(
        index_file_graph,
        inputs=[
            files,
            chunk_size,
            overlap,
            base_url,
            api_key,
            llm_model,
            entity_labels,
            relation_labels,
        ],
        outputs=[process_log],
    )

    draw_graph.click(
        draw_interactive_graph,
        inputs=[],
        outputs=[graph],
    )
