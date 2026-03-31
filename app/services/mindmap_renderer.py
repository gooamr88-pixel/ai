"""
Ruya — Mindmap Image Renderer
================================
Takes a MindMapNode tree and renders it as a professional PNG using Graphviz.
Features:
  - Color-coded depth levels with rounded rectangle nodes
  - Arabic RTL text support via Noto Sans Arabic font
  - Upload to Supabase Storage or save locally
"""

import os
import uuid
import logging
import asyncio
from typing import Optional

import graphviz

from app.schemas.mindmap import MindMapNode
from app.core.config import settings
from app.core.database import supabase

logger = logging.getLogger(__name__)

# ── Color palette for depth levels ────────────────────────────────────────────
DEPTH_COLORS = [
    {"fill": "#1a237e", "font": "#ffffff", "border": "#0d1642"},  # Level 0: Deep indigo (root)
    {"fill": "#4a148c", "font": "#ffffff", "border": "#2a0a52"},  # Level 1: Deep purple
    {"fill": "#00695c", "font": "#ffffff", "border": "#003d33"},  # Level 2: Teal
    {"fill": "#e65100", "font": "#ffffff", "border": "#8c3100"},  # Level 3: Deep orange
    {"fill": "#37474f", "font": "#ffffff", "border": "#1c2529"},  # Level 4+: Blue grey
]

ARABIC_FONT = "Noto Sans Arabic"

MEDIA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "media", "mindmaps"
)
os.makedirs(MEDIA_DIR, exist_ok=True)


def _add_nodes(dot: graphviz.Digraph, node: MindMapNode, depth: int = 0):
    """Recursively add nodes and edges to the Graphviz graph."""
    colors = DEPTH_COLORS[min(depth, len(DEPTH_COLORS) - 1)]

    # Node styling based on depth
    if depth == 0:
        # Root node — larger, bold
        dot.node(
            node.id,
            label=node.label,
            shape="box",
            style="rounded,filled,bold",
            fillcolor=colors["fill"],
            fontcolor=colors["font"],
            color=colors["border"],
            fontsize="18",
            fontname=ARABIC_FONT,
            width="3",
            height="0.8",
            penwidth="3",
        )
    else:
        dot.node(
            node.id,
            label=node.label,
            shape="box",
            style="rounded,filled",
            fillcolor=colors["fill"],
            fontcolor=colors["font"],
            color=colors["border"],
            fontsize="13" if depth <= 2 else "11",
            fontname=ARABIC_FONT,
            width="2",
            height="0.5",
            penwidth="1.5",
        )

    # Add edges to children
    if node.children:
        for child in node.children:
            _add_nodes(dot, child, depth + 1)
            dot.edge(
                node.id,
                child.id,
                color="#90a4ae",
                penwidth="1.5",
                arrowsize="0.7",
            )


def _build_graph(root_node: MindMapNode) -> graphviz.Digraph:
    """Build a Graphviz Digraph from a MindMapNode tree."""
    dot = graphviz.Digraph(
        format="png",
        engine="dot",  # Best for hierarchical trees
        graph_attr={
            "rankdir": "TB",       # Top to bottom
            "bgcolor": "#fafafa",  # Light background
            "pad": "0.5",
            "nodesep": "0.6",
            "ranksep": "0.8",
            "dpi": "150",          # High quality
            "splines": "ortho",    # Clean right-angle edges
        },
        node_attr={
            "fontname": ARABIC_FONT,
        },
        edge_attr={
            "fontname": ARABIC_FONT,
        },
    )

    _add_nodes(dot, root_node)
    return dot


async def render_mindmap_image(root_node: MindMapNode) -> str:
    """
    Render a MindMapNode tree as a professional PNG image.
    Returns: public URL (Supabase) or local /media/ path.
    """
    logger.info("[MINDMAP_RENDER] Building Graphviz graph...")

    try:
        dot = _build_graph(root_node)

        # Render to temp file
        file_id = uuid.uuid4().hex
        temp_path = os.path.join(MEDIA_DIR, f"mindmap_{file_id}")

        # graphviz.render() is blocking, run in thread
        rendered_path = await asyncio.to_thread(
            dot.render, temp_path, cleanup=True
        )

        logger.info(f"[MINDMAP_RENDER] Rendered PNG: {rendered_path}")

        # Read the rendered PNG
        with open(rendered_path, "rb") as f:
            img_bytes = f.read()

        # Upload to Supabase
        if supabase:
            try:
                dest_name = f"mindmaps/mindmap_{file_id}.png"

                def _upload():
                    supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
                        path=dest_name,
                        file=img_bytes,
                        file_options={"content-type": "image/png"},
                    )
                    return supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(dest_name)

                public_url = await asyncio.wait_for(
                    asyncio.to_thread(_upload), timeout=30
                )
                logger.info(f"[MINDMAP_RENDER] ✓ Uploaded to Supabase: {public_url[:80]}")

                # Cleanup local file after successful upload
                if os.path.exists(rendered_path):
                    os.remove(rendered_path)

                return public_url

            except Exception as e:
                logger.error(f"[MINDMAP_RENDER] Supabase upload failed: {e}")

        # Fallback: serve from local /media/
        local_url = f"/media/mindmaps/mindmap_{file_id}.png"
        logger.info(f"[MINDMAP_RENDER] ✓ Serving locally: {local_url}")
        return local_url

    except Exception as e:
        logger.error(f"[MINDMAP_RENDER] Rendering failed: {e}", exc_info=True)
        raise RuntimeError(f"Mindmap rendering failed: {e}")
