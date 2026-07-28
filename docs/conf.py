"""Sphinx configuration for the MAS4TE Bressoux Digital Twin documentation."""

project = "MAS4TE - Bressoux Digital Twin"
copyright = "2026, ULiège"
author = "ULiège"

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
]

# Generate anchors for h1-h3 headings, so cross-file links of the form
# `[text](page.md#some-heading)` resolve. Without this MyST only registers
# explicit `(label)=` targets, which are not reachable through that syntax --
# every such link across the docs was silently broken.
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"

html_theme = "sphinx_rtd_theme"
