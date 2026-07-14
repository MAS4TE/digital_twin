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

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"

html_theme = "sphinx_rtd_theme"
