"""Constants used throughout the repo."""
from pathlib import Path

# Define some Path objects to folders within the repo
# Path to the repo as a whole
DIR_REPO = Path(__file__).resolve().parent.parent.parent

# Path to the top-level data directories which contain the data
DIR_DATA = DIR_REPO / "data"
DIR_DATA_EXT = DIR_DATA / "external"
DIR_DATA_INT = DIR_DATA / "interim"
DIR_DATA_PROC = DIR_DATA / "processed"
DIR_DATA_RAW = DIR_DATA / "raw"

# Path to the notebook and report directories
DIR_NOTEBOOKS = DIR_REPO / "notebooks"
DIR_REPORTS = DIR_REPO / "reports"

# Path to paper references
DIR_REFERENCES = DIR_REPO / "references"

# Path to various locations within the documentation
DIR_DOCS = DIR_REPO / "docs"
