"""Check that the tutorial notebook is valid and its code at least compiles."""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "tutorial.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_the_notebook_has_a_valid_structure(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    assert {cell["cell_type"] for cell in notebook["cells"]} == {"markdown", "code"}

    for cell in notebook["cells"]:
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []             # no saved output: the file stays small and clean
            assert cell["execution_count"] is None


def test_the_notebook_starts_with_an_explanation(notebook):
    first = notebook["cells"][0]
    assert first["cell_type"] == "markdown"
    assert first["source"][0].startswith("# ")


def test_every_code_cell_is_valid_python(notebook):
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell {number}", "exec")


def test_the_notebook_covers_every_step(notebook):
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for function in (
        "load_csv", "clean_sales_data", "describe_numeric", "plot_monthly_revenue",
        "plot_bar", "plot_histogram", "plot_correlation_heatmap", "export_clean_data",
    ):
        assert function in text


def test_the_notebook_only_uses_functions_that_exist(notebook):
    """Run the imports of the setup cell for real, so a renamed function is caught."""
    setup = "".join(notebook["cells"][1]["source"])
    imports = [line for line in setup.splitlines() if line.startswith(("from dataviz", "import"))]
    namespace = {}
    exec("import sys\nfrom pathlib import Path\n" + "\n".join(_join_imports(setup)), namespace)
    assert imports


def _join_imports(setup):
    """Collect the import statements (including the multi-line ones) from the setup cell."""
    statements, current = [], []
    for line in setup.splitlines():
        if line.startswith(("from dataviz", "import pandas", "import matplotlib")):
            current = [line]
            if line.rstrip().endswith("("):
                continue
            statements.append("\n".join(current))
            current = []
        elif current:
            current.append(line)
            if line.strip().endswith(")"):
                statements.append("\n".join(current))
                current = []
    return statements