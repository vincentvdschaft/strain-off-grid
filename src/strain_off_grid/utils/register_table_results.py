import json


def register_table_result(
    phantom: str,
    location: str,
    method,
    mean: float,
    std: float,
    out: str = "out/table_results.json",
):
    """Register a result in a JSON file for later use in a LaTeX table."""
    try:
        with open(out, "r") as f:
            table_results = json.load(f)
    except FileNotFoundError:
        table_results = {}
    if phantom not in table_results:
        table_results[phantom] = {}
    if location not in table_results[phantom]:
        table_results[phantom][location] = {}
    if method not in table_results[phantom][location]:
        table_results[phantom][location][method] = {}
    table_results[phantom][location][method] = {"mean": mean, "std": std}
    with open(out, "w") as f:
        json.dump(table_results, f, indent=4)


def make_latex_table():
    pass
