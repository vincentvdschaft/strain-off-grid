import json
from pathlib import Path


def load_results(path):
    with open(path) as f:
        return json.load(f)


def result_cell(method_result):
    mean = method_result["mean"] * 100
    std = method_result["std"] * 100
    return f"${mean:.1f} \\pm {std:.1f}$"


def multirow(span, content):
    return f"\\multirow{{{span}}}{{*}}{{{content}}}"


def rows_in_phantom(locations):
    return sum(len(methods) for methods in locations.values())


def location_lines(location, methods, phantom_cell, phantom_span):
    span = len(methods)
    lines = []
    for index, (method, result) in enumerate(methods.items()):
        phantom_column = (
            multirow(phantom_span, phantom_cell) if phantom_cell and index == 0 else ""
        )
        location_column = multirow(span, location) if index == 0 else ""
        lines.append(
            f"{phantom_column} & {location_column} & {method} & {result_cell(result)}\\\\"
        )
    return lines


def phantom_lines(phantom, locations):
    span = rows_in_phantom(locations)
    lines = []
    for index, (location, methods) in enumerate(locations.items()):
        if index > 0:
            lines.append("\\cmidrule(l){2-4}")
        phantom_cell = phantom if index == 0 else None
        lines.extend(location_lines(location, methods, phantom_cell, span))
    return lines


def body_lines(results):
    lines = []
    for index, (phantom, locations) in enumerate(results.items()):
        if index > 0:
            lines.append("\\midrule")
        lines.extend(phantom_lines(phantom, locations))
    return lines


HEADER = "\\textbf{Phantom} & \\textbf{Location} & \\textbf{Method} & \\textbf{Error [100/s]}\\\\"


def build_document(results):
    lines = [
        "\\begin{table}",
        "\\centering",
        "\\caption{Mean strain rate error per phantom, location and method.}",
        "\\label{tab:ischemic_results}",
        "% requires \\usepackage{booktabs} and \\usepackage{multirow}",
        "\\begin{tabular}{llll}",
        "\\toprule",
        HEADER,
        "\\midrule",
        *body_lines(results),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def write_table(text, paths):
    for path in paths:
        path.write_text(text)


def main():
    results = load_results(Path("out/table_results.json"))
    text = build_document(results)
    miccai_figures_dir = Path.home() / "1-projects/papers/miccai/figures"
    write_table(
        text,
        (
            Path("out/table_results.tex"),
            miccai_figures_dir / "table.tex",
        ),
    )


if __name__ == "__main__":
    main()
