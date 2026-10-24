"""Resultados de los estudios en JSON para la web, leídos de estudios/resultados_*.md.

    .venv\\Scripts\\python -m estudios.web_resultados

Los números de la web salen de las mismas tablas que enlaza el README: nada se
copia a mano. Escribe web/data/results.json.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "web", "data", "results.json")
FILES = {"candles": "resultados_velas_regimenes.md", "comparison": "resultados_comparativa.md"}


def _cell(text: str):
    text = text.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def parse_markdown(md: str) -> dict:
    """Título, primera línea de contexto y, por cada sección `##`, sus notas y su primera tabla."""
    title, intro, sections, current = "", "", [], None
    for line in md.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            current = {"title": line[3:].strip(), "notes": [], "columns": [], "rows": []}
            sections.append(current)
        elif line.startswith("|"):
            if current is None or line.startswith("|---"):
                continue
            cells = [c for c in line.strip().strip("|").split("|")]
            if not current["columns"]:
                current["columns"] = [c.strip() for c in cells]
            elif len(cells) == len(current["columns"]):
                current["rows"].append([cells[0].strip(), *(_cell(c) for c in cells[1:])])
        elif line.strip():
            if current is None:
                intro = intro or line.strip()
            else:
                current["notes"].append(line.strip())
    return {"title": title, "intro": intro, "sections": sections}


def main() -> None:
    out = {}
    for key, name in FILES.items():
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            out[key] = {"source": f"estudios/{name}", **parse_markdown(f.read())}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("Guardado en", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
