"""[8] Redactor, segunda mitad: el entregable en el formato que pide el enunciado.

  - Markdown: tal cual.
  - PDF (Tarea C): Markdown → HTML (markdown-it) → PDF con PyMuPDF (fitz.Story), igual que
    solver-v2/generar_enunciados.py. El límite de páginas lo comprueba el código.
  - Notebook (Tarea B): una celda Markdown por parte (la explicación del redactor) seguida de
    la celda de código con el script APROBADO de esa parte, y el notebook se EJECUTA de
    principio a fin con nbconvert en un proceso con entorno vacío. Las cifras del notebook son
    salidas de celda: se calculan, no se escriben a mano.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from solver.agentes.ejecutor import COPIAR_DE_DEPENDENCIAS, entorno_vacio
from solver.agentes.lector import clave_de

CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.3; }
h1 { font-size: 14pt; margin-bottom: 4pt; }
h2 { font-size: 11.5pt; margin-top: 8pt; margin-bottom: 3pt; }
h3 { font-size: 10pt; margin-top: 6pt; margin-bottom: 2pt; }
p, li { margin-top: 2pt; margin-bottom: 3pt; text-align: justify; }
table { border-collapse: collapse; margin: 4pt 0; }
th, td { border: 0.5pt solid #888; padding: 1.5pt 3pt; font-size: 8.5pt; }
th { background-color: #eeeeee; }
code { font-family: monospace; font-size: 8.5pt; }
img { width: 300pt; }
"""


# ====================================================================== PDF
def md_a_pdf(md: str, destino: Path, base: Path) -> int:
    """Escribe el PDF y devuelve su número de páginas. Las imágenes se resuelven desde `base`."""
    import pymupdf as fitz
    from markdown_it import MarkdownIt

    html = MarkdownIt("commonmark").enable("table").render(md)
    historia = fitz.Story(html=html, user_css=CSS, archive=str(base))
    escritor = fitz.DocumentWriter(str(destino))
    a4 = fitz.paper_rect("a4")
    caja = a4 + (50, 50, -50, -50)
    mas = True
    while mas:
        dispositivo = escritor.begin_page(a4)
        mas, _ = historia.place(caja)
        historia.draw(dispositivo)
        escritor.end_page()
    escritor.close()
    with fitz.open(destino) as doc:
        return doc.page_count


# ====================================================================== notebook
def partes_markdown(md: str) -> list[tuple[str, str]]:
    """[(encabezado, texto)] partiendo por encabezados '#'/'##'. Lo previo va con encabezado ''."""
    bloques, actual, cuerpo = [], "", []
    for linea in md.splitlines():
        if re.match(r"^#{1,3}\s+\S", linea):
            if actual or any(x.strip() for x in cuerpo):
                bloques.append((actual, "\n".join(cuerpo).strip()))
            actual, cuerpo = linea.strip(), []
        else:
            cuerpo.append(linea)
    if actual or any(x.strip() for x in cuerpo):
        bloques.append((actual, "\n".join(cuerpo).strip()))
    return bloques


def _celda_md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto}


def _celda_codigo(codigo: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": codigo}


def construir_notebook(md: str, plan: list[dict], scripts: dict[str, str]) -> dict:
    """Cada encabezado «Parte N» del redactor va seguido del código APROBADO de las subtareas de
    cálculo que cubren esa parte (en el orden del plan). Si una subtarea no quedó asociada a
    ningún encabezado, su código va al final, para que ninguna cifra quede sin calcular."""
    por_clave: dict[str, list[str]] = {}
    for s in plan:
        if s.get("tipo") == "calculo" and s["id"] in scripts:
            primera = (s.get("secciones") or [""])[0]
            por_clave.setdefault(primera, []).append(s["id"])
    celdas, usadas = [], set()
    for encabezado, texto in partes_markdown(md):
        celdas.append(_celda_md(f"{encabezado}\n\n{texto}".strip()))
        clave = clave_de(encabezado.lstrip("#").strip()) if encabezado else ""
        for sid in por_clave.get(clave, []):
            if sid not in usadas:
                celdas.append(_celda_codigo(f"# {sid}: código aprobado por el crítico\n{scripts[sid]}"))
                usadas.add(sid)
    for s in plan:
        if s["id"] in scripts and s["id"] not in usadas and s.get("tipo") == "calculo":
            celdas.append(_celda_codigo(f"# {s['id']}: código aprobado por el crítico\n{scripts[s['id']]}"))
    return {"cells": celdas, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                          "language": "python"},
                                          "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def preparar_carpeta(carpeta: Path, aprobadas: dict[str, Path], datos: str | None) -> None:
    """El notebook corre en una sola carpeta: las celdas leen entrada/<id>/ como sus scripts."""
    carpeta.mkdir(parents=True, exist_ok=True)
    for sid, origen in aprobadas.items():
        destino = carpeta / "entrada" / sid
        destino.mkdir(parents=True, exist_ok=True)
        for f in Path(origen).iterdir():
            if f.is_file() and f.suffix in COPIAR_DE_DEPENDENCIAS and f.name not in {"stdout.txt", "stderr.txt"}:
                shutil.copy2(f, destino / f.name)
    if datos and Path(datos).is_dir():
        shutil.copytree(datos, carpeta / Path(datos).name, dirs_exist_ok=True)


def ejecutar_notebook(ruta: Path, timeout_s: int, tmp: Path) -> dict:
    """nbconvert --execute --inplace, en un proceso con entorno vacío y su propio grupo."""
    t0 = time.perf_counter()
    orden = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             f"--ExecutePreprocessor.timeout={timeout_s}", ruta.name]
    try:
        p = subprocess.run(orden, cwd=ruta.parent, env=entorno_vacio(ruta.parent, tmp),
                           capture_output=True, text=True, timeout=timeout_s * 3, start_new_session=True)
        rc, err = p.returncode, p.stderr[-3000:]
    except subprocess.TimeoutExpired:
        rc, err = None, f"timeout: el notebook superó {timeout_s * 3} s"
    errores = errores_notebook(ruta)
    return {"returncode": rc, "stderr": err, "errores": errores,
            "duracion_s": round(time.perf_counter() - t0, 2)}


def errores_notebook(ruta: Path) -> list[str]:
    """Lo mismo que mira el evaluador (ipynb_ejecutado): celdas sin ejecutar o con error."""
    nb = json.loads(Path(ruta).read_text(encoding="utf-8"))
    codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
    problemas = [f"celda {i}: {o.get('ename')}: {o.get('evalue')}"[:200]
                 for i, c in enumerate(codigo) for o in c.get("outputs", []) if o.get("output_type") == "error"]
    sin = sum(c.get("execution_count") is None for c in codigo)
    if sin:
        problemas.append(f"{sin} celdas de código sin ejecutar")
    if not codigo:
        problemas.append("el notebook no tiene celdas de código")
    return problemas


def texto_notebook(ruta: Path) -> str:
    """Markdown + salidas de celda: lo que el evaluador busca para cifras y procedencia."""
    nb = json.loads(Path(ruta).read_text(encoding="utf-8"))
    partes = []
    for c in nb["cells"]:
        fuente = "".join(c.get("source", ""))
        if c["cell_type"] == "markdown":
            partes.append(fuente)
        for o in c.get("outputs", []):
            partes.append("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", "")))
    return "\n".join(partes)
