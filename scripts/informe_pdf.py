"""Arma el informe: informe/informe.md → informe/informe.pdf, con las tablas generadas incluidas.

    uv run python scripts/informe_pdf.py              # → informe/informe_completo.md e informe/informe.pdf
    uv run python scripts/informe_pdf.py --revisar    # solo lista lo PENDIENTE y lo que falta incluir

En informe.md:
    <!-- incluir: resultados/final/tablas_2b/tablas.md -->      inserta un Markdown (baja sus títulos)
    <!-- incluir-codigo: resultados/Entregables_Parte_0/salida_0a.txt -->   lo inserta como bloque de código
    <!-- incluir-codigo: corridas/x/plan.json 1-40 -->           solo esas líneas
    **[PENDIENTE: …]**                                          lo que falta escribir: se lista al armar

Las tablas no se transcriben: se incluyen desde los archivos que generan los scripts, así cada
cifra del informe se puede reconstruir desde los CSV (lo exige el enunciado). Las imágenes van
con rutas relativas a la raíz del repo. El PDF se genera con el mismo md_a_pdf del solver.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

INCLUIR = re.compile(r"<!--\s*incluir(-codigo)?:\s*(\S+)(?:\s+(\d+)-(\d+))?\s*-->")
PENDIENTE = re.compile(r"\[PENDIENTE:[^\]]*\]")


def bajar_titulos(md: str, niveles: int = 2) -> str:
    fuera = []
    en_codigo = False
    for linea in md.splitlines():
        if linea.strip().startswith("```"):
            en_codigo = not en_codigo
        if not en_codigo and re.match(r"^#{1,6}\s", linea):
            linea = "#" * niveles + linea
        fuera.append(linea)
    return "\n".join(fuera)


def armar(md: str) -> tuple[str, list[str]]:
    faltan: list[str] = []

    def reemplazo(m: re.Match) -> str:
        codigo, ruta, desde, hasta = m.group(1), m.group(2), m.group(3), m.group(4)
        archivo = RAIZ / ruta
        if not archivo.exists():
            faltan.append(ruta)
            return f"**[PENDIENTE: falta `{ruta}` — se genera con los scripts de la sección Reproducibilidad]**"
        texto = archivo.read_text(encoding="utf-8", errors="replace")
        if desde:
            texto = "\n".join(texto.splitlines()[int(desde) - 1: int(hasta)])
        if codigo:
            return f"```\n{texto.rstrip()}\n```"
        return bajar_titulos(texto)
    return INCLUIR.sub(reemplazo, md), faltan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=str(RAIZ / "informe" / "informe.md"))
    ap.add_argument("--salida", default=str(RAIZ / "informe" / "informe.pdf"))
    ap.add_argument("--revisar", action="store_true")
    a = ap.parse_args()

    md, faltan = armar(Path(a.entrada).read_text(encoding="utf-8"))
    pendientes = PENDIENTE.findall(md)
    imagenes = [r for r in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", md) if not (RAIZ / r).exists()]
    for etiqueta, lista in [("archivos por incluir que no existen", faltan), ("imágenes que no existen", imagenes),
                            ("PENDIENTES", pendientes)]:
        print(f"{etiqueta}: {len(lista)}")
        for x in lista:
            print(f"   - {x[:150]}")
    if a.revisar:
        return 0
    completo = Path(a.salida).with_name("informe_completo.md")
    completo.write_text(md, encoding="utf-8")
    from solver import formatos
    formatos.CSS = formatos.CSS.replace("img { width: 300pt; }", "img { width: 440pt; }")
    paginas = formatos.md_a_pdf(md, Path(a.salida), RAIZ)
    print(f"→ {completo}\n→ {a.salida} ({paginas} páginas)")
    if pendientes or faltan or imagenes:
        print("AVISO: el PDF tiene partes pendientes; no es la versión de entrega.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
