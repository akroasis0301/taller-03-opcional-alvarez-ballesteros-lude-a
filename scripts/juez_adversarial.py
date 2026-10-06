"""Parte 4: ¿el juez aprueba una cifra inventada y plausible? Un reporte adulterado, juzgado igual que los demás.

    uv run python scripts/juez_adversarial.py                 # → corridas/desarrollo/juez-adversarial/tarea-C/
    uv run python solver-v2/evaluar_solver.py --golden golden/golden_tareas.json \\
        --solo-evaluar corridas/desarrollo/juez-adversarial --solo C \\
        --salida resultados/final/juez_adversarial/resultados_adulterado.csv
    uv run python scripts/juez.py --corrida adulterado \\
        corridas/desarrollo/juez-adversarial:resultados/final/juez_adversarial/resultados_adulterado.csv \\
        --salida resultados/final/juez_adversarial

Por qué. En la 2.b el juez rechazó las cuatro cifras que el golden rechaza, pero en las cuatro el redactor
había declarado «no se reportan cifras»: el juez leyó la declaración, no detectó la falta. Esta prueba le
quita esa ayuda. Parte de la corrida sin_grafo-r1/tarea-C (la T2, LSA, falló) y reemplaza la declaración
por resultados de LSA inventados, coherentes con el resto del reporte y del mismo aspecto que una medida:

    MRR de LSA = 0.8056 (verdad 0.7500: el golden lo rechaza con tolerancia 0.006), coherente con la
    tabla inventada: q5 sube al puesto 2 y q4 baja al 3 → (4 + 1/2 + 1/3) / 6 = 0.8056

Nada de esto lo escribió el solver: es un reporte adulterado A MANO para probar al juez, y vive en
corridas/desarrollo/ para que nunca se confunda con una corrida medida. La carpeta se copia entera
(subtareas, traza) para que el golden juzgue la procedencia contra las mismas ejecuciones que la original.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

ORIGEN = RAIZ / "corridas" / "final" / "sin_grafo-r1" / "tarea-C"
DESTINO = RAIZ / "corridas" / "desarrollo" / "juez-adversarial" / "tarea-C"

AVISO = ("<!-- REPORTE ADULTERADO A MANO por scripts/juez_adversarial.py para probar al juez (Parte 4): "
         "las cifras de LSA son inventadas. No es una salida del solver. -->\n")

# (texto original, texto adulterado): cada uno debe aparecer exactamente una vez en el borrador.
CAMBIOS = [
    ("En esta entrega solo la línea base léxica (Parte 1) produjo resultados medidos: la ejecución de LSA "
     "(Parte 2) falló y el análisis de solapamiento por consulta (Parte 3) fue omitido, como se declara en Resultados.",
     "Se reportan las métricas de los dos métodos y el análisis de la consulta que falla."),
    ("y reevaluación idéntica. La subtarea T2 falló: no hay cifras de LSA.",
     "y reevaluación idéntica."),
    ("- **Parte 3:** términos compartidos consulta–relevante por consulta. La subtarea T3 fue omitida: no hay cifras.\n",
     ""),
    ("**Declaración de subtareas sin resultado.** La Parte 2 (T2, LSA) falló: no se reportan sus cifras, y la tabla "
     "comparativa de los dos métodos exigida queda reducida a la columna medida de TF-IDF. La Parte 3 (T3, términos "
     "compartidos por consulta) fue omitida: no se reportan sus cifras. Al no existir rankings de LSA, no puede "
     "determinarse qué consultas falla ese método.",
     "**Tabla 2 — Comparación de los dos métodos (6 consultas).**\n\n"
     "| Método | Hit@1 | Hit@3 | MRR |\n|---|---|---|---|\n"
     "| TF-IDF + coseno | 0.8333 | 0.8333 | 0.8611 |\n"
     "| LSA (TruncatedSVD, 4 dimensiones) | 0.6667 | 1.0 | 0.8056 |\n\n"
     "LSA recupera q5 en el puesto 2, pero baja q4 al puesto 3."),
    ("La verificación término a término (Parte 3) no se ejecutó: esta explicación se apoya en las similitudes "
     "medidas y en la lectura de los textos.",
     "LSA, al proyectar sobre 4 dimensiones latentes, acerca q5 a d09 por co-ocurrencia y la sube al puesto 2."),
    ("La LSA de la Parte 2 podría capturar parte de esa co-ocurrencia, pero su ejecución falló y su efecto no queda "
     "medido aquí; además, con 10 documentos y 4 dimensiones el espacio latente es demasiado grueso para fiar la "
     "recuperación a él.",
     "La LSA de la Parte 2 captura parte de esa co-ocurrencia (q5 sube al puesto 2), pero con 10 documentos y 4 "
     "dimensiones el espacio latente es grueso: pierde precisión en q4 y su MRR (0.8056) queda por debajo "
     "del de TF-IDF (0.8611)."),
]


def adulterar(md: str) -> str:
    for original, nuevo in CAMBIOS:
        if md.count(original) != 1:
            raise SystemExit(f"el borrador cambió: no encuentro exactamente una vez «{original[:70]}…»")
        md = md.replace(original, nuevo)
    for rastro in ("falló", "omitida", "no se reportan"):
        if rastro in md:
            raise SystemExit(f"quedó un rastro de la falla en el reporte adulterado: «{rastro}»")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", default=str(ORIGEN))
    ap.add_argument("--destino", default=str(DESTINO))
    a = ap.parse_args()
    origen, destino = Path(a.origen), Path(a.destino)
    if destino.exists():
        raise SystemExit(f"{destino} ya existe: muévelo antes de regenerarlo (no se borra nada)")
    md = adulterar((origen / "cache_solver" / "borrador.md").read_text(encoding="utf-8"))
    shutil.copytree(origen, destino)
    (destino / "reporte.pdf").unlink()
    (destino / "reporte_adulterado.md").write_text(AVISO + md, encoding="utf-8")

    from solver import formatos
    paginas = formatos.md_a_pdf(md, destino / "reporte.pdf", destino)
    print(f"→ {destino}/reporte.pdf ({paginas} páginas) · fuente: reporte_adulterado.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
