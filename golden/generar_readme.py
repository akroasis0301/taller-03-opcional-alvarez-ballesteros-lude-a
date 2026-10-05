"""Regenera golden/README.md desde golden_tareas.json (la fuente es el JSON).

    uv run python golden/generar_readme.py
"""
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
g = json.loads((AQUI / "golden_tareas.json").read_text(encoding="utf-8"))
L = ["# Golden set del grupo (Parte 2.a)", "",
     "Generado por `golden/generar_readme.py` desde `golden_tareas.json`: cada comprobación lleva su `_por_que`.", "",
     "```bash", "# desde la RAÍZ del repo (las verdad_py de S1 y S2 importan golden/verdades_reales.py)",
     "uv run python golden/verdades_reales.py          # verdades de S1 y S2 y lo que dan las violaciones",
     "uv run python solver-v2/evaluar_solver.py --golden golden/golden_tareas.json --solo-evaluar corridas/completo-r2 --solo A B C",
     "```", "", g["_comentario"], ""]
for t in g["tareas"]:
    L += [f"## {t['id']} — `{t.get('pdf')}` → `{t['entregable']}` ({len(t['checks'])} comprobaciones)", ""]
    if t.get("_comentario"):
        L += [t["_comentario"], ""]
    L += ["| id | tipo | qué detecta |", "|---|---|---|"]
    L += [f"| {c['id']} | {c['tipo']} | {c['_por_que'].replace('|', '/')} |" for c in t["checks"]]
    L.append("")
L += ["## Validación sin gastar H200", "",
      "- A09 aprueba en las corridas con la curva estandarizada (0.976608 en el resultados.json de la T3 de "
      "`completo`, `completo-r1` y `completo-r2`) y **falla** en `corridas/prueba/tarea-a-generativo-discriminativo/` "
      "(la T3 sin `StandardScaler`, 2026-10-04): la falla que el kit no veía.",
      "- Las trampas de S2 se reproducen con `python golden/verdades_reales.py` (`s2_trampas`).",
      "- En la exploratoria (2026-10-05) la T6 de S2 calculó el intervalo correcto [−0.19375, 1.00625] pero la "
      "corrida agotó el presupuesto antes de aprobarla: S210 y S211 fallaron por presupuesto, no por la trampa."]
(AQUI / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("→ golden/README.md")
