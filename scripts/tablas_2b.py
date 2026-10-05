"""Las tablas de la Parte 2.b, derivadas de los CSV crudos y de las trazas (nada a mano).

    uv run python scripts/tablas_2b.py \\
        --variante completo  corridas/completo-r1:resultados/resultados_completo_r1.csv \\
                             corridas/completo-r2:resultados/resultados_completo_r2.csv \\
        --variante sin_grafo corridas/sin_grafo-r1:resultados/resultados_sin_grafo_r1.csv \\
        --salida resultados/tablas_2b

Cada corrida es `carpeta[:resultados.csv]`: la carpeta con las `tarea-*/traza.jsonl` y, si se
evaluó, el `resultados_*.csv` de evaluar_solver.py. Con varias corridas por variante (las
repeticiones), las tablas dan la media y el rango.

Lo que pide el enunciado y de dónde sale:
  comprobaciones aprobadas, procedencia  → resultados_*.csv (evaluar_solver.py)
  status, subtareas fallidas, intentos,
  tokens por agente, duración            → traza.jsonl (el resumen_*.csv no trae tokens por agente)

Escribe en --salida: por_tarea.csv, tokens_por_agente.csv, comprobaciones.csv y tablas.md.
Los tokens del índice de las notas del curso (una vez, en caché) NO están en las trazas de las
tareas: se reportan aparte, desde cache/graphrag/notas-*/indice.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NO_LOGRADAS = {"fallida", "omitida", "pendiente"}


# ============================================================================ lectura
def leer_traza(ruta: Path) -> list[dict]:
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def de_la_traza(filas: list[dict]) -> dict:
    """Status, subtareas, intentos, tokens por agente y duración de una corrida de una tarea."""
    fin = next((f for f in reversed(filas) if f["tipo"] == "fin"), {})
    subtareas = fin.get("subtareas") or {}
    agentes: dict[str, dict] = defaultdict(lambda: {"llamadas": 0, "tokens_entrada": 0, "tokens_salida": 0,
                                                    "latencia_s": 0.0, "errores": 0})
    for f in filas:
        if f["tipo"] == "llamada":
            a = agentes[f["agente"]]
            a["llamadas"] += 1
            a["tokens_entrada"] += f.get("tokens_entrada") or 0
            a["tokens_salida"] += f.get("tokens_salida") or 0
            a["latencia_s"] += f.get("latencia_s") or 0
            a["errores"] += bool(f.get("error"))
    # Un intento de código es un script que se ejecutó, o un intento que no llegó a ejecutarse
    # (el programador falló, no hubo script, o era idéntico a uno rechazado).
    ejecuciones = sum(f["tipo"] == "ejecucion" and f.get("agente") == "ejecutor" and f.get("subtarea") != "notebook"
                      for f in filas)
    sin_ejecutar = sum(f["tipo"] == "decision" and f.get("decision") in {"llm_fallo", "sin_script", "script_repetido"}
                       for f in filas)
    rechazos = [f for f in filas if f["tipo"] == "decision" and f.get("decision") == "rechazado"]
    ts = [f["ts"] for f in filas if "ts" in f]
    return {
        "status": fin.get("status", "sin_fin"),
        "motivo": fin.get("motivo", ""),
        "subtareas": len(subtareas),
        "no_logradas": sum(s in NO_LOGRADAS for s in subtareas.values()),
        "detalle_subtareas": " ".join(f"{k}:{v}" for k, v in subtareas.items()),
        "intentos_codigo": ejecuciones + sin_ejecutar,
        "rechazos_codigo": sum(r.get("por") == "codigo" for r in rechazos),
        "rechazos_llm": sum(r.get("por") == "llm" for r in rechazos),
        "tokens_entrada": sum(a["tokens_entrada"] for a in agentes.values()),
        "tokens_salida": sum(a["tokens_salida"] for a in agentes.values()),
        "duracion_s": round(max(ts) - min(ts), 1) if ts else 0.0,
        "presupuesto_agotado": any(f.get("decision") == "presupuesto_agotado" for f in filas),
        "agentes": dict(agentes),
        "modelo": next((f.get("modelo") for f in filas if f["tipo"] == "llamada"), ""),
    }


def leer_resultados(ruta: Path | None) -> dict[str, list[dict]]:
    if not ruta:
        return {}
    por_tarea: dict[str, list[dict]] = defaultdict(list)
    with ruta.open(encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            por_tarea[fila["tarea"]].append(fila)
    return por_tarea


def procedencia(checks: list[dict]) -> str:
    for c in checks:
        if c["tipo"] == "procedencia":
            m = re.search(r"\(([\d.]+)\)", c["detalle"])
            return m.group(1) if m else ("1.00" if c["ok"] == "1" else "")
    return ""


# ============================================================================ tablas
def _media(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _num(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 or abs(x) >= 100 else f"{x:.1f}"


def _rango(xs: list[float]) -> str:
    """Media (mínimo–máximo); un solo valor si no varía. La media no se redondea a entero si es
    chica: 1.5 subtareas no logradas no es «2»."""
    if not xs:
        return "—"
    if len(xs) == 1 or min(xs) == max(xs):
        return _num(xs[0])
    return f"{_num(_media(xs))} ({_num(min(xs))}–{_num(max(xs))})"


def tabla_md(cabecera: list[str], filas: list[list]) -> str:
    lineas = ["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
    lineas += ["| " + " | ".join(str(x) for x in f) + " |" for f in filas]
    return "\n".join(lineas)


def indice_notas() -> str:
    filas = []
    for ruta in sorted((RAIZ / "cache" / "graphrag").glob("notas-*/indice.json")):
        d = json.loads(ruta.read_text(encoding="utf-8"))
        t = d.get("tokens", {})
        filas.append([d.get("clave"), d.get("modelo"), d.get("creado"), d.get("duracion_s"),
                      t.get("tokens_entrada", 0), t.get("tokens_salida", 0),
                      len(d.get("fragmentos", [])), len(d["fusion"]["entidades"]), len(d.get("comunidades", []))])
    if not filas:
        return "_No hay índice de notas en cache/graphrag/._"
    return tabla_md(["clave", "modelo", "creado", "duración (s)", "tokens entrada", "tokens salida",
                     "fragmentos", "entidades", "comunidades"], filas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variante", nargs="+", action="append", required=True, metavar=("NOMBRE", "CORRIDA"),
                    help="nombre y una o más corridas carpeta[:resultados.csv]")
    ap.add_argument("--salida", default="resultados/tablas_2b")
    a = ap.parse_args()

    por_tarea, por_agente, comprobaciones = [], [], []
    for nombre, *corridas in a.variante:
        if not corridas:
            ap.error(f"la variante {nombre} no tiene corridas")
        for rep, spec in enumerate(corridas, 1):
            carpeta, _, csv_ruta = spec.partition(":")
            carpeta = Path(carpeta)
            resultados = leer_resultados(Path(csv_ruta) if csv_ruta else None)
            trazas = sorted(carpeta.glob("tarea-*/traza.jsonl"))
            if not trazas:
                print(f"aviso: {carpeta} no tiene tarea-*/traza.jsonl", file=sys.stderr)
            for traza in trazas:
                tarea = traza.parent.name.removeprefix("tarea-")
                t = de_la_traza(leer_traza(traza))
                checks = resultados.get(tarea, [])
                fila = {"variante": nombre, "repeticion": rep, "corrida": str(carpeta), "tarea": tarea,
                        "aprobadas": sum(c["ok"] == "1" for c in checks) if checks else "",
                        "total": len(checks) if checks else "", "procedencia": procedencia(checks),
                        **{k: v for k, v in t.items() if k != "agentes"}}
                por_tarea.append(fila)
                for agente, v in sorted(t["agentes"].items()):
                    por_agente.append({"variante": nombre, "repeticion": rep, "tarea": tarea, "agente": agente,
                                       **{k: round(x, 1) if isinstance(x, float) else x for k, x in v.items()}})
                for c in checks:
                    comprobaciones.append({"variante": nombre, "repeticion": rep, "tarea": tarea, "check": c["check"],
                                           "tipo": c["tipo"], "ok": c["ok"], "detalle": c["detalle"]})

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    for nombre_csv, filas in [("por_tarea.csv", por_tarea), ("tokens_por_agente.csv", por_agente),
                              ("comprobaciones.csv", comprobaciones)]:
        if filas:
            with (salida / nombre_csv).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(filas[0]))
                w.writeheader()
                w.writerows(filas)

    variantes = list(dict.fromkeys(f["variante"] for f in por_tarea))
    tareas = sorted({f["tarea"] for f in por_tarea})
    md = ["# Parte 2.b — Solver completo contra la ablación", "",
          "Generado por `scripts/tablas_2b.py` desde los CSV de `evaluar_solver.py` y las trazas. "
          "Con repeticiones: media (mínimo–máximo).", "",
          "Corridas: " + "; ".join(f"**{v}** = " + ", ".join(sorted({f['corrida'] for f in por_tarea if f['variante'] == v}))
                                    for v in variantes), ""]

    # --- 1. Por tarea
    filas_md = []
    for tarea in tareas:
        for v in variantes:
            fs = [f for f in por_tarea if f["variante"] == v and f["tarea"] == tarea]
            if not fs:
                continue
            aprob = [f"{f['aprobadas']}/{f['total']}" if f["total"] != "" else "—" for f in fs]
            filas_md.append([tarea, v, " · ".join(aprob), " · ".join(f["procedencia"] or "—" for f in fs),
                             " · ".join(f["status"] for f in fs),
                             _rango([f["no_logradas"] for f in fs]), _rango([f["intentos_codigo"] for f in fs]),
                             _rango([f["tokens_entrada"] for f in fs]), _rango([f["tokens_salida"] for f in fs]),
                             _rango([f["duracion_s"] for f in fs])])
    md += ["## 1. Por tarea", "",
           tabla_md(["tarea", "variante", "comprobaciones (por repetición)", "procedencia", "status",
                     "subtareas no logradas", "intentos de código", "tokens entrada", "tokens salida",
                     "duración (s)"], filas_md), ""]

    # --- 2. Totales por variante (suma de tareas, promedio entre repeticiones)
    filas_md = []
    for v in variantes:
        reps = sorted({f["repeticion"] for f in por_tarea if f["variante"] == v})
        tot = defaultdict(list)
        for r in reps:
            fs = [f for f in por_tarea if f["variante"] == v and f["repeticion"] == r]
            evaluadas = [f for f in fs if f["total"] != ""]
            tot["aprobadas"].append(sum(f["aprobadas"] for f in evaluadas))
            tot["total"].append(sum(f["total"] for f in evaluadas))
            for k in ("no_logradas", "intentos_codigo", "tokens_entrada", "tokens_salida", "duracion_s"):
                tot[k].append(sum(f[k] for f in fs))
        filas_md.append([v, len(reps), " · ".join(f"{a}/{t}" if t else "—" for a, t in zip(tot["aprobadas"], tot["total"])),
                         _rango(tot["no_logradas"]), _rango(tot["intentos_codigo"]),
                         _rango(tot["tokens_entrada"]), _rango(tot["tokens_salida"]), _rango(tot["duracion_s"])])
    md += ["## 2. Totales por variante (todas las tareas)", "",
           tabla_md(["variante", "repeticiones", "comprobaciones", "subtareas no logradas", "intentos de código",
                     "tokens entrada", "tokens salida", "duración (s)"], filas_md), ""]

    # --- 3. Tokens por agente (suma de tareas, promedio entre repeticiones)
    agentes = sorted({f["agente"] for f in por_agente})
    filas_md = []
    for v in variantes:
        reps = sorted({f["repeticion"] for f in por_agente if f["variante"] == v})
        total_v = _media([sum(f["tokens_entrada"] + f["tokens_salida"] for f in por_agente
                              if f["variante"] == v and f["repeticion"] == r) for r in reps]) or 1
        for ag in agentes:
            por_rep = [[f for f in por_agente if f["variante"] == v and f["repeticion"] == r and f["agente"] == ag]
                       for r in reps]
            if not any(por_rep):
                continue
            ent = [sum(f["tokens_entrada"] for f in fs) for fs in por_rep]
            sal = [sum(f["tokens_salida"] for f in fs) for fs in por_rep]
            ll = [sum(f["llamadas"] for f in fs) for fs in por_rep]
            filas_md.append([v, ag, _rango(ll), _rango(ent), _rango(sal),
                             f"{100 * (_media(ent) + _media(sal)) / total_v:.0f} %"])
    md += ["## 3. Tokens por agente (todas las tareas)", "",
           tabla_md(["variante", "agente", "llamadas", "tokens entrada", "tokens salida", "% del total"], filas_md), "",
           "Los tokens de **salida** incluyen el razonamiento del modelo: el de la H200 razona siempre.", ""]

    # --- 4. Comprobaciones que fallan
    fallos = defaultdict(list)
    for c in comprobaciones:
        if c["ok"] != "1":
            fallos[(c["variante"], c["tarea"], c["check"], c["tipo"])].append(f"r{c['repeticion']}: {c['detalle']}")
    md += ["## 4. Comprobaciones que fallan (insumo de la 2.c)", ""]
    md += [tabla_md(["variante", "tarea", "check", "tipo", "en qué repeticiones"],
                    [[*k, " / ".join(v)] for k, v in sorted(fallos.items())]) if fallos else "_Ninguna._", ""]

    # --- 5. Índice de las notas (costo fijo)
    md += ["## 5. Costo fijo: el índice de las notas del curso", "",
           "Se construye una vez y queda en caché; no está en las trazas de las tareas.", "", indice_notas(), ""]

    (salida / "tablas.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n→ {salida}/tablas.md, por_tarea.csv, tokens_por_agente.csv, comprobaciones.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
