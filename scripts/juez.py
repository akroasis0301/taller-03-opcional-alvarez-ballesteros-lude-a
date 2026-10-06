"""Parte 4, opción E — un juez de otra familia, calibrado contra las comprobaciones determinísticas.

    uv run python scripts/juez.py --modelos                      # qué modelos sirve el Ollama de la H200
    uv run python scripts/juez.py \\
        --corrida completo  corridas/completo-r1:resultados/resultados_completo_r1.csv \\
        --corrida sin_grafo corridas/sin_grafo-r1:resultados/resultados_sin_grafo_r1.csv \\
        --salida resultados/juez

El solver corre con GLM (vLLM). El juez es de OTRA familia (gemma3, en el Ollama de la H200,
puerto 11434). Para cada entregable recibe el enunciado (la rúbrica) y el entregable, y contesta
las preguntas del campo «juez» del golden set: las mismas comprobaciones, SIN la cifra verdadera.
Después se compara su veredicto con el de evaluar_solver.py (resultados_*.csv): acuerdo, kappa de
Cohen, matriz de confusión, por tipo de comprobación, y costo (tokens y latencia).

La pregunta que responde: ¿se puede reemplazar el verificador determinístico por un LLM-juez?
Lo que importa no es solo el acuerdo total, sino DÓNDE falla: si aprueba cifras que el golden
rechaza (falsos positivos), un juez solo habría publicado cifras incorrectas.

Escribe: juez.csv (una fila por comprobación), traza_juez.jsonl (cada llamada) y juez.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "solver-v2"))

from solver.cliente_llm import extraer_json, sin_pensamiento  # noqa: E402
from solver.config import Config  # noqa: E402,F401  (carga el .env)

SISTEMA = """Eres un evaluador de entregables de una maestría en IA. Recibes el ENUNCIADO de la tarea
(la rúbrica) y el ENTREGABLE de un estudiante. Contesta cada pregunta con true o false y una razón
breve. No puedes ejecutar código: juzga con lo que ves y con tu conocimiento. Si una pregunta pide
saber si una cifra es correcta, decide si lo es. Devuelve SOLO un objeto JSON:
{"veredictos": {"<id>": {"ok": true|false, "razon": "una frase"}}}"""


# ============================================================================ Ollama
def _url(ruta: str) -> str:
    host = os.environ.get("H200_HOST", "172.28.230.10")
    puerto = os.environ.get("JUEZ_PUERTO", os.environ.get("H200_PUERTO_EMB", "11434"))
    return f"http://{host}:{puerto}/{ruta}"


def _pedir(ruta: str, cuerpo: dict | None = None, timeout: float = 600) -> dict:
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(_url(ruta), data=datos, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def modelos() -> list[str]:
    return [m["name"] for m in _pedir("api/tags", timeout=10).get("models", [])]


def elegir_modelo(pedido: str | None) -> str:
    servidos = modelos()
    if pedido:
        if pedido not in servidos:
            raise SystemExit(f"{pedido!r} no está en el Ollama: {servidos}")
        return pedido
    gemma = sorted(m for m in servidos if m.startswith("gemma3"))
    if not gemma:
        raise SystemExit(f"no hay gemma3 en el Ollama de la H200; modelos: {servidos}. Usa --modelo.")
    return gemma[-1]


def juzgar(modelo: str, enunciado: str, entregable: str, preguntas: dict[str, str]) -> tuple[dict, dict]:
    usuario = (f"ENUNCIADO:\n{enunciado[:9000]}\n\nENTREGABLE:\n{entregable[:14000]}\n\nPREGUNTAS:\n"
               + "\n".join(f"- {k}: {v}" for k, v in preguntas.items()))
    t0 = time.perf_counter()
    r = _pedir("v1/chat/completions", {"model": modelo, "temperature": 0,
                                        "response_format": {"type": "json_object"},
                                        "messages": [{"role": "system", "content": SISTEMA},
                                                     {"role": "user", "content": usuario}]})
    texto = sin_pensamiento(r["choices"][0]["message"].get("content") or "")
    uso = r.get("usage", {})
    meta = {"tokens_entrada": uso.get("prompt_tokens", 0), "tokens_salida": uso.get("completion_tokens", 0),
            "latencia_s": round(time.perf_counter() - t0, 2), "salida": texto}
    try:
        veredictos = extraer_json(texto).get("veredictos", {})
    except ValueError as err:
        meta["error"] = str(err)
        veredictos = {}
    return veredictos, meta


def normalizar_veredicto(v) -> tuple[bool | None, str, str]:
    """(ok, razón, formato) de un veredicto, sea cual sea el formato en que lo devolvió el juez.

    Se pide {"ok": bool, "razon": str}, pero gemma3 a veces devuelve solo el booleano ("B03": true),
    aun con temperature=0 y response_format=json. No se adivina: se registra en qué formato llegó.
      completo   {"ok": true, "razon": "…"}
      sin_razon  true / false, o el texto "true" / "false"; razón vacía
      faltante   ausente o cualquier otra cosa: sin veredicto (no entra al acuerdo)
    """
    if isinstance(v, dict) and isinstance(v.get("ok"), bool):
        return v["ok"], str(v.get("razon", "")), "completo"
    if isinstance(v, bool):
        return v, "", "sin_razon"
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true", "", "sin_razon"
    return None, "", "faltante"


# ============================================================================ métricas
def kappa(pares: list[tuple[int, int]]) -> float | None:
    """Kappa de Cohen entre el golden (a) y el juez (b), binario."""
    n = len(pares)
    if not n:
        return None
    po = sum(a == b for a, b in pares) / n
    pa, pb = sum(a for a, _ in pares) / n, sum(b for _, b in pares) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def resumen(filas: list[dict]) -> list[str]:
    pares = [(int(f["golden"]), int(f["juez"])) for f in filas if f["juez"] != ""]
    c = Counter(pares)
    k = kappa(pares)
    md = ["## Acuerdo global", "",
          f"- comprobaciones juzgadas: **{len(pares)}** (sin respuesta del juez: {sum(f['juez'] == '' for f in filas)})",
          f"- acuerdo: **{sum(a == b for a, b in pares) / max(len(pares), 1):.3f}** · kappa de Cohen: "
          f"**{'—' if k is None else f'{k:.3f}'}**", "",
          "| | juez: aprueba | juez: rechaza |", "|---|---|---|",
          f"| **golden: aprueba** | {c[(1, 1)]} | {c[(1, 0)]} |",
          f"| **golden: rechaza** | {c[(0, 1)]} (falso positivo del juez) | {c[(0, 0)]} |", ""]
    formatos = Counter(f.get("formato", "completo") for f in filas)
    md += [f"Formato de las respuestas del juez (se pidió `{{\"ok\", \"razon\"}}` en todas): "
           + ", ".join(f"{k} {formatos[k]}" for k in ("completo", "sin_razon", "faltante")) + ".", ""]
    por_tipo = defaultdict(list)
    for f in filas:
        if f["juez"] != "":
            por_tipo[f["tipo"]].append((int(f["golden"]), int(f["juez"])))
    md += ["## Por tipo de comprobación", "", "| tipo | n | acuerdo | kappa | falsos positivos del juez |",
           "|---|---|---|---|---|"]
    for tipo, ps in sorted(por_tipo.items()):
        kt = kappa(ps)
        md.append(f"| {tipo} | {len(ps)} | {sum(a == b for a, b in ps) / len(ps):.2f} | "
                  f"{'—' if kt is None else f'{kt:.2f}'} | {sum(a == 0 and b == 1 for a, b in ps)} |")
    fp = [f for f in filas if f["golden"] == "0" and f["juez"] == "1"]
    md += ["", "## Donde el juez aprueba lo que el golden rechaza", ""]
    md += [f"- {f['variante']} r{f['repeticion']} · {f['tarea']} {f['check']} ({f['tipo']}): golden «{f['detalle']}» · "
           f"juez «{f['razon']}»" for f in fp] or ["_Ninguno._"]
    return md


# ============================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrida", nargs="+", action="append", metavar=("VARIANTE", "CORRIDA"),
                    help="variante y corridas carpeta:resultados.csv (como tablas_2b.py)")
    ap.add_argument("--golden", default=str(RAIZ / "golden" / "golden_tareas.json"))
    ap.add_argument("--modelo")
    ap.add_argument("--modelos", action="store_true", help="lista los modelos del Ollama y sale")
    ap.add_argument("--salida", default=str(RAIZ / "resultados" / "juez"))
    a = ap.parse_args()
    if a.modelos:
        print("\n".join(modelos()))
        return 0
    if not a.corrida:
        ap.error("falta --corrida")

    import evaluar_solver as E  # el del kit: misma lectura de entregables que el golden

    golden = json.loads(Path(a.golden).read_text(encoding="utf-8"))
    base = Path(a.golden).resolve().parent
    tareas = {t["id"]: t for t in golden["tareas"]}
    modelo = elegir_modelo(a.modelo)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    traza = (salida / "traza_juez.jsonl").open("w", encoding="utf-8")
    print(f"Juez: {modelo} en {_url('')}")

    filas, tokens = [], Counter()
    for variante, *corridas in a.corrida:
        for rep, spec in enumerate(corridas, 1):
            carpeta, _, csv_ruta = spec.partition(":")
            golden_ok = defaultdict(dict)
            if csv_ruta:
                with open(csv_ruta, encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        golden_ok[r["tarea"]][r["check"]] = r
            for tid, t in tareas.items():
                salida_t = Path(carpeta) / f"tarea-{tid}"
                principal = E.entregable(salida_t, t["entregable"]) if salida_t.exists() else None
                preguntas = {c["id"]: c["juez"] for c in t["checks"] if c.get("juez") and c["id"] in golden_ok[tid]}
                if principal is None or not preguntas:
                    continue
                enunciado = E.texto_entrada((base / (t.get("entrada") or t["pdf"])).resolve())
                veredictos, meta = juzgar(modelo, enunciado, E.texto_de(principal), preguntas)
                tokens.update({k: meta[k] for k in ("tokens_entrada", "tokens_salida", "latencia_s")})
                traza.write(json.dumps({"variante": variante, "repeticion": rep, "tarea": tid, "modelo": modelo,
                                        **meta}, ensure_ascii=False) + "\n")
                traza.flush()
                for cid in preguntas:
                    ok, razon, formato = normalizar_veredicto(veredictos.get(cid))
                    filas.append({"variante": variante, "repeticion": rep, "tarea": tid, "check": cid,
                                  "tipo": golden_ok[tid][cid]["tipo"], "golden": golden_ok[tid][cid]["ok"],
                                  "juez": "" if ok is None else str(int(bool(ok))),
                                  "detalle": golden_ok[tid][cid]["detalle"], "formato": formato,
                                  "razon": razon[:300]})
                print(f"  {variante} r{rep} {tid}: {len(preguntas)} preguntas · {meta['latencia_s']} s · "
                      f"{meta['tokens_entrada']}+{meta['tokens_salida']} tokens{' · ERROR ' + meta['error'] if meta.get('error') else ''}")
    traza.close()
    if not filas:
        print("no hubo nada que juzgar (¿faltan los resultados_*.csv?)")
        return 1
    with (salida / "juez.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    md = ["# Parte 4 — Opción E: un juez de otra familia", "",
          f"Juez `{modelo}` (Ollama de la H200) contra el golden determinístico (`evaluar_solver.py`). "
          f"Generado por `scripts/juez.py` el {time.strftime('%Y-%m-%d')}.", "",
          f"Costo: {tokens['tokens_entrada']:.0f} tokens de entrada, {tokens['tokens_salida']:.0f} de salida, "
          f"{tokens['latencia_s']:.0f} s.", ""] + resumen(filas)
    (salida / "juez.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
