"""Parte 3: los cuatro frenos, cada uno forzado en una corrida con su traza. Sin H200 ni red.

    uv run python scripts/forzar_frenos.py                 # los cuatro → corridas/frenos/<freno>/
    uv run python scripts/forzar_frenos.py --solo red

El enunciado lo permite («para forzarlos sin gastar, un modelo de guion sirve: el freno es código
tuyo»). El modelo de guion (LLMGuion) responde como los agentes; los frenos, el sandbox y la traza
son los del solver real. Cada escenario escribe su corrida completa (traza.jsonl, plan.json,
reporte.md) y al final corridas/frenos/README.md con los eventos que prueban cada freno.

  presupuesto   300 tokens por llamada de guion contra un presupuesto de 1 500 con 600 de reserva:
                el trabajo se detiene ANTES de la llamada que lo excedería y el redactor entrega con
                la reserva, declarando lo que no se hizo.
  repeticion    el programador devuelve SIEMPRE el mismo script con fuga (0.c): el intento 1 se
                rechaza, el 2 y el 3 son idénticos y no se ejecutan; al tope la subtarea queda
                fallida y la cola sigue con la siguiente.
  timeout       un `while True` con 3 s de límite: se mata el grupo de procesos (killpg); el
                intento siguiente, correcto, se aprueba.
  red           un script con fetch_openml: sin una persona que confirme, no corre. Con
                --aprobar-red se simula la aprobación (se registra igual, pero sin red el script
                fallaría: no lo usamos en la evidencia).
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from solver.cliente_llm import LLMGuion  # noqa: E402
from solver.config import Config  # noqa: E402
from solver.orquestador import Solver  # noqa: E402
from solver.traza import leer  # noqa: E402

ENUNCIADO = textwrap.dedent("""\
    # Tarea F — los frenos del solver
    **Entrega:** un reporte en Markdown (`reporte.md`).
    ## Parte 1 — Datos
    Carga el conjunto breast_cancer de scikit-learn y reporta cuántos ejemplos tiene.
    ## Parte 2 — Modelo
    Con los datos de la Parte 1, divide 70/30 con random_state=0, entrena un árbol de decisión
    y reporta su exactitud sobre el conjunto de prueba.
    ## El reporte
    Un reporte.md con estas secciones, en este orden: Resultados y Discusión. Máximo 300 palabras.
    """)

PLAN = {"subtareas": [
    {"id": "T1", "tipo": "calculo", "secciones": ["Parte 1"], "depende_de": [],
     "objetivo": "cargar breast_cancer y contar ejemplos", "criterio": "n_ejemplos en resultados.json"},
    {"id": "T2", "tipo": "calculo", "secciones": ["Parte 2"], "depende_de": ["T1"],
     "objetivo": "árbol de decisión 70/30, exactitud en prueba", "criterio": "accuracy en resultados.json"}]}

T1 = textwrap.dedent("""\
    import json
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    json.dump({"n_ejemplos": int(X.shape[0])}, open("resultados.json", "w"))
    print("n_ejemplos", X.shape[0])
    """)
T2_BIEN = textwrap.dedent("""\
    import json
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    m = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    json.dump({"accuracy": float(m.score(X_te, y_te))}, open("resultados.json", "w"))
    print("accuracy", m.score(X_te, y_te))
    """)
T2_FUGA = textwrap.dedent("""\
    import json
    from sklearn.datasets import load_breast_cancer
    from sklearn.tree import DecisionTreeClassifier
    X, y = load_breast_cancer(return_X_y=True)
    m = DecisionTreeClassifier(random_state=0).fit(X, y)
    json.dump({"accuracy": float(m.score(X, y))}, open("resultados.json", "w"))
    """)
T2_BUCLE = "import time\nwhile True:\n    time.sleep(0.1)\n"
T2_RED = textwrap.dedent("""\
    import json
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("breast-w", version=1, return_X_y=True, as_frame=False)
    json.dump({"n": int(X.shape[0])}, open("resultados.json", "w"))
    """)
REPORTE = ("## Resultados\nEl conjunto tiene 569 ejemplos. {t2}\n\n"
           "## Discusión\nCorrida de demostración de un freno del solver.\n")


def guion(scripts_t2: list[str], reporte_t2: str):
    """Responde por agente; el programador de la T2 sigue la lista (el último se repite)."""
    estado = {"t2": 0}

    def responder(m):
        sistema, usuario = m[0]["content"], m[-1]["content"]
        if "planificador" in sistema:
            return json.dumps(PLAN)
        if "programador" in sistema:
            if "SUBTAREA T1" in m[1]["content"]:
                return f"```python\n{T1}```"
            i = min(estado["t2"], len(scripts_t2) - 1)
            estado["t2"] += 1
            return f"```python\n{scripts_t2[i]}```"
        if "crítico" in sistema:
            return json.dumps({"aprobado": True, "problemas": []})
        if "redactor" in sistema:
            texto = REPORTE.format(t2=reporte_t2)
            if "DEBES DECLARAR EN EL DOCUMENTO:" in usuario:     # un redactor que obedece la instrucción
                texto += "\n" + usuario.split("DEBES DECLARAR EN EL DOCUMENTO:")[1].strip().replace("- ", "*") + "*\n"
            return texto
        return "{}"
    return responder


def escenarios(aprobar_red: bool) -> dict[str, tuple[Config, list[str]]]:
    base = dict(capa2=False, embedder=None)   # los frenos no dependen del GraphRAG: corridas mínimas
    no_hecho = "La exactitud de la Parte 2 no se obtuvo: ver la traza."
    return {
        "presupuesto": (Config(llm=LLMGuion([guion([T2_BIEN], no_hecho)], tokens_entrada=200, tokens_salida=100),
                               presupuesto_tokens=1_500, reserva_redactor=600, **base),
                        ["presupuesto_agotado"]),
        "repeticion": (Config(llm=LLMGuion([guion([T2_FUGA], no_hecho)]), max_intentos=3, **base),
                       ["rechazado", "script_repetido", "tope_intentos"]),
        "timeout": (Config(llm=LLMGuion([guion([T2_BUCLE, T2_BIEN], "Exactitud medida en la T2.")]),
                           timeout_s=3, **base),
                    ["rechazado"]),
        "red": (Config(llm=LLMGuion([guion([T2_RED], no_hecho)]), max_intentos=1,
                       confirmar=(lambda motivo: True) if aprobar_red else (lambda motivo: False), **base),
                ["confirmacion_red"]),
    }


def evidencia(filas: list[dict], claves: list[str]) -> list[str]:
    salida = []
    for f in filas:
        if f["tipo"] == "decision" and f.get("decision") in set(claves) | {"tope_intentos", "presupuesto_agotado"}:
            extra = f" (aprobado={f['aprobado']})" if "aprobado" in f else ""
            salida.append(f"- `#{f['seq']}` {f['agente']} → **{f['decision']}**{extra} [{f.get('subtarea') or ''}]: "
                          f"{str(f.get('motivo', ''))[:220]}")
        if f["tipo"] == "llamada" and f["agente"] == "redactor":
            usuario = (f.get("entrada") or [{}])[-1].get("content", "")
            nota = usuario.split("DEBES DECLARAR EN EL DOCUMENTO:")[1].strip()[:200] if "DEBES DECLARAR" in usuario else ""
            salida.append(f"- `#{f['seq']}` redactor redacta ({f['tokens_entrada']}+{f['tokens_salida']} tokens)"
                          + (f"; recibió la orden de declarar: «{nota}»" if nota else ""))
        if f["tipo"] == "ejecucion" and f.get("error"):
            salida.append(f"- `#{f['seq']}` ejecutor, intento {f.get('intento')} [{f.get('subtarea')}]: "
                          f"returncode={f.get('returncode')}, {f.get('duracion_s')} s, error: {f['error'][:160]}")
    return salida


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", nargs="*", choices=["presupuesto", "repeticion", "timeout", "red"])
    ap.add_argument("--aprobar-red", action="store_true")
    ap.add_argument("--salida", default=str(RAIZ / "corridas" / "frenos"))
    a = ap.parse_args()

    raiz = Path(a.salida)
    entrada = raiz / "_enunciado"
    entrada.mkdir(parents=True, exist_ok=True)
    (entrada / "enunciado.md").write_text(ENUNCIADO, encoding="utf-8")
    md = ["# Parte 3 — Los cuatro frenos, forzados", "",
          "Generado por `scripts/forzar_frenos.py` con un modelo de guion (sin H200): los frenos, el sandbox "
          "y la traza son los del solver real. Cada carpeta tiene su `traza.jsonl` completa.", ""]
    ok_total = True
    for nombre, (cfg, esperados) in escenarios(a.aprobar_red).items():
        if a.solo and nombre not in a.solo:
            continue
        r = Solver(cfg).solve(str(entrada), str(raiz / nombre))
        filas = leer(r["trace"])
        decisiones = {f.get("decision") for f in filas}
        disparado = all(e in decisiones for e in esperados) and (
            nombre != "timeout" or any(f["tipo"] == "ejecucion" and "timeout" in (f.get("error") or "") for f in filas))
        ok_total &= disparado
        estados = {s["id"]: f"{s['status']} ({s['intentos']} intentos)" for s in r["subtareas"]}
        print(f"{nombre:<12} {'DISPARADO' if disparado else 'NO SE DISPARÓ'}  status={r['status']}  {estados}")
        md += [f"## {nombre} — {'disparado' if disparado else 'NO se disparó'}", "",
               f"`corridas/frenos/{nombre}/` · status **{r['status']}** · subtareas {estados} · "
               f"entregables {[Path(e).name for e in r['entregables']]} · tokens {r['usage']}", ""]
        md += evidencia(filas, esperados) + [""]
    (raiz / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"→ {raiz}/README.md")
    return 0 if ok_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
