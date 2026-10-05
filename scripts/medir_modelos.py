"""Mide qué modelo conviene para DESARROLLAR: latencia real y si el script que escribe sirve.

    uv run python scripts/medir_modelos.py                                  # Ollama: 3 candidatos
    uv run python scripts/medir_modelos.py --modelos qwen3:32b gpt-oss:120b
    uv run python scripts/medir_modelos.py --puerto 12555                   # el vLLM (GLM), si volvió

Cada modelo recibe el MISMO prompt del programador (Tarea A, Parte 2) por el cliente del
solver, y su script pasa por el mismo ejecutor y el mismo crítico. Se mide la latencia, los
tokens que informa el servidor, el razonamiento (en caracteres: el Ollama no siempre lo cuenta
como tokens) y si el crítico aprobó. Las filas se AÑADEN a resultados/medicion_modelos.csv,
con fecha: la cifra que justifica el modelo de desarrollo sale de aquí, no de memoria.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

SUBTAREA = {"id": "T2", "tipo": "calculo", "secciones": ["Parte 2"], "depende_de": [],
            "objetivo": "Entrenar Naive Bayes gaussiano y regresión logística (atributos estandarizados, "
                        "escalador ajustado solo con el entrenamiento) y reportar accuracy y F1 macro en prueba.",
            "criterio": "accuracy y f1_macro de los dos modelos en resultados.json"}


def contexto() -> dict:
    md = (RAIZ / "solver-v2" / "enunciados" / "tarea-a-generativo-discriminativo.md").read_text(encoding="utf-8")
    texto = md.split("## Parte 3")[0]                    # preámbulo, Parte 1 y Parte 2
    return {"texto": texto, "previos": ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="*", default=["qwen3:32b", "gpt-oss:120b", "qwen3:8b"],
                    help="ids que sirve el endpoint (con --puerto 12555 se ignora: el vLLM sirve uno)")
    ap.add_argument("--puerto", help="12555 para el vLLM; por defecto el H200_PUERTO del .env")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    if a.puerto:
        os.environ["H200_PUERTO"] = a.puerto            # antes de importar h200.py

    from solver.agentes import critico, ejecutor, programador
    from solver.cliente_llm import cargar_h200
    from solver.config import Config
    from solver.traza import Traza
    from solver.cliente_llm import ClienteLLM

    carpeta = RAIZ / "corridas" / "medicion_modelos" / time.strftime("%Y%m%d-%H%M%S")
    traza = Traza(carpeta / "traza.jsonl")
    destino = RAIZ / "resultados" / "medicion_modelos.csv"
    destino.parent.mkdir(exist_ok=True)
    nuevo = not destino.exists()
    modelos = [None] if a.puerto == "12555" else a.modelos

    with destino.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fecha", "puerto", "modelo", "latencia_s", "tokens_entrada",
                                          "tokens_salida", "razonamiento_car", "script", "critico", "motivo"])
        if nuevo:
            w.writeheader()
        for m in modelos:
            fila = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "puerto": os.environ.get("H200_PUERTO", "")}
            try:
                h = cargar_h200(a.timeout, m)
                cli = ClienteLLM(Config(llm=h), traza)
                fila["modelo"] = h.modelo
                print(f"→ {h.modelo} …", flush=True)
                t0 = time.perf_counter()
                r = cli.chat("programador", programador.mensajes(SUBTAREA, contexto(), None, None),
                             subtarea=f"medicion-{h.modelo}")
                fila["latencia_s"] = round(time.perf_counter() - t0, 1)
                fila["tokens_entrada"] = r["uso"].get("prompt_tokens", 0)
                fila["tokens_salida"] = r["uso"].get("completion_tokens", 0)
                fila["razonamiento_car"] = len(r.get("razonamiento") or "")
                codigo = programador.extraer_codigo(r["contenido"])
                ej = ejecutor.ejecutar(codigo, carpeta / h.modelo.replace(":", "_").replace("/", "_"),
                                       timeout_s=120, tmp=carpeta / ".tmp")
                v = critico.comprobar(SUBTAREA, codigo, ej)
                fila["script"] = "corre" if ej["returncode"] == 0 else (ej["error"] or f"rc={ej['returncode']}")[:80]
                fila["critico"] = "aprobado" if v["aprobado"] else "rechazado"
                fila["motivo"] = "; ".join(v["motivos"])[:200]
            except Exception as err:
                fila.setdefault("modelo", m or "?")
                fila["motivo"] = f"{type(err).__name__}: {err}"[:200]
            w.writerow(fila)
            f.flush()
            print("  ", {k: v for k, v in fila.items() if k not in {"fecha", "puerto"}}, flush=True)
    print(f"\nfilas añadidas a {destino.relative_to(RAIZ)}; traza en {carpeta.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
