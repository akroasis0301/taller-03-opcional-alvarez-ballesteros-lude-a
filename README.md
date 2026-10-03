# Taller 03 v2 — Solver multiagente con GraphRAG

MMIA 6013 IA Generativa y Agentes · USFQ · Semana 3

Un solver que recibe el enunciado de una tarea en PDF, lo descompone, recupera contexto
con un grafo de conocimiento, escribe y ejecuta código en un sandbox, lo critica, reintenta
y redacta el entregable en el formato pedido (Markdown, PDF o notebook ejecutado).

## Instalación

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.12 (uv lo descarga si falta).

```bash
uv sync                      # crea .venv e instala todo lo fijado en uv.lock
uv sync --extra ocr          # opcional: OCR para PDF escaneados (docling, pesado)
cp .env.example .env         # y ajusta la ruta del LLM
uv run python scripts/verificar_entorno.py
```

`requirements.txt` (lo pide el enunciado) se genera desde el lock:

```bash
uv export --format requirements-txt --no-hashes --no-dev -o requirements.txt
```

## Arquitectura: ocho agentes, uno por responsabilidad

| # | Agente | Hace | Corrección |
|---|---|---|---|
| 1 | Lector | PDF → texto por página, con tablas | — |
| 2 | Indexador | grafo: secciones, `depende_de` por regla, entidades, comunidades | C2 |
| 3 | Planificador | subtareas en un DAG validado por código | C1 |
| 4 | Investigador | sección literal + dependencias + búsqueda local, con citas | C2 |
| 5 | Programador | un script por subtarea de cálculo | C4 |
| 6 | Ejecutor | corre el script en el sandbox (sin LLM) | C3 |
| 7 | Crítico | chequeos de código primero, LLM después | C4 |
| 8 | Redactor | entregable en el formato del enunciado + procedencia | C5 |

Transversal: traza en todo camino (C6) y frenos (Parte 3).

## Estructura

```
solver/
  estado.py  cliente_llm.py  traza.py  frenos.py  procedencia.py  orquestador.py
  agentes/   lector, indexador, planificador, investigador,
             programador, ejecutor, critico, redactor
  sandbox/   guarda.py (guarda estática sobre el AST)
parte0/      salidas de los tres scripts de la Parte 0
tareas/      practica/ (A, B, C del Lab-03) y reales/ (las del grupo)
corridas/    tarea-*/ con traza.jsonl, plan.json y el entregable
informe/     el PDF y sus figuras
scripts/     utilidades (verificar_entorno.py)
tests/       pruebas de la guarda, del plan y de la procedencia
```

## Contrato

```python
Solver().solve(ruta_pdf, carpeta_salida) -> dict   # status, entregables, subtareas, usage, model, trace
Solver().run(pregunta) -> dict                      # answer, trace, status, model, usage (Taller 4)
```

## Credenciales

Ninguna clave en el código, en las trazas, en los scripts generados ni en el repositorio.
Se leen de `.env`, que `.gitignore` excluye, y el sandbox corre con entorno vacío.
