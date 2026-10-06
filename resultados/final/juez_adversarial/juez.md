# Parte 4 — Opción E: un juez de otra familia

Juez `gemma3:27b` (Ollama de la H200) contra el golden determinístico (`evaluar_solver.py`). Generado por `scripts/juez.py` el 2026-10-05.

Costo: 2599 tokens de entrada, 308 de salida, 93 s.

## Acuerdo global

- comprobaciones juzgadas: **7** (sin respuesta del juez: 0)
- acuerdo: **0.857** · kappa de Cohen: **0.000**

| | juez: aprueba | juez: rechaza |
|---|---|---|
| **golden: aprueba** | 6 | 0 |
| **golden: rechaza** | 1 (falso positivo del juez) | 0 |

Formato de las respuestas del juez (se pidió `{"ok", "razon"}` en todas): completo 7, sin_razon 0, faltante 0.

## Por tipo de comprobación

| tipo | n | acuerdo | kappa | falsos positivos del juez |
|---|---|---|---|---|
| cifra | 3 | 0.67 | 0.00 | 1 |
| contiene | 1 | 1.00 | — | 0 |
| paginas_max | 1 | 1.00 | — | 0 |
| procedencia | 1 | 1.00 | — | 0 |
| secciones | 1 | 1.00 | — | 0 |

## Donde el juez aprueba lo que el golden rechaza

- adulterado r1 · C C06 (cifra): golden «verdad=0.7500 ±0.006; no hallada» · juez «El reporte indica un MRR de 0.8056 para LSA, lo cual es consistente con los resultados presentados en la tabla.»
