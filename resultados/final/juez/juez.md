# Parte 4 — Opción E: un juez de otra familia

Juez `gemma3:27b` (Ollama de la H200) contra el golden determinístico (`evaluar_solver.py`). Generado por `scripts/juez.py` el 2026-10-05.

Costo: 36587 tokens de entrada, 3487 de salida, 1194 s.

## Acuerdo global

- comprobaciones juzgadas: **74** (sin respuesta del juez: 6)
- acuerdo: **0.986** · kappa de Cohen: **0.882**

| | juez: aprueba | juez: rechaza |
|---|---|---|
| **golden: aprueba** | 69 | 1 |
| **golden: rechaza** | 0 (falso positivo del juez) | 4 |

Formato de las respuestas del juez (se pidió `{"ok", "razon"}` en todas): completo 68, sin_razon 6, faltante 6.

## Por tipo de comprobación

| tipo | n | acuerdo | kappa | falsos positivos del juez |
|---|---|---|---|---|
| cifra | 20 | 1.00 | 1.00 | 0 |
| cifra_presente | 20 | 1.00 | — | 0 |
| contiene | 7 | 1.00 | — | 0 |
| paginas_max | 4 | 1.00 | — | 0 |
| palabras_max | 4 | 1.00 | — | 0 |
| procedencia | 11 | 0.91 | 0.00 | 0 |
| secciones | 8 | 1.00 | — | 0 |

## Donde el juez aprueba lo que el golden rechaza

_Ninguno._
