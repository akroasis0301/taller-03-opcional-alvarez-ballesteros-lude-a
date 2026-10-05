# Interfaces internas del solver

El contrato entre la Parte 1 (baseline, P1) y el resto del grupo: la 2.b y la 2.c (P2),
y la Parte 3 y la 4 (P3). Si algo de aquí cambia, se avisa al grupo **antes** del commit.

## 1. Clases que importa el evaluador

`evaluar_solver.py` instancia `Solver()` **sin argumentos**, así que cada variante es su
propia clase:

| Clase | Qué cambia | Para |
|---|---|---|
| `solver.orquestador:Solver` | nada: el solver completo | 1, 2.b |
| `solver.orquestador:SolverSinGrafo` | `usar_grafo=False`: el investigador recibe los k fragmentos más similares; no corre la capa 2 | 2.b |
| `solver.orquestador:SolverSinCritico` | `usar_critico=False`: un intento por script, sin comprobaciones | 2.b |

```bash
cd solver-v2 && uv run python evaluar_solver.py --solver solver.orquestador:Solver \
  --ruta .. --corridas ../corridas/completo --salida ../resultados/resultados_solver.csv
```

Contrato (Parte 1 del enunciado): `Solver().solve(ruta_pdf, carpeta_salida) -> dict` con
`status`, `entregables`, `subtareas` (`id`, `tipo`, `status`, `intentos`), `usage`
(`tokens_entrada`, `tokens_salida`), `model`, `trace`. Y `Solver().run(pregunta) -> dict`
con `answer`, `trace`, `status`, `model`, `usage` para el Taller 4.

## 2. `Config` (`solver/config.py`)

| Campo | Defecto | Quién lo usa |
|---|---|---|
| `presupuesto_tokens`, `reserva_redactor` | `.env` (300 000 / 30 000) | P3: freno 1 |
| `max_intentos` | 3 | P3: freno 2 |
| `timeout_s` | 120 | P3: freno 3 |
| `confirmar(motivo) -> bool` | niega siempre | P3: freno 4 y extensión D |
| `usar_grafo`, `usar_critico` | `True` | P2: ablaciones |
| `llm` | `None` (= la H200 real) | P3: modelo de guion |
| `capa2` | `SOLVER_CAPA2` (1) | apagarla abarata las pruebas de frenos |
| `notas`, `cache_graphrag` | `conocimiento/notas-teoricas`, `cache/graphrag` | capa 2 (sección 10) |
| `hilos_indexador` | 8 | llamadas en paralelo del indexador |
| `presupuesto_notas` | 2 000 000 | el índice de las notas, aparte del de la tarea |
| `embedder` | `None` (= bge-m3, con respaldo léxico) | pruebas sin red |

`Solver(cfg=Config(...))` acepta una configuración explícita para pruebas y frenos.

## 3. Modelo de guion (forzar frenos sin gastar)

```python
from solver.cliente_llm import LLMGuion
from solver.config import Config
llm = LLMGuion([
    '{"subtareas": [...]}',                 # str: el contenido
    {"contenido": "", "fin": "length"},     # dict: campos de la respuesta
    RuntimeError("sin VPN"),                # excepción: se lanza
])                                          # la última respuesta se repite
Solver(cfg=Config(llm=llm, presupuesto_tokens=5_000)).solve(pdf, salida)
```

`llm.llamadas` guarda lo que recibió el modelo en cada llamada.

## 4. Dónde salta cada freno (Parte 3)

| Freno | Punto | Evento en la traza |
|---|---|---|
| Presupuesto con reserva | `ClienteLLM.chat`, **antes** de cada llamada; el orquestador va a [8] | `decision: presupuesto_agotado` |
| Tope de intentos | lazo [5]→[6]→[7]; la subtarea queda `fallida` y la cola sigue | `decision: tope_intentos` |
| Repetición | [6]: hash del script; uno idéntico a uno rechazado no se ejecuta | `decision: script_repetido` |
| Tiempo máximo | [6]: `killpg` del grupo de procesos | `ejecucion` con `error: timeout` |
| Confirmación humana | antes de [6], si la guarda detecta descarga | `decision: confirmacion_red` |

Nota: el presupuesto se comprueba antes de cada llamada, así que puede excederse como
mucho en lo que cueste una llamada. La reserva del redactor absorbe ese exceso.

## 5. La traza (`traza.jsonl`, C6)

Una línea JSON por evento, escrita al momento (sobrevive a un crash). Todo texto pasa por
`enmascarar()`.

| `tipo` | Campos |
|---|---|
| `llamada` | `agente`, `subtarea`, `modelo`, `tokens_entrada`, `tokens_salida`, `latencia_s`, `fin`, `intento`, `max_tokens`, `error`, `entrada` (mensajes), `salida`, `razonamiento_caracteres` |
| `ejecucion` | `agente`, `subtarea`, `intento`, `returncode`, `duracion_s`, `archivos`, `error` |
| `decision` | `agente`, `subtarea`, `decision`, `motivo` |
| `error` | `agente`, `subtarea`, `error` |

Agentes: `lector`, `indexador`, `planificador`, `investigador`, `programador`, `ejecutor`,
`critico`, `redactor`, `procedencia`, `orquestador`.

Tokens por agente para la 2.b:

```python
import pandas as pd
from solver.traza import leer
df = pd.DataFrame(leer("corridas/completo/tarea-A/traza.jsonl"))
df[df.tipo == "llamada"].groupby("agente")[["tokens_entrada", "tokens_salida"]].sum()
```

## 6. Una corrida en disco

```
corridas/<variante>/tarea-X/
├── traza.jsonl            C6: todo evento, en todo camino
├── grafo.json, grafo.png  [2] esqueleto del enunciado (aristas depende_de en rojo) + capa 2 en el json
├── grafo_entidades.png    [2] capa 2: entidades de cada sección y cuáles se fusionaron con las notas
├── plan.json              [3] el plan validado, en orden topológico
├── subtareas/T1/intento-N/{script.py, stdout.txt, stderr.txt, resultados.json, *.png, entrada/, data/}
├── figuras/               [8] las figuras aprobadas que cita el entregable
├── reporte.md             [8] el entregable (M1: Markdown; M2: PDF y notebook)
└── cache_solver/          lo que NO escribió una ejecución aprobada
    ├── enunciado_leido.json   [1] secciones, tablas, restricciones
    ├── graphrag/entidades_enunciado.json  [2] entidades por sección
    ├── contextos/T1.md        [4] lo que recibió el programador (evidencia para la 2.c)
    ├── rechazados/T2/intento-1/  scripts que el crítico rechazó, con su salida
    └── tmp/                   cachés del sandbox (ignorado por git)
```

`cache_solver/` lleva «cache» en el nombre a propósito: `evaluar_solver.py` no cuenta como
respaldo de procedencia nada cuya ruta contenga «cache», «traza», «plan» o «grafo». Así un
intento rechazado (p. ej. la exactitud 1,000 de una fuga) no respalda una cifra del reporte.

## 7. Correr una tarea

```bash
uv run python scripts/correr.py solver-v2/enunciados/tarea-a-generativo-discriminativo.pdf
uv run python scripts/correr.py <pdf> --variante sin_grafo --salida corridas/sin_grafo/tarea-A
uv run python scripts/correr.py --diagrama > resultados/orquestador.mmd
```

## 8. Repeticiones y carpetas (para la 2.b)

- `Solver.solve()` **nunca mezcla corridas**: si la carpeta de salida ya tiene algo, la mueve a
  `<carpeta>.anterior-AAAAMMDD-HHMMSS` y empieza vacía (evento `corrida_anterior_apartada`).
  Si no, los artefactos aprobados de la corrida vieja respaldarían cifras de la nueva.
- `scripts/correr.py` ya crea una carpeta con fecha por corrida.
- Con `evaluar_solver.py`, usen un `--corridas` distinto por repetición
  (`../corridas/completo-r1`, `-r2`…) y un `--salida` distinto. La misma tarea con el mismo
  modelo dio resultados distintos en dos corridas (Tarea A, 2026-10-04): una sola corrida por
  variante no basta para comparar.

## 9. Quién rechazó (para la 2.c)

El evento `decision: rechazado` lleva `por`: `"codigo"` (guarda, código de salida, contrato,
NaN, figura, fugas, plausibilidad, script repetido) o `"llm"` (el crítico LLM, que lee el
enunciado completo). Un rechazo del LLM siempre cita una frase literal del enunciado; si la
cita no existe, el rechazo se descarta y queda `decision: rechazo_descartado`.

## 10. GraphRAG capa 2 (C2, `solver/graphrag.py`)

Va **encima** del esqueleto y nunca lo reemplaza: la sección literal y sus dependencias
llegan siempre al programador.

1. **Notas del curso** (`conocimiento/notas-teoricas/*.md`): se parten por `##`/`###` en
   31 fragmentos; el LLM extrae entidades (dataset, metodo, metrica, concepto, restriccion) y
   relaciones; Louvain arma comunidades y el LLM resume cada una. Se construye **una vez**
   (`uv run python scripts/indexar_notas.py`) y queda en `cache/graphrag/notas-<clave>/`
   con su propia traza (`traza_notas.jsonl`). La clave cambia si cambian las notas, el modelo
   o el prompt.
2. **Enunciado**: una extracción por sección, en paralelo, **con cargo a la tarea**. Las
   entidades se fusionan con las de las notas por nombre normalizado (sin acentos, artículos
   ni plural) y por alias («accuracy» = «Exactitud»).
3. **Investigador**: búsqueda local = semillas (entidades de las secciones de la subtarea + las
   más parecidas por embeddings) → vecinos → fragmentos de las notas; búsqueda global = los
   resúmenes de comunidad más parecidos. Cada fragmento lleva su cita
   (`[notas del curso — s2-rag-y-vector-search § 4. Chunking …]`, `comunidad 3`).

Eventos en la traza:

| Evento | Qué dice |
|---|---|
| `decision: indice_notas` | `cache: true/false`; si se construyó, tokens y duración (no se cobran a la tarea) |
| `decision: capa2` | entidades por tipo, cuántas se fusionaron con las notas y ejemplos |
| `evento: embeddings` | modelo (`bge-m3` o `lexico-…`), cuántos textos, latencia |
| `decision: grafo+capa2` | por subtarea: `citas`, `semillas`, `vecinos` |
| `decision: capa2_no_disponible` | la capa 2 falló y la corrida siguió con el esqueleto |
| `decision: embeddings_lexicos` | bge-m3 no respondió; se usó el respaldo léxico |

**Para la 2.b (P2):** los tokens del índice de las notas son un costo fijo que se paga una
vez; repórtenlo aparte (están en `indice.json`, campo `tokens`). Los tokens de extraer el
enunciado sí están en `resumen_solver.csv` del solver completo, y no en `sin_grafo`.
