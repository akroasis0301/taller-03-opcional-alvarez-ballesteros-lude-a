# Interfaces internas del solver

El contrato entre la Parte 1 (baseline, P1) y el resto del grupo: la 2.b y la 2.c (P2),
y la Parte 3 y la 4 (P3). Si algo de aquí cambia, se avisa al grupo **antes** del commit.

## 1. Clases que importa el evaluador

`evaluar_solver.py` instancia `Solver()` **sin argumentos**, así que cada variante es su
propia clase:

| Clase | Qué cambia | Para |
|---|---|---|
| `solver.orquestador:Solver` | nada: el solver completo | 1, 2.b |
| `solver.orquestador:SolverSinGrafo` | `usar_grafo=False`: el investigador recibe los k fragmentos más similares | 2.b |
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
├── grafo.json, grafo.png  [2] esqueleto del enunciado (aristas depende_de en rojo)
├── plan.json              [3] el plan validado, en orden topológico
├── subtareas/T1/intento-N/{script.py, stdout.txt, stderr.txt, resultados.json, *.png, entrada/, data/}
├── figuras/               [8] las figuras aprobadas que cita el entregable
├── reporte.md             [8] el entregable (M1: Markdown; M2: PDF y notebook)
└── cache_solver/          lo que NO escribió una ejecución aprobada
    ├── enunciado_leido.json   [1] secciones, tablas, restricciones
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
