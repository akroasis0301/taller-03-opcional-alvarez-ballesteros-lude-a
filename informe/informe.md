# Taller 03 v2 — Un solver multiagente con GraphRAG

MMIA 6013 IA Generativa y Agentes · USFQ · Jessica Ballesteros, Darlyn Ludeña, Miguel Álvarez

**Ruta, modelo y fecha.** H200 de la USFQ por VPN (GlobalProtect). LLM: vLLM en `172.28.230.10:12555`,
id `zai-org/GLM-5.3-Flash`, leído de `/v1/models` (no escrito a mano). Embeddings: `bge-m3:latest` en el
Ollama del mismo servidor (puerto 11434). Corridas finales: **[PENDIENTE: fechas de corridas/final]**. Todas las
corridas de las Partes 1 y 2 usan el mismo modelo.

**Autoría y uso de asistentes.** **[PENDIENTE: quién hizo cada parte y cómo se usó un asistente de IA en el
desarrollo; la Parte 5 pide declararlo con la traza como evidencia]**

**Cómo leer este informe.** Cada tabla se genera desde los CSV crudos y las trazas con los scripts de la
sección 9; ninguna cifra se transcribió a mano.

---

## 1. Parte 0 — Tres fallas que no fallan

### 0.a — El solver que no ejecuta

<!-- incluir-codigo: resultados/Entregables_Parte_0/salida_0a.txt -->

Ruta: H200 (vLLM 12555), modelo `zai-org/GLM-5.3-Flash`, 2026-10-03, 12 028 tokens de salida, 61,72 s.

1. **Por qué que las cifras estén cerca no las hace aceptables.** El reporte dice 0,9415 para Naive Bayes
   (medida: 0,9474) y 0,9825 para la regresión logística (medida: 0,9883): están cerca, pero ninguna salió de
   una ejecución (0 de 24 cifras con respaldo). «Cerca» solo se sabe porque esta vez alguien ejecutó el
   experimento al lado; sin esa ejecución no hay forma de distinguir una cifra plausible de una correcta, y el
   reporte no declara que sean estimaciones. Una cifra sin procedencia no es un resultado aunque acierte.
2. **Qué cifra habría cambiado una conclusión.** La de la curva con pocos datos: el reporte da, con el 5 %
   del entrenamiento, Naive Bayes 0,9123 contra regresión logística 0,9064, y de ahí concluye que con pocos
   ejemplos conviene el generativo (el cruce de Ng y Jordan). La diferencia es **un acierto sobre 171**: con
   un acierto más para la regresión logística la conclusión se invierte, y ninguna de las dos cifras tiene una
   ejecución detrás (medido con el pipeline estandarizado: 0,9415 y 0,9298). En la corrida del enunciado
   (2026-10-02) el modelo llegó a inventar 0,7251 para la regresión logística: un cruce exagerado que
   confirmaba la tesis.
3. **Qué tendría que comprobar un verificador sin conocer las respuestas.** Que cada cifra del reporte
   aparezca en algo que escribió una ejecución (un `resultados.json` de un script que terminó bien) o en el
   enunciado, y devolver por su nombre las que no: es la comprobación de procedencia (corrección C5).

### 0.b — El RAG plano que pierde la referencia

<!-- incluir-codigo: resultados/Entregables_Parte_0/salida_0b.txt -->

1. **Es una falla de recuperación.** Los datos y la partición están escritos en la Parte 1; el RAG plano trae
   la Parte 4 (0,293) y la Parte 3 (0,243) y cubre 0 de 3 hechos. Al seguir la arista `depende_de`
   Parte 3 → Parte 1 cubre 3 de 3 sin tocar el generador: el generador no puede usar lo que no recibió.
2. **Qué haría un LLM con el primer contexto.** Rellenar el hueco: suponer un dataset y una partición
   razonables y presentarlos con la misma seguridad que un dato del enunciado, sin avisar que faltaban.
3. **Por qué la arista se extrae con una regla.** «La misma división de la Parte 1» es un patrón literal:
   una expresión regular lo encuentra siempre, igual en cada corrida y sin tokens. Un LLM puede omitir la
   arista o inventar otra, y si falla vuelve la misma pérdida de contexto sin ningún aviso. El LLM se reserva
   para lo que una regla no puede hacer (entidades y relaciones, capa 2).

### 0.c — Un `returncode == 0` no es un experimento

<!-- incluir-codigo: resultados/Entregables_Parte_0/salida_0c.txt -->

1. **La fuga que la comprobación estática no ve.** Un escalador ajustado con todos los datos antes de dividir
   (`StandardScaler().fit_transform(X)` y después `train_test_split`): el modelo se ajusta con `X_tr` y se
   evalúa con `X_te`, nombres distintos, y la exactitud (≈0,98) no llega al umbral de plausibilidad. Lo mismo
   una imputación o una selección de atributos ajustadas con todo. Nuestro crítico agrega una comprobación
   para el caso del escalador (`ajuste_antes_de_dividir`, `solver/agentes/critico.py`).
2. **Por qué la decisión final puede quedar en un LLM crítico.** Porque las comprobaciones de código van
   antes y no se pueden persuadir: el LLM solo ve lo que ya pasó los filtros, puede agregar rechazos pero no
   deshacer uno del código. En nuestro solver además un rechazo del LLM solo cuenta si cita una frase literal
   del enunciado; si la cita no existe, se descarta (`rechazo_descartado`).

---

## 2. Parte 1 — El solver

### 2.1 Arquitectura: ocho agentes, un orquestador

![Orquestador](resultados/orquestador.png)

| # | Agente | Hace | Lo comprueba el código |
|---|---|---|---|
| 1 | Lector (`agentes/lector.py`) | PDF → secciones por tipografía, tablas, restricciones (formato, límites, secciones, archivos exigidos) | ligaduras, PDF escaneado, límites de una sola parte |
| 2 | Indexador (`agentes/indexador.py`, `graphrag.py`) | esqueleto por regla + capa 2 (entidades, notas, embeddings, comunidades) | aristas `depende_de` por regla |
| 3 | Planificador (`agentes/planificador.py`) | subtareas con tipo, dependencias y criterio | DAG, ids, cobertura, coherencia con el grafo |
| 4 | Investigador (`agentes/investigador.py`) | contexto de cada subtarea, con citas | la sección literal y sus dependencias van siempre |
| 5 | Programador (`agentes/programador.py`) | un script por subtarea de cálculo | la guarda del sandbox |
| 6 | Ejecutor (`agentes/ejecutor.py`) | corre el script en el sandbox | — (es código) |
| 7 | Crítico (`agentes/critico.py`) | aprueba o devuelve con la corrección | código de salida, contrato, NaN, figuras, fugas, plausibilidad |
| 8 | Redactor (`agentes/redactor.py`, `formatos.py`) | Markdown, PDF con límite de páginas o notebook ejecutado | secciones, extensión, procedencia |

El orquestador (`solver/orquestador.py`) es un `StateGraph` de LangGraph 1.2.12. El ciclo
programar → ejecutar → criticar → programar sale por aprobación o por tope de intentos; la salida a redactar
es por fin de la cola o por presupuesto. Contrato: `Solver().solve(ruta_pdf, carpeta_salida)` y
`Solver().run(pregunta)` para el Taller 4.

### 2.2 Las seis correcciones, señaladas en el código

| | Corrección | Dónde | Evidencia |
|---|---|---|---|
| C1 | El plan es un DAG validado por código | `planificador.validar()` (ids, dependencias, ciclos, cobertura, aristas del grafo) y `ordenar()`; `_tras_planificar` devuelve el plan inválido con sus problemas (máx. 3) | traza: `plan_invalido` / `plan_valido` |
| C2 | GraphRAG con esqueleto determinístico | capa 1: `indexador.esqueleto()` (referencia, «la pregunta anterior», anáfora «los dos modelos»); capa 2: `graphrag.py` (entidades del LLM, fusión por nombre normalizado con las notas del curso, bge-m3, Louvain con resumen); `graphrag.busqueda_local()` con citas | `grafo.png`, `grafo_entidades.png`, traza: `capa2`, `grafo+capa2` |
| C3 | Sandbox | `sandbox/guarda.py` (AST: red, procesos, borrado, `eval`, rutas fuera, URLs como descarga); `ejecutor.py`: entorno vacío, carpeta propia, `start_new_session` + `killpg` al vencer el tiempo | Parte 3 |
| C4 | Contrato de resultados y crítico que mira antes | `resultados.json` obligatorio (prompt del programador); `critico.comprobar()` primero, `revisar_con_llm()` solo si pasan | traza: `rechazado` con `por: codigo` o `por: llm` |
| C5 | Procedencia | `procedencia.verificar()` en `_verificar`: las cifras sin origen vuelven al redactor por su nombre | traza: `devuelto_al_redactor` |
| C6 | Traza en todo camino | `traza.py` (una línea por evento, escrita al momento, con secretos enmascarados); `_nodo` atrapa todo error y lo registra; toda corrida termina en `cerrar` con `fin` | `traza.jsonl` de cada corrida |

### 2.3 El grafo del enunciado de la Tarea A

![Grafo de la Tarea A](corridas/final/completo-r1/tarea-A/grafo.png)

Las aristas `depende_de` (en rojo) salen de reglas: «la misma división de la Parte 1» (referencia) y «los dos
modelos» (anáfora, que lleva la Parte 2 a la Parte 3: sin ella la curva se entrenó sin estandarizar el
2026-10-04). La capa 2 agrega las entidades y su fusión con las notas del curso:

![Capa 2 de la Tarea A](corridas/final/completo-r1/tarea-A/grafo_entidades.png)

### 2.4 Un plan

<!-- incluir-codigo: corridas/final/completo-r1/tarea-A/plan.json -->

### 2.5 Una traza con un rechazo del crítico

Corrida `corridas/completo-r2/tarea-C` (2026-10-04): el programador de la T3 razonó hasta el tope de 32 768
tokens sin responder; con 65 536 entregó un script que no compilaba; la guarda lo detuvo y el crítico lo
rechazó por código; el siguiente intento agotó el presupuesto y el redactor entregó con la reserva declarando
lo que faltó.

<!-- incluir: informe/evidencia/r2_C_T3.md -->

Otro rechazo, de `corridas/completo-r1/tarea-A`, es un **falso positivo**: la comprobación estática de fuga
rechazó la T2 porque, además de evaluar en prueba, el script calculaba la exactitud en entrenamiento como
diagnóstico. Costó un intento (~70 s). Lo dejamos así porque la comprobación es conservadora a propósito.

---

## 3. Parte 2.a — El golden set

`golden/golden_tareas.json` parte de las 25 comprobaciones del kit (A, B, C) y agrega 22: **A09** y dos
tareas reales de *Matemáticas y Programación para IA* (S1, telemetría, 9 comprobaciones; S2, comparación
pareada, 12). La Tarea D del kit queda fuera: su paquete no está en el repositorio. Toda verdad se calcula
(`verdad_py`, `golden/verdades_reales.py`); cada comprobación lleva su `_por_que`.

- **La trampa (S210–S211).** El intervalo bootstrap de S2 exige `default_rng(20260829)` y remuestrear tiendas
  completas: lo correcto es [−0,194; 1,006]. Remuestrear A y B por separado da [−1,169; 2,000]; usar la
  semilla global antigua da [−0,169; 0,994]. Las dos violaciones fallan con tolerancia 0,005.
- **A09.** El kit no detectaba la curva de aprendizaje sin estandarizar: A09 (0,9766 al 50 %) falla en esa
  corrida y pasa en las correctas.
- **Formato.** Hay notebook (B), PDF con límite de páginas (C) y archivos exigidos en `output/` (S1, S2).

La tabla completa, comprobación por comprobación:

<!-- incluir: golden/README.md -->

---

## 4. Parte 2.b — Solver completo contra la ablación «sin grafo»

`SolverSinGrafo`: el investigador recibe los dos fragmentos del enunciado más parecidos por TF-IDF y nada
más (lo de la 0.b), sin aristas ni capa 2. Mismo golden, mismo modelo, dos repeticiones por variante.

<!-- incluir: resultados/final/tablas_2b/tablas.md -->

**[PENDIENTE: lectura de las tablas — qué compró el grafo en comprobaciones y en tokens, con las cifras]**

---

## 5. Parte 2.c — Las tres peores comprobaciones

**[PENDIENTE: elegir de la tabla 4 de la 2.b; para cada una, el agente señalado y la evidencia de
`scripts/evidencia.py`]**

---

## 6. Parte 3 — Los cuatro frenos, forzados

Los cuatro se fuerzan con un modelo de guion (`scripts/forzar_frenos.py`): los frenos, el sandbox y la traza
son los del solver real. Dónde vive cada uno: `solver/frenos.py`. Además, el freno de presupuesto se disparó
con el modelo real en `corridas/completo-r2/tarea-C` (sección 2.5).

<!-- incluir: corridas/frenos/README.md -->

---

## 7. Parte 4 — Extensión E: un juez de otra familia

`scripts/juez.py`: gemma3 (Ollama de la H200) califica cada entregable de la 2.b con las preguntas del campo
`juez` del golden —las mismas comprobaciones, sin la cifra verdadera— y se mide su acuerdo con el golden
determinístico.

<!-- incluir: resultados/final/juez/juez.md -->

**[PENDIENTE: lectura — dónde acierta, dónde aprueba lo que el golden rechaza, y qué costó]**

---

## 8. Parte 5 — Reflexión

**1. Dónde vive el objetivo.** Ningún LLM decide que la tarea está resuelta. El planificador propone
subtareas, pero el plan solo avanza si `planificador.validar()` lo acepta; el crítico LLM opina, pero solo
después de `critico.comprobar()` y nunca puede deshacer un rechazo del código; el redactor escribe, pero
`_verificar` (orquestador) decide si se publica con la forma del enunciado y la procedencia de cada cifra.
El criterio de parada está en `_tras_verificar` y el veredicto en `_cerrar`: «completado» exige todas las
subtareas de cálculo aprobadas, ningún problema de redacción pendiente y ninguna parada por presupuesto o
error. Es decir: el objetivo vive en código que se puede leer y probar, no en el juicio del modelo.
**[PENDIENTE: revisar y escribir con voz propia]**

**2. Con `resumen_solver.csv` delante.** **[PENDIENTE: con las tablas finales. Lo que ya muestran r1 y r2:
el programador gasta ~59 % de los tokens, casi todo razonamiento de salida]**

**3. Uso honesto y daño.** Usarlo honestamente exige declarar que la tarea se resolvió con el solver,
adjuntar la traza (qué recibió cada agente, qué produjo, qué decidió el código) y separar lo que hizo la
persona. En la revisión del sandbox encontramos lo que solo *parecía* proteger: la guarda estática no veía
`pd.read_csv("https://…")` (salía a la red) ni `importlib.import_module("soc"+"ket")`; los cerramos (una URL
como argumento ahora pide confirmación humana; `importlib` está prohibido), pero una guarda sobre el árbol
sintáctico nunca ve todo lo que se construye en tiempo de ejecución, y el proceso no tiene aislamiento de
red a nivel del sistema operativo. Lo que protegió de verdad: el entorno vacío (ninguna variable del `.env`
llega al script), la carpeta propia, el `killpg` y las comprobaciones de la 0.c antes del LLM. Si el sandbox
fallara, el código generado podría leer o borrar archivos del usuario o filtrar datos por la red.
**[PENDIENTE: revisar y escribir con voz propia]**

---

## 9. Reproducibilidad

```bash
uv sync && cp .env.example .env              # y la VPN
uv run pytest -q
uv run python golden/verdades_reales.py
bash scripts/corridas_2b.sh completo 1       # y 2; sin_grafo 1 y 2
bash scripts/corridas_2b.sh tablas
bash scripts/corridas_2b.sh frenos
bash scripts/corridas_2b.sh juez
uv run python scripts/evidencia.py corridas/completo-r2/tarea-C --subtarea T3 --salida informe/evidencia/r2_C_T3.md
uv run python scripts/diagrama_png.py
uv run python scripts/informe_pdf.py
```

Versiones: `uv.lock` y `requirements.txt`. Ninguna clave de API aparece en el código, las trazas ni los
scripts generados: `.env` está fuera de git y el sandbox corre con el entorno vacío.
