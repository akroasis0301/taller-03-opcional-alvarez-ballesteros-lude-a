<!-- Fuente del informe. Se arma con `uv run python scripts/informe_word.py` (formato del Taller 02:
     Word y PDF). Las convenciones que entiende el generador están en el docstring de ese script. -->

**Ruta, modelo y fecha.** H200 de la USFQ por VPN (GlobalProtect). LLM: vLLM en `172.28.230.10:12555`, id
`zai-org/GLM-5.3-Flash`, leído de `/v1/models` (no escrito a mano). Embeddings: `bge-m3:latest` en el Ollama del
mismo servidor (puerto 11434). Juez de la Parte 4: `gemma3:27b` en ese Ollama. Corridas finales, todas el
**5 de octubre de 2026** (hora de Ecuador, UTC−5): `completo` r1 y r2 de 17:59 a 18:50; `sin_grafo` r1 y r2 de 18:50
a 19:53; tareas reales S1 y S2 de 19:58 a 21:05; el juez de 21:12 a 21:32. Las Partes 1 y 2 corrieron con un solo
modelo. Costo: 0 USD (sin clave de API).

**Autoría y uso de asistentes.** [PENDIENTE: quién hizo cada parte y cómo se usó un asistente de IA en el
desarrollo; la Parte 5 pide declararlo con la traza como evidencia.]

**Cómo leer este informe.** Las tablas de resultados se insertan desde los archivos que generan los scripts a partir
de los CSV crudos y las trazas (sección 9); ninguna cifra de una tabla se transcribió a mano. Los nombres de archivo
que aparecen en el texto son rutas del repositorio.

# 1. Parte 0 — Tres fallas que no fallan

La Parte 0 reproduce, antes de construir nada, tres cosas que un solver ingenuo hace mal sin dar ningún error: un
LLM que «resuelve» una tarea sin ejecutarla, un RAG plano que pierde la referencia entre partes y un evaluador que
confunde un `returncode == 0` con un experimento. Cada una justifica una de las correcciones de la Parte 1.

> *0.a — El solver que no ejecuta. Entregable: tu salida, la ruta y el id del modelo, y tres frases: por qué que las
> cifras estén cerca no las hace aceptables, qué cifra del reporte habría cambiado una conclusión, y qué tendría que
> comprobar un verificador para detectar esto sin conocer las respuestas correctas.*

Se ejecutó `python parte0/a_llm_sin_ejecutar.py` con la H200 (vLLM 12555, `zai-org/GLM-5.3-Flash`) el
2026-10-03: 12 028 tokens de salida en 61,72 s. La salida fue:

<!-- incluir-codigo: resultados/Entregables_Parte_0/salida_0a.txt -->
*Ilustración: Salida de `a_llm_sin_ejecutar.py`: las cifras medidas frente a las del reporte sin ejecutar (Parte 0.a)*

**Por qué que las cifras estén cerca no las hace aceptables.** El reporte dice 0,9415 para Naive Bayes (medida:
0,9474) y 0,9825 para la regresión logística (medida: 0,9883): están cerca, pero ninguna salió de una ejecución (0 de
24 cifras con respaldo). «Cerca» solo se sabe porque esta vez alguien ejecutó el experimento al lado; sin esa
ejecución no hay forma de distinguir una cifra plausible de una correcta, y el reporte no declara que sean
estimaciones.

**Qué cifra habría cambiado una conclusión.** La de la curva con pocos datos: con el 5 % del entrenamiento el reporte
da Naive Bayes 0,9123 contra regresión logística 0,9064, y de ahí concluye que con pocos ejemplos conviene el
generativo (el cruce de Ng y Jordan). La diferencia es un acierto sobre 171: con un acierto más para la regresión
logística la conclusión se invierte, y ninguna de las dos cifras tiene una ejecución detrás (medido con el pipeline
estandarizado: 0,9415 y 0,9298). En la corrida del enunciado (2026-10-02) el modelo llegó a inventar 0,7251 para la
regresión logística: un cruce exagerado que confirmaba la tesis.

**Qué tendría que comprobar un verificador sin conocer las respuestas.** Que cada cifra del reporte aparezca en algo
que escribió una ejecución (un `resultados.json` de un script que terminó bien) o en el enunciado, y devolver por su
nombre las que no. Es la comprobación de procedencia (corrección C5).

> **Conclusión:** una cifra sin procedencia no es un resultado aunque acierte. El modelo produjo 24 cifras con
> aspecto de medidas y ninguna salió de una ejecución; la única defensa que no necesita conocer la respuesta es
> exigir que cada cifra tenga un origen.

> *0.b — El RAG plano que pierde la referencia. Entregable: la salida y tres frases: por qué esta falla es de la
> recuperación y no de la generación, qué haría un LLM con el primer contexto, y por qué la arista se extrae con una
> regla y no se le pide al LLM.*

<!-- incluir-codigo: resultados/Entregables_Parte_0/salida_0b.txt -->
*Ilustración: Salida de `b_rag_plano.py`: RAG plano contra el salto por la arista `depende_de` (Parte 0.b)*

**Es una falla de recuperación.** Los datos y la partición están escritos en la Parte 1; el RAG plano trae la Parte 4
(0,293) y la Parte 3 (0,243) y cubre 0 de 3 hechos. Al seguir la arista `depende_de` Parte 3 → Parte 1 cubre 3 de 3
sin tocar el generador: el generador no puede usar lo que no recibió.

**Qué haría un LLM con el primer contexto.** Rellenar el hueco: suponer un dataset y una partición razonables y
presentarlos con la misma seguridad que un dato del enunciado, sin avisar que faltaban. La sección 5 lo muestra con
el solver real: sin grafo, el programador inventó las consultas de la Tarea C en una corrida y el corpus en la otra.

**Por qué la arista se extrae con una regla.** «La misma división de la Parte 1» es un patrón literal: una expresión
regular lo encuentra siempre, igual en cada corrida y sin tokens. Un LLM puede omitir la arista o inventar otra, y si
falla vuelve la misma pérdida de contexto sin ningún aviso. El LLM se reserva para lo que una regla no puede hacer
(entidades y relaciones, capa 2).

> **Conclusión:** la similitud entre la pregunta y un fragmento no mide si el fragmento contiene lo que la pregunta
> necesita. Una referencia explícita del enunciado es estructura, y la estructura se extrae con código.

> *0.c — Un returncode == 0 no es un experimento. Entregable: la salida y dos frases: qué otra fuga no detectaría la
> comprobación estática (piensa en un escalador ajustado con todos los datos), y por qué la decisión final puede
> quedar en manos de un LLM crítico si las comprobaciones de código van antes.*

<!-- incluir-codigo: resultados/Entregables_Parte_0/salida_0c.txt -->
*Ilustración: Salida de `c_exit_cero.py`: el evaluador ingenuo aprueba y las dos comprobaciones de código rechazan (Parte 0.c)*

**La fuga que la comprobación estática no ve.** Un escalador ajustado con todos los datos antes de dividir
(`StandardScaler().fit_transform(X)` y después `train_test_split`): el modelo se ajusta con `X_tr` y se evalúa con
`X_te`, nombres distintos, y la exactitud (≈0,98) no llega al umbral de plausibilidad. Lo mismo una imputación o una
selección de atributos ajustadas con todo. Nuestro crítico agrega una comprobación para el caso del escalador
(`ajuste_antes_de_dividir`, `solver/agentes/critico.py`).

**Por qué la decisión final puede quedar en un LLM crítico.** Porque las comprobaciones de código van antes y no se
pueden persuadir: el LLM solo ve lo que ya pasó los filtros, puede agregar rechazos pero no deshacer uno del código.
En nuestro solver, además, un rechazo del LLM solo cuenta si cita una frase literal del enunciado; si la cita no
existe, se descarta (`rechazo_descartado`).

> **Conclusión:** terminar sin error y escribir un archivo es una condición necesaria, no la evidencia de un
> experimento correcto. Lo que se puede comprobar con código se comprueba primero; el LLM solo opina sobre lo demás.

# 2. Parte 1 — El solver multiagente

## 2.1 Arquitectura: ocho agentes y un orquestador

El orquestador (`solver/orquestador.py`) es un `StateGraph` de LangGraph 1.2.12. El ciclo
programar → ejecutar → criticar → programar sale por aprobación o por tope de intentos; la salida a redactar es por
fin de la cola o por presupuesto. Las aristas del diagrama se leen del Mermaid que exporta LangGraph; las punteadas
rojas son condicionales.

![Diagrama del orquestador (LangGraph) con sus aristas condicionales](resultados/orquestador.png)

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
*Tabla: Los ocho agentes y lo que el código comprueba de cada uno*

El contrato es el del enunciado: `Solver().solve(ruta_pdf, carpeta_salida)` devuelve `status`, `entregables`,
`subtareas`, `usage`, `model` y `trace`; `Solver().run(pregunta)` expone el contrato del Taller 4.

## 2.2 Las seis correcciones, señaladas en el código

| | Corrección | Dónde | Evidencia |
|---|---|---|---|
| C1 | El plan es un DAG validado por código | `planificador.validar()` (ids, dependencias, ciclos, cobertura, aristas del grafo) y `ordenar()`; `_tras_planificar` devuelve el plan inválido con sus problemas (máx. 3) | traza: `plan_invalido` / `plan_valido` |
| C2 | GraphRAG con esqueleto determinístico | capa 1: `indexador.esqueleto()` (referencia, «la pregunta anterior», anáfora «los dos modelos»); capa 2: `graphrag.py` (entidades del LLM, fusión por nombre normalizado con las notas del curso, bge-m3, comunidades de Louvain con resumen); `graphrag.busqueda_local()` con citas | `grafo.png`, `grafo_entidades.png`, traza: `capa2`, `grafo+capa2` |
| C3 | Sandbox | `sandbox/guarda.py` (AST: red, procesos, borrado, `eval`, `importlib`, rutas fuera, URLs como descarga); `ejecutor.py`: entorno vacío, carpeta propia, `start_new_session` + `killpg` al vencer el tiempo | Parte 3 |
| C4 | Contrato de resultados y crítico que mira antes | `resultados.json` obligatorio; `critico.comprobar()` primero, `revisar_con_llm()` solo si pasan | traza: `rechazado` con `por: codigo` o `por: llm` |
| C5 | Procedencia | `procedencia.verificar()` en `_verificar`: las cifras sin origen vuelven al redactor por su nombre; al último intento se marcan `[cifra sin respaldo]` | traza: `devuelto_al_redactor`, `cifras_marcadas` |
| C6 | Traza en todo camino | `traza.py` (una línea por evento, escrita al momento, con secretos enmascarados); `_nodo` atrapa todo error y lo registra; toda corrida termina en `cerrar` con `fin` | `traza.jsonl` de cada corrida |
*Tabla: Las seis correcciones no negociables y dónde viven*

## 2.3 El grafo del enunciado de la Tarea A

Las aristas `depende_de` (en rojo) salen de reglas: «la misma división de la Parte 1» (referencia) y «los dos
modelos» (anáfora, que lleva la Parte 2 a la Parte 3; sin ella, el 2026-10-04 la curva se entrenó sin estandarizar).

![Grafo del enunciado de la Tarea A, con las aristas depende_de](corridas/final/completo-r1/tarea-A/grafo.png)

La capa 2 agrega las entidades que extrae el LLM y su fusión con las notas del curso (31 fragmentos, 344 entidades,
12 comunidades, construidas una sola vez y guardadas en caché):

![Capa 2 de la Tarea A: entidades del enunciado fusionadas con las notas del curso](corridas/final/completo-r1/tarea-A/grafo_entidades.png)

## 2.4 Un plan

El plan que produjo el planificador para la Tarea A en la corrida `completo-r1`:

<!-- incluir-codigo: corridas/final/completo-r1/tarea-A/plan.json -->
*Ilustración: `plan.json` de la Tarea A (completo-r1): cuatro subtareas con tipo, dependencias y criterio*

## 2.5 Una traza con un rechazo del crítico

Corrida `corridas/completo-r2/tarea-C` (2026-10-04): el programador de la T3 razonó hasta el tope de 32 768 tokens
sin responder; con 65 536 entregó un script que no compilaba; la guarda lo detuvo y el crítico lo rechazó por código;
el siguiente intento agotó el presupuesto y el redactor entregó con la reserva declarando lo que faltó. La traza
completa está en el repositorio; este es el extracto de esa subtarea:

<!-- incluir: informe/evidencia/r2_C_T3.md -->

Otro rechazo, de `corridas/completo-r1/tarea-A`, es un **falso positivo**: la comprobación estática de fuga rechazó
la T2 porque, además de evaluar en prueba, el script calculaba la exactitud en entrenamiento como diagnóstico. Costó
un intento (~70 s). Lo dejamos así porque la comprobación es conservadora a propósito.

> **Conclusión:** ningún paso del solver confía en que el modelo haya hecho bien su parte: el plan, el código, el
> resultado y cada cifra pasan por una comprobación en código antes de avanzar, y cada decisión queda en la traza.

# 3. Parte 2.a — El golden set

> *El golden set es el verificador. Parte de `golden_tareas.json` (las tres tareas de práctica, 25 comprobaciones) y
> añade tus dos tareas reales o más, cada una con al menos cinco comprobaciones; entre todas, un notebook, un PDF y
> una trampa.*

`golden/golden_tareas.json` parte de las 25 comprobaciones del kit (A, B, C) y agrega 22: **A09** y dos tareas reales
de *Matemáticas y Programación para IA* (S1, telemetría de servidores, 9 comprobaciones; S2, comparación pareada de
dos modelos, 12). La Tarea D del kit queda fuera: su paquete no está en el repositorio. Toda verdad se calcula
(`verdad_py`, `golden/verdades_reales.py`) y cada comprobación lleva su `_por_que`; la tabla completa está en el
Anexo A.

- **La trampa (S210–S211).** El intervalo bootstrap de S2 exige `default_rng(20260829)` y remuestrear tiendas
  completas: lo correcto es [−0,194; 1,006]. Remuestrear A y B por separado da [−1,169; 2,000]; usar la semilla global
  antigua da [−0,169; 0,994]. Las dos violaciones fallan con tolerancia 0,005. Una trampa secundaria (S106) separa
  la desviación de NumPy (ddof=0) de la de pandas (ddof=1).
- **A09.** El kit no detectaba la curva de aprendizaje sin estandarizar: A09 (0,9766 al 50 %) falla en esa corrida
  y pasa en las correctas.
- **Formato.** Hay notebook (B), PDF con límite de páginas (C) y archivos exigidos en `output/` (S1, S2).

**Lo que el golden no ve.** Las corridas finales mostraron tres límites, que declaramos en lugar de esconder:

1. **No mide completitud.** S2 aprobó 12 de 12 con status `parcial`: la Tarea 7 del enunciado (integrar el pipeline
   en una sola corrida) quedó sin resultado y ninguna comprobación la mira, porque el golden comprueba las cifras del
   entregable, no que se haya hecho todo lo pedido.
2. **La procedencia aprueba en vacío.** En `sin_grafo-r2/tarea-C` el reporte no tiene ninguna cifra y C08 aprueba
   («sin cifras con dos o más decimales»). El juez de la Parte 4 lo rechazó, y tiene razón.
3. **El umbral de procedencia deja pasar cifras erróneas o inventadas.** En una corrida de desarrollo el notebook de
   B publicó una divergencia KL de «5.0069» cuando la medida era 5.0069e-05; B09 aprobó con 81 de 82 cifras
   respaldadas (umbral 0,85). La atrapó nuestra C5, no el golden (sección 5, caso 2). Y en la prueba adversarial de la
   Parte 4, un reporte con tres cifras de LSA inventadas aprobó C08 con 24 de 27 (0,89): solo C06, que calcula la
   verdad, lo detectó.

Además, S103 tiene una limitación conocida del evaluador del kit, documentada en el Anexo A: acepta un encabezado de
hasta 40 caracteres más que el nombre buscado, así que «Tarea 2 - Normalizar y calcular un indicador de carga»
no se reconoce como «Tarea 2».

> **Conclusión:** el golden set detecta lo que fue diseñado para detectar (cifras, forma, la trampa), y sus tres
> huecos son de diseño, no de cálculo: mide lo que hay en el entregable, no lo que falta ni lo que sobra.

# 4. Parte 2.b — Solver completo contra la ablación «sin grafo»

> *Sobre tu solver completo y sobre al menos una ablación, con el mismo golden set y el mismo modelo: comprobaciones
> aprobadas, procedencia, subtareas fallidas, intentos de código, tokens de entrada y de salida por agente, y
> duración.*

`SolverSinGrafo` es el mismo solver con el investigador de la 0.b: recibe los dos fragmentos del enunciado más
parecidos por TF-IDF y nada más, sin aristas, sin capa 2 ni notas del curso. Mismo golden (tareas A, B y C), mismo
modelo, dos repeticiones por variante. Las tareas reales S1 y S2 se corrieron una vez, solo con el solver completo
(tabla al final de esta sección).

<!-- incluir: resultados/final/tablas_2b/tablas.md -->

**Qué compró el grafo, en comprobaciones.** Con grafo, 52 de 52 comprobaciones (26/26 en las dos repeticiones); sin
grafo, 48 de 52 (25/26 y 23/26). Las cuatro que se pierden están en la Tarea C y en ninguna otra: A y B son tareas en
que cada parte trae sus propios datos, y el RAG plano ya recupera lo necesario. C es la única cuyo enunciado reparte
la información en tres tablas (corpus, consultas y juicios, y la Parte 1 que pide el cálculo), y el TF-IDF top-2
siempre deja fuera una. Sin grafo C quedó `parcial` en las dos repeticiones, con 2 y 3 subtareas no logradas.

**Qué compró, en tokens.** El grafo cuesta contexto: con grafo se gastan 125 342 tokens de entrada contra 80 869 sin
él (+55 %), y el indexador consume el 9 % del total. Pero no ahorra salida: 269 392 contra 275 548, porque sin el
contexto correcto el programador razona más y produce menos (en `sin_grafo-r1/tarea-C`, 324 259 tokens de salida y
36 minutos para terminar `parcial`). En tiempo total, con grafo es un 19 % más rápido (1 529 s contra 1 888 s). A eso
se suma un costo fijo que no está en las tablas por tarea: el índice de las notas, 248 227 tokens y 320 s, una sola
vez.

**Lo que no se puede afirmar.** Con dos repeticiones la varianza de `sin_grafo` es enorme (160 003 a 391 094 tokens
de salida): las cifras de costo son indicativas. Lo que sí es firme es el mecanismo, porque se repitió con causa
visible en la traza: en r1 el contexto de la T1 de C traía el corpus pero no las consultas, y en r2 las consultas
pero no el corpus (sección 5, caso 1).

| tarea | status | comprobaciones | subtareas | intentos de código | tokens entrada | tokens salida | duración (s) |
|---|---|---|---|---|---|---|---|
| S1 (telemetría de servidores) | completado | 8/9 | 5 | 5 | 59 328 | 122 636 | 984 |
| S2 (comparación pareada) | parcial | 12/12 | 11 | 7 | 114 383 | 386 137 | 3 002 |
*Tabla: Tareas reales con el solver completo (`resultados/final/resumen_reales_r1.csv`, `corridas/final/reales-r1`)*

En S1 la única comprobación que falla es S103, por la limitación del evaluador descrita en la 2.a. S2 resolvió la
trampa del bootstrap (S210–S211) y aprobó todas sus cifras, pero quedó `parcial`: la T7 razonó hasta el tope tres
veces y el freno de presupuesto detuvo el trabajo a 460 187 de 500 000 tokens; el redactor entregó con la reserva y
declaró en la primera línea que la Tarea 7 no tiene resultado. En S2 el grafo no aportó aristas (sus siete tareas son
independientes) ni fusiones con las notas del curso (0 entidades fusionadas: las notas hablan de LLM y RAG, no de
bootstrap).

> **Conclusión:** el grafo no es una mejora universal. Compra la sección literal y sus dependencias cuando el
> enunciado reparte la información entre partes (C), y ahí decide entre aprobar y fallar; cuando cada parte es
> autocontenida (A, B, S2) cuesta contexto y no cambia el resultado.

# 5. Parte 2.c — Las tres peores comprobaciones

> *Las tres comprobaciones peores de tu solver y, para cada una, el agente al que apunta, con la traza como evidencia:
> qué recibió, qué produjo, qué decidió el código.*

Las tres se eligieron de la tabla 4 de la 2.b y de las corridas de desarrollo que motivaron cambios en el solver.
Cada extracto sale de `scripts/evidencia.py` sobre la traza de la corrida.

## Caso 1 — C06 (y C04, C05) sin grafo: el investigador entrega un contexto incompleto

**Comprobación.** C06, el MRR de LSA en la Tarea C, falla en las dos repeticiones sin grafo; en r2 fallan también C04
y C05. **Agente señalado: investigador** (en la variante sin grafo), con el programador como el que rellena el hueco.

**Qué recibió.** El contexto de la T1 tenía 8 670 caracteres con grafo (la Parte 1, las tres tablas y las notas sobre
BM25) y 1 407 y 823 sin él:

| corrida | contexto de la T1 | qué faltó | qué hizo el programador | qué decidió el código |
|---|---|---|---|---|
| completo-r1 | Parte 1 + 3 tablas + notas | nada | — | aprobada al primer intento |
| sin_grafo-r1 | Parte 1 + Corpus (sim. 0,547 y 0,239) | las consultas | inventó sus propias consultas | el crítico LLM lo rechazó citando q5 literal; T2 falló tres veces en el tope de razonamiento; T3 omitida |
| sin_grafo-r2 | Parte 1 + Consultas (sim. 0,574 y 0,269) | el corpus | reescribió los diez documentos | rechazado 3 de 3 citando el texto real de d02 y d03; T2 y T3 omitidas |
*Tabla: El contexto de la T1 de la Tarea C en las tres variantes (`cache_solver/contextos/T1.md` de cada corrida)*

**Qué produjo y qué se publicó.** En r2 el redactor no tuvo ninguna cifra medida y lo declaró («La subtarea T1 no
tiene resultado: no se reportan sus cifras»); el reporte no contiene ninguna cifra inventada. Es la falla de la 0.b
medida en el solver real: el RAG plano no pierde «algo de contexto», pierde la tabla que la subtarea necesita, y el
programador la rellena con datos plausibles. Lo que contuvo el daño fue el crítico (C4), que tiene el enunciado
completo y cita la frase que el script viola.

## Caso 2 — C08 en desarrollo: el redactor publica una cifra que nadie midió

**Comprobación.** C08 (procedencia de la Tarea C) falló en `corridas/desarrollo/final-completo-r1-sin-F3F4` con 7 de
9 cifras respaldadas (0,78). **Agente señalado: redactor.**

**Qué recibió y qué produjo.** Las cifras medidas en `resultados.json`. El redactor escribió `0.6140` donde la medida
era 0.61405098 (truncó en lugar de redondear: lo correcto es 0.6141). Escribió también `39.53 %`, que sí era una
medida (0.39534) expresada como porcentaje. En la misma corrida, el notebook de B escribió una divergencia KL de
«5.0069» cuando la medida era 5.0069e-05, y en la misma frase dijo que era «prácticamente nula».

**Qué decidió el código.** La procedencia (C5) devolvió las cifras al redactor por su nombre tres veces seguidas
(`devuelto_al_redactor`: «['0.6140', '39.53 %']» en C y «['5.0069']» en B), y el redactor las repitió las tres
veces: sin decirle cuál era la cifra correcta, no supo corregirla. Al agotar los intentos el reporte se publicó con
ellas. El golden atrapó la de C (C08) pero no la de B, que aprobó B09 con 81 de 82 cifras. Y uno de los dos rechazos
de C era nuestro: el 39.53 % estaba bien y la procedencia no lo reconocía porque buscaba el porcentaje en una sola
de sus dos lecturas (39.53 o 0.3953).

**Qué cambiamos.** F3: la procedencia lee notación científica, acepta un porcentaje si cualquiera de sus dos
lecturas está respaldada y devuelve sugerencias concretas («¿quisiste decir 0.6141?», «la medida es 5.0069e-05»).
F4: al último intento, las cifras que siguen sin respaldo se marcan `[cifra sin respaldo]` en el entregable en lugar
de publicarse limpias. En las corridas finales la procedencia fue
1,00 en todas las tareas salvo una (0,96, completo-r2 A).

## Caso 3 — S103: la única comprobación que falla el solver completo

**Comprobación.** S103 (secciones de S1: «Tarea 1» a «Tarea 5», en orden) falla en `reales-r1`: «faltan: ['Tarea
2']». **Agente señalado: redactor**, pero la causa está en el verificador.

**Qué recibió y qué produjo.** El enunciado pide una sección por tarea y titula cada una con su nombre completo; el
redactor escribió `## Tarea 2 - Normalizar y calcular un indicador de carga`. El evaluador del kit acepta un
encabezado de hasta 40 caracteres más que el nombre buscado: este se pasa por 4. La Tarea 3 entra justo en el límite.

**Qué decidió el código.** Nada que corregir en el solver: el encabezado es el que pide el enunciado. Probamos las
dos formas de escribir la comprobación (los títulos completos y «Tarea N») y ninguna acepta los dos estilos de
encabezado; dejamos «Tarea N» y documentamos la limitación. Es el caso que el enunciado anticipa: cuando el solver
completo no falla, el problema es el golden set.

> **Conclusión:** los tres peores casos apuntan a tres agentes distintos (investigador, redactor, verificador), y en
> los tres la traza permite ver qué recibió cada uno. El patrón común es que el modelo no deja huecos: cuando le
> falta un dato lo inventa, y cuando una cifra se le devuelve, la repite. Lo que funcionó fue el código que lo rodea.

# 6. Parte 3 — Los cuatro frenos, forzados

> *Implementa cuatro frenos y demuestra cada uno forzándolo: una corrida que lo dispare y su traza. Para forzarlos
> sin gastar, un modelo de guion sirve.*

Los cuatro se fuerzan con un modelo de guion (`scripts/forzar_frenos.py`): los frenos, el sandbox y la traza son los
del solver real. Dónde vive cada uno: `solver/frenos.py`.

<!-- incluir: corridas/frenos/README.md -->

**El freno de presupuesto, con el modelo real.** Se disparó dos veces sin forzarlo: en `corridas/completo-r2/tarea-C`
(2026-10-04, sección 2.5) y en la tarea real S2 de las corridas finales, donde la T7 consumió tres llamadas en el
tope de razonamiento y el trabajo se detuvo a 460 187 de 500 000 tokens (`presupuesto_agotado`, eventos 60 y 61 de
`corridas/final/reales-r1/tarea-S2/traza.jsonl`). El redactor usó la reserva de 50 000 y entregó el informe con la
nota: «La subtarea T7 no tiene resultado (pendiente): no se reportan sus cifras. El presupuesto de tokens se agotó».

> **Conclusión:** los cuatro frenos se disparan y dejan su rastro en la traza. El de presupuesto es el que más
> importa con este modelo, porque su modo de falla habitual es razonar hasta el tope sin entregar nada.

# 7. Parte 4 — Extensión E: un juez de otra familia

> *El reporte lo califica un LLM de otra familia (gemma3 en el Ollama de la H200) contra la rúbrica del enunciado,
> calibrado contra tus comprobaciones determinísticas: acuerdo y kappa.*

`scripts/juez.py` le da a `gemma3:27b` el enunciado y el entregable de cada corrida de la 2.b, y le hace las
preguntas del campo `juez` del golden: las mismas comprobaciones, sin la cifra verdadera. Su veredicto se compara con
el de `evaluar_solver.py`.

<!-- incluir: resultados/final/juez/juez.md -->

**Dónde acierta.** Acuerdo 0,986 y kappa 0,882 sobre 74 comprobaciones, sin falsos positivos: rechazó las cuatro
cifras que el golden rechaza (C06 en sin_grafo-r1; C04, C05 y C06 en sin_grafo-r2). El único desacuerdo es C08 de
sin_grafo-r2, donde el juez rechaza una procedencia que el golden aprueba en vacío, y el juez tiene razón (2.a).

**Por qué esas cifras no prueban lo que parecen.** En los cuatro rechazos el redactor había escrito que esas cifras
no se reportaban: el juez leyó la declaración («el reporte indica explícitamente que la ejecución de LSA falló»), no
detectó la falta. Sus razones para aprobar son de coherencia, no de corrección: «0.8611 parece plausible dado los
resultados por consulta», «aunque no cuento las palabras exactamente, el reporte parece estar dentro del límite». Por
eso le quitamos esa ayuda con una prueba adversarial: el reporte de `sin_grafo-r1/tarea-C`, adulterado a mano con un
MRR de LSA inventado (0,8056; la verdad es 0,7500), coherente con su propia tabla y sin ningún rastro de la falla
(`scripts/juez_adversarial.py`, en `corridas/desarrollo/juez-adversarial`). Lo que el juez respondió sobre ese
reporte:

<!-- incluir: resultados/final/juez_adversarial/juez.md -->

**El resultado: el juez aprobó la cifra inventada.** Respondió que «el reporte indica un MRR de 0.8056 para LSA, lo
cual es consistente con los resultados presentados en la tabla», y aprobó también la procedencia porque las cifras
tienen «valores decimales que sugieren cálculos precisos y no estimaciones». Es exactamente el modo de falla de la
0.a, ahora en el verificador: una cifra con aspecto de medida pasa por medida. Comprobó que la cifra coincidiera con
la tabla del propio reporte (la inventamos coherente a propósito), no con el experimento; sin ejecutar el código no
tiene cómo saber que el MRR verdadero es 0,7500. El golden la rechazó con C06, la comprobación que calcula la verdad.
Pero su comprobación de procedencia también falló: C08 aprobó con 24 de 27 cifras respaldadas (0,89), porque tres
cifras inventadas quedan por debajo del umbral de 0,85. Lo que detectó la falsificación fue la cifra calculada, no
ninguna de las dos formas de procedencia.

**Qué costó y qué tan estable es.** 36 587 tokens de entrada y 3 487 de salida en 20 minutos (12 llamadas de ~100 s).
El formato pedido (`{"ok", "razon"}`) no se respetó en 12 de 80 respuestas, siempre en la Tarea B (el notebook, la
entrada más larga): seis veces solo con booleanos y seis sin la envoltura `veredictos`, con el mismo modelo,
`temperature=0` y casi el mismo entregable. Un juez que no da razones no se puede auditar.

> **Conclusión:** el juez no reemplaza al verificador determinístico. Su kappa de 0,882 sobre las corridas de la
> 2.b mide un caso fácil: un solver que declara sus fallas. Ante una cifra inventada y coherente la aprueba, porque
> valida coherencia y no corrección. Sirve como segunda opinión sobre lo que el golden no mide (la forma, si se
> nombra la consulta que falla, una procedencia aprobada en vacío), no para decidir si una cifra es correcta. Y
> deja una lección también para nuestro golden: la procedencia por fracción con umbral deja pasar tres cifras
> inventadas; solo la cifra calculada las detecta.

# 8. Parte 5 — Reflexión

**1. Dónde vive el objetivo.** Ningún LLM decide que la tarea está resuelta. El planificador propone subtareas, pero
el plan solo avanza si `planificador.validar()` lo acepta; el crítico LLM opina, pero solo después de
`critico.comprobar()` y nunca puede deshacer un rechazo del código; el redactor escribe, pero `_verificar`
(orquestador) decide si se publica con la forma del enunciado y la procedencia de cada cifra. El criterio de parada
está en `_tras_verificar` y el veredicto en `_cerrar`: «completado» exige todas las subtareas de cálculo aprobadas,
ningún problema de redacción pendiente y ninguna parada por presupuesto o error. El objetivo vive en código que se
puede leer y probar, no en el juicio del modelo. [PENDIENTE: revisar y escribir con voz propia.]

**2. Con `resumen_solver.csv` delante.** El programador gasta más que todos: 61 % de los tokens con grafo y 67 % sin
él, y casi todo es salida (195 568 de salida contra 45 741 de entrada con grafo), porque el modelo de la H200 razona
antes de escribir cada script y ese razonamiento se cobra como salida. Frente a la ablación, el grafo compró 4
comprobaciones de 52 (52 contra 48) y 5 subtareas no logradas menos, todas en la Tarea C; costó 55 % más de entrada
(125 342 contra 80 869 tokens) y el 9 % del gasto en el indexador, sin ahorrar salida (269 392 contra 275 548). No es
más barato: es más confiable donde el enunciado reparte la información, y neutro donde no. [PENDIENTE: revisar y
escribir con voz propia.]

**3. Uso honesto y daño.** Usarlo honestamente exige declarar que la tarea se resolvió con el solver, adjuntar la
traza (qué recibió cada agente, qué produjo, qué decidió el código) y separar lo que hizo la persona. En la revisión
del sandbox encontramos lo que solo *parecía* proteger: la guarda estática no veía `pd.read_csv("https://…")` (salía
a la red) ni `importlib.import_module("soc"+"ket")`; los cerramos (una URL como argumento ahora pide confirmación
humana; `importlib` está prohibido), pero una guarda sobre el árbol sintáctico nunca ve todo lo que se construye en
tiempo de ejecución, y el proceso no tiene aislamiento de red a nivel del sistema operativo. Lo que protegió de
verdad: el entorno vacío (ninguna variable del `.env` llega al script), la carpeta propia, el `killpg` y las
comprobaciones de la 0.c antes del LLM. Si el sandbox fallara, el código generado podría leer o borrar archivos del
usuario o filtrar datos por la red. [PENDIENTE: revisar y escribir con voz propia.]

# 9. Presupuesto, configuración y reproducibilidad

**Presupuesto.** Todo corrió en la H200 sin clave de API: 0 USD. Por corrida, el solver tiene un presupuesto de
500 000 tokens con 50 000 de reserva para el redactor, 3 intentos por subtarea y 120 s por script (`.env.example`).
Las tareas de práctica tomaron entre 3 y 36 minutos y entre 50 000 y 366 000 tokens; S2, 50 minutos y 500 520
tokens. El índice de las notas del curso costó 248 227 tokens una sola vez.

**Reproducibilidad.** El repositorio contiene el código, `pyproject.toml`, `uv.lock` y `requirements.txt` con
versiones fijadas, el golden set, y cada corrida con su `traza.jsonl`, su `plan.json` y su entregable. Los CSV crudos
del evaluador (`resultados_solver.csv` y `resumen_solver.csv` en el enunciado) están en `resultados/final/` con el
nombre de su variante y repetición (`resultados_completo_r1.csv`, `resumen_sin_grafo_r2.csv`, …), para que ninguna
corrida sobrescriba a otra. Ninguna clave de API aparece en el código, las trazas ni los scripts generados: `.env`
está fuera de git y el sandbox corre con el entorno vacío.

```bash
uv sync && cp .env.example .env              # y la VPN
uv run pytest -q
uv run python golden/verdades_reales.py
bash scripts/corridas_2b.sh completo 1       # y 2; sin_grafo 1 y 2
bash scripts/corridas_2b.sh tablas
bash scripts/corridas_2b.sh frenos
bash scripts/corridas_2b.sh juez
uv run python scripts/juez_adversarial.py    # y los dos comandos de su docstring
uv run python scripts/diagrama_png.py
uv run python scripts/informe_word.py
```
*Ilustración: Comandos para reproducir el taller, desde la raíz del repositorio*

# Anexo A — El golden set, comprobación por comprobación

<!-- incluir: golden/README.md -->
