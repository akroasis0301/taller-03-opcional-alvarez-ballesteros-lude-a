# Golden set del grupo (Parte 2.a)

Generado por `golden/generar_readme.py` desde `golden_tareas.json`: cada comprobación lleva su `_por_que`.

```bash
# desde la RAÍZ del repo (las verdad_py de S1 y S2 importan golden/verdades_reales.py)
uv run python golden/verdades_reales.py          # verdades de S1 y S2 y lo que dan las violaciones
uv run python solver-v2/evaluar_solver.py --golden golden/golden_tareas.json --solo-evaluar corridas/completo-r2 --solo A B C
```

Golden set del grupo (Taller 03 v2, Parte 2.a). Parte del golden del kit (solver-v2/golden_tareas.json, tareas A, B y C con sus 25 comprobaciones) y agrega: A09 (la curva de aprendizaje estandarizada, que el kit no comprobaba) y dos tareas reales de Matemáticas y Programación para IA (S1, S2) con 9 y 12 comprobaciones. La Tarea D del kit queda fuera: su paquete (hackathon3_student) no está en el repositorio. Entre todas: un notebook (B), un PDF (C) y la trampa (S210-S211: semilla y bootstrap pareado). Cada comprobación lleva en «_por_que» qué detecta; evaluar_solver.py ignora ese campo. Las verdad_py de S1 y S2 importan golden/verdades_reales.py: correr el evaluador DESDE LA RAÍZ del repo. Tipos de comprobación: ver el _comentario de solver-v2/golden_tareas.json. El campo «juez» es la pregunta, SIN la cifra verdadera, que la extensión (Parte 4, scripts/juez.py) le hace a un LLM de otra familia para medir su acuerdo con la comprobación.

## A — `../solver-v2/enunciados/tarea-a-generativo-discriminativo.pdf` → `reporte.md` (9 comprobaciones)

| id | tipo | qué detecta |
|---|---|---|
| A01 | archivo | El entregable que pide el enunciado, con su nombre exacto. |
| A02 | archivo | La Parte 3 exige la curva como figura PNG. |
| A03 | secciones | Secciones exigidas por el enunciado, en su orden. |
| A04 | palabras_max | Límite de 1 200 palabras. |
| A05 | cifra_presente | Tamaño del dataset (569 ejemplos): detecta que se cargó el conjunto correcto. |
| A06 | cifra | Exactitud de Naive Bayes sobre la partición 70/30 estratificada con semilla 42. |
| A07 | cifra | Exactitud de la regresión logística estandarizada sobre la misma partición. |
| A08 | procedencia | Procedencia: las cifras del reporte deben salir de una ejecución (respuesta a la 0.a). |
| A09 | cifra_presente | NUEVA. Curva de aprendizaje, regresión logística ESTANDARIZADA con el 50 % del entrenamiento: 0.9766. Sin estandarizar da 0.9415 o 0.9532 según max_iter. El kit no detectaba el error de la corrida del 2026-10-04 (T3 sin estandarizar, «los dos modelos» de la Parte 2): A07 mide la Parte 2, no la curva. Se usa el 50 % porque 0.9766 no coincide con ninguna otra cifra de la curva (el 5 %, 0.9298, sí coincide con la LR sin estandarizar al 25 %). |

## B — `../solver-v2/enunciados/tarea-b-temperatura-top-p.pdf` → `*.ipynb` (9 comprobaciones)

| id | tipo | qué detecta |
|---|---|---|
| B01 | archivo | El entregable es un notebook. |
| B02 | ipynb_ejecutado | El notebook se entrega EJECUTADO y sin errores. |
| B03 | cifra_presente | Entropía (bits) con T=0.5: cifra exacta que el evaluador recalcula. |
| B04 | cifra_presente | Entropía con T=1. |
| B05 | cifra_presente | Entropía con T=2. |
| B06 | cifra_presente | Masa de probabilidad renormalizada de t0 tras top-p. |
| B07 | contiene | Los tokens t0…t3 del núcleo aparecen nombrados. |
| B08 | archivo | La figura del muestreo existe (archivo o imagen embebida). |
| B09 | procedencia | Procedencia de las cifras del notebook. |

## C — `../solver-v2/enunciados/tarea-c-recuperacion-lexica-lsa.pdf` → `reporte.pdf` (8 comprobaciones)

| id | tipo | qué detecta |
|---|---|---|
| C01 | archivo | El entregable es un PDF. |
| C02 | paginas_max | Límite de dos páginas. |
| C03 | secciones | Secciones exigidas, en orden. |
| C04 | cifra | MRR de TF-IDF. |
| C05 | cifra | Hit@3 de TF-IDF. |
| C06 | cifra | MRR de LSA con TruncatedSVD(4, random_state=0). |
| C07 | contiene | La consulta que falla (q5) se nombra en el análisis. |
| C08 | procedencia | Procedencia. |

## S1 — `../tareas/reales/semana1/enunciado.pdf` → `reporte.md` (9 comprobaciones)

Matemáticas y Programación para IA, semana 1: telemetría con NumPy, Pandas y Parquet. Datos en tareas/reales/semana1/data/. Las verdades salen de golden/verdades_reales.py.

| id | tipo | qué detecta |
|---|---|---|
| S101 | archivo | Entregable principal. |
| S102 | archivo | El enunciado exige el Parquet en output/ (Tarea 5). Ruta estricta: un archivo enterrado en la carpeta de una subtarea no cumple el entregable. |
| S103 | secciones | «Una sección por tarea, en orden (Tarea 1 a Tarea 5)», tal como lo dice el enunciado. LIMITACIÓN conocida del evaluador del kit: acepta un encabezado de hasta 40 caracteres más que el nombre buscado, así que «## Tarea 2 - Normalizar y calcular un indicador de carga» (51) no se reconoce y «## Tarea 2» sí. Con los títulos completos pasa lo contrario: ninguna lista acepta las dos formas (exploratoria y completo-r1, 2026-10-05). Un fallo aquí se revisa a mano antes de atribuirlo al solver. |
| S104 | palabras_max | Máximo 800 palabras. |
| S105 | cifra_presente | Media de gpu_utilization (57.9931): las tres características en el orden exigido. |
| S106 | cifra_presente | TRAMPA SECUNDARIA. load_score de la fila 23 = 3.136218 con la desviación de NumPy (ddof=0), la que reproduce el «Resultado esperado». Con la .std() de pandas (ddof=1) sale 3.130986: la tolerancia 0.002 separa los dos. |
| S107 | cifra | Tabla resumen: potencia media de AI-SRV-01 (363.4598). |
| S108 | cifra | Tabla resumen: carga media de AI-SRV-06 = 0.6676 (con ddof=1 sale 0.6665). Comprueba que la tabla está completa (los seis servidores), no solo la fila de ejemplo. |
| S109 | procedencia | «Toda cifra del reporte debe provenir de una ejecución del código». |

## S2 — `../tareas/reales/semana2/enunciado.pdf` → `informe.md` (12 comprobaciones)

Matemáticas y Programación para IA, semana 2: comparación pareada de dos modelos. Contiene la TRAMPA del golden (S210-S211). El evaluador lee los números sin signo: las verdades negativas se escriben en valor absoluto.

| id | tipo | qué detecta |
|---|---|---|
| S201 | archivo | Entregable principal. |
| S202 | archivo | Tabla pareada exigida en output/ (Tarea 7). |
| S203 | archivo | Figura por tienda exigida en output/ (Tarea 3). |
| S204 | secciones | Las cinco partes del Anexo, en orden. |
| S205 | cifra_presente | Desviación estándar MUESTRAL de las diferencias (1.2534). Con ddof=0 sale 1.2136. |
| S206 | cifra_presente | Error estándar = sd/√n (0.3133). |
| S207 | cifra_presente | p-value bilateral de la prueba t de una muestra sobre las diferencias (0.2013). Una prueba t independiente (no pareada) da otro valor. |
| S208 | cifra_presente | p-value exacto por cambios de signo: 13 372 de 65 536 configuraciones (0.2040). La tolerancia lo separa del p de la prueba t (0.2013). |
| S209 | contiene | La semilla del bootstrap se declara (pregunta 17). |
| S210 | cifra_presente | TRAMPA. Límite inferior del bootstrap pareado: −0.1938 con default_rng(20260829) y tiendas completas. Remuestrear A y B por separado da −1.169; usar np.random.seed (la semilla global antigua) da −0.169. Las dos violaciones de la Tarea 6 fallan aquí. |
| S211 | cifra_presente | TRAMPA. Límite superior: 1.0063 (separado: 2.000; semilla global: 0.9939). |
| S212 | procedencia | «Toda cifra del informe debe provenir de una ejecución del código». |

## Validación sin gastar H200

- A09 aprueba en las corridas con la curva estandarizada (0.976608 en el resultados.json de la T3 de `completo`, `completo-r1` y `completo-r2`) y **falla** en `corridas/prueba/tarea-a-generativo-discriminativo/` (la T3 sin `StandardScaler`, 2026-10-04): la falla que el kit no veía.
- Las trampas de S2 se reproducen con `python golden/verdades_reales.py` (`s2_trampas`).
- En la exploratoria (2026-10-05) la T6 de S2 calculó el intervalo correcto [−0.19375, 1.00625] pero la corrida agotó el presupuesto antes de aprobarla: S210 y S211 fallaron por presupuesto, no por la trampa.
