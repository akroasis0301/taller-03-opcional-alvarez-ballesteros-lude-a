# Tarea C — Recuperación léxica contra semántica latente sobre un corpus mínimo

MMIA 6013 · Taller 03 v2

## Objetivo

Comparar un recuperador léxico (TF-IDF con similitud coseno) con su proyección semántica latente (LSA, `TruncatedSVD` a 4 dimensiones) sobre un corpus de 10 documentos y 6 consultas con un documento relevante cada una, midiendo Hit@1, Hit@3 y MRR, e identificar las consultas que cada método falla y por qué. En esta entrega solo la línea base léxica (Parte 1) produjo resultados medidos: la ejecución de LSA (Parte 2) falló y el análisis de solapamiento por consulta (Parte 3) fue omitido, como se declara en Resultados.

## Método

- **Corpus y juicios:** 10 documentos (d01–d10) y 6 consultas (q1–q6), con un relevante por consulta: q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02.
- **Parte 1 (léxica):** `TfidfVectorizer` con parámetros por defecto, ajustado sobre los 10 documentos (vocabulario de 105 términos); las consultas se vectorizan con ese vocabulario y se ordenan los 10 documentos por similitud coseno.
- **Métricas:** Hit@1, Hit@3 y MRR (recíproco del rango del relevante), promediados sobre las 6 consultas.
- **Parte 2 (LSA):** proyección de la misma matriz TF-IDF con `TruncatedSVD(n_components=4, random_state=0)` y reevaluación idéntica. La subtarea T2 falló: no hay cifras de LSA.
- **Parte 3:** términos compartidos consulta–relevante por consulta. La subtarea T3 fue omitida: no hay cifras.

## Resultados

**Tabla 1 — TF-IDF + coseno: métricas por consulta y agregadas.**

| Consulta | Relevante | Posición | Sim. con relevante | Hit@1 | Hit@3 | RR |
|---|---|---|---|---|---|---|
| q1 | d04 | 1 | 0.6891 | 1.0 | 1.0 | 1.0 |
| q2 | d03 | 1 | 0.5174 | 1.0 | 1.0 | 1.0 |
| q3 | d08 | 1 | 0.5604 | 1.0 | 1.0 | 1.0 |
| q4 | d05 | 1 | 0.4569 | 1.0 | 1.0 | 1.0 |
| q5 | d09 | 6 | 0.0790 | 0.0 | 0.0 | 0.1667 |
| q6 | d02 | 1 | 0.5674 | 1.0 | 1.0 | 1.0 |
| **Promedio (6 consultas)** | — | — | — | **0.8333** | **0.8333** | **0.8611** |

**Declaración de subtareas sin resultado.** La Parte 2 (T2, LSA) falló: no se reportan sus cifras, y la tabla comparativa de los dos métodos exigida queda reducida a la columna medida de TF-IDF. La Parte 3 (T3, términos compartidos por consulta) fue omitida: no se reportan sus cifras. Al no existir rankings de LSA, no puede determinarse qué consultas falla ese método.

![Mapa de calor de la similitud coseno consultas × documentos (TF-IDF, vocabulario de 105 términos).](figuras/T1_t1_mapa_calor_similitud.png)

![Rango recíproco del relevante por consulta (TF-IDF): 1.0 en q1–q4 y q6; 0.1667 en q5.](figuras/T1_t1_rr_por_consulta.png)

Detalle del único fallo (ranking TF-IDF de q5: d03, d06, d04, d01, d10, d09, …):

| Par q5–documento | Similitud |
|---|---|
| d03 (puesto 1) | 0.3107 |
| d06 (puesto 2) | 0.2009 |
| d04 (puesto 3) | 0.1896 |
| **d09 (relevante, puesto 6)** | **0.0790** |

## Discusión

**Lectura de los resultados.** TF-IDF coloca el relevante en el puesto 1 en 5 de 6 consultas (Hit@1 = Hit@3 = 0.8333; MRR = 0.8611). En esos aciertos la similitud con el relevante es alta (0.4569–0.6891), coherente con un solapamiento léxico directo de términos de contenido entre consulta y documento; q1–d04 alcanza 0.6891, el valor máximo de toda la matriz.

**El fallo en q5.** La consulta pregunta por qué el modelo es más determinista al elegir la siguiente palabra, y el relevante d09 habla de temperatura, logits, softmax y probabilidad: no comparten términos de contenido. La similitud medida con d09 (0.0790) es la más baja de los seis pares relevantes, compatible con que solo pesen palabras funcionales. Los tres primeros puestos (d03, d06, d04), que no tratan el tema, superan a d09 porque cualquier peso compartido, aunque poco discriminativo, basta cuando el relevante casi no comparte vocabulario. La verificación término a término (Parte 3) no se ejecutó: esta explicación se apoya en las similitudes medidas y en la lectura de los textos.

**Por qué un recuperador denso resolvería q5.** Un codificador entrenado a gran escala proyecta a vectores próximos textos con significado afín aunque no compartan palabras: "elegir la siguiente palabra de forma más determinista" y "temperatura baja concentra la probabilidad" describen el mismo fenómeno y quedarían cerca en el espacio de embeddings. TF-IDF, con un vocabulario de solo 105 términos del corpus, exige coincidencia léxica superficial y por eso falla. La LSA de la Parte 2 podría capturar parte de esa co-ocurrencia, pero su ejecución falló y su efecto no queda medido aquí; además, con 10 documentos y 4 dimensiones el espacio latente es demasiado grueso para fiar la recuperación a él.

**Costo frente a TF-IDF.** El recuperador denso exige un modelo de embeddings preentrenado (memoria y dependencia externa), cómputo de inferencia para codificar documentos y consultas, construcción de un índice vectorial y mayor latencia. TF-IDF no requiere entrenamiento ni modelo, es determinista, barato y transparente, y preserva la coincidencia exacta de términos, valiosa cuando la consulta usa el vocabulario literal del documento, como en q1–q4 y q6, todas resueltas en el puesto 1. El beneficio semántico se paga con complejidad y costo computacional.
