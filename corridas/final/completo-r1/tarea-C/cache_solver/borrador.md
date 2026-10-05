## Objetivo

Comparar la recuperación léxica (TF-IDF con similitud coseno) contra la semántica latente (LSA mediante TruncatedSVD a 4 dimensiones) sobre un corpus de 10 documentos y 6 consultas con un documento relevante cada una. Se miden Hit@1, Hit@3 y MRR, se identifican las consultas que cada método falla, se explican los fallos mediante los términos compartidos entre consulta y documentos, y se discute por qué un recuperador denso resolvería el caso fallado y qué costo implica.

## Método

- **Corpus y juicios:** 10 documentos (d01–d10) y 6 consultas (q1–q6), con relevantes q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02.
- **Línea base (Parte 1):** `TfidfVectorizer` de scikit-learn con parámetros por defecto, ajustado solo con los 10 documentos; similitud coseno consulta-documento y ordenamiento de los 10 documentos por consulta.
- **LSA (Parte 2):** `TruncatedSVD(n_components=4, random_state=0)` sobre la matriz TF-IDF de los documentos; las consultas se proyectan con la misma base latente y se repite el coseno.
- **Evaluación:** Hit@1, Hit@3 y MRR sobre las 6 consultas; fallo = relevante fuera del top-3. Nada se ajusta con las consultas.

## Resultados

**Tabla 1 — Métricas de los dos métodos (Parte 2).**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (línea base) | 0.8333 | 0.8333 | 0.8611 |
| LSA (k=4) | 0.6667 | 0.8333 | 0.7500 |

En TF-IDF, Hit@1 y Hit@3 coinciden porque el único fallo (q5) cae fuera del top-3. En LSA se separan: q4 pierde el primer puesto sin salir del top-3, mientras q5 sigue fallando.

**Tabla 2 — Posición del relevante por consulta (Parte 3).**

| Consulta | Relevante | Rank TF-IDF | Rank LSA |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | 6 | 6 |
| q6 | d02 | 1 | 1 |

TF-IDF falla solo q5; LSA también falla q5 y además degrada q4, donde d01 y d10 superan al relevante en el espacio latente.

![Métricas comparativas TF-IDF vs. LSA](figuras/T2_t2_metricas_comparativas.png)

![Términos compartidos entre q5 y los documentos](figuras/T3_t3_terminos_compartidos.png)

## Discusión

**Por qué falla q5 en ambos métodos (Parte 3).** La consulta q5 («¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?») y su relevante d09 («La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad») no comparten ningún término de contenido: su único término común es «la», puramente funcional. Sin solapamiento léxico de contenido, TF-IDF solo puede puntuar por funcionales o por contenido ajeno: d03 encabeza porque comparte «modelo», y d06, d04, d01 y d10 la superan compartiendo únicamente funcionales («el», «qué», «que», «al», «la»); el relevante queda fuera del top-3 (Tabla 2). LSA no repara el fallo: su espacio latente se aprende de co-ocurrencias dentro de estos 10 documentos y, con 4 componentes que retienen solo 39.53 % de la varianza, proyecta q5 cerca de documentos que comparten con la consulta «modelo» o funcionales (d03 alcanza 0.6140); d09 permanece fuera del top-3. La proyección incluso perjudica a q4. En un corpus mínimo, LSA solo suaviza co-ocurrencias ya observadas; no puede crear la conexión «determinista» ↔ «temperatura/probabilidad», que no aparece en los datos.

**Recuperador denso (Parte 4).** Un codificador entrenado a gran escala representa significado: habría aprendido que «más determinista al elegir la siguiente palabra» parafrasea «valores bajos concentran la probabilidad» (baja temperatura), de modo que q5 y d09 quedarían cercanos aunque no compartan ninguna palabra. TF-IDF exige solapamiento léxico —funciona cuando lo hay, como en q2, donde d03 es el único documento con similitud no nula— y LSA solo captura co-ocurrencia dentro del corpus, insuficiente para inferir sinonimia o paráfrasis. Costos frente a TF-IDF: exige un modelo entrenado (preentrenado o afinado) y su mantenimiento; la inferencia es más cara, pues cada consulta y documento pasan por una red neuronal frente a un producto punto disperso; la indexación requiere vectores densos de muchas dimensiones y, a escala, índices ANN aproximados, frente a un índice invertido compacto; y es menos interpretable, con riesgo de degradarse fuera del dominio de entrenamiento. TF-IDF no entrena nada, es exacto, barato e interpretable, pero queda limitado al solapamiento de términos, como muestra el fallo de q5 en ambos métodos.
