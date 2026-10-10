## Objetivo

Comparar dos recuperadores sobre un corpus mínimo de 10 documentos (d01–d10) y 6 consultas con un documento relevante cada una: (1) la línea base léxica TF-IDF con similitud coseno y (2) la semántica latente (LSA) obtenida proyectando la misma matriz TF-IDF a 4 dimensiones. Se evalúa Hit@1, Hit@3 y MRR, se analizan los fallos por consulta con evidencia léxica y se discute qué aportaría un recuperador denso entrenado.

## Método

TF-IDF: `TfidfVectorizer` con parámetros por defecto, vocabulario ajustado solo con los 10 documentos (105 términos) y consultas transformadas con ese vocabulario; ranking de los 10 documentos por similitud coseno. LSA: `TruncatedSVD(n_components=4, random_state=0)` ajustado sobre la matriz TF-IDF de documentos; las consultas se proyectan con el mismo SVD y se repite el ranking coseno. Los 4 componentes explican 0.3953 de la varianza (0.0342, 0.1266, 0.1160 y 0.1186 por componente). Métricas agregadas sobre las 6 consultas: Hit@1, Hit@3 y MRR. Criterio de fallo: relevante fuera del top 3. Como evidencia léxica se extrae la intersección de tokens entre cada consulta y su documento relevante, filtrada por el vocabulario medido de TF-IDF.

## Resultados

**Tabla 1 (Parte 2) — Métricas de los dos métodos:**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (léxico) | 0.8333 | 0.8333 | 0.8611 |
| LSA (k=4) | 0.8333 | 0.8333 | 0.7500 |

![Métricas Hit@1, Hit@3 y MRR de TF-IDF frente a LSA](figuras/T2_t2_metricas_tfidf_vs_lsa.png)

**Tabla 2 (Parte 3) — Posición del relevante y fallos por consulta:**

| Consulta | Relevante | Pos. TF-IDF | Pos. LSA | Fallo (fuera de top 3) |
|---|---|---|---|---|
| q1 | d04 | 1 | 1 | ninguno |
| q2 | d03 | 1 | 1 | ninguno |
| q3 | d08 | 1 | 1 | ninguno |
| q4 | d05 | 1 | 3 | ninguno |
| q5 | d09 | 6 | 6 | TF-IDF y LSA |
| q6 | d02 | 1 | 1 | ninguno |

Ambos métodos fallan únicamente en q5: d09 queda en la posición 6. En TF-IDF su similitud es 0.0790, por detrás de d03 (0.3107), d06 (0.2009) y d04 (0.1896); en LSA es 0.4999, por detrás de d04 (0.9427) y d01 (0.8066). LSA además degrada q4: d05 cae del puesto 1 al 3 (0.8984), superado por d01 (0.9220) y d10 (0.9206); sigue en el top 3, pero pierde el primer puesto y explica la caída del MRR de 0.8611 a 0.7500.

Evidencia léxica: q1 comparte con d04 seis términos de contenido (función, ranking, léxica, pondera, frecuencia, términos), lo que produce la similitud más alta de la matriz (0.6891) y el puesto 1. En cambio, q5 y d09 comparten un único token del vocabulario de 105 términos: «la», una palabra funcional; los términos de contenido de la consulta (determinista, palabra, elegir) no aparecen en d09, y los de d09 (temperatura, logits, softmax, probabilidad) no aparecen en q5. Los documentos que encabezan q5 solo ganan por solapamiento espurio: d03 comparte el término de contenido «modelo» (más funcionales), y d06 comparte «qué» y «el».

![Intersección léxica entre cada consulta y su documento relevante](figuras/T3_T3_mapa_interseccion.png)

## Discusión

El fallo de q5 es un fallo de vocabulario, no de ponderación: sin términos de contenido compartidos, TF-IDF no tiene señal que puntuar, y las palabras funcionales («la», «el», «qué») deciden el ranking. LSA tampoco lo resuelve: con 4 componentes que capturan solo 0.3953 de la varianza de una matriz de 10×105, la semántica latente únicamente recombina co-ocurrencias presentes en el corpus; como q5 y d09 no co-ocurren en ningún término de contenido, no hay asociación que propagar, y el suavizado incluso promueve documentos genéricos (d04, d01) que cargan en los componentes dominantes — de ahí también el retroceso de q4 y del MRR.

Un recuperador denso resolvería q5 porque sus embeddings, entrenados en corpus masivos, codifican paráfrasis y sinonimia: acercarían «más determinista al elegir la siguiente palabra» a «temperatura baja concentra la probabilidad» aunque no compartan ninguna palabra de contenido. TF-IDF y LSA, ajustados solo con 10 documentos, carecen de esa supervisión semántica externa. El costo frente a TF-IDF es doble: requiere un modelo preentrenado (memoria para sus pesos y para el índice de embeddings de los documentos, cómputo de inferencia por consulta y por documento) y añade dependencia del modelo y de su dominio de entrenamiento; TF-IDF solo cuenta términos en un espacio de 105 dimensiones, es inmediato de entrenar, interpretable y sin dependencias. En este corpus el intercambio solo compensa cuando la consulta parafrasea al documento (q5); para consultas con solapamiento léxico (q1–q4, q6) el método léxico ya recupera el relevante en el puesto 1.
