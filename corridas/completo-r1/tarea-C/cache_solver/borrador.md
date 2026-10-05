## Objetivo

Comparar dos aproximaciones de recuperación sobre un corpus mínimo de 10 documentos (d01–d10) y 6 consultas (q1–q6) con un documento relevante cada una: (1) una línea base léxica TF-IDF con similitud coseno y (2) su proyección semántica latente LSA a 4 dimensiones. Se cuantifica el desempeño con Hit@1, Hit@3 y MRR, se diagnostican los fallos por consulta a partir de los términos compartidos entre consulta y documento relevante, y se discute qué aportaría —y qué costaría— un recuperador denso.

## Método

- **Datos:** 10 documentos y 6 consultas del enunciado, con un juicio de relevancia por consulta.
- **Línea base léxica:** `TfidfVectorizer` de scikit-learn con parámetros por defecto (minúsculas, `token_pattern (?u)\b\w\w+\b`), ajustado con los 10 documentos (vocabulario de 105 términos); similitud coseno consulta–documento y ordenamiento de los 10 documentos por consulta.
- **Semántica latente:** la misma matriz TF-IDF se proyecta con `TruncatedSVD(n_components=4, random_state=0)`; las consultas se proyectan al espacio latente (folding-in, q·Vᵀ) y se recalcula la similitud coseno.
- **Métricas** sobre las 6 consultas: Hit@1, Hit@3 y MRR. Criterio de fallo: relevante fuera del top 3.

## Resultados

**Tabla 1 — Métricas globales (Parte 2):**

| Método  | Hit@1  | Hit@3  | MRR    |
|---------|--------|--------|--------|
| TF-IDF  | 0.8333 | 0.8333 | 0.8611 |
| LSA-4   | 0.6667 | 0.8333 | 0.7500 |

![Mapa de calor de similitud coseno TF-IDF entre las 6 consultas y los 10 documentos](figuras/T1_t1_heatmap_similitud.png)

**Tabla 2 — Posición del documento relevante por consulta (Parte 3):**

| Consulta | Relevante | Pos. TF-IDF | Pos. LSA |
|----------|-----------|-------------|----------|
| q1       | d04       | 1           | 1        |
| q2       | d03       | 1           | 1        |
| q3       | d08       | 1           | 1        |
| q4       | d05       | 1           | 3        |
| q5       | d09       | **6**       | **6**    |
| q6       | d02       | 1           | 1        |

Ambos métodos fallan únicamente en q5: d09 queda en la posición 6. En TF-IDF, d09 obtiene similitud 0.0790 con q5, superado por d03 (0.3107), d06 (0.2009) y d04 (0.1896). LSA no corrige q5 y además degrada q4: d05 cae de la posición 1 a la 3 (similitud latente 0.8984, por detrás de d01 y d10), lo que explica la caída de Hit@1 de 0.8333 a 0.6667 sin ganancia en Hit@3. En el resto, LSA concentra similitudes cercanas a 1 (p. ej., q2–d03: 0.9754; q1–d04: 0.9542) sin alterar el top-1.

![Análisis de fallos por consulta y términos compartidos entre cada consulta y su documento relevante](figuras/T3_t3_analisis_consultas.png)

## Discusión

**Por qué falla cada método (Parte 3).** En q1–q4 y q6 la consulta repite los términos distintivos de su relevante: q1/d04 comparten «función de ranking léxica», «pondera», «frecuencia», «términos» (similitud 0.6891); q3/d08 comparten «entrena», «matrices», «bajo», «rango» (0.5604); q6/d02 comparten «autoatención», «capas», «feed-forward» (0.5674); q2/d03 comparten «fragmentos», «prompt» (0.5174) y q4/d05 comparten «compara», «orientación», «vectores» (0.4569). Ese solapamiento léxico coloca al relevante en la posición 1. En q5 no existe ese puente: la consulta habla de «determinista», «elegir» y «siguiente palabra», mientras d09 habla de «temperatura», «logits», «softmax» y «probabilidad»; el único token compartido es el artículo «la». Es un fallo puro de desajuste léxico (sinonimia): los documentos que lo superan (d03, d06, d04) se apoyan en coincidencias débiles o palabras funcionales, no en afinidad temática. LSA no lo resuelve porque con 10 documentos y 4 componentes la co-ocurrencia es demasiado escasa para aprender la relación determinismo↔temperatura; además, la compresión distorsiona q4 y sacrifica Hit@1.

**Recuperador denso (Parte 4).** Un codificador entrenado a gran escala mapearía «más determinista al elegir la siguiente palabra» y «valores bajos concentran la probabilidad» a regiones cercanas del espacio vectorial, porque aprende sinonimia y paráfrasis más allá del vocabulario compartido; resolvería q5 incluso con solapamiento léxico nulo, exactamente el caso que TF-IDF y LSA fallan (posición 6 ambos). Sus costos frente a TF-IDF son: (i) necesita un modelo preentrenado, con el costo de entrenamiento y los datos que ello implica; (ii) mayor cómputo de inferencia por consulta y documento, y un índice de vectores (típicamente ANN) en lugar de un índice invertido ligero; (iii) riesgo de desajuste de dominio entre el modelo y el corpus; (iv) menor interpretabilidad: TF-IDF permite inspeccionar qué términos emparejan (como en el análisis anterior), mientras el embedding es opaco. En un corpus de 10 documentos, TF-IDF es exacto, sin entrenamiento y suficiente salvo para fallos de sinonimia como q5.
