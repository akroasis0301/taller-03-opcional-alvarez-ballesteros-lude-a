## Objetivo

Comparar la recuperación léxica (TF-IDF + similitud coseno) contra la semántica latente (LSA mediante TruncatedSVD con k=4) sobre un corpus de 10 documentos cortos y 6 consultas con un documento relevante cada una (q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02), midiendo Hit@1, Hit@3 y MRR, y analizar en qué consultas falla cada método y por qué.

## Método

**Línea base léxica.** `TfidfVectorizer` de scikit-learn con parámetros por defecto (minúsculas, tokens de 2+ caracteres, sin stopwords ni stemming, norma l2, smooth_idf), ajustado sobre los 10 documentos; las consultas solo se transforman. El vocabulario resultante tiene 105 términos (matriz 10×105). El ranking se obtiene por similitud coseno entre cada consulta y los diez documentos.

**Semántica latente.** Sobre la misma matriz TF-IDF se ajusta `TruncatedSVD(n_components=4, random_state=0)`; las consultas se proyectan con el mismo modelo y se repite el ranking por coseno en el espacio latente de 4 dimensiones. Los cuatro valores singulares son 1.3172, 1.0401, 1.0225 y 0.9974, con una varianza explicada total de 0.3953.

**Evaluación.** Hit@1, Hit@3 y MRR agregados sobre las 6 consultas; se define fallo como relevante fuera del top 3. Para el análisis se extrae el solape de términos consulta-documento con el analizador del propio vectorizador.

## Resultados

**Tabla 1. Métricas agregadas (6 consultas).**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (coseno) | 0.8333 | 0.8333 | 0.8611 |
| LSA (TruncatedSVD k=4) | 0.6667 | 0.8333 | 0.75 |

![Comparativa de métricas TF-IDF vs LSA](figuras/T2_comparativa_metricas_tfidf_lsa.png)

**Tabla 2. Puesto del documento relevante por consulta.**

| Consulta | Relevante | Rank TF-IDF | Rank LSA |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | 6 | 6 |
| q6 | d02 | 1 | 1 |

**Tabla 3. Fallos (relevante fuera del top 3) y solape léxico con el relevante.**

| Método | Consulta fallida | Rank relevante | Términos compartidos con el relevante |
|---|---|---|---|
| TF-IDF | q5 | 6 | solo "la" (Σ IDF = 1.4520) |
| LSA | q5 | 6 | solo "la" (Σ IDF = 1.4520) |

![Mapa de calor de similitud consulta-documento (TF-IDF)](figuras/T1_t1_mapa_calor_similitud.png)

## Discusión

**Por qué falla q5 en ambos métodos.** La consulta q5 ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?") comparte con su relevante d09 únicamente el artículo "la" (Σ IDF = 1.4520); todo su léxico de contenido —"determinista", "elegir", "hace", "más", "palabra", "sea", "siguiente"— está fuera del vocabulario de 105 términos. El ranking lo deciden pues palabras funcionales: d03 (pos 1 en TF-IDF, score 0.3107 frente a 0.0790 de d09) comparte "al", "modelo", "la"; d06 comparte "qué", "el"; d04 comparte "que", "la". La proyección latente no puentea la distancia: d09 sigue en el puesto 6 (score LSA 0.4999), porque con k=4 y solo 0.3953 de varianza explicada, LSA reorganiza la geometría del corpus pero no puede inyectar sinónimos que no aparecen en ningún documento; su top 3 (d04, d01, d03) sigue ordenado por solapes funcionales.

**Efecto de LSA en el resto.** LSA solo degrada q4 (rank 1→3): la proyección difumina el emparejamiento léxico fuerte de q4 con d05 ("vectores", "coseno", "orientación"), lo que cuesta Hit@1 (0.8333→0.6667) y baja el MRR de 0.8611 a 0.75, sin ganar ningún Hit@3. Con un corpus mínimo y consultas casi copia de su relevante, la expansión semántica no aporta y sí introduce ruido.

**Recuperador denso para q5 y su costo.** Un recuperador denso con embeddings de un modelo entrenado resolvería q5 mejor porque aprende, en grandes corpus, que "más determinista al elegir la siguiente palabra" es parafraseable por "valores bajos concentran la probabilidad" (d09): la similitud operaría en un espacio donde sinonimia y paráfrasis están cerca aunque el solape léxico sea nulo, exactamente el caso que TF-IDF y LSA no pueden cubrir. El costo frente a TF-IDF es triple: (1) datos y cómputo de entrenamiento del codificador, frente a un ajuste de una pasada sobre 10 documentos; (2) latencia de inferencia neuronal por consulta y documento, frente a un producto punto disperso e interpretable; y (3) mantenimiento del índice, pues cambiar de modelo obliga a re-calcular los embeddings de todo el corpus, mientras el vocabulario TF-IDF es estático y auditable.

![Solape de términos consulta-documento](figuras/T3_T3_solape_consulta_documento.png)
