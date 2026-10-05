## Objetivo

Comparar la recuperación léxica (TF-IDF + similitud coseno) contra la semántica latente (LSA mediante TruncatedSVD a 4 dimensiones) sobre un corpus de 10 documentos y 6 consultas con un documento relevante cada una, midiendo Hit@1, Hit@3 y MRR, y diagnosticar por qué falla cada método mirando el solapamiento léxico consulta–documento.

## Método

Se construyó el corpus d01–d10 y las consultas q1–q6 con sus juicios binarios. La línea base usa `TfidfVectorizer` de scikit-learn con parámetros por defecto, ajustado solo con los 10 documentos (vocabulario de 105 términos); las consultas se transforman con el mismo vectorizador y se rankean por similitud coseno. LSA proyecta esa matriz con `TruncatedSVD(n_components=4, random_state=0)`; las consultas se proyectan al mismo espacio latente (folding-in) y se rankean por coseno en él. Los empates se resuelven por el orden original de los documentos. Las 4 componentes capturan 0.3953 de la varianza (0.0342, 0.1266, 0.1160 y 0.1186 por componente). Se reportan Hit@1, Hit@3 y MRR sobre las 6 consultas.

## Resultados

**Tabla comparativa (Parte 2):**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF | 0.8333 | 0.8333 | 0.8611 |
| LSA(4) | 0.6667 | 0.8333 | 0.75 |

![Métricas Hit@1, Hit@3 y MRR de TF-IDF y LSA(4)](figuras/T2_T2_barras_metricas.png)

**Posición del relevante por consulta (Parte 3; fallo = posición > 3):**

| Consulta | Relevante | Pos. TF-IDF | Pos. LSA(4) |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | **6** | **6** |
| q6 | d02 | 1 | 1 |

TF-IDF falla solo en q5; LSA(4) también falla solo en q5, y además degrada q4 del puesto 1 al 3 (d05 con 0.8984 frente a d01 con 0.9220). En q5, la similitud del relevante d09 es 0.0790 con TF-IDF y 0.4999 con LSA; el top 3 lo ocupan d03 (0.3107), d06 (0.2009) y d04 (0.1896) en TF-IDF, y d04 (0.9427), d01 (0.8066) y d03 (0.6141) en LSA.

![Similitudes de q5 contra los 10 documentos en TF-IDF y LSA(4)](figuras/T3_t3_q5_similitudes.png)

## Discusión

**Fallo de q5.** La consulta y su relevante d09 comparten un único término del vocabulario: «la» (peso 0.2440 en la consulta, 0.3238 en d09, IDF 1.4520, DF 6/10), una palabra funcional sin capacidad discriminativa; no hay solapamiento de términos de contenido. Los términos de contenido de la consulta (determinista, elegir, palabra, siguiente) están fuera del vocabulario, y los característicos de d09 (temperatura, logits, softmax, probabilidad) no aparecen en la consulta: la relación consulta–documento es una paráfrasis, invisible para el matching léxico. LSA no lo corrige: aunque la similitud de d09 sube a 0.4999, el espacio de 4 dimensiones retiene solo 0.3953 de la varianza y está dominado por co-ocurrencias globales, de modo que d04 alcanza 0.9427 y d09 permanece en el puesto 6.

![Términos compartidos entre q5 y d09 según el vocabulario TF-IDF](figuras/T3_t3_q5_tokens.png)

**Recuperador denso (Parte 4).** Un modelo de embeddings entrenado colocaría «más determinista» cerca de «temperatura baja concentra la probabilidad» en el espacio semántico, resolviendo q5 sin necesidad de solapamiento léxico, que es exactamente el diagnóstico del fallo anterior. Sus costos frente a TF-IDF son: necesita datos y cómputo de entrenamiento (o depender de un modelo preentrenado), la inferencia es más cara por consulta (codificación neuronal frente a un producto disperso), el índice de vectores densos crece con la dimensión del modelo, y la interpretabilidad es menor (dimensiones latentes opacas frente a pesos por término visible); además puede sufrir desajuste de dominio entre sus datos de entrenamiento y el corpus.
