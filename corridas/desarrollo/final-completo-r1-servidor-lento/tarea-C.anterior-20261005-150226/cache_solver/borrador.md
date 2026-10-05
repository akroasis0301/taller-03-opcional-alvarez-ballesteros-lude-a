## Objetivo

Comparar dos recuperadores sobre un corpus mínimo de 10 documentos y 6 consultas (cada una con un único documento relevante): (1) la línea base léxica TF-IDF con similitud coseno y (2) su proyección semántica latente LSA a 4 dimensiones. Se evalúan Hit@1, Hit@3 y MRR, se identifican las consultas donde cada método falla (relevante fuera del top 3) y se discute por qué un recuperador denso resolvería mejor el caso fallido y a qué costo.

## Método

Ambos métodos parten de la misma matriz TF-IDF, generada con `TfidfVectorizer` de scikit-learn (parámetros por defecto) ajustado solo con los 10 documentos (vocabulario de 105 términos). La similitud coseno consulta-documento ordena los 10 documentos por consulta; los empates se resuelven por orden del corpus (d01…d10). En LSA, `TruncatedSVD(n_components=4, random_state=0)` se ajusta sobre la matriz TF-IDF de los documentos y las consultas se proyectan con el mismo modelo; la similitud se calcula en el espacio latente de 4 dimensiones. Sobre las 6 consultas se reportan Hit@k (proporción de consultas con el relevante entre los k primeros) y MRR (media del recíproco del rango del relevante). Un fallo es que el relevante quede fuera del top 3.

## Resultados

**Tabla 1 — Métricas globales de los dos métodos (Parte 2).**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF | 0.8333 | 0.8333 | 0.8611 |
| LSA (k=4) | 0.6667 | 0.8333 | 0.7500 |

![Comparativa de métricas TF-IDF frente a LSA](figuras/T2_comparativa_tfidf_vs_lsa.png)

**Tabla 2 — Posición del documento relevante por consulta y método (Parte 3).**

| Consulta | Relevante | Pos. TF-IDF | Pos. LSA |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | 6 | 6 |
| q6 | d02 | 1 | 1 |

![Posición del relevante por consulta y método](figuras/T3_T3_posicion_relevante.png)

Ambos métodos fallan únicamente en q5 (relevante d09 en la posición 6). En TF-IDF, d09 obtiene similitud 0.0790 frente a 0.3107 del top-1 (d03); en LSA, d09 sube a 0.4999, pero d04 sube más (0.9427) y el rango no mejora. LSA además degrada q4 del puesto 1 al 3. Los aciertos se apoyan en solapamiento léxico alto: en q1, consulta y d04 comparten 8 términos del vocabulario, 6 de contenido (léxica, ranking, términos, frecuencia, pondera, función), con similitud 0.6891.

## Discusión

**Por qué fallan ambos en q5.** La consulta ("más determinista al elegir la siguiente palabra") y d09 ("la temperatura reescala los logits… valores bajos concentran la probabilidad") no comparten ningún término de contenido del vocabulario: el único término común es el funcional "la". TF-IDF, que exige coincidencia exacta de tokens, produce similitud 0.0790. El top-1 d03 se explica por "modelo": d03 es el único documento que lo contiene, su IDF es alto y domina la similitud aunque el resto de la consulta no coincida. LSA no corrige el fallo: con solo 10 documentos, los 4 factores latentes (varianza explicada total 0.3953) capturan co-ocurrencia genérica del corpus, no significado; la proyección acerca a d09 (0.0790 → 0.4999), pero acerca más a documentos genéricos como d04 (0.9427), y el relevante permanece en el puesto 6.

**Recuperador denso frente a TF-IDF.** Un modelo de embeddings entrenado a gran escala aprende a colocar paráfrasis y sinónimos cerca en el espacio vectorial: representaría "más determinista al elegir la siguiente palabra" próximo a "temperatura baja concentra la probabilidad" aunque no compartan ningún token, cubriendo justo el hueco que ni TF-IDF ni una LSA entrenada con 10 documentos pueden cubrir. Sus costos frente a TF-IDF son: necesita un modelo preentrenado (datos y cómputo de entrenamiento que este escenario no tiene), más memoria y cómputo de inferencia por documento y por consulta, mayor latencia (especialmente con índice vectorial), menor interpretabilidad —TF-IDF permite inspeccionar el peso de cada término compartido, el embedding es opaco— y riesgo de degradación por desajuste de dominio. TF-IDF sigue siendo sin entrenamiento, barato y explicable; el recuperador denso compra cobertura semántica pagando esos costos.
