## Objetivo

Comparar, sobre un corpus mínimo de 10 documentos en español con 6 consultas (cada una con un documento relevante), la recuperación léxica (TF-IDF + similitud coseno) contra la semántica latente (LSA mediante TruncatedSVD a 4 dimensiones). Se evalúan Hit@1, Hit@3 y MRR sobre las seis consultas y se analiza en qué consultas falla cada método.

## Método

- **Corpus y juicios:** documentos d01–d10 y consultas q1–q6, con relevancia q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02.
- **Línea base léxica (T1):** `TfidfVectorizer` de scikit-learn con parámetros por defecto (sin eliminación de palabras vacías); el vocabulario medido tiene 105 términos. Para cada consulta se rankean los 10 documentos por similitud coseno.
- **Semántica latente (T2):** la misma matriz TF-IDF se proyecta a 4 dimensiones con `TruncatedSVD(n_components=4, random_state=0)`; consultas y documentos se comparan por coseno en el espacio latente.
- **Métricas:** Hit@1, Hit@3 y MRR promediadas sobre las 6 consultas.

## Resultados

**Tabla única de métricas (Parte 2):**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF + coseno | 0.8333 | 0.8333 | 0.8611 |
| LSA (TruncatedSVD, 4D) + coseno | 0.6667 | 0.8333 | 0.7500 |

Diferencia (LSA − TF-IDF): Hit@1 −0.1667, Hit@3 0.0000, MRR −0.1111.

**Posición del documento relevante por consulta** (medida en T1/T2):

| Consulta | Relevante | Pos. TF-IDF | Pos. LSA |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | 6 | 6 |
| q6 | d02 | 1 | 1 |

![Mapa de calor de la similitud coseno TF-IDF entre las seis consultas y los diez documentos](figuras/T1_t1_heatmap_similitud_tfidf.png)

![Posición del documento relevante por consulta, TF-IDF frente a LSA](figuras/T2_posicion_relevante_por_consulta.png)

**Declaración:** la subtarea T3 (Parte 3, análisis de términos compartidos entre consulta y documentos) no tiene resultado (fallida): no se reportan sus cifras. Las subtareas T4 y T5, que dependían de T3, tampoco se ejecutaron; la Discusión se argumenta directamente con las medidas de T1 y T2.

## Discusión

**TF-IDF** resuelve 5 de 6 consultas en el puesto 1 y solo falla q5: el relevante d09 queda en el puesto 6, con similitud medida 0.0790, muy por debajo de d03 (0.3107), que encabeza el ranking sin ser relevante. A falta del análisis sistemático de T3, la inspección directa del vocabulario medido (105 términos) lo explica: q5 («…más determinista al elegir la siguiente palabra») y d09 («La temperatura reescala los logits antes del softmax…») no comparten ningún término de contenido —solo la palabra funcional «la»—, mientras que d03 comparte con la consulta el término de contenido «modelo» (además de «al» y «la»). Es un fallo de cobertura léxica, no de ordenación.

**LSA** no corrige q5 (d09 sigue en el puesto 6) y degrada q4: d05, que en TF-IDF encabeza con 0.4569, cae al puesto 3, superado por d01 y d10 en el ranking medido. Con solo 10 documentos, los 4 ejes latentes capturan co-ocurrencias locales y mezclan temas; por eso Hit@1 baja de 0.8333 a 0.6667 y MRR de 0.8611 a 0.7500, sin ganancia en Hit@3 (0.8333 en ambos). La semántica latente solo agrupa términos que co-ocurren en el corpus: no puede inyectar conocimiento externo.

**Parte 4 (conceptual):** un recuperador denso resolvería q5 mejor porque sus embeddings, entrenados en grandes corpus, acercan textos semánticamente afines aunque no compartan términos: «más determinista al elegir la siguiente palabra» y «la temperatura… concentra la probabilidad» describen el mismo fenómeno (control de la aleatoriedad de la salida), asociación que ni TF-IDF (similitud medida 0.0790) ni LSA, entrenado solo con las co-ocurrencias de estos 10 documentos, pueden establecer. El costo frente a TF-IDF: exige un modelo preentrenado (descarga y memoria), cómputo de inferencia para embeber consultas y documentos (latencia, posible GPU), mantenimiento de versiones y pérdida de interpretabilidad —en TF-IDF puede auditarse qué términos coinciden, como se hizo con q5, mientras que un vector denso es opaco—. En un corpus de 10 documentos, además, ese costo no se amortiza.
