## Objetivo

Comparar dos estrategias de recuperación sobre un corpus mínimo de diez documentos (d01–d10) y seis consultas (q1–q6), cada una con un único documento relevante (q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02): (1) una línea base léxica con TF-IDF y similitud coseno, y (2) semántica latente (LSA) proyectando la misma matriz TF-IDF a 4 dimensiones. Se evalúan Hit@1, Hit@3 y MRR, se analizan los fallos por consulta y se discute qué aportaría un recuperador denso, en particular para q5.

## Método

- **Parte 1 (TF-IDF):** vectorización de los diez documentos con `TfidfVectorizer` de scikit-learn (parámetros por defecto); para cada consulta se calcula la similitud coseno contra los diez documentos y se ordenan. Métricas sobre las seis consultas: Hit@1 (relevante en la posición 1), Hit@3 (relevante dentro del top 3) y MRR (promedio del recíproco del rango del relevante).
- **Parte 2 (LSA):** sobre la misma matriz TF-IDF, proyección a 4 dimensiones con `TruncatedSVD(n_components=4, random_state=0)`; ranking por coseno en el espacio latente y recálculo de las tres métricas. Ambos métodos se reportan en una sola tabla comparativa.
- **Parte 3:** con los rankings de ambos métodos, identificación de las consultas cuyo relevante queda fuera del top 3 y análisis de los términos compartidos entre cada consulta, su relevante y los documentos no relevantes mejor posicionados.

## Resultados

**Declaración de resultados no disponibles.** La ejecución que debía producir las cifras no se completó:

- La subtarea **T1 (Parte 1)** no tiene resultado (fallida): **no se reportan sus cifras** (Hit@1, Hit@3 y MRR de TF-IDF).
- La subtarea **T2 (Parte 2)** no tiene resultado (omitida): **no se reportan sus cifras** (métricas de LSA ni tabla comparativa).
- La subtarea **T3 (Parte 3)** no tiene resultado (omitida): no se disponen de los rankings, por lo que **no puede listarse en qué consultas falla cada método** ni los términos compartidos calculados.

La tabla comparativa exigida queda por tanto vacía; cualquier cifra en ella sería inventada y se omite:

| Métrica | TF-IDF | LSA (4 dim.) |
|---|---|---|
| Hit@1 | — | — |
| Hit@3 | — | — |
| MRR | — | — |

(celdas sin dato: mediciones no disponibles; pendiente re-ejecución del pipeline).

## Discusión

**Análisis por consulta (sin soporte medido).** Al no existir el resultado de T3, no puede afirmarse con evidencia de ejecución en qué consultas falla cada método. Sí puede razonarse cualitativamente a partir del texto del corpus: q5 («¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?») es la única consulta cuyo relevante (d09: «La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad») no comparte ningún término de contenido con la consulta («determinista» vs. «concentran la probabilidad»; «palabra» vs. «logits/softmax»). TF-IDF, que depende del solapamiento léxico exacto, es candidata a fallar ahí; LSA a 4 dimensiones podría mitigarlo vía co-ocurrencia, pero con solo diez documentos el espacio latente es grueso y puede mezclar temas, degradando consultas que sí dependen de términos precisos (q1–q4 y q6 comparten literalmente «función de ranking léxica», «frecuencia de términos», «fragmentos… prompt», «matrices de bajo rango», «orientación… vectores» y «autoatención… feed-forward» con sus relevantes). Estas hipótesis quedan sin confirmar mientras no se re-ejecuten T1–T3.

**Recuperador denso para q5 y sus costos.** Un recuperador denso con embeddings de un modelo entrenado resolvería mejor q5 porque aprende, en grandes corpus, a alinear paráfrasis y relaciones semánticas: «más determinista» quedaría cerca de «valores bajos concentran la probabilidad» aunque no compartan vocabulario, resolviendo justo el desajuste léxico que penaliza a TF-IDF y que LSA, entrenada implícitamente sobre diez documentos, apenas puede captar. Frente a TF-IDF, sus costos son: (i) entrenamiento o uso de un modelo preentrenado, con datos y cómputo asociados; (ii) inferencia neuronal (paso de red) por documento y por consulta, con mayor latencia y posible necesidad de GPU; (iii) almacenamiento e indexación de vectores densos; (iv) mantenimiento: versionado del modelo, re-cálculo de embeddings ante actualizaciones y sensibilidad a cambio de dominio; y (v) menor interpretabilidad que los pesos léxicos. TF-IDF sigue siendo gratuito de entrenar, rápido e interpretable, y suficiente cuando la consulta y el documento comparten términos, como ocurre en q1–q4 y q6.
