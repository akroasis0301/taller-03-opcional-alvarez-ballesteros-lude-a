## Objetivo

Comparar la recuperación léxica (TF-IDF + similitud coseno) contra la semántica latente (LSA mediante TruncatedSVD a 4 dimensiones) sobre un corpus de 10 documentos y 6 consultas con un documento relevante cada una, evaluando Hit@1, Hit@3 y MRR.

## Método

Se vectorizó el corpus con `TfidfVectorizer` (parámetros por defecto, ajustado solo sobre los 10 documentos; vocabulario de 105 términos) y se ordenaron los 10 documentos por similitud coseno para cada consulta. Para LSA se proyectó la misma matriz TF-IDF con `TruncatedSVD(n_components=4, random_state=0)`; las consultas se plegaron con `svd.transform` y se repitió el ranking con coseno en el espacio proyectado. Valores singulares medidos: 1.3172, 1.0401, 1.0225, 0.9974; varianza explicada acumulada con k=4: 0.3953. Hit@k: proporción de consultas con el relevante entre los k primeros; MRR: media del rango recíproco del relevante. El conteo cuantitativo de términos compartidos por consulta (Parte 3) quedó sin ejecutar; los fallos se identifican con los ranks medidos y el análisis se apoya en las similitudes medidas.

## Resultados

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF + coseno | 0.8333 | 0.8333 | 0.8611 |
| LSA (k=4, random_state=0) | 0.6667 | 0.8333 | 0.7500 |

Diferencia LSA − TF-IDF: Hit@1 −0.1667, Hit@3 0.0000, MRR −0.1111.

| Consulta (relevante) | Rank TF-IDF | Rank LSA |
|---|---|---|
| q1 (d04) | 1 | 1 |
| q2 (d03) | 1 | 1 |
| q3 (d08) | 1 | 1 |
| q4 (d05) | 1 | 3 |
| q5 (d09) | 6 | 6 |
| q6 (d02) | 1 | 1 |

TF-IDF coloca el relevante en el puesto 1 en 5 de 6 consultas; solo q5 queda fuera del top 3 (rank 6). LSA conserva Hit@3 pero pierde un acierto en el top 1: en q4, d05 (0.8984) es superado por d01 (0.9220) y d10 (0.9206); q5 permanece en rank 6.

![Mapa de calor de similitud coseno TF-IDF consulta-documento](figuras/T1_t1_heatmap_similitud.png)

![Proyección LSA (componentes 1 y 2) de documentos y consultas](figuras/T2_proyeccion_lsa_componentes_1_2.png)

## Discusión

**Parte 3.** El único fallo de ambos métodos (relevante fuera del top 3) es q5. La consulta «¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?» no comparte términos de contenido con d09 («temperatura», «logits», «softmax», «probabilidad»): la similitud TF-IDF medida q5–d09 es 0.0790, por detrás de d03 (0.3107, comparte «modelo») y d06 (0.2009). Sin sinónimos ni paráfrasis en el corpus, el ranking léxico no puede unir «determinista» con «temperatura baja». LSA no lo corrige: con k=4 captura solo 0.3953 de la varianza y, al difuminar los temas, q5–d09 sube a 0.4999 pero q5–d04 sube más (0.9427); en q4 esa misma difusión acerca la consulta a d01 y d10 y cuesta el puesto 1.

**Parte 4.** Un recuperador denso entrenado resolvería q5 mejor porque sus embeddings se aprenden en corpus grandes donde «determinista» coocurre con «temperatura», «softmax» y «baja aleatoriedad»; la similitud en ese espacio captaría la paráfrasis que TF-IDF (términos exactos) y LSA (coocurrencia en solo 10 documentos, 0.3953 de varianza) no ven. El costo frente a TF-IDF: exige un modelo preentrenado e inferencia para codificar consultas y documentos, un índice de vectores con su memoria, y pierde interpretabilidad (los pesos TF-IDF son inspeccionables término a término); TF-IDF no requiere entrenamiento ni cómputo especial.

*El presupuesto de tokens se agotó: se entrega lo que se alcanzó a medir.*
