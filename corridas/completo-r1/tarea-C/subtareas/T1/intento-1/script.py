#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
Subtarea T1 — Línea base léxica: TF-IDF + similitud coseno.

Corpus de 10 documentos (d01–d10) y 6 consultas (q1–q6) con sus juicios de
relevancia, tomados íntegramente del enunciado (no se descarga nada).

Procedimiento:
  1. TfidfVectorizer de scikit-learn con TODOS los parámetros por defecto,
     ajustado SOLO con los 10 documentos (las consultas solo se transforman).
  2. Similitud coseno entre cada consulta y los 10 documentos.
  3. Ranking completo (10 documentos) por consulta, orden descendente de
     similitud (empates resueltos por el orden original d01..d10).
  4. Métricas sobre las 6 consultas: Hit@1, Hit@3 y MRR.

Salidas:
  - resultados.json            (contrato: rankings completos, similitudes y métricas)
  - t1_heatmap_similitud.png   (mapa de calor consulta x documento)
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1. Datos del enunciado: corpus, consultas y juicios de relevancia
# ----------------------------------------------------------------------------
DOCS = {
    "d01": ("El mecanismo de atención pondera cada token según su similitud con la "
            "consulta; la atención escalada divide por la raíz de la dimensión."),
    "d02": ("Los transformadores apilan capas de autoatención y redes feed-forward, "
            "con conexiones residuales y normalización."),
    "d03": ("La recuperación aumentada con generación busca fragmentos relevantes y "
            "los añade al prompt del modelo."),
    "d04": ("BM25 es una función de ranking léxica que pondera la frecuencia de "
            "términos y la longitud del documento."),
    "d05": ("Los embeddings densos representan textos como vectores; la similitud "
            "coseno compara su orientación."),
    "d06": ("Un agente con herramientas decide en cada paso qué función llamar y "
            "observa el resultado."),
    "d07": ("ReAct intercala razonamiento y acciones; Reflexion añade una "
            "autocrítica verbal entre intentos."),
    "d08": ("El ajuste fino con LoRA entrena matrices de bajo rango y congela los "
            "pesos originales."),
    "d09": ("La temperatura reescala los logits antes del softmax; valores bajos "
            "concentran la probabilidad."),
    "d10": ("La cuantización reduce la precisión de los pesos a 8 o 4 bits para "
            "ahorrar memoria."),
}

# consulta -> (texto, documento relevante)
QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente "
           "palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(DOCS.keys())
query_ids = list(QUERIES.keys())
doc_texts = [DOCS[d] for d in doc_ids]
query_texts = [QUERIES[q][0] for q in query_ids]
relevant = {q: QUERIES[q][1] for q in query_ids}

# ----------------------------------------------------------------------------
# 2. Vectorización TF-IDF (parámetros por defecto) y similitud coseno
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()                  # sin ajustes: defaults de sklearn
X_docs = vectorizer.fit_transform(doc_texts)    # vocabulario e IDF solo del corpus
X_queries = vectorizer.transform(query_texts)   # las consultas solo se transforman

S = cosine_similarity(X_queries, X_docs)        # forma (6 consultas, 10 documentos)

# ----------------------------------------------------------------------------
# 3. Ranking completo de los 10 documentos por consulta
# ----------------------------------------------------------------------------
rankings = {}
for i, q in enumerate(query_ids):
    order = np.argsort(-S[i], kind="stable")    # estable: empates quedan d01..d10
    rankings[q] = [doc_ids[j] for j in order]

# ----------------------------------------------------------------------------
# 4. Métricas por consulta y agregadas: Hit@1, Hit@3, MRR
# ----------------------------------------------------------------------------
def metricas_ranking(ranking, doc_relevante):
    pos = ranking.index(doc_relevante) + 1      # posición 1-based del relevante
    return {
        "posicion_relevante": pos,
        "hit@1": 1.0 if pos <= 1 else 0.0,
        "hit@3": 1.0 if pos <= 3 else 0.0,
        "reciprocal_rank": 1.0 / pos,
    }

per_query = {}
for q in query_ids:
    m = metricas_ranking(rankings[q], relevant[q])
    per_query[q] = {
        "consulta": QUERIES[q][0],
        "relevante": relevant[q],
        "ranking": rankings[q],
        "ranking_con_similitud": [
            {"doc": d, "similitud": float(S[i, doc_ids.index(d)])}
            for d in rankings[q]
        ],
        **m,
    }

hit_at_1 = float(np.mean([per_query[q]["hit@1"] for q in query_ids]))
hit_at_3 = float(np.mean([per_query[q]["hit@3"] for q in query_ids]))
mrr = float(np.mean([per_query[q]["reciprocal_rank"] for q in query_ids]))

# ----------------------------------------------------------------------------
# 5. Figura: mapa de calor de similitudes consulta x documento
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.5))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0)
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids, rotation=45, ha="right")
ax.set_yticks(range(len(query_ids)))
ax.set_yticklabels(query_ids)
for i in range(len(query_ids)):
    for j in range(len(doc_ids)):
        val = S[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                color="white" if val < 0.6 else "black")
fig.colorbar(im, ax=ax, label="similitud coseno")
ax.set_title("T1 — Línea base TF-IDF: similitud coseno consulta-documento")
fig.tight_layout()
fig.savefig("t1_heatmap_similitud.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. Escritura del contrato: resultados.json
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "descripcion": ("Línea base léxica: TfidfVectorizer (parámetros por defecto, "
                    "ajustado con los 10 documentos) + similitud coseno"),
    "n_documentos": len(doc_ids),
    "n_consultas": len(query_ids),
    "metricas": {
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "mrr": mrr,
    },
    "metricas_por_consulta": per_query,
    "ranking_por_consulta": rankings,
    "matriz_similitud": {
        "consultas": query_ids,
        "documentos": doc_ids,
        "valores": [[float(v) for v in fila] for fila in S],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 7. Resumen breve por consola
# ----------------------------------------------------------------------------
print("=" * 72)
print("T1 — Línea base TF-IDF (coseno) | 10 documentos, 6 consultas")
print("=" * 72)
for q in query_ids:
    m = per_query[q]
    print(f"{q} (relevante={m['relevante']}): "
          f"ranking={' > '.join(m['ranking'])}")
    print(f"    posición del relevante = {m['posicion_relevante']:2d} | "
          f"Hit@1={m['hit@1']:.0f} | Hit@3={m['hit@3']:.0f} | "
          f"RR={m['reciprocal_rank']:.4f}")
print("-" * 72)
print(f"Hit@1 = {hit_at_1:.4f}  ({int(round(hit_at_1 * len(query_ids)))}/{len(query_ids)} consultas)")
print(f"Hit@3 = {hit_at_3:.4f}  ({int(round(hit_at_3 * len(query_ids)))}/{len(query_ids)} consultas)")
print(f"MRR   = {mrr:.4f}")
print("Archivos generados: resultados.json, t1_heatmap_similitud.png")
