#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica: TF-IDF (parámetros por defecto) + similitud coseno
sobre el corpus mínimo de 10 documentos y 6 consultas del enunciado.

Salidas:
  - resultados.json                (contrato: rankings por consulta + métricas)
  - matriz_similitud_tfidf.png     (mapa de calor consultas × documentos)
  - metricas_tfidf_baseline.png    (barras Hit@1, Hit@3, MRR)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1) Corpus y consultas (textuales del enunciado, sin modificaciones)
# ----------------------------------------------------------------------
DOCS = {
    "d01": "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "d02": "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "d03": "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "d04": "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "d05": "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "d06": "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "d07": "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "d08": "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "d09": "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "d10": "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
}

QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(DOCS.keys())
doc_texts = [DOCS[d] for d in doc_ids]
query_ids = list(QUERIES.keys())
query_texts = [QUERIES[q][0] for q in query_ids]
relevant = {q: QUERIES[q][1] for q in query_ids}

# ----------------------------------------------------------------------
# 2) TF-IDF con parámetros por defecto.
#    El vocabulario se ajusta SOLO con los 10 documentos (índice de
#    recuperación); las consultas se proyectan con el mismo vectorizador.
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # defaults: lowercase=True, token_pattern=(?u)\b\w\w+\b, etc.
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)

# ----------------------------------------------------------------------
# 3) Similitud coseno entre cada consulta y los 10 documentos
# ----------------------------------------------------------------------
S = cosine_similarity(X_queries, X_docs)  # forma (6, 10)

# ----------------------------------------------------------------------
# 4) Ranking por consulta (descendente; empates resueltos por orden d01..d10)
# ----------------------------------------------------------------------
rankings = {}
per_query = []
for i, qid in enumerate(query_ids):
    order = np.argsort(-S[i], kind="stable")
    ranked_ids = [doc_ids[j] for j in order]
    ranked_scores = [float(S[i, j]) for j in order]
    rankings[qid] = {"ranking": ranked_ids, "scores": ranked_scores}

    rel = relevant[qid]
    rank_pos = ranked_ids.index(rel) + 1  # 1-based
    per_query.append({
        "query_id": qid,
        "consulta": query_texts[i],
        "relevante": rel,
        "rank_del_relevante": int(rank_pos),
        "hit@1": int(rank_pos == 1),
        "hit@3": int(rank_pos <= 3),
        "reciprocal_rank": float(1.0 / rank_pos),
        "similitud_del_relevante": float(S[i, doc_ids.index(rel)]),
    })

df_q = pd.DataFrame(per_query)
hit1 = float(df_q["hit@1"].mean())
hit3 = float(df_q["hit@3"].mean())
mrr = float(df_q["reciprocal_rank"].mean())

# ----------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "metodo": "TfidfVectorizer (parametros por defecto) + similitud coseno; "
              "vocabulario ajustado solo con los 10 documentos, consultas transformadas",
    "n_documentos": len(doc_ids),
    "n_consultas": len(query_ids),
    "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    "vocabulario": sorted(vectorizer.vocabulary_.keys()),
    "matriz_similitud": {
        "filas": query_ids,
        "columnas": doc_ids,
        "valores": [[float(v) for v in fila] for fila in S],
    },
    "rankings_por_consulta": {
        qid: {
            "consulta": QUERIES[qid][0],
            "relevante": relevant[qid],
            "ranking": rankings[qid]["ranking"],
            "similitudes": rankings[qid]["scores"],
        }
        for qid in query_ids
    },
    "metricas_por_consulta": per_query,
    "tabla_metricas": [
        {"metodo": "TF-IDF (linea base lexica)", "Hit@1": hit1, "Hit@3": hit3, "MRR": mrr}
    ],
    "metricas_globales": {"Hit@1": hit1, "Hit@3": hit3, "MRR": mrr},
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6) Figura: mapa de calor de la matriz de similitud
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5))
im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(len(doc_ids)))
ax.set_xticklabels(doc_ids, rotation=45)
ax.set_yticks(np.arange(len(query_ids)))
ax.set_yticklabels(query_ids)
for i in range(len(query_ids)):
    for j in range(len(doc_ids)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if S[i, j] < 0.6 else "black")
ax.set_title("Similitud coseno TF-IDF: consultas × documentos (línea base léxica)")
fig.colorbar(im, ax=ax, label="similitud coseno")
fig.tight_layout()
plt.savefig("matriz_similitud_tfidf.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 7) Figura: métricas globales
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
vals = [hit1, hit3, mrr]
ax.bar(["Hit@1", "Hit@3", "MRR"], vals, color=["#4c72b0", "#55a868", "#c44e52"])
ax.set_ylim(0, 1.05)
for k, v in enumerate(vals):
    ax.text(k, v + 0.02, f"{v:.4f}", ha="center")
ax.set_ylabel("valor")
ax.set_title("Línea base léxica TF-IDF (6 consultas)")
fig.tight_layout()
plt.savefig("metricas_tfidf_baseline.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 8) Resumen en consola
# ----------------------------------------------------------------------
print("=== T1: Línea base léxica TF-IDF + coseno ===")
print(f"Vocabulario: {len(vectorizer.vocabulary_)} términos | Docs: {len(doc_ids)} | Consultas: {len(query_ids)}")
print("\nTabla por consulta:")
print(df_q[["query_id", "relevante", "rank_del_relevante", "hit@1", "hit@3", "reciprocal_rank"]].to_string(index=False))
print(f"\nGlobales: Hit@1 = {hit1:.4f} | Hit@3 = {hit3:.4f} | MRR = {mrr:.4f}")
print("\nRankings (mejor → peor):")
for qid in query_ids:
    print(f"  {qid} (rel={relevant[qid]}): {' > '.join(rankings[qid]['ranking'])}")
print("\nArchivos escritos: resultados.json, matriz_similitud_tfidf.png, metricas_tfidf_baseline.png")
