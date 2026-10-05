#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica (Tarea C, MMIA 6013).

Corpus de 10 documentos y 6 consultas con juicio de relevancia (textuales del
enunciado). Vectorización con TfidfVectorizer de scikit-learn (parámetros por
defecto), ajustado SOLO con los documentos del corpus; las consultas se
transforman con ese vocabulario (sin fuga de datos). Similitud coseno
consulta-documento, ranking completo de los 10 documentos por consulta y
evaluación de Hit@1, Hit@3 y MRR sobre las 6 consultas.

Salidas (carpeta actual):
  - resultados.json                    (contrato: rankings, similitudes, métricas)
  - figura_t1_similitud_coseno.png     (mapa de calor 6x10)
  - figura_t1_metricas.png             (Hit@1, Hit@3, MRR agregados)
  - figura_t1_rr_por_consulta.png      (reciprocal rank por consulta)
"""

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1. Corpus y consultas (exactamente como en el enunciado)
# ----------------------------------------------------------------------------
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

doc_ids = list(DOCS)
doc_texts = [DOCS[d] for d in doc_ids]
query_ids = list(QUERIES)
query_texts = [QUERIES[q][0] for q in query_ids]
relevantes = {q: QUERIES[q][1] for q in query_ids}

# ----------------------------------------------------------------------------
# 2. Vectorización TF-IDF (parámetros por defecto) y similitud coseno
# ----------------------------------------------------------------------------
# El vectorizador se ajusta únicamente con el corpus de documentos; las
# consultas solo se transforman con el vocabulario aprendido.
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(doc_texts)      # (10, V)
X_queries = vectorizer.transform(query_texts)     # (6, V)

S = cosine_similarity(X_queries, X_docs)          # (6, 10)

# ----------------------------------------------------------------------------
# 3. Ranking completo (10 documentos) por consulta y métricas por consulta
# ----------------------------------------------------------------------------
filas = []
for i, qid in enumerate(query_ids):
    orden = np.argsort(-S[i], kind="stable")      # descendente; desempate = orden del corpus
    ranking = [doc_ids[j] for j in orden]
    rel = relevantes[qid]
    rank = ranking.index(rel) + 1
    filas.append(
        {
            "consulta": qid,
            "texto_consulta": query_texts[i],
            "relevante": rel,
            "ranking_completo": ranking,
            "rank_del_relevante": int(rank),
            "reciprocal_rank": float(1.0 / rank),
            "hit@1": int(rank == 1),
            "hit@3": int(rank <= 3),
            "sim_coseno_con_relevante": float(S[i, doc_ids.index(rel)]),
            "sim_coseno_top1": float(S[i, orden[0]]),
        }
    )

df = pd.DataFrame(filas)

# Métricas agregadas sobre las 6 consultas
hit1 = float(df["hit@1"].mean())
hit3 = float(df["hit@3"].mean())
mrr = float(df["reciprocal_rank"].mean())

# ----------------------------------------------------------------------------
# 4. resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1_linea_base_lexica_tfidf",
    "descripcion": "TF-IDF (TfidfVectorizer por defecto) + similitud coseno; ranking de 10 documentos por consulta; Hit@1, Hit@3 y MRR sobre 6 consultas.",
    "n_documentos": len(doc_ids),
    "n_consultas": len(query_ids),
    "vectorizador": "TfidfVectorizer (parámetros por defecto), ajustado solo con los 10 documentos",
    "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    "vocabulario": sorted(vectorizer.vocabulary_.keys()),
    "desempate_ranking": "argsort estable descendente (empates resueltos por orden del corpus d01..d10)",
    "matriz_similitud_coseno": {
        "filas_consultas": query_ids,
        "columnas_documentos": doc_ids,
        "valores": [[float(v) for v in fila] for fila in S],
    },
    "por_consulta": filas,
    "metricas_agregadas": {
        "hit@1": hit1,
        "hit@3": hit3,
        "mrr": mrr,
    },
    "tabla_metricas": [
        {"metrica": "Hit@1", "valor": hit1},
        {"metrica": "Hit@3", "valor": hit3},
        {"metrica": "MRR", "valor": mrr},
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 5. Figuras
# ----------------------------------------------------------------------------
# 5.1 Mapa de calor de similitudes coseno (6 x 10)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
im = ax.imshow(S, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids, rotation=0)
ax.set_yticks(range(len(query_ids)))
ax.set_yticklabels(query_ids)
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                color="white", fontsize=7)
fig.colorbar(im, ax=ax, label="similitud coseno")
ax.set_xlabel("Documentos")
ax.set_ylabel("Consultas")
ax.set_title("T1 · Línea base TF-IDF: similitud coseno consulta-documento")
fig.tight_layout()
fig.savefig("figura_t1_similitud_coseno.png", dpi=120)
plt.close(fig)

# 5.2 Métricas agregadas
fig, ax = plt.subplots(figsize=(6, 4))
nombres = ["Hit@1", "Hit@3", "MRR"]
valores = [hit1, hit3, mrr]
barras = ax.bar(nombres, valores, color=["#4c72b0", "#55a868", "#c44e52"])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Valor (6 consultas)")
ax.set_title("T1 · Línea base léxica TF-IDF: métricas agregadas")
for b, v in zip(barras, valores):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
fig.tight_layout()
fig.savefig("figura_t1_metricas.png", dpi=120)
plt.close(fig)

# 5.3 Reciprocal rank por consulta
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(df["consulta"], df["reciprocal_rank"], color="#8172b2")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Reciprocal rank (1/rank del relevante)")
ax.set_xlabel("Consulta")
ax.set_title("T1 · Reciprocal rank por consulta (línea base TF-IDF)")
for x, (rr, rk) in enumerate(zip(df["reciprocal_rank"], df["rank_del_relevante"])):
    ax.text(x, rr + 0.02, f"rank={rk}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("figura_t1_rr_por_consulta.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. Resumen en consola
# ----------------------------------------------------------------------------
print("=" * 72)
print("T1 · Línea base léxica: TF-IDF (por defecto) + similitud coseno")
print("=" * 72)
print(f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos")
print("\nRanking por consulta (top-3) y posición del documento relevante:")
for _, r in df.iterrows():
    top3 = " > ".join(r["ranking_completo"][:3])
    print(
        f"  {r['consulta']}  relevante={r['relevante']}  rank={r['rank_del_relevante']:>2}  "
        f"RR={r['reciprocal_rank']:.4f}  | top-3: {top3}"
    )
print("\nTabla agregada (6 consultas):")
print(df[["consulta", "relevante", "rank_del_relevante", "reciprocal_rank", "hit@1", "hit@3"]]
      .to_string(index=False))
print("-" * 72)
print(f"Hit@1 = {hit1:.6f}   Hit@3 = {hit3:.6f}   MRR = {mrr:.6f}")
print("Salidas: resultados.json, figura_t1_similitud_coseno.png, "
      "figura_t1_metricas.png, figura_t1_rr_por_consulta.png")
