#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica: TF-IDF + similitud coseno (Tarea C, MMIA 6013).

Construye el corpus de 10 documentos y las 6 consultas con sus juicios de
relevancia (texto EXACTO del enunciado), vectoriza con TfidfVectorizer de
scikit-learn con parámetros por defecto (ajustado SOLO con el corpus; las
consultas únicamente se transforman), calcula la similitud coseno
consulta-documento, ordena los 10 documentos por consulta y evalúa
Hit@1, Hit@3 y MRR sobre las 6 consultas.

Salidas (rutas relativas a la carpeta actual):
  - entrada/T1/resultados.json  : contrato con rankings completos y métricas
  - resultados.json             : copia del contrato en la carpeta actual
  - t1_similitud_coseno.png     : mapa de calor de similitudes q-d
  - t1_metricas_baseline.png    : Hit@1, Hit@3, MRR
  - t1_rankings.csv             : rankings completos por consulta
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1) Corpus: 10 documentos (texto exacto del enunciado, Parte 1, página 1)
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

# ----------------------------------------------------------------------------
# 2) Consultas y juicios de relevancia (1 documento relevante por consulta)
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 3) Vectorización TF-IDF (parámetros por defecto) y similitud coseno
#    El vectorizador se ajusta SOLO con el corpus (práctica IR estándar);
#    las consultas solo se transforman. No se ajusta nada con las consultas.
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, tal como pide el enunciado
X_docs = vectorizer.fit_transform(doc_texts)

q_texts = [q for q, _ in QUERIES.values()]
X_queries = vectorizer.transform(q_texts)
sim = cosine_similarity(X_queries, X_docs)  # forma (6, 10)

# ----------------------------------------------------------------------------
# 4) Ranking por consulta y métricas Hit@1, Hit@3, MRR
# ----------------------------------------------------------------------------
per_query = {}
hit1_list, hit3_list, rr_list = [], [], []

for i, (qid, (qtext, rel)) in enumerate(QUERIES.items()):
    sims = sim[i]
    order = np.argsort(-sims, kind="stable")  # orden descendente, desempate estable
    ranking = [doc_ids[j] for j in order]
    rank = ranking.index(rel) + 1  # posición 1-based del documento relevante
    h1 = 1.0 if rank == 1 else 0.0
    h3 = 1.0 if rank <= 3 else 0.0
    rr = 1.0 / rank
    hit1_list.append(h1)
    hit3_list.append(h3)
    rr_list.append(rr)
    per_query[qid] = {
        "consulta": qtext,
        "documento_relevante": rel,
        "ranking_completo": ranking,
        "similitudes_ordenadas": [float(sims[j]) for j in order],
        "rank_del_relevante": int(rank),
        "hit@1": h1,
        "hit@3": h3,
        "reciprocal_rank": rr,
    }

hit_at_1 = float(np.mean(hit1_list))
hit_at_3 = float(np.mean(hit3_list))
mrr = float(np.mean(rr_list))

# ----------------------------------------------------------------------------
# 5) Contrato: resultados.json (rankings completos + métricas agregadas)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "titulo": "Línea base léxica: TF-IDF (parámetros por defecto) + similitud coseno",
    "config": {
        "vectorizador": "TfidfVectorizer (scikit-learn, parámetros por defecto)",
        "ajustado_con": "solo los 10 documentos del corpus",
        "similitud": "coseno",
        "n_documentos": len(doc_ids),
        "n_consultas": len(QUERIES),
        "n_terminos_vocabulario": len(vectorizer.vocabulary_),
    },
    "hit@1": hit_at_1,
    "hit@3": hit_at_3,
    "mrr": mrr,
    "ranks_del_documento_relevante": {q: per_query[q]["rank_del_relevante"] for q in QUERIES},
    "rankings": {q: per_query[q]["ranking_completo"] for q in QUERIES},
    "similitudes_por_consulta": {
        q: {doc_ids[j]: float(sim[i, j]) for j in range(len(doc_ids))}
        for i, q in enumerate(QUERIES)
    },
    "detalle_por_consulta": per_query,
}

out_dir = Path("entrada/T1")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 6) CSV de rankings (insumo para el reporte)
# ----------------------------------------------------------------------------
rows = []
for qid, info in per_query.items():
    for pos, (doc_id, s) in enumerate(
        zip(info["ranking_completo"], info["similitudes_ordenadas"]), start=1
    ):
        rows.append(
            {
                "consulta": qid,
                "rank": pos,
                "documento": doc_id,
                "similitud_coseno": s,
                "es_relevante": doc_id == info["documento_relevante"],
            }
        )
pd.DataFrame(rows).to_csv("t1_rankings.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------------
# 7) Figuras
# ----------------------------------------------------------------------------
# 7a) Mapa de calor de similitudes consulta-documento
fig, ax = plt.subplots(figsize=(10.5, 4.8))
im = ax.imshow(sim, cmap="viridis", aspect="auto", vmin=0.0)
ax.set_xticks(np.arange(len(doc_ids)))
ax.set_xticklabels(doc_ids)
ax.set_yticks(np.arange(len(QUERIES)))
ax.set_yticklabels(list(QUERIES.keys()))
for i in range(sim.shape[0]):
    for j in range(sim.shape[1]):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
for i, (qid, (_, rel)) in enumerate(QUERIES.items()):
    j = doc_ids.index(rel)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
fig.colorbar(im, ax=ax, label="Similitud coseno")
ax.set_xlabel("Documentos")
ax.set_ylabel("Consultas")
ax.set_title("T1 · TF-IDF (defecto): similitud coseno consulta–documento\n(rectángulo rojo = documento relevante)")
plt.tight_layout()
plt.savefig("t1_similitud_coseno.png", dpi=120)
plt.close()

# 7b) Métricas agregadas
fig, ax = plt.subplots(figsize=(6, 4))
nombres = ["Hit@1", "Hit@3", "MRR"]
valores = [hit_at_1, hit_at_3, mrr]
barras = ax.bar(nombres, valores, color=["#4C72B0", "#55A868", "#C44E52"], width=0.55)
ax.axhline(1.0, color="gray", ls="--", lw=0.8)
ax.set_ylim(0, 1.1)
for b, v in zip(barras, valores):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=10)
ax.set_ylabel("Valor (promedio sobre 6 consultas)")
ax.set_title("T1 · Línea base TF-IDF + coseno")
plt.tight_layout()
plt.savefig("t1_metricas_baseline.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# 8) Resumen
# ----------------------------------------------------------------------------
print("=" * 70)
print("T1 · Línea base léxica: TF-IDF (parámetros por defecto) + coseno")
print("=" * 70)
for qid, info in per_query.items():
    top3 = ", ".join(
        f"{d}({s:.3f})"
        for d, s in list(zip(info["ranking_completo"], info["similitudes_ordenadas"]))[:3]
    )
    print(
        f"{qid} [rel={info['documento_relevante']}] "
        f"rank_relevante={info['rank_del_relevante']:>2} | top-3: {top3}"
    )
print("-" * 70)
n_ok1 = int(round(hit_at_1 * len(QUERIES)))
n_ok3 = int(round(hit_at_3 * len(QUERIES)))
print(f"Hit@1 = {hit_at_1:.6f}  ({n_ok1}/{len(QUERIES)} consultas)")
print(f"Hit@3 = {hit_at_3:.6f}  ({n_ok3}/{len(QUERIES)} consultas)")
print(f"MRR   = {mrr:.6f}")
print("-" * 70)
print("Archivos generados: entrada/T1/resultados.json, resultados.json,")
print("t1_similitud_coseno.png, t1_metricas_baseline.png, t1_rankings.csv")
