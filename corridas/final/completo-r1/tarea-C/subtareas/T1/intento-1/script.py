# ============================================================================
# T1 — Línea base léxica: TF-IDF (parámetros por defecto) + similitud coseno
# Corpus d01-d10, consultas q1-q6 con un documento relevante cada una.
# Salidas: resultados.json, t1_rankings.csv, t1_mapa_calor_similitud.png,
#          t1_mrr_por_consulta.png
# ============================================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1. Corpus (d01-d10) — texto literal del enunciado
# ----------------------------------------------------------------------------
CORPUS = {
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
# 2. Consultas (q1-q6) y juicios de relevancia (1 documento relevante c/u)
# ----------------------------------------------------------------------------
CONSULTAS = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(CORPUS.keys())
doc_texts = [CORPUS[d] for d in doc_ids]
q_ids = list(CONSULTAS.keys())
q_texts = [CONSULTAS[q][0] for q in q_ids]
relevantes = {q: CONSULTAS[q][1] for q in q_ids}

# ----------------------------------------------------------------------------
# 3. Vectorización TF-IDF con TfidfVectorizer por defecto.
#    Se ajusta (fit) SOLO con el corpus; las consultas únicamente se
#    transforman con el vocabulario aprendido (práctica estándar de IR).
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)   # (10, V)
X_queries = vectorizer.transform(q_texts)      # (6, V)
S = cosine_similarity(X_queries, X_docs)       # (6, 10)

# ----------------------------------------------------------------------------
# 4. Ranking de los 10 documentos para cada consulta (descendente por coseno;
#    empates resueltos de forma determinista por orden original).
# ----------------------------------------------------------------------------
rankings = {}
for i, q in enumerate(q_ids):
    orden = np.argsort(-S[i], kind="stable")
    rankings[q] = [
        {"pos": int(pos + 1), "doc_id": doc_ids[j], "score": float(S[i, j])}
        for pos, j in enumerate(orden)
    ]

# ----------------------------------------------------------------------------
# 5. Métricas: Hit@1, Hit@3 y MRR (por consulta y agregadas sobre las 6)
# ----------------------------------------------------------------------------
por_consulta = {}
hit1_list, hit3_list, rr_list = [], [], []
for q in q_ids:
    rank_rel = next(r["pos"] for r in rankings[q] if r["doc_id"] == relevantes[q])
    rr = 1.0 / rank_rel
    h1, h3 = int(rank_rel == 1), int(rank_rel <= 3)
    hit1_list.append(h1)
    hit3_list.append(h3)
    rr_list.append(rr)
    por_consulta[q] = {
        "consulta": CONSULTAS[q][0],
        "relevante": relevantes[q],
        "rank_relevante": int(rank_rel),
        "reciprocal_rank": float(rr),
        "hit_at_1": int(h1),
        "hit_at_3": int(h3),
    }

metricas = {
    "hit_at_1": float(np.mean(hit1_list)),
    "hit_at_3": float(np.mean(hit3_list)),
    "mrr": float(np.mean(rr_list)),
    "n_consultas": len(q_ids),
}

# ----------------------------------------------------------------------------
# 6. Tabla de rankings en CSV (consulta, posición, documento, score)
# ----------------------------------------------------------------------------
filas = [
    {"consulta": q, "pos": r["pos"], "doc_id": r["doc_id"],
     "score": r["score"], "es_relevante": int(r["doc_id"] == relevantes[q])}
    for q in q_ids for r in rankings[q]
]
pd.DataFrame(filas).to_csv("t1_rankings.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------------
# 7. Figura 1: mapa de calor de similitudes (relevante enmarcado en rojo)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.6))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0)
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids, fontsize=9)
ax.set_yticks(range(len(q_ids)))
ax.set_yticklabels(q_ids, fontsize=9)
for i in range(len(q_ids)):
    for j in range(len(doc_ids)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if S[i, j] < 0.6 * S.max() else "black")
for i, q in enumerate(q_ids):
    j = doc_ids.index(relevantes[q])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="red", linewidth=2))
ax.set_xlabel("Documentos")
ax.set_ylabel("Consultas")
ax.set_title("T1 · Similitud coseno consulta-documento (TF-IDF por defecto)")
fig.colorbar(im, ax=ax, label="similitud coseno")
fig.tight_layout()
fig.savefig("t1_mapa_calor_similitud.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 8. Figura 2: rango recíproco por consulta + métricas agregadas
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
vals = [por_consulta[q]["reciprocal_rank"] for q in q_ids]
barras = ax.bar(q_ids, vals, color="#4C72B0", edgecolor="black", linewidth=0.5)
for q, b in zip(q_ids, barras):
    ax.text(b.get_x() + b.get_width() / 2, por_consulta[q]["reciprocal_rank"] + 0.03,
            f"rank {por_consulta[q]['rank_relevante']}", ha="center", fontsize=8)
ax.axhline(metricas["mrr"], color="#C44E52", linestyle="--", linewidth=1.2,
           label=f"MRR = {metricas['mrr']:.3f}")
ax.set_ylim(0, 1.18)
ax.set_ylabel("1 / rank del documento relevante")
ax.set_title("T1 · Línea base TF-IDF — "
             f"Hit@1 = {metricas['hit_at_1']:.3f} · "
             f"Hit@3 = {metricas['hit_at_3']:.3f} · "
             f"MRR = {metricas['mrr']:.3f}")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("t1_mrr_por_consulta.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 9. Contrato: resultados.json con todas las cifras de la subtarea
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "titulo": "Línea base léxica: TF-IDF (TfidfVectorizer por defecto) + similitud coseno",
    "config": {
        "vectorizador": "sklearn.feature_extraction.text.TfidfVectorizer (parámetros por defecto)",
        "nota_defaults": "lowercase=True, token_pattern='(?u)\\b\\w\\w\\b+' (descarta tokens de 1 carácter), sin stopwords, sin stemming, norm='l2', smooth_idf=True",
        "ajuste": "fit sobre los 10 documentos del corpus; las consultas solo se transforman",
        "similitud": "coseno (sklearn.metrics.pairwise.cosine_similarity)",
        "n_documentos": len(doc_ids),
        "n_consultas": len(q_ids),
        "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    },
    "juicios_relevancia": relevantes,
    "metricas_agregadas": metricas,
    "metricas_por_consulta": por_consulta,
    "rankings": rankings,
    "matriz_similitud": {
        "consultas": q_ids,
        "documentos": doc_ids,
        "valores": [[float(v) for v in fila] for fila in S],
    },
    "figuras": ["t1_mapa_calor_similitud.png", "t1_mrr_por_consulta.png"],
    "tablas": ["t1_rankings.csv"],
}
Path("resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ----------------------------------------------------------------------------
# 10. Resumen en consola
# ----------------------------------------------------------------------------
print("=" * 78)
print("T1 · Línea base TF-IDF (parámetros por defecto) + similitud coseno")
print(f"Vocabulario aprendido del corpus: {len(vectorizer.vocabulary_)} términos")
print("-" * 78)
for q in q_ids:
    pc = por_consulta[q]
    top3 = " | ".join(f"{r['doc_id']} ({r['score']:.3f})" for r in rankings[q][:3])
    print(f"{q} [rel={pc['relevante']}] rank_relevante={pc['rank_relevante']} "
          f"RR={pc['reciprocal_rank']:.3f}  top-3: {top3}")
print("-" * 78)
print(f"Hit@1 = {metricas['hit_at_1']:.4f}   "
      f"Hit@3 = {metricas['hit_at_3']:.4f}   "
      f"MRR  = {metricas['mrr']:.4f}")
print("Archivos generados: resultados.json, t1_rankings.csv, "
      "t1_mapa_calor_similitud.png, t1_mrr_por_consulta.png")
print("=" * 78)
