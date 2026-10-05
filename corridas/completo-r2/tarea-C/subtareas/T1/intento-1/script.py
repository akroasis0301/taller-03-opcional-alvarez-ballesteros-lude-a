# =====================================================================
# T1 · Línea base léxica: TF-IDF (parámetros por defecto) + coseno
# Corpus de 10 documentos y 6 consultas con un relevante cada una
# (datos tomados íntegramente del enunciado; sin descargas ni red).
# Salidas: resultados.json (contrato) y t1_heatmap_similitud.png
# =====================================================================
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------- corpus
doc_ids = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]
documentos = [
    "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
]

# ------------------------------------- consultas y juicios de relevancia
consultas = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
query_ids = sorted(consultas.keys())  # q1..q6

# ------------------------------------------ vectorización TF-IDF
# TfidfVectorizer con TODOS los parámetros por defecto, tal como pide
# la Parte 1. Las consultas se transforman con el vocabulario aprendido
# SOLO de los documentos (no se ajusta nada con datos de prueba).
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(documentos)          # (10, V)
X_queries = vectorizer.transform([consultas[q][0] for q in query_ids])  # (6, V)

# ------------------------------------------ similitud coseno q-d
S = cosine_similarity(X_queries, X_docs)  # forma (6, 10)

# ------------------------------------------ rankings y evaluación
rankings = {}
por_consulta = {}
hits1, hits3, rrs = [], [], []

for i, qid in enumerate(query_ids):
    orden = np.argsort(-S[i], kind="stable")  # descendente; desempates por id
    ranking = [doc_ids[j] for j in orden]
    scores = [float(S[i, j]) for j in orden]
    relevante = consultas[qid][1]
    rank = ranking.index(relevante) + 1
    h1 = 1.0 if rank <= 1 else 0.0
    h3 = 1.0 if rank <= 3 else 0.0
    rr = 1.0 / rank
    hits1.append(h1)
    hits3.append(h3)
    rrs.append(rr)
    rankings[qid] = {
        "consulta": consultas[qid][0],
        "relevante": relevante,
        "ranking_completo_10_docs": ranking,
        "similitudes_en_orden": scores,
        "rank_del_relevante": int(rank),
    }
    por_consulta[qid] = {
        "relevante": relevante,
        "rank": int(rank),
        "hit@1": h1,
        "hit@3": h3,
        "reciprocal_rank": rr,
    }

hit_at_1 = float(np.mean(hits1))
hit_at_3 = float(np.mean(hits3))
mrr = float(np.mean(rrs))

resultados = {
    "subtarea": "T1_linea_base_lexica_tfidf_coseno",
    "config": {
        "vectorizador": "TfidfVectorizer (parametros por defecto)",
        "similitud": "coseno",
        "n_documentos": len(doc_ids),
        "n_consultas": len(query_ids),
        "vocabulario_tamano": int(len(vectorizer.vocabulary_)),
    },
    "metricas_agregadas": {
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "mrr": mrr,
    },
    "metricas_por_consulta": por_consulta,
    "rankings_por_consulta": rankings,
    "matriz_similitud_coseno": {
        "filas_consultas": query_ids,
        "columnas_documentos": doc_ids,
        "valores": [[float(v) for v in fila] for fila in S],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------- figura
fig, ax = plt.subplots(figsize=(10, 4.8))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0)
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids, rotation=45)
ax.set_yticks(range(len(query_ids)))
ax.set_yticklabels(query_ids)
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                color="white", fontsize=7)
fig.colorbar(im, ax=ax, label="Similitud coseno")
ax.set_title("T1 · TF-IDF (defecto) + coseno: similitud consulta–documento")
ax.set_xlabel("Documentos")
ax.set_ylabel("Consultas")
plt.tight_layout()
plt.savefig("t1_heatmap_similitud.png", dpi=120)
plt.close(fig)

# --------------------------------------------------------- resumen
print("=== T1 · Línea base léxica (TF-IDF + coseno) ===")
print(f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos")
for qid in query_ids:
    r = rankings[qid]
    print(f"{qid} | relevante={r['relevante']} | rank={r['rank_del_relevante']:>2} | "
          f"top3={r['ranking_completo_10_docs'][:3]}")
print(f"Hit@1 = {hit_at_1:.4f}")
print(f"Hit@3 = {hit_at_3:.4f}")
print(f"MRR   = {mrr:.4f}")
print("Salidas: resultados.json, t1_heatmap_similitud.png")
