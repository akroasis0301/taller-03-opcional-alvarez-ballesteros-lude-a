# =====================================================================
# T1 · Línea base léxica: TF-IDF + similitud coseno
# Corpus de 10 documentos, 6 consultas con juicio de relevancia.
# Vectorizador: TfidfVectorizer de scikit-learn con parámetros por
# defecto (ajustado SOLO con los documentos; las consultas se
# transforman con el mismo vectorizador, práctica estándar de
# recuperación). Ranking por similitud coseno + Hit@1, Hit@3, MRR.
# =====================================================================
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# 1) Corpus (10 documentos) — datos completos del enunciado
# ---------------------------------------------------------------------
corpus = {
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

# 2) Consultas (6) con su documento relevante (juicios de relevancia)
consultas = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(corpus.keys())
doc_texts = [corpus[d] for d in doc_ids]
query_ids = list(consultas.keys())
query_texts = [consultas[q][0] for q in query_ids]
relevante = {q: consultas[q][1] for q in query_ids}

# ---------------------------------------------------------------------
# 3) Vectorización TF-IDF (parámetros por defecto) y similitud coseno
# ---------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # defaults: lowercase, token_pattern (?u)\b\w\w+\b, tf-idf l2
X_docs = vectorizer.fit_transform(doc_texts)      # ajuste SOLO con documentos
X_queries = vectorizer.transform(query_texts)     # consultas solo se transforman
S = cosine_similarity(X_queries, X_docs)          # matriz (6 consultas x 10 documentos)

# ---------------------------------------------------------------------
# 4) Ranking de los 10 documentos por consulta y métricas
# ---------------------------------------------------------------------
por_consulta = {}
hits1, hits3, rr_list, pos_list = [], [], [], []

for i, q in enumerate(query_ids):
    orden = np.argsort(-S[i], kind="stable")          # descendente, determinista
    ranking = [doc_ids[j] for j in orden]
    rel = relevante[q]
    pos = ranking.index(rel) + 1                      # posición 1-indexada
    h1 = int(pos == 1)
    h3 = int(pos <= 3)
    rr = 1.0 / pos
    por_consulta[q] = {
        "consulta": query_texts[i],
        "documento_relevante": rel,
        "ranking": ranking,
        "posicion_del_relevante": int(pos),
        "Hit@1": h1,
        "Hit@3": h3,
        "reciprocal_rank": float(rr),
        "similitud_coseno": {doc_ids[j]: float(S[i, j]) for j in range(len(doc_ids))},
    }
    hits1.append(h1)
    hits3.append(h3)
    rr_list.append(rr)
    pos_list.append(int(pos))

hit1 = float(np.mean(hits1))
hit3 = float(np.mean(hits3))
mrr = float(np.mean(rr_list))

resultados = {
    "subtarea": "T1_linea_base_lexica",
    "metodo": "TfidfVectorizer (parametros por defecto) + similitud coseno; ranking de 10 documentos por consulta",
    "n_documentos": len(doc_ids),
    "n_consultas": len(query_ids),
    "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    "vocabulario": sorted(vectorizer.vocabulary_.keys()),
    "rankings_por_consulta": {q: por_consulta[q]["ranking"] for q in query_ids},
    "posicion_del_relevante_por_consulta": {q: por_consulta[q]["posicion_del_relevante"] for q in query_ids},
    "detalle_por_consulta": por_consulta,
    "metricas_tfidf": {
        "Hit@1": hit1,
        "Hit@3": hit3,
        "MRR": mrr,
    },
    "tabla_agregada": [
        {"metodo": "TF-IDF (lexica)", "Hit@1": hit1, "Hit@3": hit3, "MRR": mrr}
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------
# 5) Figura 1: mapa de calor de similitudes (relevante enmarcado)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0)
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids)
ax.set_yticks(range(len(query_ids)))
ax.set_yticklabels(query_ids)
for i in range(len(query_ids)):
    for j in range(len(doc_ids)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
for i, q in enumerate(query_ids):
    j = doc_ids.index(relevante[q])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
fig.colorbar(im, ax=ax, label="Similitud coseno")
ax.set_title("T1 · TF-IDF: similitud coseno consulta–documento (cuadro rojo = relevante)")
ax.set_xlabel("Documentos")
ax.set_ylabel("Consultas")
plt.tight_layout()
plt.savefig("t1_heatmap_similitud_tfidf.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------
# 6) Figura 2: posición del documento relevante por consulta
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(query_ids, pos_list, color="#4c72b0")
ax.axhline(1, color="green", linestyle="--", linewidth=1, label="Hit@1")
ax.axhline(3, color="orange", linestyle="--", linewidth=1, label="Hit@3")
ax.set_ylim(0, 10.5)
ax.set_ylabel("Posición del documento relevante")
ax.set_xlabel("Consultas")
ax.set_title("T1 · TF-IDF: posición del relevante por consulta")
for i, p in enumerate(pos_list):
    ax.text(i, p + 0.15, str(p), ha="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig("t1_posicion_relevante.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------
# 7) Resumen
# ---------------------------------------------------------------------
print("T1 · Línea base léxica (TF-IDF + similitud coseno)")
print(f"  Vocabulario: {len(vectorizer.vocabulary_)} términos | Documentos: {len(doc_ids)} | Consultas: {len(query_ids)}")
for q in query_ids:
    d = por_consulta[q]
    print(f"  {q} [relevante={d['documento_relevante']}] pos={d['posicion_del_relevante']:>2} | "
          f"ranking: {' > '.join(d['ranking'])}")
print(f"  Agregado -> Hit@1={hit1:.4f}  Hit@3={hit3:.4f}  MRR={mrr:.4f}")
print("  Archivos escritos: resultados.json, t1_heatmap_similitud_tfidf.png, t1_posicion_relevante.png")
