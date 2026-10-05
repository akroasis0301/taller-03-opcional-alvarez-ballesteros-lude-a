# -*- coding: utf-8 -*-
"""
T1 · Línea base léxica: TF-IDF (TfidfVectorizer por defecto) + similitud coseno
Corpus mínimo de 10 documentos y 6 consultas (1 documento relevante por consulta),
tal como aparecen en el enunciado de la Tarea C (MMIA 6013).

Protocolo:
  - El vectorizador se ajusta SOLO con los 10 documentos del corpus; las consultas
    únicamente se transforman con ese vocabulario (protocolo IR estándar: nada se
    ajusta con el lado de evaluación).
  - Similitud coseno consulta-documento; se ordenan los 10 documentos por consulta.
  - Métricas agregadas sobre las 6 consultas: Hit@1, Hit@3 y MRR.

Salidas:
  - resultados.json               (contrato: rankings, posiciones del relevante, métricas)
  - tfidf_similitud_coseno.png    (mapa de calor consulta-documento)
  - tfidf_rr_por_consulta.png     (reciprocal rank por consulta)
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Corpus: 10 documentos cortos (tabla del enunciado, página 1)
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
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]

# ----------------------------------------------------------------------
# 2) Consultas y juicios de relevancia (tabla del enunciado, página 1)
#    (id, texto, documento relevante)
# ----------------------------------------------------------------------
QUERIES = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

# ----------------------------------------------------------------------
# 3) Vectorización TF-IDF con parámetros por defecto + similitud coseno
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform([DOCS[d] for d in DOC_IDS])   # (10, V) solo documentos
X_cons = vectorizer.transform([q for _, q, _ in QUERIES])       # (6, V) mismo vocabulario
S = cosine_similarity(X_cons, X_docs)                           # (6, 10)

# ----------------------------------------------------------------------
# 4) Ranking de los 10 documentos por consulta + posición del relevante
# ----------------------------------------------------------------------
por_consulta = []
recip = []
for i, (qid, qtext, rel) in enumerate(QUERIES):
    orden = np.argsort(-S[i], kind="stable")          # descendente; empates -> orden d01..d10
    ranking = [DOC_IDS[j] for j in orden]
    sims_orden = [float(S[i, j]) for j in orden]
    pos = ranking.index(rel) + 1                      # posición 1-based del documento relevante
    rr = 1.0 / pos
    recip.append(rr)
    por_consulta.append({
        "consulta_id": qid,
        "consulta": qtext,
        "relevante": rel,
        "ranking": ranking,
        "similitudes_en_orden_ranking": sims_orden,
        "similitudes": {DOC_IDS[j]: float(S[i, j]) for j in range(len(DOC_IDS))},
        "posicion_del_relevante": pos,
        "reciprocal_rank": rr,
        "hit@1": int(pos == 1),
        "hit@3": int(pos <= 3),
    })

hit1 = float(np.mean([p["hit@1"] for p in por_consulta]))
hit3 = float(np.mean([p["hit@3"] for p in por_consulta]))
mrr = float(np.mean(recip))
n_hit1 = int(sum(p["hit@1"] for p in por_consulta))
n_hit3 = int(sum(p["hit@3"] for p in por_consulta))

# ----------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "metodo": "TF-IDF (TfidfVectorizer, parámetros por defecto) + similitud coseno",
    "protocolo": "vectorizador ajustado solo con los 10 documentos; consultas transformadas",
    "n_documentos": len(DOC_IDS),
    "n_consultas": len(QUERIES),
    "vocabulario_tamano": int(len(vectorizer.vocabulary_)),
    "por_consulta": por_consulta,
    "metricas": {
        "Hit@1": hit1,
        "Hit@3": hit3,
        "MRR": mrr,
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "mrr": mrr,
    },
    "conteos": {
        "hit@1_aciertos": n_hit1,
        "hit@3_aciertos": n_hit3,
        "n_consultas": len(QUERIES),
    },
    "matriz_similitud_coseno": {
        "filas_consultas": [q[0] for q in QUERIES],
        "columnas_documentos": DOC_IDS,
        "valores": [[float(v) for v in fila] for fila in S],
    },
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6) Figuras
# ----------------------------------------------------------------------
# 6a) Mapa de calor de similitudes consulta-documento
fig, ax = plt.subplots(figsize=(9, 4.6))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0, vmax=max(float(S.max()), 1e-12))
ax.set_xticks(range(len(DOC_IDS)))
ax.set_xticklabels(DOC_IDS)
ax.set_yticks(range(len(QUERIES)))
ax.set_yticklabels([q[0] for q in QUERIES])
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if S[i, j] < 0.55 * S.max() else "black")
ax.set_title("T1 · TF-IDF + coseno: similitud consulta-documento")
fig.colorbar(im, ax=ax, label="similitud coseno")
fig.tight_layout()
fig.savefig("tfidf_similitud_coseno.png", dpi=120)
plt.close(fig)

# 6b) Reciprocal rank por consulta
fig, ax = plt.subplots(figsize=(7, 4))
cols = ["#2c7fb8" if p["posicion_del_relevante"] <= 3 else "#d95f0e" for p in por_consulta]
ax.bar([p["consulta_id"] for p in por_consulta], recip, color=cols)
ax.axhline(mrr, ls="--", lw=1, color="k")
ax.text(len(QUERIES) - 0.45, mrr + 0.02, f"MRR = {mrr:.3f}", ha="right", fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Reciprocal rank (1/posición del relevante)")
ax.set_title("T1 · TF-IDF: reciprocal rank por consulta (naranja = relevante fuera del top 3)")
for x, p in enumerate(por_consulta):
    ax.text(x, recip[x] + 0.02, f"{recip[x]:.2f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("tfidf_rr_por_consulta.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 7) Resumen en pantalla (criterio de éxito de la subtarea)
# ----------------------------------------------------------------------
print("=" * 78)
print("T1 · Línea base léxica: TF-IDF (parámetros por defecto) + similitud coseno")
print(f"Vocabulario ajustado solo con los {len(DOC_IDS)} documentos: "
      f"{len(vectorizer.vocabulary_)} términos")
print("=" * 78)
for p in por_consulta:
    print(f"\n{p['consulta_id']}: {p['consulta']}")
    print(f"   Relevante esperado: {p['relevante']}  ->  posición "
          f"{p['posicion_del_relevante']}/10   (RR={p['reciprocal_rank']:.4f}, "
          f"Hit@1={p['hit@1']}, Hit@3={p['hit@3']})")
    for r, (doc, s) in enumerate(zip(p["ranking"], p["similitudes_en_orden_ranking"]), start=1):
        marca = "  <= documento relevante" if doc == p["relevante"] else ""
        print(f"   {r:>2}. {doc}  sim={s:.4f}{marca}")
print("-" * 78)
print("Métricas agregadas sobre las 6 consultas (línea base TF-IDF):")
print(f"   Hit@1 = {hit1:.6f}  ({n_hit1}/6 consultas)")
print(f"   Hit@3 = {hit3:.6f}  ({n_hit3}/6 consultas)")
print(f"   MRR   = {mrr:.6f}")
print("-" * 78)
print("Archivos escritos: resultados.json, tfidf_similitud_coseno.png, tfidf_rr_por_consulta.png")
