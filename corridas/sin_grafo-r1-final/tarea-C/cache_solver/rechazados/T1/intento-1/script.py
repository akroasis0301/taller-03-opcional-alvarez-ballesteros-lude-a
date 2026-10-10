# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica: TF-IDF + similitud coseno.

- Vectoriza los 10 documentos y las 6 consultas con TfidfVectorizer
  (parámetros por defecto; el vectorizador se ajusta SOLO sobre los
  documentos y las consultas se transforman con ese vocabulario/IDF).
- Recupera por similitud coseno ordenando los 10 documentos por consulta.
- Calcula Hit@1, Hit@3 y MRR sobre las seis consultas con los juicios de
  relevancia: q1->d04, q2->d03, q3->d08, q4->d05, q5->d09, q6->d02.

Salidas (carpeta actual):
  - resultados.json                 (contrato de la subtarea)
  - t1_mapa_calor_similitud.png
  - t1_rr_por_consulta.png
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
# 1. Corpus: 10 documentos (texto del enunciado)
# ----------------------------------------------------------------------
docs = {
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
doc_ids = [f"d{i:02d}" for i in range(1, 11)]
doc_texts = [docs[d] for d in doc_ids]

# ----------------------------------------------------------------------
# 2. Las 6 consultas de la tarea y sus juicios de relevancia (1 relevante
#    por consulta, según el enunciado)
# ----------------------------------------------------------------------
queries = {
    "q1": "¿Cómo funciona BM25 como función de ranking?",
    "q2": "¿Qué es la generación aumentada con recuperación?",
    "q3": "¿Qué es el ajuste fino con LoRA?",
    "q4": "¿Qué son los embeddings densos y cómo se comparan con la similitud coseno?",
    "q5": "¿Qué hace la temperatura antes del softmax?",
    "q6": "¿Cómo es la arquitectura de los transformadores?",
}
relevancia = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}

query_ids = [f"q{i}" for i in range(1, 7)]
query_texts = [queries[q] for q in query_ids]

# ----------------------------------------------------------------------
# 3. TF-IDF (parámetros por defecto) y similitud coseno
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()                    # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)      # ajuste SOLO sobre documentos
X_queries = vectorizer.transform(query_texts)     # mismas columnas/vocabulario

S = cosine_similarity(X_queries, X_docs)          # matriz (6 consultas x 10 docs)

# ----------------------------------------------------------------------
# 4. Ranking de los 10 documentos por consulta (descendente por similitud;
#    empates resueltos de forma determinista por orden del documento)
# ----------------------------------------------------------------------
rankings = {}
for i, q in enumerate(query_ids):
    orden = np.argsort(-S[i], kind="stable")
    rankings[q] = [doc_ids[j] for j in orden]

# ----------------------------------------------------------------------
# 5. Métricas por consulta y agregadas: Hit@1, Hit@3, MRR
# ----------------------------------------------------------------------
por_consulta = {}
hit1, hit3, rr = [], [], []
for i, q in enumerate(query_ids):
    rel = relevancia[q]
    pos = rankings[q].index(rel) + 1              # posición 1-based del relevante
    h1 = 1.0 if pos == 1 else 0.0
    h3 = 1.0 if pos <= 3 else 0.0
    r = 1.0 / pos
    hit1.append(h1)
    hit3.append(h3)
    rr.append(r)
    por_consulta[q] = {
        "consulta": queries[q],
        "documento_relevante": rel,
        "posicion_del_relevante": int(pos),
        "similitud_con_relevante": float(S[i, doc_ids.index(rel)]),
        "hit@1": h1,
        "hit@3": h3,
        "reciprocal_rank": r,
        "ranking": rankings[q],
    }

Hit1 = float(np.mean(hit1))
Hit3 = float(np.mean(hit3))
MRR = float(np.mean(rr))

# ----------------------------------------------------------------------
# 6. Tabla de similitudes y resultados.json (contrato)
# ----------------------------------------------------------------------
sim_df = pd.DataFrame(S, index=query_ids, columns=doc_ids)

resultados = {
    "subtarea": "T1_linea_base_lexica_tfidf_coseno",
    "configuracion": {
        "vectorizador": "TfidfVectorizer (parametros por defecto)",
        "vocabulario_ajustado_sobre": "los 10 documentos",
        "similitud": "coseno",
        "n_documentos": 10,
        "n_consultas": 6,
        "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    },
    "consultas": queries,
    "juicios_de_relevancia": relevancia,
    "orden_consultas": query_ids,
    "orden_documentos": doc_ids,
    "matriz_similitud_consultas_x_documentos": [[float(v) for v in fila] for fila in S],
    "similitud_consulta_documento": {
        q: {d: float(sim_df.loc[q, d]) for d in doc_ids} for q in query_ids
    },
    "ranking_por_consulta": {q: rankings[q] for q in query_ids},
    "metricas_por_consulta": por_consulta,
    "Hit@1": Hit1,
    "Hit@3": Hit3,
    "MRR": MRR,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Figuras
# ----------------------------------------------------------------------
# 7a. Mapa de calor de similitudes (relevante marcado con *)
fig, ax = plt.subplots(figsize=(10, 4.8))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0, vmax=float(S.max()))
ax.set_xticks(np.arange(len(doc_ids)))
ax.set_xticklabels(doc_ids)
ax.set_yticks(np.arange(len(query_ids)))
ax.set_yticklabels(query_ids)
ax.set_xlabel("Documentos")
ax.set_ylabel("Consultas")
ax.set_title("T1 TF-IDF: similitud coseno consulta-documento (relevante marcado con *)")
for i, q in enumerate(query_ids):
    for j, d in enumerate(doc_ids):
        marca = "*" if relevancia[q] == d else ""
        ax.text(j, i, f"{S[i, j]:.2f}{marca}", ha="center", va="center", fontsize=8,
                color="white" if S[i, j] < 0.6 * S.max() else "black")
fig.colorbar(im, ax=ax, label="similitud coseno")
fig.tight_layout()
fig.savefig("t1_mapa_calor_similitud.png", dpi=120)
plt.close(fig)

# 7b. Rango recíproco por consulta
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(query_ids, rr, color="#4472c4")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Rango recíproco (1/posición)")
ax.set_title(f"T1 TF-IDF — Hit@1={Hit1:.3f}  Hit@3={Hit3:.3f}  MRR={MRR:.3f}")
for k, q in enumerate(query_ids):
    ax.text(k, rr[k] + 0.02, f"pos {por_consulta[q]['posicion_del_relevante']}",
            ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("t1_rr_por_consulta.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 8. Resumen por consola
# ----------------------------------------------------------------------
print("=" * 72)
print("T1 — Línea base léxica TF-IDF + similitud coseno")
print("=" * 72)
for q in query_ids:
    m = por_consulta[q]
    print(f"{q}: {m['consulta']}")
    print(f"   ranking: {' > '.join(m['ranking'])}")
    print(f"   relevante {m['documento_relevante']} en posición "
          f"{m['posicion_del_relevante']} (sim={m['similitud_con_relevante']:.4f})  "
          f"RR={m['reciprocal_rank']:.4f}")
print("-" * 72)
print(f"Hit@1 = {Hit1:.4f}")
print(f"Hit@3 = {Hit3:.4f}")
print(f"MRR   = {MRR:.4f}")
print("Archivos generados: resultados.json, t1_mapa_calor_similitud.png, "
      "t1_rr_por_consulta.png")
