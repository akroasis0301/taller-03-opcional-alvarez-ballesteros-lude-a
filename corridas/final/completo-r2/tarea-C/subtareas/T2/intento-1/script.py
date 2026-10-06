# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la misma matriz TF-IDF de la Parte 1.

Reconstruye el corpus d01–d10, las consultas q1–q6 y los juicios de relevancia
de la Parte 1, vectoriza con TfidfVectorizer (parámetros por defecto,
vocabulario ajustado SOLO con los 10 documentos), proyecta documentos y
consultas a 4 dimensiones con TruncatedSVD(n_components=4, random_state=0),
repite el ranking por similitud coseno y evalúa Hit@1, Hit@3 y MRR de los dos
métodos (TF-IDF y LSA) en una sola tabla.

Salidas: resultados.json, t2_tabla_metricas.csv,
         t2_metricas_tfidf_vs_lsa.png, t2_mapa_calor_lsa.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1) Corpus, consultas y juicios (textuales del enunciado, Parte 1)
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
relevante = {q: QUERIES[q][1] for q in query_ids}

# ----------------------------------------------------------------------------
# 2) Misma matriz TF-IDF que la Parte 1 (parámetros por defecto; vocabulario
#    ajustado SOLO con los 10 documentos, las consultas solo se transforman)
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(doc_texts)      # (10, |V|)
X_queries = vectorizer.transform(query_texts)     # (6, |V|)

# ----------------------------------------------------------------------------
# 3) LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)            # (10, 4)
X_queries_lsa = svd.transform(X_queries)          # (6, 4) — mismas componentes

# ----------------------------------------------------------------------------
# 4) Ranking por similitud coseno (empates por orden de id, como en la Parte 1)
# ----------------------------------------------------------------------------
def rankear(X_q, X_d):
    S = cosine_similarity(X_q, X_d)
    sal = {}
    for i, qid in enumerate(query_ids):
        orden = np.argsort(-S[i], kind="stable")
        sal[qid] = {
            "ranking": [doc_ids[j] for j in orden],
            "similitudes": [float(S[i, j]) for j in orden],
        }
    return S, sal

S_tfidf, rank_tfidf = rankear(X_queries, X_docs)
S_lsa, rank_lsa = rankear(X_queries_lsa, X_docs_lsa)

# ----------------------------------------------------------------------------
# 5) Evaluación: Hit@1, Hit@3 y MRR sobre las 6 consultas
# ----------------------------------------------------------------------------
def evaluar(rank):
    posiciones, rr, h1, h3 = {}, [], 0, 0
    for qid in query_ids:
        pos = rank[qid]["ranking"].index(relevante[qid]) + 1
        posiciones[qid] = pos
        h1 += int(pos == 1)
        h3 += int(pos <= 3)
        rr.append(1.0 / pos)
    n = len(query_ids)
    return {"Hit@1": h1 / n, "Hit@3": h3 / n, "MRR": float(np.mean(rr))}, posiciones

met_tfidf, pos_tfidf = evaluar(rank_tfidf)
met_lsa, pos_lsa = evaluar(rank_lsa)

# ----------------------------------------------------------------------------
# 6) Tabla única comparativa (las tres métricas para los dos métodos)
# ----------------------------------------------------------------------------
METRICAS = ["Hit@1", "Hit@3", "MRR"]
tabla = pd.DataFrame(
    {"TF-IDF": [met_tfidf[m] for m in METRICAS],
     "LSA (k=4)": [met_lsa[m] for m in METRICAS]},
    index=METRICAS,
)
tabla.index.name = "métrica"
tabla.to_csv("t2_tabla_metricas.csv", encoding="utf-8")
print("Tabla única — TF-IDF vs LSA (TruncatedSVD, 4 componentes):")
print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))

# ----------------------------------------------------------------------------
# 7) Verificación opcional contra la subtarea previa T1 (no altera resultados)
# ----------------------------------------------------------------------------
check_t1 = None
t1_path = Path("entrada") / "T1" / "resultados.json"
if t1_path.exists():
    try:
        t1 = json.loads(t1_path.read_text(encoding="utf-8"))
        t1_rank = t1.get("rankings_por_consulta", {})
        comparables = [q for q in query_ids if q in t1_rank and "ranking" in t1_rank[q]]
        if comparables:
            check_t1 = all(
                list(t1_rank[q]["ranking"]) == rank_tfidf[q]["ranking"]
                for q in comparables
            )
    except Exception:
        check_t1 = None
if check_t1 is not None:
    print(f"[check] Rankings TF-IDF idénticos a los de T1: {check_t1}")

# ----------------------------------------------------------------------------
# 8) Figuras
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(len(METRICAS))
w = 0.35
b1 = ax.bar(x - w / 2, [met_tfidf[m] for m in METRICAS], w, label="TF-IDF", color="#4C72B0")
b2 = ax.bar(x + w / 2, [met_lsa[m] for m in METRICAS], w, label="LSA (k=4)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(METRICAS)
ax.set_ylim(0, 1.12)
ax.set_ylabel("valor")
ax.set_title("Hit@1, Hit@3 y MRR: TF-IDF vs LSA (TruncatedSVD k=4, random_state=0)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("t2_metricas_tfidf_vs_lsa.png", dpi=120)
plt.close(fig)

vmin = min(0.0, float(S_lsa.min()))
vmax = max(float(S_lsa.max()), 1e-9)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
im = ax.imshow(S_lsa, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids, rotation=90, fontsize=8)
ax.set_yticks(range(len(query_ids)))
ax.set_yticklabels(query_ids, fontsize=8)
for i in range(len(query_ids)):
    for j in range(len(doc_ids)):
        t = (S_lsa[i, j] - vmin) / (vmax - vmin)
        ax.text(j, i, f"{S_lsa[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if t < 0.55 else "black")
ax.set_title("Similitud coseno consulta–documento en el espacio LSA (k=4)")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
fig.savefig("t2_mapa_calor_lsa.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 9) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T2 — Semántica latente (LSA) sobre la matriz TF-IDF de la Parte 1",
    "metodo": (
        "TfidfVectorizer (parámetros por defecto, vocabulario ajustado solo con los 10 "
        "documentos, consultas transformadas) -> TruncatedSVD(n_components=4, random_state=0) "
        "ajustado sobre la matriz TF-IDF de documentos; consultas proyectadas con el mismo "
        "SVD; ranking por similitud coseno"
    ),
    "n_documentos": len(doc_ids),
    "n_consultas": len(query_ids),
    "n_componentes": 4,
    "random_state": 0,
    "tam_vocabulario": len(vectorizer.vocabulary_),
    "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
    "varianza_explicada_acumulada": float(svd.explained_variance_ratio_.sum()),
    "vectores_documentos_lsa": {d: [float(v) for v in X_docs_lsa[i]]
                                for i, d in enumerate(doc_ids)},
    "vectores_consultas_lsa": {q: [float(v) for v in X_queries_lsa[i]]
                               for i, q in enumerate(query_ids)},
    "matriz_similitud_lsa": {
        "filas": query_ids,
        "columnas": doc_ids,
        "valores": [[float(S_lsa[i, j]) for j in range(len(doc_ids))]
                    for i in range(len(query_ids))],
    },
    "rankings_por_consulta_lsa": {
        qid: {
            "consulta": QUERIES[qid][0],
            "relevante": relevante[qid],
            "ranking": rank_lsa[qid]["ranking"],
            "similitudes": rank_lsa[qid]["similitudes"],
            "posicion_del_relevante": pos_lsa[qid],
        }
        for qid in query_ids
    },
    "rankings_por_consulta_tfidf": {
        qid: {
            "ranking": rank_tfidf[qid]["ranking"],
            "posicion_del_relevante": pos_tfidf[qid],
        }
        for qid in query_ids
    },
    "metricas": {"TF-IDF": met_tfidf, "LSA": met_lsa},
    "tabla_comparativa": {m: {"TF-IDF": met_tfidf[m], "LSA": met_lsa[m]} for m in METRICAS},
    "tabla_comparativa_como_filas": [
        {"métrica": m, "TF-IDF": met_tfidf[m], "LSA": met_lsa[m]} for m in METRICAS
    ],
    "verificacion_rankings_tfidf_contra_T1": check_t1,
    "figuras": ["t2_metricas_tfidf_vs_lsa.png", "t2_mapa_calor_lsa.png"],
    "tabla_csv": "t2_tabla_metricas.csv",
}
Path("resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ----------------------------------------------------------------------------
# 10) Resumen breve
# ----------------------------------------------------------------------------
print("\nResumen T2")
print(f"  Varianza explicada por las 4 componentes LSA: "
      f"{svd.explained_variance_ratio_.sum():.4f}")
print("  Posición del documento relevante (TF-IDF -> LSA):")
for qid in query_ids:
    print(f"    {qid}: {pos_tfidf[qid]} -> {pos_lsa[qid]}  (relevante: {relevante[qid]})")
print(f"  Hit@1: TF-IDF={met_tfidf['Hit@1']:.4f} | LSA={met_lsa['Hit@1']:.4f}")
print(f"  Hit@3: TF-IDF={met_tfidf['Hit@3']:.4f} | LSA={met_lsa['Hit@3']:.4f}")
print(f"  MRR  : TF-IDF={met_tfidf['MRR']:.4f} | LSA={met_lsa['MRR']:.4f}")
print("  Archivos: resultados.json, t2_tabla_metricas.csv, "
      "t2_metricas_tfidf_vs_lsa.png, t2_mapa_calor_lsa.png")
