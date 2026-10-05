#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la matriz TF-IDF de la Parte 1 (T1).

Reconstruye la misma matriz TF-IDF de T1 (TfidfVectorizer con parámetros por
defecto, ajustado SOLO con los 10 documentos del corpus), la proyecta a 4
dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA), proyecta
las 6 consultas al espacio latente (folding-in: q_lsa = q_tfidf · V^T),
recalcula la similitud coseno y reevalúa las mismas 6 consultas y juicios de
relevancia. Construye una sola tabla con Hit@1, Hit@3 y MRR de los dos métodos
(TF-IDF y LSA). El SVD se ajusta únicamente con los documentos; las consultas
solo se proyectan (no se usa nada del "test" para ajustar).

Salidas (carpeta actual):
  resultados.json               contrato: métricas, rankings LSA por consulta, tabla única
  t2_tabla_metricas.png         tabla única de métricas (TF-IDF vs LSA)
  t2_comparativa_metricas.png   barras agrupadas Hit@1 / Hit@3 / MRR
  t2_lsa_proyeccion.png         documentos y consultas en las componentes LSA 1–2
"""

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ---------------------------------------------------------------------------
# Corpus, consultas y juicios (idénticos al enunciado y a la subtarea T1)
# ---------------------------------------------------------------------------
DOCS = [
    ("d01", "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión."),
    ("d02", "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización."),
    ("d03", "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo."),
    ("d04", "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento."),
    ("d05", "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación."),
    ("d06", "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado."),
    ("d07", "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos."),
    ("d08", "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales."),
    ("d09", "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad."),
    ("d10", "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria."),
]

QUERIES = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

doc_ids = [d[0] for d in DOCS]
doc_texts = [d[1] for d in DOCS]
q_ids = [q[0] for q in QUERIES]
q_texts = [q[1] for q in QUERIES]
relevante = {q[0]: q[2] for q in QUERIES}


def evaluar_metodo(S):
    """Evalúa matriz de similitudes (n_consultas, n_docs): rankings, Hit@1, Hit@3, MRR."""
    por_consulta = {}
    hits1, hits3, rrs = [], [], []
    for i, qid in enumerate(q_ids):
        sims = np.asarray(S[i], dtype=float)
        orden = np.argsort(-sims, kind="stable")  # descendente, desempate estable por id
        ranking = [doc_ids[j] for j in orden]
        rel = relevante[qid]
        pos = ranking.index(rel) + 1
        h1 = 1.0 if pos == 1 else 0.0
        h3 = 1.0 if pos <= 3 else 0.0
        rr = 1.0 / pos
        hits1.append(h1)
        hits3.append(h3)
        rrs.append(rr)
        por_consulta[qid] = {
            "consulta": q_texts[i],
            "relevante": rel,
            "ranking": ranking,
            "ranking_con_similitud": [{"doc": doc_ids[j], "similitud": float(sims[j])} for j in orden],
            "posicion_relevante": int(pos),
            "hit@1": h1,
            "hit@3": h3,
            "reciprocal_rank": rr,
        }
    metricas = {
        "hit@1": float(np.mean(hits1)),
        "hit@3": float(np.mean(hits3)),
        "mrr": float(np.mean(rrs)),
    }
    return metricas, por_consulta


# ---------------------------------------------------------------------------
# 1) Misma matriz TF-IDF que T1 (vectorizador por defecto, ajustado con los 10 docs)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, igual que en T1
X_docs = vectorizer.fit_transform(doc_texts)   # (10, V)
X_queries = vectorizer.transform(q_texts)      # (6, V)
V = X_docs.shape[1]
if V <= 4:
    raise RuntimeError(f"Vocabulario demasiado pequeño ({V}) para TruncatedSVD(n_components=4).")

# Línea base TF-IDF + coseno (re-ejecutada aquí para la tabla comparativa)
S_tfidf = cosine_similarity(X_queries, X_docs)  # (6, 10)
m_tfidf, rank_tfidf = evaluar_metodo(S_tfidf)

# ---------------------------------------------------------------------------
# 2) LSA: TruncatedSVD(n_components=4, random_state=0) y folding-in de consultas
#    X_docs ≈ (U·Σ)·V^T ; docs_lsa = U·Σ ; q_lsa = q_tfidf · V^T
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)         # (10, 4)
X_queries_lsa = svd.transform(X_queries)       # (6, 4) folding-in
if hasattr(X_queries_lsa, "toarray"):
    X_queries_lsa = X_queries_lsa.toarray()
S_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)  # (6, 10)
m_lsa, rank_lsa = evaluar_metodo(S_lsa)

# ---------------------------------------------------------------------------
# 3) Tabla única con las tres métricas de los dos métodos
# ---------------------------------------------------------------------------
tabla_comparativa = [
    {"metodo": "TF-IDF (línea base, Parte 1)",
     "hit@1": m_tfidf["hit@1"], "hit@3": m_tfidf["hit@3"], "mrr": m_tfidf["mrr"]},
    {"metodo": "LSA (TruncatedSVD k=4, Parte 2)",
     "hit@1": m_lsa["hit@1"], "hit@3": m_lsa["hit@3"], "mrr": m_lsa["mrr"]},
]
diferencias = {k: m_lsa[k] - m_tfidf[k] for k in ("hit@1", "hit@3", "mrr")}

# ---------------------------------------------------------------------------
# 4) Verificación contra los resultados de T1 (entrada/T1/resultados.json)
# ---------------------------------------------------------------------------
t1_path = Path("entrada") / "T1" / "resultados.json"
t1_metricas = None
coincide_t1 = None
if t1_path.exists():
    try:
        with t1_path.open(encoding="utf-8") as fh:
            t1_json = json.load(fh)
        t1_metricas = t1_json.get("metricas")
        if isinstance(t1_metricas, dict):
            claves = [k for k in ("hit@1", "hit@3", "mrr") if k in t1_metricas]
            if claves:
                max_diff = max(abs(float(t1_metricas[k]) - m_tfidf[k]) for k in claves)
                coincide_t1 = bool(max_diff < 1e-9)
    except (OSError, ValueError):
        t1_metricas = None
        coincide_t1 = None

# ---------------------------------------------------------------------------
# 5) Figuras (PNG en la carpeta actual)
# ---------------------------------------------------------------------------
# 5a) Tabla única de métricas
fig, ax = plt.subplots(figsize=(7.4, 2.6))
ax.axis("off")
col_labels = ["Método", "Hit@1", "Hit@3", "MRR"]
cell_text = [
    ["TF-IDF (línea base)", f"{m_tfidf['hit@1']:.4f}", f"{m_tfidf['hit@3']:.4f}", f"{m_tfidf['mrr']:.4f}"],
    ["LSA (k=4)", f"{m_lsa['hit@1']:.4f}", f"{m_lsa['hit@3']:.4f}", f"{m_lsa['mrr']:.4f}"],
    ["Diferencia (LSA - TF-IDF)", f"{diferencias['hit@1']:+.4f}", f"{diferencias['hit@3']:+.4f}", f"{diferencias['mrr']:+.4f}"],
]
tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.7)
ax.set_title("Tarea C · Parte 2 — Hit@1, Hit@3 y MRR: TF-IDF vs LSA (6 consultas, 10 documentos)",
             fontsize=11, pad=14)
plt.savefig("t2_tabla_metricas.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 5b) Barras agrupadas
fig, ax = plt.subplots(figsize=(7.0, 4.2))
claves = ["hit@1", "hit@3", "mrr"]
nombres = ["Hit@1", "Hit@3", "MRR"]
x = np.arange(len(claves))
ancho = 0.35
b_tfidf = ax.bar(x - ancho / 2, [m_tfidf[c] for c in claves], ancho,
                 label="TF-IDF (línea base)", color="#4C72B0")
b_lsa = ax.bar(x + ancho / 2, [m_lsa[c] for c in claves], ancho,
               label="LSA (TruncatedSVD k=4)", color="#DD8452")
for barras in (b_tfidf, b_lsa):
    for rect in barras:
        alto = rect.get_height()
        ax.annotate(f"{alto:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, alto),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(nombres)
ax.set_ylim(0.0, 1.15)
ax.set_ylabel("Valor de la métrica")
ax.set_title("Recuperación TF-IDF vs LSA — 10 documentos, 6 consultas")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.savefig("t2_comparativa_metricas.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 5c) Proyección LSA (componentes 1 y 2 de 4)
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.scatter(X_docs_lsa[:, 0], X_docs_lsa[:, 1], s=70, c="#4C72B0", zorder=3,
           label="Documentos (LSA)")
for j, did in enumerate(doc_ids):
    ax.annotate(did, (X_docs_lsa[j, 0], X_docs_lsa[j, 1]), xytext=(5, 4),
                textcoords="offset points", fontsize=9)
ax.scatter(X_queries_lsa[:, 0], X_queries_lsa[:, 1], s=95, c="#C44E52", marker="x",
           zorder=3, label="Consultas (LSA)")
for i, qid in enumerate(q_ids):
    ax.annotate(qid, (X_queries_lsa[i, 0], X_queries_lsa[i, 1]), xytext=(5, 4),
                textcoords="offset points", fontsize=9, color="#C44E52")
    j = doc_ids.index(relevante[qid])
    ax.plot([X_queries_lsa[i, 0], X_docs_lsa[j, 0]],
            [X_queries_lsa[i, 1], X_docs_lsa[j, 1]],
            ls="--", lw=0.9, c="#55A868", alpha=0.8, zorder=2)
ax.set_xlabel("Componente LSA 1")
ax.set_ylabel("Componente LSA 2")
ax.set_title("Proyección LSA (componentes 1–2 de 4); punteado = par consulta–relevante")
ax.grid(alpha=0.3)
ax.legend(loc="best")
plt.savefig("t2_lsa_proyeccion.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": (
        "Semántica latente: la misma matriz TF-IDF de T1 (TfidfVectorizer por defecto, "
        "ajustado con los 10 documentos) se proyecta a 4 dimensiones con "
        "TruncatedSVD(n_components=4, random_state=0); las consultas se proyectan al "
        "espacio latente (folding-in q·V^T) y se reevalúan las mismas 6 consultas y "
        "juicios con similitud coseno. Tabla única con Hit@1, Hit@3 y MRR de TF-IDF y LSA."
    ),
    "config": {
        "n_documentos": len(DOCS),
        "n_consultas": len(QUERIES),
        "tam_vocabulario_tfidf": int(V),
        "n_componentes_lsa": 4,
        "random_state": 0,
        "vectorizador": "TfidfVectorizer (parámetros por defecto)",
        "metrica_similitud": "coseno",
    },
    "metricas": {"tfidf": m_tfidf, "lsa": m_lsa},
    "tfidf": {
        "metricas": m_tfidf,
        "rankings_por_consulta": rank_tfidf,
        "matriz_similitud": [[float(v) for v in fila] for fila in S_tfidf],
    },
    "lsa": {
        "metricas": m_lsa,
        "rankings_por_consulta": rank_lsa,
        "matriz_similitud": [[float(v) for v in fila] for fila in S_lsa],
        "vectores_documentos": {doc_ids[j]: [float(v) for v in X_docs_lsa[j]]
                                for j in range(len(doc_ids))},
        "vectores_consultas": {q_ids[i]: [float(v) for v in X_queries_lsa[i]]
                               for i in range(len(q_ids))},
        "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
        "varianza_explicada_total": float(np.sum(svd.explained_variance_ratio_)),
    },
    "tabla_comparativa": tabla_comparativa,
    "diferencia_lsa_menos_tfidf": diferencias,
    "verificacion_con_T1": {
        "archivo": "entrada/T1/resultados.json",
        "metricas_t1": t1_metricas,
        "tfidf_recalculado_coincide_con_T1": coincide_t1,
    },
    "figuras": [
        "t2_tabla_metricas.png",
        "t2_comparativa_metricas.png",
        "t2_lsa_proyeccion.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 7) Resumen breve
# ---------------------------------------------------------------------------
print("=" * 70)
print("T2 — TF-IDF vs LSA (TruncatedSVD n_components=4, random_state=0)")
print("=" * 70)
print(f"Matriz TF-IDF: {X_docs.shape[0]} docs x {V} terminos -> LSA {X_docs_lsa.shape[1]} dims")
print(f"Varianza explicada por las 4 componentes LSA: "
      f"{resultados['lsa']['varianza_explicada_total']:.4f}")
print("-" * 70)
print(f"{'Método':<32}{'Hit@1':>10}{'Hit@3':>10}{'MRR':>10}")
for fila in tabla_comparativa:
    print(f"{fila['metodo']:<32}{fila['hit@1']:>10.4f}{fila['hit@3']:>10.4f}{fila['mrr']:>10.4f}")
print("-" * 70)
print("Rankings LSA por consulta (posición del relevante):")
for qid in q_ids:
    r = rank_lsa[qid]
    cadena = " > ".join(r["ranking"])
    print(f"  {qid} [rel={r['relevante']}, pos={r['posicion_relevante']}, "
          f"RR={r['reciprocal_rank']:.3f}]: {cadena}")
print("-" * 70)
if t1_metricas is not None:
    if coincide_t1 is None:
        estado = "no verificable"
    elif coincide_t1:
        estado = "coincide con el recalculado"
    else:
        estado = "NO coincide con el recalculado"
    print(f"Verificación T1: métricas TF-IDF de T1 {estado}.")
print("Salidas: resultados.json, t2_tabla_metricas.png, "
      "t2_comparativa_metricas.png, t2_lsa_proyeccion.png")
