# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la misma matriz TF-IDF de la T1.

Reconstruye la matriz TF-IDF (TfidfVectorizer por defecto, fit sobre los 10
documentos, consultas solo transformadas — idéntico a T1), la proyecta a 4
dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA), repite el
ranking por similitud coseno y la evaluación (Hit@1, Hit@3, MRR) con las mismas
seis consultas y los mismos juicios, y construye UNA tabla comparativa única
TF-IDF vs LSA. La línea base TF-IDF se verifica contra entrada/T1/resultados.json.
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
# 1) Corpus, consultas y juicios (idénticos a T1; tomados del enunciado)
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
# 2) Funciones de ranking y evaluación
# ----------------------------------------------------------------------------
def rankear(sim_matrix):
    """Ordena los 10 documentos por similitud descendente para cada consulta.

    Desempate determinista: orden estable conserva el orden d01..d10.
    """
    rankings = {}
    for i, q in enumerate(query_ids):
        scores = sim_matrix[i]
        order = np.argsort(-scores, kind="stable")
        rankings[q] = [
            {"pos": int(pos + 1), "doc_id": doc_ids[j], "score": float(scores[j])}
            for pos, j in enumerate(order)
        ]
    return rankings


def evaluar(rankings):
    """Hit@1, Hit@3 y MRR (agregados sobre las 6 consultas y por consulta)."""
    por_consulta, h1s, h3s, rrs = {}, [], [], []
    for q in query_ids:
        rel = relevante[q]
        rank = next(r["pos"] for r in rankings[q] if r["doc_id"] == rel)
        rr = 1.0 / rank
        h1, h3 = int(rank == 1), int(rank <= 3)
        por_consulta[q] = {
            "consulta": QUERIES[q][0],
            "relevante": rel,
            "rank_relevante": int(rank),
            "reciprocal_rank": float(rr),
            "hit_at_1": h1,
            "hit_at_3": h3,
        }
        h1s.append(h1)
        h3s.append(h3)
        rrs.append(rr)
    agregadas = {
        "hit_at_1": float(np.mean(h1s)),
        "hit_at_3": float(np.mean(h3s)),
        "mrr": float(np.mean(rrs)),
        "n_consultas": len(query_ids),
    }
    return agregadas, por_consulta


# ----------------------------------------------------------------------------
# 3) Misma matriz TF-IDF que T1 (parámetros por defecto, fit sobre los 10 docs)
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)

# Línea base TF-IDF (recomputada de forma determinista; se verifica contra T1)
sim_tfidf = cosine_similarity(X_queries, X_docs)
rank_tfidf = rankear(sim_tfidf)
agg_tfidf, perq_tfidf = evaluar(rank_tfidf)

# ----------------------------------------------------------------------------
# 4) LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)
sim_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)
rank_lsa = rankear(sim_lsa)
agg_lsa, perq_lsa = evaluar(rank_lsa)

# ----------------------------------------------------------------------------
# 5) Verificación de la línea base contra los resultados de T1
# ----------------------------------------------------------------------------
t1_path = Path("entrada") / "T1" / "resultados.json"
verif = {"ruta": "entrada/T1/resultados.json", "disponible": t1_path.exists()}
if t1_path.exists():
    t1 = json.loads(t1_path.read_text(encoding="utf-8"))
    t1_agg = t1.get("metricas_agregadas", {})
    claves = ["hit_at_1", "hit_at_3", "mrr"]
    verif["metricas_T1"] = {k: t1_agg.get(k) for k in claves}
    verif["metricas_recomputadas_T2"] = {k: agg_tfidf[k] for k in claves}
    diffs = [abs(agg_tfidf[k] - t1_agg[k]) for k in claves if k in t1_agg]
    verif["diferencia_max_abs"] = float(max(diffs)) if diffs else None
    verif["metricas_coinciden"] = bool(diffs) and all(d < 1e-9 for d in diffs)
    t1_rank = t1.get("rankings", {})
    ranks_t1 = {
        q: next(r["pos"] for r in t1_rank[q] if r["doc_id"] == relevante[q])
        for q in query_ids
        if q in t1_rank
    }
    if ranks_t1:
        verif["ranks_por_consulta_T1"] = ranks_t1
        verif["ranks_coinciden"] = all(
            ranks_t1[q] == perq_tfidf[q]["rank_relevante"] for q in ranks_t1
        )

# ----------------------------------------------------------------------------
# 6) Tabla comparativa única (ambos métodos, las tres métricas, 6 consultas)
# ----------------------------------------------------------------------------
tabla = pd.DataFrame(
    [
        {
            "metodo": "TF-IDF (coseno)",
            "hit_at_1": agg_tfidf["hit_at_1"],
            "hit_at_3": agg_tfidf["hit_at_3"],
            "mrr": agg_tfidf["mrr"],
        },
        {
            "metodo": "LSA (TruncatedSVD k=4)",
            "hit_at_1": agg_lsa["hit_at_1"],
            "hit_at_3": agg_lsa["hit_at_3"],
            "mrr": agg_lsa["mrr"],
        },
    ]
)
tabla.to_csv("tabla_comparativa.csv", index=False)

tabla_q = pd.DataFrame(
    [
        {
            "consulta_id": q,
            "relevante": relevante[q],
            "rank_tfidf": perq_tfidf[q]["rank_relevante"],
            "rank_lsa": perq_lsa[q]["rank_relevante"],
            "hit1_tfidf": perq_tfidf[q]["hit_at_1"],
            "hit1_lsa": perq_lsa[q]["hit_at_1"],
            "hit3_tfidf": perq_tfidf[q]["hit_at_3"],
            "hit3_lsa": perq_lsa[q]["hit_at_3"],
            "rr_tfidf": perq_tfidf[q]["reciprocal_rank"],
            "rr_lsa": perq_lsa[q]["reciprocal_rank"],
        }
        for q in query_ids
    ]
)
tabla_q.to_csv("tabla_comparativa_por_consulta.csv", index=False)

# ----------------------------------------------------------------------------
# 7) Figuras (PNG, backend Agg)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.2))
labels = ["Hit@1", "Hit@3", "MRR"]
vals_t = [agg_tfidf["hit_at_1"], agg_tfidf["hit_at_3"], agg_tfidf["mrr"]]
vals_l = [agg_lsa["hit_at_1"], agg_lsa["hit_at_3"], agg_lsa["mrr"]]
x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w / 2, vals_t, w, label="TF-IDF", color="#4C72B0")
b2 = ax.bar(x + w / 2, vals_l, w, label="LSA (k=4)", color="#DD8452")
ax.bar_label(b1, fmt="%.3f", fontsize=8)
ax.bar_label(b2, fmt="%.3f", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.0, 1.08)
ax.set_ylabel("Valor de la métrica")
ax.set_title("Recuperación sobre 6 consultas: TF-IDF vs LSA (TruncatedSVD, k=4)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("comparativa_metricas_tfidf_lsa.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(len(query_ids))
r_t = [perq_tfidf[q]["rank_relevante"] for q in query_ids]
r_l = [perq_lsa[q]["rank_relevante"] for q in query_ids]
b1 = ax.bar(x - w / 2, r_t, w, label="TF-IDF", color="#4C72B0")
b2 = ax.bar(x + w / 2, r_l, w, label="LSA (k=4)", color="#DD8452")
ax.bar_label(b1, fmt="%d", fontsize=8)
ax.bar_label(b2, fmt="%d", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(query_ids)
ax.set_ylim(0, 10.8)
ax.set_ylabel("Rank del documento relevante (menor = mejor)")
ax.set_title("Rank del documento relevante por consulta")
ax.legend()
fig.tight_layout()
fig.savefig("ranks_por_consulta_tfidf_lsa.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 8) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tipo no serializable: {type(o)}")


resultados = {
    "subtarea": "T2",
    "titulo": "Semántica latente (LSA, TruncatedSVD k=4) sobre la misma matriz TF-IDF y tabla comparativa TF-IDF vs LSA",
    "config": {
        "tfidf": "TfidfVectorizer de scikit-learn con parámetros por defecto (idéntico a T1); fit sobre los 10 documentos, consultas solo transformadas",
        "lsa": "TruncatedSVD(n_components=4, random_state=0) ajustado sobre la matriz TF-IDF de los 10 documentos; consultas proyectadas con el mismo modelo",
        "n_componentes": 4,
        "random_state": 0,
        "similitud": "coseno (espacio TF-IDF para la línea base; espacio latente de 4 dimensiones para LSA)",
        "forma_matriz_tfidf_docs": list(X_docs.shape),
        "tam_vocabulario": int(len(vectorizer.vocabulary_)),
        "valores_singulares": [float(v) for v in svd.singular_values_],
        "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
        "varianza_explicada_total": float(svd.explained_variance_ratio_.sum()),
    },
    "juicios_relevancia": relevante,
    "tabla_comparativa": {
        "descripcion": "Tabla única con Hit@1, Hit@3 y MRR de los dos métodos (TF-IDF y LSA) sobre las seis consultas",
        "columnas": ["metodo", "hit_at_1", "hit_at_3", "mrr"],
        "filas": tabla.to_dict(orient="records"),
    },
    "metricas_agregadas": {"TF-IDF": agg_tfidf, "LSA": agg_lsa},
    "metricas_por_consulta": {"TF-IDF": perq_tfidf, "LSA": perq_lsa},
    "tabla_comparativa_por_consulta": tabla_q.to_dict(orient="records"),
    "rankings_lsa_por_consulta": rank_lsa,
    "rankings_tfidf_por_consulta": rank_tfidf,
    "documentos_en_espacio_lsa": {
        doc_ids[i]: [float(v) for v in X_docs_lsa[i]] for i in range(len(doc_ids))
    },
    "consultas_en_espacio_lsa": {
        query_ids[i]: [float(v) for v in X_queries_lsa[i]] for i in range(len(query_ids))
    },
    "verificacion_con_T1": verif,
    "archivos_generados": [
        "resultados.json",
        "tabla_comparativa.csv",
        "tabla_comparativa_por_consulta.csv",
        "comparativa_metricas_tfidf_lsa.png",
        "ranks_por_consulta_tfidf_lsa.png",
    ],
}

Path("resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2, default=_json_default),
    encoding="utf-8",
)

# ----------------------------------------------------------------------------
# 9) Resumen
# ----------------------------------------------------------------------------
print("T2 — LSA (TruncatedSVD k=4, random_state=0) vs TF-IDF sobre 6 consultas")
print("\nTabla comparativa (agregada sobre las 6 consultas):")
print(tabla.to_string(index=False))
print("\nRank del documento relevante por consulta (TF-IDF vs LSA):")
print(tabla_q[["consulta_id", "relevante", "rank_tfidf", "rank_lsa"]].to_string(index=False))
print(f"\nVarianza explicada por LSA (4 componentes): {svd.explained_variance_ratio_.sum():.4f}")
if verif.get("disponible"):
    print(f"Verificación contra entrada/T1/resultados.json: coincide = {verif.get('metricas_coinciden')}")
print("\nArchivos generados: resultados.json, tabla_comparativa.csv, "
      "tabla_comparativa_por_consulta.csv, comparativa_metricas_tfidf_lsa.png, "
      "ranks_por_consulta_tfidf_lsa.png")
