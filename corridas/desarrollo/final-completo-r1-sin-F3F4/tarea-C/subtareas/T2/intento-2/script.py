# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la misma matriz TF-IDF de T1.

Reutiliza el corpus de 10 documentos, las 6 consultas y los juicios de T1,
proyecta la matriz TF-IDF a 4 dimensiones con TruncatedSVD(n_components=4,
random_state=0) (LSA), repite la recuperación con similitud coseno en el
espacio latente y construye UNA tabla con Hit@1, Hit@3 y MRR de los dos
métodos, más el ranking por consulta del método LSA.

Corrección: las matrices TF-IDF son dispersas (scipy.sparse); el producto
matricial disperso debe convertirse a denso con .toarray() ANTES de
np.asarray (np.asarray sobre una matriz dispersa produce un array 0-d,
que causaba el IndexError).
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
from sklearn.preprocessing import normalize


def a_denso(matriz):
    """Convierte una matriz (dispersa o densa) a un ndarray denso 2-D float."""
    if hasattr(matriz, "toarray"):
        matriz = matriz.toarray()
    arr = np.asarray(matriz, dtype=float)
    if arr.ndim == 0:  # salvaguarda extra
        arr = np.asarray(matriz.item(), dtype=float)
    return np.atleast_2d(arr)


# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios (idénticos a T1, tomados del enunciado)
# ---------------------------------------------------------------------------
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]
DOCS = [
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
QUERY_IDS = [f"q{i}" for i in range(1, 7)]
QUERIES = [
    "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "¿Cómo se añaden fragmentos recuperados al prompt?",
    "¿Qué técnica entrena matrices de bajo rango?",
    "¿Cómo se compara la orientación de dos vectores de texto?",
    "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "¿Qué arquitectura combina autoatención con capas feed-forward?",
]
RELEVANTE = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}
TEXTO_Q = dict(zip(QUERY_IDS, QUERIES))

# ---------------------------------------------------------------------------
# 2) Resultados previos de T1 (para la tabla comparativa y verificación)
# ---------------------------------------------------------------------------
p_t1 = Path("entrada") / "T1" / "resultados.json"
t1 = None
if p_t1.exists():
    try:
        t1 = json.loads(p_t1.read_text(encoding="utf-8"))
    except Exception:
        t1 = None

# ---------------------------------------------------------------------------
# 3) Misma matriz TF-IDF que T1 (parámetros por defecto, fit solo con los docs)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, como en T1
X_docs = vectorizer.fit_transform(DOCS)      # fit SOLO con los 10 documentos
X_queries = vectorizer.transform(QUERIES)    # consultas solo se transforman

X_docs_n = normalize(X_docs, norm="l2")
X_queries_n = normalize(X_queries, norm="l2")

# Coseno TF-IDF: producto disperso -> denso (6, 10)
sim_tfidf = a_denso(X_queries_n @ X_docs_n.T)

# ---------------------------------------------------------------------------
# 4) LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
L_docs = svd.fit_transform(X_docs)       # documentos en el espacio latente (denso)
L_queries = svd.transform(X_queries)     # consultas proyectadas con la misma base

L_docs_n = normalize(L_docs, norm="l2")
L_queries_n = normalize(L_queries, norm="l2")

# Coseno en el espacio latente (denso, (6, 10))
sim_lsa = a_denso(L_queries_n @ L_docs_n.T)

# ---------------------------------------------------------------------------
# 5) Rankings y métricas (Hit@1, Hit@3, MRR sobre las 6 consultas)
# ---------------------------------------------------------------------------
def ranking_de_fila(fila):
    fila = np.asarray(fila, dtype=float).ravel()
    orden = np.argsort(-fila, kind="stable")  # descendente, desempate por id
    return [DOC_IDS[j] for j in orden], orden


def metricas(ranks):
    r = np.asarray(ranks, dtype=float)
    return {
        "hit@1": float(np.mean(r == 1)),
        "hit@3": float(np.mean(r <= 3)),
        "mrr": float(np.mean(1.0 / r)),
    }


def evaluar(sim):
    sim = a_denso(sim)
    assert sim.shape == (len(QUERY_IDS), len(DOC_IDS)), f"forma inesperada: {sim.shape}"
    rankings, ranks_rel, sims_ord = {}, {}, {}
    for i, qid in enumerate(QUERY_IDS):
        ids_ord, orden = ranking_de_fila(sim[i])
        rankings[qid] = ids_ord
        ranks_rel[qid] = int(ids_ord.index(RELEVANTE[qid]) + 1)
        sims_ord[qid] = [float(sim[i, j]) for j in orden]
    return rankings, ranks_rel, sims_ord, metricas(list(ranks_rel.values()))


rank_tfidf, rel_tfidf, sims_ord_tfidf, met_tfidf = evaluar(sim_tfidf)
rank_lsa, rel_lsa, sims_ord_lsa, met_lsa = evaluar(sim_lsa)

# ---------------------------------------------------------------------------
# 6) Tabla única de los dos métodos
# ---------------------------------------------------------------------------
tabla_unica = {
    "TF-IDF (línea base T1)": {
        "Hit@1": met_tfidf["hit@1"], "Hit@3": met_tfidf["hit@3"], "MRR": met_tfidf["mrr"],
    },
    "LSA (TruncatedSVD k=4)": {
        "Hit@1": met_lsa["hit@1"], "Hit@3": met_lsa["hit@3"], "MRR": met_lsa["mrr"],
    },
}
tabla_metricas = [
    {"metodo": metodo, "Hit@1": m["Hit@1"], "Hit@3": m["Hit@3"], "MRR": m["MRR"]}
    for metodo, m in tabla_unica.items()
]
filas_md = ["| Método | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|"]
for metodo, m in tabla_unica.items():
    filas_md.append(f"| {metodo} | {m['Hit@1']} | {m['Hit@3']} | {m['MRR']} |")
tabla_unica_md = "\n".join(filas_md)

tabla_por_consulta = [
    {
        "consulta_id": qid,
        "consulta": TEXTO_Q[qid],
        "documento_relevante": RELEVANTE[qid],
        "rank_tfidf": rel_tfidf[qid],
        "rank_lsa": rel_lsa[qid],
        "hit@1_tfidf": float(rel_tfidf[qid] == 1),
        "hit@3_tfidf": float(rel_tfidf[qid] <= 3),
        "reciprocal_rank_tfidf": 1.0 / rel_tfidf[qid],
        "hit@1_lsa": float(rel_lsa[qid] == 1),
        "hit@3_lsa": float(rel_lsa[qid] <= 3),
        "reciprocal_rank_lsa": 1.0 / rel_lsa[qid],
    }
    for qid in QUERY_IDS
]
df_pc = pd.DataFrame(tabla_por_consulta)

delta = {k: met_lsa[k] - met_tfidf[k] for k in ("hit@1", "hit@3", "mrr")}

# Verificación contra T1 (la línea base TF-IDF debe reproducirse exactamente)
verif = {"archivo_T1": str(p_t1), "disponible": t1 is not None}
if t1 is not None:
    verif.update(
        {
            "hit@1_coincide": bool(np.isclose(t1.get("hit@1", np.nan), met_tfidf["hit@1"], atol=1e-9)),
            "hit@3_coincide": bool(np.isclose(t1.get("hit@3", np.nan), met_tfidf["hit@3"], atol=1e-9)),
            "mrr_coincide": bool(np.isclose(t1.get("mrr", np.nan), met_tfidf["mrr"], atol=1e-9)),
            "rankings_coinciden": bool(t1.get("rankings") == rank_tfidf),
        }
    )

# ---------------------------------------------------------------------------
# 7) Figuras
# ---------------------------------------------------------------------------
# Figura 1: barras comparativas de las tres métricas
fig, ax = plt.subplots(figsize=(7.2, 4.2))
nombres = ["Hit@1", "Hit@3", "MRR"]
claves = ["hit@1", "hit@3", "mrr"]
x = np.arange(3)
w = 0.35
vals_t = [met_tfidf[c] for c in claves]
vals_l = [met_lsa[c] for c in claves]
b1 = ax.bar(x - w / 2, vals_t, w, label="TF-IDF (línea base)", color="#4C72B0")
b2 = ax.bar(x + w / 2, vals_l, w, label="LSA (TruncatedSVD k=4)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(nombres)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Valor de la métrica")
ax.set_title("Recuperación sobre 6 consultas: TF-IDF vs LSA (k=4, random_state=0)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("t2_metricas_comparativas.png", dpi=120)
plt.close(fig)

# Figura 2: mapa de calor de similitudes coseno LSA (relevante enmarcado)
fig, ax = plt.subplots(figsize=(8.4, 4.6))
vmax = max(float(sim_lsa.max()), 1e-9)
vmin = min(float(sim_lsa.min()), 0.0)
im = ax.imshow(sim_lsa, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(DOC_IDS)))
ax.set_xticklabels(DOC_IDS, rotation=45)
ax.set_yticks(range(len(QUERY_IDS)))
ax.set_yticklabels(QUERY_IDS)
for i in range(len(QUERY_IDS)):
    for j in range(len(DOC_IDS)):
        v = float(sim_lsa[i, j])
        frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if frac > 0.6 else "black")
for i, qid in enumerate(QUERY_IDS):
    j = DOC_IDS.index(RELEVANTE[qid])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
fig.colorbar(im, ax=ax, label="similitud coseno (espacio latente)")
ax.set_title("LSA k=4: similitudes consulta-documento (rojo = documento relevante)")
fig.tight_layout()
fig.savefig("t2_heatmap_similitud_lsa.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
sim_lsa_por_consulta = {
    qid: {DOC_IDS[j]: float(sim_lsa[i, j]) for j in range(len(DOC_IDS))}
    for i, qid in enumerate(QUERY_IDS)
}
detalle_lsa = {
    qid: {
        "consulta": TEXTO_Q[qid],
        "documento_relevante": RELEVANTE[qid],
        "ranking_completo": rank_lsa[qid],
        "similitudes_ordenadas": sims_ord_lsa[qid],
        "rank_del_relevante": rel_lsa[qid],
        "hit@1": float(rel_lsa[qid] == 1),
        "hit@3": float(rel_lsa[qid] <= 3),
        "reciprocal_rank": 1.0 / rel_lsa[qid],
    }
    for qid in QUERY_IDS
}

resultados = {
    "subtarea": "T2",
    "titulo": "Semántica latente: LSA (TruncatedSVD k=4, random_state=0) vs línea base TF-IDF",
    "config": {
        "vectorizador": "TfidfVectorizer (scikit-learn, parámetros por defecto), fit solo con los 10 documentos",
        "lsa": "TruncatedSVD(n_components=4, random_state=0) ajustado sobre la matriz TF-IDF de los 10 documentos",
        "proyeccion_consultas": "svd.transform sobre la matriz TF-IDF de las consultas (misma base latente)",
        "similitud": "coseno en el espacio latente",
        "n_documentos": 10,
        "n_consultas": 6,
        "n_componentes_lsa": 4,
        "random_state": 0,
        "n_terminos_vocabulario": int(len(vectorizer.vocabulary_)),
        "varianza_explicada_por_componente": svd.explained_variance_ratio_.tolist(),
        "varianza_explicada_acumulada": float(svd.explained_variance_ratio_.sum()),
        "valores_singulares": svd.singular_values_.tolist(),
    },
    "tabla_unica": tabla_unica,
    "tabla_metricas": tabla_metricas,
    "tabla_unica_md": tabla_unica_md,
    "metricas_tfidf": met_tfidf,
    "metricas_lsa": met_lsa,
    "delta_lsa_menos_tfidf": delta,
    "tabla_por_consulta": tabla_por_consulta,
    "rankings_lsa": rank_lsa,
    "rankings_tfidf_recalculado": rank_tfidf,
    "ranks_del_documento_relevante_lsa": rel_lsa,
    "ranks_del_documento_relevante_tfidf": rel_tfidf,
    "similitudes_lsa_por_consulta": sim_lsa_por_consulta,
    "similitudes_ordenadas_lsa": sims_ord_lsa,
    "detalle_por_consulta_lsa": detalle_lsa,
    "verificacion_con_T1": verif,
    "figuras": ["t2_metricas_comparativas.png", "t2_heatmap_similitud_lsa.png"],
}

Path("resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ---------------------------------------------------------------------------
# 9) Resumen
# ---------------------------------------------------------------------------
print("=" * 64)
print("T2 — LSA (TruncatedSVD k=4, random_state=0) vs TF-IDF")
print("=" * 64)
print(tabla_unica_md)
print("-" * 64)
print("Rank del documento relevante por consulta (LSA): "
      + ", ".join(f"{q}:{rel_lsa[q]}" for q in QUERY_IDS))
print("Rank del documento relevante por consulta (TF-IDF): "
      + ", ".join(f"{q}:{rel_tfidf[q]}" for q in QUERY_IDS))
print(f"Varianza explicada acumulada (4 componentes): "
      f"{svd.explained_variance_ratio_.sum():.4f}")
if t1 is not None:
    ok = verif["hit@1_coincide"] and verif["hit@3_coincide"] and verif["mrr_coincide"] and verif["rankings_coinciden"]
    print(f"Verificación contra T1 (línea base TF-IDF reproducida): {'OK' if ok else 'REVISAR'}")
print("Archivos escritos: resultados.json, t2_metricas_comparativas.png, t2_heatmap_similitud_lsa.png")
