# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la misma matriz TF-IDF, consultas y juicios de T1.

Pasos:
1) Reconstruye el corpus (10 documentos) y las 6 consultas con sus juicios (idénticos a T1).
2) Reconstruye la matriz TF-IDF con TfidfVectorizer (parámetros por defecto, ajustado SOLO
   sobre los documentos) -> idéntica a la de T1; se verifica contra entrada/T1/resultados.json.
3) Proyecta esa matriz a 4 dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA)
   y pliega las consultas al mismo espacio latente con svd.transform.
4) Repite el ranking por similitud coseno y la evaluación (Hit@1, Hit@3, MRR sobre 6 consultas).
5) Construye UNA sola tabla comparativa TF-IDF vs LSA y escribe resultados.json + figuras PNG.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------------
# 1) Corpus, consultas y juicios: idénticos a los de T1 (datos del enunciado)
# ----------------------------------------------------------------------------
doc_ids = [f"d{i:02d}" for i in range(1, 11)]
documentos = {
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
doc_texts = [documentos[d] for d in doc_ids]

consultas = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
qids = list(consultas.keys())
query_texts = [consultas[q][0] for q in qids]
relevantes = {q: consultas[q][1] for q in qids}

# ----------------------------------------------------------------------------
# 2) Resultados oficiales de T1 (línea base para la tabla y verificación)
# ----------------------------------------------------------------------------
t1_path = Path("entrada") / "T1" / "resultados.json"
t1 = None
if t1_path.exists():
    try:
        with t1_path.open(encoding="utf-8") as fh:
            t1 = json.load(fh)
    except Exception:
        t1 = None

# ----------------------------------------------------------------------------
# 3) Misma matriz TF-IDF que T1: TfidfVectorizer por defecto, ajustado SOLO
#    sobre los 10 documentos (determinista -> matriz idéntica a la de T1)
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, igual que en T1
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(query_texts)  # las consultas NO reajustan el vectorizador
vocab_size = int(len(vectorizer.vocabulary_))

# Similitud coseno TF-IDF (verificación de la línea base de T1)
sim_tfidf = normalize(X_queries).toarray() @ normalize(X_docs).toarray().T

# ----------------------------------------------------------------------------
# 4) Ranking y evaluación (Hit@1, Hit@3, MRR sobre las 6 consultas)
# ----------------------------------------------------------------------------
def ranking_y_eval(sim, qids, relevantes, doc_ids):
    """Ordena los 10 documentos por consulta (similitud descendente, empates
    en orden de documento) y calcula Hit@1, Hit@3 y MRR."""
    detalle, ranks = {}, {}
    for i, qid in enumerate(qids):
        orden = np.argsort(-sim[i], kind="stable")
        orden_ids = [doc_ids[j] for j in orden]
        rank = orden_ids.index(relevantes[qid]) + 1
        ranks[qid] = rank
        detalle[qid] = {
            "ranking_completo_10_docs": orden_ids,
            "similitudes_en_orden": [float(sim[i, j]) for j in orden],
            "rank_del_relevante": int(rank),
            "hit@1": 1.0 if rank == 1 else 0.0,
            "hit@3": 1.0 if rank <= 3 else 0.0,
            "reciprocal_rank": 1.0 / rank,
        }
    agregado = {
        "hit@1": float(np.mean([detalle[q]["hit@1"] for q in qids])),
        "hit@3": float(np.mean([detalle[q]["hit@3"] for q in qids])),
        "mrr": float(np.mean([detalle[q]["reciprocal_rank"] for q in qids])),
    }
    return detalle, ranks, agregado

det_tfidf, ranks_tfidf, agg_tfidf_recalc = ranking_y_eval(sim_tfidf, qids, relevantes, doc_ids)

# --- Verificación contra T1 (misma matriz TF-IDF, mismas consultas y juicios) ---
verif = {
    "t1_cargado": t1 is not None,
    "ruta_t1": "entrada/T1/resultados.json",
    "vocabulario_tamano_t1": None,
    "vocabulario_coincide": None,
    "rankings_tfidf_coinciden_con_T1": None,
    "metricas_agregadas_tfidf_coinciden_con_T1": None,
}
tfidf_oficial = dict(agg_tfidf_recalc)
ranks_tfidf_oficial = {q: int(ranks_tfidf[q]) for q in qids}

if t1 is not None:
    v_t1 = (t1.get("config", {}) or {}).get("vocabulario_tamano")
    verif["vocabulario_tamano_t1"] = v_t1
    verif["vocabulario_coincide"] = bool(v_t1 == vocab_size)

    t1_rank = t1.get("rankings_por_consulta", {}) or {}
    comparables = [q for q in qids
                   if isinstance(t1_rank.get(q), dict) and "ranking_completo_10_docs" in t1_rank[q]]
    if comparables:
        verif["rankings_tfidf_coinciden_con_T1"] = bool(all(
            list(t1_rank[q]["ranking_completo_10_docs"]) == det_tfidf[q]["ranking_completo_10_docs"]
            for q in comparables))

    t1_agg = t1.get("metricas_agregadas", {}) or {}
    if all(m in t1_agg for m in ("hit@1", "hit@3", "mrr")):
        verif["metricas_agregadas_tfidf_coinciden_con_T1"] = bool(all(
            abs(float(t1_agg[m]) - agg_tfidf_recalc[m]) < 1e-9
            for m in ("hit@1", "hit@3", "mrr")))
        tfidf_oficial = {m: float(t1_agg[m]) for m in ("hit@1", "hit@3", "mrr")}

    t1_mpc = t1.get("metricas_por_consulta", {}) or {}
    for q in qids:
        if isinstance(t1_mpc.get(q), dict) and "rank" in t1_mpc[q]:
            ranks_tfidf_oficial[q] = int(t1_mpc[q]["rank"])

# ----------------------------------------------------------------------------
# 5) LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
#    de T1; consultas plegadas al mismo espacio latente
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)     # proyección de los 10 documentos
X_queries_lsa = svd.transform(X_queries)   # plegado de consultas: q_tfidf @ Vt.T

sim_lsa = normalize(X_queries_lsa) @ normalize(X_docs_lsa).T  # coseno en espacio latente

det_lsa, ranks_lsa, agg_lsa = ranking_y_eval(sim_lsa, qids, relevantes, doc_ids)

# ----------------------------------------------------------------------------
# 6) Tabla única comparativa: TF-IDF vs LSA
# ----------------------------------------------------------------------------
tabla = pd.DataFrame([
    {"metodo": "TF-IDF (coseno, línea base T1)",
     "hit@1": tfidf_oficial["hit@1"], "hit@3": tfidf_oficial["hit@3"], "mrr": tfidf_oficial["mrr"]},
    {"metodo": "LSA (TruncatedSVD k=4, random_state=0)",
     "hit@1": agg_lsa["hit@1"], "hit@3": agg_lsa["hit@3"], "mrr": agg_lsa["mrr"]},
])
tabla.to_csv("tabla_comparativa.csv", index=False, encoding="utf-8")

comparado_por_consulta = {
    q: {"relevante": relevantes[q],
        "rank_tfidf": int(ranks_tfidf_oficial[q]),
        "rank_lsa": int(ranks_lsa[q])}
    for q in qids
}
diferencias = {m: float(agg_lsa[m] - tfidf_oficial[m]) for m in ("hit@1", "hit@3", "mrr")}

# ----------------------------------------------------------------------------
# 7) Figuras (PNG, dpi=120)
# ----------------------------------------------------------------------------
# 7a) La tabla única, renderizada como figura
fig, ax = plt.subplots(figsize=(8.4, 2.4))
ax.axis("off")
cell_text = [[r["metodo"], f"{r['hit@1']:.4f}", f"{r['hit@3']:.4f}", f"{r['mrr']:.4f}"]
             for _, r in tabla.iterrows()]
tbl = ax.table(cellText=cell_text, colLabels=["Método", "Hit@1", "Hit@3", "MRR"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
ax.set_title("Hit@1, Hit@3 y MRR de los dos métodos (6 consultas, 10 documentos)", pad=18)
fig.savefig("tabla_comparativa.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 7b) Barras agrupadas de las tres métricas
metricas = ["hit@1", "hit@3", "mrr"]
etiquetas = ["Hit@1", "Hit@3", "MRR"]
tf_vals = [tfidf_oficial[m] for m in metricas]
lsa_vals = [agg_lsa[m] for m in metricas]
x = np.arange(len(metricas))
w = 0.35
fig, ax = plt.subplots(figsize=(7.2, 4.6))
b1 = ax.bar(x - w / 2, tf_vals, w, label="TF-IDF (T1)", color="#4C72B0")
b2 = ax.bar(x + w / 2, lsa_vals, w, label="LSA (TruncatedSVD k=4)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(etiquetas)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Valor de la métrica")
ax.set_title("Recuperación sobre 6 consultas: TF-IDF vs LSA (k=4)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("comparacion_tfidf_vs_lsa.png", dpi=120)
plt.close(fig)

# 7c) Mapas de calor de similitudes (TF-IDF vs LSA), relevante marcado en rojo
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True)
for ax, sim, titulo in zip(axes, (sim_tfidf, sim_lsa),
                           ("TF-IDF (coseno) — línea base T1", "LSA k=4 (coseno)")):
    vmin, vmax = float(sim.min()), float(sim.max())
    im = ax.imshow(sim, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(doc_ids)))
    ax.set_xticklabels(doc_ids, rotation=90, fontsize=8)
    ax.set_yticks(range(len(qids)))
    ax.set_yticklabels(qids, fontsize=9)
    ax.set_title(titulo, fontsize=11)
    for i in range(len(qids)):
        for j in range(len(doc_ids)):
            frac = (sim[i, j] - vmin) / (vmax - vmin + 1e-12)
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if frac < 0.55 else "black")
    for i, q in enumerate(qids):
        j = doc_ids.index(relevantes[q])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="red", linewidth=1.8))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.suptitle("Similitud coseno consulta–documento (rectángulo rojo = documento relevante)")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("mapa_calor_similitud_tfidf_vs_lsa.png", dpi=120)
plt.close(fig)

# 7d) Proyección LSA: documentos y consultas en componentes 1–2
fig, ax = plt.subplots(figsize=(7.6, 6.0))
ax.scatter(X_docs_lsa[:, 0], X_docs_lsa[:, 1], s=70, c="#4C72B0", zorder=3, label="Documentos")
ax.scatter(X_queries_lsa[:, 0], X_queries_lsa[:, 1], s=80, c="#DD8452", marker="^",
           zorder=3, label="Consultas")
for k, d in enumerate(doc_ids):
    ax.annotate(d, (X_docs_lsa[k, 0], X_docs_lsa[k, 1]), xytext=(6, 4),
                textcoords="offset points", fontsize=9)
for i, q in enumerate(qids):
    ax.annotate(q, (X_queries_lsa[i, 0], X_queries_lsa[i, 1]), xytext=(6, 4),
                textcoords="offset points", fontsize=9, color="#a34a1f")
    j = doc_ids.index(relevantes[q])
    ax.plot([X_queries_lsa[i, 0], X_docs_lsa[j, 0]],
            [X_queries_lsa[i, 1], X_docs_lsa[j, 1]],
            color="gray", lw=0.9, ls="--", zorder=1)
ax.axhline(0, color="gray", lw=0.6)
ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel("Componente latente 1")
ax.set_ylabel("Componente latente 2")
ax.set_title("Proyección LSA (TruncatedSVD k=4): componentes 1–2")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("proyeccion_lsa_componentes_1_2.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 8) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T2_lsa_truncatedsvd4_vs_tfidf_tabla_comparativa",
    "config": {
        "vectorizador": "TfidfVectorizer (parametros por defecto), ajustado solo sobre los 10 documentos (misma matriz TF-IDF que T1)",
        "reduccion": "TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF de T1 (LSA); consultas plegadas con svd.transform",
        "n_componentes": 4,
        "random_state": 0,
        "similitud": "coseno en el espacio proyectado",
        "n_documentos": 10,
        "n_consultas": 6,
        "vocabulario_tamano": vocab_size,
        "valores_singulares": [float(v) for v in svd.singular_values_],
        "varianza_explicada_ratio_por_componente": [float(v) for v in svd.explained_variance_ratio_],
        "varianza_explicada_ratio_acumulada_k4": float(np.sum(svd.explained_variance_ratio_)),
    },
    "verificacion_con_T1": verif,
    "metricas_agregadas_tfidf": tfidf_oficial,
    "metricas_agregadas_lsa": agg_lsa,
    "diferencia_lsa_menos_tfidf": diferencias,
    "tabla_comparativa": tabla.to_dict(orient="records"),
    "tabla_comparativa_por_consulta": comparado_por_consulta,
    "rankings_por_consulta_lsa": {
        q: {
            "consulta": consultas[q][0],
            "relevante": relevantes[q],
            "ranking_completo_10_docs": det_lsa[q]["ranking_completo_10_docs"],
            "similitudes_lsa_en_orden": det_lsa[q]["similitudes_en_orden"],
            "rank_del_relevante": det_lsa[q]["rank_del_relevante"],
            "hit@1": det_lsa[q]["hit@1"],
            "hit@3": det_lsa[q]["hit@3"],
            "reciprocal_rank": det_lsa[q]["reciprocal_rank"],
        }
        for q in qids
    },
    "matriz_similitud_coseno_lsa": {
        "filas_consultas": qids,
        "columnas_documentos": doc_ids,
        "valores": [[float(v) for v in fila] for fila in sim_lsa],
    },
    "coordenadas_lsa": {
        "documentos": {d: [float(x) for x in X_docs_lsa[k]] for k, d in enumerate(doc_ids)},
        "consultas": {q: [float(x) for x in X_queries_lsa[i]] for i, q in enumerate(qids)},
    },
    "rankings_por_consulta_tfidf_recalculados": {
        q: det_tfidf[q]["ranking_completo_10_docs"] for q in qids
    },
    "figuras": [
        "tabla_comparativa.png",
        "comparacion_tfidf_vs_lsa.png",
        "mapa_calor_similitud_tfidf_vs_lsa.png",
        "proyeccion_lsa_componentes_1_2.png",
    ],
    "archivos": ["resultados.json", "tabla_comparativa.csv"],
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 9) Resumen
# ----------------------------------------------------------------------------
print("=" * 72)
print("T2 — LSA (TruncatedSVD n_components=4, random_state=0) vs TF-IDF (T1)")
print("=" * 72)
print(f"Vocabulario TF-IDF: {vocab_size} términos "
      f"(T1 reporta {verif['vocabulario_tamano_t1']}; coincide: {verif['vocabulario_coincide']})")
print(f"Rankings TF-IDF idénticos a T1: {verif['rankings_tfidf_coinciden_con_T1']}")
print(f"Varianza explicada por los 4 componentes LSA: "
      f"{resultados['config']['varianza_explicada_ratio_acumulada_k4']:.4f}")
print("-" * 72)
print("Tabla única (6 consultas, 10 documentos):")
print(tabla.to_string(index=False))
print("-" * 72)
print("Rank del documento relevante por consulta (TF-IDF -> LSA):")
for q in qids:
    print(f"  {q} (relevante {relevantes[q]}): {ranks_tfidf_oficial[q]} -> {ranks_lsa[q]}")
print("-" * 72)
print("Rankings LSA por consulta:")
for q in qids:
    print(f"  {q}: " + " > ".join(det_lsa[q]["ranking_completo_10_docs"]))
print("-" * 72)
print("Archivos generados: resultados.json, tabla_comparativa.csv y 4 figuras PNG")
