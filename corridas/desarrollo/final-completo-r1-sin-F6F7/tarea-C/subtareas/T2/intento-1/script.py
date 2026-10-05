#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 — Semántica latente (LSA, TruncatedSVD a 4 dims) vs línea base TF-IDF.

Reconstruye el corpus de 10 documentos, las 6 consultas y sus juicios de
relevancia exactamente como en el enunciado (mismos datos que la Parte 1),
replica la matriz TF-IDF de T1 (TfidfVectorizer por defecto ajustado SOLO con
los documentos; consultas transformadas), la proyecta a 4 dimensiones con
TruncatedSVD(n_components=4, random_state=0) (LSA), calcula similitud coseno
en el espacio latente, ordena los 10 documentos por consulta y evalúa
Hit@1, Hit@3 y MRR sobre las 6 consultas. Genera UNA tabla comparativa con
las tres métricas de los dos métodos (TF-IDF y LSA(4)) y verifica que la
línea base TF-IDF recalculada coincida con los rankings guardados de T1.
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

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios (idénticos al enunciado / Parte 1 / T1)
# ---------------------------------------------------------------------------
DOCUMENTOS = {
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

CONSULTAS = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

DOC_IDS = list(DOCUMENTOS)
QUERY_IDS = list(CONSULTAS)
DOC_TEXTS = [DOCUMENTOS[d] for d in DOC_IDS]
QUERY_TEXTS = [CONSULTAS[q][0] for q in QUERY_IDS]
RELEVANTE = {q: CONSULTAS[q][1] for q in QUERY_IDS}

# ---------------------------------------------------------------------------
# 2) Misma matriz TF-IDF que la Parte 1: vectorizador ajustado SOLO con los
#    10 documentos; las consultas solo se transforman (protocolo de T1).
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, como en T1
X_docs = vectorizer.fit_transform(DOC_TEXTS)
X_queries = vectorizer.transform(QUERY_TEXTS)
sim_tfidf = cosine_similarity(X_queries, X_docs)  # 6 x 10


def detalle_por_consulta(sim_matrix, metodo):
    """Ranking descendente por consulta (empates en orden original d01..d10)."""
    filas = []
    for i, qid in enumerate(QUERY_IDS):
        orden = np.argsort(-np.asarray(sim_matrix[i], dtype=float), kind="stable")
        ranking = [DOC_IDS[j] for j in orden]
        rel = RELEVANTE[qid]
        pos = ranking.index(rel) + 1
        filas.append(
            {
                "consulta_id": qid,
                "consulta": CONSULTAS[qid][0],
                "relevante": rel,
                "metodo": metodo,
                "ranking": ranking,
                "similitudes": {DOC_IDS[j]: float(sim_matrix[i, j]) for j in range(len(DOC_IDS))},
                "similitudes_en_orden_ranking": [float(sim_matrix[i, j]) for j in orden],
                "posicion_del_relevante": pos,
                "reciprocal_rank": 1.0 / pos,
                "hit@1": int(pos == 1),
                "hit@3": int(pos <= 3),
            }
        )
    return filas


def metricas_desde_detalle(filas):
    return {
        "Hit@1": float(np.mean([f["hit@1"] for f in filas])),
        "Hit@3": float(np.mean([f["hit@3"] for f in filas])),
        "MRR": float(np.mean([f["reciprocal_rank"] for f in filas])),
    }


detalle_tfidf = detalle_por_consulta(sim_tfidf, "TF-IDF")
metricas_tfidf = metricas_desde_detalle(detalle_tfidf)

# ---------------------------------------------------------------------------
# 3) LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF
#    documental; las consultas se proyectan al espacio latente con las mismas
#    componentes (folding-in). Nada se ajusta con las consultas.
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
docs_latente = svd.fit_transform(X_docs)     # 10 x 4
queries_latente = svd.transform(X_queries)   # 6 x 4
sim_lsa = cosine_similarity(queries_latente, docs_latente)  # 6 x 10

detalle_lsa = detalle_por_consulta(sim_lsa, "LSA(4)")
metricas_lsa = metricas_desde_detalle(detalle_lsa)

# ---------------------------------------------------------------------------
# 4) Verificación de consistencia con T1 (rankings TF-IDF de la Parte 1)
# ---------------------------------------------------------------------------
t1_path = Path("entrada") / "T1" / "resultados.json"
consistencia = {"archivo_T1": str(t1_path), "disponible": t1_path.exists()}
if consistencia["disponible"]:
    try:
        with open(t1_path, encoding="utf-8") as fh:
            t1 = json.load(fh)
        t1_por_consulta = {f["consulta_id"]: f for f in t1.get("por_consulta", [])}
        comparacion = {}
        for f in detalle_tfidf:
            ref = t1_por_consulta.get(f["consulta_id"])
            if ref:
                comparacion[f["consulta_id"]] = {
                    "ranking_igual": ref.get("ranking") == f["ranking"],
                    "posicion_igual": ref.get("posicion_del_relevante") == f["posicion_del_relevante"],
                }
        consistencia["por_consulta"] = comparacion
        if comparacion:
            consistencia["rankings_tfidf_coinciden_con_T1"] = bool(
                all(c["ranking_igual"] for c in comparacion.values())
            )
        if "vocabulario_tamano" in t1:
            consistencia["vocabulario_tamano_T1"] = t1["vocabulario_tamano"]
            consistencia["vocabulario_coincide"] = bool(
                int(t1["vocabulario_tamano"]) == len(vectorizer.vocabulary_)
            )
    except Exception as exc:
        consistencia["error_lectura"] = repr(exc)

# ---------------------------------------------------------------------------
# 5) ÚNICA tabla comparativa: TF-IDF vs LSA(4) con Hit@1, Hit@3 y MRR
# ---------------------------------------------------------------------------
tabla = pd.DataFrame(
    [
        {"metodo": "TF-IDF", **metricas_tfidf},
        {"metodo": "LSA(4)", **metricas_lsa},
    ]
).set_index("metodo")
tabla.to_csv("T2_tabla_comparativa.csv", encoding="utf-8")

posiciones = [
    {
        "consulta_id": qid,
        "relevante": RELEVANTE[qid],
        "pos_TF-IDF": detalle_tfidf[i]["posicion_del_relevante"],
        "pos_LSA(4)": detalle_lsa[i]["posicion_del_relevante"],
        "RR_TF-IDF": detalle_tfidf[i]["reciprocal_rank"],
        "RR_LSA(4)": detalle_lsa[i]["reciprocal_rank"],
    }
    for i, qid in enumerate(QUERY_IDS)
]
pos_df = pd.DataFrame(posiciones)
pos_df.to_csv("T2_posiciones_por_consulta.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------------------
# 6) Figuras (PNG, dpi=120)
# ---------------------------------------------------------------------------
# 6a) La tabla comparativa como imagen
fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.axis("off")
celdas = [
    ["TF-IDF", f"{metricas_tfidf['Hit@1']:.4f}", f"{metricas_tfidf['Hit@3']:.4f}", f"{metricas_tfidf['MRR']:.4f}"],
    ["LSA(4)", f"{metricas_lsa['Hit@1']:.4f}", f"{metricas_lsa['Hit@3']:.4f}", f"{metricas_lsa['MRR']:.4f}"],
]
tbl = ax.table(
    cellText=celdas,
    colLabels=["metodo", "Hit@1", "Hit@3", "MRR"],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.auto_set_column_width(col=list(range(4)))
tbl.set_fontsize(12)
tbl.scale(1, 1.7)
for (r, c), celda in tbl.get_celld().items():
    if r == 0:
        celda.set_facecolor("#D9E2F3")
        celda.set_text_props(weight="bold")
    elif c == 0:
        celda.set_text_props(weight="bold")
ax.set_title(
    "Tarea C · Parte 2 — Tabla comparativa (6 consultas, 10 documentos)\n"
    "Hit@1, Hit@3 y MRR: TF-IDF vs LSA(4)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig("T2_tabla_comparativa.png", dpi=120)
plt.close(fig)

# 6b) Barras agrupadas de las tres métricas
fig, ax = plt.subplots(figsize=(7.0, 3.8))
x = np.arange(len(tabla.columns))
ancho = 0.35
barras_tfidf = ax.bar(x - ancho / 2, tabla.loc["TF-IDF"].values.astype(float), ancho,
                      label="TF-IDF", color="#4C72B0")
barras_lsa = ax.bar(x + ancho / 2, tabla.loc["LSA(4)"].values.astype(float), ancho,
                    label="LSA(4)", color="#DD8452")


def etiquetar(barras):
    try:
        ax.bar_label(barras, fmt="%.3f", fontsize=9, padding=2)
    except Exception:
        for rect in barras:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.3f}",
                    ha="center", fontsize=9)


etiquetar(barras_tfidf)
etiquetar(barras_lsa)
ax.set_xticks(x)
ax.set_xticklabels(list(tabla.columns))
ax.set_ylim(0, 1.15)
ax.set_ylabel("valor de la métrica")
ax.set_title("Recuperación sobre 6 consultas: TF-IDF vs LSA(4)")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("T2_barras_metricas.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "metodo": "LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF de la Parte 1 + similitud coseno en el espacio latente",
    "metodo_linea_base": "TF-IDF (TfidfVectorizer, parámetros por defecto) + similitud coseno, idéntico a T1",
    "protocolo": "TF-IDF ajustado solo con los 10 documentos; SVD ajustado solo con la matriz TF-IDF documental; consultas proyectadas al espacio latente con las mismas componentes (folding-in); empates resueltos por el orden original de documentos",
    "n_documentos": len(DOC_IDS),
    "n_consultas": len(QUERY_IDS),
    "n_componentes_lsa": 4,
    "random_state_svd": 0,
    "vocabulario_tamano": len(vectorizer.vocabulary_),
    "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
    "varianza_explicada_acumulada": float(np.sum(svd.explained_variance_ratio_)),
    "tabla_comparativa": {
        "descripcion": "Única tabla comparativa: Hit@1, Hit@3 y MRR de TF-IDF y LSA(4) sobre las 6 consultas",
        "columnas": ["metodo", "Hit@1", "Hit@3", "MRR"],
        "filas": [
            {"metodo": "TF-IDF", "Hit@1": metricas_tfidf["Hit@1"],
             "Hit@3": metricas_tfidf["Hit@3"], "MRR": metricas_tfidf["MRR"]},
            {"metodo": "LSA(4)", "Hit@1": metricas_lsa["Hit@1"],
             "Hit@3": metricas_lsa["Hit@3"], "MRR": metricas_lsa["MRR"]},
        ],
    },
    "metricas": {"TF-IDF": metricas_tfidf, "LSA(4)": metricas_lsa},
    "lsa_por_consulta": detalle_lsa,
    "tfidf_por_consulta": detalle_tfidf,
    "posiciones_por_consulta": posiciones,
    "consistencia_con_T1": consistencia,
    "archivos_generados": [
        "resultados.json",
        "T2_tabla_comparativa.csv",
        "T2_tabla_comparativa.png",
        "T2_barras_metricas.png",
        "T2_posiciones_por_consulta.csv",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen breve
# ---------------------------------------------------------------------------
print("=" * 64)
print("T2 — Semántica latente (LSA) vs línea base TF-IDF")
print("=" * 64)
print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
print(
    f"\nVarianza explicada por las 4 componentes LSA: "
    f"{resultados['varianza_explicada_acumulada']:.4f} "
    f"(por componente: {[round(v, 4) for v in resultados['varianza_explicada_por_componente']]})"
)
if "rankings_tfidf_coinciden_con_T1" in consistencia:
    print(f"Rankings TF-IDF idénticos a los de T1: {consistencia['rankings_tfidf_coinciden_con_T1']}")
print("\nPosición del documento relevante por consulta (1 = primer lugar):")
print(pos_df.to_string(index=False))
print("\nRankings LSA(4) por consulta (top 3):")
for f in detalle_lsa:
    print(f"  {f['consulta_id']} (rel={f['relevante']}): "
          f"{' > '.join(f['ranking'][:3])} ... pos_relevante={f['posicion_del_relevante']}")
print("\nArchivos generados:")
for nombre in resultados["archivos_generados"]:
    print(f"  - {nombre}")
