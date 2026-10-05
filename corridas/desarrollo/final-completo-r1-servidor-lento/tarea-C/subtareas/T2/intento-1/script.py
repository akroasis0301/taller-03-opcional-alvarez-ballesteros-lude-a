# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la línea base léxica TF-IDF.

Reconstruye la misma matriz TF-IDF de la Parte 1 (TfidfVectorizer con parámetros
por defecto sobre los mismos 10 documentos del enunciado), proyecta documentos y
consultas a 4 dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA),
repite la evaluación con las mismas 6 consultas y los mismos juicios, y construye
una tabla única con Hit@1, Hit@3 y MRR de los dos métodos, más los rankings LSA.

Salidas (carpeta actual):
  - resultados.json                            (contrato de la subtarea)
  - tabla_metricas_tfidf_vs_lsa.png            (tabla única de métricas)
  - comparativa_metricas_tfidf_vs_lsa.png      (barras agrupadas)
  - posicion_relevante_por_consulta.png        (posición del relevante por consulta)
  - tabla_metricas.csv                         (misma tabla en CSV)
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
# Datos del enunciado (idénticos a T1): corpus, consultas y juicios
# ---------------------------------------------------------------------------
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
DOC_IDS = list(DOCS)
DOC_TEXTS = [DOCS[d] for d in DOC_IDS]

QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
QUERY_IDS = list(QUERIES)

N_COMPONENTS = 4
RANDOM_STATE = 0
METRICAS = ["Hit@1", "Hit@3", "MRR"]


# ---------------------------------------------------------------------------
# Utilidades de evaluación
# ---------------------------------------------------------------------------
def rankear(fila_scores):
    """Ordena los 10 documentos por similitud descendente (empates por id de documento)."""
    orden = np.argsort(-np.asarray(fila_scores, dtype=float), kind="stable")
    return [DOC_IDS[j] for j in orden]


def evaluar(S):
    """Rankings, posición del relevante y métricas Hit@1, Hit@3, MRR sobre las 6 consultas."""
    rankings, posiciones, detalle = {}, {}, {}
    hits1, hits3, rranks = [], [], []
    for i, qid in enumerate(QUERY_IDS):
        texto_q, relevante = QUERIES[qid]
        ranking = rankear(S[i])
        pos = ranking.index(relevante) + 1
        rankings[qid] = ranking
        posiciones[qid] = pos
        hits1.append(int(pos == 1))
        hits3.append(int(pos <= 3))
        rranks.append(1.0 / pos)
        detalle[qid] = {
            "consulta": texto_q,
            "documento_relevante": relevante,
            "ranking": ranking,
            "posicion_del_relevante": pos,
            "Hit@1": int(pos == 1),
            "Hit@3": int(pos <= 3),
            "reciprocal_rank": 1.0 / pos,
            "similitud_coseno": {DOC_IDS[j]: float(S[i, j]) for j in range(len(DOC_IDS))},
        }
    metricas = {
        "Hit@1": float(np.mean(hits1)),
        "Hit@3": float(np.mean(hits3)),
        "MRR": float(np.mean(rranks)),
    }
    return metricas, rankings, posiciones, detalle


def _buscar_metricas_agregadas(obj):
    """Busca recursivamente en resultados.json de T1 un dict con Hit@1, Hit@3 y MRR."""
    if isinstance(obj, dict):
        claves = {str(k).lower() for k in obj}
        if {"hit@1", "hit@3", "mrr"} <= claves:
            return {k: obj[k] for k in obj if str(k).lower() in {"hit@1", "hit@3", "mrr"}}
        for v in obj.values():
            r = _buscar_metricas_agregadas(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _buscar_metricas_agregadas(v)
            if r is not None:
                return r
    return None


def main():
    # -----------------------------------------------------------------------
    # 1) Misma matriz TF-IDF que T1: TfidfVectorizer por defecto, ajustado
    #    solo con los 10 documentos (determinista -> idéntica a la de T1).
    # -----------------------------------------------------------------------
    vectorizer = TfidfVectorizer()
    X_docs = vectorizer.fit_transform(DOC_TEXTS)
    X_queries = vectorizer.transform([QUERIES[q][0] for q in QUERY_IDS])

    # -----------------------------------------------------------------------
    # 2) Línea base TF-IDF + coseno (misma evaluación de la Parte 1)
    # -----------------------------------------------------------------------
    S_tfidf = cosine_similarity(X_queries, X_docs)
    met_tfidf, rank_tfidf, pos_tfidf, det_tfidf = evaluar(S_tfidf)

    # -----------------------------------------------------------------------
    # 3) LSA: proyección a 4 dimensiones con TruncatedSVD(n_components=4,
    #    random_state=0), ajustada sobre la matriz TF-IDF de los documentos.
    # -----------------------------------------------------------------------
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    X_docs_lsa = svd.fit_transform(X_docs)
    X_queries_lsa = svd.transform(X_queries)
    S_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)
    met_lsa, rank_lsa, pos_lsa, det_lsa = evaluar(S_lsa)

    # -----------------------------------------------------------------------
    # 4) Tabla única con las tres métricas de los dos métodos
    # -----------------------------------------------------------------------
    tabla = {
        "columnas": METRICAS,
        "filas": {
            "TF-IDF + coseno": {m: met_tfidf[m] for m in METRICAS},
            "LSA (TruncatedSVD, 4D) + coseno": {m: met_lsa[m] for m in METRICAS},
        },
    }
    deltas = {m: float(met_lsa[m] - met_tfidf[m]) for m in METRICAS}
    filas_tabla = list(tabla["filas"])

    # -----------------------------------------------------------------------
    # 5) Verificación opcional contra los resultados de T1
    # -----------------------------------------------------------------------
    verif = {
        "archivo": "entrada/T1/resultados.json",
        "disponible": Path("entrada/T1/resultados.json").exists(),
    }
    if verif["disponible"]:
        try:
            t1 = json.loads(Path("entrada/T1/resultados.json").read_text(encoding="utf-8"))
            r1 = t1.get("rankings_por_consulta")
            if isinstance(r1, dict):
                verif["rankings_tfidf_coinciden"] = {q: (r1.get(q) == rank_tfidf[q]) for q in QUERY_IDS}
                verif["todos_los_rankings_coinciden"] = all(verif["rankings_tfidf_coinciden"].values())
            p1 = t1.get("posicion_del_relevante_por_consulta")
            if isinstance(p1, dict):
                verif["posiciones_coinciden"] = all(
                    int(p1.get(q, -1)) == pos_tfidf[q] for q in QUERY_IDS
                )
            m1 = _buscar_metricas_agregadas(t1)
            if m1 is not None:
                verif["metricas_t1"] = {k: float(v) for k, v in m1.items()}
                verif["metricas_coinciden"] = all(
                    abs(float(m1[k]) - met_tfidf[k]) <= 1e-9 for k in METRICAS if k in m1
                )
        except Exception as exc:  # la verificación es opcional
            verif["error"] = f"{type(exc).__name__}: {exc}"

    # -----------------------------------------------------------------------
    # 6) Figuras
    # -----------------------------------------------------------------------
    # Figura 1: tabla única de métricas
    celdas = [[nombre] + [f"{tabla['filas'][nombre][m]:.4f}" for m in METRICAS] for nombre in filas_tabla]
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.axis("off")
    tb = ax.table(cellText=celdas, colLabels=["Método"] + METRICAS, cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 1.7)
    for j in range(len(METRICAS) + 1):
        tb[0, j].set_facecolor("#D9E2F3")
        tb[0, j].set_text_props(weight="bold")
    ax.set_title("Hit@1, Hit@3 y MRR — TF-IDF vs LSA (6 consultas, 10 documentos)", pad=14)
    plt.savefig("tabla_metricas_tfidf_vs_lsa.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Figura 2: barras agrupadas de las tres métricas
    v_tfidf = [met_tfidf[m] for m in METRICAS]
    v_lsa = [met_lsa[m] for m in METRICAS]
    x = np.arange(len(METRICAS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    b1 = ax.bar(x - w / 2, v_tfidf, w, label="TF-IDF + coseno", color="#4C72B0")
    b2 = ax.bar(x + w / 2, v_lsa, w, label="LSA (TruncatedSVD, 4D) + coseno", color="#DD8452")
    for barras in (b1, b2):
        for b in barras:
            ax.annotate(
                f"{b.get_height():.3f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(METRICAS)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Recuperación sobre 6 consultas: TF-IDF vs LSA (4 dimensiones)")
    ax.legend(loc="upper right")
    plt.savefig("comparativa_metricas_tfidf_vs_lsa.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Figura 3: posición del documento relevante por consulta (1 = primero)
    pt = [pos_tfidf[q] for q in QUERY_IDS]
    pl = [pos_lsa[q] for q in QUERY_IDS]
    xq = np.arange(len(QUERY_IDS))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(xq - w / 2, pt, w, label="TF-IDF + coseno", color="#4C72B0")
    ax.bar(xq + w / 2, pl, w, label="LSA (TruncatedSVD, 4D) + coseno", color="#DD8452")
    for xi, (a, b) in enumerate(zip(pt, pl)):
        ax.annotate(str(a), (xi - w / 2, a), ha="center", va="bottom", fontsize=9)
        ax.annotate(str(b), (xi + w / 2, b), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xq)
    ax.set_xticklabels(QUERY_IDS)
    ax.set_ylim(0, 10.8)
    ax.set_ylabel("Posición del documento relevante")
    ax.set_title("Posición del relevante por consulta (menor es mejor)")
    ax.legend()
    plt.savefig("posicion_relevante_por_consulta.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 7) Tabla en CSV (misma tabla única, formato plano)
    # -----------------------------------------------------------------------
    df_tabla = pd.DataFrame(tabla["filas"]).T[METRICAS]
    df_tabla.to_csv("tabla_metricas.csv", encoding="utf-8")

    # -----------------------------------------------------------------------
    # 8) resultados.json (contrato de la subtarea)
    # -----------------------------------------------------------------------
    resultados = {
        "subtarea": "T2_lsa_truncated_svd_4d",
        "descripcion": (
            "LSA: la matriz TF-IDF de la Parte 1 se proyecta a 4 dimensiones con "
            "TruncatedSVD(n_components=4, random_state=0); con las mismas 6 consultas "
            "y los mismos juicios se repite la evaluación (similitud coseno en el "
            "espacio latente) y se construye una tabla única con Hit@1, Hit@3 y MRR "
            "de los dos métodos."
        ),
        "n_documentos": len(DOC_IDS),
        "n_consultas": len(QUERY_IDS),
        "tam_vocabulario": int(len(vectorizer.vocabulary_)),
        "n_componentes_lsa": N_COMPONENTS,
        "random_state": RANDOM_STATE,
        "juicios_por_consulta": {q: QUERIES[q][1] for q in QUERY_IDS},
        "tabla_metricas": tabla,
        "metricas_tfidf": met_tfidf,
        "metricas_lsa": met_lsa,
        "diferencia_lsa_menos_tfidf": deltas,
        "rankings_tfidf_por_consulta": rank_tfidf,
        "rankings_lsa_por_consulta": rank_lsa,
        "posicion_del_relevante_tfidf_por_consulta": pos_tfidf,
        "posicion_del_relevante_lsa_por_consulta": pos_lsa,
        "detalle_por_consulta_tfidf": det_tfidf,
        "detalle_por_consulta_lsa": det_lsa,
        "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_],
        "varianza_explicada_ratio_por_componente": [float(v) for v in svd.explained_variance_ratio_],
        "varianza_explicada_acumulada": float(np.sum(svd.explained_variance_ratio_)),
        "verificacion_contra_T1": verif,
        "figuras": [
            "tabla_metricas_tfidf_vs_lsa.png",
            "comparativa_metricas_tfidf_vs_lsa.png",
            "posicion_relevante_por_consulta.png",
        ],
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # 9) Resumen
    # -----------------------------------------------------------------------
    print("=" * 66)
    print("T2 — LSA (TruncatedSVD, 4 dimensiones) vs línea base TF-IDF")
    print("=" * 66)
    print(
        f"Documentos: {len(DOC_IDS)} | Consultas: {len(QUERY_IDS)} | "
        f"Vocabulario: {len(vectorizer.vocabulary_)} | "
        f"Varianza explicada (4 comp.): {np.sum(svd.explained_variance_ratio_):.4f}"
    )
    print("")
    print(f"{'Método':<34}{'Hit@1':>9}{'Hit@3':>9}{'MRR':>9}")
    for nombre in filas_tabla:
        vals = tabla["filas"][nombre]
        print(f"{nombre:<34}{vals['Hit@1']:>9.4f}{vals['Hit@3']:>9.4f}{vals['MRR']:>9.4f}")
    print("")
    print("Rankings LSA por consulta (relevante entre corchetes):")
    for q in QUERY_IDS:
        rel = QUERIES[q][1]
        marcado = [f"[{d}]" if d == rel else d for d in rank_lsa[q]]
        print(f"  {q}: {' > '.join(marcado)}  (pos. {pos_lsa[q]})")
    print("")
    if verif["disponible"]:
        print("Verificación contra T1:", json.dumps(verif, ensure_ascii=False))
    else:
        print("Verificación contra T1: entrada/T1/resultados.json no disponible (omitida)")
    print("")
    print(
        "Archivos generados: resultados.json, tabla_metricas.csv, "
        "tabla_metricas_tfidf_vs_lsa.png, comparativa_metricas_tfidf_vs_lsa.png, "
        "posicion_relevante_por_consulta.png"
    )


if __name__ == "__main__":
    main()
