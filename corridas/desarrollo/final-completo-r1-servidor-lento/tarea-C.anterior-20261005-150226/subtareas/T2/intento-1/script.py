#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Parte 2: Semántica latente (LSA) sobre la misma matriz TF-IDF de la Parte 1.

Pasos:
1) Reconstruir la matriz TF-IDF de la Parte 1 (TfidfVectorizer por defecto, ajustado
   solo con los 10 documentos; las 6 consultas se transforman con el mismo vectorizador).
2) Proyectar a 4 dimensiones con TruncatedSVD(n_components=4, random_state=0) (LSA).
3) Similitud coseno en el espacio latente y ranking de los 10 documentos por consulta
   (mismo criterio de desempate que en T1: orden del corpus d01..d10).
4) Repetir la evaluación (Hit@1, Hit@3, MRR sobre las 6 consultas) para ambos métodos
   y construir UNA sola tabla comparativa (2 métodos × 3 métricas).

Salidas: resultados.json, tabla_comparativa_tfidf_lsa.csv, comparativa_tfidf_vs_lsa.png
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
# 1) Datos del enunciado (idénticos a la Parte 1 / T1)
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

QUERY_IDS = ["q1", "q2", "q3", "q4", "q5", "q6"]
QUERIES = [
    "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "¿Cómo se añaden fragmentos recuperados al prompt?",
    "¿Qué técnica entrena matrices de bajo rango?",
    "¿Cómo se compara la orientación de dos vectores de texto?",
    "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "¿Qué arquitectura combina autoatención con capas feed-forward?",
]
RELEVANTE = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}


# ---------------------------------------------------------------------------
# 2) Utilidades (mismos criterios que la Parte 1)
# ---------------------------------------------------------------------------
def ranking_descendente_estable(scores):
    """Orden descendente; empates resueltos por orden del corpus (d01..d10), como en T1."""
    orden = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return [DOC_IDS[i] for i in orden]


def evaluar_matriz_similitud(S):
    """Rankings por consulta + Hit@1, Hit@3 y MRR (promediados sobre las 6 consultas)."""
    por_consulta = []
    for i, qid in enumerate(QUERY_IDS):
        sims = np.asarray(S[i], dtype=float)
        ranking = ranking_descendente_estable(sims)
        rel = RELEVANTE[qid]
        rank = ranking.index(rel) + 1
        por_consulta.append(
            {
                "consulta": qid,
                "texto_consulta": QUERIES[i],
                "relevante": rel,
                "ranking_completo": ranking,
                "rank_del_relevante": int(rank),
                "reciprocal_rank": float(1.0 / rank),
                "hit@1": int(rank == 1),
                "hit@3": int(rank <= 3),
                "sim_coseno_con_relevante": float(sims[DOC_IDS.index(rel)]),
                "sim_coseno_top1": float(sims.max()),
            }
        )
    metricas = {
        "hit@1": float(np.mean([c["hit@1"] for c in por_consulta])),
        "hit@3": float(np.mean([c["hit@3"] for c in por_consulta])),
        "mrr": float(np.mean([c["reciprocal_rank"] for c in por_consulta])),
    }
    return por_consulta, metricas


def matriz_a_json(S):
    return {
        "filas_consultas": list(QUERY_IDS),
        "columnas_documentos": list(DOC_IDS),
        "valores": [[float(v) for v in fila] for fila in np.asarray(S)],
    }


# ---------------------------------------------------------------------------
# 3) Pipeline principal
# ---------------------------------------------------------------------------
def main():
    # --- 3.1 Misma matriz TF-IDF que la Parte 1 (parámetros por defecto) ---
    vectorizer = TfidfVectorizer()  # parámetros por defecto, igual que en T1
    X_docs = vectorizer.fit_transform(DOCS)
    X_queries = vectorizer.transform(QUERIES)
    S_tfidf = cosine_similarity(X_queries, X_docs)
    por_consulta_tfidf, metricas_tfidf = evaluar_matriz_similitud(S_tfidf)

    # --- 3.2 LSA: TruncatedSVD a 4 dimensiones (semilla 0), ajustado SOLO con los documentos ---
    svd = TruncatedSVD(n_components=4, random_state=0)
    X_docs_lsa = svd.fit_transform(X_docs)      # 10 × 4
    X_queries_lsa = svd.transform(X_queries)    # 6 × 4 (mismas componentes latentes)
    S_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)
    por_consulta_lsa, metricas_lsa = evaluar_matriz_similitud(S_lsa)

    # --- 3.3 Verificación suave contra el resultado de la Parte 1 (si está disponible) ---
    ruta_t1 = Path("entrada") / "T1" / "resultados.json"
    if ruta_t1.exists():
        try:
            t1 = json.loads(ruta_t1.read_text(encoding="utf-8"))
            S_t1 = np.asarray(t1["matriz_similitud_coseno"]["valores"], dtype=float)
            matriz_ok = bool(S_t1.shape == S_tfidf.shape and np.allclose(S_t1, S_tfidf, rtol=1e-9, atol=1e-12))
            ranks_t1 = {c["consulta"]: int(c["rank_del_relevante"]) for c in t1.get("por_consulta", [])}
            ranks_t2 = {c["consulta"]: int(c["rank_del_relevante"]) for c in por_consulta_tfidf}
            ranks_ok = bool(ranks_t1 == ranks_t2)
            vocab_ok = bool(int(t1.get("tam_vocabulario", -1)) == len(vectorizer.vocabulary_))
            verificacion = {
                "fuente": "entrada/T1/resultados.json",
                "tam_vocabulario_coincide": vocab_ok,
                "matriz_similitud_tfidf_coincide": matriz_ok,
                "ranks_del_relevante_tfidf_coinciden": ranks_ok,
                "misma_linea_base": bool(matriz_ok and ranks_ok and vocab_ok),
            }
        except Exception as exc:
            verificacion = {
                "fuente": "entrada/T1/resultados.json",
                "error": f"No se pudo verificar contra T1 ({type(exc).__name__}: {exc}); se usa la línea base TF-IDF recomputada con los mismos parámetros.",
            }
    else:
        verificacion = {
            "fuente": "entrada/T1/resultados.json",
            "nota": "Archivo de T1 no encontrado; la línea base TF-IDF se recomputó con los mismos parámetros de la Parte 1.",
        }

    # --- 3.4 Tabla comparativa única (2 métodos × 3 métricas) ---
    tabla = pd.DataFrame(
        [
            {"metodo": "TF-IDF (línea base léxica)", **metricas_tfidf},
            {"metodo": "LSA (TruncatedSVD, 4 dimensiones, random_state=0)", **metricas_lsa},
        ],
        columns=["metodo", "hit@1", "hit@3", "mrr"],
    )
    tabla.to_csv("tabla_comparativa_tfidf_lsa.csv", index=False, encoding="utf-8")
    diferencias = {m: float(metricas_lsa[m] - metricas_tfidf[m]) for m in ("hit@1", "hit@3", "mrr")}

    # --- 3.5 Figura comparativa ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    nombres_metricas = ["Hit@1", "Hit@3", "MRR"]
    vals_tfidf = [metricas_tfidf["hit@1"], metricas_tfidf["hit@3"], metricas_tfidf["mrr"]]
    vals_lsa = [metricas_lsa["hit@1"], metricas_lsa["hit@3"], metricas_lsa["mrr"]]
    x = np.arange(3)
    ancho = 0.36
    b1 = ax1.bar(x - ancho / 2, vals_tfidf, ancho, label="TF-IDF", color="#4C72B0")
    b2 = ax1.bar(x + ancho / 2, vals_lsa, ancho, label="LSA (4D)", color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres_metricas)
    ax1.set_ylim(0.0, 1.15)
    ax1.set_ylabel("Valor (promedio sobre 6 consultas)")
    ax1.set_title("Métricas agregadas: TF-IDF vs LSA")
    ax1.legend(loc="upper right", fontsize=8)
    for barras in (b1, b2):
        for b in barras:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                     f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ranks_tfidf = [c["rank_del_relevante"] for c in por_consulta_tfidf]
    ranks_lsa = [c["rank_del_relevante"] for c in por_consulta_lsa]
    xq = np.arange(len(QUERY_IDS))
    ax2.bar(xq - ancho / 2, ranks_tfidf, ancho, label="TF-IDF", color="#4C72B0")
    ax2.bar(xq + ancho / 2, ranks_lsa, ancho, label="LSA (4D)", color="#DD8452")
    ax2.axhline(1, color="gray", lw=0.8, ls="--")
    ax2.axhline(3, color="gray", lw=0.8, ls=":")
    ax2.text(len(QUERY_IDS) - 0.45, 1.12, "Hit@1", fontsize=7, color="gray", va="bottom", ha="right")
    ax2.text(len(QUERY_IDS) - 0.45, 3.12, "Hit@3", fontsize=7, color="gray", va="bottom", ha="right")
    ax2.set_xticks(xq)
    ax2.set_xticklabels(QUERY_IDS)
    ax2.set_ylim(0, 10.5)
    ax2.set_ylabel("Rank del documento relevante (menor = mejor)")
    ax2.set_title("Rank del relevante por consulta")
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle("Tarea C · Parte 2 — Recuperación léxica (TF-IDF) vs semántica latente (LSA, 4D)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("comparativa_tfidf_vs_lsa.png", dpi=120)
    plt.close(fig)

    # --- 3.6 resultados.json (contrato de la subtarea) ---
    coordenadas_lsa = {
        "documentos": {DOC_IDS[i]: [float(v) for v in X_docs_lsa[i]] for i in range(len(DOC_IDS))},
        "consultas": {QUERY_IDS[i]: [float(v) for v in X_queries_lsa[i]] for i in range(len(QUERY_IDS))},
    }
    resultados = {
        "subtarea": "T2_semantica_latente_lsa",
        "descripcion": (
            "LSA: TruncatedSVD(n_components=4, random_state=0) sobre la misma matriz TF-IDF de la Parte 1; "
            "similitud coseno en el espacio latente de 4 dimensiones; evaluación repetida (Hit@1, Hit@3, MRR "
            "sobre las 6 consultas) y una única tabla comparativa con los dos métodos."
        ),
        "configuracion": {
            "n_documentos": len(DOCS),
            "n_consultas": len(QUERIES),
            "vectorizador": "TfidfVectorizer (parámetros por defecto), ajustado solo con los 10 documentos (idéntico a la Parte 1)",
            "tam_vocabulario": int(len(vectorizer.vocabulary_)),
            "lsa": {
                "modelo": "TruncatedSVD",
                "n_componentes": 4,
                "random_state": 0,
                "ajustado_sobre": "matriz TF-IDF de los 10 documentos; consultas proyectadas con el mismo modelo",
            },
            "similitud": "coseno (cosine_similarity de scikit-learn) en el espacio latente de 4 dimensiones",
            "desempate_ranking": "argsort estable descendente (empates resueltos por orden del corpus d01..d10), igual que en T1",
        },
        "verificacion_contra_T1": verificacion,
        "lsa": {
            "valores_singulares": [float(v) for v in svd.singular_values_],
            "varianza_explicada_ratio": [float(v) for v in svd.explained_variance_ratio_],
            "varianza_explicada_total_4_componentes": float(np.sum(svd.explained_variance_ratio_)),
            "coordenadas_latentes": coordenadas_lsa,
        },
        "matriz_similitud_coseno_tfidf": matriz_a_json(S_tfidf),
        "matriz_similitud_coseno_lsa": matriz_a_json(S_lsa),
        "por_consulta_tfidf": por_consulta_tfidf,
        "por_consulta_lsa": por_consulta_lsa,
        "metricas_tfidf": metricas_tfidf,
        "metricas_lsa": metricas_lsa,
        "diferencias_lsa_menos_tfidf": diferencias,
        "tabla_comparativa": {
            "columnas": ["metodo", "hit@1", "hit@3", "mrr"],
            "filas": tabla.to_dict(orient="records"),
            "nota": "Única tabla comparativa: 2 métodos (TF-IDF, LSA) × 3 métricas (Hit@1, Hit@3, MRR), promediadas sobre las 6 consultas.",
        },
        "tabla_comparativa_matriz": [["metodo", "hit@1", "hit@3", "mrr"]]
        + [[r["metodo"], r["hit@1"], r["hit@3"], r["mrr"]] for r in tabla.to_dict(orient="records")],
        "archivos_generados": [
            "resultados.json",
            "tabla_comparativa_tfidf_lsa.csv",
            "comparativa_tfidf_vs_lsa.png",
        ],
    }
    Path("resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 3.7 Resumen ---
    print("=" * 74)
    print("T2 · Parte 2 — LSA (TruncatedSVD, 4D, random_state=0) vs TF-IDF")
    print("=" * 74)
    print(f"Documentos: {len(DOCS)} | Consultas: {len(QUERIES)} | "
          f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos")
    if isinstance(verificacion, dict) and verificacion.get("misma_linea_base"):
        print("Verificación: la matriz TF-IDF recomputada coincide con entrada/T1/resultados.json.")
    print(f"Varianza explicada por las 4 componentes latentes: "
          f"{float(np.sum(svd.explained_variance_ratio_)):.4f}")

    print("\nTabla comparativa (2 métodos × 3 métricas, promedio sobre 6 consultas):")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nRank del relevante por consulta (TF-IDF → LSA):")
    for ct, cl in zip(por_consulta_tfidf, por_consulta_lsa):
        print(f"  {ct['consulta']}: {ct['rank_del_relevante']:>2} → {cl['rank_del_relevante']:>2}"
              f"   (relevante {ct['relevante']})")

    print("\nRankings LSA por consulta (top-3; entre corchetes el relevante):")
    for c in por_consulta_lsa:
        top3 = list(c["ranking_completo"][:3])
        marcado = [f"[{d}]" if d == c["relevante"] else d for d in top3]
        extra = "" if c["relevante"] in top3 else f" … (relevante {c['relevante']} en rank {c['rank_del_relevante']})"
        print(f"  {c['consulta']}: {' > '.join(marcado)}{extra}")

    print("\nArchivos generados:")
    for nombre in resultados["archivos_generados"]:
        print(f"  - {nombre}")


if __name__ == "__main__":
    main()
