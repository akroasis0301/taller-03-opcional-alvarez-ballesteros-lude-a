```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Parte 3: análisis por consulta.

· Reproduce los rankings de ambos métodos con la misma pipeline de T1/T2:
    - TF-IDF: TfidfVectorizer por defecto (fit en los 10 documentos) + coseno.
    - LSA:    TruncatedSVD(n_components=4, random_state=0) + coseno.
· Detecta los fallos de cada método: consulta cuyo documento relevante queda
  fuera del top 3 (rank_relevante > 3).
· Extrae, como evidencia, los términos que comparte cada consulta con su
  documento relevante y con los documentos que cada método coloca por encima,
  usando la misma tokenización del vectorizador (minúsculas, tokens de 2+
  caracteres, sin stopwords ni stemming).

Salidas en la carpeta actual:
  resultados.json
  T3_solape_consulta_documento.png
  T3_fallo_q5.png
  T3_tabla_fallos.csv
  T3_tabla_consultas_metodos.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ----------------------------------------------------------------------------
# 1) Corpus, consultas y juicios de relevancia (Parte 1 del enunciado)
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
    "q1": "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "q2": "¿Cómo se añaden fragmentos recuperados al prompt?",
    "q3": "¿Qué técnica entrena matrices de bajo rango?",
    "q4": "¿Cómo se compara la orientación de dos vectores de texto?",
    "q5": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "q6": "¿Qué arquitectura combina autoatención con capas feed-forward?",
}
RELEVANTE = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}

doc_ids = [f"d{i:02d}" for i in range(1, 11)]
query_ids = [f"q{i}" for i in range(1, 7)]
METODOS = ("TF-IDF", "LSA")

# ----------------------------------------------------------------------------
# 2) Misma pipeline que T1/T2 y rankings de ambos métodos
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, idéntico a T1/T2
X_docs = vectorizer.fit_transform([DOCS[d] for d in doc_ids])
X_queries = vectorizer.transform([QUERIES[q] for q in query_ids])

svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)

S_tfidf = cosine_similarity(X_queries, X_docs)
S_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)


def rankings_from_matrix(S):
    """Ranking de los 10 documentos por consulta (empates en orden d01..d10)."""
    out = {}
    for i, qid in enumerate(query_ids):
        order = np.argsort(-S[i], kind="stable")
        out[qid] = [{"pos": j + 1, "doc_id": doc_ids[k], "score": float(S[i, k])}
                    for j, k in enumerate(order)]
    return out


RANKINGS = {"TF-IDF": rankings_from_matrix(S_tfidf),
            "LSA": rankings_from_matrix(S_lsa)}

# ----------------------------------------------------------------------------
# 3) Términos compartidos consulta–documento (tokenización del vectorizador)
# ----------------------------------------------------------------------------
analyzer = vectorizer.build_analyzer()
vocabulario = set(vectorizer.get_feature_names_out())
idf = vectorizer.idf_
col_of = vectorizer.vocabulary_

tokens_doc = {d: set(analyzer(DOCS[d])) for d in doc_ids}
tokens_query_set = {q: set(analyzer(QUERIES[q])) for q in query_ids}
tokens_query_lista = {q: analyzer(QUERIES[q]) for q in query_ids}


def terminos_compartidos(qid, did):
    """Términos compartidos con idf y pesos TF-IDF en consulta y documento."""
    qi, di = query_ids.index(qid), doc_ids.index(did)
    return [{"termino": t,
             "idf": float(idf[col_of[t]]),
             "peso_en_consulta": float(X_queries[qi, col_of[t]]),
             "peso_en_documento": float(X_docs[di, col_of[t]])}
            for t in sorted(tokens_query_set[qid] & tokens_doc[did])]


def analizar(qid, metodo):
    rel = RELEVANTE[qid]
    rl = RANKINGS[metodo][qid]
    rank_rel = next(it["pos"] for it in rl if it["doc_id"] == rel)
    comp_rel = terminos_compartidos(qid, rel)
    docs_sobre = []
    for it in rl:
        if it["pos"] < rank_rel:
            det = terminos_compartidos(qid, it["doc_id"])
            docs_sobre.append({
                "pos": it["pos"],
                "doc_id": it["doc_id"],
                "score": it["score"],
                "terminos_compartidos_con_consulta": [x["termino"] for x in det],
                "detalle_terminos": det,
                "suma_idf_terminos_compartidos": float(sum(x["idf"] for x in det)),
            })
    return {
        "metodo": metodo,
        "consulta_id": qid,
        "consulta": QUERIES[qid],
        "relevante": rel,
        "rank_relevante": int(rank_rel),
        "fallo_top3": bool(rank_rel > 3),
        "hit_at_1": int(rank_rel == 1),
        "hit_at_3": int(rank_rel <= 3),
        "reciprocal_rank": 1.0 / rank_rel,
        "top3": [it["doc_id"] for it in rl if it["pos"] <= 3],
        "docs_sobre_relevante": docs_sobre,
        "terminos_compartidos_con_relevante": [x["termino"] for x in comp_rel],
        "detalle_terminos_compartidos_con_relevante": comp_rel,
        "n_terminos_compartidos_con_relevante": len(comp_rel),
        "suma_idf_terminos_compartidos_con_relevante": float(sum(x["idf"] for x in comp_rel)),
    }


analisis = {qid: {m: analizar(qid, m) for m in METODOS} for qid in query_ids}

# ----------------------------------------------------------------------------
# 4) Tabla de fallos por método y por consulta + evidencia de términos
# ----------------------------------------------------------------------------
def fmt(ts):
    return ", ".join(f"'{t}'" for t in ts) if ts else "ninguno"


filas_fallos = []
for qid in query_ids:
    for m in METODOS:
        a = analisis[qid][m]
        if not a["fallo_top3"]:
            continue
        oov = sorted(t for t in tokens_query_set[qid] if t not in vocabulario)
        encima = "; ".join(
            f"{d['doc_id']} (pos {d['pos']}) → {fmt(d['terminos_compartidos_con_consulta'])}"
            for d in a["docs_sobre_relevante"])
        diagnostico = (
            f"{m}: el relevante {a['relevante']} queda en el puesto {a['rank_relevante']} "
            f"(fuera del top 3). La consulta solo comparte con él "
            f"{fmt(a['terminos_compartidos_con_relevante'])} "
            f"(suma IDF = {a['suma_idf_terminos_compartidos_con_relevante']:.3f}); además "
            f"{fmt(oov)} —el contenido léxico de la consulta— no aparece en ningún documento "
            f"(fuera del vocabulario), de modo que el ranking se decide con palabras "
            f"funcionales. Documentos colocados arriba y su solape con la consulta: {encima}."
        )
        filas_fallos.append({
            "metodo": m,
            "consulta_id": qid,
            "consulta": QUERIES[qid],
            "relevante": a["relevante"],
            "rank_relevante": a["rank_relevante"],
            "top3": a["top3"],
            "docs_sobre_relevante": [d["doc_id"] for d in a["docs_sobre_relevante"]],
            "terminos_compartidos_con_relevante": a["terminos_compartidos_con_relevante"],
            "n_terminos_compartidos_con_relevante": a["n_terminos_compartidos_con_relevante"],
            "suma_idf_terminos_compartidos_con_relevante": a["suma_idf_terminos_compartidos_con_relevante"],
            "terminos_consulta_fuera_de_vocabulario": oov,
            "terminos_compartidos_con_docs_sobre_relevante": {
                d["doc_id"]: d["terminos_compartidos_con_consulta"] for d in a["docs_sobre_relevante"]},
            "diagnostico": diagnostico,
        })

fallidos = {(a["metodo"], qid) for qid in query_ids for a in analisis[qid].values() if a["fallo_top3"]}
cubiertas = {(f["metodo"], f["consulta_id"]) for f in filas_fallos}
cobertura_ok = (fallidos == cubiertas) and {("TF-IDF", "q5"), ("LSA", "q5")} <= fallidos

fallos_por_metodo = {}
for m in METODOS:
    fallidas = [qid for qid in query_ids if analisis[qid][m]["fallo_top3"]]
    fallos_por_metodo[m] = {
        "consultas_fallidas": fallidas,
        "n_fallos": len(fallidas),
        "hit_at_3_recalculado": float(np.mean([analisis[qid][m]["hit_at_3"] for qid in query_ids])),
        "mrr_recalculado": float(np.mean([analisis[qid][m]["reciprocal_rank"] for qid in query_ids])),
    }

casos_limite = [
    {"metodo": a["metodo"], "consulta_id": qid, "rank_relevante": a["rank_relevante"],
     "nota": "el relevante queda en el puesto 3: no es fallo según la definición "
             "(fuera del top 3), aunque el método pierde Hit@1"}
    for qid in query_ids for a in analisis[qid].values()
    if a["hit_at_3"] == 1 and a["hit_at_1"] == 0]

# Términos compartidos de CADA consulta con su relevante (y docs de arriba por método)
terminos_por_consulta = {}
for qid in query_ids:
    rel = RELEVANTE[qid]
    comp_rel = analisis[qid]["TF-IDF"]["detalle_terminos_compartidos_con_relevante"]
    terminos_por_consulta[qid] = {
        "consulta": QUERIES[qid],
        "relevante": rel,
        "tokens_consulta": tokens_query_lista[qid],
        "tokens_consulta_unicos": sorted(tokens_query_set[qid]),
        "tokens_consulta_en_vocabulario": sorted(t for t in tokens_query_set[qid] if t in vocab
