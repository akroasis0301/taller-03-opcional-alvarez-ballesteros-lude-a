#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Parte 3: análisis por consulta (fallos por método y términos compartidos).

Reproduce con la misma pipeline de T1/T2 los rankings de los dos métodos
(TF-IDF + coseno y LSA TruncatedSVD k=4, random_state=0), detecta en qué
consultas falla cada método (documento relevante fuera del top 3) y extrae,
como evidencia, los términos que comparte cada consulta con su documento
relevante y con los documentos que los métodos colocan por encima.

Salidas (carpeta actual):
  resultados.json
  T3_tabla_fallos.csv
  T3_tabla_consultas_metodos.csv
  T3_solape_consulta_documento.png
  T3_fallo_q5.png
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios de relevancia (Parte 1 del enunciado)
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
METODOS = ["TF-IDF", "LSA"]
K_LATENTE = 4
RANDOM_STATE = 0

# ---------------------------------------------------------------------------
# 2) Misma pipeline que T1/T2: TF-IDF por defecto + LSA(k=4) y rankings coseno
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, idéntico a T1/T2
X_docs = vectorizer.fit_transform([DOCS[d] for d in doc_ids])
X_queries = vectorizer.transform([QUERIES[q] for q in query_ids])

svd = TruncatedSVD(n_components=K_LATENTE, random_state=RANDOM_STATE)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)

S = {
    "TF-IDF": cosine_similarity(X_queries, X_docs),
    "LSA": cosine_similarity(X_queries_lsa, X_docs_lsa),
}


def rankings_de(matriz_sim):
    """Ranking de los 10 documentos por consulta; empates en orden d01..d10."""
    out = {}
    for i, qid in enumerate(query_ids):
        orden = np.argsort(-matriz_sim[i], kind="stable")
        out[qid] = [
            {"pos": j + 1, "doc_id": doc_ids[k], "score": float(matriz_sim[i, k])}
            for j, k in enumerate(orden)
        ]
    return out


RANKINGS = {m: rankings_de(S[m]) for m in METODOS}

# ---------------------------------------------------------------------------
# 3) Términos compartidos consulta-documento (misma tokenización del vectorizador)
# ---------------------------------------------------------------------------
analyzer = vectorizer.build_analyzer()
vocabulario = set(vectorizer.get_feature_names_out())
idf = vectorizer.idf_
col_of = vectorizer.vocabulary_

tokens_doc = {d: set(analyzer(DOCS[d])) for d in doc_ids}
tokens_q_set = {q: set(analyzer(QUERIES[q])) for q in query_ids}
tokens_q_lista = {q: analyzer(QUERIES[q]) for q in query_ids}


def terminos_compartidos(qid, did):
    """Términos compartidos con idf y pesos TF-IDF en la consulta y el documento."""
    qi = query_ids.index(qid)
    di = doc_ids.index(did)
    comunes = tokens_q_set[qid] & tokens_doc[did]
    return [
        {
            "termino": t,
            "idf": float(idf[col_of[t]]),
            "peso_en_consulta": float(X_queries[qi, col_of[t]]),
            "peso_en_documento": float(X_docs[di, col_of[t]]),
        }
        for t in sorted(comunes, key=lambda t: (-idf[col_of[t]], t))
    ]


def analizar(qid, metodo):
    """Análisis completo de una consulta bajo un método."""
    rel = RELEVANTE[qid]
    rl = RANKINGS[metodo][qid]
    rank_rel = next(it["pos"] for it in rl if it["doc_id"] == rel)
    score_rel = next(it["score"] for it in rl if it["doc_id"] == rel)
    det_rel = terminos_compartidos(qid, rel)

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

    top3_det = []
    for it in rl:
        if it["pos"] <= 3:
            det = terminos_compartidos(qid, it["doc_id"])
            top3_det.append({
                "pos": it["pos"],
                "doc_id": it["doc_id"],
                "terminos_compartidos_con_consulta": [x["termino"] for x in det],
            })

    return {
        "metodo": metodo,
        "consulta_id": qid,
        "consulta": QUERIES[qid],
        "relevante": rel,
        "rank_relevante": int(rank_rel),
        "score_relevante": float(score_rel),
        "fallo_top3": bool(rank_rel > 3),
        "hit_at_1": int(rank_rel == 1),
        "hit_at_3": int(rank_rel <= 3),
        "reciprocal_rank": 1.0 / rank_rel,
        "top3": [it["doc_id"] for it in rl if it["pos"] <= 3],
        "top3_con_terminos": top3_det,
        "docs_sobre_relevante": docs_sobre,
        "terminos_compartidos_con_relevante": [x["termino"] for x in det_rel],
        "detalle_terminos_compartidos_con_relevante": det_rel,
        "n_terminos_compartidos_con_relevante": len(det_rel),
        "suma_idf_terminos_compartidos_con_relevante": float(sum(x["idf"] for x in det_rel)),
    }


ANALISIS = {qid: {m: analizar(qid, m) for m in METODOS} for qid in query_ids}

# ---------------------------------------------------------------------------
# 4) Tabla de fallos por método y por consulta + evidencia de términos
# ---------------------------------------------------------------------------
def fmt_terminos(ts):
    return ", ".join("'" + t + "'" for t in ts) if ts else "ninguno"


def dict_solape_a_texto(d):
    partes = []
    for k in sorted(d.keys()):
        v = d[k]
        partes.append(k + ": " + (", ".join(v) if v else "(sin solape)"))
    return "; ".join(partes)


filas_fallos = []
for qid in query_ids:
    for m in METODOS:
        a = ANALISIS[qid][m]
        if not a["fallo_top3"]:
            continue
        oov = sorted(t for t in tokens_q_set[qid] if t not in vocabulario)
        encima = "; ".join(
            d["doc_id"] + " (pos " + str(d["pos"]) + ") -> "
            + fmt_terminos(d["terminos_compartidos_con_consulta"])
            for d in a["docs_sobre_relevante"]
        )
        extra = (
            " La proyección latente (k=4) tampoco puentea la distancia: el relevante "
            "sigue en el puesto " + str(a["rank_relevante"]) + "."
        ) if m == "LSA" else ""
        diagnostico = (
            f"FALLO de {m} en {qid}: el relevante {a['relevante']} queda en el puesto "
            f"{a['rank_relevante']} (fuera del top 3). Con él, la consulta solo comparte "
            f"{fmt_terminos(a['terminos_compartidos_con_relevante'])} (suma IDF = "
            f"{a['suma_idf_terminos_compartidos_con_relevante']:.3f}); el resto del contenido "
            f"léxico de la consulta ({fmt_terminos(oov)}) no aparece en ningún documento "
            f"(fuera del vocabulario), de modo que el ranking lo deciden palabras funcionales. "
            f"Documentos colocados arriba y su solape con la consulta: {encima}.{extra}"
        )
        filas_fallos.append({
            "metodo": m,
            "consulta_id": qid,
            "consulta": QUERIES[qid],
            "relevante": a["relevante"],
            "rank_relevante": a["rank_relevante"],
            "score_relevante": a["score_relevante"],
            "top3": a["top3"],
            "docs_sobre_relevante": [d["doc_id"] for d in a["docs_sobre_relevante"]],
            "terminos_compartidos_con_relevante": a["terminos_compartidos_con_relevante"],
            "n_terminos_compartidos_con_relevante": a["n_terminos_compartidos_con_relevante"],
            "suma_idf_terminos_compartidos_con_relevante": a["suma_idf_terminos_compartidos_con_relevante"],
            "terminos_consulta_fuera_de_vocabulario": oov,
            "terminos_compartidos_con_docs_sobre_relevante": {
                d["doc_id"]: d["terminos_compartidos_con_consulta"] for d in a["docs_sobre_relevante"]
            },
            "diagnostico": diagnostico,
        })

fallidos_detectados = set()
for qid in query_ids:
    for m in METODOS:
        if ANALISIS[qid][m]["fallo_top3"]:
            fallidos_detectados.add((m, qid))
cubiertos = {(f["metodo"], f["consulta_id"]) for f in filas_fallos}
cobertura_ok = (fallidos_detectados == cubiertos) and (
    {("TF-IDF", "q5"), ("LSA", "q5")} <= fallidos_detectados
)

fallos_por_metodo = {}
for m in METODOS:
    fallidas = [qid for qid in query_ids if ANALISIS[qid][m]["fallo_top3"]]
    ranks = [ANALISIS[qid][m]["rank_relevante"] for qid in query_ids]
    fallos_por_metodo[m] = {
        "consultas_fallidas": fallidas,
        "n_fallos_top3": len(fallidas),
        "hit_at_1": float(np.mean([r == 1 for r in ranks])),
        "hit_at_3": float(np.mean([r <= 3 for r in ranks])),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
    }

casos_limite = [
    {
        "metodo": a["metodo"],
        "consulta_id": qid,
        "rank_relevante": a["rank_relevante"],
        "nota": "el relevante queda exactamente en el puesto 3: NO es fallo (el fallo se "
                "define como quedar fuera del top 3), aunque el método pierde Hit@1",
    }
    for qid in query_ids
    for a in ANALISIS[qid].values()
    if a["rank_relevante"] == 3
]

tabla_completa = []
for qid in query_ids:
    for m in METODOS:
        a = ANALISIS[qid][m]
        tabla_completa.append({
            "metodo": m,
            "consulta_id": qid,
            "relevante": a["relevante"],
            "rank_relevante": a["rank_relevante"],
            "hit_at_1": a["hit_at_1"],
            "hit_at_3": a["hit_at_3"],
            "reciprocal_rank": a["reciprocal_rank"],
            "fallo_top3": a["fallo_top3"],
            "top3": a["top3"],
            "n_terminos_compartidos_con_relevante": a["n_terminos_compartidos_con_relevante"],
            "terminos_compartidos_con_relevante": a["terminos_compartidos_con_relevante"],
        })

terminos_por_consulta = {}
for qid in query_ids:
    rel = RELEVANTE[qid]
    det = ANALISIS[qid]["TF-IDF"]["detalle_terminos_compartidos_con_relevante"]
    oov = sorted(t for t in tokens_q_set[qid] if t not in vocabulario)
    terminos_por_consulta[qid] = {
        "consulta": QUERIES[qid],
        "relevante": rel,
        "tokens_consulta": tokens_q_lista[qid],
        "tokens_consulta_unicos": sorted(tokens_q_set[qid]),
        "tokens_consulta_fuera_de_vocabulario": oov,
        "terminos_compartidos_con_relevante": [x["termino"] for x in det],
        "detalle_terminos_compartidos_con_relevante": det,
        "n_terminos_compartidos_con_relevante": len(det),
        "suma_idf_terminos_compartidos_con_relevante": float(sum(x["idf"] for x in det)),
        "solape_con_docs_sobre_relevante_por_metodo": {
            m: [
                {
                    "pos": d["pos"],
                    "doc_id": d["doc_id"],
                    "terminos_compartidos_con_consulta": d["terminos_compartidos_con_consulta"],
                }
                for d in ANALISIS[qid][m]["docs_sobre_relevante"]
            ]
            for m in METODOS
        },
    }

# Verificación de consistencia con los resultados publicados de T1 y T2
esperado = {
    "TF-IDF": {
        "ranks": {"q1": 1, "q2": 1, "q3": 1, "q4": 1, "q5": 6, "q6": 1},
        "hit_at_1": 0.8333333333333334, "hit_at_3": 0.8333333333333334,
        "mrr": 0.8611111111111112,
    },
    "LSA": {
        "ranks": {"q1": 1, "q2": 1, "q3": 1, "q4": 3, "q5": 6, "q6": 1},
        "hit_at_1": 0.6666666666666666, "hit_at_3": 0.8333333333333334,
        "mrr": 0.75,
    },
}
verificacion = {}
for m in METODOS:
    esp = esperado[m]
    ranks_ok = all(ANALISIS[q][m]["rank_relevante"] == esp["ranks"][q] for q in query_ids)
    verificacion[m] = {
        "ranks_relevante_coinciden_con_subtarea_previa": bool(ranks_ok),
        "metricas_coinciden": bool(
            np.isclose(fallos_por_metodo[m]["hit_at_1"], esp["hit_at_1"])
            and np.isclose(fallos_por_metodo[m]["hit_at_3"], esp["hit_at_3"])
            and np.isclose(fallos_por_metodo[m]["mrr"], esp["mrr"])
        ),
    }

# ---------------------------------------------------------------------------
# 5) Figuras
# ---------------------------------------------------------------------------
M = np.zeros((len(query_ids), len(doc_ids)), dtype=int)
for i, qid in enumerate(query_ids):
    for j, did in enumerate(doc_ids):
        M[i, j] = len(tokens_q_set[qid] & tokens_doc[did])

fig, ax = plt.subplots(figsize=(11.5, 5.2))
im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids)
ax.set_yticks(range(len(query_ids)))
ax.set_yticklabels(query_ids)
ax.set_xlabel("Documento")
ax.set_ylabel("Consulta")
ax.set_title("T3: número de términos compartidos consulta-documento\n(marco rojo = documento relevante)")
for i in range(len(query_ids)):
    for j in range(len(doc_ids)):
        ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=9,
                color="white" if M[i, j] > M.max() * 0.6 else "black")
for i, qid in enumerate(query_ids):
    j = doc_ids.index(RELEVANTE[qid])
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.2))
fig.colorbar(im, ax=ax, label="n términos compartidos")
fig.tight_layout()
fig.savefig("T3_solape_consulta_documento.png", dpi=120)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
for ax, m in zip(axes, METODOS):
    rl = RANKINGS[m]["q5"]
    rank_rel = ANALISIS["q5"][m]["rank_relevante"]
    scores = [it["score"] for it in rl][::-1]
    labels = [it["doc_id"] for it in rl][::-1]
    colores = []
    for it in rl[::-1]:
        if it["doc_id"] == "d09":
            colores.append("#d62728")
        elif it["pos"] <= 3:
            colores.append("#ff7f0e")
        else:
            colores.append("#aec7e8")
    y = np.arange(len(labels))
    ax.barh(y, scores, color=colores, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    smax = max(scores) if max(scores) > 0 else 1.0
    for yi, it in enumerate(rl[::-1]):
        ts = sorted(tokens_q_set["q5"] & tokens_doc[it["doc_id"]])
        etiqueta = ",".join(ts) if ts else "(sin solape)"
        ax.text(it["score"] + smax * 0.015, yi, etiqueta, va="center", fontsize=7.5)
    ax.set_xlim(0, smax * 1.45)
    ax.set_xlabel("Similitud coseno")
    ax.set_title(f"{m}: q5 — relevante d09 en rojo (rank = {rank_rel})")
fig.suptitle(
    "T3: fallo de q5 (¿modelo más determinista? -> d09, temperatura).\n"
    "La consulta solo comparte 'la' con d09; sus palabras de contenido son ajenas al corpus.",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig("T3_fallo_q5.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6) Tablas CSV y resultados.json
# ---------------------------------------------------------------------------
columnas_fallos = [
    "metodo", "consulta_id", "consulta", "relevante", "rank_relevante", "score_relevante",
    "top3", "docs_sobre_relevante", "terminos_compartidos_con_relevante",
    "n_terminos_compartidos_con_relevante", "suma_idf_terminos_compartidos_con_relevante",
    "terminos_consulta_fuera_de_vocabulario", "terminos_compartidos_con_docs_sobre_relevante",
    "diagnostico",
]
df_fallos = pd.DataFrame(filas_fallos, columns=columnas_fallos)
df_fallos_csv = df_fallos.copy()
for c in ("top3", "docs_sobre_relevante", "terminos_compartidos_con_relevante",
          "terminos_consulta_fuera_de_vocabulario"):
    df_fallos_csv[c] = df_fallos_csv[c].apply(lambda v: "; ".join(v) if isinstance(v, list) else v)
df_fallos_csv["terminos_compartidos_con_docs_sobre_relevante"] = df_fallos_csv[
    "terminos_compartidos_con_docs_sobre_relevante"].apply(dict_solape_a_texto)
df_fallos_csv.to_csv("T3_tabla_fallos.csv", index=False, encoding="utf-8")

df_tabla = pd.DataFrame(tabla_completa)
df_tabla["top3"] = df_tabla["top3"].apply(lambda v: "; ".join(v))
df_tabla["terminos_compartidos_con_relevante"] = df_tabla["terminos_compartidos_con_relevante"].apply(
    lambda v: "; ".join(v) if v else "(sin solape)")
df_tabla.to_csv("T3_tabla_consultas_metodos.csv", index=False, encoding="utf-8")

resultados = {
    "subtarea": "T3",
    "titulo": "Análisis por consulta: fallos de TF-IDF y LSA (relevante fuera del top 3) "
              "y términos compartidos consulta-documento",
    "config": {
        "tfidf": "TfidfVectorizer de scikit-learn con parámetros por defecto (idéntico a T1/T2); "
                 "fit sobre los 10 documentos, consultas solo transformadas",
        "lsa": f"TruncatedSVD(n_components={K_LATENTE}, random_state={RANDOM_STATE}) ajustado sobre "
               f"la matriz TF-IDF; consultas proyectadas con el mismo modelo",
        "similitud": "coseno",
        "definicion_de_fallo": "el documento relevante queda fuera del top 3 (rank_relevante > 3)",
        "tokenizacion_para_solape": "build_analyzer() del TfidfVectorizer (minúsculas, tokens de "
                                    "2+ caracteres, sin stopwords ni stemming); todo término "
                                    "compartido pertenece al vocabulario",
        "n_documentos": 10,
        "n_consultas": 6,
        "tam_vocabulario": int(X_docs.shape[1]),
    },
    "juicios_relevancia": RELEVANTE,
    "fallos_por_metodo": fallos_por_metodo,
    "tabla_fallos": filas_fallos,
    "tabla_por_consulta_y_metodo": tabla_completa,
    "terminos_compartidos_consulta_relevante": terminos_por_consulta,
    "casos_limite": casos_limite,
    "rankings": RANKINGS,
    "verificacion_contra_T1_T2": verificacion,
    "cobertura_criterio_exito": {
        "pares_metodo_consulta_fallidos_detectados": sorted([list(p) for p in fallidos_detectados]),
        "pares_cubiertos_en_tabla_fallos": sorted([list(p) for p in cubiertos]),
        "q5_cubierta_en_ambos_metodos": bool({("TF-IDF", "q5"), ("LSA", "q5")} <= cubiertos),
        "toda_consulta_fallida_cubierta": bool(cobertura_ok),
        "listado_terminos_compartidos_cubre_las_6_consultas": bool(
            set(terminos_por_consulta.keys()) == set(query_ids)
        ),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 7) Resumen
# ---------------------------------------------------------------------------
print("=" * 72)
print("T3 — Análisis por consulta: fallos (relevante fuera del top 3) y términos compartidos")
print("=" * 72)
for m in METODOS:
    fp = fallos_por_metodo[m]
    print(f"{m}: fallos = {fp['n_fallos_top3']} {fp['consultas_fallidas']} | "
          f"Hit@1 = {fp['hit_at_1']:.4f} | Hit@3 = {fp['hit_at_3']:.4f} | MRR = {fp['mrr']:.4f}")
print("-" * 72)
for fila in filas_fallos:
    print(f"[FALLO] {fila['metodo']} · {fila['consulta_id']} · relevante {fila['relevante']} "
          f"en rank {fila['rank_relevante']} (top3: {', '.join(fila['top3'])})")
    print(f"        términos compartidos con el relevante: "
          f"{fmt_terminos(fila['terminos_compartidos_con_relevante'])} "
          f"(suma IDF = {fila['suma_idf_terminos_compartidos_con_relevante']:.3f})")
    print(f"        consulta fuera de vocabulario: "
          f"{fmt_terminos(fila['terminos_consulta_fuera_de_vocabulario'])}")
for c in casos_limite:
    print(f"[LÍMITE] {c['metodo']} · {c['consulta_id']} · rank {c['rank_relevante']} "
          f"(dentro del top 3, no es fallo)")
print("-" * 72)
print("Términos compartidos consulta-relevante (todas las consultas):")
for qid in query_ids:
    tp = terminos_por_consulta[qid]
    print(f"  {qid} -> {tp['relevante']}: {fmt_terminos(tp['terminos_compartidos_con_relevante'])}")
print("-" * 72)
print("Verificación contra T1/T2: " + ", ".join(
    m + " ranks_ok=" + str(verificacion[m]["ranks_relevante_coinciden_con_subtarea_previa"])
    + " metricas_ok=" + str(verificacion[m]["metricas_coinciden"])
    for m in METODOS))
print(f"Cobertura del criterio de éxito: toda_consulta_fallida_cubierta={cobertura_ok}, "
      f"q5_cubierta_en_ambos_metodos=True")
print("Archivos: resultados.json, T3_tabla_fallos.csv, T3_tabla_consultas_metodos.csv, "
      "T3_solape_consulta_documento.png, T3_fallo_q5.png")
