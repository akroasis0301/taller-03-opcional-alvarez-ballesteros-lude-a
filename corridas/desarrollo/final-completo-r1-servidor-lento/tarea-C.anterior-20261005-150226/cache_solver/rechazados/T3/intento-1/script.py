```python
# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Análisis por consulta.

A partir de los rankings de los dos métodos de las partes previas
  · TF-IDF + similitud coseno        (T1: TfidfVectorizer por defecto, ajustado solo con los 10 documentos)
  · LSA de 4 dimensiones + coseno    (T2: TruncatedSVD(n_components=4, random_state=0))
esta subtarea:
  1) identifica en qué consultas falla cada método (documento relevante fuera del
     top 3), guardando la posición del relevante en cada caso;
  2) calcula, para cada consulta, qué términos comparte la consulta con su documento
     relevante (solapamiento léxico sobre el vocabulario TF-IDF de 105 términos),
     con los pesos TF-IDF de cada término compartido y una clasificación auxiliar
     función/contenido;
  3) explica los fallos mirando ese solapamiento.

Salidas: resultados.json, T3_posicion_relevante.png, T3_solapamiento_terminos.png
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP_K = 3  # criterio de fallo del enunciado: relevante fuera del top 3

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios de relevancia (datos completos del enunciado)
# ---------------------------------------------------------------------------
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]
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
Q_IDS = [f"q{i}" for i in range(1, 7)]
CONSULTAS = {
    "q1": "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "q2": "¿Cómo se añaden fragmentos recuperados al prompt?",
    "q3": "¿Qué técnica entrena matrices de bajo rango?",
    "q4": "¿Cómo se compara la orientación de dos vectores de texto?",
    "q5": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "q6": "¿Qué arquitectura combina autoatención con capas feed-forward?",
}
RELEVANTE = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}

# ---------------------------------------------------------------------------
# 2) Métodos de las partes previas (mismos parámetros exactos)
# ---------------------------------------------------------------------------
vec = TfidfVectorizer()  # idéntico a T1: parámetros por defecto, ajustado SOLO con los 10 documentos
X_docs = vec.fit_transform([DOCUMENTOS[d] for d in DOC_IDS])
X_cons = vec.transform([CONSULTAS[q] for q in Q_IDS])
vocab = [str(t) for t in vec.get_feature_names_out()]
vocab_set = set(vocab)
idx_term = {t: j for j, t in enumerate(vocab)}
Xd = X_docs.toarray()
Xq = X_cons.toarray()
S_tfidf = cosine_similarity(X_cons, X_docs)

svd = TruncatedSVD(n_components=4, random_state=0)  # idéntico a T2
D_lat = svd.fit_transform(X_docs)
Q_lat = svd.transform(X_cons)
S_lsa_recalc = cosine_similarity(Q_lat, D_lat)


def rankings_desde_similitud(S):
    """Ranking de los 10 documentos por consulta: descendente por similitud,
    empates resueltos por orden del corpus d01..d10 (argsort estable), como en T1/T2."""
    return [[DOC_IDS[j] for j in np.argsort(-S[i], kind="stable")] for i in range(S.shape[0])]


def posicion_de(doc_id, ranking):
    return ranking.index(doc_id) + 1


R_tfidf = rankings_desde_similitud(S_tfidf)
pos_tfidf = {q: posicion_de(RELEVANTE[q], R_tfidf[i]) for i, q in enumerate(Q_IDS)}

# ---------------------------------------------------------------------------
# 3) Verificación contra los resultados oficiales de T1 y T2 (si están disponibles)
# ---------------------------------------------------------------------------
def cargar_json(ruta):
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except Exception:
        return None


t1 = cargar_json("entrada/T1/resultados.json")
t2 = cargar_json("entrada/T2/resultados.json")
verif = {"T1": {"archivo": "entrada/T1/resultados.json" if t1 is not None else "no disponible"},
         "T2": {"archivo": "entrada/T2/resultados.json" if t2 is not None else "no disponible"}}

S_lsa = S_lsa_recalc
fuente_lsa = "recalculado aquí con TruncatedSVD(n_components=4, random_state=0)"
if t2 is not None:
    try:
        coords = t2["lsa"]["coordenadas_latentes"]
        D2 = np.array([coords["documentos"][d] for d in DOC_IDS], dtype=float)
        Q2 = np.array([coords["consultas"][q] for q in Q_IDS], dtype=float)
        S2 = cosine_similarity(Q2, D2)
        verif["T2"]["similitud_lsa_coincide_con_recalculo"] = bool(np.allclose(S2, S_lsa_recalc, atol=1e-6))
        verif["T2"]["max_dif_similitud_lsa"] = float(np.max(np.abs(S2 - S_lsa_recalc)))
        S_lsa = S2  # se usan las coordenadas oficiales de T2 para garantizar consistencia
        fuente_lsa = "coordenadas latentes oficiales leídas de entrada/T2/resultados.json"
        if "valores_singulares" in t2.get("lsa", {}):
            verif["T2"]["valores_singulares_coinciden"] = bool(np.allclose(
                np.array(t2["lsa"]["valores_singulares"], dtype=float), svd.singular_values_, atol=1e-6))
    except Exception as exc:
        verif["T2"]["error_leyendo_coordenadas"] = str(exc)

R_lsa = rankings_desde_similitud(S_lsa)
pos_lsa = {q: posicion_de(RELEVANTE[q], R_lsa[i]) for i, q in enumerate(Q_IDS)}

if t1 is not None:
    try:
        M1 = np.array(t1["matriz_similitud_coseno"]["valores"], dtype=float)
        verif["T1"]["matriz_similitud_tfidf_coincide"] = bool(M1.shape == S_tfidf.shape and np.allclose(M1, S_tfidf, atol=1e-8))
        verif["T1"]["max_dif_matriz_tfidf"] = float(np.max(np.abs(M1 - S_tfidf)))
        filas1 = {str(f["consulta"]): f for f in t1["por_consulta"]}
        verif["T1"]["rankings_completos_coinciden"] = bool(
            all(q in filas1 and list(filas1[q]["ranking_completo"]) == R_tfidf[i] for i, q in enumerate(Q_IDS)))
        verif["T1"]["ranks_del_relevante_coinciden"] = bool(
            all(q in filas1 and int(filas1[q]["rank_del_relevante"]) == pos_tfidf[q] for q in Q_IDS))
    except Exception as exc:
        verif["T1"]["error"] = str(exc)


def tablas_rank_encontradas(obj):
    """Busca listas de dicts con 'consulta'/'id' y 'rank_del_relevante' (tablas por consulta guardadas por T2)."""
    hallazgos = []

    def rec(o, ruta):
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, ruta + [str(k)])
        elif isinstance(o, list):
            if (len(o) == len(Q_IDS) and all(isinstance(x, dict) for x in o)
                    and all(("consulta" in x or "id" in x) and "rank_del_relevante" in x for x in o)):
                ids = [str(x.get("consulta", x.get("id"))) for x in o]
                if set(ids) == set(Q_IDS):
                    hallazgos.append((".".join(ruta),
                                      {str(x.get("consulta", x.get("id"))): int(x["rank_del_relevante"]) for x in o}))
            for i, v in enumerate(o):
                if isinstance(v, (dict, list)):
                    rec(v, ruta + [str(i)])

    rec(obj, [])
    return hallazgos


if t2 is not None:
    coinc = []
    for ruta, ranks in tablas_rank_encontradas(t2):
        if all(ranks[q] == pos_tfidf[q] for q in Q_IDS):
            coinc.append({"ruta": ruta, "coincide_con": "tfidf"})
        elif all(ranks[q] == pos_lsa[q] for q in Q_IDS):
            coinc.append({"ruta": ruta, "coincide_con": "lsa"})
        else:
            coinc.append({"ruta": ruta, "coincide_con": "ninguno"})
    if coinc:
        verif["T2"]["tablas_rank_del_relevante_encontradas"] = coinc

# ---------------------------------------------------------------------------
# 4) Análisis de fallos: relevante fuera del top 3 en cada método
# ---------------------------------------------------------------------------
def detalle_fallos(pos, R, S):
    out = []
    for i, q in enumerate(Q_IDS):
        if pos[q] > TOP_K:
            rel = RELEVANTE[q]
            out.append({
                "consulta": q,
                "texto_consulta": CONSULTAS[q],
                "relevante": rel,
                "posicion_relevante": int(pos[q]),
                "top3": list(R[i][:TOP_K]),
                "sim_coseno_top1": float(S[i, DOC_IDS.index(R[i][0])]),
                "sim_coseno_relevante": float(S[i, DOC_IDS.index(rel)]),
            })
    return out


fallos_tfidf = detalle_fallos(pos_tfidf, R_tfidf, S_tfidf)
fallos_lsa = detalle_fallos(pos_lsa, R_lsa, S_lsa)
fallan_tfidf = [f["consulta"] for f in fallos_tfidf]
fallan_lsa = [f["consulta"] for f in fallos_lsa]
fallan_ambos = set(fallan_tfidf) & set(fallan_lsa)

# ---------------------------------------------------------------------------
# 5) Solapamiento léxico consulta–documento relevante sobre el vocabulario TF-IDF
# ---------------------------------------------------------------------------
analyzer = vec.build_analyzer()
CANDIDATAS_FUNCION = {"un", "una", "el", "la", "los", "al", "del", "de", "en", "con", "para", "por",
                      "
