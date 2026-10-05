#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subtarea T3 — Parte 3: análisis por consulta (fallos y términos compartidos).

(1) Con los rankings de T1 (TF-IDF) y T2 (LSA-4) identifica en qué consultas
    falla cada método: el documento relevante queda fuera del top 3.
(2) Para cada consulta fallida —en particular q5 con d09— extrae los términos
    que comparten la consulta y su documento relevante según el vocabulario
    TF-IDF de la Parte 1, con sus pesos, y redacta la explicación del fallo.

Entradas: entrada/T1/resultados.json y entrada/T2/resultados.json (rankings
oficiales de T1/T2; si faltara alguno se recomputa con los parámetros exactos
del enunciado: TfidfVectorizer por defecto y TruncatedSVD(4, random_state=0)).
Salidas:  resultados.json y figuras PNG en la carpeta actual.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios de relevancia (idénticos al enunciado)
# ---------------------------------------------------------------------------
DOC_IDS = ["d%02d" % i for i in range(1, 11)]
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
QUERY_IDS = ["q%d" % i for i in range(1, 7)]
QUERIES = {
    "q1": "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "q2": "¿Cómo se añaden fragmentos recuperados al prompt?",
    "q3": "¿Qué técnica entrena matrices de bajo rango?",
    "q4": "¿Cómo se compara la orientación de dos vectores de texto?",
    "q5": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "q6": "¿Qué arquitectura combina autoatención con capas feed-forward?",
}
RELEVANTE = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}
TOP_K = 3
METODOS = ["TF-IDF", "LSA(4)"]

# Palabras funcionales (solo para etiquetar términos al explicar el fallo;
# no interviene en ningún cálculo ni en el vectorizador).
FUNCIONALES = {
    "el", "la", "los", "las", "del", "al", "que", "qué", "cómo", "como", "con",
    "de", "en", "y", "o", "u", "para", "por", "su", "un", "una", "es", "sea",
    "son", "más", "se", "lo", "a", "ni", "si", "ya", "muy", "cada", "según",
    "sin", "sobre", "entre", "hacia", "ante", "bajo", "tras", "desde", "hasta",
}

# ---------------------------------------------------------------------------
# 2) Recomputación de referencia (parámetros exactos de T1/T2)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, como en T1
X = vectorizer.fit_transform([DOCS[d] for d in DOC_IDS])    # solo los 10 documentos
Q = vectorizer.transform([QUERIES[q] for q in QUERY_IDS])   # consultas transformadas
S_TFIDF = cosine_similarity(Q, X)                            # (6, 10)

svd = TruncatedSVD(n_components=4, random_state=0)           # como en T2
L_D = svd.fit_transform(X)                                   # (10, 4)
L_Q = svd.transform(Q)                                       # folding-in: (6, 4)
S_LSA = cosine_similarity(L_Q, L_D)                          # (6, 10)

IDX_Q = {q: i for i, q in enumerate(QUERY_IDS)}
IDX_D = {d: i for i, d in enumerate(DOC_IDS)}


def ranking_de_fila(fila):
    """Orden descendente por similitud; empates por orden original de documentos."""
    orden = np.argsort(-np.asarray(fila), kind="stable")
    return [DOC_IDS[i] for i in orden]


RANK_REC = {
    "TF-IDF": {q: ranking_de_fila(S_TFIDF[IDX_Q[q]]) for q in QUERY_IDS},
    "LSA(4)": {q: ranking_de_fila(S_LSA[IDX_Q[q]]) for q in QUERY_IDS},
}
SIM_REC = {
    "TF-IDF": {q: {d: float(S_TFIDF[IDX_Q[q], IDX_D[d]]) for d in DOC_IDS} for q in QUERY_IDS},
    "LSA(4)": {q: {d: float(S_LSA[IDX_Q[q], IDX_D[d]]) for d in DOC_IDS} for q in QUERY_IDS},
}

# ---------------------------------------------------------------------------
# 3) Rankings oficiales de T1/T2 (con respaldo por recomputación)
# ---------------------------------------------------------------------------
def cargar_json(ruta):
    p = Path(ruta)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


T1 = cargar_json("entrada/T1/resultados.json")
T2 = cargar_json("entrada/T2/resultados.json")


def extraer_por_consulta(res, clave):
    out = {}
    if isinstance(res, dict):
        for item in res.get(clave) or []:
            cid = item.get("consulta_id")
            if cid and item.get("ranking"):
                out[cid] = {
                    "relevante": item.get("relevante"),
                    "ranking": list(item["ranking"]),
                    "similitudes": dict(item.get("similitudes") or {}),
                }
    return out


FILE_RANK = {
    "TF-IDF": extraer_por_consulta(T1, "por_consulta"),
    "LSA(4)": extraer_por_consulta(T2, "lsa_por_consulta"),
}

RANK, SIMS, ORIGEN = {}, {}, {}
for m in METODOS:
    RANK[m], SIMS[m] = {}, {}
    fr = FILE_RANK[m]
    carpeta = "T1" if m == "TF-IDF" else "T2"
    if fr:
        ORIGEN[m] = "entrada/%s/resultados.json (%d/6 consultas)" % (carpeta, len(fr))
    else:
        ORIGEN[m] = "recomputación local con los parámetros del enunciado"
    for q in QUERY_IDS:
        if q in fr:
            RANK[m][q] = fr[q]["ranking"]
            s = {d: float(v) for d, v in fr[q]["similitudes"].items() if d in DOC_IDS}
            for d in DOC_IDS:
                s.setdefault(d, SIM_REC[m][q][d])
            SIMS[m][q] = s
        else:
            RANK[m][q] = RANK_REC[m][q]
            SIMS[m][q] = dict(SIM_REC[m][q])

VERIF = {}
for m in METODOS:
    comp = [q for q in QUERY_IDS if q in FILE_RANK[m]]
    VERIF[m] = {
        "consultas_con_ranking_en_archivo": len(comp),
        "rankings_identicos_a_recomputacion": sum(
            1 for q in comp if FILE_RANK[m][q]["ranking"] == RANK_REC[m][q]
        ),
        "juicios_de_relevancia_coinciden": all(
            FILE_RANK[m][q]["relevante"] in (None, RELEVANTE[q]) for q in comp
        ),
    }

VOCAB_TAM = len(vectorizer.vocabulary_)
CHECK_VOCAB = None
if isinstance(T1, dict) and isinstance(T1.get("vocabulario_tamano"), int):
    CHECK_VOCAB = bool(T1["vocabulario_tamano"] == VOCAB_TAM)

# ---------------------------------------------------------------------------
# 4) Posición del relevante, fallos (fuera del top 3) y métricas
# ---------------------------------------------------------------------------
def posicion_de(rank, rel):
    return rank.index(rel) + 1 if rel in rank else len(rank) + 1


POS = {m: {q: posicion_de(RANK[m][q], RELEVANTE[q]) for q in QUERY_IDS} for m in METODOS}
FALLOS = {m: [q for q in QUERY_IDS if POS[m][q] > TOP_K] for m in METODOS}

METRICAS = {}
for m in METODOS:
    METRICAS[m] = {
        "Hit@1": sum(1 for q in QUERY_IDS if POS[m][q] == 1) / len(QUERY_IDS),
        "Hit@3": sum(1 for q in QUERY_IDS if POS[m][q] <= TOP_K) / len(QUERY_IDS),
        "MRR": float(np.mean([1.0 / POS[m][q] for q in QUERY_IDS])),
    }

# ---------------------------------------------------------------------------
# 5) Términos compartidos consulta–documento relevante (vocabulario TF-IDF)
# ---------------------------------------------------------------------------
ANALYZER = vectorizer.build_analyzer()
VOCAB = vectorizer.vocabulary_
Xd = X.toarray()
Qd = Q.toarray()
DF_COUNTS = (Xd > 0).sum(axis=0)


def terminos_compartidos(qid, did):
    """Términos del vocabulario TF-IDF presentes en la consulta y en el documento."""
    toks_q = ANALYZER(QUERIES[qid])
    toks_d = ANALYZER(DOCS[did])
    filas = []
    for t in sorted(set(toks_q) & set(toks_d)):
        j = VOCAB[t]
        filas.append({
            "termino": t,
            "peso_consulta_tfidf": float(Qd[IDX_Q[qid], j]),
            "peso_documento_tfidf": float(Xd[IDX_D[did], j]),
            "idf": float(vectorizer.idf_[j]),
            "df": int(DF_COUNTS[j]),
            "frecuencia_en_consulta": int(toks_q.count(t)),
            "frecuencia_en_documento": int(toks_d.count(t)),
            "tipo": "funcional" if t in FUNCIONALES else "contenido",
        })
    filas.sort(key=lambda r: (-r["idf"], r["termino"]))
    en_vocab = sorted({t for t in toks_q if t in VOCAB})
    fuera_vocab = sorted({t for t in toks_q if t not in VOCAB})
    ausentes = sorted(set(toks_d) - set(toks_q))
    return filas, en_vocab, fuera_vocab, ausentes


def terminos_comunes(qid, did):
    return sorted(set(ANALYZER(QUERIES[qid])) & set(ANALYZER(DOCS[did])))


def acotar(lista, n=8):
    lista = list(lista)
    if len(lista) <= n:
        return ", ".join(lista)
    return ", ".join(lista[:n]) + ", …"


def fmt_terminos(filas, rel):
    partes = []
    for f in filas:
        partes.append(
            "'%s' (peso en la consulta=%.4f, peso en %s=%.4f, IDF=%.4f, DF=%d/10)"
            % (f["termino"], f["peso_consulta_tfidf"], rel,
               f["peso_documento_tfidf"], f["idf"], f["df"])
        )
    return "; ".join(partes)


def construir_explicacion(metodo, qid, rel, filas, pos, sim_rel, top3, pos_otro, ausentes, var_exp):
    if filas:
        base = (
            "La consulta %s y su relevante %s comparten %d término(s) del vocabulario TF-IDF: %s."
            % (qid, rel, len(filas), fmt_terminos(filas, rel))
        )
    else:
        base = ("La consulta %s y su relevante %s no comparten ningún término del vocabulario TF-IDF."
                % (qid, rel))
    partes_top3 = []
    for t in top3:
        partes_top3.append(
            "%s (pos %d, sim %.4f, comparte: %s)"
            % (t["documento"], t["posicion"], t["similitud"],
               acotar(t["terminos_compartidos_con_consulta"], 6) or "—")
        )
    top3_txt = "; ".join(partes_top3)
    contenido = [f["termino"] for f in filas if f["tipo"] == "contenido"]
    if contenido:
        if metodo == "LSA(4)":
            causa = (
                " El fallo de %s en %s no se debe a falta de solapamiento léxico —hay términos de "
                "contenido compartidos (%s)— sino a la proyección latente a 4 dimensiones: con solo "
                "10 documentos y %.1f%% de varianza explicada, las componentes latentes mezclan temas "
                "y diluyen el match léxico fuerte; la similitud de %s (%.4f) cae a la posición %d, "
                "fuera del top 3 (%s). TF-IDF, que sí explota el match léxico directo, colocaba a %s "
                "en la posición %d."
                % (metodo, qid, acotar(contenido, 8), 100.0 * var_exp, rel, sim_rel, pos,
                   top3_txt, rel, pos_otro)
            )
        else:
            causa = (
                " Aunque hay solapamiento de contenido (%s), otros documentos comparten con la "
                "consulta términos de contenido más frecuentes o más específicos y los superan; "
                "top 3: %s." % (acotar(contenido, 8), top3_txt)
            )
    else:
        if filas:
            causa = (
                " Todos los términos compartidos son palabras funcionales de IDF baja (DF hasta "
                "%d/10), sin capacidad discriminativa: no hay solapamiento de términos de contenido."
                % max(f["df"] for f in filas)
            )
        else:
            causa = " No existe solapamiento léxico de ningún tipo."
        causa += (
            " Los términos de contenido de la consulta no aparecen en %s y los términos "
            "característicos de %s (%s) no aparecen en la consulta: la relación consulta–documento "
            "es puramente semántica (una paráfrasis), invisible para el matching léxico."
            % (rel, rel, acotar(ausentes, 8))
        )
        if metodo == "TF-IDF":
            causa += (
                " La similitud coseno de %s (%.4f) solo recoge el cruce de palabras funcionales y "
                "queda por debajo de la de documentos que comparten más (o más específicas) palabras "
                "con la consulta — top 3: %s —, de modo que %s cae a la posición %d, fuera del top 3."
                % (rel, sim_rel, top3_txt, rel, pos)
            )
        else:
            causa += (
                " LSA solo puede acercar vectores mediante co-ocurrencias observadas en el corpus; "
                "al no co-ocurrir esos términos en los 10 documentos, la proyección a 4 componentes "
                "no salva la paráfrasis y %s queda en la posición %d (top 3: %s)."
                % (rel, pos, top3_txt)
            )
    return base + causa

# ---------------------------------------------------------------------------
# 6) Registro detallado de cada fallo
# ---------------------------------------------------------------------------
var_exp = float(svd.explained_variance_ratio_.sum())
ANALISIS_FALLOS = []
TABLA_TERMINOS = []
for m in METODOS:
    otro = "LSA(4)" if m == "TF-IDF" else "TF-IDF"
    for qid in FALLOS[m]:
        rel = RELEVANTE[qid]
        pos = POS[m][qid]
        sim_rel = float(SIMS[m][qid][rel])
        rank = RANK[m][qid]
        top3 = []
        for i in range(min(TOP_K, len(rank))):
            top3.append({
                "posicion": i + 1,
                "documento": rank[i],
                "similitud": float(SIMS[m][qid][rank[i]]),
                "terminos_compartidos_con_consulta": terminos_comunes(qid, rank[i]),
            })
        filas, en_vocab, fuera_vocab, ausentes = terminos_compartidos(qid, rel)
        contenido = [f["termino"] for f in filas if f["tipo"] == "contenido"]
        rec = {
            "metodo": m,
            "consulta_id": qid,
            "consulta": QUERIES[qid],
            "relevante": rel,
            "texto_relevante": DOCS[rel],
            "posicion_del_relevante": int(pos),
            "similitud_del_relevante": sim_rel,
            "posicion_del_relevante_con_el_otro_metodo": int(POS[otro][qid]),
            "similitud_del_relevante_con_el_otro_metodo": float(SIMS[otro][qid][rel]),
            "ranking": list(rank),
            "top3": top3,
            "terminos_compartidos_consulta_relevante": filas,
            "n_terminos_compartidos": len(filas),
            "terminos_de_contenido_compartidos": contenido,
            "diagnostico": (
                "proyeccion_latente_diluye_match_lexico" if contenido
                else "parafasis_sin_solapamiento_lexico_de_contenido"
            ),
            "tokens_de_la_consulta_en_vocabulario": en_vocab,
            "tokens_de_la_consulta_fuera_del_vocabulario": fuera_vocab,
            "tokens_del_relevante_ausentes_en_la_consulta": ausentes,
            "explicacion": construir_explicacion(
                m, qid, rel, filas, pos, sim_rel, top3, POS[otro][qid], ausentes, var_exp
            ),
        }
        ANALISIS_FALLOS.append(rec)
        for f in filas:
            TABLA_TERMINOS.append({
                "metodo": m,
                "consulta": qid,
                "relevante": rel,
                "termino": f["termino"],
                "peso_consulta_tfidf": f["peso_consulta_tfidf"],
                "peso_documento_tfidf": f["peso_documento_tfidf"],
                "idf": f["idf"],
                "df": f["df"],
                "tipo": f["tipo"],
            })

# ---------------------------------------------------------------------------
# 7) Caso destacado: q5 con d09
# ---------------------------------------------------------------------------
filas_q5, en_vocab_q5, fuera_q5, ausentes_d09 = terminos_compartidos("q5", "d09")
if filas_q5:
    compartidos_q5 = ", ".join("'%s'" % f["termino"] for f in filas_q5)
    hay_contenido = any(f["tipo"] == "contenido" for f in filas_q5)
    if hay_contenido:
        puente = "Hay términos de contenido compartidos."
    elif len(filas_q5) == 1:
        puente = "El único puente léxico es una palabra funcional, sin poder discriminativo."
    else:
        puente = "Todos los términos compartidos son palabras funcionales, sin poder discriminativo."
    base_q5 = (
        "q5 y d09 comparten %d término(s) del vocabulario TF-IDF: %s (%s). %s "
        "De los términos de la consulta presentes en el vocabulario (%s), solo %s aparece(n) en "
        "d09 —el resto tiene peso 0 en d09—; además, %d término(s) de la consulta ni siquiera "
        "pertenecen al vocabulario (%s), y los términos que definen a d09 (%s) no aparecen en la "
        "consulta: 'más determinista' es una paráfrasis de 'la temperatura baja concentra la "
        "probabilidad'. "
        % (
            len(filas_q5),
            compartidos_q5,
            "; ".join(
                "'%s': peso en la consulta=%.4f, peso en d09=%.4f, IDF=%.4f, DF=%d/10"
                % (f["termino"], f["peso_consulta_tfidf"], f["peso_documento_tfidf"],
                   f["idf"], f["df"])
                for f in filas_q5
            ),
            puente,
            acotar(en_vocab_q5, 10),
            compartidos_q5,
            len(fuera_q5),
            acotar(fuera_q5, 10),
            acotar(ausentes_d09, 10),
        )
    )
    if POS["TF-IDF"]["q5"] > TOP_K:
        cola_tf = (
            "Sin solapamiento de contenido, TF-IDF puntúa a d09 solo por el cruce de palabras "
            "funcionales y lo deja en la posición %d (similitud %.4f), fuera del top 3. "
            % (POS["TF-IDF"]["q5"], SIMS["TF-IDF"]["q5"]["d09"])
        )
    else:
        cola_tf = (
            "Aun con ese puente mínimo, TF-IDF coloca a d09 en la posición %d (similitud %.4f), "
            "dentro del top 3. " % (POS["TF-IDF"]["q5"], SIMS["TF-IDF"]["q5"]["d09"])
        )
    if POS["LSA(4)"]["q5"] <= TOP_K:
        cola_lsa = (
            "LSA(4) sí mitiga el hueco semántico en esta consulta: d09 sube a la posición %d "
            "(similitud %.4f)." % (POS["LSA(4)"]["q5"], SIMS["LSA(4)"]["q5"]["d09"])
        )
    else:
        cola_lsa = (
            "LSA(4) tampoco lo rescata (posición %d, similitud %.4f): con solo 10 documentos, los "
            "términos de la paráfrasis nunca co-ocurren y la proyección a 4 componentes no puede "
            "acercarlos." % (POS["LSA(4)"]["q5"], SIMS["LSA(4)"]["q5"]["d09"])
        )
    lectura_q5 = base_q5 + cola_tf + cola_lsa
else:
    lectura_q5 = "q5 y d09 no comparten ningún término del vocabulario TF-IDF."

CASO_Q5 = {
    "consulta_id": "q5",
    "consulta": QUERIES["q5"],
    "relevante": "d09",
    "texto_relevante": DOCS["d09"],
    "fallo_en_tf_idf": bool("q5" in FALLOS["TF-IDF"]),
    "posicion_en_tf_idf": int(POS["TF-IDF"]["q5"]),
    "similitud_en_tf_idf": float(SIMS["TF-IDF"]["q5"]["d09"]),
    "fallo_en_lsa": bool("q5" in FALLOS["LSA(4)"]),
    "posicion_en_lsa": int(POS["LSA(4)"]["q5"]),
    "similitud_en_lsa": float(SIMS["LSA(4)"]["q5"]["d09"]),
    "terminos_compartidos": filas_q5,
    "n_terminos_compartidos": len(filas_q5),
    "tokens_de_la_consulta_en_vocabulario": en_vocab_q5,
    "tokens_de_la_consulta_fuera_del_vocabulario": fuera_q5,
    "tokens_de_d09_ausentes_en_la_consulta": ausentes_d09,
    "lectura": lectura_q5,
}

# ---------------------------------------------------------------------------
# 8) Figuras
# ---------------------------------------------------------------------------
FIGURAS = []

# (a) posición del relevante por consulta y método
fig, ax = plt.subplots(figsize=(8.2, 4.4))
xpos = np.arange(len(QUERY_IDS))
w = 0.38
vals_tf = [POS["TF-IDF"][q] for q in QUERY_IDS]
vals_lsa = [POS["LSA(4)"][q] for q in QUERY_IDS]
cols_tf = ["#c44e52" if v > TOP_K else "#4878CF" for v in vals_tf]
cols_lsa = ["#c44e52" if v > TOP_K else "#4878CF" for v in vals_lsa]
ax.bar(xpos - w / 2.0, vals_tf, w, color=cols_tf, edgecolor="black", linewidth=0.4, label="TF-IDF (T1)")
ax.bar(xpos + w / 2.0, vals_lsa, w, color=cols_lsa, edgecolor="black", linewidth=0.4, label="LSA(4) (T2)")
ax.axhline(TOP_K + 0.5, color="gray", linestyle="--", linewidth=1.0)
ax.text(len(QUERY_IDS) - 0.55, TOP_K + 0.65, "límite del top 3", color="gray", fontsize=8, ha="right")
for i in range(len(QUERY_IDS)):
    ax.text(xpos[i] - w / 2.0, vals_tf[i] + 0.15, str(vals_tf[i]), ha="center", fontsize=8)
    ax.text(xpos[i] + w / 2.0, vals_lsa[i] + 0.15, str(vals_lsa[i]), ha="center", fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels(QUERY_IDS)
ax.set_ylim(0, 11)
ax.set_ylabel("posición del documento relevante")
ax.set_title("T3 — Posición del relevante por consulta y método (rojo = fuera del top 3)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("t3_posicion_relevante.png", dpi=120)
plt.close(fig)
FIGURAS.append("t3_posicion_relevante.png")

# (b) q5: similitudes de los 10 documentos con TF-IDF y LSA(4), d09 destacado
if filas_q5:
    cap_q5 = "términos compartidos q5–d09: " + ", ".join("'" + f["termino"] + "'" for f in filas_q5)
else:
    cap_q5 = "q5 y d09 no comparten términos del vocabulario TF-IDF"
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
for ax, m in zip(axes, METODOS):
    orden = RANK[m]["q5"]
    sims = [SIMS[m]["q5"][d] for d in orden]
    cols = ["#c44e52" if d == "d09" else ("#55A868" if i < TOP_K else "#9ba3ab")
            for i, d in enumerate(orden)]
    ypos = np.arange(len(orden))[::-1]
    ax.barh(ypos, sims, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(orden)
    ax.set_xlabel("similitud coseno con q5")
    ax.set_title("%s — relevante d09 en pos %d" % (m, POS[m]["q5"]))
    ax.axvline(0, color="black", linewidth=0.6)
fig.suptitle("T3 — q5 ('más determinista al elegir la siguiente palabra'): %s" % cap_q5, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("t3_q5_similitudes.png", dpi=120)
plt.close(fig)
FIGURAS.append("t3_q5_similitudes.png")

# (c) q5: pesos TF-IDF de los tokens de la consulta y presencia en d09
toks_d09 = ANALYZER(DOCS["d09"])
set_d09 = set(toks_d09)
pares = []
vistos = set()
for t in ANALYZER(QUERIES["q5"]):
    if t in vistos:
        continue
    vistos.add(t)
    if t in VOCAB:
        pares.append((t, float(Qd[IDX_Q["q5"], VOCAB[t]]), t in set_d09))
    else:
        pares.append((t, 0.0, False))
pares.sort(key=lambda p: -p[1])
fig, ax = plt.subplots(figsize=(8.6, 4.2))
nombres = [p[0] for p in pares]
pesos = [p[1] for p in pares]
cols = ["#55A868" if p[2] else "#c44e52" for p in pares]
ax.bar(np.arange(len(pares)), pesos, color=cols, edgecolor="black", linewidth=0.4)
ax.set_xticks(np.arange(len(pares)))
ax.set_xticklabels(nombres, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("peso TF-IDF del término en la consulta q5")
ax.set_title("T3 — Tokens de q5: verdes = presentes en d09; rojos = ausentes en d09 (o fuera de vocabulario)")
fig.tight_layout()
fig.savefig("t3_q5_tokens.png", dpi=120)
plt.close(fig)
FIGURAS.append("t3_q5_tokens.png")

# ---------------------------------------------------------------------------
# 9) resultados.json
# ---------------------------------------------------------------------------
RESULTADOS = {
    "subtarea": "T3",
    "descripcion": (
        "Análisis por consulta: fallos (relevante fuera del top 3) de TF-IDF (T1) y LSA(4) (T2) "
        "y términos compartidos consulta–documento relevante según el vocabulario TF-IDF"
    ),
    "criterio_de_fallo": "posición del documento relevante > 3",
    "top_k": TOP_K,
    "fuentes": {
        "TF-IDF": ORIGEN["TF-IDF"],
        "LSA(4)": ORIGEN["LSA(4)"],
        "verificacion_contra_recomputacion": VERIF,
        "vocabulario_tamano": VOCAB_TAM,
        "vocabulario_coincide_con_T1": CHECK_VOCAB,
    },
    "posicion_del_relevante": {m: {q: int(POS[m][q]) for q in QUERY_IDS} for m in METODOS},
    "consultas_fallidas": {
        m: {
            "consultas": FALLOS[m],
            "n_fallos": len(FALLOS[m]),
            "detalle": {q: {"relevante": RELEVANTE[q], "posicion": int(POS[m][q])} for q in FALLOS[m]},
        }
        for m in METODOS
    },
    "metricas_verificacion": METRICAS,
    "analisis_de_fallos": ANALISIS_FALLOS,
    "tabla_terminos_compartidos": TABLA_TERMINOS,
    "caso_q5_d09": CASO_Q5,
    "figuras": FIGURAS,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(RESULTADOS, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 10) Resumen
# ---------------------------------------------------------------------------
def si_no(b):
    return "sí" if b else "no"


print("=" * 72)
print("T3 — Análisis por consulta (fallos y términos compartidos)")
print("=" * 72)
print("Vocabulario TF-IDF: %d términos (coincide con T1: %s)" % (VOCAB_TAM, CHECK_VOCAB))
for m in METODOS:
    print("")
    print("[%s] fuente: %s" % (m, ORIGEN[m]))
    print("  posición del relevante: " + ", ".join("%s:%d" % (q, POS[m][q]) for q in QUERY_IDS))
    if FALLOS[m]:
        print("  CONSULTAS FALLIDAS (relevante fuera del top 3): " + ", ".join(FALLOS[m]))
        for qid in FALLOS[m]:
            rel = RELEVANTE[qid]
            filas, _, _, _ = terminos_compartidos(qid, rel)
            if filas:
                lista = ", ".join(
                    "'%s' (q=%.4f, %s=%.4f, IDF=%.4f, DF=%d)"
                    % (f["termino"], f["peso_consulta_tfidf"], rel,
                       f["peso_documento_tfidf"], f["idf"], f["df"])
                    for f in filas
                )
                print("    %s vs %s (pos %d): %s" % (qid, rel, POS[m][qid], lista))
            else:
                print("    %s vs %s (pos %d): sin términos compartidos" % (qid, rel, POS[m][qid]))
    else:
        print("  sin fallos: el relevante está en el top 3 en las 6 consultas")
print("")
print("Caso q5 (relevante d09):")
print("  TF-IDF: posición %d (fallo: %s) | LSA(4): posición %d (fallo: %s)" % (
    POS["TF-IDF"]["q5"], si_no("q5" in FALLOS["TF-IDF"]),
    POS["LSA(4)"]["q5"], si_no("q5" in FALLOS["LSA(4)"]),
))
if filas_q5:
    print("  términos compartidos q5–d09: " + ", ".join(
        "'%s' (q=%.4f, d09=%.4f, IDF=%.4f, DF=%d)"
        % (f["termino"], f["peso_consulta_tfidf"], f["peso_documento_tfidf"], f["idf"], f["df"])
        for f in filas_q5
    ))
else:
    print("  sin términos compartidos")
print("")
print("Figuras: " + ", ".join(FIGURAS))
print("Resultados escritos en resultados.json")
