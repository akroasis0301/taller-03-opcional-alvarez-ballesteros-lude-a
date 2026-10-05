```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subtarea T3 — Parte 3: análisis por consulta.

(1) A partir de los rankings de T1 (TF-IDF) y T2 (LSA-4) identifica en qué
    consultas falla cada método (documento relevante fuera del top 3).
(2) Para cada consulta fallida —en particular q5 con d09— extrae los términos
    que comparten la consulta y su documento relevante según el vocabulario
    TF-IDF de la Parte 1, con sus pesos en la consulta y en el documento, IDF
    y DF, y redacta la explicación del fallo.

Entradas: entrada/T1/resultados.json y entrada/T2/resultados.json (rankings
oficiales de T1/T2; si no estuvieran disponibles se recomputan con los
parámetros exactos del enunciado).
Salidas : resultados.json y figuras PNG en la carpeta actual.
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

# -----------------------------------------------------------------------------
# 1) Corpus, consultas y juicios de relevancia (idénticos al enunciado)
# -----------------------------------------------------------------------------
DOC_IDS = [f"d{i:02d}" for i in range(1, 11)]
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
QUERY_IDS = [f"q{i}" for i in range(1, 7)]
QUERIES = {
    "q1": "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "q2": "¿Cómo se añaden fragmentos recuperados al prompt?",
    "q3": "¿Qué técnica entrena matrices de bajo rango?",
    "q4": "¿Cómo se compara la orientación de dos vectores de texto?",
    "q5": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "q6": "¿Qué arquitectura combina autoatención con capas feed-forward?",
}
RELEVANTES = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}
TOP_K = 3
METODOS = ["TF-IDF", "LSA(4)"]

# Lista cerrada de palabras funcionales (solo etiqueta términos al redactar la
# explicación; no interviene en ningún cálculo ni en el vectorizador).
FUNCIONALES = {
    "el", "la", "los", "las", "del", "al", "que", "qué", "cómo", "como", "con", "de",
    "en", "y", "o", "u", "para", "por", "su", "un", "una", "es", "sea", "son", "más",
    "se", "lo", "cada", "según", "ante", "bajo", "entre", "sin", "sobre", "hacia",
}

# -----------------------------------------------------------------------------
# 2) Recomputación de referencia con los parámetros exactos de T1/T2
# -----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, igual que en T1
X = vectorizer.fit_transform([DOCS[d] for d in DOC_IDS])   # solo los 10 documentos
Q = vectorizer.transform([QUERIES[q] for q in QUERY_IDS])  # consultas transformadas
S_TFIDF = cosine_similarity(Q, X)  # (6, 10)

svd = TruncatedSVD(n_components=4, random_state=0)  # igual que en T2
L_D = svd.fit_transform(X)   # (10, 4)
L_Q = svd.transform(Q)       # folding-in de las consultas: (6, 4)
S_LSA = cosine_similarity(L_Q, L_D)  # (6, 10)


def ranking_de_fila(fila):
    """Orden descendente por similitud; empates resueltos por orden original."""
    orden = np.argsort(-np.asarray(fila), kind="stable")
    return [DOC_IDS[i] for i in orden]


RANK_REC = {
    "TF-IDF": {q: ranking_de_fila(S_TFIDF[i]) for i, q in enumerate(QUERY_IDS)},
    "LSA(4)": {q: ranking_de_fila(S_LSA[i]) for i, q in enumerate(QUERY_IDS)},
}
SIM_REC = {
    "TF-IDF": {q: {d: float(S_TFIDF[i, j]) for j, d in enumerate(DOC_IDS)} for i, q in enumerate(QUERY_IDS)},
    "LSA(4)": {q: {d: float(S_LSA[i, j]) for j, d in enumerate(DOC_IDS)} for i, q in enumerate(QUERY_IDS)},
}

# -----------------------------------------------------------------------------
# 3) Rankings oficiales de T1/T2 (con recomputación de respaldo y verificación)
# -----------------------------------------------------------------------------
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
    ORIGEN[m] = (
        f"entrada/{carpeta}/resultados.json ({len(fr)}/6 consultas)"
        if fr
        else "recomputación local con los parámetros del enunciado"
    )
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
            FILE_RANK[m][q]["relevante"] in (None, RELEVANTES[q]) for q in comp
        ),
    }

VOCAB_TAM = len(vectorizer.vocabulary_)
CHECK_VOCAB = None
if isinstance(T1, dict) and isinstance(T1.get("vocabulario_tamano"), int):
    CHECK_VOCAB = bool(T1["vocabulario_tamano"] == VOCAB_TAM)

# -----------------------------------------------------------------------------
# 4) Posiciones del relevante, fallos (fuera del top 3) y métricas
# -----------------------------------------------------------------------------
def posicion_de(rank, rel):
    return rank.index(rel) + 1 if rel in rank else len(rank) + 1


POS = {m: {q: posicion_de(RANK[m][q], RELEVANTES[q]) for q in QUERY_IDS} for m in METODOS}
FALLOS = {m: [q for q in QUERY_IDS if POS[m][q] > TOP_K] for m in METODOS}

METRICAS = {}
for m in METODOS:
    METRICAS[m] = {
        "Hit@1": sum(1 for q in QUERY_IDS if POS[m][q] == 1) / len(QUERY_IDS),
        "Hit@3": sum(1 for q in QUERY_IDS if POS[m][q] <= TOP_K) / len(QUERY_IDS),
        "MRR": float(np.mean([1.0 / POS[m][q] for q in QUERY_IDS])),
    }

# -----------------------------------------------------------------------------
# 5) Términos compartidos consulta–documento relevante (vocabulario TF-IDF)
# -----------------------------------------------------------------------------
ANALYZER = vectorizer.build_analyzer()
VOCAB = vectorizer.vocabulary_
Xd = X.toarray()
Qd = Q.toarray()
DF_COUNTS = (Xd > 0).sum(axis=0)
IDX_Q = {q: i for i, q in enumerate(QUERY_IDS)}
IDX_D = {d: i for i, d in enumerate(DOC_IDS)}


def terminos_compartidos(qid, did):
    toks_q = ANALYZER(QUERIES[qid])
    toks_d = ANALYZER(DOCS[did])
    sq, sd = set(toks_q), set(toks_d)
    filas = []
    for t in sorted(sq & sd):
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
    ausentes_en_consulta = sorted(sd - sq)
    return filas, en_vocab, fuera_vocab, ausentes_en_consulta


def terminos_comunes(qid, did):
    return sorted(set(ANALYZER(QUERIES[qid])) & set(ANALYZER(DOCS[did])))


def acotar(lista, n=8):
    lista = list(lista)
    if len(lista) <= n:
        return ", ".join(lista)
    return ", ".join(lista[:n]) + ", …"


def fmt_terminos(filas, rel):
    return "; ".join(
        f"'{f['termino']}' (peso en la consulta={f['peso_consulta_tfidf']:.4f}, "
        f"peso en {rel}={f['peso_documento_tfidf']:.4f}, IDF={f['idf']:.4f}, DF={f['df']}/10)"
        for f in filas
    )


def construir_explicacion(metodo, qid, rel, filas, pos, sim_rel, top3, pos_otro, ausentes, var_exp):
    if filas:
        base = (
            f"La consulta {qid} y su relevante {rel} comparten {len(filas)} término(s) del vocabulario TF-IDF: "
            + fmt_terminos(filas, rel) + "."
        )
    else:
        base = f"La consulta {qid} y su relevante {rel} no comparten ningún término del vocabulario TF-IDF."
    top3_txt = "; ".join(
        f"{t['documento']} (pos {t['posicion']}, sim {t['similitud']:.4f}, "
        f"comparte: {acotar(t['terminos_compartidos_con_consulta'], 6) or '—'})"
        for t in top3
    )
    contenido = [f["termino"] for f in filas if f["tipo"] == "contenido"]
    if contenido:
        if metodo == "LSA(4)":
            causa = (
                f" El fallo de {metodo} en {qid} no se debe a falta de solapamiento léxico —hay términos de contenido "
                f"compartidos ({acotar(contenido, 8)})— sino a la proyección latente a 4 dimensiones: con solo 10 documentos y "
                f"{var_exp:.1%} de varianza explicada, las componentes latentes mezclan temas y diluyen el match léxico fuerte; "
                f"la similitud de {rel} ({sim_rel:.4f}) cae a la posición {pos}, fuera del top 3 ({top3_txt}). "
                f"TF-IDF, que sí explota el match léxico directo, colocaba a {rel} en la posición {pos_otro}."
            )
        else:
            causa = (
                f" Aunque hay solapamiento de contenido ({acotar(contenido, 8)}), otros documentos comparten con la consulta "
                f"términos de contenido más frecuentes o más específicos y los superan: top 3: {top3_txt}."
            )
    else:
        if filas:
            causa = (
                f" Todos los términos compartidos son palabras funcionales de IDF baja "
                f"(DF hasta {max(f['df'] for f in filas)}/10), sin capacidad discriminativa: "
                f"no hay solapamiento de términos de contenido."
            )
        else:
            causa = " No existe solapamiento léxico de ningún tipo."
        causa += (
            f" Los términos de contenido de la consulta no aparecen en {rel} y los términos característicos de {rel} "
            f"({acotar(ausentes, 8)}) no aparecen en la consulta: la relación consulta–documento es puramente semántica "
            f"(una paráfrasis), invisible para el matching léxico."
        )
        if metodo == "TF-IDF":
            causa += (
                f" La similitud coseno de {rel} ({sim_rel:.4f}) solo recoge el cruce de palabras funcionales y queda por "
                f"debajo de la de documentos que comparten más (o más específicas) palabras funcionales con la consulta — "
                f"top 3: {top3_txt} —, de modo que {rel} cae a la posición {pos}, fuera del top 3."
            )
        else:
            causa += (
                f" LSA solo puede acercar vectores mediante co-ocurrencias observadas en el corpus; al no co-ocurrir esos "
                f"términos en los 10 documentos, la proyección a 4 componentes no salva la paráfrasis y {rel} queda en la "
                f"posición {pos} (top 3: {top3_txt})."
            )
    return base + causa

# -----------------------------------------------------------------------------
# 6) Registro detallado de cada fallo
# -----------------------------------------------------------------------------
var_exp = float(svd.explained_variance_ratio_.sum())
ANALISIS_FALLOS = []
TABLA_TERMINOS = []
for m in METODOS:
    otro = "LSA(4)" if m == "TF-IDF" else "TF-IDF"
    for qid in FALLOS[m]:
        rel = RELEVANTES[qid]
        pos = POS[m][qid]
        sim_rel = float(SIMS[m][qid][rel])
        rank = RANK[m][qid]
        top3 = [
            {
                "posicion": i + 1,
                "documento": rank[i],
                "similitud": float(SIMS[m][qid][rank[i]]),
                "terminos_compartidos_con_consulta": terminos_comunes(qid, rank[i]),
            }
            for i in range(min(TOP_K, len(rank)))
        ]
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

TERM_POR_FALLO = {m: {} for m in METODOS}
for rec in ANALISIS_FALLOS:
    TERM_POR_FALLO[rec["metodo"]][rec["consulta_id"]] = rec["terminos_compartidos_consulta_relevante"]

# -----------------------------------------------------------------------------
# 7) Caso destacado: q5 con d09
# -----------------------------------------------------------------------------
filas_q5, en_vocab_q5, fuera_q5, ausentes_d09 = terminos_compartidos("q5", "d09")
compartidos_q5 = ", ".join(f"'{f['termino']}'" for f in filas_q5) if filas_q5 else "ninguno"
if filas_q5:
    if len(filas_q5) == 1 and filas_q5[0]["tipo"] == "funcional":
        puente = "El único puente léxico es una palabra funcional, sin poder discriminativo."
    elif not [f for f in filas_q5 if f["tipo"] == "contenido"]:
        puente = "Todos los términos compartidos son palabras funcionales, sin poder discriminativo."
    else:
        puente = "Hay términos de contenido compartidos."
    lectura_q5 = (
        f"q5 y d09 comparten {len(filas_q5)} término del vocabulario TF-IDF: {compartidos_q5} ("
        + "; ".join(
            f"'{f['termino']}': peso en la consulta={f['peso_consulta_tfidf']:.4f}, "
            f"peso en d09={f['peso_documento_tfidf']:.4f}, IDF={f['idf']:.4f}, DF={f['df']}/10"
            for f in filas_q5
        )
        + f"). {puente} "
        f"De los términos de la consulta presentes en el vocabulario ({acotar(en_vocab_q5, 10)}), "
        f"solo {compartidos_q5} aparece(n) en d09 —el resto tiene peso 0 en d09—; además, "
        f"{len(fuera_q5)} término(s) de la consulta ni siquiera pertenecen al vocabulario "
        f"({acotar(fuera_q5, 10)}), y los términos que definen a d09 ({acotar(ausentes_d09, 10)}) "
        f"no aparecen en la consulta: 'más determinista' es una paráfrasis de 'la temperatura baja "
        f"concentra la probabilidad'. Sin solapamiento de contenido, TF-IDF puntúa a d09 solo por el "
        f"cruce de palabras funcionales y lo deja en la posición {POS['TF-IDF']['q5']} "
        f"(similitud {SIMS['TF-IDF']['q5']['d09']:.4f}), fuera del top 3. "
    )
    if POS["LSA(4)"]["q5"] <= TOP_K:
        lectura_q5 += (
            f"LSA(4) sí mitiga el hueco semántico en esta consulta: d09 sube a la posición "
            f"{POS['LSA(4)']['q5']} (similitud {SIMS['LSA(4)']['q5']['d09']:.4f})."
        )
    else:
        lectura_q5 += (
            f"LSA(4) tampoco lo rescata (posición {POS['LSA(4)']['q5']}, similitud "
            f"{SIMS['LSA(4)']['q5']['d09']:.4f}): con solo 10 documentos, los términos de la paráfrasis "
            f"nunca co-ocurren y la proyección a 4 componentes no puede acercarlos."
        )
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

# -----------------------------------------------------------------------------
# 8) Figuras
# -----------------------------------------------------------------------------
FIGURAS = []

# (a) posición del relevante por consulta y método
fig, ax = plt.subplots(figsize=(8.2, 4.4))
xpos = np.arange(len(QUERY_IDS))
w = 0.38
vals_tf = [POS["TF-IDF"][q] for q in QUERY_IDS]
vals_lsa = [POS["LSA(4)"][q] for q in QUERY_IDS]
b1 = ax.bar(xpos - w / 2, vals_tf, w, label="TF-IDF (T1)", color="#4878CF")
b2 = ax.bar(xpos + w / 2, vals_lsa, w, label
