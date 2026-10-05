# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Análisis por consulta (corpus mínimo de 10 documentos y 6 consultas).

A partir de los rankings de los dos métodos de las partes previas
  · TF-IDF + similitud coseno       (T1: TfidfVectorizer por defecto, ajustado SOLO con los 10 documentos)
  · LSA de 4 dimensiones + coseno   (T2: TruncatedSVD(n_components=4, random_state=0))
esta subtarea:
  1) identifica en qué consultas falla cada método (documento relevante fuera del
     top 3), guardando la posición del relevante en cada caso;
  2) calcula, para cada consulta, qué términos comparte la consulta con su documento
     relevante (solapamiento léxico sobre el vocabulario TF-IDF), con pesos TF-IDF
     por término y una clasificación auxiliar funcional/contenido;
  3) genera una explicación de los fallos a partir de ese solapamiento.

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
DOC_IDS = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]
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
Q_IDS = ["q1", "q2", "q3", "q4", "q5", "q6"]
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
vec = TfidfVectorizer()  # idéntico a T1: parámetros por defecto, ajustado solo con los 10 documentos
X_docs = vec.fit_transform([DOCUMENTOS[d] for d in DOC_IDS])
X_cons = vec.transform([CONSULTAS[q] for q in Q_IDS])
vocab = [str(t) for t in vec.get_feature_names_out()]
vocab_set = set(vocab)
term_index = {t: j for j, t in enumerate(vocab)}
Xd = X_docs.toarray()
Xq = X_cons.toarray()
S_tfidf = cosine_similarity(X_cons, X_docs)

svd = TruncatedSVD(n_components=4, random_state=0)  # idéntico a T2
D_lat = svd.fit_transform(X_docs)
Q_lat = svd.transform(X_cons)
S_lsa_recalc = cosine_similarity(Q_lat, D_lat)


def rankings_desc(S):
    """Ranking descendente por similitud; empates resueltos por orden del corpus d01..d10."""
    return [[DOC_IDS[j] for j in np.argsort(-S[i], kind="stable")] for i in range(S.shape[0])]


def pos_de(doc, ranking):
    return ranking.index(doc) + 1


R_tfidf = rankings_desc(S_tfidf)
pos_tfidf = {q: pos_de(RELEVANTE[q], R_tfidf[i]) for i, q in enumerate(Q_IDS)}

# ---------------------------------------------------------------------------
# 3) Verificación contra los resultados oficiales de T1 y T2 (si están disponibles)
# ---------------------------------------------------------------------------
def cargar(ruta):
    p = Path(ruta)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


t1 = cargar("entrada/T1/resultados.json")
t2 = cargar("entrada/T2/resultados.json")
verif = {
    "T1": {"archivo": "entrada/T1/resultados.json" if t1 is not None else "no_encontrado"},
    "T2": {"archivo": "entrada/T2/resultados.json" if t2 is not None else "no_encontrado"},
}

S_lsa = S_lsa_recalc
fuente_lsa = "recalculado aqui con TruncatedSVD(n_components=4, random_state=0)"

if t2 is not None:
    try:
        coords = t2["lsa"]["coordenadas_latentes"]
        D2 = np.array([coords["documentos"][d] for d in DOC_IDS], dtype=float)
        Q2 = np.array([coords["consultas"][q] for q in Q_IDS], dtype=float)
        S2 = cosine_similarity(Q2, D2)
        verif["T2"]["similitud_lsa_coincide_con_recalculo"] = bool(np.allclose(S2, S_lsa_recalc, atol=1e-6))
        verif["T2"]["max_dif_similitud_lsa"] = float(np.max(np.abs(S2 - S_lsa_recalc)))
        S_lsa = S2  # se usan las coordenadas oficiales de T2 para garantizar consistencia
        fuente_lsa = "coordenadas latentes oficiales leidas de entrada/T2/resultados.json"
    except Exception as exc:
        verif["T2"]["error_leyendo_coordenadas"] = str(exc)

if t1 is not None:
    try:
        M1 = np.array(t1["matriz_similitud_coseno"]["valores"], dtype=float)
        verif["T1"]["matriz_similitud_tfidf_coincide"] = bool(
            M1.shape == S_tfidf.shape and np.allclose(M1, S_tfidf, atol=1e-8))
        filas1 = {str(f["consulta"]): f for f in t1["por_consulta"]}
        verif["T1"]["rankings_completos_coinciden"] = bool(
            all(q in filas1 and list(filas1[q]["ranking_completo"]) == R_tfidf[i]
                for i, q in enumerate(Q_IDS)))
        verif["T1"]["ranks_del_relevante_coinciden"] = bool(
            all(q in filas1 and int(filas1[q]["rank_del_relevante"]) == pos_tfidf[q] for q in Q_IDS))
    except Exception as exc:
        verif["T1"]["error"] = str(exc)

R_lsa = rankings_desc(S_lsa)
pos_lsa = {q: pos_de(RELEVANTE[q], R_lsa[i]) for i, q in enumerate(Q_IDS)}

# ---------------------------------------------------------------------------
# 4) Análisis de fallos: relevante fuera del top 3 en cada método
# ---------------------------------------------------------------------------
def lista_fallos(pos, R, S, metodo):
    out = []
    for i, q in enumerate(Q_IDS):
        if pos[q] > TOP_K:
            rel = RELEVANTE[q]
            out.append({
                "metodo": metodo,
                "consulta": q,
                "texto_consulta": CONSULTAS[q],
                "relevante": rel,
                "posicion_del_relevante": int(pos[q]),
                "top3_del_metodo": list(R[i][:TOP_K]),
                "sim_coseno_top1": float(S[i, DOC_IDS.index(R[i][0])]),
                "sim_coseno_relevante": float(S[i, DOC_IDS.index(rel)]),
            })
    return out


fallos_tfidf = lista_fallos(pos_tfidf, R_tfidf, S_tfidf, "tfidf")
fallos_lsa = lista_fallos(pos_lsa, R_lsa, S_lsa, "lsa")
fallan_tfidf = [f["consulta"] for f in fallos_tfidf]
fallan_lsa = [f["consulta"] for f in fallos_lsa]
fallan_ambos = sorted(set(fallan_tfidf) & set(fallan_lsa))

# ---------------------------------------------------------------------------
# 5) Solapamiento léxico consulta <-> documento relevante sobre el vocabulario TF-IDF
# ---------------------------------------------------------------------------
analyzer = vec.build_analyzer()
FUNCIONALES = {"un", "una", "el", "la", "los", "al", "del", "de", "en", "con", "para",
               "por", "que", "qué", "como", "cada", "su", "es", "o", "y", "se", "a"}

solapamiento = {}
for i, q in enumerate(Q_IDS):
    rel = RELEVANTE[q]
    toks_q = analyzer(CONSULTAS[q])
    toks_d = analyzer(DOCUMENTOS[rel])
    sq = set(t for t in toks_q if t in vocab_set)
    sd = set(t for t in toks_d if t in vocab_set)
    compartidos = sorted(sq & sd, key=lambda t: -float(Xd[DOC_IDS.index(rel), term_index[t]]))
    union = sq | sd
    detalle = []
    for t in compartidos:
        j = term_index[t]
        detalle.append({
            "termino": t,
            "tipo": "funcional" if t in FUNCIONALES else "contenido",
            "peso_tfidf_en_consulta": float(Xq[i, j]),
            "peso_tfidf_en_documento": float(Xd[DOC_IDS.index(rel), j]),
        })
    solapamiento[q] = {
        "consulta": q,
        "texto_consulta": CONSULTAS[q],
        "relevante": rel,
        "texto_relevante": DOCUMENTOS[rel],
        "terminos_consulta_en_vocabulario": sorted(sq, key=lambda t: term_index[t]),
        "terminos_documento_en_vocabulario": sorted(sd, key=lambda t: term_index[t]),
        "terminos_consulta_fuera_del_vocabulario": sorted(set(toks_q) - vocab_set),
        "terminos_compartidos": compartidos,
        "detalle_por_termino": detalle,
        "n_compartidos": len(compartidos),
        "n_compartidos_contenido": sum(1 for f in detalle if f["tipo"] == "contenido"),
        "n_compartidos_funcionales": sum(1 for f in detalle if f["tipo"] == "funcional"),
        "terminos_contenido_compartidos": [f["termino"] for f in detalle if f["tipo"] == "contenido"],
        "terminos_funcionales_compartidos": [f["termino"] for f in detalle if f["tipo"] == "funcional"],
        "jaccard_vocabulario": (len(compartidos) / len(union)) if union else 0.0,
        "posicion_relevante_tfidf": int(pos_tfidf[q]),
        "posicion_relevante_lsa": int(pos_lsa[q]),
        "falla_tfidf": bool(pos_tfidf[q] > TOP_K),
        "falla_lsa": bool(pos_lsa[q] > TOP_K),
    }

# ---------------------------------------------------------------------------
# 6) Explicación de los fallos a partir del solapamiento
# ---------------------------------------------------------------------------
explicaciones = {}
for q in Q_IDS:
    ov = solapamiento[q]
    nc = ov["n_compartidos_contenido"]
    nf = ov["n_compartidos_funcionales"]
    cont = ", ".join(ov["terminos_contenido_compartidos"]) if nc else "(ninguno)"
    partes = []
    partes.append(
        f"{q} vs {ov['relevante']}: comparten {ov['n_compartidos']} termino(s) del vocabulario "
        f"TF-IDF ({nc} de contenido: {cont}; {nf} funcionales)."
    )
    if ov["terminos_consulta_fuera_del_vocabulario"]:
        partes.append(
            "Terminos clave de la consulta ausentes del vocabulario del corpus: "
            + ", ".join(ov["terminos_consulta_fuera_del_vocabulario"]) + "."
        )
    if ov["falla_tfidf"] and ov["falla_lsa"]:
        if nc == 0:
            partes.append(
                f"FALLO en ambos metodos (posicion {ov['posicion_relevante_tfidf']} en TF-IDF y "
                f"{ov['posicion_relevante_lsa']} en LSA): consulta y relevante no comparten NINGUN "
                "termino de contenido del vocabulario, solo palabras funcionales, de modo que ni la "
                "coincidencia literal ni la estructura latente elevan el relevante al top 3."
            )
        else:
            partes.append(
                f"FALLO en ambos metodos (posicion {ov['posicion_relevante_tfidf']} en TF-IDF y "
                f"{ov['posicion_relevante_lsa']} en LSA): el solapamiento de contenido es debil "
                "frente a otros documentos del corpus."
            )
    elif ov["falla_tfidf"]:
        partes.append(
            f"FALLO solo en TF-IDF (posicion {ov['posicion_relevante_tfidf']}); LSA coloca el "
            f"relevante en {ov['posicion_relevante_lsa']} apoyandose en coocurrencias latentes."
        )
    elif ov["falla_lsa"]:
        partes.append(
            f"FALLO solo en LSA (posicion {ov['posicion_relevante_lsa']}); TF-IDF coloca el "
            f"relevante en {ov['posicion_relevante_tfidf']} gracias a la coincidencia literal de terminos."
        )
    else:
        partes.append(
            f"Sin fallos: el relevante queda en el top 3 en ambos metodos (TF-IDF pos "
            f"{ov['posicion_relevante_tfidf']}, LSA pos {ov['posicion_relevante_lsa']}), apoyado en "
            f"{nc} termino(s) de contenido compartido(s)."
        )
    explicaciones[q] = " ".join(partes)

# ---------------------------------------------------------------------------
# 7) Figura 1: posición del relevante por consulta y método
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(Q_IDS))
w = 0.38
b1 = ax.bar(x - w / 2, [pos_tfidf[q] for q in Q_IDS], w, label="TF-IDF + coseno", color="#4C72B0")
b2 = ax.bar(x + w / 2, [pos_lsa[q] for q in Q_IDS], w, label="LSA (4 dim) + coseno", color="#DD8452")
ax.axhline(TOP_K + 0.5, color="crimson", ls="--", lw=1.2)
ax.text(len(Q_IDS) - 0.45, TOP_K + 0.65, "umbral de fallo (fuera del top 3)",
        color="crimson", fontsize=8, ha="right")
for barras in (b1, b2):
    for r in barras:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.08,
                str(int(r.get_height())), ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(Q_IDS)
ax.set_ylim(0, 10.8)
ax.set_yticks(range(1, 11))
ax.set_xlabel("Consulta")
ax.set_ylabel("Posición del documento relevante")
ax.set_title("T3 · Posición del relevante por consulta y método (fallo si queda fuera del top 3)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("T3_posicion_relevante.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8) Figura 2: solapamiento léxico consulta <-> relevante
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.5]})
nc = [solapamiento[q]["n_compartidos_contenido"] for q in Q_IDS]
nf = [solapamiento[q]["n_compartidos_funcionales"] for q in Q_IDS]
ax1.bar(x, nc, 0.6, label="términos de contenido", color="#55A868")
ax1.bar(x, nf, 0.6, bottom=nc, label="términos funcionales", color="#CB7EA7")
for i, q in enumerate(Q_IDS):
    total = nc[i] + nf[i]
    ax1.text(i, total + 0.12, str(total), ha="center", fontsize=9)
    marcas = []
    if solapamiento[q]["falla_tfidf"]:
        marcas.append("TF-IDF")
    if solapamiento[q]["falla_lsa"]:
        marcas.append("LSA")
    if marcas:
        ax1.text(i, total + 0.55, "fallo: " + "+".join(marcas), ha="center",
                 fontsize=8, color="crimson")
ax1.set_xticks(x)
ax1.set_xticklabels(Q_IDS)
ax1.set_ylim(0, max(nc[i] + nf[i] for i in range(len(Q_IDS))) + 1.8)
ax1.set_xlabel("Consulta")
ax1.set_ylabel("Nº de términos compartidos con el relevante")
ax1.set_title("Solapamiento léxico consulta <-> documento relevante")
ax1.legend(loc="upper right", fontsize=8)

ax2.axis("off")
lineas = []
for q in Q_IDS:
    ov = solapamiento[q]
    terms = ", ".join(ov["terminos_compartidos"]) if ov["terminos_compartidos"] else "(ninguno)"
    e_t = f"TF-IDF pos {ov['posicion_relevante_tfidf']}" + (" FALLA" if ov["falla_tfidf"] else "")
    e_l = f"LSA pos {ov['posicion_relevante_lsa']}" + (" FALLA" if ov["falla_lsa"] else "")
    lineas.append(f"{q} -> {ov['relevante']}  [{e_t} | {e_l}]")
    lineas.append(f"    compartidos ({ov['n_compartidos']}): {terms}")
ax2.text(0.0, 1.0, "\n".join(lineas), va="top", ha="left", family="monospace", fontsize=9)
ax2.set_title("Términos compartidos (vocabulario TF-IDF) y estado por método", fontsize=11)
fig.tight_layout()
fig.savefig("T3_solapamiento_terminos.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 9) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T3_analisis_por_consulta",
    "descripcion": (
        "Parte 3: identificación de las consultas en las que falla cada método "
        "(documento relevante fuera del top 3, con su posición) y solapamiento léxico "
        "consulta-documento relevante sobre el vocabulario TF-IDF, como base de la "
        "explicación de los fallos."
    ),
    "configuracion": {
        "n_documentos": 10,
        "n_consultas": 6,
        "top_k": TOP_K,
        "criterio_de_fallo": "el documento relevante queda fuera del top 3 del ranking del método",
        "vectorizador": "TfidfVectorizer (parámetros por defecto), ajustado solo con los 10 documentos",
        "tam_vocabulario": len(vocab),
        "lsa": {
            "modelo": "TruncatedSVD",
            "n_componentes": 4,
            "random_state": 0,
            "fuente_coordenadas": fuente_lsa,
        },
        "desempate_ranking": "argsort estable descendente (empates por orden del corpus d01..d10)",
    },
    "verificacion_con_previas": verif,
    "rankings": {
        "tfidf": {
            q: {
                "ranking_completo": R_tfidf[i],
                "relevante": RELEVANTE[q],
                "posicion_del_relevante": int(pos_tfidf[q]),
            }
            for i, q in enumerate(Q_IDS)
        },
        "lsa": {
            q: {
                "ranking_completo": R_lsa[i],
                "relevante": RELEVANTE[q],
                "posicion_del_relevante": int(pos_lsa[q]),
            }
            for i, q in enumerate(Q_IDS)
        },
    },
    "consultas_falladas": {
        "tfidf": fallos_tfidf,
        "lsa": fallos_lsa,
        "ambos_metodos": fallan_ambos,
        "resumen": {
            "n_fallos_tfidf": len(fallos_tfidf),
            "n_fallos_lsa": len(fallos_lsa),
            "lista_falladas_tfidf": fallan_tfidf,
            "lista_falladas_lsa": fallan_lsa,
            "lista_falladas_ambos": fallan_ambos,
        },
    },
    "solapamiento_lexico_consulta_relevante": solapamiento,
    "explicaciones_por_consulta": explicaciones,
}

Path("resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ---------------------------------------------------------------------------
# 10) Resumen por consola
# ---------------------------------------------------------------------------
print("=" * 70)
print("T3 - Parte 3: Análisis por consulta (fallo = relevante fuera del top 3)")
print("=" * 70)
print(f"Vocabulario TF-IDF: {len(vocab)} términos | Verificación T1: "
      f"{verif['T1'].get('ranks_del_relevante_coinciden', 's/d')} | "
      f"LSA coincide con T2: {verif['T2'].get('similitud_lsa_coincide_con_recalculo', 's/d')}")
print("-" * 70)
for q in Q_IDS:
    ov = solapamiento[q]
    print(f"{q} -> {ov['relevante']} | TF-IDF pos {ov['posicion_relevante_tfidf']}"
          f"{' (FALLA)' if ov['falla_tfidf'] else ''} | LSA pos {ov['posicion_relevante_lsa']}"
          f"{' (FALLA)' if ov['falla_lsa'] else ''} | compartidos ({ov['n_compartidos']}): "
          f"{', '.join(ov['terminos_compartidos']) if ov['terminos_compartidos'] else '(ninguno)'}")
print("-" * 70)
print(f"Fallan en TF-IDF: {fallan_tfidf if fallan_tfidf else '(ninguna)'}")
print(f"Fallan en LSA:    {fallan_lsa if fallan_lsa else '(ninguna)'}")
print(f"Fallan en ambos:  {fallan_ambos if fallan_ambos else '(ninguna)'}")
print("Salidas: resultados.json, T3_posicion_relevante.png, T3_solapamiento_terminos.png")
