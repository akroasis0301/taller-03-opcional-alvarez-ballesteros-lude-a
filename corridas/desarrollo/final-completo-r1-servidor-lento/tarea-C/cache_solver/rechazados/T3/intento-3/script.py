```python
# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Tarea C, MMIA 6013).

Reconstruye la línea base TF-IDF (TfidfVectorizer por defecto) y la proyección
LSA (TruncatedSVD(n_components=4, random_state=0)) exactamente como en T1/T2,
identifica en qué consultas falla cada método (documento relevante fuera del
top 3) y extrae del vocabulario TF-IDF los términos que comparte cada consulta
con su documento relevante y con los documentos que lo superan en el ranking.

Salidas en la carpeta actual:
  - resultados.json                       (contrato de la subtarea)
  - T3_posicion_relevante_y_terminos.png  (posición del relevante + solapamiento)
  - T3_terminos_compartidos_q5.png        (caso crítico q5)
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# 1) Corpus, consultas y juicios (datos literales del enunciado, como en T1/T2)
# ---------------------------------------------------------------------------
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
consultas = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(documentos)
query_ids = list(consultas)
relevante = {q: consultas[q][1] for q in query_ids}
TOP_K = 3

# ---------------------------------------------------------------------------
# 2) Línea base TF-IDF (parámetros por defecto, idéntica a T1)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform([documentos[d] for d in doc_ids])
X_q = vectorizer.transform([consultas[q][0] for q in query_ids])

vocab = [str(t) for t in vectorizer.get_feature_names_out()]
vocab_set = set(vocab)
col_de = vectorizer.vocabulary_
idf = {t: float(vectorizer.idf_[col_de[t]]) for t in vocab}

sims_tfidf = (normalize(X_q) @ normalize(X_docs).T).toarray()  # coseno, 6x10

# ---------------------------------------------------------------------------
# 3) LSA: TruncatedSVD a 4 dimensiones (idéntico a T2)
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
L_docs = normalize(svd.fit_transform(X_docs))
L_q = normalize(svd.transform(X_q))
sims_lsa = L_q @ L_docs.T  # coseno en el espacio latente, 6x10

# ---------------------------------------------------------------------------
# 4) Rankings y posición del documento relevante
# ---------------------------------------------------------------------------
def ranking_de(fila):
    orden = np.argsort(-np.asarray(fila, dtype=float).ravel(), kind="stable")
    return [doc_ids[i] for i in orden]

rank_tfidf = {q: ranking_de(sims_tfidf[i]) for i, q in enumerate(query_ids)}
rank_lsa = {q: ranking_de(sims_lsa[i]) for i, q in enumerate(query_ids)}
pos_tfidf = {q: rank_tfidf[q].index(relevante[q]) + 1 for q in query_ids}
pos_lsa = {q: rank_lsa[q].index(relevante[q]) + 1 for q in query_ids}

fallos_tfidf = [q for q in query_ids if pos_tfidf[q] > TOP_K]
fallos_lsa = [q for q in query_ids if pos_lsa[q] > TOP_K]

# ---------------------------------------------------------------------------
# 5) Términos compartidos consulta-documento (tokenización de TfidfVectorizer)
# ---------------------------------------------------------------------------
PATRON = re.compile(r"(?u)\b\w\w+\b")  # token_pattern por defecto de sklearn

def tokenizar(texto):
    return PATRON.findall(texto.lower())

tokens_doc = {d: set(tokenizar(documentos[d])) for d in doc_ids}
tokens_q = {q: set(tokenizar(consultas[q][0])) for q in query_ids}

# Palabras funcionales presentes en el vocabulario (lista manual; solo se usa
# para separar 'contenido' de 'función' al leer los resultados).
FUNCIONALES = {
    "al", "antes", "cada", "como", "con", "de", "del", "el", "en", "entre",
    "es", "la", "los", "para", "por", "que", "qué", "según", "su", "un", "una",
}

def compartidos(q, d):
    return sorted(tokens_q[q] & tokens_doc[d] & vocab_set)

analisis = {}
for i, q in enumerate(query_ids):
    rel = relevante[q]
    j = doc_ids.index(rel)
    t_rel = compartidos(q, rel)
    t_cont = [t for t in t_rel if t not in FUNCIONALES]
    sup_tfidf = rank_tfidf[q][: pos_tfidf[q] - 1]
    sup_lsa = rank_lsa[q][: pos_lsa[q] - 1]
    analisis[q] = {
        "consulta": consultas[q][0],
        "documento_relevante": rel,
        "posicion_tfidf": pos_tfidf[q],
        "posicion_lsa": pos_lsa[q],
        "terminos_consulta_en_vocabulario_tfidf": sorted(tokens_q[q] & vocab_set),
        "terminos_consulta_fuera_del_vocabulario": sorted(tokens_q[q] - vocab_set),
        "terminos_compartidos_con_relevante": t_rel,
        "n_terminos_compartidos_con_relevante": len(t_rel),
        "terminos_compartidos_de_contenido_con_relevante": t_cont,
        "n_terminos_compartidos_de_contenido_con_relevante": len(t_cont),
        "detalle_terminos_compartidos_con_relevante": [
            {
                "termino": t,
                "idf": idf[t],
                "peso_tfidf_en_consulta": float(X_q[i, col_de[t]]),
                "peso_tfidf_en_relevante": float(X_docs[j, col_de[t]]),
            }
            for t in t_rel
        ],
        "similitud_coseno_tfidf_consulta_relevante": float(sims_tfidf[i, j]),
        "similitud_coseno_lsa_consulta_relevante": float(sims_lsa[i, j]),
        "documentos_que_superan_al_relevante_tfidf": sup_tfidf,
        "terminos_compartidos_con_los_que_lo_superan_tfidf": {d: compartidos(q, d) for d in sup_tfidf},
        "documentos_que_superan_al_relevante_lsa": sup_lsa,
        "terminos_compartidos_con_los_que_lo_superan_lsa": {d: compartidos(q, d) for d in sup_lsa},
    }

for q in query_ids:
    a = analisis[q]
    n_t = a["n_terminos_compartidos_con_relevante"]
    n_c = a["n_terminos_compartidos_de_contenido_con_relevante"]
    lista_c = ", ".join(a["terminos_compartidos_de_contenido_con_relevante"]) or "ninguno"
    base = (f"Comparte {n_t} término(s) del vocabulario TF-IDF con {a['documento_relevante']} "
            f"({n_c} de contenido: {lista_c}).")
    if a["posicion_tfidf"] > TOP_K or a["posicion_lsa"] > TOP_K:
        base += (" FALLO: el relevante queda fuera del top 3 "
                 f"(posición {max(a['posicion_tfidf'], a['posicion_lsa'])} en el peor método).")
    elif a["posicion_lsa"] > a["posicion_tfidf"]:
        base += f" Sin fallo, pero LSA degrada al relevante a la posición {a['posicion_lsa']} (límite del top 3)."
    else:
        base += " Sin fallo: el relevante está en el top 3 en ambos métodos."
    a["lectura"] = base

# ---------------------------------------------------------------------------
# 6) Caso q5 (única consulta que falla en los dos métodos)
# ---------------------------------------------------------------------------
a5 = analisis["q5"]
analisis_q5 = {
    "consulta": consultas["q5"][0],
    "documento_relevante": "d09",
    "texto_del_relevante": documentos["d09"],
    "terminos_compartidos_q5_d09": a5["terminos_compartidos_con_relevante"],
    "terminos_de_contenido_compartidos_q5_d09": a5["terminos_compartidos_de_contenido_con_relevante"],
    "posicion_d09_tfidf": pos_tfidf["q5"],
    "posicion_d09_lsa": pos_lsa["q5"],
    "documentos_que_superan_a_d09": {
        d: {
            "posicion_tfidf": rank_tfidf["q5"].index(d) + 1,
            "posicion_lsa": rank_lsa["q5"].index(d) + 1,
            "terminos_compartidos_con_q5": a5["terminos_compartidos_con_los_que_lo_superan_tfidf"][d],
        }
        for d in a5["documentos_que_superan_al_relevante_tfidf"]
    },
    "explicacion_del_fallo": (
        "q5 comparte con d09 un único término del vocabulario TF-IDF: el artículo «la» (cero términos de "
        "contenido). La consulta plantea la idea con «determinista», palabra que no aparece en d09; d09 la "
        "expresa con «temperatura», «softmax» y «probabilidad». Los documentos que superan a d09 comparten "
        "más términos de la consulta: d03 comparte «la» y además «modelo» (término de contenido), d06 "
        "comparte «el» y «qué», d04 comparte «la» y «que», d01 comparte «el» y «la», y d10 comparte «la». "
        "Al basarse en el solapamiento léxico, TF-IDF puntúa esos documentos por encima de d09 y lo deja "
        "en la posición 6."
    ),
    "por_que_lsa_tampoco_lo_recupera": (
        "LSA con 4 componentes latentes, aprendidas a partir de solo 10 documentos breves, no logra asociar "
        "«determinista» con el tema de d09: el corpus no aporta las co-ocurrencias necesarias para suplir la "
        "ausencia de solapamiento léxico y d09 permanece en la posición 6 también en el espacio latente."
    ),
}

# ---------------------------------------------------------------------------
# 7) Tabla de fallos por consulta y resumen
# ---------------------------------------------------------------------------
sup_q4 = analisis["q4"]["documentos_que_superan_al_relevante_lsa"]
det_q4 = "; ".join(
    f"{d} comparte {', '.join(analisis['q4']['terminos_compartidos_con_los_que_lo_superan_lsa'][d]) or 'nada'}"
    for d in sup_q4
)
notas = {
    "q4": (f"En LSA el relevante degrada de la posición 1 a la 3: sigue dentro del top 3 (no es fallo), pero "
           f"queda en el límite; los que lo superan ({', '.join(sup_q4)}) solo comparten palabras funcionales "
           f"con la consulta ({det_q4})."),
    "q5": ("Fallo en los dos métodos: d09 queda en la posición 6; es la única consulta sin ningún término de "
           "contenido compartido con su relevante (solo el artículo «la»)."),
}

tabla_fallos = [
    {
        "consulta": q,
        "texto": consultas[q][0],
        "relevante": relevante[q],
        "posicion_tfidf": pos_tfidf[q],
        "fallo_tfidf": bool(pos_tfidf[q] > TOP_K),
        "posicion_lsa": pos_lsa[q],
        "fallo_lsa": bool(pos_lsa[q] > TOP_K),
        "cambio_posicion_lsa_menos_tfidf": int(pos_lsa[q] - pos_tfidf[q]),
        "nota": notas.get(q, ""),
    }
    for q in query_ids
]

resumen_fallos = {
    "fallan_ambos_metodos": [q for q in query_ids if q in fallos_tfidf and q in fallos_lsa],
    "falla_solo_tfidf": [q for q in query_ids if q in fallos_tfidf and q not in fallos_lsa],
    "falla_solo_lsa": [q for q in query_ids if q not in fallos_tfidf and q in fallos_lsa],
    "no_falla_ningun_metodo": [q for q in query_ids if q not in fallos_tfidf and q not in fallos_lsa],
}

# ---------------------------------------------------------------------------
# 8) Métricas de contexto (reproducen T1/T2) y coherencia con subtareas previas
# ---------------------------------------------------------------------------
def metricas_de(posiciones):
    n = len(posiciones)
    return {
        "Hit@1": sum(1 for p in posiciones.values() if p == 1) / n,
        "Hit@3": sum(1 for p in posiciones.values() if p <= TOP_K) / n,
        "MRR": sum(1.0 / p for p in posiciones.values()) / n,
    }

metricas_tfidf = metricas_de(pos_tfidf)
metricas_lsa = metricas_de(pos_lsa)

sims_tfidf_por_consulta = {
    q: {d: float(sims_tfidf[i, j]) for j, d in enumerate(doc_ids)}
    for i, q in enumerate(query_ids)
}
sims_lsa_por_consulta = {
    q: {d: float(sims_lsa[i, j]) for j, d in enumerate(doc_ids)}
    for i, q in enumerate(query_ids)
}

def cargar_previo(sid):
    p = Path("entrada") / sid / "resultados.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

coherencia = {}
prev_t1 = cargar_previo("T1")
if prev_t1 is not None:
    coherencia["T1"] = {
        "vocabulario_igual": prev_t1.get("vocabulario") == vocab,
        "rankings_tfidf_iguales": prev_t1.get("rankings_por_consulta") == rank_tfidf,
        "posiciones_iguales": prev_t1.get("posicion_del_relevante_por_consulta") == pos_tfidf,
    }
prev_t2 = cargar_previo("T2")
if prev_t2 is not None:
    coherencia["T2"] = {
        "rankings_tfidf_iguales": prev_t2.get("rankings_tfidf_por_consulta") == rank_tfidf,
        "rankings_lsa_iguales": prev_t2.get("rankings_lsa_por_consulta") == rank_lsa,
        "posiciones_tfidf_iguales": prev_t2.get("posicion_del_relevante_tfidf_por_consulta") == pos_tfidf,
        "posiciones_lsa_iguales": prev_t2.get("posicion_del_relevante_lsa_por_consulta") == pos_lsa,
    }
if not coherencia:
    coherencia = {"nota": "entrada/T1 y entrada/T2 no disponibles; todo se recalculó desde los datos del enunciado"}

# ---------------------------------------------------------------------------
# 9) Figuras
# ---------------------------------------------------------------------------
# Figura 1: posición del relevante por método + nº de términos compartidos
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
x = np.arange(len(query_ids))
w = 0.38
b1 = ax.bar(x - w / 2, [pos_tfidf[q] for q in query_ids], w, label="TF-IDF + coseno", color="#4C72B0")
b2 = ax.bar(x + w / 2, [pos_lsa[q] for q in query_ids], w, label="LSA (4D) + coseno", color="#DD8452")
ax.axhline(TOP_K, color="crimson", ls="--", lw=1.3)
ax.axhspan(TOP_K, 7.6, color="crimson", alpha=0.07)
ax.text(-0.4, 3.25, "umbral top-3", color="crimson", fontsize=9, ha="left", va="bottom")
ax.text(2.3, 6.9, "zona de fallo (posición > 3)", color="crimson", fontsize=9, ha="left", va="center")
for barras in (b1, b2):
    for r in barras:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.12, f"{int(r.get_height())}",
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(query_ids)
ax.set_ylim(0, 7.6)
ax.set_ylabel("Posición del documento relevante")
ax.set_title("Posición del relevante por consulta (fallo si > 3)")
ax.legend(loc="upper left", fontsize=9)

ax = axes[1]
counts = [analisis[q]["n_terminos_compartidos_con_relevante"] for q in query_ids]
colores = ["#C44E52" if analisis[q]["n_terminos_compartidos_de_contenido_con_relevante"] == 0 else "#55A868"
           for q in query_ids]
barras = ax.bar(query_ids, counts, color=colores)
for r, c in zip(barras, counts):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.1, str(c), ha="center", fontsize=9)
ax.set_ylabel("Nº de términos compartidos con el relevante")
ax.set_title("Términos del vocabulario TF-IDF compartidos consulta–relevante")
ax.set_ylim(0, max(counts) + 1.8)
ax.legend(handles=[Patch(color="#55A868", label="comparte términos de contenido"),
                   Patch(color="#C44E52", label="solo términos funcionales")],
          loc="upper right", fontsize=8)

fig.tight_layout()
fig.suptitle("T3 — Análisis por consulta: fallos y solapamiento léxico", y=1.02, fontsize=12)
fig.savefig("T3_posicion_relevante_y_terminos.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Figura 2: q5 — términos compartidos con cada documento del ranking
fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.axis("off")
cols = ["pos. TF-IDF", "pos. LSA", "doc", "¿relevante?", "términos compartidos con q5"]
celdas = []
for d in rank_tfidf["q5"]:
    celdas.append([
        str(rank_tfidf["q5"].index(d) + 1),
        str(rank_lsa["q5"].index(d) + 1),
        d,
        "sí" if d == relevante["q5"] else "-",
        ", ".join(compartidos("q5", d)) or "-",
    ])
tabla = ax.table(cellText=celdas, colLabels=cols, cellLoc="center", loc="center",
                 bbox=[0.02, 0.10, 0.96, 0.76])
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
for j in range(len(cols)):
    tabla[0, j].set_facecolor("#D9D9D9")
    tabla[0, j].set_text_props(weight="bold")
fila_rel = rank_tfidf["q5"].index(relevante["q5"]) + 1
for j in range(len(cols)):
    tabla[fila_rel, j].set_facecolor("#F5C6C6")
    tabla[fila_rel, j].set_text_props(weight="bold")
ax.set_title("q5: «" + consultas["q5"][0] + "»   (relevante: d09)\n
