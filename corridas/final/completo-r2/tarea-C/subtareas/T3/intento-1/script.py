#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Parte 3 de la Tarea C).

A partir de los rankings de los dos métodos de las subtareas previas
  * T1: TF-IDF (TfidfVectorizer por defecto) + similitud coseno
  * T2: LSA (TruncatedSVD n_components=4, random_state=0) + similitud coseno
esta subtarea:
  1) determina en qué consultas falla cada método (criterio del enunciado:
     el documento relevante queda FUERA DEL TOP 3);
  2) para cada consulta —en especial las fallidas, como q5— extrae los
     términos que comparten la consulta y su documento relevante
     (intersección de tokens según el vocabulario de TF-IDF), que constituye
     la evidencia léxica que explica cada fallo.

Salidas (carpeta actual):
  - resultados.json
  - T3_fallos_y_evidencia.png
  - T3_mapa_interseccion.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------------
# 1) Corpus, consultas y juicios de relevancia (idénticos al enunciado: T1/T2)
# ----------------------------------------------------------------------------
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
TOP_K = 3  # criterio de fallo: relevante fuera del top 3

# Palabras funcionales (artículos, preposiciones, pronombres, etc.) que pueden
# aparecer en el vocabulario TF-IDF; solo se usan para separar la evidencia
# de contenido de la evidencia puramente gramatical.
PALABRAS_FUNCIONALES = {
    "de", "del", "el", "la", "los", "las", "un", "una", "al", "con", "en",
    "por", "para", "que", "qué", "como", "su", "es", "entre", "según",
    "cada", "antes", "se", "y", "o", "más", "si", "no", "ya",
}

# ----------------------------------------------------------------------------
# 2) Reproducción del pipeline de T1/T2 (mismos datos, mismos parámetros)
# ----------------------------------------------------------------------------
textos_docs = [DOCUMENTOS[d] for d in DOC_IDS]
textos_cons = [CONSULTAS[q][0] for q in QUERY_IDS]

vectorizer = TfidfVectorizer()  # parámetros por defecto, como en T1
X_docs = vectorizer.fit_transform(textos_docs)
X_cons = vectorizer.transform(textos_cons)
vocabulario = vectorizer.vocabulary_
analizador = vectorizer.build_analyzer()

svd = TruncatedSVD(n_components=4, random_state=0)  # como en T2
X_docs_lsa = svd.fit_transform(X_docs)
X_cons_lsa = svd.transform(X_cons)

S_tfidf = normalize(X_cons).toarray() @ normalize(X_docs).toarray().T
S_lsa = normalize(X_cons_lsa) @ normalize(X_docs_lsa).T


def ranking_de(fila):
    orden = np.argsort(-np.asarray(fila), kind="stable")
    return [DOC_IDS[i] for i in orden]


rank_tfidf_calc = {q: ranking_de(S_tfidf[i]) for i, q in enumerate(QUERY_IDS)}
rank_lsa_calc = {q: ranking_de(S_lsa[i]) for i, q in enumerate(QUERY_IDS)}

# ----------------------------------------------------------------------------
# 3) Rankings de ambos métodos: se toman de las subtareas previas si están
#    disponibles (entrada/T1, entrada/T2) y se verifican contra el recálculo.
# ----------------------------------------------------------------------------
def leer_rankings_previos(carpeta, clave):
    ruta = Path("entrada") / carpeta / "resultados.json"
    if not ruta.exists():
        return None
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return None
    bloque = datos.get(clave) if isinstance(datos, dict) else None
    if not isinstance(bloque, dict):
        return None
    rankings = {}
    for q in QUERY_IDS:
        e = bloque.get(q)
        if (isinstance(e, dict) and isinstance(e.get("ranking"), list)
                and len(e["ranking"]) == len(DOC_IDS)
                and set(map(str, e["ranking"])) == set(DOC_IDS)):
            rankings[q] = [str(x) for x in e["ranking"]]
    return rankings if len(rankings) == len(QUERY_IDS) else None


prev_tfidf = leer_rankings_previos("T1", "rankings_por_consulta")
prev_lsa = leer_rankings_previos("T2", "rankings_por_consulta_lsa")

rank_tfidf = prev_tfidf if prev_tfidf is not None else rank_tfidf_calc
rank_lsa = prev_lsa if prev_lsa is not None else rank_lsa_calc


def verificacion(prev, calc):
    if prev is None:
        return {"fuente": "recalculado con el mismo pipeline (archivo de la subtarea previa no disponible)",
                "ranking_identico_al_previo": None,
                "posicion_del_relevante_identica_al_previo": None}
    exacto = all(prev[q] == calc[q] for q in QUERY_IDS)
    pos_igual = all(prev[q].index(CONSULTAS[q][1]) == calc[q].index(CONSULTAS[q][1]) for q in QUERY_IDS)
    return {"fuente": "entrada/<subtarea>/resultados.json (rankings de la subtarea previa)",
            "ranking_identico_al_previo": bool(exacto),
            "posicion_del_relevante_identica_al_previo": bool(pos_igual)}


verif_tfidf = verificacion(prev_tfidf, rank_tfidf_calc)
verif_lsa = verificacion(prev_lsa, rank_lsa_calc)

# ----------------------------------------------------------------------------
# 4) Posición del relevante y consultas fallidas (fuera del top 3)
# ----------------------------------------------------------------------------
pos_tfidf = {q: rank_tfidf[q].index(CONSULTAS[q][1]) + 1 for q in QUERY_IDS}
pos_lsa = {q: rank_lsa[q].index(CONSULTAS[q][1]) + 1 for q in QUERY_IDS}
fallos_tfidf = [q for q in QUERY_IDS if pos_tfidf[q] > TOP_K]
fallos_lsa = [q for q in QUERY_IDS if pos_lsa[q] > TOP_K]

# ----------------------------------------------------------------------------
# 5) Evidencia léxica: intersección de tokens consulta ∩ documento relevante,
#    filtrada por el vocabulario de TF-IDF
# ----------------------------------------------------------------------------
tokens_docs = {d: analizador(DOCUMENTOS[d]) for d in DOC_IDS}


def unicos(ts):
    return list(dict.fromkeys(ts))


evidencia = {}
toks_q_vocab = {}
for q in QUERY_IDS:
    texto_q, rel = CONSULTAS[q]
    toks_q_todos = analizador(texto_q)
    toks_q = unicos([t for t in toks_q_todos if t in vocabulario])
    toks_q_vocab[q] = toks_q
    toks_rel = unicos(tokens_docs[rel])
    inter = unicos([t for t in toks_q if t in set(toks_rel)])
    inter_cont = [t for t in inter if t not in PALABRAS_FUNCIONALES]
    inter_func = [t for t in inter if t in PALABRAS_FUNCIONALES]
    oov = unicos([t for t in toks_q_todos if t not in vocabulario])
    oov_cont = [t for t in oov if t not in PALABRAS_FUNCIONALES]
    ausentes = unicos([t for t in tokens_docs[rel]
                       if t not in set(toks_q_todos) and t not in PALABRAS_FUNCIONALES])
    evidencia[q] = {
        "consulta": texto_q,
        "relevante": rel,
        "tokens_consulta_en_vocabulario": toks_q,
        "tokens_relevante_en_vocabulario": toks_rel,
        "interseccion": inter,
        "n_interseccion": len(inter),
        "interseccion_terminos_de_contenido": inter_cont,
        "n_interseccion_contenido": len(inter_cont),
        "interseccion_palabras_funcionales": inter_func,
        "tokens_consulta_fuera_del_vocabulario": oov,
        "tokens_consulta_fuera_del_vocabulario_de_contenido": oov_cont,
        "terminos_de_contenido_del_relevante_ausentes_en_la_consulta": ausentes,
    }

# Matriz completa de intersecciones consulta × documento (contexto del fallo)
M_inter = np.zeros((len(QUERY_IDS), len(DOC_IDS)), dtype=int)
for i, q in enumerate(QUERY_IDS):
    sq = set(toks_q_vocab[q])
    for j, d in enumerate(DOC_IDS):
        M_inter[i, j] = len(sq & set(tokens_docs[d]))

# ----------------------------------------------------------------------------
# 6) Rankings completos y detalle de los fallos
# ----------------------------------------------------------------------------
def bloque_rankings(ranks, sims, posiciones, fallos):
    out = {}
    for i, q in enumerate(QUERY_IDS):
        sim_map = {DOC_IDS[j]: float(sims[i, j]) for j in range(len(DOC_IDS))}
        out[q] = {
            "consulta": CONSULTAS[q][0],
            "relevante": CONSULTAS[q][1],
            "ranking": ranks[q],
            "similitudes_en_orden_del_ranking": [sim_map[d] for d in ranks[q]],
            "posicion_relevante": posiciones[q],
            "fallo_fuera_top3": q in fallos,
        }
    return out


rankings_json = {
    "tfidf": bloque_rankings(rank_tfidf, S_tfidf, pos_tfidf, fallos_tfidf),
    "lsa": bloque_rankings(rank_lsa, S_lsa, pos_lsa, fallos_lsa),
}


def detalle_de_fallos(nombre, ranks, sims, posiciones, fallos):
    det = {}
    for q in fallos:
        texto_q, rel = CONSULTAS[q]
        i = QUERY_IDS.index(q)
        ev = evidencia[q]
        sim_map = {DOC_IDS[j]: float(sims[i, j]) for j in range(len(DOC_IDS))}
        top3 = []
        for d in ranks[q][:TOP_K]:
            inter_qd = unicos([t for t in toks_q_vocab[q] if t in set(tokens_docs[d])])
            top3.append({
                "doc": d,
                "similitud": sim_map[d],
                "terminos_compartidos_con_la_consulta": inter_qd,
                "n_terminos_compartidos": len(inter_qd),
            })
        superiores = "; ".join(
            f"{t['doc']} (comparte: {', '.join(t['terminos_compartidos_con_la_consulta']) if t['terminos_compartidos_con_la_consulta'] else 'nada'})"
            for t in top3
        )
        partes = [
            f"Fallo de {nombre} en {q}: el documento relevante {rel} queda en la posición {posiciones[q]} de 10, fuera del top {TOP_K}.",
            f"Evidencia léxica: la consulta y {rel} comparten {ev['n_interseccion']} término(s) del vocabulario TF-IDF "
            f"({', '.join(ev['interseccion']) if ev['interseccion'] else 'ninguno'}), de los cuales "
            f"{ev['n_interseccion_contenido']} son de contenido "
            f"({', '.join(ev['interseccion_terminos_de_contenido']) if ev['interseccion_terminos_de_contenido'] else 'ninguno'}).",
        ]
        if ev["tokens_consulta_fuera_del_vocabulario_de_contenido"]:
            partes.append(
                "Los términos de contenido de la consulta ("
                + ", ".join(ev["tokens_consulta_fuera_del_vocabulario_de_contenido"])
                + ") no aparecen en ningún documento (están fuera del vocabulario TF-IDF): la consulta parafrasea el concepto sin usar el léxico del relevante."
            )
        if ev["terminos_de_contenido_del_relevante_ausentes_en_la_consulta"]:
            partes.append(
                "Los términos de contenido del relevante ("
                + ", ".join(ev["terminos_de_contenido_del_relevante_ausentes_en_la_consulta"])
                + ") no aparecen en la consulta."
            )
        partes.append(f"Documentos que desplazan al relevante dentro del top 3 → {superiores}.")
        det[q] = {
            "posicion_relevante": posiciones[q],
            "top3": top3,
            "terminos_compartidos_con_el_relevante": ev["interseccion"],
            "explicacion": " ".join(partes),
        }
    return det


detalle_fallos = {
    "tfidf": detalle_de_fallos("TF-IDF", rank_tfidf, S_tfidf, pos_tfidf, fallos_tfidf),
    "lsa": detalle_de_fallos("LSA", rank_lsa, S_lsa, pos_lsa, fallos_lsa),
}

tabla_resumen = []
for q in QUERY_IDS:
    ev = evidencia[q]
    tabla_resumen.append({
        "consulta": q,
        "relevante": ev["relevante"],
        "terminos_compartidos": ev["interseccion"],
        "n_terminos_compartidos": ev["n_interseccion"],
        "n_terminos_de_contenido_compartidos": ev["n_interseccion_contenido"],
        "posicion_relevante_tfidf": pos_tfidf[q],
        "fallo_tfidf": q in fallos_tfidf,
        "posicion_relevante_lsa": pos_lsa[q],
        "fallo_lsa": q in fallos_lsa,
    })

# ----------------------------------------------------------------------------
# 7) Conclusión
# ----------------------------------------------------------------------------
var_lsa = float(svd.explained_variance_ratio_.sum())
fallidas_union = sorted(set(fallos_tfidf) | set(fallos_lsa))
contenidos_ok = [evidencia[q]["n_interseccion_contenido"] for q in QUERY_IDS if q not in fallidas_union]
conclusion = (
    "Con el criterio del enunciado (documento relevante fuera del top 3), TF-IDF falla en "
    f"{', '.join(fallos_tfidf) if fallos_tfidf else 'ninguna consulta'} y LSA en "
    f"{', '.join(fallos_lsa) if fallos_lsa else 'ninguna consulta'}. "
    "La evidencia léxica lo explica: q5 es la única consulta cuyo solape con su documento relevante (d09) se reduce a un "
    "término del vocabulario TF-IDF, el artículo 'la' (0 términos de contenido). La consulta parafrasea el concepto —más "
    "determinista al elegir la siguiente palabra— sin usar el léxico de d09 (temperatura, reescala, logits, softmax, "
    "valores bajos, concentran, probabilidad), y sus términos de contenido (determinista, elegir, siguiente, palabra, "
    "hace, sea) ni siquiera pertenecen al vocabulario. Sin coincidencias léxicas de contenido, TF-IDF puntúa d09 casi "
    "solo por palabras funcionales, y LSA no lo rescata: su semántica latente se aprende de las co-ocurrencias de la "
    f"misma matriz TF-IDF con solo 10 documentos (4 componentes explican el {var_lsa:.1%} de la varianza), insuficiente "
    "para relacionar ambos campos léxicos. En las consultas que sí aciertan, la consulta y el relevante comparten "
    f"entre {min(contenidos_ok)} y {max(contenidos_ok)} términos de contenido (p. ej. q1: función, ranking, léxica, "
    "pondera, frecuencia, términos), lo que coloca al relevante dentro del top 3 en ambos métodos."
)

# ----------------------------------------------------------------------------
# 8) Figuras
# ----------------------------------------------------------------------------
idx_fallidas = [QUERY_IDS.index(q) for q in fallidas_union]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
x = np.arange(len(QUERY_IDS))
ancho = 0.38
for ax in axes:
    for idx in idx_fallidas:
        ax.axvspan(idx - 0.5, idx + 0.5, color="crimson", alpha=0.07, zorder=0)

pos_t = [pos_tfidf[q] for q in QUERY_IDS]
pos_l = [pos_lsa[q] for q in QUERY_IDS]
b1 = axes[0].bar(x - ancho / 2, pos_t, ancho, color="#4c72b0", label="TF-IDF (T1)")
b2 = axes[0].bar(x + ancho / 2, pos_l, ancho, color="#55a868", label="LSA (T2)")
axes[0].axhline(TOP_K + 0.5, ls="--", color="gray", lw=1)
axes[0].text(len(QUERY_IDS) - 0.45, TOP_K + 0.62, "umbral de fallo (fuera del top 3)",
             ha="right", va="bottom", fontsize=8, color="dimgray")
for barras in (b1, b2):
    for b in barras:
        axes[0].annotate(str(int(b.get_height())),
                         (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha="center", va="bottom", fontsize=8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(QUERY_IDS)
axes[0].set_ylabel("posición del documento relevante")
axes[0].set_xlabel("consulta")
axes[0].set_ylim(0, max(pos_t + pos_l) + 1.2)
axes[0].set_title("Posición del relevante (fallo si > 3)")
axes[0].legend(fontsize=8, loc="upper left")

n_tot = [evidencia[q]["n_interseccion"] for q in QUERY_IDS]
n_con = [evidencia[q]["n_interseccion_contenido"] for q in QUERY_IDS]
b3 = axes[1].bar(x - ancho / 2, n_tot, ancho, color="#4c72b0", label="términos compartidos (total)")
b4 = axes[1].bar(x + ancho / 2, n_con, ancho, color="#c44e52", label="solo términos de contenido")
for barras in (b3, b4):
    for b in barras:
        axes[1].annotate(str(int(b.get_height())),
                         (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha="center", va="bottom", fontsize=8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(QUERY_IDS)
axes[1].set_ylabel("n.º de términos compartidos con el relevante")
axes[1].set_xlabel("consulta")
axes[1].set_ylim(0, max(n_tot) + 1.2)
axes[1].set_title("Evidencia léxica: consulta ∩ relevante (vocabulario TF-IDF)")
axes[1].legend(fontsize=8, loc="upper right")

fig.suptitle("T3 — Fallos por método y evidencia léxica (banda roja: consultas fallidas)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("T3_fallos_y_evidencia.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.5, 4.8))
im = ax.imshow(M_inter, cmap="Blues", aspect="auto", vmin=0)
ax.set_xticks(range(len(DOC_IDS)))
ax.set_xticklabels(DOC_IDS)
ax.set_yticks(range(len(QUERY_IDS)))
ax.set_yticklabels(QUERY_IDS)
maximo = M_inter.max()
for i in range(len(QUERY_IDS)):
    for j in range(len(DOC_IDS)):
        v = int(M_inter[i, j])
        ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                color="white" if maximo > 0 and v > 0.6 * maximo else "black")
for i, q in enumerate(QUERY_IDS):
    j = DOC_IDS.index(CONSULTAS[q][1])
    fallo = (q in fallos_tfidf) and (q in fallos_lsa)
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                           edgecolor="#d62728" if fallo else "#2ca02c",
                           lw=2.2, ls="-" if fallo else "--"))
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("n.º de términos compartidos (vocabulario TF-IDF)")
ax.set_xlabel("documento")
ax.set_ylabel("consulta")
ax.set_title("T3 — Intersección de tokens consulta × documento\n"
             "(recuadro = documento relevante; rojo continuo: fuera del top 3 en ambos métodos)")
fig.tight_layout()
fig.savefig("T3_mapa_interseccion.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 9) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T3 — Análisis por consulta: consultas fallidas por método y evidencia léxica (términos compartidos consulta–documento relevante)",
    "criterio_fallo": "el documento relevante queda fuera del top 3 (posición >= 4 en el ranking de 10 documentos)",
    "metodo": ("Rankings de TF-IDF (T1) y LSA (T2); intersección de tokens entre cada consulta y su documento relevante, "
               "filtrada por el vocabulario de TfidfVectorizer (parámetros por defecto, ajustado solo con los 10 documentos; "
               "LSA: TruncatedSVD(n_components=4, random_state=0)). Términos de contenido = tokens que no son palabras funcionales."),
    "parametros": {"tfidf": "TfidfVectorizer() por defecto", "lsa": "TruncatedSVD(n_components=4, random_state=0)", "top_k": TOP_K},
    "n_documentos": len(DOC_IDS),
    "n_consultas": len(QUERY_IDS),
    "tam_vocabulario_tfidf": len(vocabulario),
    "varianza_explicada_lsa_4_componentes": var_lsa,
    "verificacion_con_subtareas_previas": {"tfidf_T1": verif_tfidf, "lsa_T2": verif_lsa},
    "consultas_fallidas": {"tfidf": fallos_tfidf, "lsa": fallos_lsa},
    "n_consultas_fallidas": {"tfidf": len(fallos_tfidf), "lsa": len(fallos_lsa)},
    "posicion_relevante_por_consulta": {"tfidf": pos_tfidf, "lsa": pos_lsa},
    "rankings_por_consulta": rankings_json,
    "terminos_compartidos_por_consulta": evidencia,
    "matriz_interseccion_consulta_documento": {"filas": QUERY_IDS, "columnas": DOC_IDS, "valores": M_inter.tolist()},
    "detalle_fallos": detalle_fallos,
    "tabla_resumen": tabla_resumen,
    "conclusion": conclusion,
    "figuras": ["T3_fallos_y_evidencia.png", "T3_mapa_interseccion.png"],
}

Path("resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------------------------------------------------------------------
# 10) Resumen por consola
# ----------------------------------------------------------------------------
print("=" * 78)
print("T3 — Análisis por consulta: fallos y evidencia léxica")
print("=" * 78)
print(f"Vocabulario TF-IDF: {len(vocabulario)} términos | varianza LSA (4 comp.): {var_lsa:.1%}")
print(f"Verificación T1 (rankings TF-IDF): {verif_tfidf}")
print(f"Verificación T2 (rankings LSA):    {verif_lsa}")
print(f"Consultas fallidas TF-IDF (relevante fuera del top 3): {fallos_tfidf if fallos_tfidf else 'ninguna'}")
print(f"Consultas fallidas LSA     (relevante fuera del top 3): {fallos_lsa if fallos_lsa else 'ninguna'}")
print("-" * 78)
print(f"{'consulta':<10}{'rel.':<7}{'compartidos':<14}{'contenido':<12}{'pos TF-IDF':<13}{'pos LSA':<8}")
for q in QUERY_IDS:
    ev = evidencia[q]
    print(f"{q:<10}{ev['relevante']:<7}{ev['n_interseccion']:<14}{ev['n_interseccion_contenido']:<12}"
          f"{pos_tfidf[q]:<13}{pos_lsa[q]:<8}")
print("-" * 78)
for q in fallidas_union:
    print(f"Evidencia {q} (relevante {evidencia[q]['relevante']}) — intersección: {evidencia[q]['interseccion']}")
    for metodo in ("tfidf", "lsa"):
        if q in detalle_fallos[metodo]:
            print(f"  [{metodo}] {detalle_fallos[metodo][q]['explicacion']}")
print("-" * 78)
print("Archivos escritos: resultados.json, T3_fallos_y_evidencia.png, T3_mapa_interseccion.png")
