# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Tarea C · MMIA 6013).

Compara, consulta a consulta, los rankings de TF-IDF (línea base de T1) y de
LSA (TruncatedSVD k=4, random_state=0, de T2) para:
  1) listar qué consultas falla cada método (documento relevante fuera del
     top-3) y en qué posición queda el relevante;
  2) extraer, con el tokenizador del propio TfidfVectorizer, los términos que
     comparte cada consulta con su documento relevante y con los documentos
     mal posicionados (los que quedan por encima del relevante), de modo que
     los fallos queden explicados con datos.

Se reconstruye EXACTAMENTE la configuración de T1/T2 (mismos textos del
enunciado, TfidfVectorizer por defecto ajustado SOLO con los 10 documentos,
SVD k=4 con random_state=0; las consultas solo se transforman/proyectan) y se
verifica el resultado contra entrada/T1/resultados.json y entrada/T2/resultados.json.

Salidas: resultados.json · t3_posicion_del_relevante.png · t3_terminos_compartidos.png
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------- datos del enunciado (verbatim)
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
QUERIES = {  # id -> (texto, documento relevante)
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
DOC_IDS = list(DOCS)
Q_IDS = list(QUERIES)
TOP_K = 3  # criterio del enunciado: fallo si el relevante queda fuera del top-3

# ----------------------------------------------------------------- 1) reproducir T1 y T2
vec = TfidfVectorizer()  # parámetros por defecto, ajustado solo con los 10 documentos (como T1)
X_docs = vec.fit_transform([DOCS[d] for d in DOC_IDS])
X_q = vec.transform([QUERIES[q][0] for q in Q_IDS])
S_tfidf = cosine_similarity(X_q, X_docs)

svd = TruncatedSVD(n_components=4, random_state=0)  # como T2
L_docs = svd.fit_transform(X_docs)
L_q = svd.transform(X_q)
S_lsa = cosine_similarity(L_q, L_docs)


def ranking_de(fila):
    """Orden de documentos por similitud descendente; empates en orden original (como T1)."""
    return [DOC_IDS[j] for j in np.argsort(-fila, kind="stable")]


rank_tfidf = {q: ranking_de(S_tfidf[i]) for i, q in enumerate(Q_IDS)}
rank_lsa = {q: ranking_de(S_lsa[i]) for i, q in enumerate(Q_IDS)}
ranks_rel_tfidf = {q: rank_tfidf[q].index(QUERIES[q][1]) + 1 for q in Q_IDS}
ranks_rel_lsa = {q: rank_lsa[q].index(QUERIES[q][1]) + 1 for q in Q_IDS}


def metricas(ranks):
    r = np.array([ranks[q] for q in Q_IDS], dtype=float)
    return {"hit@1": float(np.mean(r == 1)),
            "hit@3": float(np.mean(r <= TOP_K)),
            "mrr": float(np.mean(1.0 / r))}


met_tfidf, met_lsa = metricas(ranks_rel_tfidf), metricas(ranks_rel_lsa)
var_lsa = float(svd.explained_variance_ratio_.sum())

# ----------------------------------------------------------------- 2) análisis de términos compartidos
analyzer = vec.build_analyzer()  # misma tokenización que usa el modelo
tok_q = {q: analyzer(QUERIES[q][0]) for q in Q_IDS}
tok_d = {d: analyzer(DOCS[d]) for d in DOC_IDS}
idf_map = {t: float(v) for t, v in zip(vec.get_feature_names_out(), vec.idf_)}

# Lista mínima FIJA de palabras gramaticales (solo para ETIQUETAR términos; no afecta al modelo)
FUNC = {"a","al","ante","antes","como","cómo","con","contra","cada","cuando","de","del","desde","dos","el","ella",
        "ello","ellas","ellos","en","entre","era","eran","es","esa","esas","ese","eso","esos","esta","está","están",
        "estas","este","esto","estos","fue","fueron","ha","han","hasta","hay","la","las","le","les","lo","los","mas",
        "más","me","menos","mi","mis","mucho","muy","nada","ni","no","nos","nuestra","nuestro","o","otra","otras",
        "otro","otros","para","pero","poco","por","porque","qué","que","quien","quienes","se","según","ser","sea",
        "sean","si","sí","sin","sobre","solo","son","su","sus","tal","también","tan","tanto","te","tiene","tienen",
        "toda","todas","todo","todos","tu","tus","un","una","unas","unos","va","van","y","ya","yo","tres","cuatro",
        "tras","mediante"}


def shared(qid, did):
    return sorted(set(tok_q[qid]) & set(tok_d[did]))


def split(terms):
    return ([t for t in terms if t not in FUNC],
            [t for t in terms if t in FUNC])


def fmt(terms):
    return ", ".join(f"'{t}'" for t in terms) if terms else "ninguno"


def docs_sobre_relevante(nombre, qid):
    """Documentos MAL POSICIONADOS: los que quedan por ENCIMA del relevante en ese método."""
    ranking = rank_tfidf[qid] if nombre == "TF-IDF" else rank_lsa[qid]
    S = S_tfidf if nombre == "TF-IDF" else S_lsa
    rel = QUERIES[qid][1]
    qi = Q_IDS.index(qid)
    out = []
    for pos, d in enumerate(ranking[: ranking.index(rel)], start=1):
        inter = shared(qid, d)
        cont, func = split(inter)
        out.append({
            "posicion": pos,
            "documento": d,
            "similitud": float(S[qi, DOC_IDS.index(d)]),
            "terminos_compartidos_con_consulta": inter,
            "terminos_contenido": cont,
            "terminos_funcionales": func,
            "detalle_idf": [{"termino": t, "idf": idf_map[t],
                             "tipo": "funcional" if t in FUNC else "contenido"} for t in inter],
        })
    return out


analisis = {}
for qid in Q_IDS:
    rel = QUERIES[qid][1]
    qi, di = Q_IDS.index(qid), DOC_IDS.index(rel)
    inter = shared(qid, rel)
    cont, func = split(inter)
    q_content = [t for t in dict.fromkeys(tok_q[qid]) if t not in FUNC]
    rel_content_ausente = [t for t in dict.fromkeys(tok_d[rel]) if t not in FUNC and t not in set(tok_q[qid])]
    oov = sorted({t for t in tok_q[qid] if t not in vec.vocabulary_})
    rt, rl = ranks_rel_tfidf[qid], ranks_rel_lsa[qid]
    if rt > TOP_K and rl > TOP_K:
        diag = "fallo en AMBOS métodos"
    elif rt > TOP_K:
        diag = "fallo solo en TF-IDF"
    elif rl > TOP_K:
        diag = "fallo solo en LSA"
    else:
        diag = "sin fallo (relevante dentro del top-3 en ambos métodos)"
    analisis[qid] = {
        "consulta": QUERIES[qid][0],
        "documento_relevante": rel,
        "tokens_consulta": tok_q[qid],
        "tokens_documento_relevante": tok_d[rel],
        "tokens_consulta_fuera_del_vocabulario": oov,
        "terminos_compartidos_con_relevante": inter,
        "n_terminos_compartidos_con_relevante": len(inter),
        "terminos_contenido_compartidos": cont,
        "n_terminos_contenido_compartidos": len(cont),
        "terminos_funcionales_compartidos": func,
        "n_terminos_funcionales_compartidos": len(func),
        "terminos_contenido_del_relevante_ausentes_en_la_consulta": rel_content_ausente,
        "rank_tfidf": rt,
        "rank_lsa": rl,
        "fallo_tfidf": bool(rt > TOP_K),
        "fallo_lsa": bool(rl > TOP_K),
        "similitud_tfidf_del_relevante": float(S_tfidf[qi, di]),
        "similitud_lsa_del_relevante": float(S_lsa[qi, di]),
        "diagnostico": diag,
        "documentos_mal_posicionados": {"TF-IDF": docs_sobre_relevante("TF-IDF", qid),
                                        "LSA": docs_sobre_relevante("LSA", qid)},
    }

# ----------------------------------------------------------------- 3) fallos por método (relevante fuera del top-3)
fallos = {"TF-IDF": [], "LSA": []}
for qid in Q_IDS:
    rel = QUERIES[qid][1]
    for nombre in ("TF-IDF", "LSA"):
        ranking = rank_tfidf[qid] if nombre == "TF-IDF" else rank_lsa[qid]
        pos = ranking.index(rel) + 1
        if pos > TOP_K:
            inter = shared(qid, rel)
            cont, func = split(inter)
            fallos[nombre].append({
                "consulta_id": qid,
                "consulta": QUERIES[qid][0],
                "documento_relevante": rel,
                "posicion_del_relevante": pos,
                "top3_del_metodo": ranking[:TOP_K],
                "terminos_compartidos_con_relevante": inter,
                "n_terminos_compartidos_con_relevante": len(inter),
                "terminos_contenido_compartidos_con_relevante": cont,
                "terminos_funcionales_compartidos_con_relevante": func,
                "documentos_mal_posicionados_sobre_el_relevante": docs_sobre_relevante(nombre, qid),
            })

degradaciones = []
for qid in Q_IDS:
    rt, rl = ranks_rel_tfidf[qid], ranks_rel_lsa[qid]
    if rl > rt and rl <= TOP_K:
        degradaciones.append({"consulta_id": qid, "rank_tfidf": rt, "rank_lsa": rl,
                              "nota": "el relevante sigue dentro del top-3 (no es fallo), pero LSA empeora su posición"})

set_t = {f["consulta_id"] for f in fallos["TF-IDF"]}
set_l = {f["consulta_id"] for f in fallos["LSA"]}
resumen_fallos = {
    "consultas_falladas_tfidf": sorted(set_t),
    "posicion_del_relevante_tfidf": {f["consulta_id"]: f["posicion_del_relevante"] for f in fallos["TF-IDF"]},
    "consultas_falladas_lsa": sorted(set_l),
    "posicion_del_relevante_lsa": {f["consulta_id"]: f["posicion_del_relevante"] for f in fallos["LSA"]},
    "n_fallos_tfidf": len(set_t),
    "n_fallos_lsa": len(set_l),
    "fallo_comun_a_ambos": sorted(set_t & set_l),
    "solo_falla_tfidf": sorted(set_t - set_l),
    "solo_falla_lsa": sorted(set_l - set_t),
}

# ----------------------------------------------------------------- 4) explicaciones sustentadas en los términos
rel5 = QUERIES["q5"][1]
inter5 = shared("q5", rel5)
q5_content = [t for t in dict.fromkeys(tok_q["q5"]) if t not in FUNC]
d09_content = [t for t in dict.fromkeys(tok_d[rel5]) if t not in FUNC]
q5_content_en_corpus = [t for t in q5_content if t in vec.vocabulary_]
q5_content_oov = [t for t in q5_content if t not in vec.vocabulary_]
qi5, di5 = Q_IDS.index("q5"), DOC_IDS.index(rel5)


def resumen_competidores(nombre, qid):
    ranking = rank_tfidf[qid] if nombre == "TF-IDF" else rank_lsa[qid]
    rel = QUERIES[qid][1]
    partes = []
    for pos, d in enumerate(ranking[: ranking.index(rel)], start=1):
        cont, func = split(shared(qid, d))
        if cont:
            partes.append(f"{d} (pos {pos}; contenido: {fmt(cont)})")
        elif func:
            partes.append(f"{d} (pos {pos}; solo funcionales: {fmt(func)})")
        else:
            partes.append(f"{d} (pos {pos}; sin ninguna coincidencia)")
    return "; ".join(partes) if partes else "ninguno"


exp_q5_tfidf = (
    f"Fallo por desajuste de vocabulario. q5 y su relevante d09 comparten un único término, {fmt(inter5)} "
    f"(funcional; idf={idf_map['la']:.3f}): {len(q5_content_oov)} de los {len(q5_content)} términos de contenido de la "
    f"consulta ({fmt(q5_content)}) no aparecen en d09 —y {fmt(q5_content_oov)} no aparecen en NINGÚN documento del "
    f"corpus—, mientras que los {len(d09_content)} términos de contenido de d09 ({fmt(d09_content)}) no aparecen en la "
    f"consulta. La única palabra de contenido de la consulta que existe en el corpus es {fmt(q5_content_en_corpus)} y "
    f"pertenece a d03, no a d09: por eso d03 encabeza el ranking. Los documentos mal posicionados solo coinciden con la "
    f"consulta en funcionales o en una coincidencia incidental: {resumen_competidores('TF-IDF', 'q5')}. Esas pocas "
    f"coincidencias tienen IDF alto por ser raras ('modelo', 'qué' y 'que' aparecen en un solo documento, "
    f"idf={idf_map['modelo']:.3f}) y bastan para dejar a d09 (similitud={float(S_tfidf[qi5, di5]):.3f}, que solo recibe "
    f"el aporte de 'la') en el puesto {ranks_rel_tfidf['q5']}."
)
exp_q5_lsa = (
    f"LSA tampoco corrige el fallo: d09 queda en el puesto {ranks_rel_lsa['q5']} (similitud latente "
    f"{float(S_lsa[qi5, di5]):.3f}). Con k=4 y solo {var_lsa:.1%} de la varianza explicada, el espacio latente solo "
    f"codifica coocurrencias presentes en 10 documentos; como los términos de d09 ({fmt(d09_content)}) nunca coocurren "
    f"con los de la consulta ({fmt(q5_content)}), no existe término puente que los acerque. Documentos que superan a "
    f"d09 en LSA: {resumen_competidores('LSA', 'q5')}. La semántica latente no crea sinonimia donde el corpus no aporta señal."
)
i4_d01, i4_d10 = shared("q4", "d01"), shared("q4", "d10")
exp_q4_lsa = (
    f"Degradación sin fallo técnico: q4 pasa del puesto {ranks_rel_tfidf['q4']} (TF-IDF) al {ranks_rel_lsa['q4']} "
    f"(LSA), sigue dentro del top-3. d01 y d10, que con la consulta solo comparten funcionales ({fmt(i4_d01)} y "
    f"{fmt(i4_d10)}; ningún término de contenido), se colocan por encima de d05 en el espacio latente, que agrupa "
    f"documentos por coocurrencia global (d01: 'similitud', 'consulta'; d10: 'pesos', 'precisión') y no por coincidencia "
    f"literal. Es la suavización semántica de LSA con k=4 ({var_lsa:.1%} de varianza): difumina la señal léxica fuerte "
    f"({fmt(analisis['q4']['terminos_contenido_compartidos'])}) que en TF-IDF colocaba a d05 primero."
)
conteos = ", ".join(f"{q}: {analisis[q]['n_terminos_contenido_compartidos']}" for q in Q_IDS)
exp_general = (
    f"Términos de contenido compartidos consulta–relevante por consulta ({conteos}). Las cinco consultas con solape de "
    f"contenido (q1, q2, q3, q4, q6) se resuelven bien: TF-IDF las coloca 1.ª y LSA también, salvo q4 que baja al puesto "
    f"3 sin salirse del top-3. El único fallo, común a los dos métodos, es q5: consulta y relevante no comparten NINGÚN "
    f"término de contenido (solo el artículo 'la'), un hueco léxico que ni TF-IDF ni un LSA k=4 ajustado con 10 "
    f"documentos puede salvar; un recuperador denso entrenado sí podría ('determinista' ↔ 'temperatura baja/softmax')."
)
explicaciones = {"general": exp_general, "q5_TF-IDF": exp_q5_tfidf, "q5_LSA": exp_q5_lsa,
                 "q4_LSA_degradacion": exp_q4_lsa}

clave = set()
for qid in Q_IDS:
    rel = QUERIES[qid][1]
    clave.update(shared(qid, rel))
    for nombre in ("TF-IDF", "LSA"):
        ranking = rank_tfidf[qid] if nombre == "TF-IDF" else rank_lsa[qid]
        for d in ranking[: ranking.index(rel)]:
            clave.update(shared(qid, d))
clave.update(d09_content)
idf_terminos_clave = {t: idf_map[t] for t in sorted(clave) if t in idf_map}

filas = ["| consulta | relevante | rank TF-IDF | rank LSA | fallo TF-IDF | fallo LSA | contenido compartido con el relevante |",
         "|---|---|---|---|---|---|---|"]
for qid in Q_IDS:
    a = analisis[qid]
    filas.append(f"| {qid} | {a['documento_relevante']} | {a['rank_tfidf']} | {a['rank_lsa']} | "
                 f"{'SÍ' if a['fallo_tfidf'] else 'no'} | {'SÍ' if a['fallo_lsa'] else 'no'} | "
                 f"{', '.join(a['terminos_contenido_compartidos']) or '—'} |")
tabla_md = "\n".join(filas)

# ----------------------------------------------------------------- verificación contra T1/T2
verif = {"T1_ranks_coinciden": None, "T2_ranks_coinciden": None, "detalle": ""}
try:
    t1 = json.loads(Path("entrada", "T1", "resultados.json").read_text(encoding="utf-8"))
    prev = t1.get("ranks_del_documento_relevante", {})
    verif["T1_ranks_coinciden"] = bool(prev) and all(int(prev[q]) == ranks_rel_tfidf[q] for q in Q_IDS)
except Exception as exc:
    verif["detalle"] += f"T1 no verificado ({type(exc).__name__}); "
try:
    t2 = json.loads(Path("entrada", "T2", "resultados.json").read_text(encoding="utf-8"))
    filas_t2 = t2.get("tabla_por_consulta", [])
    verif["T2_ranks_coinciden"] = len(filas_t2) == len(Q_IDS) and all(
        int(f["rank_tfidf"]) == ranks_rel_tfidf[f["consulta_id"]]
        and int(f["rank_lsa"]) == ranks_rel_lsa[f["consulta_id"]] for f in filas_t2)
except Exception as exc:
    verif["detalle"] += f"T2 no verificado ({type(exc).__name__})."

# ----------------------------------------------------------------- 5) figuras
fig, ax = plt.subplots(figsize=(8.4, 4.6))
x = np.arange(len(Q_IDS))
w = 0.38
ax.bar(x - w / 2, [ranks_rel_tfidf[q] for q in Q_IDS], w, label="TF-IDF", color="#4C72B0")
ax.bar(x + w / 2, [ranks_rel_lsa[q] for q in Q_IDS], w, label="LSA (k=4)", color="#DD8452")
ax.axhline(TOP_K + 0.5, color="#C44E52", ls="--", lw=1.2)
ax.text(len(Q_IDS) - 0.45, TOP_K + 0.62, "fallo: relevante fuera del top-3", color="#C44E52", fontsize=8, ha="right")
for i, q in enumerate(Q_IDS):
    ax.text(i - w / 2, ranks_rel_tfidf[q] + 0.08, str(ranks_rel_tfidf[q]), ha="center", fontsize=8)
    ax.text(i + w / 2, ranks_rel_lsa[q] + 0.08, str(ranks_rel_lsa[q]), ha="center", fontsize=8)
    if ranks_rel_tfidf[q] > TOP_K:
        ax.text(i - w / 2, ranks_rel_tfidf[q] + 0.45, "fallo", ha="center", fontsize=8,
                color="#C44E52", fontweight="bold")
    if ranks_rel_lsa[q] > TOP_K:
        ax.text(i + w / 2, ranks_rel_lsa[q] + 0.45, "fallo", ha="center", fontsize=8,
                color="#C44E52", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(Q_IDS)
ax.set_ylim(0, 7.4)
ax.set_ylabel("posición del documento relevante")
ax.set_title("T3 · Posición del relevante por consulta: TF-IDF vs. LSA (fallo si queda fuera del top-3)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("t3_posicion_del_relevante.png", dpi=120)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
cont = [analisis[q]["n_terminos_contenido_compartidos"] for q in Q_IDS]
func = [analisis[q]["n_terminos_funcionales_compartidos"] for q in Q_IDS]
axes[0].bar(Q_IDS, cont, color="#4C72B0", label="términos de contenido")
axes[0].bar(Q_IDS, func, bottom=cont, color="#BAB0AC", label="términos funcionales")
for i, q in enumerate(Q_IDS):
    axes[0].text(i, cont[i] + func[i] + 0.15, str(cont[i] + func[i]), ha="center", fontsize=9)
axes[0].text(4, cont[4] + func[4] + 0.55, "solo 'la'", ha="center", fontsize=8, color="#C44E52")
axes[0].set_ylim(0, max(c + f for c, f in zip(cont, func)) + 1.6)
axes[0].set_ylabel("nº de términos compartidos")
axes[0].set_title("Consulta vs. documento relevante")
axes[0].legend(fontsize=8, loc="upper right")

orden5 = rank_tfidf["q5"]
qset5 = set(tok_q["q5"])
c_cont = [len([t for t in qset5 & set(tok_d[d]) if t not in FUNC]) for d in orden5]
c_func = [len([t for t in qset5 & set(tok_d[d]) if t in FUNC]) for d in orden5]
xpos = np.arange(len(orden5))
bordes = ["#C44E52" if d == rel5 else "none" for d in orden5]
axes[1].bar(xpos, c_cont, color="#4C72B0", edgecolor=bordes, linewidth=1.8, label="contenido")
axes[1].bar(xpos, c_func, bottom=c_cont, color="#BAB0AC", edgecolor=bordes, linewidth=1.8, label="funcionales")
for i, d in enumerate(orden5):
    axes[1].text(i, c_cont[i] + c_func[i] + 0.07, str(c_cont[i] + c_func[i]), ha="center", fontsize=8)
axes[1].set_xticks(xpos)
axes[1].set_xticklabels([f"{d}\n{pos}.º" for pos, d in enumerate(orden5, start=1)], fontsize=8)
for lbl, d in zip(axes[1].get_xticklabels(), orden5):
    if d == rel5:
        lbl.set_color("#C44E52")
        lbl.set_fontweight("bold")
axes[1].set_ylim(0, 3.4)
axes[1].set_ylabel("nº de términos compartidos con q5")
axes[1].set_xlabel("documentos en el orden del ranking TF-IDF de q5 (borde rojo = relevante d09)")
axes[1].set_title("q5 (fallo común): solape léxico de la consulta con cada documento")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("t3_terminos_compartidos.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------- 6) resultados.json (contrato)
resultados = {
    "subtarea": "T3",
    "titulo": "Análisis por consulta: consultas falladas por TF-IDF y por LSA (relevante fuera del top-3) y términos compartidos que explican los fallos",
    "config": {
        "vectorizador": "TfidfVectorizer (scikit-learn, parámetros por defecto), ajustado solo con los 10 documentos (idéntico a T1/T2)",
        "lsa": "TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF de los 10 documentos (idéntico a T2)",
        "proyeccion_consultas": "svd.transform sobre la matriz TF-IDF de las consultas (misma base latente)",
        "similitud": "coseno (espacio TF-IDF y espacio latente)",
        "criterio_de_fallo": "el documento relevante queda fuera del top-3 (posición > 3)",
        "documentos_mal_posicionados": "documentos que ocupan posiciones por encima del relevante en el ranking de cada método",
        "tokenizador_del_analisis": "build_analyzer() del TfidfVectorizer (minúsculas, tokens de >= 2 caracteres, sin stopwords ni stemming)",
        "etiquetado_funcional": "lista mínima fija de palabras gramaticales, solo para etiquetar; no afecta al modelo",
        "n_documentos": len(DOC_IDS),
        "n_consultas": len(Q_IDS),
        "n_terminos_vocabulario": int(len(vec.vocabulary_)),
        "varianza_explicada_lsa": var_lsa,
        "nada_ajustado_con_las_consultas": True,
    },
    "metricas_contexto": {"TF-IDF": met_tfidf, "LSA": met_lsa},
    "ranks_del_documento_relevante": {"TF-IDF": ranks_rel_tfidf, "LSA": ranks_rel_lsa},
    "fallos_por_metodo": fallos,
    "resumen_fallos": resumen_fallos,
    "degradaciones_sin_fallo": degradaciones,
    "top3_por_metodo": {"TF-IDF": {q: rank_tfidf[q][:TOP_K] for q in Q_IDS},
                        "LSA": {q: rank_lsa[q][:TOP_K] for q in Q_IDS}},
    "rankings_completos": {"TF-IDF": rank_tfidf, "LSA": rank_lsa},
    "analisis_terminos_por_consulta": analisis,
    "explicaciones": explicaciones,
    "idf_terminos_clave": idf_terminos_clave,
    "tabla_por_consulta_md": tabla_md,
    "verificacion_con_subtareas_previas": verif,
    "figuras": ["t3_posicion_del_relevante.png", "t3_terminos_compartidos.png"],
}
Path("resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------------------------------------------------------- resumen
print("=" * 78)
print("T3 · Análisis por consulta — fallos (relevante fuera del top-3) y términos compartidos")
print(f"Vocabulario: {len(vec.vocabulary_)} términos | varianza explicada LSA (k=4): {var_lsa:.4f}")
print(f"Métricas recomputadas -> TF-IDF: {met_tfidf}")
print(f"                         LSA   : {met_lsa}")
print(f"Fallos TF-IDF: {resumen_fallos['consultas_falladas_tfidf']} -> posiciones {resumen_fallos['posicion_del_relevante_tfidf']}")
print(f"Fallos LSA   : {resumen_fallos['consultas_falladas_lsa']} -> posiciones {resumen_fallos['posicion_del_relevante_lsa']}")
if degradaciones:
    print(f"Degradaciones sin fallo (LSA): {[(d['consulta_id'], d['rank_tfidf'], '->', d['rank_lsa']) for d in degradaciones]}")
print("-" * 78)
print("Términos compartidos consulta ↔ documento relevante:")
for qid in Q_IDS:
    a = analisis[qid]
    print(f"  {qid} (rel {a['documento_relevante']} | rank TF-IDF {a['rank_tfidf']} | rank LSA {a['rank_lsa']}): "
          f"total {a['n_terminos_compartidos_con_relevante']} {a['terminos_compartidos_con_relevante']} "
          f"| contenido {a['terminos_contenido_compartidos']}")
print("-" * 78)
print("Conclusión:", explicaciones["general"])
print(f"Verificación con T1/T2: {verif}")
print("Archivos escritos: resultados.json, t3_posicion_del_relevante.png, t3_terminos_compartidos.png")
