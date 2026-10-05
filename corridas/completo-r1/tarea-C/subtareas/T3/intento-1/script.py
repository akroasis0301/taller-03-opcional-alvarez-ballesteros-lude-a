#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Tarea C · MMIA 6013 · Taller 03 v2).

Con los rankings de ambos métodos (TF-IDF de T1/T2 y LSA-4 de T2):
  1) tabla de fallos por método: 6 consultas x 2 métodos, con la posición del
     documento relevante y marca de fallo (relevante fuera del top 3);
  2) términos que comparte cada consulta con su documento relevante
     (intersección de tokens según el vocabulario TF-IDF), con DF e IDF;
  3) documentación explícita del caso q5 (fallo en ambos métodos).

Entradas: entrada/T1/resultados.json, entrada/T2/resultados.json
          (con respaldo de un recálculo local con la configuración exacta de
          T1/T2: TfidfVectorizer por defecto + TruncatedSVD(4, random_state=0)).
Salidas:  resultados.json, t3_tabla_fallos.csv, t3_terminos_compartidos.csv,
          t3_analisis_consultas.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 0) Corpus, consultas y juicios de relevancia (datos completos del enunciado)
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
CONSULTAS = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
DOC_IDS = list(DOCS)
QIDS = list(CONSULTAS)
METODOS = ["tfidf", "lsa"]

# ---------------------------------------------------------------------------
# 1) Carga de resultados previos (T1 y T2)
# ---------------------------------------------------------------------------
def cargar_json(ruta):
    p = Path(ruta)
    if p.is_file():
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Aviso: no se pudo leer {ruta}: {exc}")
    return None


T1 = cargar_json("entrada/T1/resultados.json")
T2 = cargar_json("entrada/T2/resultados.json")


def _por_consulta(bloque):
    """Normaliza {qid: {...}} a {qid: {'ranking': [...], 'sims': {...}|None}}."""
    out = {}
    if not isinstance(bloque, dict):
        return out
    for qid, entrada in bloque.items():
        if not isinstance(entrada, dict):
            continue
        ranking = entrada.get("ranking")
        sims = None
        if isinstance(entrada.get("ranking_con_similitud"), list):
            pares = [p for p in entrada["ranking_con_similitud"]
                     if isinstance(p, dict) and "doc" in p]
            sims = {p["doc"]: p.get("similitud") for p in pares}
            if ranking is None:
                ranking = [p["doc"] for p in pares]
        if isinstance(ranking, list) and ranking:
            out[str(qid)] = {"ranking": [str(d) for d in ranking], "sims": sims}
    return out


def _buscar_bloque(nodo):
    """Localiza el primer sub-diccionario {qid: {..., 'ranking': ...}}."""
    if not isinstance(nodo, dict):
        return {}
    for clave in ("rankings_por_consulta", "metricas_por_consulta",
                  "resultados_por_consulta", "por_consulta"):
        if clave in nodo:
            cand = _por_consulta(nodo[clave])
            if cand:
                return cand
    for valor in nodo.values():
        if isinstance(valor, dict):
            cand = _por_consulta(valor)
            if cand:
                return cand
    return {}


def rankings_en_t2(metodo):
    if isinstance(T2, dict) and isinstance(T2.get(metodo), dict):
        cand = _buscar_bloque(T2[metodo])
        if cand:
            return cand
    return {}


bloque_t1 = _buscar_bloque(T1) if isinstance(T1, dict) else {}

# ---------------------------------------------------------------------------
# 2) Recálculo de referencia con la configuración exacta de T1/T2
#    (TfidfVectorizer por defecto + TruncatedSVD(n_components=4, random_state=0))
# ---------------------------------------------------------------------------
textos_docs = [DOCS[d] for d in DOC_IDS]
textos_q = [CONSULTAS[q][0] for q in QIDS]

vectorizer = TfidfVectorizer()                      # parámetros por defecto, como en T1
X = vectorizer.fit_transform(textos_docs)           # 10 x |V|
Q = vectorizer.transform(textos_q)                  # 6 x |V|

svd = TruncatedSVD(n_components=4, random_state=0)  # LSA, como en T2
X_lsa = svd.fit_transform(X)                        # 10 x 4
Q_lsa = np.asarray(Q.dot(svd.components_.T))        # folding-in: q · V^T

sims_recalc = {
    "tfidf": cosine_similarity(Q, X),
    "lsa": cosine_similarity(Q_lsa, X_lsa),
}


def rankings_desde_sims(S):
    """Orden descendente por similitud; empates por orden original d01..d10."""
    return {qid: [DOC_IDS[i] for i in np.argsort(-S[k], kind="stable")]
            for k, qid in enumerate(QIDS)}


rank_recalc = {m: rankings_desde_sims(sims_recalc[m]) for m in METODOS}

# ---------------------------------------------------------------------------
# 3) Rankings oficiales de ambos métodos (archivos de T2/T1, con respaldo)
# ---------------------------------------------------------------------------
rankings, sims_oficiales, fuente_rankings = {}, {}, {}
for m in METODOS:
    r_t2 = rankings_en_t2(m)
    if r_t2:
        r, src = r_t2, f"entrada/T2/resultados.json ({m})"
    elif m == "tfidf" and bloque_t1:
        r, src = bloque_t1, "entrada/T1/resultados.json (tfidf)"
    else:
        r = {q: {"ranking": rank_recalc[m][q], "sims": None} for q in QIDS}
        src = "recalculo local (misma configuración que T1/T2)"
    rankings[m] = {q: r[q]["ranking"] for q in QIDS if q in r}
    sims_oficiales[m] = {q: r[q].get("sims") for q in QIDS if q in r}
    for q in QIDS:
        rankings[m].setdefault(q, rank_recalc[m][q])
        sims_oficiales[m].setdefault(q, None)
    fuente_rankings[m] = src

usar_sims_t2 = {m: fuente_rankings[m].startswith("entrada/T2") for m in METODOS}

# ---------------------------------------------------------------------------
# 4) Verificación de consistencia (recálculo local y métricas de T1/T2)
# ---------------------------------------------------------------------------
def metricas_de_rankings(rank_dict):
    rr, h1, h3 = [], [], []
    for q in QIDS:
        pos = rank_dict[q].index(CONSULTAS[q][1]) + 1
        rr.append(1.0 / pos)
        h1.append(1.0 if pos == 1 else 0.0)
        h3.append(1.0 if pos <= 3 else 0.0)
    return {"hit@1": float(np.mean(h1)), "hit@3": float(np.mean(h3)),
            "mrr": float(np.mean(rr))}


metricas_rankings = {m: metricas_de_rankings(rankings[m]) for m in METODOS}

verificacion = {}
for m in METODOS:
    verificacion[f"rankings_{m}_coinciden_con_recalculo"] = bool(
        all(rankings[m][q] == rank_recalc[m][q] for q in QIDS))
if bloque_t1:
    verificacion["tfidf_T1_coincide_con_fuente_usada"] = bool(all(
        bloque_t1.get(q, {}).get("ranking") == rankings["tfidf"][q]
        for q in QIDS if q in bloque_t1))
if isinstance(T2, dict) and isinstance(T2.get("metricas"), dict):
    for m in METODOS:
        prev = T2["metricas"].get(m)
        if isinstance(prev, dict):
            verificacion[f"metricas_{m}_coinciden_con_T2"] = bool(all(
                np.isclose(metricas_rankings[m][k], prev.get(k, np.nan), atol=1e-9)
                for k in ("hit@1", "hit@3", "mrr")))

# ---------------------------------------------------------------------------
# 5) Tabla de fallos por método (6 consultas x 2 métodos)
# ---------------------------------------------------------------------------
filas = []
for m in METODOS:
    for q in QIDS:
        texto_q, rel = CONSULTAS[q]
        ranking = rankings[m][q]
        pos = ranking.index(rel) + 1
        sim_rel = None
        if usar_sims_t2[m] and sims_oficiales[m].get(q):
            sim_rel = sims_oficiales[m][q].get(rel)
        if sim_rel is None:
            sim_rel = float(sims_recalc[m][QIDS.index(q), DOC_IDS.index(rel)])
        filas.append({
            "metodo": m,
            "consulta": q,
            "texto_consulta": texto_q,
            "relevante": rel,
            "posicion_relevante": int(pos),
            "en_top3": bool(pos <= 3),
            "fallo": bool(pos > 3),
            "similitud_relevante": float(sim_rel),
            "top3": list(ranking[:3]),
            "fuente_ranking": fuente_rankings[m],
        })
tabla_fallos = pd.DataFrame(filas)

fallos_por_metodo = {
    m: tabla_fallos.loc[(tabla_fallos["metodo"] == m) & (tabla_fallos["fallo"]),
                        "consulta"].tolist()
    for m in METODOS
}
posiciones = {m: {q: int(rankings[m][q].index(CONSULTAS[q][1]) + 1) for q in QIDS}
              for m in METODOS}
degradadas_lsa = [q for q in QIDS if posiciones["lsa"][q] > posiciones["tfidf"][q]]

# ---------------------------------------------------------------------------
# 6) Términos compartidos consulta–documento relevante (vocabulario TF-IDF)
# ---------------------------------------------------------------------------
analizador = vectorizer.build_analyzer()
vocab = vectorizer.vocabulary_
idf_por_indice = vectorizer.idf_
df_por_indice = np.bincount(X.nonzero()[1], minlength=X.shape[1])

tokens_docs = {d: set(analizador(DOCS[d])) for d in DOC_IDS}
docs_de_termino = {}
for d in DOC_IDS:
    for t in tokens_docs[d]:
        docs_de_termino.setdefault(t, []).append(d)
for t in docs_de_termino:
    docs_de_termino[t].sort()

conjunto_vocab = set(vocab)
terminos_compartidos = {}
for q in QIDS:
    texto_q, rel = CONSULTAS[q]
    tokens_q = set(analizador(texto_q))
    compartidos = sorted(tokens_q & tokens_docs[rel] & conjunto_vocab)
    en_vocab = sorted(t for t in tokens_q if t in conjunto_vocab)
    fuera_vocab = sorted(t for t in tokens_q if t not in conjunto_vocab)
    detalle = {t: {"df": int(df_por_indice[vocab[t]]),
                   "idf": float(idf_por_indice[vocab[t]]),
                   "documentos": list(docs_de_termino.get(t, []))}
               for t in compartidos}
    lectura = (f"{q} ↔ {rel}: comparten {len(compartidos)} término(s) del vocabulario "
               "TF-IDF: "
               + ", ".join(f"{t} (DF={v['df']}/10, IDF={v['idf']:.3f})"
                           for t, v in detalle.items())
               + ". Los términos con DF bajo (IDF alto) aportan más similitud; los de DF alto aportan poco.")
    exclusivos = [t for t, v in detalle.items() if v["df"] == 1]
    if exclusivos:
        lectura += f" Términos exclusivos del par consulta–documento (DF=1): {', '.join(exclusivos)}."
    terminos_compartidos[q] = {
        "consulta": texto_q,
        "relevante": rel,
        "texto_relevante": DOCS[rel],
        "n_terminos_compartidos": len(compartidos),
        "terminos_compartidos": compartidos,
        "df_e_idf_por_termino": detalle,
        "tokens_consulta_en_vocabulario": en_vocab,
        "tokens_consulta_fuera_del_vocabulario": fuera_vocab,
        "posicion_relevante": {m: posiciones[m][q] for m in METODOS},
        "fallo": {m: bool(posiciones[m][q] > 3) for m in METODOS},
        "lectura": lectura,
    }

# ---------------------------------------------------------------------------
# 7) Caso q5: fallo documentado en ambos métodos
# ---------------------------------------------------------------------------
q5 = terminos_compartidos["q5"]
pos_q5 = {m: posiciones[m]["q5"] for m in METODOS}
idf_min = float(idf_por_indice.min())
terminos_idf_minimo = sorted(t for t, i in vocab.items()
                             if abs(idf_por_indice[i] - idf_min) < 1e-12)

no_compartidos_en_vocab = [t for t in q5["tokens_consulta_en_vocabulario"]
                           if t not in q5["terminos_compartidos"]]
donde_aparecen = {t: {"df": int(df_por_indice[vocab[t]]),
                      "idf": float(idf_por_indice[vocab[t]]),
                      "documentos": list(docs_de_termino.get(t, []))}
                  for t in no_compartidos_en_vocab}

fallo_q5_ambos = all(p > 3 for p in pos_q5.values())
if "la" in vocab and q5["terminos_compartidos"] == ["la"]:
    idf_la = float(idf_por_indice[vocab["la"]])
    df_la = int(df_por_indice[vocab["la"]])
    frase_min = ""
    if abs(idf_la - idf_min) < 1e-12:
        otros = ", ".join(f"'{t}'" for t in terminos_idf_minimo if t != "la")
        frase_min = f", el IDF mínimo del corpus (empatado con {otros})"
    cierre = (
        f"En consecuencia d09 queda en el puesto {pos_q5['tfidf']} con TF-IDF y en el "
        f"puesto {pos_q5['lsa']} con LSA: fuera del top 3 en ambos métodos (fallo documentado)."
        if fallo_q5_ambos else
        f"Posiciones obtenidas: TF-IDF {pos_q5['tfidf']}, LSA {pos_q5['lsa']}."
    )
    explicacion_q5 = (
        f"q5 («{CONSULTAS['q5'][0]}») tiene como relevante d09 («{DOCS['d09']}»). "
        f"La intersección de tokens según el vocabulario TF-IDF se reduce a un único "
        f"término, 'la' (DF={df_la}/10, IDF={idf_la:.4f}{frase_min}). "
        f"Los términos de contenido de la consulta están fuera del vocabulario "
        f"({', '.join(q5['tokens_consulta_fuera_del_vocabulario'])}); los demás tokens "
        f"que sí pertenecen al vocabulario aparecen en otros documentos: "
        + "; ".join(f"{t} → {{{', '.join(v['documentos'])}}}"
                    for t, v in donde_aparecen.items())
        + f". La evidencia léxica favorece a otros documentos (TF-IDF encabeza con "
        f"{rankings['tfidf']['q5'][0]} y {rankings['tfidf']['q5'][1]}). " + cierre
    )
    caso_q5 = {
        "consulta": CONSULTAS["q5"][0],
        "relevante": "d09",
        "texto_relevante": DOCS["d09"],
        "posicion_relevante": pos_q5,
        "fallo_en_ambos_metodos": bool(fallo_q5_ambos),
        "terminos_compartidos": q5["terminos_compartidos"],
        "df_e_idf_terminos_compartidos": q5["df_e_idf_por_termino"],
        "tokens_consulta_en_vocabulario": q5["tokens_consulta_en_vocabulario"],
        "tokens_consulta_fuera_del_vocabulario": q5["tokens_consulta_fuera_del_vocabulario"],
        "donde_aparecen_los_demas_tokens_del_vocabulario": donde_aparecen,
        "idf_minimo_corpus": idf_min,
        "terminos_con_idf_minimo": terminos_idf_minimo,
        "explicacion": explicacion_q5,
    }
else:
    explicacion_q5 = (f"q5 comparte con d09 los términos {q5['terminos_compartidos']}; "
                      f"posiciones: TF-IDF {pos_q5['tfidf']}, LSA {pos_q5['lsa']}.")
    caso_q5 = {
        "consulta": CONSULTAS["q5"][0],
        "relevante": "d09",
        "posicion_relevante": pos_q5,
        "fallo_en_ambos_metodos": bool(fallo_q5_ambos),
        "terminos_compartidos": q5["terminos_compartidos"],
        "explicacion": explicacion_q5,
    }

# ---------------------------------------------------------------------------
# 8) Figura resumen (PNG)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
x = np.arange(len(QIDS))
w = 0.38
pt = [posiciones["tfidf"][q] for q in QIDS]
pl = [posiciones["lsa"][q] for q in QIDS]
ax1.bar(x - w / 2, pt, w, label="TF-IDF", color="#4C72B0")
ax1.bar(x + w / 2, pl, w, label="LSA (4 dim)", color="#DD8452")
ax1.axhline(3, color="crimson", ls="--", lw=1.2, label="umbral top-3")
ax1.set_xticks(x)
ax1.set_xticklabels(QIDS)
ax1.set_ylim(0, 11)
ax1.set_yticks(range(0, 11))
ax1.set_xlabel("consulta")
ax1.set_ylabel("posición del documento relevante")
ax1.set_title("Posición del relevante por método\n(fallo si queda por debajo del umbral)",
              fontsize=10)
for xi, (a, b) in enumerate(zip(pt, pl)):
    ax1.text(xi - w / 2, a + 0.15, str(a), ha="center", fontsize=8)
    ax1.text(xi + w / 2, b + 0.15, str(b), ha="center", fontsize=8)
ax1.legend(fontsize=8, loc="upper left")

n_terms = [terminos_compartidos[q]["n_terminos_compartidos"] for q in QIDS]
colores = ["crimson" if q == "q5" else "#55A868" for q in QIDS]
ax2.bar(x, n_terms, 0.6, color=colores)
ax2.set_xticks(x)
ax2.set_xticklabels(QIDS)
ax2.set_ylim(0, max(n_terms) + 1.6)
ax2.set_xlabel("consulta")
ax2.set_ylabel("n.º de términos compartidos")
ax2.set_title("Solapamiento léxico consulta–relevante\n(vocabulario TF-IDF; q5 en rojo)",
              fontsize=10)
for xi, v in enumerate(n_terms):
    ax2.text(xi, v + 0.12, str(v), ha="center", fontsize=9)
fig.suptitle("T3 · Análisis por consulta: fallos por método y términos compartidos",
             fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig("t3_analisis_consultas.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 9) Tablas CSV de apoyo para el reporte
# ---------------------------------------------------------------------------
tabla_fallos[["metodo", "consulta", "relevante", "posicion_relevante",
              "en_top3", "fallo", "similitud_relevante"]].to_csv(
    "t3_tabla_fallos.csv", index=False, encoding="utf-8")

filas_term = [{
    "consulta": q,
    "relevante": terminos_compartidos[q]["relevante"],
    "n_terminos_compartidos": terminos_compartidos[q]["n_terminos_compartidos"],
    "terminos_compartidos": "; ".join(terminos_compartidos[q]["terminos_compartidos"]),
    "tokens_consulta_fuera_del_vocabulario": "; ".join(
        terminos_compartidos[q]["tokens_consulta_fuera_del_vocabulario"]),
} for q in QIDS]
pd.DataFrame(filas_term).to_csv("t3_terminos_compartidos.csv", index=False,
                                encoding="utf-8")

# ---------------------------------------------------------------------------
# 10) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": (
        "Análisis por consulta: con los rankings de TF-IDF (T1/T2) y LSA-4 (T2) se "
        "identifica en qué consultas falla cada método (documento relevante fuera del "
        "top 3, anotando la posición obtenida) y se extraen los términos que comparten "
        "cada consulta y su documento relevante (intersección de tokens según el "
        "vocabulario TF-IDF), documentando explícitamente el fallo de q5 en ambos métodos."
    ),
    "config": {
        "n_documentos": len(DOC_IDS),
        "n_consultas": len(QIDS),
        "metodos": METODOS,
        "criterio_de_fallo": "posición del documento relevante > 3 (fuera del top 3)",
        "tam_vocabulario_tfidf": int(len(vocab)),
        "definicion_terminos_compartidos": (
            "intersección de los tokens de la consulta y de su documento relevante tras "
            "el análisis de TfidfVectorizer por defecto (minúsculas, token_pattern "
            "(?u)\\b\\w\\w+\\b), restringida al vocabulario ajustado con los 10 documentos"
        ),
    },
    "fuentes": {
        "rankings": fuente_rankings,
        "verificacion": verificacion,
        "metricas_desde_rankings": metricas_rankings,
        "nota": ("Los rankings se toman de los resultados oficiales de T2 (el TF-IDF de "
                 "T2 reutiliza la matriz de T1); un recálculo local con la misma "
                 "configuración los reproduce."),
    },
    "rankings_usados": {m: {q: rankings[m][q] for q in QIDS} for m in METODOS},
    "tabla_fallos": filas,
    "fallos_por_metodo": fallos_por_metodo,
    "posiciones_relevante": posiciones,
    "degradadas_en_lsa_vs_tfidf": degradadas_lsa,
    "terminos_compartidos": terminos_compartidos,
    "caso_q5": caso_q5,
    "resumen": {
        "consultas_que_fallan_tfidf": fallos_por_metodo["tfidf"],
        "consultas_que_fallan_lsa": fallos_por_metodo["lsa"],
        "n_fallos_tfidf": len(fallos_por_metodo["tfidf"]),
        "n_fallos_lsa": len(fallos_por_metodo["lsa"]),
        "q5_falla_en_ambos_metodos": bool(fallo_q5_ambos),
        "posicion_q5": pos_q5,
        "terminos_compartidos_q5": q5["terminos_compartidos"],
        "n_terminos_compartidos_por_consulta": {
            q: terminos_compartidos[q]["n_terminos_compartidos"] for q in QIDS},
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 11) Resumen breve
# ---------------------------------------------------------------------------
print("=" * 78)
print("T3 · Análisis por consulta — fallos por método y términos compartidos")
print("=" * 78)
print(tabla_fallos[["metodo", "consulta", "relevante", "posicion_relevante",
                    "en_top3", "fallo"]].to_string(index=False))
print(f"\nFallos TF-IDF (relevante fuera del top 3): {fallos_por_metodo['tfidf']}")
print(f"Fallos LSA-4  (relevante fuera del top 3): {fallos_por_metodo['lsa']}")
print(f"Consultas degradadas por LSA frente a TF-IDF: {degradadas_lsa}")
print("\nMétricas recomputadas a partir de los rankings usados:")
for m in METODOS:
    print(f"  {m}: {metricas_rankings[m]}")
print("\nTérminos compartidos consulta–documento relevante (vocabulario TF-IDF):")
for q in QIDS:
    info = terminos_compartidos[q]
    print(f"  {q} ↔ {info['relevante']} ({info['n_terminos_compartidos']}): "
          f"{', '.join(info['terminos_compartidos']) if info['terminos_compartidos'] else '—'}")
print(f"\nCaso q5: {explicacion_q5}")
print(f"\nVerificación: {verificacion}")
print("Archivos generados: resultados.json, t3_tabla_fallos.csv, "
      "t3_terminos_compartidos.csv, t3_analisis_consultas.png")
