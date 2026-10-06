```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la misma matriz TF-IDF de la Parte 1.

Reutiliza la matriz TF-IDF, las consultas y los juicios de relevancia de la
subtarea T1 (carpeta entrada/T1/), proyecta la representación a 4 dimensiones
con TruncatedSVD(n_components=4, random_state=0) (LSA), repite la evaluación
completa (ranking por consulta, Hit@1, Hit@3, MRR sobre las 6 consultas) y
construye una única tabla comparativa con las tres métricas de TF-IDF y LSA
lado a lado.

Salidas (carpeta actual):
  - resultados.json                           (contrato de la subtarea)
  - tabla_comparativa_tfidf_vs_lsa.csv        (tabla comparativa única)
  - tabla_comparativa_por_consulta.csv        (detalle por consulta)
  - tabla_comparativa_tfidf_vs_lsa.png        (la tabla, en PNG)
  - comparativa_metricas_tfidf_vs_lsa.png     (barras Hit@1 / Hit@3 / MRR)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Constantes del enunciado (respaldo; lo normal es tomarlas de entrada/T1)
# ----------------------------------------------------------------------------
K_LSA = 4
SEMILLA = 0
ORDEN_Q_DEFAULT = [f"q{i}" for i in range(1, 7)]
ORDEN_D_DEFAULT = [f"d{i:02d}" for i in range(1, 11)]
JUICIOS_DEFAULT = {"q1": "d04", "q2": "d03", "q3": "d08",
                   "q4": "d05", "q5": "d09", "q6": "d02"}
CONSULTAS_DEFAULT = {
    "q1": "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "q2": "¿Cómo se añaden fragmentos recuperados al prompt?",
    "q3": "¿Qué técnica entrena matrices de bajo rango?",
    "q4": "¿Cómo se compara la orientación de dos vectores de texto?",
    "q5": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "q6": "¿Qué arquitectura combina autoatención con capas feed-forward?",
}
T1_DIR = Path("entrada") / "T1"

# ----------------------------------------------------------------------------
# 1) Artefactos de la subtarea previa (T1)
# ----------------------------------------------------------------------------
ruta_t1 = T1_DIR / "resultados.json"
if not ruta_t1.exists():
    raise SystemExit("[T2] ERROR: no se encontró entrada/T1/resultados.json "
                     "(se requieren los resultados de la Parte 1).")
t1 = json.loads(ruta_t1.read_text(encoding="utf-8"))

orden_q = [str(x) for x in (t1.get("orden_consultas") or ORDEN_Q_DEFAULT)]
orden_d = [str(x) for x in (t1.get("orden_documentos") or ORDEN_D_DEFAULT)]
consultas = dict(CONSULTAS_DEFAULT)
consultas.update({str(k): str(v) for k, v in (t1.get("consultas") or {}).items()})
juicios = dict(JUICIOS_DEFAULT)
juicios.update({str(k): str(v) for k, v in (t1.get("juicios_de_relevancia") or {}).items()})

S_t1 = None
for clave in ("matriz_similitud_consultas_x_documentos", "matriz_similitud_tfidf",
              "similitud_consultas_x_documentos"):
    if isinstance(t1.get(clave), list):
        S_t1 = np.asarray(t1[clave], dtype=float)
        break

conf_t1 = t1.get("configuracion") or {}
tam_vocab_t1 = None
for k in ("tam_vocabulario", "tamano_vocabulario", "n_vocabulario", "vocabulario_tam"):
    if isinstance(conf_t1.get(k), (int, float)):
        tam_vocab_t1 = int(conf_t1[k])
        break

# ----------------------------------------------------------------------------
# 2) Funciones auxiliares
# ----------------------------------------------------------------------------
def norm_id(s):
    """Normaliza identificadores de documentos: 'Doc_1' -> 'd01'."""
    s = str(s).strip().lower()
    m = re.fullmatch(r"(?:doc|d)_?0*(\d+)", s)
    return f"d{int(m.group(1)):02d}" if m else s


def es_prosa(x, min_len=15):
    return isinstance(x, str) and len(x.strip()) >= min_len and (" " in x.strip())


def sim_coseno(Q, D):
    return cosine_similarity(Q, D)


def error_vs_t1(S):
    if S_t1 is None or np.asarray(S).shape != S_t1.shape:
        return 0.0
    return float(np.abs(np.asarray(S, dtype=float) - S_t1).max())


def rankear(S):
    """Ranking de los 10 documentos por consulta (similitud descendente,
    empates resueltos por el orden d01..d10)."""
    out = {}
    for i, q in enumerate(orden_q):
        idx = np.argsort(-np.asarray(S[i], dtype=float), kind="stable")
        out[q] = [orden_d[j] for j in idx]
    return out


def evaluar(rankings):
    """Hit@1, Hit@3, MRR por consulta y globales (promedio sobre las 6)."""
    por_q, h1, h3, rr = {}, [], [], []
    for q in orden_q:
        rel = juicios[q]
        pos = None
        for i, d in enumerate(rankings[q]):
            if norm_id(d) == norm_id(rel):
                pos = i + 1
                break
        if pos is None:
            raise SystemExit(f"[T2] ERROR: el relevante '{rel}' de {q} no está en el ranking.")
        por_q[q] = {"consulta": consultas[q],
                    "documento_relevante": rel,
                    "posicion_relevante": int(pos),
                    "Hit@1": float(1.0 if pos == 1 else 0.0),
                    "Hit@3": float(1.0 if pos <= 3 else 0.0),
                    "RR": float(1.0 / pos)}
        h1.append(pos == 1)
        h3.append(pos <= 3)
        rr.append(1.0 / pos)
    glob = {"Hit@1": float(np.mean(h1)),
            "Hit@3": float(np.mean(h3)),
            "MRR": float(np.mean(rr))}
    return por_q, glob


def leer_tabla(p):
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf == ".tsv":
        return pd.read_csv(p, sep="\t")
    return pd.read_parquet(p)

# ----------------------------------------------------------------------------
# 3) Búsqueda de los textos de los 10 documentos dentro de entrada/T1
# ----------------------------------------------------------------------------
def textos_docs_desde_json(obj, orden_d):
    ids_norm = {norm_id(d) for d in orden_d}
    n = len(orden_d)
    hallazgos = []

    def prioridad(clave):
        return 0 if clave and re.search(r"document|corpus|coleccion|texto|passage", clave, re.I) else 1

    def visit(node, clave):
        if isinstance(node, dict):
            mapeo = {norm_id(k): node[k] for k in node}
            if ids_norm.issubset(mapeo.keys()):
                vals = [mapeo[norm_id(d)] for d in orden_d]
                if all(es_prosa(v) for v in vals):
                    hallazgos.append((prioridad(clave), vals))
                else:
                    for tk in ("texto", "text", "contenido", "content", "cuerpo", "body",
                               "documento", "passage"):
                        if all(isinstance(v, dict) and es_prosa(v.get(tk)) for v in vals):
                            hallazgos.append((prioridad(clave), [v[tk] for v in vals]))
                            break
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    visit(v, str(k))
        elif isinstance(node, list):
            if len(node) == n and all(es_prosa(x) for x in node):
                hallazgos.append((prioridad(clave), list(node)))
            elif len(node) == n and all(isinstance(x, dict) for x in node):
                for tk in ("texto", "text", "contenido", "content", "cuerpo", "body",
                           "documento", "passage"):
                    if all(es_prosa(x.get(tk)) for x in node):
                        idk = None
                        for ik in ("id", "doc_id", "id_doc", "documento_id", "nombre",
                                   "doc", "documento"):
                            if ik != tk and all(ik in x for x in node):
                                idk = ik
                                break
                        if idk is not None:
                            mapeo = {norm_id(x[idk]): x[tk] for x in node}
                            if ids_norm.issubset(mapeo.keys()):
                                hallazgos.append((0, [mapeo[norm_id(d)] for d in orden_d]))
                            else:
                                hallazgos.append((1, [x[tk] for x in node]))
                        else:
                            hallazgos.append((1, [x[tk] for x in node]))
                        break
            for x in node:
                if isinstance(x, (dict, list)):
                    visit(x, clave)

    visit(obj, None)
    if not hallazgos:
        return None
    hallazgos.sort(key=lambda h: h[0])
    return hallazgos[0][1]


def textos_docs_desde_tablas(rutas, orden_d):
    ids_norm = {norm_id(d) for d in orden_d}
    hallazgos = []
    for p in rutas:
        try:
            df = leer_tabla(p)
        except Exception:
            continue
        if len(df) != len(orden_d):
            continue
        col_id = None
        for c in df.columns:
            try:
                if {norm_id(v) for v in df[c].astype(str)} == ids_norm:
                    col_id = c
                    break
            except Exception:
                continue
        for c in df.columns:
            if c == col_id:
                continue
            serie = df[c].astype(str)
            if serie.str.strip().str.len().ge(15).all() and serie.str.contains(" ", regex=False).all():
                if col_id is not None:
                    mapeo = {norm_id(a): b for a, b in zip(df[col_id], serie)}
                    if ids_norm.issubset(mapeo.keys()):
                        prior = 0 if re.search(r"text|contenido|doc|cuerpo|body", str(c), re.I) else 1
                        hallazgos.append((prior, [mapeo[norm_id(d)] for d in orden_d]))
                else:
                    hallazgos.append((1, list(serie)))
    if not hallazgos:
        return None
    hallazgos.sort(key=lambda h: h[0])
    return hallazgos[0][1]


def textos_docs_desde_txts(rutas, orden_d):
    mapeo = {}
    for p in rutas:
        m = re.fullmatch(r"(?:doc|d)_?0*(\d+)", p.stem.strip().lower())
        if m:
            mapeo[f"d{int(m.group(1)):02d}"] = p.read_text(encoding="utf-8", errors="replace")
    if {norm_id(d) for d in orden_d}.issubset(mapeo.keys()):
        return [mapeo[norm_id(d)] for d in orden_d]
    return None

# ----------------------------------------------------------------------------
# 4) Búsqueda de matrices TF-IDF guardadas por T1 (por si no hay textos)
# ----------------------------------------------------------------------------
def pares_tfidf_guardados(t1_json, rutas_npz, rutas_npy, rutas_tabla, n_d, n_q):
    planas = []

    def agregar(etiqueta, a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return
        if a.ndim != 2 or not np.isfinite(a).all() or a.min() < -1e-12:
            return
        if a.shape[0] not in (n_d, n_q, n_d + n_q):
            if a.shape[1] in (n_d, n_q, n_d + n_q) and a.shape[0] > 20:
                a = a.T
            else:
                return
        if a.shape[1] <= 20:  # descarta matrices de similitud (6 o 10 columnas)
            return
        planas.append((etiqueta, a))

    def visit(node, etiqueta):
        if isinstance(node, dict):
            for k, v in node.items():
                visit(v, f"{etiqueta}.{k}" if etiqueta else str(k))
        elif isinstance(node, list):
            try:
                a = np.asarray(node, dtype=float)
            except Exception:
                a = None
            if a is not None and a.ndim == 2:
                agregar(etiqueta, a)
            else:
                for x in node:
                    if isinstance(x, (dict, list)):
                        visit(x, etiqueta)

    visit(t1_json, "resultados.json")

    for p in rutas_npz:
        try:
            z = np.load(p)
            for k in z.files:
                agregar(f"{p.name}:{k}", z[k])
        except Exception:
            pass

    for p in rutas_npy:
        try:
            agregar(p.name, np.load(p))
        except Exception:
            pass

    for p in rutas_tabla:
        try:
            df = leer_tabla(p)
        except Exception:
            continue
        num = df.select_dtypes(include=[np.number])
        if num.shape[0] == 0 or num.shape[1] == 0:
            continue
        drop = [c for c in num.columns
                if re.match(r"unnamed", str(c), re.I)
                or np.array_equal(num[c].to_numpy(dtype=float),
                                  np.arange(num.shape[0], dtype=float))]
        if drop:
            num = num.drop(columns=drop)
        agregar(p.name, num.to_numpy(dtype=float))

    pares = []
    for etiqueta, a in planas:
        if a.shape[0] == n_d + n_q:  # matriz apilada docs+consultas
            pares.append((f"{etiqueta} [apilada docs+consultas]", a[:n_d], a[n_d:]))
    docs = [(e, a) for e, a in planas if a.shape[0] == n_d]
    quers = [(e, a) for e, a in planas if a.shape[0] == n_q]
    for ed, ad in docs:
        for eq, aq in quers:
            if ad.shape[1] == aq.shape[1]:
                pares.append((f"{ed} + {eq}", ad, aq))
    return pares

# ----------------------------------------------------------------------------
# 5) Recuperar la representación TF-IDF de la Parte 1
# ----------------------------------------------------------------------------
rutas_json = sorted(T1_DIR.rglob("*.json"))
rutas_npz = sorted(T1_DIR.rglob("*.npz"))
rutas_npy = sorted(T1_DIR.rglob("*.npy"))
rutas_tabla = (sorted(T1_DIR.rglob("*.csv")) + sorted(T1_DIR.rglob("*.tsv"))
               + sorted(T1_DIR.rglob("*.parquet")) + sorted(T1_DIR.rglob("*.pq")))
rutas_txt = sorted(T1_DIR.rglob("*.txt")) + sorted(T1_DIR.rglob("*.md"))

fuentes = []  # (error_de_verificacion, descripcion, X_docs, X_queries)

for etiqueta, Xd, Xq in pares_tfidf_guardados(t1, rutas_npz, rutas_npy, rutas_tabla,
                                              len(orden_d), len(orden_q)):
    fuentes.append((error_vs_t1(sim_coseno(Xq, Xd)),
                    f"matrices TF-IDF leídas de entrada/T1 [{etiqueta}]", Xd, Xq))

textos_docs = textos_docs_desde_json(t1, orden_d)
if textos_docs is None:
    for p in [r for r in rutas_json if r.name != ruta_t1.name]:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        textos_docs = textos_docs_desde_json(obj, orden_d)
        if textos_docs is not None:
            break
if textos_docs is None:
    textos_docs = textos_docs_desde_tablas(rutas_tabla, orden_d)
if textos_docs is None:
    textos_docs = textos_docs_desde_txts(rutas_txt, orden_d)

if textos_docs is not None:
    vec_tfidf = TfidfVectorizer()  # parámetros por defecto, idéntico a la Parte 1
    Xd = vec_tfidf.fit_transform(textos_docs)
    Xq = vec_tfidf.transform([consultas[q] for q in orden_q])
    fuentes.append((error_vs_t1(sim_coseno(Xq, Xd)),
                    "TfidfVectorizer (parámetros por defecto) reajustado sobre los 10 documentos "
                    "cuyos textos están en entrada/T1; consultas proyectadas con el mismo vocabulario",
                    Xd, Xq))

fuentes = [f for f in fuentes if f[2].shape[1] > K_LSA]
if not fuentes:
    raise SystemExit("[T2] ERROR: no se pudo recuperar la matriz TF-IDF de la Parte 1 "
                     "(ni matrices guardadas ni textos de los documentos en entrada/T1).")

fuentes.sort(key=lambda f: f[0])
err_verif, desc_fuente, X_docs, X_queries = fuentes[0]
if S_t1 is not None and err_verif > 1e-6:
    print(f"[AVISO] La representación recuperada difiere de la similitud TF-IDF de T1 "
          f"(error máximo = {err_verif:.3e}).")

# ----------------------------------------------------------------------------
# 6) Línea base TF-IDF (Parte 1): rankings de T1 y verificación de la matriz
# ----------------------------------------------------------------------------
S_tfidf = sim_coseno(X_queries, X_docs)
rankings_tfidf_propios = rankear(S_tfidf)
t1_rank = t1.get("ranking_por_consulta") or {}
rankings_tfidf = {q: [str(d) for d in (t1_rank.get(q) or rankings_tfidf_propios[q])]
                  for q in orden_q}
rankings_identicos = all(
    [norm_id(d) for d in rankings_tfidf_propios[q]] == [norm_id(d) for d in rankings_tfidf[q]]
    for q in orden_q)
metricas_tfidf_q, metricas_tfidf_glob = evaluar(rankings_tfidf)

# ----------------------------------------------------------------------------
# 7) LSA: TruncatedSVD(n_components=4, random_state=0) y nueva evaluación
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=K_LSA, random_state=SEMILLA)
D_lsa = svd.fit_transform(X_docs)   # ajuste SOLO con los 10 documentos
Q_lsa = svd.transform(X_queries)    # las consultas solo se proyectan
S_lsa = sim_coseno(Q_lsa, D_lsa)
rankings_lsa = rankear(S_lsa)
metricas_lsa_q, metricas_lsa_glob = evaluar(rankings_lsa)

# ----------------------------------------------------------------------------
# 8) Tabla comparativa única (CSV) y detalle por consulta
# ----------------------------------------------------------------------------
tabla = pd.DataFrame({
    "Metodo": ["TF-IDF + coseno (Parte 1)",
               f"LSA TruncatedSVD(k={K_LSA}) + coseno (Parte 2)"],
    "Hit@1": [metricas_tfidf_glob["Hit@1"], metricas_lsa_glob["Hit@1"]],
    "Hit@3": [metricas_tfidf_glob["Hit@3"], metricas_lsa_glob["Hit@3"]],
    "MRR": [metricas_tfidf_glob["MRR"], metricas_lsa_glob["MRR"]],
})
tabla.to_csv("tabla_comparativa_tfidf_vs_lsa.csv", index=False, encoding="utf-8")

filas_q = []
for q in orden_q:
    filas_q.append({
        "consulta_id": q,
        "consulta": consultas[q],
        "documento_relevante": juicios[q],
        "pos_TFIDF": metricas_tfidf_q[q]["posicion_relevante"],
        "pos_LSA": metricas_lsa_q[q]["posicion_relevante"],
        "Hit@1_TFIDF": metricas_tfidf_q[q]["Hit@1"],
        "Hit@1_LSA": metricas_lsa_q[q]["Hit@1
