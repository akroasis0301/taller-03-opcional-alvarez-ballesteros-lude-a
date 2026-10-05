# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Tarea C · MMIA 6013).

1) Identifica, por método, en qué consultas falla cada sistema de recuperación
   (documento relevante fuera del top 3):
      - TF-IDF + coseno  -> rankings de T1 (entrada/T1/resultados.json)
      - LSA (SVD k=4)    -> rankings de T2 (entrada/T2/resultados.json)
2) Calcula los términos que comparten cada consulta con su documento relevante
   y con los documentos recuperados en el top 3, para sustentar el análisis.

Salidas: resultados.json, t3_rank_relevante_por_metodo.png,
         t3_heatmap_terminos_compartidos.png, t3_q5_solape_por_documento.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1) Corpus y consultas (datos completos del enunciado, idénticos a T1/T2)
# ---------------------------------------------------------------------------
DOCS = [
    ("d01", "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión."),
    ("d02", "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización."),
    ("d03", "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo."),
    ("d04", "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento."),
    ("d05", "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación."),
    ("d06", "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado."),
    ("d07", "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos."),
    ("d08", "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales."),
    ("d09", "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad."),
    ("d10", "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria."),
]
QUERIES = [
    ("q1", "¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    ("q2", "¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    ("q3", "¿Qué técnica entrena matrices de bajo rango?", "d08"),
    ("q4", "¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    ("q5", "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    ("q6", "¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
]

doc_ids = [d for d, _ in DOCS]
doc_textos = [t for _, t in DOCS]
q_ids = [q for q, _, _ in QUERIES]
q_textos = [t for _, t, _ in QUERIES]
relevante = {q: r for q, _, r in QUERIES}

# ---------------------------------------------------------------------------
# 2) Carga de resultados previos (rankings oficiales de T1 y T2)
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

t1 = cargar_json(Path("entrada") / "T1" / "resultados.json")
t2 = cargar_json(Path("entrada") / "T2" / "resultados.json")

# ---------------------------------------------------------------------------
# 3) Recalculo de verificación con la MISMA configuración de T1/T2
#    (TfidfVectorizer por defecto; TruncatedSVD(n_components=4, random_state=0))
# ---------------------------------------------------------------------------
vec = TfidfVectorizer()                      # parámetros por defecto, igual que T1
X_docs = vec.fit_transform(doc_textos)       # ajustado SOLO sobre los 10 documentos
X_q = vec.transform(q_textos)                # las consultas solo se transforman
vocab = vec.vocabulary_
idf = vec.idf_
analyzer = vec.build_analyzer()
Xd = X_docs.toarray()
Xq = X_q.toarray()

S_tfidf = cosine_similarity(X_q, X_docs)

svd = TruncatedSVD(n_components=4, random_state=0)
D_lsa = svd.fit_transform(X_docs)
Q_lsa = svd.transform(X_q)
S_lsa = cosine_similarity(Q_lsa, D_lsa)
var_exp = float(svd.explained_variance_ratio_.sum())
S_map = {"tfidf": S_tfidf, "lsa": S_lsa}

def rankings_desde_sim(S):
    out = {}
    for i, q in enumerate(q_ids):
        orden = np.argsort(-S[i], kind="stable")   # empates en orden d01..d10 (como T1)
        rk = [doc_ids[j] for j in orden]
        out[q] = {"ranking": rk, "rank_rel": rk.index(relevante[q]) + 1}
    return out

rk_tfidf_rec = rankings_desde_sim(S_tfidf)
rk_lsa_rec = rankings_desde_sim(S_lsa)

def _rankings_de_json(res, clave):
    out = {}
    for q, info in ((res or {}).get(clave) or {}).items():
        if isinstance(info, dict) and "ranking_completo_10_docs" in info:
            out[q] = {"ranking": [str(d) for d in info["ranking_completo_10_docs"]],
                      "rank_rel": int(info["rank_del_relevante"])}
    return out

rk_t1_file = _rankings_de_json(t1, "rankings_por_consulta")
rk_t2_file = _rankings_de_json(t2, "rankings_por_consulta_lsa")
usar_t1 = len(rk_t1_file) == len(q_ids)
usar_t2 = len(rk_t2_file) == len(q_ids)

# Rankings de referencia: los publicados por T1/T2 (fallback: recálculo verificado)
rk_tfidf = rk_t1_file if usar_t1 else rk_tfidf_rec
rk_lsa = rk_t2_file if usar_t2 else rk_lsa_rec
RKS = {"tfidf": rk_tfidf, "lsa": rk_lsa}

# ---------------------------------------------------------------------------
# 4) Términos compartidos: intersección de tokens (analizador de TfidfVectorizer)
# ---------------------------------------------------------------------------
tok_q = {q: set(analyzer(t)) for q, t, _ in QUERIES}
tok_d = {d: set(analyzer(t)) for d, t in DOCS}
oov = {q: sorted(tok_q[q] - set(vocab)) for q in q_ids}

def comunes(q, d):
    """Términos compartidos consulta-documento, ordenados por idf descendente."""
    return sorted(tok_q[q] & tok_d[d], key=lambda t: (-idf[vocab[t]], t))

def detalle_terminos(q, d):
    qi, di = q_ids.index(q), doc_ids.index(d)
    return [{"termino": t,
             "idf": float(idf[vocab[t]]),
             "tfidf_en_consulta": float(Xq[qi, vocab[t]]),
             "tfidf_en_documento": float(Xd[di, vocab[t]])}
            for t in comunes(q, d)]

# ---------------------------------------------------------------------------
# 5) Fallos por método (relevante fuera del top 3) y métricas coherentes
# ---------------------------------------------------------------------------
fallos = {m: [q for q in q_ids if RKS[m][q]["rank_rel"] > 3] for m in ("tfidf", "lsa")}
hit1 = {m: float(sum(RKS[m][q]["rank_rel"] == 1 for q in q_ids) / len(q_ids)) for m in ("tfidf", "lsa")}
hit3 = {m: float(sum(RKS[m][q]["rank_rel"] <= 3 for q in q_ids) / len(q_ids)) for m in ("tfidf", "lsa")}

fallos_detalle = {
    m: [{"consulta": q,
         "relevante": relevante[q],
         "rank_del_relevante": RKS[m][q]["rank_rel"],
         "top3_recuperado": RKS[m][q]["ranking"][:3]}
        for q in fallos[m]]
    for m in ("tfidf", "lsa")
}

estado_por_consulta = {}
for q, texto, rel in QUERIES:
    estado_por_consulta[q] = {
        "consulta": texto,
        "relevante": rel,
        "rank_tfidf": RKS["tfidf"][q]["rank_rel"],
        "rank_lsa": RKS["lsa"][q]["rank_rel"],
        "fallo_tfidf": RKS["tfidf"][q]["rank_rel"] > 3,
        "fallo_lsa": RKS["lsa"][q]["rank_rel"] > 3,
        "top3_tfidf": RKS["tfidf"][q]["ranking"][:3],
        "top3_lsa": RKS["lsa"][q]["ranking"][:3],
        "ranking_tfidf": RKS["tfidf"][q]["ranking"],
        "ranking_lsa": RKS["lsa"][q]["ranking"],
    }

# ---------------------------------------------------------------------------
# 6) Tabla de términos compartidos consulta-relevante y solape con el top-3
# ---------------------------------------------------------------------------
tabla_relevante = []
terminos_por_consulta = {}
for q, texto, rel in QUERIES:
    comunes_rel = comunes(q, rel)
    tabla_relevante.append({
        "consulta": q,
        "texto_consulta": texto,
        "documento_relevante": rel,
        "terminos_compartidos": comunes_rel,
        "n_terminos_compartidos": len(comunes_rel),
        "rank_tfidf": RKS["tfidf"][q]["rank_rel"],
        "rank_lsa": RKS["lsa"][q]["rank_rel"],
        "fallo_tfidf": RKS["tfidf"][q]["rank_rel"] > 3,
        "fallo_lsa": RKS["lsa"][q]["rank_rel"] > 3,
    })
    i = q_ids.index(q)
    entrada = {
        "texto_consulta": texto,
        "relevante": rel,
        "tokens_consulta": sorted(tok_q[q]),
        "n_tokens_consulta": len(tok_q[q]),
        "terminos_consulta_fuera_de_vocabulario": oov[q],
        "con_documento_relevante": {
            "terminos_compartidos": comunes_rel,
            "n_terminos_compartidos": len(comunes_rel),
            "detalle_por_termino": detalle_terminos(q, rel),
        },
    }
    for m in ("tfidf", "lsa"):
        top3 = RKS[m][q]["ranking"][:3]
        entrada[m] = {
            "top3": top3,
            "rank_del_relevante": RKS[m][q]["rank_rel"],
            "fallo": RKS[m][q]["rank_rel"] > 3,
            "similitud_coseno_relevante": float(S_map[m][i, doc_ids.index(rel)]),
            "similitud_coseno_top3": {d: float(S_map[m][i, doc_ids.index(d)]) for d in top3},
            "terminos_compartidos_con_cada_doc_del_top3": {
                d: {"terminos_compartidos": comunes(q, d),
                    "n_terminos_compartidos": len(tok_q[q] & tok_d[d])}
                for d in top3
            },
        }
    terminos_por_consulta[q] = entrada

M = np.array([[len(tok_q[q] & tok_d[d]) for d in doc_ids] for q in q_ids], dtype=int)
matriz_solape = {"filas_consultas": q_ids, "columnas_documentos": doc_ids,
                 "n_terminos_compartidos": M.tolist()}

# ---------------------------------------------------------------------------
# 7) Explicación de los fallos con los términos compartidos
# ---------------------------------------------------------------------------
def texto_fallo(q, m):
    rk = RKS[m]
    rel = relevante[q]
    rank_rel = rk[q]["rank_rel"]
    top3 = rk[q]["ranking"][:3]
    comunes_rel = comunes(q, rel)
    partes = [f"[{m.upper()}] {q} («{q_textos[q_ids.index(q)}»): el relevante {rel} queda en el puesto {rank_rel}, fuera del top 3."]
    if comunes_rel:
        det = ", ".join(f"«{t}» (idf={idf[vocab[t]]:.2f})" for t in comunes_rel)
        partes.append(f"Términos compartidos consulta-{rel}: {det}; no hay solape en términos de contenido.")
    else:
        partes.append(f"La consulta y {rel} no comparten ningún término del vocabulario.")
    extra = set()
    for d in top3:
        extra |= tok_q[q] & tok_d[d]
    extra -= set(comunes_rel)
    extra = sorted(extra, key=lambda t: (-idf[vocab[t]], t))
    if extra:
        partes.append(f"El top-3 recuperado ({', '.join(top3)}) comparte además "
                      f"{', '.join('«' + t + '»' for t in extra)}, y por eso el método los puntúa por encima de {rel}.")
    if oov[q]:
        partes.append(f"{len(oov[q])} de {len(tok_q[q])} términos de la consulta no aparecen en ningún documento "
                      f"({', '.join('«' + t + '»' for t in oov[q])}): fallo de paráfrasis/vocabulario.")
    if m == "tfidf":
        partes.append(f"TF-IDF es puramente léxico: sin solape de contenido no puede colocar a {rel} arriba.")
    else:
        if q in fallos["tfidf"]:
            partes.append(f"LSA (k=4, {var_exp:.1%} de varianza explicada) no corrige el fallo: el espacio latente "
                          f"aprendido de solo 10 documentos no conecta el campo léxico de la consulta con el de {rel}.")
        else:
            partes.append(f"LSA (k=4, {var_exp:.1%} de varianza explicada) introduce el fallo al difuminar el "
                          f"solape léxico que TF-IDF sí aprovechaba.")
    return " ".join(partes)

explicaciones = {m: {q: texto_fallo(q, m) for q in fallos[m]} for m in ("tfidf", "lsa")}

notas_cambios = []
for q in q_ids:
    rt, rl = RKS["tfidf"][q]["rank_rel"], RKS["lsa"][q]["rank_rel"]
    if rt != rl and rt <= 3 and rl <= 3:
        rel = relevante[q]
        top_lsa = RKS["lsa"][q]["ranking"][:3]
        encima = top_lsa[:top_lsa.index(rel)] if rel in top_lsa else []
        det = "; ".join(f"{d} (comparte: {', '.join(comunes(q, d)) or 'nada'})" for d in encima)
        i = q_ids.index(q)
        sims = ", ".join(f"{d}={S_lsa[i, doc_ids.index(d)]:.3f}" for d in top_lsa)
        notas_cambios.append(
            f"{q}: pasa del puesto {rt} (TF-IDF) al {rl} (LSA) sin salir del top 3 (no cuenta como fallo). "
            f"Top-3 LSA: {', '.join(top_lsa)} (similitudes: {sims}). "
            f"Documentos por encima del relevante {rel}: {det}. "
            f"La proyección a k=4 componentes ({var_exp:.1%} de la varianza) difumina el peso de los términos de contenido."
        )

qs_fallo = [q for q in q_ids if (q in fallos["tfidf"]) or (q in fallos["lsa"])]
sintesis = "; ".join(
    f"{m.upper()} falla en {len(fallos[m])}/6 consultas" + (f" ({', '.join(fallos[m])})" if fallos[m] else "")
    for m in ("tfidf", "lsa")
)
sintesis += f". Hit@1: TF-IDF {hit1['tfidf']:.4f}, LSA {hit1['lsa']:.4f}; Hit@3: TF-IDF {hit3['tfidf']:.4f}, LSA {hit3['lsa']:.4f}."
if qs_fallo:
    det = "; ".join(f"{q}: su relevante {relevante[q]} comparte {len(tok_q[q] & tok_d[relevante[q]])} término(s) con la consulta"
                    for q in qs_fallo)
    sintesis += " Causa: " + det + "; el top-3 recuperado comparte más términos con la consulta."
    for q in qs_fallo:
        if oov[q]:
            sintesis += f" En {q}, {len(oov[q])} de {len(tok_q[q])} términos de la consulta no aparecen en ningún documento (paráfrasis)."
    sintesis += " LSA (k=4) no corrige este fallo de vocabulario con un corpus de solo 10 documentos."

# ---------------------------------------------------------------------------
# 8) Verificación contra T1/T2
# ---------------------------------------------------------------------------
tam_vocab_t1, vocab_coincide = None, None
if t1 is not None:
    v1 = (t1.get("config") or {}).get("vocabulario_tamano")
    if v1 is not None:
        tam_vocab_t1 = int(v1)
        vocab_coincide = (tam_vocab_t1 == len(vocab))

mismos_tfidf = None
if usar_t1:
    mismos_tfidf = all(rk_t1_file[q]["ranking"] == rk_tfidf_rec[q]["ranking"]
                       and rk_t1_file[q]["rank_rel"] == rk_tfidf_rec[q]["rank_rel"] for q in q_ids)
mismos_lsa = None
if usar_t2:
    mismos_lsa = all(rk_t2_file[q]["ranking"] == rk_lsa_rec[q]["ranking"]
                     and rk_t2_file[q]["rank_rel"] == rk_lsa_rec[q]["rank_rel"] for q in q_ids)

ranks_m_t1 = {q: int(v["rank"]) for q, v in ((t1 or {}).get("metricas_por_consulta") or {}).items()
              if isinstance(v, dict) and "rank" in v}
coincide_metricas_t1 = (all(ranks_m_t1.get(q) == RKS["tfidf"][q]["rank_rel"] for q in q_ids)
                        if len(ranks_m_t1) == len(q_ids) else None)

tab2 = (t2 or {}).get("tabla_comparativa_por_consulta") or {}
coincide_tab2 = None
if len(tab2) == len(q_ids):
    coincide_tab2 = all(int(tab2[q].get("rank_tfidf", -999)) == RKS["tfidf"][q]["rank_rel"]
                        and int(tab2[q].get("rank_lsa", -999)) == RKS["lsa"][q]["rank_rel"]
                        for q in q_ids)

def _cerca(a, b):
    if a is None or b is None:
        return None
    return bool(abs(float(a) - float(b)) < 1e-9)

hit3_t1_ok = _cerca(((t1 or {}).get("metricas_agregadas") or {}).get("hit@3"), hit3["tfidf"])
hit3_t2_ok = _cerca(((t2 or {}).get("metricas_agregadas_lsa") or {}).get("hit@3"), hit3["lsa"])

# ---------------------------------------------------------------------------
# 9) Figuras
# ---------------------------------------------------------------------------
# Figura 1: rank del relevante por consulta y método
fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(q_ids))
w = 0.38
r_t = [RKS["tfidf"][q]["rank_rel"] for q in q_ids]
r_l = [RKS["lsa"][q]["rank_rel"] for q in q_ids]
b1 = ax.bar(x - w / 2, r_t, w, label="TF-IDF (rankings T1)", color="#4C72B0")
b2 = ax.bar(x + w / 2, r_l, w, label="LSA k=4 (rankings T2)", color="#DD8452")
for barras, ranks in ((b1, r_t), (b2, r_l)):
    for rect, r in zip(barras, ranks):
        if r > 3:
            rect.set_edgecolor("red")
            rect.set_linewidth(2)
        ax.text(rect.get_x() + rect.get_width() / 2, r + 0.15, str(r), ha="center", va="bottom", fontsize=9)
ax.axhline(3, color="red", ls="--", lw=1.2, label="Umbral top-3 (fallo si rank > 3)")
ax.set_xticks(x)
ax.set_xticklabels(q_ids)
ax.set_yticks(range(0, 11))
ax.set_ylim(0, 10.8)
ax.set_xlabel("Consulta")
ax.set_ylabel("Rank del documento relevante")
ax.set_title("T3 · Rank del documento relevante por consulta y método (borde rojo = fallo)")
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
plt.tight_layout()
plt.savefig("t3_rank_relevante_por_metodo.png", dpi=120)
plt.close(fig)

# Figura 2: mapa de calor del solape léxico consulta-documento
vmax = max(1, int(M.max()))
fig, ax = plt.subplots(figsize=(8.8, 4.8))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids)
ax.set_yticks(range(len(q_ids)))
ax.set_yticklabels(q_ids)
for i in range(len(q_ids)):
    for j in range(len(doc_ids)):
        v = int(M[i, j])
        ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                color="white" if v > 0.55 * vmax else "black")
for i, q in enumerate(q_ids):
    j = doc_ids.index(relevante[q])
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
ax.set_xlabel("Documento")
ax.set_ylabel("Consulta")
ax.set_title("T3 · Términos compartidos consulta-documento (recuadro rojo = relevante)")
plt.colorbar(im, ax=ax, label="Nº de términos compartidos", shrink=0.85)
plt.tight_layout()
plt.savefig("t3_heatmap_terminos_compartidos.png", dpi=120)
plt.close(fig)

# Figura 3: q5 (fallo en ambos métodos) — solape con cada documento
i5 = q_ids.index("q5")
vals = M[i5]
top3_t5 = RKS["tfidf"]["q5"]["ranking"][:3]
colores = ["#C44E52" if d == relevante["q5"] else ("#DD8452" if d in top3_t5 else "#B8B8B8") for d in doc_ids]
fig, ax = plt.subplots(figsize=(8.8, 4.2))
barras = ax.bar(doc_ids, vals, color=colores, edgecolor="black", linewidth=0.5)
for rect, v in zip(barras, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.06, str(int(v)), ha="center", va="bottom", fontsize=9)
comunes_q5 = comunes("q5", relevante["q5"])
j9 = doc_ids.index(relevante["q5"])
etiqueta = "solo comparte: " + ", ".join(comunes_q5) if comunes_q5 else "sin solape con la consulta"
ax.text(j9, vals[j9] + 0.5, etiqueta, ha="center", fontsize=8, color="#C44E52")
ax.set_ylim(0, max(float(vals.max()) + 1.4, 2.0))
ax.set_xlabel("Documento")
ax.set_ylabel("Nº de términos compartidos con q5")
ax.set_title("T3 · q5 (falla en TF-IDF y LSA): solape léxico con cada documento")
ax.legend(handles=[Patch(color="#C44E52", label=f"Relevante {relevante['q5']}"),
                   Patch(color="#DD8452", label="Top-3 TF-IDF"),
                   Patch(color="#B8B8B8", label="Resto")],
          loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("t3_q5_solape_por_documento.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 10) Contrato de resultados
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T3_analisis_por_consulta_fallos_y_terminos_compartidos",
    "config": {
        "metodos": {
            "tfidf": "TF-IDF (TfidfVectorizer por defecto) + similitud coseno; rankings de T1",
            "lsa": "LSA (TruncatedSVD n_components=4, random_state=0) + coseno; rankings de T2",
        },
        "criterio_de_fallo": "el documento relevante queda fuera del top 3 (rank > 3)",
        "tokenizacion_terminos_compartidos": ("analizador de TfidfVectorizer por defecto (minúsculas, tokens de >=2 caracteres, "
                                              "sin stemming ni stopwords); términos compartidos = intersección de tokens "
                                              "consulta-documento (siempre dentro del vocabulario)"),
        "fuentes_de_rankings": {
            "tfidf": "entrada/T1/resultados.json" if usar_t1 else "recalculado (T1 no disponible o incompleto)",
            "lsa": "entrada/T2/resultados.json" if usar_t2 else "recalculado (T2 no disponible o incompleto)",
        },
        "n_documentos": 10,
        "n_consultas": 6,
        "vocabulario_tamano": int(len(vocab)),
        "varianza_explicada_lsa_k4": var_exp,
    },
    "consultas_falladas_por_metodo": fallos,
    "fallos_detalle_por_metodo": fallos_detalle,
    "n_fallos_por_metodo": {"tfidf": len(fallos["tfidf"]), "lsa": len(fallos["lsa"])},
    "hit1_por_metodo": hit1,
    "hit3_por_metodo": hit3,
    "estado_por_consulta": estado_por_consulta,
    "tabla_terminos_compartidos_con_relevante": tabla_relevante,
    "terminos_compartidos_por_consulta": terminos_por_consulta,
    "matriz_solape_consulta_documento": matriz_solape,
    "explicacion_fallos": explicaciones,
    "notas_cambios_rank_sin_fallo": notas_cambios,
    "sintesis": sintesis,
    "verificacion_con_T1_T2": {
        "T1_cargado": t1 is not None,
        "T2_cargado": t2 is not None,
        "rankings_tfidf_tomados_de_T1": bool(usar_t1),
        "rankings_lsa_tomados_de_T2": bool(usar_t2),
        "recalculo_tfidf_coincide_con_T1": mismos_tfidf,
        "recalculo_lsa_coincide_con_T2": mismos_lsa,
        "ranks_coinciden_con_metricas_por_consulta_T1": coincide_metricas_t1,
        "ranks_coinciden_con_tabla_comparativa_T2": coincide_tab2,
        "hit3_tfidf_coincide_con_T1": hit3_t1_ok,
        "hit3_lsa_coincide_con_T2": hit3_t2_ok,
        "vocabulario_tamano_T1": tam_vocab_t1,
        "vocabulario_coincide_con_T1": vocab_coincide,
    },
    "figuras": [
        "t3_rank_relevante_por_metodo.png",
        "t3_heatmap_terminos_compartidos.png",
        "t3_q5_solape_por_documento.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 11) Resumen
# ---------------------------------------------------------------------------
print("=" * 74)
print("T3 · Análisis por consulta: fallos (relevante fuera del top 3) y términos compartidos")
print("=" * 74)
print(f"Vocabulario TF-IDF: {len(vocab)} términos"
      + (f" (coincide con T1: {tam_vocab_t1})" if vocab_coincide else ""))
print(f"Consultas falladas TF-IDF: {', '.join(fallos['tfidf']) if fallos['tfidf'] else 'ninguna'}")
print(f"Consultas falladas LSA   : {', '.join(fallos['lsa']) if fallos['lsa'] else 'ninguna'}")
print(f"Hit@1 -> TF-IDF: {hit1['tfidf']:.4f} | LSA: {hit1['lsa']:.4f}")
print(f"Hit@3 -> TF-IDF: {hit3['tfidf']:.4f} | LSA: {hit3['lsa']:.4f}")
print("-" * 74)
print("Términos compartidos entre cada consulta y su documento relevante:")
for fila in tabla_relevante:
    terminos = ", ".join(fila["terminos_compartidos"]) if fila["terminos_compartidos"] else "(ninguno)"
    print(f"  {fila['consulta']} - relevante {fila['documento_relevante']} "
          f"(rank TF-IDF={fila['rank_tfidf']}, rank LSA={fila['rank_lsa']}): "
          f"{fila['n_terminos_compartidos']} -> {terminos}")
print("-" * 74)
print("Explicación de los fallos:")
for m in ("tfidf", "lsa"):
    for texto in explicaciones[m].values():
        print(f"  {texto}")
for nota in notas_cambios:
    print(f"  nota: {nota}")
print("-" * 74)
print("Figuras PNG: " + ", ".join(resultados["figuras"]))
print("Contrato escrito en resultados.json")
