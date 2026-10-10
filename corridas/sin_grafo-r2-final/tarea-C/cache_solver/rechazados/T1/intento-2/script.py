# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica TF-IDF sobre el corpus de 10 documentos y 6 consultas.

- Corpus d01..d10: los diez textos EXACTOS de la tabla del enunciado (sin
  parafrasear ni ampliar). Los datos están completos en el enunciado: no se
  descarga nada.
- Consultas q1..q6 con sus juicios de relevancia: q1→d04, q2→d03, q3→d08,
  q4→d05, q5→d09, q6→d02.
- TfidfVectorizer (parámetros por defecto), ajustado SOLO con los 10
  documentos; las consultas únicamente se transforman.
- Ranking de los 10 documentos por similitud coseno para cada consulta.
- Métricas: Hit@1, Hit@3 y MRR sobre las 6 consultas.

Salidas: resultados.json (contrato) y t1_similitud_tfidf.png (mapa de calor).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1) Corpus de 10 documentos — textos EXACTOS de la tabla del enunciado
# ----------------------------------------------------------------------
corpus = {
    "d01": ("El mecanismo de atención pondera cada token según su similitud con la "
            "consulta; la atención escalada divide por la raíz de la dimensión."),
    "d02": ("La arquitectura Transformer apila bloques que combinan autoatención "
            "multicabeza con redes feed-forward y conexiones residuales."),
    "d03": ("En la generación aumentada por recuperación, los fragmentos recuperados de "
            "una base de conocimiento externa se añaden al prompt como contexto."),
    "d04": ("BM25 es una función de ranking léxica que pondera la frecuencia de términos "
            "con saturación y la rareza del término en la colección."),
    "d05": ("La similitud coseno compara la orientación de dos vectores mediante el "
            "coseno del ángulo entre ellos, sin depender de sus magnitudes."),
    "d06": ("Los embeddings representan palabras o documentos como vectores densos donde "
            "la cercanía refleja parecido semántico."),
    "d07": ("La tokenización divide el texto en subpalabras mediante algoritmos como "
            "BPE, equilibrando el tamaño del vocabulario y la longitud de la secuencia."),
    "d08": ("LoRA entrena matrices de bajo rango que se suman a los pesos congelados del "
            "modelo, reduciendo los parámetros a actualizar."),
    "d09": ("La temperatura escala los logits antes del muestreo: con valores bajos la "
            "distribución se concentra en los tokens probables y la generación resulta "
            "predecible."),
    "d10": ("La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar "
            "memoria."),
}

# ----------------------------------------------------------------------
# 2) Consultas y juicios de relevancia (según el enunciado)
# ----------------------------------------------------------------------
consultas = {
    "q1": {"texto": "¿Qué función de ranking léxica pondera la frecuencia de términos?",
           "relevante": "d04"},
    "q2": {"texto": "¿Cómo se añaden fragmentos recuperados al prompt?",
           "relevante": "d03"},
    "q3": {"texto": "¿Qué técnica entrena matrices de bajo rango?",
           "relevante": "d08"},
    "q4": {"texto": "¿Cómo se compara la orientación de dos vectores de texto?",
           "relevante": "d05"},
    "q5": {"texto": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
           "relevante": "d09"},
    "q6": {"texto": "¿Qué arquitectura combina autoatención con capas feed-forward?",
           "relevante": "d02"},
}

doc_ids = [f"d{i:02d}" for i in range(1, 11)]
q_ids = [f"q{i}" for i in range(1, 7)]
assert set(doc_ids) == set(corpus.keys()), "El corpus debe tener exactamente d01..d10"
assert set(q_ids) == set(consultas.keys()), "Debe haber exactamente q1..q6"
assert all(v["relevante"] in corpus for v in consultas.values()), "Relevante fuera del corpus"

doc_texts = [corpus[d] for d in doc_ids]
q_texts = [consultas[q]["texto"] for q in q_ids]

# ----------------------------------------------------------------------
# 3) Vectorización TF-IDF (por defecto) y similitud coseno
#    El vectorizador se ajusta SOLO con los documentos; las consultas
#    únicamente se transforman (no se ajusta nada con las consultas).
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(doc_texts)   # 10 x V
X_q = vectorizer.transform(q_texts)            #  6 x V
S = cosine_similarity(X_q, X_docs)             #  6 x 10

# ----------------------------------------------------------------------
# 4) Ranking por consulta y métricas Hit@1, Hit@3, MRR
# ----------------------------------------------------------------------
rankings = {}
rank_rel = {}
rr = {}
for i, qid in enumerate(q_ids):
    order = np.argsort(-S[i], kind="stable")  # descendente; empates en orden d01..d10
    ranked = [doc_ids[j] for j in order]
    rankings[qid] = [{"pos": p + 1, "doc": ranked[p], "score": float(S[i, order[p]])}
                     for p in range(len(ranked))]
    rel = consultas[qid]["relevante"]
    r = ranked.index(rel) + 1
    rank_rel[qid] = int(r)
    rr[qid] = 1.0 / r

hit_at_1 = float(np.mean([rank_rel[q] == 1 for q in q_ids]))
hit_at_3 = float(np.mean([rank_rel[q] <= 3 for q in q_ids]))
mrr = float(np.mean([rr[q] for q in q_ids]))

# ----------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1_linea_base_tfidf",
    "n_documentos": len(doc_ids),
    "n_consultas": len(q_ids),
    "orden_documentos": doc_ids,
    "corpus_textos_enunciado": corpus,
    "juicios_relevancia": {q: consultas[q]["relevante"] for q in q_ids},
    "consultas": {q: consultas[q]["texto"] for q in q_ids},
    "vectorizador": ("TfidfVectorizer con parámetros por defecto, ajustado únicamente "
                     "con los 10 documentos; similitud coseno para rankear."),
    "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    "ranking_por_consulta": rankings,          # ranking completo (10 docs) por consulta
    "rank_del_documento_relevante": rank_rel,  # posición (1-based) del relevante
    "reciprocal_rank_por_consulta": rr,
    "hit_at_1": hit_at_1,
    "hit_at_3": hit_at_3,
    "mrr": mrr,
    "matriz_similitud_coseno": {q: [float(x) for x in S[i]] for i, q in enumerate(q_ids)},
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6) Figura: mapa de calor de similitudes consulta-documento
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.6))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0)
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids, rotation=0)
ax.set_yticks(range(len(q_ids)))
ax.set_yticklabels([f"{q} -> {consultas[q]['relevante']}" for q in q_ids])
thr = 0.6 * float(S.max()) if S.max() > 0 else 1.0
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if S[i, j] > thr else "black")
fig.colorbar(im, ax=ax, label="similitud coseno")
ax.set_title("T1: similitud coseno TF-IDF (consulta x documento)")
fig.tight_layout()
fig.savefig("t1_similitud_tfidf.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 7) Resumen
# ----------------------------------------------------------------------
print("=" * 70)
print("T1 — Línea base TF-IDF (parámetros por defecto) + similitud coseno")
print("=" * 70)
for q in q_ids:
    top = rankings[q][0]
    ok = "HIT@1" if rank_rel[q] == 1 else ("top-3" if rank_rel[q] <= 3 else "FALLO")
    print(f"{q} | relevante={consultas[q]['relevante']} | top-1={top['doc']} "
          f"(score={top['score']:.4f}) | rank del relevante={rank_rel[q]} "
          f"[{ok}] | RR={rr[q]:.4f}")
print("-" * 70)
print(f"Hit@1 = {hit_at_1:.4f} ({int(round(hit_at_1 * len(q_ids)))}/{len(q_ids)})")
print(f"Hit@3 = {hit_at_3:.4f} ({int(round(hit_at_3 * len(q_ids)))}/{len(q_ids)})")
print(f"MRR   = {mrr:.4f}")
print("Archivos escritos: resultados.json, t1_similitud_tfidf.png")
