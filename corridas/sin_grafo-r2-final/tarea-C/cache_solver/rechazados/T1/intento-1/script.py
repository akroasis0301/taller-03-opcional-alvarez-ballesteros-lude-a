# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica TF-IDF sobre un corpus de 10 documentos y 6 consultas.

- Corpus de 10 documentos (d01..d10) y 6 consultas (q1..q6) con juicios de
  relevancia según el enunciado: q1->d04, q2->d03, q3->d08, q4->d05, q5->d09, q6->d02.
- TfidfVectorizer (parámetros por defecto), ajustado SOLO con los 10 documentos;
  las consultas únicamente se transforman.
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
# 1) Corpus de 10 documentos (español, temas distintos entre sí)
# ----------------------------------------------------------------------
corpus = {
    "d01": ("Los embeddings representan palabras o documentos como vectores densos de números "
            "reales en un espacio continuo. Entrenados sobre grandes colecciones, capturan "
            "parecido semántico: las unidades con significados afines quedan cerca en ese "
            "espacio geométrico y sirven de entrada a muchos sistemas modernos de PLN."),
    "d02": ("La arquitectura Transformer apila capas donde cada bloque combina autoatención "
            "multicabeza con una red feed-forward, junto con conexiones residuales y "
            "normalización. Al prescindir de la recurrencia, paraleliza el entrenamiento y "
            "captura dependencias largas; es la base de los grandes modelos de lenguaje actuales."),
    "d03": ("En la generación aumentada por recuperación (RAG), el sistema busca en una base de "
            "conocimiento externa y los fragmentos recuperados se añaden al prompt como "
            "contexto; así el modelo redacta una respuesta apoyada en evidencia actual y puede "
            "citar sus fuentes."),
    "d04": ("BM25 es una función de ranking léxica clásica de la búsqueda: pondera la frecuencia "
            "de términos en el documento, la rareza de cada término en la colección (idf) y la "
            "longitud del documento con saturación; sigue siendo una referencia fuerte para la "
            "búsqueda por palabras clave."),
    "d05": ("La similitud coseno compara la orientación de dos vectores: calcula el coseno del "
            "ángulo entre ellos, de modo que dos textos con la misma dirección, aunque con "
            "magnitudes distintas, obtienen un valor cercano a uno; es la medida estándar para "
            "comparar representaciones vectoriales de texto."),
    "d06": ("La tokenización divide el texto crudo en unidades que el modelo puede consumir. Los "
            "métodos de subpalabra, como BPE o WordPiece, equilibran tamaño de vocabulario y "
            "longitud de secuencia: las palabras frecuentes se vuelven un solo token y las raras "
            "se parten en piezas menores, lo que evita tokens desconocidos."),
    "d07": ("El ajuste fino supervisado toma un modelo preentrenado y continúa su entrenamiento "
            "con ejemplos etiquetados de la tarea objetivo, actualizando todos los pesos o solo "
            "una parte; con instrucciones y diálogos de ejemplo, el modelo aprende a seguir "
            "órdenes y a responder en formatos útiles."),
    "d08": ("LoRA es una técnica de adaptación eficiente que entrena matrices de bajo rango junto "
            "a los pesos congelados del modelo: en lugar de actualizar toda la red, aprende las "
            "descomposiciones A y B cuyo producto aproxima el cambio necesario, lo que reduce "
            "memoria y costo de cómputo."),
    "d09": ("La temperatura controla la aleatoriedad del decodificador: al dividir los logits por "
            "un número pequeño la distribución se afila, lo que hace que el modelo sea más "
            "determinista al elegir la siguiente palabra; con valores altos el muestreo se "
            "vuelve más diverso y arriesgado."),
    "d10": ("La evaluación de modelos de lenguaje mezcla pruebas automáticas y juicio humano: se "
            "mide exactitud, F1 o coincidencia exacta sobre conjuntos de prueba reservados, y se "
            "complementa con comparaciones pareadas y rúbricas; un buen protocolo separa los "
            "datos de evaluación del entrenamiento para detectar contaminación."),
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
#    únicamente se transforman (no se ajusta nada con datos de prueba).
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
    "juicios_relevancia": {q: consultas[q]["relevante"] for q in q_ids},
    "consultas": {q: consultas[q]["texto"] for q in q_ids},
    "corpus": corpus,
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
    print(f"{q} | relevante={consultas[q]['relevante']} | top-1={top['doc']} "
          f"(score={top['score']:.4f}) | rank del relevante={rank_rel[q]} | RR={rr[q]:.4f}")
print("-" * 70)
print(f"Hit@1 = {hit_at_1:.4f} ({int(round(hit_at_1 * len(q_ids)))}/{len(q_ids)})")
print(f"Hit@3 = {hit_at_3:.4f} ({int(round(hit_at_3 * len(q_ids)))}/{len(q_ids)})")
print(f"MRR   = {mrr:.4f}")
print("Archivos escritos: resultados.json, t1_similitud_tfidf.png")
