# Subtarea T3: Comprobación empírica de la distribución renormalizada (top-p p=0.9, T=1)
# - 10,000 muestras con numpy.random.default_rng(0)
# - Gráfico de barras: frecuencias observadas vs probabilidades teóricas
# - Divergencia KL (empírica || teórica) en bits

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Cargar la distribución renormalizada de la Parte 2 (entrada/T2)
# ------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    dist_renorm = t2["distribucion_renormalizada"]
    tokens = list(dist_renorm.keys())
    p_teorica = np.array([dist_renorm[t] for t in tokens], dtype=float)
    fuente = "entrada/T2/resultados.json"
else:
    # Respaldo: recalcular softmax(T=1) + top-p (p=0.9) a partir de los logits
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    tokens_todos = ["t0", "t1", "t2", "t3", "t4", "t5"]
    z = logits - logits.max()
    p_full = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p_full)
    acum = np.cumsum(p_full[orden])
    k = int(np.searchsorted(acum, 0.9) + 1)  # conjunto mínimo que alcanza 0.9
    supervivientes = np.sort(orden[:k])
    tokens = [tokens_todos[i] for i in supervivientes]
    p_teorica = p_full[supervivientes] / p_full[supervivientes].sum()
    fuente = "recalculada desde logits (softmax T=1 + top-p 0.9)"

p_teorica = p_teorica / p_teorica.sum()  # seguridad numérica

# ------------------------------------------------------------------
# 2. Tomar 10,000 muestras con numpy.random.default_rng(0)
# ------------------------------------------------------------------
rng = np.random.default_rng(0)
n_muestras = 10000
indices = rng.choice(len(tokens), size=n_muestras, p=p_teorica)

# ------------------------------------------------------------------
# 3. Distribución empírica (frecuencias observadas)
# ------------------------------------------------------------------
conteos = np.bincount(indices, minlength=len(tokens))
frecuencias = conteos / n_muestras

# ------------------------------------------------------------------
# 4. Divergencia KL en bits: KL(empírica || teórica) = Σ q_i log2(q_i / p_i)
# ------------------------------------------------------------------
mask = frecuencias > 0
kl_bits = float(np.sum(frecuencias[mask] * np.log2(frecuencias[mask] / p_teorica[mask])))

# ------------------------------------------------------------------
# 5. Gráfico de barras: frecuencias observadas vs probabilidades teóricas
# ------------------------------------------------------------------
x = np.arange(len(tokens))
ancho = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - ancho / 2, frecuencias, ancho,
            label="Frecuencia observada (empírica)", color="#4C72B0")
b2 = ax.bar(x + ancho / 2, p_teorica, ancho,
            label="Probabilidad teórica (renormalizada)", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_title("Parte 3 — Comprobación empírica\n"
             "Top-p (p=0.9, T=1): 10,000 muestras, numpy.random.default_rng(0)")
ymax = max(frecuencias.max(), p_teorica.max())
ax.set_ylim(0, ymax * 1.18)
for barra, valor in list(zip(b1, frecuencias)) + list(zip(b2, p_teorica)):
    ax.text(barra.get_x() + barra.get_width() / 2, valor + ymax * 0.012,
            f"{valor:.4f}", ha="center", va="bottom", fontsize=8)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("parte3_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 6. Escribir resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": ("Comprobación empírica: 10,000 muestras de la distribución renormalizada "
                    "de la Parte 2 (top-p p=0.9 sobre softmax T=1), comparación de frecuencias "
                    "observadas vs probabilidades teóricas y divergencia KL en bits"),
    "parametros": {
        "n_muestras": n_muestras,
        "semilla": 0,
        "generador": "numpy.random.default_rng(0)",
        "p_top_p": 0.9,
        "temperatura": 1.0,
    },
    "fuente_distribucion": fuente,
    "tokens": tokens,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(tokens, p_teorica)},
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(fr) for t, fr in zip(tokens, frecuencias)},
    "tabla_comparativa": [
        {
            "token": t,
            "frecuencia_observada": float(fr),
            "probabilidad_teorica": float(p),
            "diferencia_observada_menos_teorica": float(fr - p),
        }
        for t, fr, p in zip(tokens, frecuencias, p_teorica)
    ],
    "divergencia_kl_bits": kl_bits,
    "definicion_kl": "KL(empírica || teórica) = Σ q_i * log2(q_i / p_i), sobre los tokens supervivientes",
    "figura": "parte3_frecuencias_vs_teoricas.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 7. Resumen
# ------------------------------------------------------------------
print("=== Subtarea T3 — Comprobación empírica ===")
print(f"Fuente de la distribución: {fuente}")
print(f"Muestras: {n_muestras} con numpy.random.default_rng(0)")
print(f"Tokens supervivientes (top-p p=0.9): {tokens}")
print("Token | frecuencia observada | prob. teórica | diferencia")
for t, fr, p in zip(tokens, frecuencias, p_teorica):
    print(f"  {t}  |  {fr:.6f}  |  {p:.6f}  |  {fr - p:+.6f}")
print(f"Divergencia KL (empírica || teórica) en bits: {kl_bits:.8f}")
print("Figura guardada: parte3_frecuencias_vs_teoricas.png")
print("Resultados escritos en resultados.json")
