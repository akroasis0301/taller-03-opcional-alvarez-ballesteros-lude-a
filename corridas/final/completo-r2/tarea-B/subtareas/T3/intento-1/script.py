import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# T3: Comprobación empírica de la distribución renormalizada (top-p, p=0.9)
# ---------------------------------------------------------------

# 1) Cargar la distribución renormalizada de la Parte 2 (subtarea T2)
ruta_t2 = Path("entrada/T2/resultados.json")
with open(ruta_t2, "r", encoding="utf-8") as f:
    t2 = json.load(f)

tokens = t2["tokens_sobrevivientes"]          # ["t0", "t1", "t2", "t3"]
dist_renorm = t2["distribucion_renormalizada"]

# Probabilidades teóricas (distribución renormalizada) en orden de tokens
p_teorica = np.array([dist_renorm[t] for t in tokens], dtype=float)
assert abs(p_teorica.sum() - 1.0) < 1e-12, "La distribución teórica debe sumar 1"

# 2) Tomar 10,000 muestras con numpy.random.default_rng(0)
SEMILLA = 0
N_MUESTRAS = 10_000
rng = np.random.default_rng(SEMILLA)
indices = rng.choice(len(tokens), size=N_MUESTRAS, p=p_teorica)

# 3) Distribución empírica (frecuencias observadas)
conteos = np.bincount(indices, minlength=len(tokens))
f_obs = conteos / N_MUESTRAS

# 4) Divergencia KL (empírica || teórica) en bits
#    D_KL(P||Q) = sum_i P_i * log2(P_i / Q_i); términos con P_i = 0 valen 0
mask = f_obs > 0
kl_bits = float(np.sum(f_obs[mask] * np.log2(f_obs[mask] / p_teorica[mask])))

# 5) Diagrama de barras: frecuencias observadas vs probabilidades teóricas
x = np.arange(len(tokens))
ancho = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - ancho / 2, f_obs, ancho, color="#4C72B0",
            edgecolor="black", label="Frecuencia observada (empírica)")
b2 = ax.bar(x + ancho / 2, p_teorica, ancho, color="#DD8452",
            edgecolor="black", label="Probabilidad teórica (renormalizada)")
for barras in (b1, b2):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / Frecuencia relativa")
ax.set_title("Parte 3 — Muestreo top-p (p=0.9): 10,000 muestras con default_rng(0)\n"
             f"Divergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
ax.set_ylim(0, max(f_obs.max(), p_teorica.max()) * 1.18)
ax.legend()
plt.tight_layout()
plt.savefig("T3_comparacion_frecuencias_kl.png", dpi=120)
plt.close(fig)

# 6) Escribir resultados.json (contrato de la subtarea)
resultados = {
    "subtarea": "T3_comprobacion_empirica_kl",
    "n_muestras": N_MUESTRAS,
    "semilla": SEMILLA,
    "generador": "numpy.random.default_rng(0)",
    "fuente_distribucion": "entrada/T2/resultados.json (distribución renormalizada top-p p=0.9 de la Parte 2)",
    "tokens_sobrevivientes": tokens,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(tokens, p_teorica)},
    "probabilidades_teoricas_lista": [float(p) for p in p_teorica],
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(fr) for t, fr in zip(tokens, f_obs)},
    "frecuencias_observadas_lista": [float(fr) for fr in f_obs],
    "suma_frecuencias_observadas": float(f_obs.sum()),
    "divergencia_KL_bits": kl_bits,
    "tabla_comparacion": {
        "columnas": ["token", "prob_teorica", "conteo_observado",
                     "frecuencia_observada", "diferencia_obs-teo"],
        "filas": [
            [t, f"{p:.4f}", int(c), f"{fr:.4f}", f"{fr - p:+.4f}"]
            for t, p, c, fr in zip(tokens, p_teorica, conteos, f_obs)
        ],
    },
    "figura": "T3_comparacion_frecuencias_kl.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# 7) Resumen breve
print("=== T3: Comprobación empírica (top-p p=0.9) ===")
print(f"Muestras: {N_MUESTRAS} con numpy.random.default_rng({SEMILLA})")
print(f"Tokens supervivientes: {tokens}")
for t, p, c, fr in zip(tokens, p_teorica, conteos, f_obs):
    print(f"  {t}: teórica={p:.4f} | observada={fr:.4f} (conteo={c})")
print(f"Divergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
print("Figura guardada: T3_comparacion_frecuencias_kl.png")
print("Resultados escritos en resultados.json")
