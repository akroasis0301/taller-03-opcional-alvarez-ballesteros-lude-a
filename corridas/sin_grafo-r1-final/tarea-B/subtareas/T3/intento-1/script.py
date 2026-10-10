import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# T3: Comprobación empírica de la distribución renormalizada (top-p=0.9)
# 1) Muestrear 10,000 veces con numpy.random.default_rng(0)
# 2) Gráfico de barras: frecuencias observadas vs probabilidades teóricas
# 3) Divergencia KL en bits de la empírica respecto de la teórica
# ------------------------------------------------------------------

# --- 1. Cargar la distribución renormalizada de la subtarea T2 ---
ruta_t2 = Path("entrada/T2/resultados.json")
with open(ruta_t2, "r", encoding="utf-8") as f:
    res_t2 = json.load(f)

tokens = list(res_t2["tokens_supervivientes"])
probs_teoricas = np.array(
    [res_t2["distribucion_renormalizada"][t] for t in tokens], dtype=float
)
probs_teoricas = probs_teoricas / probs_teoricas.sum()  # seguridad numérica

# --- 2. Muestreo de 10,000 muestras con semilla 0 ---
N = 10000
rng = np.random.default_rng(0)
indices = np.arange(len(tokens))
muestras = rng.choice(indices, size=N, p=probs_teoricas)

conteos = np.bincount(muestras, minlength=len(tokens))
frecuencias_obs = conteos / N

# --- 3. Divergencia KL en bits: D_KL(empírica || teórica) ---
# KL(q||p) = sum_i q_i * log2(q_i / p_i), con la convención 0*log(0/p)=0
mask = frecuencias_obs > 0
kl_bits = float(
    np.sum(
        frecuencias_obs[mask]
        * np.log2(frecuencias_obs[mask] / probs_teoricas[mask])
    )
)

# --- 4. Gráfico de barras: observado vs teórico ---
x = np.arange(len(tokens))
ancho = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(
    x - ancho / 2, frecuencias_obs, width=ancho,
    label="Frecuencia observada (empírica)", color="#4C72B0", edgecolor="black",
)
b2 = ax.bar(
    x + ancho / 2, probs_teoricas, width=ancho,
    label="Probabilidad teórica (renormalizada top-p=0.9)",
    color="#DD8452", edgecolor="black",
)
for rect in list(b1) + list(b2):
    h = rect.get_height()
    ax.annotate(
        f"{h:.4f}",
        xy=(rect.get_x() + rect.get_width() / 2, h),
        xytext=(0, 3), textcoords="offset points",
        ha="center", va="bottom", fontsize=8,
    )
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad / Frecuencia relativa")
ax.set_ylim(0, max(frecuencias_obs.max(), probs_teoricas.max()) * 1.18)
ax.set_title(
    f"Muestreo de la distribución renormalizada (núcleo top-p = 0.9)\n"
    f"N = {N} muestras, semilla = 0  |  KL(empírica || teórica) = {kl_bits:.6f} bits"
)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("barras_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# --- 5. Escribir resultados.json (contrato de la subtarea) ---
resultados = {
    "subtarea": "T3_comprobacion_empirica_kl",
    "distribucion_muestreada": "renormalizada top-p=0.9 (Parte 2)",
    "num_muestras": N,
    "semilla": 0,
    "generador": "numpy.random.default_rng(0)",
    "tokens": tokens,
    "probabilidades_teoricas": {
        t: float(p) for t, p in zip(tokens, probs_teoricas)
    },
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {
        t: float(fr) for t, fr in zip(tokens, frecuencias_obs)
    },
    "suma_frecuencias_observadas": float(frecuencias_obs.sum()),
    "divergencia_KL_bits": kl_bits,
    "figura": "barras_frecuencias_vs_teoricas.png",
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# --- 6. Resumen ---
print("=== T3: Comprobación empírica de la distribución renormalizada ===")
print(f"Muestras: {N} | Generador: numpy.random.default_rng(0)")
print(f"{'Token':<10}{'Conteo':>8}{'Freq. obs.':>14}{'P teórica':>14}")
for t, c, fo, pt in zip(tokens, conteos, frecuencias_obs, probs_teoricas):
    print(f"{t:<10}{c:>8d}{fo:>14.6f}{pt:>14.6f}")
print(f"\nDivergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
print("Figura guardada: barras_frecuencias_vs_teoricas.png")
print("Resultados escritos en resultados.json")
