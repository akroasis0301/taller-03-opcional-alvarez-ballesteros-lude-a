# -*- coding: utf-8 -*-
"""
SUBTAREA T3: Comprobación empírica de la distribución renormalizada de T2.
- Toma 10,000 muestras con numpy.random.default_rng(0).
- Grafica en barras las frecuencias observadas frente a las probabilidades teóricas.
- Calcula la divergencia KL (en bits) de la distribución empírica respecto de la teórica.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 1. Cargar la distribución renormalizada de T2 ----------
with open("entrada/T2/resultados.json", "r", encoding="utf-8") as f:
    t2 = json.load(f)

tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
renorm = t2["distribucion_renormalizada"]  # contiene solo los tokens supervivientes
p_teo = np.array([renorm.get(tok, 0.0) for tok in tokens], dtype=float)

assert abs(p_teo.sum() - 1.0) < 1e-9, "La distribución teórica debe sumar 1"

# ---------- 2. Muestreo: 10,000 muestras con default_rng(0) ----------
N = 10_000
rng = np.random.default_rng(0)
idx = np.arange(len(tokens))
muestras = rng.choice(idx, size=N, p=p_teo)

conteos = np.bincount(muestras, minlength=len(tokens))
f_obs = conteos / N

# ---------- 3. Divergencia KL (bits): KL(P_emp || P_teo) ----------
# KL = sum_i p_emp_i * log2(p_emp_i / p_teo_i)
# Los términos con p_emp_i = 0 contribuyen 0 (límite x*log(x)->0).
# p_teo_i = 0 (t4, t5) solo ocurre donde p_emp_i = 0 (nunca se muestrean),
# por lo que la divergencia está bien definida.
mask = f_obs > 0
kl_bits = float(np.sum(f_obs[mask] * np.log2(f_obs[mask] / p_teo[mask])))
kl_nats = float(np.sum(f_obs[mask] * np.log(f_obs[mask] / p_teo[mask])))

# ---------- 4. Gráfico de barras: observado vs teórico ----------
x = np.arange(len(tokens))
width = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - width / 2, f_obs, width,
            label="Frecuencia observada (10,000 muestras)",
            color="#4C72B0", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + width / 2, p_teo, width,
            label="Probabilidad teórica (top-p p=0.9 renormalizada)",
            color="#DD8452", edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_title("Frecuencias observadas vs probabilidades teóricas\n"
             "Distribución renormalizada top-p (T=1, p=0.9); "
             "10,000 muestras, default_rng(0)")
ax.legend(loc="upper right")
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, max(f_obs.max(), p_teo.max()) * 1.18)
plt.tight_layout()
plt.savefig("T3_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# ---------- 5. Guardar resultados.json (contrato de la subtarea) ----------
resultados = {
    "subtarea": ("T3 - Comprobacion empirica: 10,000 muestras de la distribucion "
                 "renormalizada de T2, comparacion de frecuencias observadas vs "
                 "probabilidades teoricas y divergencia KL en bits"),
    "generador": "numpy.random.default_rng(0)",
    "n_muestras": N,
    "fuente_distribucion_teorica": "entrada/T2/resultados.json (distribucion_renormalizada)",
    "distribucion_teorica_renormalizada": {tok: float(p) for tok, p in zip(tokens, p_teo)},
    "conteos_observados": {tok: int(c) for tok, c in zip(tokens, conteos)},
    "frecuencias_observadas": {tok: float(fr) for tok, fr in zip(tokens, f_obs)},
    "comparacion_por_token": [
        {"token": tok,
         "frecuencia_observada": float(fr),
         "probabilidad_teorica": float(p),
         "diferencia_obs_menos_teo": float(fr - p)}
        for tok, fr, p in zip(tokens, f_obs, p_teo)
    ],
    "kl_bits_empirica_respecto_teorica": kl_bits,
    "kl_nats_empirica_respecto_teorica": kl_nats,
    "definicion_kl": ("KL(P_emp || P_teo) = sum_i p_emp_i * log2(p_emp_i / p_teo_i); "
                      "los terminos con p_emp_i = 0 contribuyen 0. Los tokens t4 y t5 "
                      "tienen probabilidad teorica 0 y recibieron 0 muestras."),
    "figura": "T3_frecuencias_vs_teoricas.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------- 6. Resumen ----------
print("=== T3: Comprobacion empirica ===")
print(f"Muestras: {N} con numpy.random.default_rng(0)")
print(f"{'token':<6}{'conteo':>10}{'observada':>12}{'teorica':>12}")
for tok, c, fr, p in zip(tokens, conteos, f_obs, p_teo):
    print(f"{tok:<6}{int(c):>10d}{fr:>12.6f}{p:>12.6f}")
print(f"Divergencia KL (empirica || teorica) = {kl_bits:.6f} bits "
      f"({kl_nats:.6f} nats)")
print("Figura guardada: T3_frecuencias_vs_teoricas.png")
print("Resultados guardados en resultados.json")
