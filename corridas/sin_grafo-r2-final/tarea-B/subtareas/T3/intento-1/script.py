import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# T3: Comprobación empírica del muestreo top-p (Parte 3)
# ---------------------------------------------------------------

# 1) Cargar la distribución renormalizada de la Parte 2 (T2)
ruta_t2 = Path("entrada/T2/resultados.json")
with open(ruta_t2, "r", encoding="utf-8") as f:
    t2 = json.load(f)

tokens = list(t2["tokens_sobrevivientes"])  # ["t0", "t1", "t2", "t3"]
p_teo = np.array([t2["distribucion_renormalizada"][t] for t in tokens], dtype=float)

# Verificación de sanidad de la distribución teórica
assert abs(p_teo.sum() - 1.0) < 1e-9, "La distribución renormalizada no suma 1"

# 2) Generar 10,000 muestras con la semilla exigida
N = 10000
rng = np.random.default_rng(0)
indices = np.arange(len(tokens))
muestras = rng.choice(indices, size=N, p=p_teo)

# 3) Frecuencias observadas
conteos = np.bincount(muestras, minlength=len(tokens))
p_emp = conteos / N

# 4) Divergencia KL de la empírica respecto de la teórica, en bits:
#    D_KL(P_emp || P_teo) = sum_i p_emp_i * log2(p_emp_i / p_teo_i)
# (todos los tokens del núcleo tienen p >= 0.09, así que no hay conteos nulos,
#  pero se protege el cálculo por robustez)
mask = p_emp > 0
kl_bits = float(np.sum(p_emp[mask] * np.log2(p_emp[mask] / p_teo[mask])))

# 5) Gráfico de barras: frecuencias observadas vs probabilidades teóricas
x = np.arange(len(tokens))
width = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width / 2, p_emp, width, label="Frecuencia observada (empírica)",
            color="#4C72B0", edgecolor="black")
b2 = ax.bar(x + width / 2, p_teo, width, label="Probabilidad teórica (renormalizada)",
            color="#DD8452", edgecolor="black")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.4f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad / frecuencia relativa")
ax.set_ylim(0, max(p_teo.max(), p_emp.max()) * 1.18)
ax.set_title(f"Muestreo del núcleo top-p = 0.9 (T = 1): {N} muestras, default_rng(0)\n"
             f"KL(empírica || teórica) = {kl_bits:.6f} bits")
ax.legend()
fig.tight_layout()
fig.savefig("barras_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# 6) Escribir resultados.json (contrato de la subtarea)
resultados = {
    "subtarea": "T3_comprobacion_empirica_top_p",
    "n_muestras": N,
    "semilla": 0,
    "rng": "numpy.random.default_rng(0)",
    "fuente": "distribucion_renormalizada de entrada/T2/resultados.json",
    "tokens": tokens,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(tokens, p_teo)},
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(f) for t, f in zip(tokens, p_emp)},
    "suma_frecuencias_observadas": float(p_emp.sum()),
    "divergencia_KL_bits": kl_bits,
    "divergencia_KL_bits_4dec": f"{kl_bits:.4f}",
    "figura": "barras_frecuencias_vs_teoricas.png",
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# 7) Resumen breve
print("T3 — Comprobación empírica del muestreo top-p (núcleo de la Parte 2)")
print(f"Muestras: {N} | Semilla: numpy.random.default_rng(0)")
print(f"{'token':<8}{'p_teorica':>14}{'conteo':>10}{'p_empirica':>14}")
for t, pt, c, pe in zip(tokens, p_teo, conteos, p_emp):
    print(f"{t:<8}{pt:>14.6f}{c:>10d}{pe:>14.6f}")
print(f"Divergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
print("Figura guardada: barras_frecuencias_vs_teoricas.png")
print("Resultados escritos en resultados.json")
