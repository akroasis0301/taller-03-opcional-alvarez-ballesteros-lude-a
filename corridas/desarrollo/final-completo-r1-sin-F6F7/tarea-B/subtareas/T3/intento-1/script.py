import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# T3 — Comprobación empírica
# 1) Cargar la distribución renormalizada (top-p p=0.9, T=1) de T2
# 2) Tomar 10,000 muestras con numpy.random.default_rng(0)
# 3) Gráfico de barras: frecuencias observadas vs probabilidades teóricas
# 4) Divergencia KL en bits de la empírica respecto de la teórica
# ---------------------------------------------------------------

N = 10000
SEED = 0
FIGURA = "fig_T3_frecuencias_vs_teoricas.png"

# --- 1) Distribución teórica: preferir el resultado de T2; si no existe, recalcular ---
ruta_t2 = Path("entrada/T2/resultados.json")
fuente = ""
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        res_t2 = json.load(f)
    dist_renorm = res_t2["distribucion_renormalizada"]
    fuente = "entrada/T2/resultados.json (clave 'distribucion_renormalizada')"
else:
    # Respaldo determinista: softmax(T=1) + top-p p=0.9 sobre los logits del enunciado
    tokens_fb = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()
    p_t1 = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p_t1, kind="stable")
    acum = np.cumsum(p_t1[orden])
    k = int(np.searchsorted(acum, 0.9) + 1)  # conjunto mínimo que alcanza 0.9
    nucleo = orden[:k]
    p_ren = np.zeros_like(p_t1)
    p_ren[nucleo] = p_t1[nucleo] / p_t1[nucleo].sum()
    dist_renorm = {t: float(p) for t, p in zip(tokens_fb, p_ren)}
    fuente = "recalculada desde logits (softmax T=1, top-p p=0.9)"

tokens = sorted(dist_renorm.keys())  # t0..t5
p_teorica = np.array([dist_renorm[t] for t in tokens], dtype=float)

# --- 2) Muestreo: 10,000 muestras con default_rng(0) ---
rng = np.random.default_rng(SEED)
indices = rng.choice(len(tokens), size=N, p=p_teorica)
conteos = np.bincount(indices, minlength=len(tokens))
frecuencias = conteos / N

# --- 3) Divergencia KL en bits: KL(empírica || teórica) = Σ q_i log2(q_i / p_i) ---
# Los tokens con p_teorica = 0 (t4, t5) no reciben muestras (q_i = 0 => término 0).
mask = frecuencias > 0
kl_bits = float(np.sum(frecuencias[mask] * np.log2(frecuencias[mask] / p_teorica[mask])))

# --- 4) Gráfico de barras: observado vs teórico por token ---
x = np.arange(len(tokens))
ancho = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - ancho / 2, p_teorica, width=ancho, color="#4C72B0",
            edgecolor="black", label="Probabilidad teórica (top-p renormalizada)")
b2 = ax.bar(x + ancho / 2, frecuencias, width=ancho, color="#DD8452",
            edgecolor="black", label=f"Frecuencia observada (n={N})")
for barras in (b1, b2):
    for rect in barras:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / Frecuencia")
ax.set_title("Parte 3 — Frecuencias observadas vs probabilidades teóricas\n"
             f"top-p p=0.9 (T=1), {N} muestras, default_rng({SEED}), KL = {kl_bits:.6f} bits")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURA, dpi=120)
plt.close(fig)

# --- 5) Resultados (contrato) ---
tabla_4dec = [[t, f"{p:.4f}", f"{q:.4f}", str(int(c))]
              for t, p, q, c in zip(tokens, p_teorica, frecuencias, conteos)]

resultados = {
    "subtarea": "T3 — Comprobación empírica: muestreo de la distribución renormalizada (top-p p=0.9, T=1) y divergencia KL",
    "n_muestras": N,
    "semilla": SEED,
    "generador": "numpy.random.default_rng(0)",
    "fuente_distribucion_teorica": fuente,
    "tokens": tokens,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(tokens, p_teorica)},
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(q) for t, q in zip(tokens, frecuencias)},
    "divergencia_KL_bits": kl_bits,
    "KL_bits": kl_bits,
    "definicion_KL": "KL(empírica || teórica) = Σ_i q_i * log2(q_i / p_i); términos con q_i = 0 aportan 0",
    "tabla_4_decimales": {
        "columnas": ["token", "prob_teorica", "frecuencia_observada", "conteo"],
        "filas": tabla_4dec,
    },
    "figura": FIGURA,
    "verificaciones": {
        "suma_probabilidades_teoricas": float(p_teorica.sum()),
        "suma_frecuencias_observadas": float(frecuencias.sum()),
        "total_muestras": int(conteos.sum()),
        "tokens_sin_muestras": [t for t, c in zip(tokens, conteos) if c == 0],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# --- Resumen breve ---
print("=== T3 — Comprobación empírica ===")
print(f"Fuente de la distribución teórica: {fuente}")
print(f"Muestras: {N} | Generador: numpy.random.default_rng({SEED})")
print(f"{'Token':<6}{'Teórica':>12}{'Observada':>12}{'Conteo':>10}")
for t, p, q, c in zip(tokens, p_teorica, frecuencias, conteos):
    print(f"{t:<6}{p:>12.6f}{q:>12.6f}{c:>10d}")
print(f"Divergencia KL (bits), KL(empírica || teórica): {kl_bits:.8f}")
print(f"Figura guardada: {FIGURA}")
print("Resultados escritos en resultados.json")
