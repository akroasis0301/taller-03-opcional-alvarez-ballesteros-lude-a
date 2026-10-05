# =====================================================================
# T3 — Comprobación empírica de la distribución renormalizada (top-p)
# Toma 10,000 muestras con numpy.random.default_rng(0) de la distribución
# renormalizada de la Parte 2 (entrada/T2), dibuja un gráfico de barras
# comparando frecuencias observadas vs. probabilidades teóricas y calcula
# la divergencia KL (en bits) de la empírica respecto de la teórica.
# =====================================================================
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- Parámetros del enunciado -------------------------
N_MUESTRAS = 10_000          # muestras exigidas
SEMILLA = 0                  # numpy.random.default_rng(0)
P_TOP = 0.9                  # núcleo de la Parte 2
T = 1.0                      # temperatura de la Parte 1

TODOS_TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)

# ---------------------------------------------------------------------------
# 1) Distribución renormalizada de la Parte 2 (resultado de T2)
# ---------------------------------------------------------------------------
ruta_t2 = Path("entrada") / "T2" / "resultados.json"
res_t2 = None
if ruta_t2.exists():
    with ruta_t2.open("r", encoding="utf-8") as f:
        res_t2 = json.load(f)

# Recálculo independiente (softmax T=1 + top-p p=0.9) para verificación
z = LOGITS - LOGITS.max()
p_t1 = np.exp(z) / np.exp(z).sum()
orden = np.argsort(-p_t1, kind="stable")
acum = np.cumsum(p_t1[orden])
k = int(np.searchsorted(acum, P_TOP - 1e-12) + 1)  # conjunto mínimo que alcanza 0.9
idx_nucleo = np.sort(orden[:k])
p_teo_recalc = p_t1[idx_nucleo] / p_t1[idx_nucleo].sum()

if res_t2 is not None:
    tokens_nucleo = list(res_t2["tokens_supervivientes"])
    p_teo = np.array([float(res_t2["distribucion_renormalizada"][t])
                      for t in tokens_nucleo], dtype=float)
else:
    tokens_nucleo = [TODOS_TOKENS[i] for i in idx_nucleo]
    p_teo = p_teo_recalc.copy()

max_diff_t2_vs_recalc = float(np.max(np.abs(p_teo - p_teo_recalc)))

# ---------------------------------------------------------------------------
# 2) Muestreo: 10,000 muestras con numpy.random.default_rng(0)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEMILLA)
idx_muestras = rng.choice(len(tokens_nucleo), size=N_MUESTRAS, p=p_teo)

conteos = np.bincount(idx_muestras, minlength=len(tokens_nucleo)).astype(int)
f_obs = conteos / N_MUESTRAS  # distribución empírica

# ---------------------------------------------------------------------------
# 3) Divergencia KL en bits: D_KL(empírica || teórica) = Σ q_i log2(q_i / p_i)
# ---------------------------------------------------------------------------
mask = f_obs > 0  # términos con q_i = 0 aportan 0 por convención
kl_bits = float(np.sum(f_obs[mask] * np.log2(f_obs[mask] / p_teo[mask])))
kl_nats = float(np.sum(f_obs[mask] * np.log(f_obs[mask] / p_teo[mask])))

# ---------------------------------------------------------------------------
# 4) Gráfico de barras: frecuencias observadas vs. probabilidades teóricas
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2))
x = np.arange(len(tokens_nucleo))
w = 0.38
b1 = ax.bar(x - w / 2, f_obs, width=w, color="#4C72B0", edgecolor="black",
            label="Frecuencia observada (10,000 muestras)")
b2 = ax.bar(x + w / 2, p_teo, width=w, color="#DD8452", edgecolor="black",
            label="Probabilidad teórica (renormalizada, top-p 0.9)")
for barras in (b1, b2):
    for r in barras:
        h = r.get_height()
        ax.annotate(f"{h:.4f}", (r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens_nucleo)
ax.set_xlabel("Token del núcleo")
ax.set_ylabel("Probabilidad / frecuencia relativa")
ax.set_ylim(0, max(f_obs.max(), p_teo.max()) * 1.18)
ax.set_title("Parte 3 — Frecuencias observadas vs. probabilidades teóricas\n"
             f"top-p p={P_TOP} sobre softmax T={T} · {N_MUESTRAS:,} muestras · "
             f"rng({SEMILLA}) · KL = {kl_bits:.6f} bits")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("T3_frecuencias_observadas_vs_teoricas.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
tabla = [
    {
        "token": t,
        "prob_teorica": float(p),
        "prob_teorica_4dec": f"{p:.4f}",
        "conteo_observado": int(c),
        "frecuencia_observada": float(fq),
        "frecuencia_observada_4dec": f"{fq:.4f}",
        "diferencia_obs_menos_teo": float(fq - p),
    }
    for t, p, c, fq in zip(tokens_nucleo, p_teo, conteos, f_obs)
]

resultados = {
    "subtarea": ("T3 - Comprobación empírica: 10,000 muestras de la distribución "
                 "renormalizada (top-p p=0.9, T=1), gráfico de barras y divergencia KL en bits"),
    "fuente_distribucion_renormalizada": "entrada/T2/resultados.json",
    "temperatura": T,
    "p_top": P_TOP,
    "generador": "numpy.random.default_rng(0)",
    "semilla": SEMILLA,
    "n_muestras": N_MUESTRAS,
    "tokens_nucleo": tokens_nucleo,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(tokens_nucleo, p_teo)},
    "conteos_observados": {t: int(c) for t, c in zip(tokens_nucleo, conteos)},
    "frecuencias_observadas": {t: float(fq) for t, fq in zip(tokens_nucleo, f_obs)},
    "frecuencias_observadas_4dec": {t: f"{fq:.4f}" for t, fq in zip(tokens_nucleo, f_obs)},
    "probabilidades_teoricas_4dec": {t: f"{p:.4f}" for t, p in zip(tokens_nucleo, p_teo)},
    "divergencia_KL_bits": kl_bits,
    "divergencia_KL_bits_6dec": f"{kl_bits:.6f}",
    "divergencia_KL_nats": kl_nats,
    "tabla_comparativa": tabla,
    "figura": "T3_frecuencias_observadas_vs_teoricas.png",
    "verificaciones": {
        "suma_probabilidades_teoricas": float(p_teo.sum()),
        "suma_frecuencias_observadas": float(f_obs.sum()),
        "total_conteos": int(conteos.sum()),
        "max_abs_diff_T2_vs_recalculo_softmax_topp": max_diff_t2_vs_recalc,
        "tokens_con_conteo_cero": [t for t, c in zip(tokens_nucleo, conteos) if c == 0],
        "formula_KL": "D_KL(empirica || teorica) = sum_i q_i * log2(q_i / p_i)",
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
print("=" * 72)
print("T3 — Comprobación empírica (top-p p=0.9, T=1, rng(0), 10,000 muestras)")
print("=" * 72)
print(f"Tokens del núcleo: {tokens_nucleo}")
print(f"{'token':<6}{'p_teorica':>12}{'conteo':>10}{'f_obs':>12}{'diff':>12}")
for t, p, c, fq in zip(tokens_nucleo, p_teo, conteos, f_obs):
    print(f"{t:<6}{p:>12.6f}{c:>10d}{fq:>12.6f}{fq - p:>12.6f}")
print(f"\nDivergencia KL (empírica || teórica) = {kl_bits:.8f} bits "
      f"({kl_nats:.8f} nats)")
print("Figura guardada: T3_frecuencias_observadas_vs_teoricas.png")
print("Resultados guardados: resultados.json")
