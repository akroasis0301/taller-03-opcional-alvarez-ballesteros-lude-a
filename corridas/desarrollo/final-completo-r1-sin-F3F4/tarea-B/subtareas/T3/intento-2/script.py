# -*- coding: utf-8 -*-
"""
T3 — Comprobación empírica (Parte 3 de la Tarea B).

- Toma 10,000 muestras de la distribución renormalizada de la Parte 2
  (softmax con T=1 + top-p con p=0.9) usando numpy.random.default_rng(0).
- Compara las frecuencias observadas con las probabilidades teóricas
  en un gráfico de barras (PNG).
- Calcula la divergencia KL (en bits) de la distribución empírica
  respecto de la teórica: KL(P_emp || P_teo) = Σ_i P_emp(i) log2(P_emp(i)/P_teo(i)).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Parámetros fijos -----------------------------
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
TEMPERATURA = 1.0
P_NUCLEO = 0.9
N_MUESTRAS = 10_000
SEMILLA = 0
FIGURA = "T3_comparacion_frecuencias.png"

# ------- 1) Distribución teórica: softmax(T=1) + top-p(0.9) renormalizada -------
def softmax_con_temperatura(z, temperatura=1.0):
    z = np.asarray(z, dtype=float)
    z = z - z.max()  # estabilidad numérica
    e = np.exp(z / temperatura)
    return e / e.sum()

p_T1 = softmax_con_temperatura(LOGITS, TEMPERATURA)
orden = np.argsort(-p_T1, kind="stable")           # prob. descendente
acumulada = np.cumsum(p_T1[orden])
k = int(np.searchsorted(acumulada, P_NUCLEO) + 1)  # conjunto mínimo que alcanza p=0.9
nucleo = orden[:k]
p_recalculada = np.zeros_like(p_T1)
p_recalculada[nucleo] = p_T1[nucleo] / p_T1[nucleo].sum()

# Usar la distribución oficial de la Parte 2 (entrada/T2/resultados.json) si existe
p_teorica = p_recalculada
fuente = "recalculada desde los logits (softmax T=1 + top-p p=0.9)"
consistente_con_T2 = None
ruta_T2 = Path("entrada/T2/resultados.json")
if ruta_T2.exists():
    with open(ruta_T2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    p_T2 = np.array([float(t2["distribucion_renormalizada"][t]) for t in TOKENS])
    consistente_con_T2 = bool(np.allclose(p_T2, p_recalculada, rtol=0.0, atol=1e-12))
    p_teorica = p_T2
    fuente = "entrada/T2/resultados.json (distribucion_renormalizada de la Parte 2)"

tokens_sobrevivientes = [TOKENS[i] for i in np.sort(nucleo)]

# ----------------------------- 2) Muestreo -----------------------------
rng = np.random.default_rng(SEMILLA)
muestras = rng.choice(len(TOKENS), size=N_MUESTRAS, p=p_teorica)
conteos = np.bincount(muestras, minlength=len(TOKENS)).astype(float)
p_emp = conteos / N_MUESTRAS

# ----------------------------- 3) Divergencia KL (bits) -----------------------------
# KL(P_emp || P_teo) = Σ_i P_emp(i) * log2(P_emp(i) / P_teo(i))
# Términos con P_emp(i)=0 aportan 0 por convención; como se muestrea de P_teo,
# P_teo(i)>0 siempre que P_emp(i)>0 (no hay infinitos).
mask = p_emp > 0
kl_bits = float(np.sum(p_emp[mask] * np.log2(p_emp[mask] / p_teorica[mask])))

# Entropía empírica en bits (informativo)
entropia_emp_bits = float(-np.sum(p_emp[mask] * np.log2(p_emp[mask])))

# ----------------------------- 4) Gráfico de barras -----------------------------
x = np.arange(len(TOKENS))
w = 0.38
etiqueta_obs = f"Frecuencia observada ({N_MUESTRAS} muestras)"
etiqueta_teo = "Probabilidad teórica (top-p p=0.9 renormalizada)"
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(x - w / 2, p_emp, w, color="#4C72B0", label=etiqueta_obs)
ax.bar(x + w / 2, p_teorica, w, color="#DD8452", label=etiqueta_teo)
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia relativa")
ax.set_title("Parte 3 — Comprobación empírica del muestreo top-p (T=1, p=0.9)\n"
             f"KL(empírica ‖ teórica) = {kl_bits:.6f} bits")
ax.set_ylim(0.0, max(p_teorica.max(), p_emp.max()) * 1.18)
for xi, e, th in zip(x, p_emp, p_teorica):
    ax.text(xi - w / 2, e + 0.004, f"{e:.4f}", ha="center", va="bottom", fontsize=8)
    ax.text(xi + w / 2, th + 0.004, f"{th:.4f}", ha="center", va="bottom", fontsize=8)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGURA, dpi=120)
plt.close(fig)

# ----------------------------- 5) resultados.json -----------------------------
resultados = {
    "subtarea": "T3 — Comprobación empírica: 10,000 muestras de la distribución "
                "renormalizada de la Parte 2 (top-p p=0.9, T=1), comparación "
                "observado vs teórico y divergencia KL en bits",
    "fuente_distribucion": fuente,
    "generador": "numpy.random.default_rng(0)",
    "semilla": SEMILLA,
    "n_muestras": N_MUESTRAS,
    "tokens": TOKENS,
    "tokens_sobrevivientes_top_p": tokens_sobrevivientes,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(TOKENS, p_teorica)},
    "conteos_observados": {t: int(c) for t, c in zip(TOKENS, conteos)},
    "frecuencias_observadas": {t: float(f) for t, f in zip(TOKENS, p_emp)},
    "diferencia_absoluta_obs_teo": {t: float(abs(e - th))
                                    for t, e, th in zip(TOKENS, p_emp, p_teorica)},
    "divergencia_KL_bits": kl_bits,
    "kl_bits_empirica_respecto_teorica": kl_bits,
    "entropia_bits_empirica": entropia_emp_bits,
    "primeras_10_muestras": [TOKENS[i] for i in muestras[:10]],
    "figura": FIGURA,
    "verificacion": {
        "total_muestras": int(conteos.sum()),
        "suma_frecuencias_observadas": float(p_emp.sum()),
        "suma_probabilidades_teoricas": float(p_teorica.sum()),
        "consistente_con_T2": consistente_con_T2,
        "kl_pequena_cercana_a_cero": bool(kl_bits < 0.01),
        "tokens_sin_muestras": [TOKENS[i] for i in range(len(TOKENS)) if conteos[i] == 0],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------- 6) Resumen -----------------------------
print("=" * 64)
print("T3 — Comprobación empírica (top-p p=0.9, T=1)")
print("=" * 64)
print(f"Generador: numpy.random.default_rng({SEMILLA}) | Muestras: {N_MUESTRAS}")
print(f"Distribución teórica: {fuente}")
print(f"Tokens del núcleo: {tokens_sobrevivientes}")
print("-" * 64)
print(f"{'Token':<6}{'Teórica':>12}{'Observada':>12}{'Conteo':>10}{'|diff|':>12}")
for t, th, e, c in zip(TOKENS, p_teorica, p_emp, conteos):
    print(f"{t:<6}{th:>12.6f}{e:>12.6f}{int(c):>10d}{abs(e - th):>12.6f}")
print("-" * 64)
print(f"Divergencia KL(empírica || teórica) = {kl_bits:.6f} bits")
print(f"Entropía empírica = {entropia_emp_bits:.6f} bits")
print(f"Figura guardada: {FIGURA}")
print("Resultados escritos en resultados.json")
