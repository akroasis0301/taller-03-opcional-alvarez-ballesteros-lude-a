import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Subtarea T3 — Parte 3: Comprobación empírica del muestreo top-p
#   * 10,000 muestras de la distribución renormalizada de la Parte 2
#   * generador numpy.random.default_rng(0)
#   * Diagrama de barras: frecuencias observadas vs. probabilidades teóricas
#   * Divergencia KL (en bits) de la distribución empírica respecto de la teórica
# ----------------------------------------------------------------------

N_MUESTRAS = 10_000
SEMILLA = 0
P_TOP = 0.9
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
FIGURA = "T3_frecuencias_vs_teorica.png"


def softmax_T1(logits):
    """Softmax con T=1, numéricamente estable (resta del máximo)."""
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def top_p_renormalizada(p, p_top):
    """Conjunto mínimo de tokens cuya acumulada alcanza p_top; renormaliza."""
    orden = np.argsort(-p, kind="stable")
    acum = np.cumsum(p[orden])
    k = int(np.searchsorted(acum, p_top, side="left")) + 1
    sup = np.sort(orden[:k])
    q = np.zeros_like(p)
    q[sup] = p[sup] / p[sup].sum()
    return q, [TOKENS[i] for i in sup]


# --- 1) Distribución renormalizada de la Parte 2 (salida de T2) --------
ruta_t2 = Path("entrada") / "T2" / "resultados.json"
fuente = "entrada/T2/resultados.json"
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    p_renorm = np.asarray(t2["distribucion_renormalizada_vector_alineado_t0_t5"], dtype=float)
    tokens_sup = list(t2["tokens_supervivientes"])
else:
    fuente = "recalculada en T3 (softmax T=1 + top-p 0.9)"
    p_renorm, tokens_sup = top_p_renormalizada(softmax_T1(LOGITS), P_TOP)

# Verificación determinista: el recálculo debe coincidir con T2
p_renorm_check, tokens_sup_check = top_p_renormalizada(softmax_T1(LOGITS), P_TOP)
max_diff_check = float(np.max(np.abs(p_renorm_check - p_renorm)))

# --- 2) Muestreo: 10,000 muestras con default_rng(0) -------------------
p = p_renorm / p_renorm.sum()  # garantiza suma exactamente 1 para rng.choice
rng = np.random.default_rng(SEMILLA)
muestras = rng.choice(len(TOKENS), size=N_MUESTRAS, p=p)
conteos = np.bincount(muestras, minlength=len(TOKENS))
frecuencias = conteos / N_MUESTRAS

# --- 3) Divergencia KL en bits: D(empírica || teórica) -----------------
mask = frecuencias > 0  # los términos con q_i = 0 aportan 0 a la suma
kl_bits = float(np.sum(frecuencias[mask] * np.log2(frecuencias[mask] / p[mask])))
kl_nats = float(np.sum(frecuencias[mask] * np.log(frecuencias[mask] / p[mask])))
# Referencia asintótica: E[KL] ≈ (k-1)/(2n) nats con k categorías con p>0
kl_ref_bits = float((len(tokens_sup) - 1) / (2.0 * N_MUESTRAS * np.log(2.0)))

# --- 4) Diagrama de barras: observado vs. teórico ----------------------
idx_sup = [TOKENS.index(t) for t in tokens_sup]
probs_sup = p_renorm[idx_sup]
freqs_sup = frecuencias[idx_sup]

x = np.arange(len(tokens_sup))
w = 0.38
fig, ax = plt.subplots(figsize=(8.0, 5.0))
b_teo = ax.bar(x - w / 2, probs_sup, w, color="#4C72B0", edgecolor="black",
               linewidth=0.5, label="Probabilidad teórica (renormalizada)")
b_obs = ax.bar(x + w / 2, freqs_sup, w, color="#DD8452", edgecolor="black",
               linewidth=0.5, label=f"Frecuencia observada (n={N_MUESTRAS:,})")
for barras in (b_teo, b_obs):
    for rect in barras:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens_sup)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / Frecuencia")
ax.set_ylim(0.0, 1.15 * max(probs_sup.max(), freqs_sup.max()))
ax.set_title("Parte 3 — Muestreo top-p (T=1, p=0.9): observado vs. teórico\n"
             f"10,000 muestras, numpy.random.default_rng({SEMILLA}) · KL = {kl_bits:.6f} bits")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURA, dpi=120)
plt.close(fig)

# --- 5) resultados.json (contrato de la subtarea) ----------------------
resultados = {
    "subtarea": "T3_comprobacion_empirica",
    "parte": "Parte 3 — Comprobación empírica",
    "fuente_distribucion_renormalizada": fuente,
    "temperatura": 1.0,
    "p_top": P_TOP,
    "semilla": SEMILLA,
    "generador": f"numpy.random.default_rng({SEMILLA})",
    "n_muestras": N_MUESTRAS,
    "tokens": TOKENS,
    "tokens_supervivientes": tokens_sup,
    "probabilidades_teoricas_renormalizadas": {
        t: float(p_renorm[i]) for i, t in enumerate(TOKENS) if p_renorm[i] > 0.0
    },
    "probabilidades_teoricas_alineadas_t0_t5": [float(v) for v in p_renorm],
    "conteos_observados": {t: int(conteos[i]) for i, t in enumerate(TOKENS)},
    "frecuencias_observadas": {t: float(frecuencias[i]) for i, t in enumerate(TOKENS)},
    "frecuencias_observadas_alineadas_t0_t5": [float(v) for v in frecuencias],
    "suma_frecuencias_observadas": float(frecuencias.sum()),
    "divergencia_KL_bits": kl_bits,
    "divergencia_KL_nats": kl_nats,
    "KL_esperada_bits_referencia_asintotica": kl_ref_bits,
    "verificacion_recalculo_softmax_topp_max_abs_diff": max_diff_check,
    "tokens_supervivientes_verificacion": tokens_sup_check,
    "figura": FIGURA,
    "tabla_comparativa": [
        {
            "token": tok,
            "prob_teorica_renormalizada": float(p_renorm[TOKENS.index(tok)]),
            "conteo_observado": int(conteos[TOKENS.index(tok)]),
            "frecuencia_observada": float(frecuencias[TOKENS.index(tok)]),
            "diferencia_obs_menos_teo": float(
                frecuencias[TOKENS.index(tok)] - p_renorm[TOKENS.index(tok)]
            ),
        }
        for tok in tokens_sup
    ],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# --- 6) Resumen ---------------------------------------------------------
print("=== T3 — Comprobación empírica (Parte 3) ===")
print(f"Fuente distribución renormalizada   : {fuente}")
print(f"Tokens supervivientes (top-p {P_TOP})    : {tokens_sup}")
print(f"Muestras: {N_MUESTRAS} con numpy.random.default_rng({SEMILLA})")
for tok in tokens_sup:
    i = TOKENS.index(tok)
    print(f"  {tok}: teórica={p_renorm[i]:.6f}  observada={frecuencias[i]:.6f}  (conteo={conteos[i]})")
print(f"Divergencia KL(empírica || teórica) = {kl_bits:.6f} bits ({kl_nats:.6f} nats)")
print(f"Referencia asintótica E[KL] ≈ {kl_ref_bits:.6e} bits")
print(f"Verificación recálculo softmax+top-p: max|diff| = {max_diff_check:.3e}")
print(f"Figura guardada: {FIGURA}")
print("Resultados escritos en resultados.json")
