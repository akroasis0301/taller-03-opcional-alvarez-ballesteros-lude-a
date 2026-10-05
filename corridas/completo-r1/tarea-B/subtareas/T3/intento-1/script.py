# T3: Comprobación empírica del muestreo top-p (Parte 3)
# - Toma 10,000 muestras de la distribución renormalizada de T2 con numpy.random.default_rng(0)
# - Grafica frecuencias observadas vs probabilidades teóricas (barras)
# - Calcula la divergencia KL en bits de la empírica respecto de la teórica

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_MUESTRAS = 10000
SEMILLA = 0
P_TOP = 0.9
T_TEMP = 1.0

# ------------------------------------------------------------------
# 1) Cargar la distribución renormalizada de T2 (con respaldo)
# ------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    tokens_sup = list(t2["tokens_supervivientes_top_p"])
    p_teorica = np.array(t2["distribucion_renormalizada"], dtype=float)
    fuente = "entrada/T2/resultados.json"
else:
    # Respaldo: recalcular softmax(T=1) + top-p(0.9) + renormalización
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()
    p1 = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p1)
    acum = np.cumsum(p1[orden])
    k = int(np.searchsorted(acum, P_TOP) + 1)
    idx = np.sort(orden[:k])
    p_teorica = p1[idx] / p1[idx].sum()
    tokens_sup = [f"t{i}" for i in idx]
    fuente = "recalculada (softmax T=1 + top-p 0.9)"

assert abs(p_teorica.sum() - 1.0) < 1e-12, "La distribución teórica debe sumar 1"

# ------------------------------------------------------------------
# 2) Muestreo: 10,000 muestras con default_rng(0)
# ------------------------------------------------------------------
rng = np.random.default_rng(SEMILLA)
muestras = rng.choice(len(tokens_sup), size=N_MUESTRAS, p=p_teorica)

conteos = np.bincount(muestras, minlength=len(tokens_sup))
frec_obs = conteos / N_MUESTRAS

# ------------------------------------------------------------------
# 3) Divergencia KL en bits: D_KL(empírica || teórica)
#    (términos con frecuencia observada 0 aportan 0)
# ------------------------------------------------------------------
mask = frec_obs > 0
kl_bits = float(np.sum(frec_obs[mask] * np.log2(frec_obs[mask] / p_teorica[mask])))
kl_nats = float(np.sum(frec_obs[mask] * np.log(frec_obs[mask] / p_teorica[mask])))
tv = float(0.5 * np.sum(np.abs(frec_obs - p_teorica)))  # distancia total (informativa)

# ------------------------------------------------------------------
# 4) Gráfico de barras: frecuencias observadas vs probabilidades teóricas
# ------------------------------------------------------------------
x = np.arange(len(tokens_sup))
ancho = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - ancho / 2, frec_obs, ancho, color="#4C72B0",
            label="Frecuencia observada (10,000 muestras)")
b2 = ax.bar(x + ancho / 2, p_teorica, ancho, color="#DD8452",
            label="Probabilidad teórica (top-p renormalizada)")
for rect, val in list(zip(b1, frec_obs)) + list(zip(b2, p_teorica)):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.006,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens_sup)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_title(f"Muestreo top-p (T=1, p={P_TOP}): {N_MUESTRAS} muestras — "
             f"KL(empírica||teórica) = {kl_bits:.6f} bits")
ax.set_ylim(0, max(frec_obs.max(), p_teorica.max()) * 1.18)
ax.legend()
fig.tight_layout()
fig.savefig("T3_frecuencias_observadas_vs_teoricas.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5) Escribir resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T3_comprobacion_empirica",
    "n_muestras": N_MUESTRAS,
    "semilla": SEMILLA,
    "generador": "numpy.random.default_rng(0)",
    "fuente_distribucion": fuente,
    "tokens_supervivientes_top_p": tokens_sup,
    "probabilidades_teoricas_renormalizadas": p_teorica.tolist(),
    "conteos_observados": conteos.tolist(),
    "frecuencias_observadas": frec_obs.tolist(),
    "suma_frecuencias_observadas": float(frec_obs.sum()),
    "divergencia_KL_bits_empirica_respecto_teorica": kl_bits,
    "divergencia_KL_nats_empirica_respecto_teorica": kl_nats,
    "distancia_variacion_total": tv,
    "tabla_comparativa": [
        {
            "token": tok,
            "prob_teorica": float(pt),
            "conteo_observado": int(co),
            "frecuencia_observada": float(fo),
            "diferencia_obs_menos_teo": float(fo - pt),
        }
        for tok, pt, co, fo in zip(tokens_sup, p_teorica, conteos, frec_obs)
    ],
    "figura": "T3_frecuencias_observadas_vs_teoricas.png",
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("=== T3: Comprobación empírica del muestreo top-p ===")
print(f"Muestras: {N_MUESTRAS} | Generador: numpy.random.default_rng({SEMILLA})")
print(f"Fuente de la distribución: {fuente}")
print(f"Tokens supervivientes: {tokens_sup}")
print("Token |  teórica  |  observada | conteo")
for tok, pt, fo, co in zip(tokens_sup, p_teorica, frec_obs, conteos):
    print(f"  {tok}  | {pt:.6f} | {fo:.6f} | {co}")
print(f"Divergencia KL (empírica || teórica) = {kl_bits:.6f} bits "
      f"({kl_nats:.6f} nats) -> cercana a 0: {kl_bits < 0.01}")
print(f"Distancia de variación total = {tv:.6f}")
print("Figura guardada: T3_frecuencias_observadas_vs_teoricas.png")
print("Resultados escritos en resultados.json")
