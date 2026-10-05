# -*- coding: utf-8 -*-
"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1.

- Carga la distribución de T = 1 obtenida en la Parte 1 (entrada/T1/resultados.json);
  si no estuviera disponible, la recalcula con la softmax numéricamente estable.
- Aplica top-p con p = 0.9: ordena los tokens por probabilidad descendente,
  conserva el conjunto MÍNIMO cuya probabilidad acumulada alcanza 0.9 y renormaliza.
- Escribe resultados.json (contrato), guarda fig_T2_top_p.png e imprime un resumen.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parámetros fijos del enunciado
# ----------------------------------------------------------------------
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
TOKENS = [f"t{i}" for i in range(LOGITS.size)]
TEMPERATURA = 1.0
P = 0.9

# ----------------------------------------------------------------------
# 1) Distribución de T = 1 de la Parte 1 (resultado previo T1)
# ----------------------------------------------------------------------
ruta_t1 = Path("entrada") / "T1" / "resultados.json"
origen = None
probs = None
if ruta_t1.exists():
    with ruta_t1.open("r", encoding="utf-8") as f:
        res_t1 = json.load(f)
    dists = res_t1.get("distribuciones", {})
    for clave in ("T=1.0", "T=1"):
        if clave in dists:
            probs = np.array([float(dists[clave][t]) for t in TOKENS])
            origen = f"{ruta_t1} (clave '{clave}')"
            break
if probs is None:
    # Respaldo: softmax con temperatura, numéricamente estable (resta del máximo)
    z = LOGITS / TEMPERATURA
    z = z - z.max()
    ez = np.exp(z)
    probs = ez / ez.sum()
    origen = "recalculada con softmax estable (T = 1)"

# ----------------------------------------------------------------------
# 2) Top-p con p = 0.9: conjunto mínimo con acumulada >= p
# ----------------------------------------------------------------------
orden = np.argsort(-probs, kind="stable")      # índices de mayor a menor probabilidad
probs_ord = probs[orden]
acum_ord = np.cumsum(probs_ord)

# Número mínimo de tokens para alcanzar p: el token que cruza el umbral SE incluye
k = int(np.searchsorted(acum_ord, P, side="left")) + 1
k = min(k, probs.size)

nucleo_idx = orden[:k]
fuera_idx = orden[k:]

en_nucleo = np.zeros(probs.size, dtype=bool)
en_nucleo[nucleo_idx] = True

probs_nucleo = np.where(en_nucleo, probs, 0.0)   # los tokens fuera del núcleo quedan en 0
suma_nucleo = float(probs_nucleo.sum())
probs_renorm = probs_nucleo / suma_nucleo        # renormalización

tokens_nucleo = [TOKENS[i] for i in nucleo_idx]
tokens_fuera = [TOKENS[i] for i in fuera_idx]
acum_nucleo = float(acum_ord[k - 1])

# ----------------------------------------------------------------------
# 3) Verificaciones del criterio de éxito
# ----------------------------------------------------------------------
suma_renorm = float(probs_renorm.sum())
alcanza_p = bool(acum_nucleo >= P)
es_minimo = bool(k == 1 or acum_ord[k - 2] < P)          # sin el último token no se alcanza p
fuera_en_cero = bool(np.all(probs_renorm[~en_nucleo] == 0.0))
suma_exacta = bool(abs(suma_renorm - 1.0) < 1e-12)

# ----------------------------------------------------------------------
# 4) Figura
# ----------------------------------------------------------------------
x = np.arange(probs.size)
ancho = 0.38
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax1.bar(x - ancho / 2, probs, width=ancho, color="#8d99ae", label="Original (T = 1)")
ax1.bar(x + ancho / 2, probs_renorm, width=ancho,
        color=["#2a9d8f" if m else "#e76f51" for m in en_nucleo],
        label="Renormalizada (top-p, p = 0.9)")
ax1.set_xticks(x)
ax1.set_xticklabels(TOKENS)
ax1.set_ylabel("Probabilidad")
ax1.set_ylim(0, 0.65)
ax1.set_title("Distribución original vs. renormalizada\n(verde = núcleo, rojo = fuera del núcleo)")
ax1.legend(fontsize=8)

ax2.plot(range(1, probs.size + 1), acum_ord, marker="o", color="#264653",
         label="Acumulada (orden descendente)")
ax2.axhline(P, color="#e63946", linestyle="--", label=f"p = {P}")
ax2.axvline(k, color="#2a9d8f", linestyle=":", label=f"k mínimo = {k}")
ax2.set_xticks(range(1, probs.size + 1))
ax2.set_xticklabels([TOKENS[i] for i in orden])
ax2.set_xlabel("Tokens ordenados por probabilidad descendente")
ax2.set_ylabel("Probabilidad acumulada")
ax2.set_title(f"Acumulada del núcleo: {acum_nucleo:.4f} ≥ {P}")
ax2.legend(fontsize=8)

fig.suptitle("T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig_T2_top_p.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
tabla_filas = []
for r, idx in enumerate(orden, start=1):
    tabla_filas.append([
        TOKENS[idx],
        f"{probs[idx]:.4f}",
        f"{acum_ord[r - 1]:.4f}",
        "sí" if en_nucleo[idx] else "no",
        f"{probs_renorm[idx]:.4f}",
    ])

resultados = {
    "subtarea": "T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1",
    "p": P,
    "temperatura": TEMPERATURA,
    "fuente_distribucion_T1": origen,
    "distribucion_original_T1": {t: float(v) for t, v in zip(TOKENS, probs)},
    "orden_descendente": [TOKENS[i] for i in orden],
    "probs_ordenadas": [float(v) for v in probs_ord],
    "acumuladas_ordenadas": [float(v) for v in acum_ord],
    "k_tokens_minimo": k,
    "tokens_nucleo": tokens_nucleo,
    "prob_acumulada_nucleo": acum_nucleo,
    "tokens_fuera_nucleo": tokens_fuera,
    "distribucion_renormalizada": {t: float(v) for t, v in zip(TOKENS, probs_renorm)},
    "suma_distribucion_renormalizada": suma_renorm,
    "verificaciones": {
        "acumulada_nucleo_alcanza_p": alcanza_p,
        "conjunto_es_minimo": es_minimo,
        "tokens_fuera_con_probabilidad_cero": fuera_en_cero,
        "suma_renormalizada_es_1": suma_exacta,
    },
    "tabla_4_decimales": {
        "columnas": ["token", "prob_original_T1", "acumulada_ordenada",
                     "en_nucleo", "prob_renormalizada"],
        "filas": tabla_filas,
    },
    "figura": "fig_T2_top_p.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6) Resumen por consola
# ----------------------------------------------------------------------
print("=" * 66)
print("T2 — Top-p (nucleus sampling) con p = 0.9 sobre la distribución T = 1")
print("=" * 66)
print(f"Fuente de la distribución T = 1 : {origen}")
print("\nTokens ordenados por probabilidad descendente:")
for r, idx in enumerate(orden, start=1):
    marca = "NÚCLEO" if en_nucleo[idx] else "fuera "
    print(f"  {r}. {TOKENS[idx]}  p = {probs[idx]:.6f}  "
          f"acum = {acum_ord[r - 1]:.6f}  [{marca}]")
print(f"\nNúcleo mínimo (k = {k}): {tokens_nucleo}")
print(f"Probabilidad acumulada del núcleo: {acum_nucleo:.6f}  (>= {P}: {alcanza_p})")
print(f"Tokens fuera del núcleo: {tokens_fuera}  (quedan con probabilidad 0)")
print("\nDistribución renormalizada:")
for t, v in zip(TOKENS, probs_renorm):
    print(f"  {t}: {v:.6f}")
print(f"Suma de la distribución renormalizada: {suma_renorm:.12f}")
print("\nVerificaciones:", resultados["verificaciones"])
print("Figura guardada: fig_T2_top_p.png | Contrato escrito: resultados.json")
