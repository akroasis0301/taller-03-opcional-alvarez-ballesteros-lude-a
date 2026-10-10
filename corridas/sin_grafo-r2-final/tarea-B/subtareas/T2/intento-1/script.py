# -*- coding: utf-8 -*-
"""
Subtarea T2: Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1
obtenida en la subtarea T1 (softmax con temperatura).

Pasos:
  1) Cargar la distribución de T = 1.0 desde entrada/T1/resultados.json.
  2) Ordenar los tokens por probabilidad descendente y acumular.
  3) Conservar el conjunto MÍNIMO de tokens cuya probabilidad acumulada alcanza p = 0.9.
  4) Renormalizar las probabilidades del núcleo para que sumen 1.0.
  5) Escribir resultados.json (contrato), una figura PNG y una tabla CSV.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 0.9  # umbral del núcleo (top-p)

# ----------------------------------------------------------------------
# 1) Cargar la distribución de T = 1.0 producida por la subtarea T1
# ----------------------------------------------------------------------
ruta_t1 = Path("entrada") / "T1" / "resultados.json"
if ruta_t1.exists():
    with open(ruta_t1, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    tokens = list(t1["tokens"])
    dists = t1["distribuciones"]
    if "T=1.0" in dists:
        probs_T1 = np.asarray(dists["T=1.0"], dtype=float)
    else:  # respaldo: localizar la clave de temperatura 1.0
        clave = next(k for k in dists if abs(float(str(k).split("=")[1]) - 1.0) < 1e-12)
        probs_T1 = np.asarray(dists[clave], dtype=float)
    fuente = "distribucion T=1.0 de entrada/T1/resultados.json"
else:
    # Respaldo: recalcular la softmax con T = 1 a partir de los logits de la Parte 1
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()          # estabilidad numérica
    e = np.exp(z)
    probs_T1 = e / e.sum()
    fuente = "softmax(T=1) recalculada desde los logits de la Parte 1"

# ----------------------------------------------------------------------
# 2) Top-p: ordenar descendente, acumular y conservar el conjunto mínimo
#    cuya probabilidad acumulada alcanza p = 0.9
# ----------------------------------------------------------------------
orden = np.argsort(-probs_T1, kind="stable")            # índices en orden descendente
tokens_ord = [tokens[i] for i in orden]
probs_ord = probs_T1[orden]
acumulada = np.cumsum(probs_ord)

k = 0
acum = 0.0
for pr in probs_ord:                                    # conjunto mínimo con acumulada >= p
    k += 1
    acum += float(pr)
    if acum >= P:
        break
masa_nucleo = acum                                      # probabilidad acumulada del núcleo
masa_descartada = 1.0 - masa_nucleo

idx_nucleo = orden[:k]                                  # índices que sobreviven
tokens_sobrev = [tokens[i] for i in idx_nucleo]
tokens_fuera = [tokens[i] for i in orden[k:]]

# ----------------------------------------------------------------------
# 3) Renormalizar la distribución del núcleo
# ----------------------------------------------------------------------
probs_nucleo = probs_T1[idx_nucleo]
probs_renorm = probs_nucleo / probs_nucleo.sum()
dist_renorm = {tokens[i]: float(q) for i, q in zip(idx_nucleo, probs_renorm)}
suma_renorm = float(probs_renorm.sum())

# ----------------------------------------------------------------------
# 4) Tabla resumen (valores completos y a 4 decimales)
# ----------------------------------------------------------------------
acum_por_token = {}
ac_aux = 0.0
for tk, pr in zip(tokens_ord, probs_ord):
    ac_aux += float(pr)
    acum_por_token[tk] = ac_aux

tabla = pd.DataFrame({
    "token": tokens,
    "prob_T1": probs_T1,
    "prob_acumulada_desc": [acum_por_token[tk] for tk in tokens],
    "sobrevive_top_p": ["sí" if tk in tokens_sobrev else "no" for tk in tokens],
    "prob_renormalizada": [dist_renorm.get(tk, np.nan) for tk in tokens],
})

tabla_4dec = {
    tk: {
        "prob_T1": f"{probs_T1[i]:.4f}",
        "acumulada": f"{acum_por_token[tk]:.4f}",
        "sobrevive": "sí" if tk in tokens_sobrev else "no",
        "renormalizada": (f"{dist_renorm[tk]:.4f}" if tk in dist_renorm else "—"),
    }
    for i, tk in enumerate(tokens)
}

# ----------------------------------------------------------------------
# 5) Figura: distribución ordenada con frontera del núcleo + renormalizada
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

colores = ["#2a9d8f" if tk in tokens_sobrev else "#e76f51" for tk in tokens_ord]
x = np.arange(len(tokens_ord))
axes[0].bar(x, probs_ord, color=colores, edgecolor="black")
axes[0].set_xticks(x)
axes[0].set_xticklabels(tokens_ord)
axes[0].set_ylabel("Probabilidad (T = 1)")
axes[0].set_title("Distribución T=1 ordenada (verde = núcleo top-p)")
for xi, pr in zip(x, probs_ord):
    axes[0].text(xi, pr + 0.008, f"{pr:.4f}", ha="center", fontsize=8)

ax2 = axes[0].twinx()
ax2.plot(x, acumulada, "o--", color="#264653", lw=1.5, label="Acumulada")
ax2.axhline(P, color="red", ls=":", lw=1.5, label=f"p = {P}")
ax2.set_ylabel("Probabilidad acumulada")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="lower right", fontsize=8)

xr = np.arange(k)
axes[1].bar(xr, probs_renorm, color="#2a9d8f", edgecolor="black")
axes[1].set_xticks(xr)
axes[1].set_xticklabels(tokens_sobrev)
axes[1].set_ylabel("Probabilidad renormalizada")
axes[1].set_title(f"Distribución renormalizada (suma = {suma_renorm:.4f})")
for xi, q in zip(xr, probs_renorm):
    axes[1].text(xi, q + 0.008, f"{q:.4f}", ha="center", fontsize=8)
axes[1].set_ylim(0, float(probs_renorm.max()) * 1.18)

fig.suptitle(f"Muestreo de núcleo (top-p) con p = {P} sobre la distribución de T = 1", y=1.02)
fig.tight_layout()
fig.savefig("top_p_nucleo_T1.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# 6) Guardar tabla CSV y resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
tabla.to_csv("tabla_top_p_T1.csv", index=False)

resultados = {
    "subtarea": "T2_top_p_nucleus_sampling",
    "p": P,
    "fuente": fuente,
    "tokens": tokens,
    "probabilidades_T1": probs_T1.tolist(),
    "orden_descendente": tokens_ord,
    "probabilidades_ordenadas": probs_ord.tolist(),
    "probabilidad_acumulada_ordenada": acumulada.tolist(),
    "tokens_sobrevivientes": tokens_sobrev,
    "tokens_descartados": tokens_fuera,
    "num_tokens_nucleo": k,
    "probabilidad_acumulada_nucleo": masa_nucleo,
    "masa_descartada": masa_descartada,
    "probabilidades_originales_nucleo": {tokens[i]: float(probs_T1[i]) for i in idx_nucleo},
    "distribucion_renormalizada": dist_renorm,
    "suma_distribucion_renormalizada": suma_renorm,
    "tabla_4_decimales": tabla_4dec,
    "figuras": ["top_p_nucleo_T1.png", "tabla_top_p_T1.csv"],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7) Resumen por consola
# ----------------------------------------------------------------------
print("=== T2: Muestreo de núcleo (top-p), p = 0.9, sobre la distribución de T = 1 ===")
print("Fuente:", fuente)
print("Distribución T=1:", {tk: round(float(p), 6) for tk, p in zip(tokens, probs_T1)})
print("Orden descendente:", tokens_ord)
print("Acumulada:", [round(float(a), 6) for a in acumulada])
print(f"Tokens sobrevivientes (conjunto mínimo con acumulada >= {P}): {tokens_sobrev}")
print(f"Tokens descartados: {tokens_fuera}")
print(f"Probabilidad acumulada del núcleo: {masa_nucleo:.6f} (masa descartada: {masa_descartada:.6f})")
print("Distribución renormalizada:")
for tk in tokens_sobrev:
    print(f"  {tk}: {dist_renorm[tk]:.6f}")
print(f"Suma de la distribución renormalizada: {suma_renorm:.12f}")
print("Archivos escritos: resultados.json, top_p_nucleo_T1.png, tabla_top_p_T1.csv")
