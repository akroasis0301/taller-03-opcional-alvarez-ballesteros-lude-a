# -*- coding: utf-8 -*-
"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1.

Entrada : entrada/T1/resultados.json  (distribución softmax con T=1 de la subtarea T1)
Salidas : resultados.json (contrato de la subtarea), top_p_T2.png (figura)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P_NUCLEO = 0.9

# ------------------------------------------------------------------
# 1) Cargar la distribución de T=1 obtenida en T1
# ------------------------------------------------------------------
fuente = "entrada/T1/resultados.json"
try:
    with open(fuente, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    tokens = list(t1["tokens"])
    p_T1 = np.asarray(t1["distribuciones"]["T=1.0"], dtype=float)
    origen = "entrada/T1/resultados.json (salida de T1)"
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    # Respaldo defensivo: recalcular softmax(T=1) de forma numéricamente estable
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()
    e = np.exp(z)
    p_T1 = e / e.sum()
    origen = "recalculada con softmax estable (T=1); entrada/T1 no disponible"

# ------------------------------------------------------------------
# 2) Top-p: ordenar por probabilidad descendente y acumular
# ------------------------------------------------------------------
orden = np.argsort(-p_T1, kind="stable")          # índices de mayor a menor prob.
tokens_ord = [tokens[i] for i in orden]
probs_ord = p_T1[orden]
acum = np.cumsum(probs_ord)                        # probabilidad acumulada

# Conjunto MÍNIMO cuya acumulada alcanza p: primer k con acum[k] >= p
k = int(np.searchsorted(acum, P_NUCLEO, side="left"))
if k >= len(acum):                                 # salvaguarda por punto flotante
    k = len(acum) - 1

idx_nucleo = orden[: k + 1]
tokens_sup = [tokens[i] for i in idx_nucleo]       # tokens supervivientes
probs_sup = p_T1[idx_nucleo]
masa_nucleo = float(probs_sup.sum())

# ------------------------------------------------------------------
# 3) Renormalizar el núcleo
# ------------------------------------------------------------------
probs_ren = probs_sup / masa_nucleo
suma_ren = float(probs_ren.sum())
tokens_desc = [t for t in tokens if t not in tokens_sup]
renorm_map = {tok: float(p) for tok, p in zip(tokens_sup, probs_ren)}

# ------------------------------------------------------------------
# 4) Tabla legible (4 decimales) + resultados.json (contrato)
# ------------------------------------------------------------------
tabla = []
for tok, p, a in zip(tokens_ord, probs_ord, acum):
    fila = {
        "token": tok,
        "prob_T1": f"{float(p):.4f}",
        "prob_acumulada": f"{float(a):.4f}",
        "en_nucleo": tok in renorm_map,
        "prob_renormalizada": (f"{renorm_map[tok]:.4f}" if tok in renorm_map else None),
    }
    tabla.append(fila)

resultados = {
    "subtarea": "T2_top_p_nucleus_sampling_p0.9_sobre_T1",
    "p_nucleo": P_NUCLEO,
    "fuente_distribucion": origen,
    "tokens": tokens,
    "distribucion_T1_entrada": {tok: float(p) for tok, p in zip(tokens, p_T1)},
    "orden_descendente": tokens_ord,
    "probs_ordenadas": [float(x) for x in probs_ord],
    "prob_acumulada_ordenada": [float(x) for x in acum],
    "criterio_conjunto_minimo": "primer k tal que la suma de las k+1 mayores probabilidades >= 0.9",
    "tokens_supervivientes": tokens_sup,
    "num_tokens_supervivientes": len(tokens_sup),
    "masa_acumulada_nucleo": masa_nucleo,
    "tokens_descartados": tokens_desc,
    "distribucion_renormalizada": renorm_map,
    "distribucion_renormalizada_orden_tokens": tokens_sup,
    "distribucion_renormalizada_valores": [float(x) for x in probs_ren],
    "suma_distribucion_renormalizada": suma_ren,
    "tabla_4_decimales": tabla,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# 5) Figura: distribución original (con acumulada y umbral 0.9) y núcleo renormalizado
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

x = np.arange(len(tokens))
axes[0].bar(x, p_T1, color="steelblue", label="p_i (T=1)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(tokens)
axes[0].set_ylabel("Probabilidad")
axes[0].set_title("Distribución de T=1 (entrada de T1)")
axes[0].set_ylim(0, max(p_T1) * 1.20)
for i, p in enumerate(p_T1):
    axes[0].text(i, p + 0.008, f"{p:.4f}", ha="center", fontsize=8)
ax2 = axes[0].twinx()
ax2.plot(x, acum, "o--", color="crimson", label="acumulada")
ax2.axhline(P_NUCLEO, color="black", ls=":", lw=1.2)
ax2.text(len(tokens) - 0.5, P_NUCLEO + 0.015, "p = 0.9", ha="right", fontsize=9)
ax2.set_ylabel("Probabilidad acumulada")
ax2.set_ylim(0, 1.08)
axes[0].axvspan(k - 0.5, k + 0.5, color="orange", alpha=0.15)

x2 = np.arange(len(tokens_sup))
axes[1].bar(x2, probs_ren, color="darkorange")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(tokens_sup)
axes[1].set_ylabel("Probabilidad renormalizada")
axes[1].set_title(f"Núcleo top-p (p=0.9): {tokens_sup}")
axes[1].set_ylim(0, max(probs_ren) * 1.20)
for i, p in enumerate(probs_ren):
    axes[1].text(i, p + 0.008, f"{p:.4f}", ha="center", fontsize=8)

fig.suptitle("T2 — Muestreo de núcleo (top-p, p=0.9) sobre la distribución de T=1", y=1.02)
fig.tight_layout()
plt.savefig("top_p_T2.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("=== T2: Muestreo de núcleo (top-p) con p=0.9 sobre la distribución de T=1 ===")
print(f"Fuente de la distribución: {origen}")
print("Orden descendente : " + ", ".join(f"{t}={p:.4f}" for t, p in zip(tokens_ord, probs_ord)))
print("Acumulada         : " + ", ".join(f"{a:.4f}" for a in acum))
print(f"Tokens supervivientes (conjunto mínimo con acumulada >= {P_NUCLEO}): {tokens_sup}")
print(f"Tokens descartados: {tokens_desc}")
print(f"Masa acumulada del núcleo (antes de renormalizar): {masa_nucleo:.6f}")
print("Distribución renormalizada:")
for tok, p in zip(tokens_sup, probs_ren):
    print(f"  {tok}: {p:.6f}")
print(f"Suma de la distribución renormalizada: {suma_ren:.10f}")
print("Figura guardada: top_p_T2.png | Resultados escritos en resultados.json")
