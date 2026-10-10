import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# T2 — Muestreo de núcleo (top-p, p = 0.9) sobre la
# distribución de T = 1 obtenida en la Parte 1 (T1).
# ============================================================

# ---------- 1) Cargar la distribución de T=1 de la Parte 1 ----------
ruta_t1 = Path("entrada/T1/resultados.json")
if ruta_t1.exists():
    with open(ruta_t1, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    tokens = list(t1["tokens"])
    logits = np.asarray(t1["logits"], dtype=float)
    probs = np.asarray(t1["distribuciones"]["T=1.0"], dtype=float)
else:
    # Respaldo: recomputar la softmax numéricamente estable con T=1
    tokens = ["token_1", "token_2", "token_3", "token_4", "token_5", "token_6"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()
    e = np.exp(z)
    probs = e / e.sum()

# Verificación de consistencia: la distribución cargada debe coincidir
# con la softmax estable (T=1) recalculada a partir de los logits.
z = logits - logits.max()
probs_ref = np.exp(z) / np.exp(z).sum()
max_diff = float(np.max(np.abs(probs - probs_ref)))

# ---------- 2) Top-p (núcleo) con p = 0.9 ----------
p = 0.9
orden = np.argsort(-probs, kind="stable")            # orden descendente de probabilidad
probs_ord = probs[orden]
tokens_ord = [tokens[i] for i in orden]
acum = np.cumsum(probs_ord)

# Conjunto MÍNIMO cuya probabilidad acumulada alcanza p:
# primer índice j (0-based) con acum[j] >= p  ->  k = j + 1 tokens sobreviven
j = int(np.searchsorted(acum, p, side="left"))
k = j + 1
idx_sup = orden[:k]
idx_desc = orden[k:]

tokens_sup = [tokens[i] for i in idx_sup]
probs_sup = probs[idx_sup]
acum_sup_lista = np.cumsum(probs_sup)                # acumulada dentro del núcleo
acum_sup = float(acum[j])                            # prob. acumulada del núcleo (>= 0.9)
renorm = probs_sup / probs_sup.sum()                 # distribución renormalizada

tokens_desc = [tokens[i] for i in idx_desc]
probs_desc = probs[idx_desc]

minimal = bool(j == 0 or acum[j - 1] < p)            # el token previo NO alcanzaba p

# ---------- 3) Figura ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

x = np.arange(len(tokens))
colores = ["#2a9d8f" if tk in tokens_sup else "#e76f51" for tk in tokens_ord]
axes[0].bar(x, probs_ord, color=colores, label="prob. token")
axes[0].plot(x, acum, "o--", color="#264653", label="prob. acumulada")
axes[0].axhline(p, color="gray", ls=":", label="p = 0.9")
axes[0].set_xticks(x)
axes[0].set_xticklabels(tokens_ord, rotation=30)
axes[0].set_ylabel("probabilidad")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Distribución T=1 ordenada (verde = núcleo top-p)")
axes[0].legend(fontsize=8)

x2 = np.arange(k)
axes[1].bar(x2, renorm, color="#2a9d8f")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(tokens_sup, rotation=30)
axes[1].set_ylabel("probabilidad renormalizada")
axes[1].set_ylim(0, 1.0)
axes[1].set_title(f"Distribución renormalizada (suma = {renorm.sum():.6f})")
for xi, vi in zip(x2, renorm):
    axes[1].text(xi, vi + 0.02, f"{vi:.4f}", ha="center", fontsize=8)

fig.tight_layout()
fig.savefig("T2_top_p_renormalizada.png", dpi=120)
plt.close(fig)

# ---------- 4) resultados.json ----------
tabla_md = "| Token | p_original | p_acumulada | p_renormalizada |\n|---|---|---|---|\n"
for tk, po, pa, pr in zip(tokens_sup, probs_sup, acum_sup_lista, renorm):
    tabla_md += f"| {tk} | {po:.6f} | {pa:.6f} | {pr:.6f} |\n"

resultados = {
    "subtarea": "T2_muestreo_nucleo_top_p",
    "p": p,
    "distribucion_base": "T=1.0 (Parte 1)",
    "tokens": tokens,
    "distribucion_original_T1": {tk: float(v) for tk, v in zip(tokens, probs)},
    "verificacion_softmax_T1_max_diff": max_diff,
    "tokens_ordenados_descendente": tokens_ord,
    "probs_ordenadas_descendente": [float(v) for v in probs_ord],
    "prob_acumulada_ordenada": [float(v) for v in acum],
    "num_tokens_supervivientes": k,
    "tokens_supervivientes": tokens_sup,
    "probs_supervivientes_originales": [float(v) for v in probs_sup],
    "prob_acumulada_por_token_superviviente": [float(v) for v in acum_sup_lista],
    "prob_acumulada_nucleo": acum_sup,
    "cumple_umbral_p": bool(acum_sup >= p),
    "conjunto_minimo": minimal,
    "tokens_descartados": tokens_desc,
    "probs_descartados": [float(v) for v in probs_desc],
    "masa_descartada": float(probs_desc.sum()),
    "distribucion_renormalizada": {tk: float(v) for tk, v in zip(tokens_sup, renorm)},
    "suma_distribucion_renormalizada": float(renorm.sum()),
    "tabla_markdown": tabla_md,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------- 5) Resumen ----------
print("T2 — Muestreo de núcleo top-p (p=0.9) sobre la distribución T=1 de la Parte 1")
print(f"Tokens ordenados (desc): {tokens_ord}")
print(f"Prob. acumulada: {[round(v, 6) for v in acum]}")
print(f"Núcleo (conjunto mínimo que alcanza 0.9): {tokens_sup}  | acumulada = {acum_sup:.6f} (>= 0.9: {acum_sup >= p}, mínimo: {minimal})")
print(f"Descartados: {tokens_desc} | masa descartada = {probs_desc.sum():.6f}")
print("Distribución renormalizada:")
for tk, v in zip(tokens_sup, renorm):
    print(f"  {tk}: {v:.6f}")
print(f"Suma de la distribución renormalizada = {renorm.sum():.12f}")
print("Figura guardada: T2_top_p_renormalizada.png")
print("Resultados escritos en resultados.json")
