"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1.

Toma la distribución softmax con T = 1 obtenida en la subtarea T1
(entrada/T1/resultados.json), ordena los tokens por probabilidad descendente,
conserva el conjunto MÍNIMO cuya probabilidad acumulada alcanza p = 0.9,
renormaliza y reporta los tokens supervivientes y la distribución resultante
(con probabilidad 0 para los tokens fuera del núcleo).

Salidas:
  - resultados.json          (contrato de la subtarea)
  - T2_top_p_nucleus.png     (figura comparativa)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Cargar la distribución de T = 1 de la Parte 1 (subtarea T1)
# ----------------------------------------------------------------------
ruta_t1 = Path("entrada/T1/resultados.json")
if ruta_t1.exists():
    with open(ruta_t1, "r", encoding="utf-8") as f:
        res_t1 = json.load(f)
    tokens = list(res_t1["tokens"])
    logits = np.array(res_t1["logits"], dtype=float)
    p_T1 = np.array(res_t1["distribuciones_listas"]["T=1.0"], dtype=float)
    fuente = "entrada/T1/resultados.json"
else:
    # Respaldo defensivo: recomputar la softmax con T = 1 (numéricamente estable)
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()
    p_T1 = np.exp(z) / np.exp(z).sum()
    fuente = "softmax T=1 recalculada"

# Verificación de consistencia (softmax estable con T = 1)
z = logits - logits.max()
p_check = np.exp(z) / np.exp(z).sum()
consistente = bool(np.allclose(p_T1, p_check, atol=1e-12))

# ----------------------------------------------------------------------
# 2. Top-p (nucleus sampling) con p = 0.9
# ----------------------------------------------------------------------
p_umbral = 0.9

# Ordenar tokens por probabilidad descendente
orden = np.argsort(-p_T1)                       # índices de mayor a menor
p_ordenada = p_T1[orden]
tokens_ordenados = [tokens[i] for i in orden]

# Probabilidad acumulada (orden descendente)
acumulada = np.cumsum(p_ordenada)

# Conjunto mínimo cuya acumulada alcanza p:
# primer índice k tal que acumulada[k] >= p  ->  se conservan k+1 tokens
k = int(np.searchsorted(acumulada, p_umbral, side="left") + 1)

nucleo_idx = orden[:k]      # tokens supervivientes
fuera_idx = orden[k:]       # tokens eliminados

# Renormalización: probabilidad 0 para los tokens fuera del núcleo
p_renorm = np.zeros_like(p_T1)
masa_nucleo = float(p_T1[nucleo_idx].sum())
p_renorm[nucleo_idx] = p_T1[nucleo_idx] / masa_nucleo

# ----------------------------------------------------------------------
# 3. Construir resultados (contrato resultados.json)
# ----------------------------------------------------------------------
dist_original = {tokens[i]: float(p_T1[i]) for i in range(len(tokens))}
dist_renorm = {tokens[i]: float(p_renorm[i]) for i in range(len(tokens))}
acum_dict = {tokens[i]: float(a) for i, a in zip(orden, acumulada)}
sobrevivientes = [tokens[i] for i in nucleo_idx]
eliminados = [tokens[i] for i in fuera_idx]

# Entropía en bits de la distribución renormalizada (informativo)
p_pos = p_renorm[p_renorm > 0]
entropia_renorm_bits = float(-(p_pos * np.log2(p_pos)).sum())

resultados = {
    "subtarea": "T2 — top-p (nucleus sampling) con p = 0.9 sobre la distribución de T = 1",
    "fuente_distribucion": fuente,
    "p": p_umbral,
    "temperatura": 1.0,
    "tokens": tokens,
    "logits": [float(x) for x in logits],
    "distribucion_T1": dist_original,
    "tokens_ordenados_desc": tokens_ordenados,
    "probs_ordenadas_desc": [float(x) for x in p_ordenada],
    "prob_acumulada_ordenada": acum_dict,
    "num_tokens_nucleo": k,
    "tokens_sobrevivientes": sobrevivientes,
    "tokens_eliminados": eliminados,
    "masa_prob_nucleo": masa_nucleo,
    "distribucion_renormalizada": dist_renorm,
    "distribucion_renormalizada_solo_nucleo": {
        tokens[i]: float(p_renorm[i]) for i in nucleo_idx
    },
    "entropia_bits_renormalizada": entropia_renorm_bits,
    "verificacion": {
        "softmax_T1_consistente_con_T1": consistente,
        "suma_renormalizada_total": float(p_renorm.sum()),
        "suma_renormalizada_solo_nucleo": float(p_renorm[nucleo_idx].sum()),
        "prob_fuera_nucleo_cero": bool(np.all(p_renorm[fuera_idx] == 0.0)),
        "acumulada_nucleo_alcanza_p": bool(masa_nucleo >= p_umbral),
        "acumulada_sin_ultimo_token_no_alcanza_p": bool(
            (masa_nucleo - float(p_T1[nucleo_idx[-1]])) < p_umbral
        ),
    },
    "tabla_4_decimales": [
        {
            "token": tokens[i],
            "prob_T1": f"{p_T1[i]:.4f}",
            "prob_acumulada": f"{acum_dict[tokens[i]]:.4f}",
            "en_nucleo": bool(tokens[i] in sobrevivientes),
            "prob_renormalizada": f"{p_renorm[i]:.4f}",
        }
        for i in orden
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 4. Figura: distribución original vs. distribución top-p renormalizada
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(tokens))

axes[0].bar(x, p_T1, color="steelblue")
axes[0].set_title("Softmax con T = 1 (original)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(tokens)
axes[0].set_ylabel("probabilidad")
axes[0].set_ylim(0, 1.05)
for i, v in enumerate(p_T1):
    axes[0].text(i, v + 0.015, f"{v:.4f}", ha="center", fontsize=8)

colores = ["steelblue" if tokens[i] in sobrevivientes else "lightgray"
           for i in range(len(tokens))]
axes[1].bar(x, p_renorm, color=colores)
axes[1].set_title(f"Top-p (p = {p_umbral}) renormalizada")
axes[1].set_xticks(x)
axes[1].set_xticklabels(tokens)
axes[1].set_ylim(0, 1.05)
for i, v in enumerate(p_renorm):
    axes[1].text(i, v + 0.015, f"{v:.4f}", ha="center", fontsize=8)

fig.suptitle(
    f"Núcleo mínimo: {{{', '.join(sobrevivientes)}}} — "
    f"masa acumulada = {masa_nucleo:.4f} ≥ {p_umbral} "
    f"(sin el último token: {masa_nucleo - float(p_T1[nucleo_idx[-1]]):.4f} < {p_umbral})"
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("T2_top_p_nucleus.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5. Resumen por consola
# ----------------------------------------------------------------------
print("=" * 64)
print("T2 — Top-p (nucleus sampling), p = 0.9, sobre la distribución T = 1")
print("=" * 64)
print(f"Fuente de la distribución: {fuente}")
print("\nTokens ordenados por probabilidad descendente:")
print(f"  {'token':<6}{'p(T=1)':>10}{'acumulada':>12}{'¿en núcleo?':>14}")
for t in tokens_ordenados:
    en = "sí" if t in sobrevivientes else "no"
    print(f"  {t:<6}{dist_original[t]:>10.4f}{acum_dict[t]:>12.4f}{en:>14}")
print(f"\nConjunto mínimo que alcanza p = {p_umbral}: {sobrevivientes} (k = {k})")
print(f"Masa de probabilidad del núcleo: {masa_nucleo:.4f}")
print(f"Tokens eliminados (probabilidad 0 tras renormalizar): {eliminados}")
print("\nDistribución renormalizada (suma = 1):")
for t in tokens:
    print(f"  {t}: {dist_renorm[t]:.6f}")
print(f"\nSuma de la distribución renormalizada: {float(p_renorm.sum()):.12f}")
print(f"Entropía de la distribución renormalizada: {entropia_renorm_bits:.4f} bits")
print("\nArchivos escritos: resultados.json, T2_top_p_nucleus.png")
