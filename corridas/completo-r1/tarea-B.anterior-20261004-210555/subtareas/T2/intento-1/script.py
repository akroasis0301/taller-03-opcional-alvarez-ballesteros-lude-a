# T2 — Muestreo de núcleo (top-p, p=0.9) sobre la distribución de T=1 de la Parte 1 (T1)
# Solo librerías permitidas. Sin red, sin subprocess, sin rutas absolutas.

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Cargar la distribución de T=1 calculada en T1 (entrada/T1/)
# ------------------------------------------------------------------
ruta_t1 = os.path.join("entrada", "T1", "resultados.json")
if os.path.exists(ruta_t1):
    with open(ruta_t1, "r", encoding="utf-8") as f:
        res_t1 = json.load(f)
    tokens = list(res_t1["tokens"])
    logits = np.array(res_t1["logits"], dtype=float)
    p_T1 = np.array(res_t1["distribuciones"]["T=1.0"], dtype=float)
    fuente = "entrada/T1/resultados.json"
else:
    # Respaldo: recalcular la softmax con T=1 (numéricamente estable)
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()
    p_T1 = np.exp(z) / np.exp(z).sum()
    fuente = "recalculada (respaldo)"

# Verificación: recomputar softmax(T=1) y comparar con lo cargado
z = logits / 1.0
z = z - z.max()
p_recalc = np.exp(z) / np.exp(z).sum()
diff_max = float(np.max(np.abs(p_T1 - p_recalc)))

# ------------------------------------------------------------------
# 2. Top-p (nucleus) con p = 0.9
#    Ordenar por probabilidad descendente, conservar el conjunto MÍNIMO
#    cuya probabilidad acumulada alcanza p (incluido el token que cruza
#    el umbral) y renormalizar sobre los supervivientes.
# ------------------------------------------------------------------
p_corte = 0.9

orden = np.argsort(-p_T1, kind="stable")          # índices en orden descendente
probs_ord = p_T1[orden]
tokens_ord = [tokens[i] for i in orden]
acumulada = np.cumsum(probs_ord)

# Número mínimo k de tokens con acumulada >= p (searchsorted con side='left'
# incluye el token en el que la acumulada iguala o supera el umbral)
k = int(np.searchsorted(acumulada, p_corte, side="left")) + 1
k = min(k, len(tokens))

idx_sup = orden[:k]
tokens_sup = [tokens[i] for i in idx_sup]
probs_sup = p_T1[idx_sup]
acum_sup = float(acumulada[k - 1])                # acumulada del núcleo (>= 0.9)

# Renormalización sobre los supervivientes
probs_renorm = probs_sup / probs_sup.sum()
suma_renorm = float(probs_renorm.sum())

tokens_desc = [tokens[i] for i in orden[k:]]
masa_desc = float(p_T1[orden[k:]].sum())

# Acumulada por token superviviente (para la tabla)
acum_por_sup = [float(acumulada[j]) for j in range(k)]

# ------------------------------------------------------------------
# 3. Escribir resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "p": p_corte,
    "temperatura": 1.0,
    "fuente_distribucion": fuente,
    "tokens": tokens,
    "logits": logits.tolist(),
    "distribucion_T1": p_T1.tolist(),
    "orden_descendente": tokens_ord,
    "probs_ordenadas_descendente": probs_ord.tolist(),
    "prob_acumulada_ordenada": acumulada.tolist(),
    "tokens_supervivientes_top_p": tokens_sup,
    "probs_supervivientes_originales": probs_sup.tolist(),
    "prob_acumulada_por_superviviente": acum_por_sup,
    "prob_acumulada_conjunto_superviviente": acum_sup,
    "cumple_acumulada_ge_p": bool(acum_sup >= p_corte),
    "distribucion_renormalizada": probs_renorm.tolist(),
    "suma_distribucion_renormalizada": suma_renorm,
    "tokens_descartados": tokens_desc,
    "masa_probabilidad_descartada": masa_desc,
    "tabla_resumen": [
        {
            "token": tokens_sup[j],
            "prob_original_T1": float(probs_sup[j]),
            "prob_acumulada": acum_por_sup[j],
            "prob_renormalizada": float(probs_renorm[j]),
        }
        for j in range(k)
    ],
    "verificacion_softmax_T1_max_abs_diff": diff_max,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# 4. Figura: distribución original vs. núcleo top-p renormalizado
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

x_full = np.arange(len(tokens))
colores = ["darkorange" if i in set(idx_sup) else "lightgray" for i in range(len(tokens))]
axes[0].bar(x_full, p_T1, color=colores, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(x_full)
axes[0].set_xticklabels(tokens)
axes[0].set_ylim(0, p_T1.max() * 1.18)
axes[0].set_ylabel("probabilidad")
axes[0].set_title("Distribución original (T=1)\n(naranja = supervivientes top-p)")
for i, v in enumerate(p_T1):
    axes[0].text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=8)

x_sup = np.arange(k)
axes[1].bar(x_sup, probs_renorm, color="darkorange", edgecolor="black", linewidth=0.5)
axes[1].set_xticks(x_sup)
axes[1].set_xticklabels(tokens_sup)
axes[1].set_ylim(0, probs_renorm.max() * 1.18)
axes[1].set_ylabel("probabilidad renormalizada")
axes[1].set_title(f"Núcleo top-p (p={p_corte}): {k} tokens\nacumulada = {acum_sup:.4f} >= {p_corte}")
for i, v in enumerate(probs_renorm):
    axes[1].text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=8)

fig.suptitle("Muestreo de núcleo (top-p) sobre la distribución de T=1", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("top_p_nucleus.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5. Resumen breve
# ------------------------------------------------------------------
print("=== T2: Muestreo de núcleo top-p (p=0.9) sobre la distribución T=1 ===")
print(f"Fuente de la distribución: {fuente}")
print("Orden descendente y acumulada:")
for t, v, a in zip(tokens_ord, probs_ord, acumulada):
    print(f"  {t}: p={v:.6f}  acum={a:.6f}")
print(f"Tokens supervivientes (conjunto mínimo con acumulada >= {p_corte}): {tokens_sup}")
print(f"Probabilidad acumulada del núcleo: {acum_sup:.6f} (>= 0.9: {acum_sup >= p_corte})")
print("Distribución renormalizada:")
for t, pr in zip(tokens_sup, probs_renorm):
    print(f"  {t}: {pr:.6f}")
print(f"Suma de la distribución renormalizada: {suma_renorm:.6f}")
print(f"Tokens descartados: {tokens_desc} (masa descartada = {masa_desc:.6f})")
print(f"Verificación softmax T=1 (máx. dif. absoluta): {diff_max:.2e}")
print("Figura guardada: top_p_nucleus.png | Resultados en resultados.json")
