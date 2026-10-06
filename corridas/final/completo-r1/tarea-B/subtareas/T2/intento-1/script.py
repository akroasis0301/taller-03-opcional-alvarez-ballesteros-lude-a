# T2: Muestreo de núcleo (top-p) con p=0.9 sobre la distribución de T=1 de la Parte 1
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Parámetros fijados por el enunciado
# ------------------------------------------------------------------
P = 0.9
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)


def softmax_con_temperatura(logits, T):
    """Softmax con temperatura, numéricamente estable (resta del máximo)."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# ------------------------------------------------------------------
# 1) Distribución de T=1 obtenida en la Parte 1 (entrada/T1/resultados.json)
# ------------------------------------------------------------------
ruta_T1 = Path("entrada") / "T1" / "resultados.json"
dist_T1 = None
fuente = None
if ruta_T1.exists():
    with ruta_T1.open("r", encoding="utf-8") as f:
        res_T1 = json.load(f)
    d = res_T1.get("distribuciones", {}).get("T=1")
    if d is not None:
        dist_T1 = np.asarray(d, dtype=float)
        fuente = "entrada/T1/resultados.json (distribucion de T=1 de la Parte 1)"
if dist_T1 is None:
    # Respaldo: recalcular exactamente como en la Parte 1
    dist_T1 = softmax_con_temperatura(LOGITS, 1.0)
    fuente = "recalculada con softmax numericamente estable (T=1)"

# Verificación de procedencia: coincide con la softmax estable de la Parte 1
dist_T1_recalc = softmax_con_temperatura(LOGITS, 1.0)
max_abs_diff = float(np.max(np.abs(dist_T1 - dist_T1_recalc)))

# ------------------------------------------------------------------
# 2) Top-p (nucleus) con p = 0.9
#    Ordenar por probabilidad descendente, conservar el conjunto MINIMO
#    cuya acumulada alcanza p, y renormalizar los supervivientes.
# ------------------------------------------------------------------
orden = np.argsort(-dist_T1, kind="stable")          # índices en orden descendente
probs_ord = dist_T1[orden]
acum_ord = np.cumsum(probs_ord)

# primer índice (en el orden) cuya acumulada >= p  ->  conjunto mínimo
k = int(np.searchsorted(acum_ord, P, side="left"))
k = min(k, len(TOKENS) - 1)
supervivientes_ord = orden[: k + 1]
acum_corte = float(acum_ord[k])

es_sup = np.zeros(len(TOKENS), dtype=bool)
es_sup[supervivientes_ord] = True

dist_renorm = np.zeros(len(TOKENS), dtype=float)
dist_renorm[es_sup] = dist_T1[es_sup] / acum_corte   # renormalización

tokens_sup = [TOKENS[i] for i in supervivientes_ord]
tokens_eliminados = [TOKENS[i] for i in range(len(TOKENS)) if not es_sup[i]]
probs_sup = [float(dist_T1[i]) for i in supervivientes_ord]
acum_por_sup = {TOKENS[orden[j]]: float(acum_ord[j]) for j in range(k + 1)}
dist_renorm_dict = {TOKENS[i]: float(dist_renorm[i]) for i in supervivientes_ord}

suma_renorm = float(dist_renorm[es_sup].sum())
p_ren = dist_renorm[es_sup]
entropia_ren_bits = float(-np.sum(p_ren * np.log2(p_ren)))
entropia_T1_bits = float(-np.sum(dist_T1 * np.log2(dist_T1)))

# ------------------------------------------------------------------
# 3) Tabla del proceso (precisión completa + markdown a 4 decimales)
# ------------------------------------------------------------------
tabla = []
for j, idx in enumerate(orden):
    tabla.append({
        "posicion": j + 1,
        "token": TOKENS[idx],
        "prob_T1": float(dist_T1[idx]),
        "acumulada": float(acum_ord[j]),
        "sobrevive": bool(es_sup[idx]),
        "prob_renormalizada": float(dist_renorm[idx]),
    })

filas_md = []
for fila in tabla:
    filas_md.append(
        f"| {fila['posicion']} | {fila['token']} | {fila['prob_T1']:.4f} | "
        f"{fila['acumulada']:.4f} | {'si' if fila['sobrevive'] else 'no'} | "
        f"{fila['prob_renormalizada']:.4f} |"
    )
tabla_md = (
    "| Posicion | Token | p (T=1) | Acumulada | Sobrevive? | p renormalizada |\n"
    "|---|---|---|---|---|---|\n" + "\n".join(filas_md)
)

# ------------------------------------------------------------------
# 4) Figura: distribución original vs renormalizada + curva acumulada
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
x = np.arange(len(TOKENS))
ax.bar(x - 0.2, dist_T1, width=0.4, color="#8d99ae", label="Distribucion T=1 (Parte 1)")
ax.bar(x + 0.2, dist_renorm, width=0.4,
       color=["#2a9d8f" if s else "#e63946" for s in es_sup],
       label="Top-p renormalizada")
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_ylabel("Probabilidad")
ax.set_title(f"Top-p (p={P}) sobre la distribucion de T=1")
ax.legend(fontsize=8)

ax = axes[1]
ax.step(np.arange(1, len(TOKENS) + 1), acum_ord, where="post",
        marker="o", color="#264653", label="Acumulada (orden descendente)")
ax.axhline(P, color="#e76f51", ls="--", label=f"p = {P}")
ax.axvline(k + 1, color="#2a9d8f", ls=":", label="Corte del nucleo")
ax.set_xticks(np.arange(1, len(TOKENS) + 1))
ax.set_xticklabels([TOKENS[i] for i in orden])
ax.set_ylim(0, 1.05)
ax.set_xlabel("Tokens ordenados por probabilidad descendente")
ax.set_ylabel("Probabilidad acumulada")
ax.set_title("Probabilidad acumulada y corte top-p")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("T2_top_p_nucleus.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T2_top_p_nucleus",
    "p": P,
    "temperatura": 1.0,
    "fuente_distribucion_T1": fuente,
    "tokens": TOKENS,
    "distribucion_T1_parte1": [float(v) for v in dist_T1],
    "verificacion_coincidencia_softmax_T1_max_abs_diff": max_abs_diff,
    "orden_descendente": [TOKENS[i] for i in orden],
    "probs_ordenadas": [float(v) for v in probs_ord],
    "acumulada_ordenada": [float(v) for v in acum_ord],
    "tokens_supervivientes": tokens_sup,
    "probabilidades_originales_supervivientes": probs_sup,
    "probabilidad_acumulada_por_token_superviviente": acum_por_sup,
    "probabilidad_acumulada_del_corte": acum_corte,
    "probabilidad_acumulada_alcanza_p": bool(acum_corte >= P),
    "tokens_eliminados": tokens_eliminados,
    "distribucion_renormalizada": dist_renorm_dict,
    "distribucion_renormalizada_vector_alineado_t0_t5": [float(v) for v in dist_renorm],
    "suma_distribucion_renormalizada": suma_renorm,
    "error_suma_vs_1": abs(suma_renorm - 1.0),
    "entropia_bits_T1_original": entropia_T1_bits,
    "entropia_bits_renormalizada": entropia_ren_bits,
    "tabla_proceso": tabla,
    "tabla_markdown_4_decimales": tabla_md,
    "figura": "T2_top_p_nucleus.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("=== T2: Muestreo de nucleo (top-p) con p=0.9 sobre la distribucion de T=1 ===")
print(f"Fuente de la distribucion T=1: {fuente}")
print(f"Verificacion vs softmax estable de la Parte 1 (max |diff|): {max_abs_diff:.3e}")
print(f"Orden descendente: {[TOKENS[i] for i in orden]}")
print(f"Acumulada por token: {acum_por_sup}")
print(f"Tokens supervivientes: {tokens_sup} (acumulada del corte = {acum_corte:.10f} >= {P})")
print(f"Tokens eliminados: {tokens_eliminados}")
print("Distribucion renormalizada:")
for tok, val in dist_renorm_dict.items():
    print(f"  {tok}: {val:.10f}")
print(f"Suma de la distribucion renormalizada: {suma_renorm:.12f}")
print(f"Entropia (bits): T=1 original = {entropia_T1_bits:.6f} | renormalizada = {entropia_ren_bits:.6f}")
print("Figura guardada: T2_top_p_nucleus.png | Resultados en resultados.json")
