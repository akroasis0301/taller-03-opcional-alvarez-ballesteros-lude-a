import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución T=1
# =====================================================================

P = 0.9


def softmax_T1(logits):
    """Softmax con temperatura T=1, numéricamente estable (resta del máximo)."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------
# 1) Distribución de T=1 de la Parte 1 (entrada/T1/resultados.json)
# ---------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
p_t1 = softmax_T1(logits)
fuente = "respaldo: softmax estable recalculada con T=1"

ruta_t1 = Path("entrada/T1/resultados.json")
if ruta_t1.exists():
    try:
        with ruta_t1.open("r", encoding="utf-8") as f:
            res_t1 = json.load(f)
        tokens = list(res_t1["tokens"])
        logits = np.asarray(res_t1["logits"], dtype=float)
        p_t1 = np.asarray(res_t1["distribuciones"]["T=1"], dtype=float)
        fuente = "entrada/T1/resultados.json (distribución de T=1 de la Parte 1)"
        # Consistencia: lo cargado coincide con la softmax T=1 bien calculada
        assert np.allclose(p_t1, softmax_T1(logits), atol=1e-12)
    except (KeyError, TypeError, AssertionError):
        tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
        logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
        p_t1 = softmax_T1(logits)
        fuente = "respaldo: softmax estable recalculada con T=1"

# ---------------------------------------------------------------------
# 2) Ordenar tokens por probabilidad descendente y acumular
# ---------------------------------------------------------------------
orden = np.argsort(-p_t1, kind="stable")
tokens_ord = [tokens[i] for i in orden]
probs_ord = p_t1[orden]
acum_ord = np.cumsum(probs_ord)

# ---------------------------------------------------------------------
# 3) Conjunto MÍNIMO cuya probabilidad acumulada alcanza p = 0.9
#    (se incluye el token que hace cruzar el umbral)
# ---------------------------------------------------------------------
n_nucleo = int(np.searchsorted(acum_ord, P, side="left")) + 1
idx_nucleo = orden[:n_nucleo]
tokens_nucleo = [tokens[i] for i in idx_nucleo]
tokens_fuera = [tokens[i] for i in orden[n_nucleo:]]
acum_nucleo = float(acum_ord[n_nucleo - 1])

# ---------------------------------------------------------------------
# 4) Renormalizar las probabilidades de los supervivientes
# ---------------------------------------------------------------------
p_nucleo = p_t1[idx_nucleo]
p_renorm = p_nucleo / p_nucleo.sum()
suma_renorm = float(p_renorm.sum())
dist_renorm_dict = {tokens[i]: float(p_renorm[j]) for j, i in enumerate(idx_nucleo)}

# ---------------------------------------------------------------------
# 5) Verificaciones del criterio de éxito
# ---------------------------------------------------------------------
assert abs(suma_renorm - 1.0) < 1e-12, "La distribución renormalizada debe sumar 1"
assert acum_nucleo >= P, "La probabilidad acumulada del núcleo debe alcanzar p"
if n_nucleo > 1:
    assert acum_ord[n_nucleo - 2] < P, "El núcleo debe ser el conjunto mínimo"
assert np.allclose(p_renorm, p_t1[idx_nucleo] / acum_nucleo, atol=0)

# ---------------------------------------------------------------------
# 6) Tabla (4 decimales) con el procedimiento completo
# ---------------------------------------------------------------------
p_renorm_full = np.concatenate([p_renorm, np.zeros(len(tokens_ord) - n_nucleo)])
tabla_filas = []
for j, tok in enumerate(tokens_ord):
    es_nuc = j < n_nucleo
    tabla_filas.append([
        tok,
        f"{probs_ord[j]:.4f}",
        f"{acum_ord[j]:.4f}",
        "sí" if es_nuc else "no",
        f"{p_renorm_full[j]:.4f}" if es_nuc else None,
    ])

# ---------------------------------------------------------------------
# 7) Figura: distribución original vs renormalizada y acumulada
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(tokens_ord))
colores = ["#2a9d8f" if j < n_nucleo else "#d95f02" for j in range(len(tokens_ord))]

ax1.bar(x - 0.2, probs_ord, width=0.4, color=colores, label="T=1 original")
ax1.bar(x + 0.2, p_renorm_full, width=0.4, color=colores, alpha=0.45,
        label="top-p renormalizada")
ax1.set_xticks(x)
ax1.set_xticklabels(tokens_ord)
ax1.set_ylabel("probabilidad")
ax1.set_title(f"Top-p (p={P}): núcleo = {{{', '.join(tokens_nucleo)}}}")
ax1.legend()

ax2.plot(np.arange(1, len(tokens_ord) + 1), acum_ord, marker="o",
         color="#457b9d", label="prob. acumulada (orden desc.)")
ax2.axhline(P, color="#d95f02", ls="--", label=f"p = {P}")
ax2.axvline(n_nucleo, color="gray", ls=":", label="corte del núcleo")
ax2.annotate(f"acum. núcleo = {acum_nucleo:.4f}",
             xy=(n_nucleo, acum_nucleo), xytext=(n_nucleo + 0.2, acum_nucleo - 0.12),
             fontsize=9)
ax2.set_xticks(np.arange(1, len(tokens_ord) + 1))
ax2.set_xticklabels(tokens_ord)
ax2.set_ylabel("probabilidad acumulada")
ax2.set_title("Acumulado y corte top-p")
ax2.legend(loc="lower right")

fig.tight_layout()
fig.savefig("T2_top_p_nucleus.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------
# 8) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------
resultados = {
    "subtarea": "T2_top_p_nucleus_sampling",
    "p": P,
    "temperatura": 1.0,
    "fuente_distribucion": fuente,
    "tokens": tokens,
    "logits": logits.tolist(),
    "distribucion_T1_original": p_t1.tolist(),
    "suma_distribucion_T1_original": float(p_t1.sum()),
    "orden_descendente": tokens_ord,
    "probs_ordenadas": probs_ord.tolist(),
    "prob_acumulada_ordenada": acum_ord.tolist(),
    "tokens_sobrevivientes": tokens_nucleo,
    "tokens_filtrados": tokens_fuera,
    "num_tokens_nucleo": n_nucleo,
    "prob_acumulada_nucleo": acum_nucleo,
    "distribucion_renormalizada": dist_renorm_dict,
    "distribucion_renormalizada_orden_descendente": p_renorm.tolist(),
    "suma_distribucion_renormalizada": suma_renorm,
    "tabla_cuatro_decimales": {
        "columnas": ["token", "prob_T1_original", "prob_acumulada",
                     "superviviente", "prob_renormalizada"],
        "filas": tabla_filas,
    },
    "figura": "T2_top_p_nucleus.png",
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------
# 9) Resumen
# ---------------------------------------------------------------------
print("T2 — Top-p (nucleus sampling) con p=0.9 sobre la distribución T=1 de la Parte 1")
print(f"Fuente de la distribución: {fuente}")
print(f"Orden descendente: {tokens_ord}")
print(f"Prob. acumulada (orden desc.): {[round(a, 4) for a in acum_ord.tolist()]}")
print(f"Tokens supervivientes ({n_nucleo}): {tokens_nucleo} "
      f"| prob. acumulada del núcleo = {acum_nucleo:.6f}")
print(f"Tokens filtrados: {tokens_fuera}")
print("Distribución renormalizada:")
for t, v in dist_renorm_dict.items():
    print(f"  {t}: {v:.6f}")
print(f"Suma de la distribución renormalizada: {suma_renorm:.12f}")
print("Figura: T2_top_p_nucleus.png | Resultados: resultados.json")
