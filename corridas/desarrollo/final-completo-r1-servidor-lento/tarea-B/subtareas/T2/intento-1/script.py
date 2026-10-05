# -*- coding: utf-8 -*-
"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1 (Parte 1).

Procedimiento:
  1) Se carga la distribución de T=1 obtenida en T1 (entrada/T1/resultados.json).
  2) Se ordenan los tokens por probabilidad descendente y se acumulan las probabilidades.
  3) Se conserva el conjunto MÍNIMO de tokens cuya probabilidad acumulada alcanza p = 0.9
     (incluye el token con el que la acumulada cruza el umbral).
  4) Se renormalizan las probabilidades de los supervivientes (dividiendo por su masa),
     de modo que sumen exactamente 1 y cada una sea la probabilidad original de T=1
     dividida por la masa del núcleo (coherencia con la Parte 1).

Salidas:
  - resultados.json       (contrato de la subtarea: supervivientes, renormalizada, verificaciones)
  - T2_top_p_nucleo.png   (figura: corte del núcleo y distribución renormalizada)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 0.9
T = 1.0
RUTA_T1 = Path("entrada") / "T1" / "resultados.json"
RUTA_SALIDA = Path("resultados.json")
RUTA_FIG = Path("T2_top_p_nucleo.png")


def softmax_con_temperatura(logits, temperatura):
    """Softmax con temperatura, numéricamente estable (resta el máximo)."""
    z = np.asarray(logits, dtype=float) / float(temperatura)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# ---------------------------------------------------------- 1) Distribución de T=1 (viene de T1)
tokens, logits, dist_T1, fuente = None, None, None, None
if RUTA_T1.exists():
    with open(RUTA_T1, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    tokens = list(t1.get("tokens", ["t0", "t1", "t2", "t3", "t4", "t5"]))
    logits = np.asarray(t1.get("logits", [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]), dtype=float)
    dists = t1.get("distribuciones", {})
    clave = "T=1.0" if "T=1.0" in dists else ("T=1" if "T=1" in dists else None)
    if clave is not None:
        dist_T1 = np.asarray(dists[clave], dtype=float)
        fuente = str(RUTA_T1)
if dist_T1 is None:  # respaldo: recalcular exactamente como en T1
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
    dist_T1 = softmax_con_temperatura(logits, T)
    fuente = "recalculada con softmax numéricamente estable (T=1)"

dist_T1 = dist_T1 / dist_T1.sum()  # blindaje numérico (no altera los valores de T1)

# ---------------------------------------------------------- 2) Top-p: núcleo con p = 0.9
orden = np.argsort(-dist_T1, kind="stable")            # índices en orden descendente
tokens_orden = [tokens[i] for i in orden]
probs_orden = dist_T1[orden]
acumulada = np.cumsum(probs_orden)

# Conjunto MÍNIMO cuya probabilidad acumulada alcanza p
# (searchsorted da el primer índice con acumulada >= p; +1 incluye ese token)
n_nucleo = int(np.searchsorted(acumulada, P, side="left")) + 1
n_nucleo = min(n_nucleo, len(tokens))

idx_nucleo = orden[:n_nucleo]
tokens_nucleo = [tokens[i] for i in idx_nucleo]
probs_nucleo = dist_T1[idx_nucleo]
masa_nucleo = float(probs_nucleo.sum())
probs_renorm = probs_nucleo / masa_nucleo

idx_fuera = orden[n_nucleo:]
tokens_excluidos = [tokens[i] for i in idx_fuera]
masa_descartada = float(dist_T1[idx_fuera].sum()) if idx_fuera.size else 0.0

# ---------------------------------------------------------- 3) Verificaciones
suma_renorm = float(probs_renorm.sum())
razones = {tk: float(pr / po) for tk, pr, po in zip(tokens_nucleo, probs_renorm, probs_nucleo)}
razon_esperada = 1.0 / masa_nucleo
razon_constante = bool(np.allclose(list(razones.values()), razon_esperada, rtol=1e-12, atol=0.0))
coherente = bool(np.allclose(probs_renorm * masa_nucleo, probs_nucleo, rtol=1e-12, atol=0.0))
ok_suma = bool(abs(suma_renorm - 1.0) <= 1e-12)
max_diff_T1 = float(np.max(np.abs(dist_T1 - softmax_con_temperatura(logits, T))))

# ---------------------------------------------------------- 4) Tablas y resultados
dist_original_dict = {tk: float(p) for tk, p in zip(tokens, dist_T1)}
probs_orig_nucleo_dict = {tk: float(p) for tk, p in zip(tokens_nucleo, probs_nucleo)}
renorm_dict = {tk: float(p) for tk, p in zip(tokens_nucleo, probs_renorm)}

tabla_ordenada = [
    {
        "token": tk,
        "prob_T1": float(pr),
        "prob_T1_4dec": f"{pr:.4f}",
        "acumulada": float(ac),
        "acumulada_4dec": f"{ac:.4f}",
        "en_nucleo": tk in tokens_nucleo,
    }
    for tk, pr, ac in zip(tokens_orden, probs_orden, acumulada)
]
tabla_renorm_4dec = {tk: f"{p:.4f}" for tk, p in renorm_dict.items()}

resultados = {
    "subtarea": "T2 - Muestreo de núcleo (top-p) con p=0.9 sobre la distribución de T=1",
    "p": P,
    "temperatura": T,
    "fuente_distribucion_T1": fuente,
    "tokens": tokens,
    "logits": logits.tolist(),
    "distribucion_T1_original": dist_original_dict,
    "orden_descendente_probabilidad": tokens_orden,
    "probs_orden_descendente": [float(x) for x in probs_orden],
    "prob_acumulada_orden_descendente": [float(x) for x in acumulada],
    "tokens_supervivientes": tokens_nucleo,
    "num_tokens_nucleo": n_nucleo,
    "masa_probabilidad_nucleo": masa_nucleo,
    "probabilidades_originales_supervivientes": probs_orig_nucleo_dict,
    "distribucion_renormalizada": renorm_dict,
    "distribucion_renormalizada_4dec": tabla_renorm_4dec,
    "tokens_excluidos": tokens_excluidos,
    "masa_descartada": masa_descartada,
    "suma_distribucion_renormalizada": suma_renorm,
    "verificaciones": {
        "suma_renormalizada_es_1": ok_suma,
        "error_absoluto_suma": abs(suma_renorm - 1.0),
        "razon_renormalizada_original_por_token": razones,
        "razon_esperada_1_sobre_masa_nucleo": razon_esperada,
        "razon_constante_entre_tokens": razon_constante,
        "renormalizada_por_masa_recupera_prob_T1": coherente,
        "max_abs_diff_distribucion_T1_vs_softmax_recalculada": max_diff_T1,
    },
    "tabla_ordenada_top_p": tabla_ordenada,
}

with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------- 5) Figura
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

colores = ["#2a9d8f" if tk in tokens_nucleo else "#d95f02" for tk in tokens_orden]
ax = axes[0]
ax.bar(range(len(tokens_orden)), probs_orden, color=colores, edgecolor="black", linewidth=0.6)
ax.plot(range(len(tokens_orden)), acumulada, "o--", color="#40517d", label="Prob. acumulada")
ax.axhline(P, color="crimson", linestyle=":", linewidth=1.5, label=f"p = {P}")
ax.set_xticks(range(len(tokens_orden)))
ax.set_xticklabels(tokens_orden)
ax.set_ylim(0, 1.08)
ax.set_xlabel("Token (orden descendente)")
ax.set_ylabel("Probabilidad")
ax.set_title("Top-p (p=0.9) sobre la distribución de T=1\nverde = núcleo, naranja = descartado")
ax.legend(fontsize=8, loc="upper left")

ax = axes[1]
ax.bar(range(n_nucleo), probs_renorm, color="#2a9d8f", edgecolor="black", linewidth=0.6)
ax.set_xticks(range(n_nucleo))
ax.set_xticklabels(tokens_nucleo)
ax.set_ylim(0, max(probs_renorm.max() * 1.2, 0.1))
ax.set_xlabel("Token superviviente")
ax.set_ylabel("Probabilidad renormalizada")
ax.set_title(f"Distribución renormalizada del núcleo\n(suma = {suma_renorm:.4f})")
for i, pv in enumerate(probs_renorm):
    ax.text(i, pv + 0.012, f"{pv:.4f}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(str(RUTA_FIG), dpi=120)
plt.close(fig)

# ---------------------------------------------------------- 6) Resumen
print("T2 — Top-p (muestreo de núcleo) con p=0.9 sobre la distribución de T=1")
print(f"Fuente de la distribución T=1: {fuente}")
print("Orden descendente : " + ", ".join(f"{tk}={pv:.4f}" for tk, pv in zip(tokens_orden, probs_orden)))
print("Acumulada         : " + ", ".join(f"{av:.4f}" for av in acumulada))
print(f"Núcleo (conjunto mínimo con acumulada >= {P}): {tokens_nucleo}  (masa = {masa_nucleo:.4f})")
print(f"Tokens excluidos  : {tokens_excluidos}  (masa descartada = {masa_descartada:.4f})")
print("Distribución renormalizada (suma = {:.12f}):".format(suma_renorm))
for tk in tokens_nucleo:
    print(f"  {tk}: renorm = {renorm_dict[tk]:.6f}   (original T=1 = {probs_orig_nucleo_dict[tk]:.6f},"
          f" razón = {razones[tk]:.6f})")
print(f"Verificaciones: suma=1 -> {ok_suma} | razón constante 1/masa -> {razon_constante} | "
      f"coherente con T1 -> {coherente}")
print(f"Resultados escritos en '{RUTA_SALIDA}' y figura en '{RUTA_FIG}'.")
