"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1 (Parte 1).

Pasos:
1. Carga la distribución de T=1 desde entrada/T1/resultados.json (subtarea previa)
   y la verifica recomputando la softmax con temperatura de forma numéricamente
   estable (si el archivo no existe, la recalcula desde los logits del enunciado).
2. Ordena los tokens por probabilidad descendente y calcula la probabilidad acumulada.
3. Conserva el conjunto MÍNIMO de tokens cuya acumulada alcanza p = 0.9
   (se incluye el token que cruza el umbral).
4. Renormaliza las probabilidades de los supervivientes (debe sumar 1).
5. Escribe resultados.json (contrato) y una figura PNG; imprime un resumen.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parámetros de la subtarea
# ----------------------------------------------------------------------
P = 0.9            # umbral top-p
TEMPERATURA = 1.0  # distribución de la Parte 1
RUTA_T1 = Path("entrada/T1/resultados.json")
LOGITS_ENUNCIADO = {"t0": 2.0, "t1": 1.0, "t2": 0.5, "t3": 0.2, "t4": -1.0, "t5": -3.0}


def softmax_con_temperatura(logits, T):
    """Softmax con temperatura, numéricamente estable (resta el máximo)."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------
# 1. Distribución de T=1 (de la subtarea T1, con verificación)
# ----------------------------------------------------------------------
if RUTA_T1.exists():
    with open(RUTA_T1, "r", encoding="utf-8") as f:
        res_t1 = json.load(f)
    tokens = list(res_t1["tokens"])
    logits = np.array([res_t1["logits"][t] for t in tokens], dtype=float)
    p_t1 = np.array([res_t1["distribuciones"]["T=1.0"][t] for t in tokens], dtype=float)
    fuente = str(RUTA_T1)
    coincide_softmax = bool(
        np.allclose(p_t1, softmax_con_temperatura(logits, TEMPERATURA), atol=1e-9)
    )
else:
    tokens = list(LOGITS_ENUNCIADO.keys())
    logits = np.array(list(LOGITS_ENUNCIADO.values()), dtype=float)
    p_t1 = softmax_con_temperatura(logits, TEMPERATURA)
    fuente = "softmax T=1 recalculada desde los logits del enunciado"
    coincide_softmax = True

# ----------------------------------------------------------------------
# 2. Orden descendente y probabilidad acumulada
# ----------------------------------------------------------------------
orden = np.argsort(-p_t1, kind="stable")
tokens_ordenados = [tokens[i] for i in orden]
probs_ordenadas = p_t1[orden]
acumuladas = np.cumsum(probs_ordenadas)

# ----------------------------------------------------------------------
# 3. Corte top-p: conjunto mínimo con acumulada >= p
# ----------------------------------------------------------------------
idx_corte = int(np.searchsorted(acumuladas, P, side="left"))
if idx_corte >= len(acumuladas):  # salvaguarda (p < suma total = 1)
    idx_corte = len(acumuladas) - 1
n_sup = idx_corte + 1

tokens_sup = tokens_ordenados[:n_sup]
probs_sup = probs_ordenadas[:n_sup]
masa_sup = float(probs_sup.sum())

# ----------------------------------------------------------------------
# 4. Renormalización de los supervivientes
# ----------------------------------------------------------------------
p_renorm = probs_sup / masa_sup
suma_renorm = float(p_renorm.sum())

tabla_ordenada = [
    {
        "token": tokens_ordenados[i],
        "probabilidad": float(probs_ordenadas[i]),
        "acumulada": float(acumuladas[i]),
        "sobrevive": i < n_sup,
    }
    for i in range(len(tokens_ordenados))
]
distribucion_renormalizada = {
    tokens_sup[i]: float(p_renorm[i]) for i in range(n_sup)
}

# ----------------------------------------------------------------------
# 5. Contrato: resultados.json
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": "Muestreo de núcleo (top-p) con p=0.9 sobre la distribución de T=1",
    "parametros": {"p": P, "temperatura": TEMPERATURA},
    "fuente_distribucion": fuente,
    "distribucion_T1_original": {t: float(p_t1[i]) for i, t in enumerate(tokens)},
    "orden_descendente": {
        "tokens": tokens_ordenados,
        "probabilidades": [float(x) for x in probs_ordenadas],
        "probabilidades_acumuladas": [float(x) for x in acumuladas],
    },
    "tabla_ordenada": tabla_ordenada,
    "corte_top_p": {
        "p": P,
        "indice_corte": idx_corte,
        "num_supervivientes": n_sup,
        "tokens_supervivientes": tokens_sup,
        "tokens_descartados": tokens_ordenados[n_sup:],
        "masa_probabilidad_supervivientes": masa_sup,
        "masa_descartada": float(1.0 - masa_sup),
    },
    "distribucion_renormalizada": distribucion_renormalizada,
    "suma_distribucion_renormalizada": suma_renorm,
    "verificacion": {
        "coincide_con_softmax_recalculada": coincide_softmax,
        "suma_renormalizada_es_1": bool(np.isclose(suma_renorm, 1.0)),
        "acumulada_ultimo_superviviente_alcanza_p": bool(acumuladas[idx_corte] >= P),
        "acumulada_anterior_menor_que_p": bool(
            idx_corte == 0 or acumuladas[idx_corte - 1] < P
        ),
    },
}
resultados["verificacion"]["conjunto_es_minimo"] = bool(
    resultados["verificacion"]["acumulada_ultimo_superviviente_alcanza_p"]
    and resultados["verificacion"]["acumulada_anterior_menor_que_p"]
)

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 6. Figura: probs. ordenadas + acumulada + corte, y renormalizada
# ----------------------------------------------------------------------
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = ejes[0]
x = np.arange(len(tokens_ordenados))
barras = ax.bar(x, probs_ordenadas, color="#4C72B0", label="p_i (T=1)")
for i in range(n_sup, len(x)):
    barras[i].set_color("#C44E52")
    barras[i].set_alpha(0.45)
ax2 = ax.twinx()
ax2.plot(x, acumuladas, "o-", color="#55A868", label="acumulada")
ax2.axhline(P, color="k", ls="--", lw=1, label=f"p = {P}")
ax2.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(tokens_ordenados)
ax.set_ylabel("probabilidad")
ax2.set_ylabel("probabilidad acumulada")
ax.set_title("Top-p (p=0.9): probabilidades ordenadas y acumulada")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)

ax = ejes[1]
xs = np.arange(n_sup)
ax.bar(xs, p_renorm, color="#55A868")
ax.set_xticks(xs)
ax.set_xticklabels(tokens_sup)
ax.set_ylim(0, max(p_renorm) * 1.15)
ax.set_ylabel("probabilidad renormalizada")
ax.set_title(f"Distribución renormalizada (suma = {suma_renorm:.6f})")
for i, v in enumerate(p_renorm):
    ax.text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=8)

fig.tight_layout()
fig.savefig("T2_top_p_renormalizacion.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 7. Resumen
# ----------------------------------------------------------------------
print("T2 — Top-p (nucleus sampling) con p=0.9 sobre la distribución de T=1")
print("-" * 70)
print(f"{'token':<6}{'probabilidad':>16}{'acumulada':>14}{'sobrevive':>12}")
for fila in tabla_ordenada:
    print(
        f"{fila['token']:<6}{fila['probabilidad']:>16.10f}"
        f"{fila['acumulada']:>14.6f}{str(fila['sobrevive']):>12}"
    )
print("-" * 70)
print(f"Supervivientes (conjunto mínimo con acumulada >= {P}): {tokens_sup}")
print(f"Masa de probabilidad de los supervivientes: {masa_sup:.10f}")
print(f"Tokens descartados: {tokens_ordenados[n_sup:]} (masa descartada {1.0 - masa_sup:.10f})")
print("Distribución renormalizada:")
for t, v in distribucion_renormalizada.items():
    print(f"  {t}: {v:.10f}")
print(f"Suma de la distribución renormalizada: {suma_renorm:.12f}")
print("Figura guardada: T2_top_p_renormalizacion.png")
print("Resultados escritos en resultados.json")
