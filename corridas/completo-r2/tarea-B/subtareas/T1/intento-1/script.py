# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Softmax con temperatura (NumPy, estable) + entropía en bits.
Tarea B — Temperatura, top-p y entropía de la distribución de salida (MMIA 6013).

Calcula la distribución softmax con temperatura para T = 0.5, 1 y 2 con los
logits t0..t5 = 2.0, 1.0, 0.5, 0.2, -1.0, -3.0, la entropía en bits de cada
una, y presenta todo en una tabla con cuatro decimales.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Datos del enunciado
# ----------------------------------------------------------------------------
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
TEMPERATURAS = [0.5, 1.0, 2.0]


# ----------------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable (max-subtraction)
# ----------------------------------------------------------------------------
def softmax_con_temperatura(logits, T):
    """p_i = exp(z_i / T) / sum_j exp(z_j / T), restando el máximo de los
    logits escalados para evitar overflow/underflow."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - np.max(z)          # estabilidad numérica: max-subtraction
    e = np.exp(z)
    return e / np.sum(e)


def entropia_bits(p):
    """H(p) = -sum p_i * log2(p_i), en bits (términos nulos se omiten)."""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


# ----------------------------------------------------------------------------
# Cálculo de distribuciones y entropías
# ----------------------------------------------------------------------------
distribuciones = {}
entropias = {}
sumas = {}

for T in TEMPERATURAS:
    p = softmax_con_temperatura(LOGITS, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)
    sumas[T] = float(np.sum(p))
    assert abs(sumas[T] - 1.0) < 1e-12, f"La distribución con T={T} no suma 1.0"

# ----------------------------------------------------------------------------
# Tabla con cuatro decimales (distribuciones + entropías)
# ----------------------------------------------------------------------------
tabla = pd.DataFrame(
    {f"T={T}": [f"{p:.4f}" for p in distribuciones[T]] + [f"{entropias[T]:.4f}"]
     for T in TEMPERATURAS},
    index=TOKENS + ["Entropia_bits"],
)
tabla.index.name = "Token"

# ----------------------------------------------------------------------------
# resultados.json (contrato): valores sin redondear + tabla a 4 decimales
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1 - Softmax con temperatura y entropia en bits",
    "logits": {t: float(z) for t, z in zip(TOKENS, LOGITS)},
    "temperaturas": [float(T) for T in TEMPERATURAS],
    "softmax_con_temperatura": {
        f"T={T}": {
            "distribucion": {t: float(p) for t, p in zip(TOKENS, distribuciones[T])},
            "suma_distribucion": sumas[T],
            "entropia_bits": entropias[T],
        }
        for T in TEMPERATURAS
    },
    "entropias_bits": {f"T={T}": entropias[T] for T in TEMPERATURAS},
    "tabla_4_decimales": {
        str(idx): {col: tabla.loc[idx, col] for col in tabla.columns}
        for idx in tabla.index
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# Figura: distribuciones softmax para las tres temperaturas
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(TOKENS))
width = 0.25
for i, T in enumerate(TEMPERATURAS):
    ax.bar(x + (i - 1) * width, distribuciones[T], width, label=f"T = {T}")
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_ylabel("Probabilidad")
ax.set_xlabel("Token")
ax.set_title("Softmax con temperatura (logits t0..t5)")
ax.legend()
fig.tight_layout()
plt.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------------
print("Softmax con temperatura (distribuciones y entropia en bits):")
print(tabla.to_string())
print()
for T in TEMPERATURAS:
    print(f"T={T}: suma={sumas[T]:.6f}  entropia={entropias[T]:.4f} bits")
print("\nArchivos generados: resultados.json, t1_softmax_temperatura.png")
