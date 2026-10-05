# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Softmax con temperatura (NumPy, estable) y entropía en bits.
Tarea B — Temperatura, top-p y entropía de la distribución de salida (MMIA 6013).

Calcula la distribución softmax con temperatura para T = 0.5, 1 y 2 sobre los
logits dados (t0..t5), la entropía en bits de cada una, y presenta una tabla
con cuatro decimales. Guarda los resultados en resultados.json y una figura PNG.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Datos del enunciado
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]


# ----------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable
# ----------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    """
    p_i = exp(z_i / T) / sum_j exp(z_j / T)

    Estabilidad numérica: se resta el máximo de los logits escalados
    (z/T) antes de la exponencial, de modo que el mayor exponente es 0
    y no ocurre desbordamiento (overflow) en exp.
    """
    z = np.asarray(z, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("La temperatura debe ser positiva.")
    escalados = z / T
    escalados = escalados - np.max(escalados)  # truco de estabilidad
    e = np.exp(escalados)
    return e / np.sum(e)


def entropia_bits(p):
    """H(p) = -sum p_i * log2(p_i), en bits (se omiten p_i = 0)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ----------------------------------------------------------------------
# Cálculo de las tres distribuciones y sus entropías
# ----------------------------------------------------------------------
distribuciones = {}
entropias = {}
for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)

# Verificaciones del criterio de éxito
sumas = {f"T={T}": float(np.sum(distribuciones[T])) for T in temperaturas}
orden_entropia_ok = (entropias[0.5] < entropias[1.0]) and (entropias[1.0] < entropias[2.0])

# ----------------------------------------------------------------------
# Tabla con cuatro decimales
# ----------------------------------------------------------------------
tabla = pd.DataFrame(index=tokens)
for T in temperaturas:
    tabla[f"T={T}"] = [f"{p:.4f}" for p in distribuciones[T]]
tabla.loc["Entropía (bits)"] = {f"T={T}": f"{entropias[T]:.4f}" for T in temperaturas}
tabla.loc["Suma"] = {f"T={T}": f"{sumas[f'T={T}']:.4f}" for T in temperaturas}

print("=" * 60)
print("T1 — Softmax con temperatura y entropía (bits)")
print("=" * 60)
print("Logits:", dict(zip(tokens, logits)))
print()
print(tabla.to_string())
print()

# ----------------------------------------------------------------------
# Figura: distribuciones para las tres temperaturas
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(tokens))
ancho = 0.25
colores = ["#1f77b4", "#2ca02c", "#d62728"]
for i, T in enumerate(temperaturas):
    ax.bar(x + (i - 1) * ancho, distribuciones[T], width=ancho,
           label=f"T={T}  (H={entropias[T]:.4f} bits)", color=colores[i])
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad")
ax.set_xlabel("Token")
ax.set_title("Softmax con temperatura — logits [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "descripcion": "Softmax con temperatura (NumPy, numéricamente estable) y entropía en bits",
    "tokens": tokens,
    "logits": {t: float(z) for t, z in zip(tokens, logits)},
    "temperaturas": temperaturas,
    "distribuciones": {
        f"T={T}": {t: float(p) for t, p in zip(tokens, distribuciones[T])}
        for T in temperaturas
    },
    "entropias_bits": {f"T={T}": entropias[T] for T in temperaturas},
    "sumas_distribuciones": sumas,
    "tabla_4_decimales": {
        "probabilidades": {
            f"T={T}": [f"{p:.4f}" for p in distribuciones[T]] for T in temperaturas
        },
        "entropias_bits": {f"T={T}": f"{entropias[T]:.4f}" for T in temperaturas},
    },
    "verificacion": {
        "distribuciones_suman_1": all(abs(s - 1.0) < 1e-12 for s in sumas.values()),
        "H_T0.5_menor_H_T1": bool(entropias[0.5] < entropias[1.0]),
        "H_T1_menor_H_T2": bool(entropias[1.0] < entropias[2.0]),
        "orden_entropias_correcto": bool(orden_entropia_ok),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------
print("Resumen:")
for T in temperaturas:
    print(f"  T={T}: H = {entropias[T]:.4f} bits | suma dist = {sumas[f'T={T}']:.6f}")
print(f"  Orden de entropías H(0.5) < H(1) < H(2): {orden_entropia_ok}")
print("Archivos generados: resultados.json, t1_softmax_temperatura.png")
