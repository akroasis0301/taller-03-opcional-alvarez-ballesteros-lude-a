import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================
# SUBTAREA T1 — Softmax con temperatura (NumPy, estable) + entropía en bits
# ==================================================================

LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
TOKENS = [f"token_{i+1}" for i in range(LOGITS.size)]
TEMPERATURAS = [0.5, 1.0, 2.0]


def softmax_con_temperatura(z, T):
    """p_i = exp(z_i/T) / sum_j exp(z_j/T).

    Numéricamente estable: se resta el máximo de los logits escalados
    antes de exponenciar (no cambia la distribución, evita overflow).
    """
    z = np.asarray(z, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("La temperatura T debe ser positiva.")
    z_escalado = z / T
    z_escalado = z_escalado - np.max(z_escalado)  # estabilidad numérica
    exp_z = np.exp(z_escalado)
    return exp_z / np.sum(exp_z)


def entropia_en_bits(p):
    """H(p) = -sum p_i log2(p_i), ignorando entradas nulas (0*log0 = 0)."""
    p = np.asarray(p, dtype=float)
    mascara = p > 0
    return float(-np.sum(p[mascara] * np.log2(p[mascara])))


# ---------------- Distribuciones y entropías ----------------
distribuciones = {T: softmax_con_temperatura(LOGITS, T) for T in TEMPERATURAS}
entropias = {T: entropia_en_bits(distribuciones[T]) for T in TEMPERATURAS}
sumas = {T: float(distribuciones[T].sum()) for T in TEMPERATURAS}

# Verificación: cada distribución suma 1
for T in TEMPERATURAS:
    assert np.isclose(sumas[T], 1.0, atol=1e-12), f"T={T} no suma 1"

# ---------------- Tabla (4 decimales) ----------------
# Una fila por temperatura: 6 probabilidades + entropía en bits.
tabla = pd.DataFrame(
    data=[np.concatenate([distribuciones[T], [entropias[T]]]) for T in TEMPERATURAS],
    index=[f"T={T}" for T in TEMPERATURAS],
    columns=TOKENS + ["Entropia_bits"],
)
tabla_celdas = [[f"{v:.4f}" for v in fila] for fila in tabla.to_numpy()]

encabezado = "| Temperatura | " + " | ".join(TOKENS) + " | Entropía (bits) |"
separador = "|" + "---|" * (len(TOKENS) + 2)
filas_md = [
    f"| {idx} | " + " | ".join(celdas) + " |"
    for idx, celdas in zip(tabla.index, tabla_celdas)
]
tabla_markdown = "\n".join([encabezado, separador] + filas_md)

# ---------------- Figura: efecto de la temperatura ----------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(TOKENS))
ancho = 0.25
for k, T in enumerate(TEMPERATURAS):
    ax.bar(x + (k - 1) * ancho, distribuciones[T], width=ancho, label=f"T={T}")
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_ylabel("Probabilidad")
ax.set_title("Softmax con temperatura (logits 2.0, 1.0, 0.5, 0.2, -1.0, -3.0)")
ax.legend()
fig.tight_layout()
fig.savefig("softmax_temperaturas.png", dpi=120)
plt.close(fig)

# ---------------- resultados.json (contrato) ----------------
resultados = {
    "subtarea": "T1_softmax_con_temperatura",
    "logits": LOGITS.tolist(),
    "tokens": TOKENS,
    "temperaturas": TEMPERATURAS,
    "distribuciones": {f"T={T}": distribuciones[T].tolist() for T in TEMPERATURAS},
    "entropias_bits": {f"T={T}": entropias[T] for T in TEMPERATURAS},
    "suma_de_cada_distribucion": {f"T={T}": sumas[T] for T in TEMPERATURAS},
    "tabla_cuatro_decimales": {
        "columnas": TOKENS + ["Entropia_bits"],
        "filas": list(tabla.index),
        "valores": tabla_celdas,
    },
    "tabla_markdown": tabla_markdown,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------- Resumen ----------------
print("T1 — Softmax con temperatura (distribución y entropía en bits)")
print(tabla_markdown)
print("\nSuma de cada fila de probabilidades:",
      {f"T={T}": f"{sumas[T]:.12f}" for T in TEMPERATURAS})
print("Figura guardada: softmax_temperaturas.png")
print("Resultados escritos en resultados.json")
