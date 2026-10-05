# T1: Softmax con temperatura (NumPy, estable), distribuciones para T=0.5, 1, 2,
# entropía en bits y tabla con cuatro decimales.

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Datos del enunciado ----------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]


# ---------- Implementación ----------
def softmax_con_temperatura(z, T):
    """p_i = exp(z_i/T) / sum_j exp(z_j/T), numéricamente estable.

    Se resta el máximo de los logits escalados antes de exponenciar para
    evitar desbordamiento (overflow) en exp.
    """
    z = np.asarray(z, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("La temperatura debe ser positiva.")
    z_esc = z / T
    z_esc = z_esc - np.max(z_esc)  # estabilidad numérica
    e = np.exp(z_esc)
    return e / np.sum(e)


def entropia_bits(p):
    """H = -sum p_i * log2(p_i), en bits (se ignoran p_i = 0)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ---------- Cálculos ----------
distribuciones = {T: softmax_con_temperatura(logits, T) for T in temperaturas}
entropias = {T: entropia_bits(distribuciones[T]) for T in temperaturas}
sumas = {T: float(np.sum(distribuciones[T])) for T in temperaturas}
monotona = bool(entropias[0.5] < entropias[1.0] < entropias[2.0])


# ---------- Tabla con cuatro decimales ----------
tabla = pd.DataFrame(
    {f"T={T}": [f"{p:.4f}" for p in distribuciones[T]] + [f"{entropias[T]:.4f}"]
     for T in temperaturas},
    index=tokens + ["Entropía (bits)"],
)


# ---------- Figura ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, T in zip(axes, temperaturas):
    ax.bar(tokens, distribuciones[T], color="steelblue", edgecolor="black")
    ax.set_title(f"T = {T}   (H = {entropias[T]:.4f} bits)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Token")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Probabilidad")
fig.suptitle("Softmax con temperatura sobre logits [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]")
fig.tight_layout()
plt.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)


# ---------- resultados.json (contrato) ----------
resultados = {
    "logits": {t: float(z) for t, z in zip(tokens, logits)},
    "temperaturas": temperaturas,
    "distribuciones": {
        f"T={T}": {t: float(p) for t, p in zip(tokens, distribuciones[T])}
        for T in temperaturas
    },
    "entropias_bits": {f"T={T}": entropias[T] for T in temperaturas},
    "suma_distribuciones": {f"T={T}": sumas[T] for T in temperaturas},
    "entropia_monotona_creciente": monotona,
    "tabla_4_decimales": {
        "filas": tokens + ["entropia_bits"],
        "columnas": [f"T={T}" for T in temperaturas],
        "valores": (
            [[f"{distribuciones[T][i]:.4f}" for T in temperaturas] for i in range(len(tokens))]
            + [[f"{entropias[T]:.4f}" for T in temperaturas]]
        ),
    },
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)


# ---------- Resumen ----------
print("Tabla (4 decimales):")
print(tabla.to_string())
print("\nSuma de cada distribución:", {f"T={T}": round(s, 12) for T, s in sumas.items()})
print("Entropías (bits):", {f"T={T}": round(h, 6) for T, h in entropias.items()})
print("Entropía monótona creciente con T:", monotona)
print("Archivos generados: resultados.json, t1_softmax_temperatura.png")
