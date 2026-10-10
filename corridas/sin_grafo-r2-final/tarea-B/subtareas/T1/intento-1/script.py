# Subtarea T1: Softmax con temperatura (NumPy, estable) + entropía en bits
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Funciones ----------
def softmax_con_temperatura(logits, T):
    """p_i = exp(z_i/T) / sum_j exp(z_j/T), restando el máximo para estabilidad."""
    z = np.asarray(logits, dtype=np.float64)
    x = z / float(T)
    x = x - np.max(x)          # truco de estabilidad numérica
    e = np.exp(x)
    return e / np.sum(e)


def entropia_bits(p):
    """H(p) = -sum p_i log2 p_i, en bits."""
    p = np.asarray(p, dtype=np.float64)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


# ---------- Datos del enunciado ----------
logits = [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]
tokens = [f"t{i}" for i in range(6)]
temperaturas = [0.5, 1.0, 2.0]

# ---------- Cálculos ----------
distribuciones = {T: softmax_con_temperatura(logits, T) for T in temperaturas}
entropias = {T: entropia_bits(distribuciones[T]) for T in temperaturas}
sumas = {T: float(np.sum(distribuciones[T])) for T in temperaturas}

# ---------- Tabla con cuatro decimales ----------
tabla = pd.DataFrame(
    {f"T={T}": distribuciones[T] for T in temperaturas},
    index=tokens,
)
tabla.loc["Entropía (bits)"] = [entropias[T] for T in temperaturas]
tabla_fmt = tabla.apply(lambda col: col.map(lambda v: f"{v:.4f}"))

# ---------- Guardar tabla como CSV (apoyo) ----------
tabla_fmt.to_csv("tabla_softmax_temperatura.csv")

# ---------- Figura 1: tabla ----------
col_labels = ["Token"] + [f"T={T}" for T in temperaturas]
cell_text = [[tokens[i]] + [f"{distribuciones[T][i]:.4f}" for T in temperaturas]
             for i in range(6)]
cell_text.append(["Entropía (bits)"] + [f"{entropias[T]:.4f}" for T in temperaturas])

fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.axis("off")
tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.5)
ax.set_title("Softmax con temperatura: distribuciones y entropía (bits)", pad=12)
plt.savefig("tabla_softmax_temperatura.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------- Figura 2: barras de las tres distribuciones ----------
fig2, ax2 = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(tokens))
width = 0.25
for k, T in enumerate(temperaturas):
    ax2.bar(x + (k - 1) * width, distribuciones[T], width, label=f"T={T}")
ax2.set_xticks(x)
ax2.set_xticklabels(tokens)
ax2.set_ylabel("Probabilidad")
ax2.set_xlabel("Token")
ax2.set_title("Distribución softmax para T = 0.5, 1 y 2")
ax2.legend()
plt.savefig("softmax_temperatura_barras.png", dpi=120, bbox_inches="tight")
plt.close(fig2)

# ---------- resultados.json (contrato) ----------
resultados = {
    "subtarea": "T1_softmax_con_temperatura",
    "logits": logits,
    "tokens": tokens,
    "temperaturas": temperaturas,
    "distribuciones": {
        f"T={T}": [float(p) for p in distribuciones[T]] for T in temperaturas
    },
    "entropias_bits": {f"T={T}": entropias[T] for T in temperaturas},
    "suma_distribuciones": {f"T={T}": sumas[T] for T in temperaturas},
    "tabla_4_decimales": tabla_fmt.to_dict(orient="index"),
    "figuras": [
        "tabla_softmax_temperatura.png",
        "softmax_temperatura_barras.png",
        "tabla_softmax_temperatura.csv",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------- Resumen ----------
print("=== T1: Softmax con temperatura ===")
print(f"Logits: {logits}")
print(tabla_fmt.to_string())
print("\nSumas de cada distribución:")
for T in temperaturas:
    print(f"  T={T}: suma = {sumas[T]:.10f}")
print("\nEntropías (bits):")
for T in temperaturas:
    print(f"  T={T}: H = {entropias[T]:.4f} bits")
print("\nArchivos generados: resultados.json, tabla_softmax_temperatura.png, "
      "softmax_temperatura_barras.png, tabla_softmax_temperatura.csv")
