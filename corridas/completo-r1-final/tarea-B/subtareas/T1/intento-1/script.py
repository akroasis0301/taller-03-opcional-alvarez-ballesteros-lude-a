# T1 — Softmax con temperatura (NumPy, estable) + entropía en bits
# Tarea B · MMIA 6013 · Taller 03 v2

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ datos
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
TEMPERATURAS = [0.5, 1.0, 2.0]


# --------------------------------------------- softmax con temperatura
def softmax_con_temperatura(logits: np.ndarray, T: float) -> np.ndarray:
    """p_i = exp(z_i/T) / sum_j exp(z_j/T), numéricamente estable
    (se resta el máximo de los logits escalados antes de la exponencial)."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - np.max(z)  # estabilidad numérica
    e = np.exp(z)
    return e / e.sum()


def entropia_bits(p: np.ndarray) -> float:
    """Entropía de Shannon H(p) = -sum p_i log2(p_i), en bits."""
    p = np.asarray(p, dtype=float)
    p_nz = p[p > 0]
    return float(-np.sum(p_nz * np.log2(p_nz)))


# ------------------------------------------------------------- cálculos
distribuciones = {T: softmax_con_temperatura(LOGITS, T) for T in TEMPERATURAS}
entropias = {T: entropia_bits(p) for T, p in distribuciones.items()}
sumas = {T: float(p.sum()) for T, p in distribuciones.items()}

# Verificación del criterio de éxito: cada fila de probabilidades suma 1
for T, s in sumas.items():
    assert abs(s - 1.0) < 1e-12, f"La distribución T={T} no suma 1 (suma={s})"

# ------------------------------------------- tabla con cuatro decimales
filas = []
for T in TEMPERATURAS:
    fila = {"Temperatura": f"{T:g}"}
    for tok, pi in zip(TOKENS, distribuciones[T]):
        fila[tok] = f"{pi:.4f}"
    fila["Entropia_bits"] = f"{entropias[T]:.4f}"
    filas.append(fila)

tabla = pd.DataFrame(filas)

# Tabla en formato markdown (sin dependencias externas)
claves = ["Temperatura"] + TOKENS + ["Entropia_bits"]
lineas = [
    "| " + " | ".join(claves) + " |",
    "|" + "---|" * len(claves),
]
for fila in filas:
    lineas.append("| " + " | ".join(str(fila[k]) for k in claves) + " |")
tabla_md = "\n".join(lineas)

# ------------------------------------------------------- resultados.json
resultados = {
    "tokens": TOKENS,
    "logits": LOGITS.tolist(),
    "temperaturas": TEMPERATURAS,
    "softmax_numerically_estable": True,  # se resta el máximo de los logits
    "distribuciones": {f"T={T:g}": distribuciones[T].tolist() for T in TEMPERATURAS},
    "entropias_bits": {f"T={T:g}": entropias[T] for T in TEMPERATURAS},
    "suma_de_cada_fila": {f"T={T:g}": sumas[T] for T in TEMPERATURAS},
    "tabla_4_decimales": filas,
    "tabla_markdown_4_decimales": tabla_md,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------- figura
fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(TOKENS))
ancho = 0.25
colores = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i, T in enumerate(TEMPERATURAS):
    ax.bar(x + (i - 1) * ancho, distribuciones[T], width=ancho,
           color=colores[i], label=f"T={T:g}  (H = {entropias[T]:.4f} bits)")
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad")
ax.set_title("Softmax con temperatura — logits [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------- resumen
print("T1 — Softmax con temperatura (implementación NumPy estable, restando el máximo)")
print("Logits:", LOGITS.tolist())
print()
print(tabla.to_string(index=False))
print()
for T in TEMPERATURAS:
    print(f"T={T:g}: suma de probabilidades = {sumas[T]:.12f} | "
          f"entropía = {entropias[T]:.4f} bits")
print()
print("Archivos generados: resultados.json, t1_softmax_temperatura.png")
