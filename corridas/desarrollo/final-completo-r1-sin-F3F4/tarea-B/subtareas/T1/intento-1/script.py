"""
Subtarea T1 — Softmax con temperatura y entropía en bits (NumPy).

Tarea B (MMIA 6013) · Parte 1: implementar la softmax con temperatura
    p_i = exp(z_i / T) / Σ_j exp(z_j / T)
de forma numéricamente estable (restando el máximo antes de exponenciar),
calcularla para T = 0.5, 1 y 2 sobre los logits dados, y reportar la
entropía en bits de cada distribución en una tabla con cuatro decimales.

Salidas:
    - resultados.json  (contrato: distribuciones, entropías, tabla 4 decimales)
    - tabla_softmax_temperatura.csv
    - softmax_temperatura.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Datos del enunciado
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]


# ----------------------------------------------------------------------
# 2. Softmax con temperatura, numéricamente estable
#    p_i = exp((z_i - max(z)) / T) / Σ_j exp((z_j - max(z)) / T)
#    Restar el máximo de los logits antes de exponenciar evita
#    desbordamiento (overflow) sin alterar el resultado.
# ----------------------------------------------------------------------
def softmax_con_temperatura(z: np.ndarray, T: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if T <= 0:
        raise ValueError("La temperatura T debe ser positiva.")
    z_escalado = z / T
    z_escalado = z_escalado - np.max(z_escalado)  # estabilidad numérica
    exp_z = np.exp(z_escalado)
    return exp_z / np.sum(exp_z)


def entropia_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # 0 * log2(0) = 0 por convención
    return float(-np.sum(p * np.log2(p)))


# ----------------------------------------------------------------------
# 3. Cálculo de distribuciones y entropías
# ----------------------------------------------------------------------
distribuciones = {}
entropias = {}
for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)

# ----------------------------------------------------------------------
# 4. Tabla con cuatro decimales (distribuciones + entropías)
# ----------------------------------------------------------------------
filas = []
for T in temperaturas:
    fila = {"T": T}
    for tok, p_ij in zip(tokens, distribuciones[T]):
        fila[tok] = f"{p_ij:.4f}"
    fila["entropia_bits"] = f"{entropias[T]:.4f}"
    filas.append(fila)

tabla = pd.DataFrame(filas)
tabla.to_csv("tabla_softmax_temperatura.csv", index=False)

# Versión numérica (sin formatear) para verificación
tabla_num = pd.DataFrame(
    {
        "T": temperaturas,
        **{
            tok: [distribuciones[T][i] for T in temperaturas]
            for i, tok in enumerate(tokens)
        },
        "entropia_bits": [entropias[T] for T in temperaturas],
    }
)

# ----------------------------------------------------------------------
# 5. Verificaciones del criterio de éxito
# ----------------------------------------------------------------------
sumas = {f"T={T}": float(np.sum(distribuciones[T])) for T in temperaturas}
entropia_crece = bool(
    entropias[0.5] < entropias[1.0] < entropias[2.0]
)

# ----------------------------------------------------------------------
# 6. Figura: distribuciones por temperatura
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(tokens))
ancho = 0.25
for k, T in enumerate(temperaturas):
    ax.bar(x + (k - 1) * ancho, distribuciones[T], width=ancho, label=f"T = {T}")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad")
ax.set_xlabel("Token")
ax.set_title("Softmax con temperatura — logits [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]")
ax.legend()
fig.tight_layout()
plt.savefig("softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 7. resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {
        f"T={T}": {tok: float(p) for tok, p in zip(tokens, distribuciones[T])}
        for T in temperaturas
    },
    "distribuciones_listas": {
        f"T={T}": distribuciones[T].tolist() for T in temperaturas
    },
    "entropias_bits": {f"T={T}": entropias[T] for T in temperaturas},
    "tabla_4_decimales": filas,
    "verificacion": {
        "suma_por_distribucion": sumas,
        "todas_suman_1": bool(
            all(abs(s - 1.0) < 1e-12 for s in sumas.values())
        ),
        "entropia_crece_con_T": entropia_crece,
        "entropias_ordenadas": [
            entropias[T] for T in sorted(temperaturas)
        ],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 8. Resumen
# ----------------------------------------------------------------------
print("Subtarea T1 — Softmax con temperatura y entropía en bits")
print(f"Logits: {logits.tolist()}")
print("\nTabla (4 decimales):")
print(tabla.to_string(index=False))
print("\nVerificación:")
for T in temperaturas:
    print(
        f"  T={T}: suma = {sumas[f'T={T}']:.12f}, "
        f"H = {entropias[T]:.4f} bits"
    )
print(f"  ¿Todas las distribuciones suman 1? {resultados['verificacion']['todas_suman_1']}")
print(f"  ¿La entropía crece al aumentar T?  {entropia_crece}")
print("\nArchivos generados: resultados.json, tabla_softmax_temperatura.csv, softmax_temperatura.png")
