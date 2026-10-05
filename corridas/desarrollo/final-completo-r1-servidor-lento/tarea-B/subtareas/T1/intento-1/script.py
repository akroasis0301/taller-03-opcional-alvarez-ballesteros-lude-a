# T1 — Softmax con temperatura (NumPy, numéricamente estable) + entropía en bits
# MMIA 6013 · Tarea B · Taller 03 v2

import json

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
# Implementación
# ----------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    """p_i = exp(z_i/T) / sum_j exp(z_j/T), restando el máximo de los
    logits escalados para estabilidad numérica."""
    z = np.asarray(z, dtype=float)
    escalados = z / float(T)
    escalados = escalados - escalados.max()  # estabilidad numérica
    exp_z = np.exp(escalados)
    return exp_z / exp_z.sum()


def entropia_bits(p):
    """Entropía de Shannon en bits: H = -sum_i p_i * log2(p_i)."""
    p = np.asarray(p, dtype=float)
    mascara = p > 0
    return float(-np.sum(p[mascara] * np.log2(p[mascara])))


# ----------------------------------------------------------------------
# Cálculo de distribuciones y entropías
# ----------------------------------------------------------------------
distribuciones = {}
entropias = {}
sumas = {}
for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)
    sumas[T] = float(p.sum())

# ----------------------------------------------------------------------
# Tabla con cuatro decimales (probabilidades por token + entropía)
# ----------------------------------------------------------------------
tabla_4dec = pd.DataFrame(
    {f"T={T}": [f"{distribuciones[T][i]:.4f}" for i in range(len(tokens))]
     for T in temperaturas},
    index=tokens,
)
tabla_4dec.loc["Entropía (bits)"] = [f"{entropias[T]:.4f}" for T in temperaturas]

# ----------------------------------------------------------------------
# Verificaciones (cada distribución debe sumar 1)
# ----------------------------------------------------------------------
verificacion = {
    f"T={T}": {
        "suma": sumas[T],
        "suma_redondeada_4dec": f"{sumas[T]:.4f}",
        "ok_suma_1": bool(abs(sumas[T] - 1.0) < 1e-12),
    }
    for T in temperaturas
}

# ----------------------------------------------------------------------
# Figura: distribuciones para las tres temperaturas
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(tokens))
ancho = 0.25
colores = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for k, T in enumerate(temperaturas):
    ax.bar(
        x + (k - 1) * ancho,
        distribuciones[T],
        width=ancho,
        color=colores[k],
        label=f"T={T}  (H = {entropias[T]:.4f} bits)",
    )
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad")
ax.set_title("Softmax con temperatura — logits [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1 - Softmax con temperatura y entropía en bits",
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {
        f"T={T}": distribuciones[T].tolist() for T in temperaturas
    },
    "entropias_bits": {f"T={T}": entropias[T] for T in temperaturas},
    "suma_distribuciones": {f"T={T}": sumas[T] for T in temperaturas},
    "verificacion_suma_1": verificacion,
    "tabla_4_decimales": tabla_4dec.to_dict(),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------
print("T1 — Softmax con temperatura (implementación NumPy estable)")
print("Logits:", logits.tolist())
print()
print("Tabla (4 decimales):")
print(tabla_4dec.to_string())
print()
for T in temperaturas:
    print(
        f"T={T}: suma de la distribución = {sumas[T]:.6f} "
        f"(ok={verificacion[f'T={T}']['ok_suma_1']}), "
        f"entropía = {entropias[T]:.4f} bits"
    )
print()
print("Figura guardada: t1_softmax_temperatura.png")
print("Resultados guardados: resultados.json")
