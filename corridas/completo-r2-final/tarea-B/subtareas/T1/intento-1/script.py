# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Softmax con temperatura (NumPy, numéricamente estable) y entropía en bits.
Tarea B — Temperatura, top-p y entropía de la distribución de salida (MMIA 6013).

Salidas:
  - resultados.json            (contrato: distribuciones, entropías, tabla a 4 decimales)
  - t1_softmax_temperatura.png (figura: distribuciones y entropía vs. temperatura)
"""

import json

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
etiquetas = ["T=0.5", "T=1", "T=2"]


# ----------------------------------------------------------------------
# 2. Softmax con temperatura, numéricamente estable
# ----------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    """
    p_i = exp(z_i / T) / sum_j exp(z_j / T), implementación estable.

    Restar una constante m a todos los logits no cambia la softmax:
        exp((z_i - m)/T) / sum_j exp((z_j - m)/T) == exp(z_i/T) / sum_j exp(z_j/T)
    Tomando m = max(z) se evita el overflow de exp para logits grandes.
    """
    z = np.asarray(z, dtype=float)
    m = np.max(z)                      # constante de estabilidad numérica
    e = np.exp((z - m) / T)
    return e / e.sum()


def entropia_bits(p):
    """Entropía de Shannon en bits: H(p) = -sum_i p_i * log2(p_i), con 0*log2(0)=0."""
    p = np.asarray(p, dtype=float)
    p_pos = p[p > 0]
    return float(-np.sum(p_pos * np.log2(p_pos)))


# ----------------------------------------------------------------------
# 3. Cálculo de las tres distribuciones y sus entropías
# ----------------------------------------------------------------------
distribuciones = {lab: softmax_con_temperatura(logits, T)
                  for lab, T in zip(etiquetas, temperaturas)}
entropias = {lab: entropia_bits(distribuciones[lab]) for lab in etiquetas}
sumas = {lab: float(distribuciones[lab].sum()) for lab in etiquetas}

# ----------------------------------------------------------------------
# 4. Tabla con cuatro decimales (distribuciones + entropías)
# ----------------------------------------------------------------------
def f4(v):
    return f"{v:.4f}"

tabla = pd.DataFrame(
    {lab: [f4(v) for v in distribuciones[lab]] for lab in etiquetas},
    index=tokens,
)
tabla.loc["entropia_bits"] = [f4(entropias[lab]) for lab in etiquetas]

# ----------------------------------------------------------------------
# 5. Figura PNG
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

x = np.arange(len(tokens))
ancho = 0.26
colores = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for k, lab in enumerate(etiquetas):
    ax1.bar(x + (k - 1) * ancho, distribuciones[lab], width=ancho,
            label=f"{lab} (H = {entropias[lab]:.4f} bits)", color=colores[k])
ax1.set_xticks(x)
ax1.set_xticklabels(tokens)
ax1.set_xlabel("Token")
ax1.set_ylabel("Probabilidad")
ax1.set_title("Softmax con temperatura")
ax1.legend(fontsize=8)

ax2.plot(temperaturas, [entropias[lab] for lab in etiquetas], "o-", color="#d62728")
for T, lab in zip(temperaturas, etiquetas):
    ax2.annotate(f"{entropias[lab]:.4f}", (T, entropias[lab]),
                 textcoords="offset points", xytext=(6, 4), fontsize=9)
ax2.set_xticks(temperaturas)
ax2.set_xlabel("Temperatura T")
ax2.set_ylabel("Entropía (bits)")
ax2.set_title("Entropía vs. temperatura")
ax2.grid(alpha=0.3)

fig.tight_layout()
plt.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 6. resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1_softmax_con_temperatura",
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {lab: distribuciones[lab].tolist() for lab in etiquetas},
    "entropias_bits": {lab: entropias[lab] for lab in etiquetas},
    "suma_de_cada_distribucion": sumas,
    "tabla_cuatro_decimales": {
        "columnas": ["token"] + etiquetas,
        "filas": (
            [[tok] + [tabla.loc[tok, lab] for lab in etiquetas] for tok in tokens]
            + [["entropia_bits"] + [tabla.loc["entropia_bits", lab] for lab in etiquetas]]
        ),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Resumen
# ----------------------------------------------------------------------
print("SUBTAREA T1 — Softmax con temperatura y entropía en bits")
print(f"Logits: {logits.tolist()}")
print()
print(tabla.to_string())
print()
for lab in etiquetas:
    print(f"{lab}: suma de probabilidades = {sumas[lab]:.12f} | "
          f"entropía = {entropias[lab]:.4f} bits")
print()
print("Archivos escritos: resultados.json, t1_softmax_temperatura.png")
