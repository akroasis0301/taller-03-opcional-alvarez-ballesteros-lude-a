# =====================================================================
# SUBTAREA T1: Softmax con temperatura (NumPy, estable) + entropía en bits
# Tarea B — MMIA 6013 · Taller 03 v2
# =====================================================================
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Datos del enunciado
# ---------------------------------------------------------------------
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
TEMPERATURAS = [0.5, 1.0, 2.0]

# ---------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable (max-subtraction):
# p_i = exp(z_i/T) / sum_j exp(z_j/T), restando el máximo de los logits
# ---------------------------------------------------------------------
def softmax_con_temperatura(logits, T):
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - np.max(z)          # estabilidad: evita overflow de exp
    e = np.exp(z)
    return e / np.sum(e)

def entropia_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]               # evita log2(0)
    return float(-np.sum(p * np.log2(p)))

# ---------------------------------------------------------------------
# Cálculo de distribuciones, entropías y verificación de suma = 1
# ---------------------------------------------------------------------
distribuciones, entropias, sumas = {}, {}, {}
for T in TEMPERATURAS:
    p = softmax_con_temperatura(LOGITS, T)
    key = f"T={T}"
    distribuciones[key] = p
    entropias[key] = entropia_bits(p)
    sumas[key] = float(np.sum(p))

# Comprobación de estabilidad: desplazar los logits +1000 no cambia nada
estabilidad_ok = bool(np.allclose(
    softmax_con_temperatura(LOGITS + 1000.0, 1.0),
    distribuciones["T=1.0"], rtol=0, atol=1e-12))

# ---------------------------------------------------------------------
# Tabla con cuatro decimales (filas: tokens y entropía; columnas: T)
# ---------------------------------------------------------------------
datos = {}
for T in TEMPERATURAS:
    key = f"T={T}"
    datos[key] = [f"{v:.4f}" for v in distribuciones[key]] + [f"{entropias[key]:.4f}"]
tabla = pd.DataFrame(datos, index=TOKENS + ["Entropía (bits)"])

print("Tabla (4 decimales):")
print(tabla.to_string())

# ---------------------------------------------------------------------
# Figura: distribuciones para las tres temperaturas
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(TOKENS))
ancho = 0.25
for i, T in enumerate(TEMPERATURAS):
    key = f"T={T}"
    ax.bar(x + (i - 1) * ancho, distribuciones[key], width=ancho,
           label=f"T={T}  (H = {entropias[key]:.4f} bits)")
ax.set_xticks(x)
ax.set_xticklabels(TOKENS)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad $p_i$")
ax.set_title("Softmax con temperatura — logits (2.0, 1.0, 0.5, 0.2, -1.0, -3.0)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------
tabla_json = [
    {"fila": fila, **{col: tabla.loc[fila, col] for col in tabla.columns}}
    for fila in tabla.index
]

resultados = {
    "subtarea": "T1_softmax_con_temperatura",
    "tokens": TOKENS,
    "logits": {tok: float(z) for tok, z in zip(TOKENS, LOGITS)},
    "temperaturas": TEMPERATURAS,
    "distribuciones": {k: v.tolist() for k, v in distribuciones.items()},
    "suma_distribuciones": sumas,
    "entropias_bits": entropias,
    "softmax_estable_max_subtraction": estabilidad_ok,
    "tabla_4_decimales": tabla_json,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------
# Resumen breve
# ---------------------------------------------------------------------
print("\nResumen:")
for T in TEMPERATURAS:
    key = f"T={T}"
    print(f"  {key}: suma={sumas[key]:.6f} | H = {entropias[key]:.4f} bits")
print(f"  Estabilidad (logits +1000 sin cambio): {estabilidad_ok}")
print("Archivos generados: resultados.json, t1_softmax_temperatura.png")
