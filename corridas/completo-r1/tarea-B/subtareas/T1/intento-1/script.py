# T1: Softmax con temperatura (NumPy, estable numéricamente) + entropía en bits
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Datos del enunciado
# ------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]

# ------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable (max-shift)
# p_i = exp(z_i / T) / sum_j exp(z_j / T)
# ------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    z = np.asarray(z, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("La temperatura T debe ser > 0.")
    escalados = z / T
    m = np.max(escalados)              # resta del máximo -> estabilidad
    exps = np.exp(escalados - m)
    return exps / np.sum(exps)

def entropia_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                       # términos 0*log2(0) = 0
    return float(-np.sum(p * np.log2(p)))

# ------------------------------------------------------------------
# Cálculo de distribuciones y entropías
# ------------------------------------------------------------------
distribuciones = {}
entropias = {}
for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    distribuciones[f"T={T}"] = p.tolist()
    entropias[f"T={T}"] = entropia_bits(p)

sumas = {k: float(np.sum(v)) for k, v in distribuciones.items()}

# Verificación: T=1 debe coincidir con la softmax estándar de los logits
exps_std = np.exp(logits - np.max(logits))
softmax_estandar = exps_std / np.sum(exps_std)
diff_T1 = float(np.max(np.abs(softmax_estandar - np.array(distribuciones["T=1.0"]))))

# ------------------------------------------------------------------
# Tabla con cuatro decimales
# ------------------------------------------------------------------
tabla = {}
for T in temperaturas:
    key = f"T={T}"
    tabla[key] = {
        "distribucion_4dec": [round(v, 4) for v in distribuciones[key]],
        "entropia_bits_4dec": round(entropias[key], 4),
    }

df = pd.DataFrame(
    {
        "T": [f"T={T}" for T in temperaturas],
        **{t: [round(distribuciones[f"T={T}"][i], 4) for T in temperaturas]
           for i, t in enumerate(tokens)},
        "entropia_bits": [round(entropias[f"T={T}"], 4) for T in temperaturas],
        "suma": [round(sumas[f"T={T}"], 6) for T in temperaturas],
    }
)

# ------------------------------------------------------------------
# Contrato: resultados.json
# ------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": distribuciones,
    "entropias_bits": entropias,
    "tabla_4_decimales": tabla,
    "sumas_distribuciones": sumas,
    "verificacion_T1_vs_softmax_estandar_max_abs_diff": diff_T1,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# Figura: distribuciones según temperatura
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(tokens))
ancho = 0.25
for i, T in enumerate(temperaturas):
    ax.bar(x + (i - 1) * ancho, distribuciones[f"T={T}"],
           width=ancho, label=f"T={T}")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad")
ax.set_title("Softmax con temperatura (logits [2.0, 1.0, 0.5, 0.2, -1.0, -3.0])")
ax.legend()
fig.tight_layout()
plt.savefig("softmax_temperatura.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# Resumen
# ------------------------------------------------------------------
print("=== T1: Softmax con temperatura ===")
print(f"Logits: {logits.tolist()}")
print()
print(df.to_string(index=False))
print()
for T in temperaturas:
    k = f"T={T}"
    print(f"{k}: suma = {sumas[k]:.10f} | H = {entropias[k]:.4f} bits")
print()
print(f"Verificación T=1 vs softmax estándar (max abs diff): {diff_T1:.3e}")
print("Archivos generados: resultados.json, softmax_temperatura.png")
