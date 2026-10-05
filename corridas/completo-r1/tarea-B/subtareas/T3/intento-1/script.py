# T3 — Comprobación empírica: muestreo de la distribución renormalizada de T2,
# comparación de frecuencias observadas vs probabilidades teóricas y divergencia KL (bits).
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cargar_distribucion_t2():
    """Lee la distribución renormalizada de T2; si no existiera, la recomputa desde los logits."""
    try:
        with open("entrada/T2/resultados.json", "r", encoding="utf-8") as f:
            t2 = json.load(f)
        tokens = t2["distribucion_renormalizada_orden_tokens"]
        p_teo = np.array(t2["distribucion_renormalizada_valores"], dtype=float)
        fuente = "entrada/T2/resultados.json (distribución renormalizada de T2)"
        return tokens, p_teo, fuente
    except (FileNotFoundError, KeyError):
        # Fallback: softmax con T=1 + top-p (p=0.9) sobre los logits del enunciado
        tokens_full = ["t0", "t1", "t2", "t3", "t4", "t5"]
        logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
        z = logits - logits.max()
        p1 = np.exp(z) / np.exp(z).sum()  # T = 1
        orden = np.argsort(-p1)
        acum = np.cumsum(p1[orden])
        k = int(np.searchsorted(acum, 0.9) + 1)  # conjunto mínimo que alcanza 0.9
        nucleo = orden[:k]
        p_nuc = p1[nucleo] / p1[nucleo].sum()
        tokens = [tokens_full[i] for i in nucleo]
        fuente = "recomputada desde logits (softmax T=1 + top-p p=0.9)"
        return tokens, p_nuc, fuente


def main():
    tokens, p_teo, fuente = cargar_distribucion_t2()
    n = 10000

    # Muestreo: 10,000 muestras con la semilla exigida
    rng = np.random.default_rng(0)
    muestras = rng.choice(len(tokens), size=n, p=p_teo)

    counts = np.bincount(muestras, minlength=len(tokens))
    freq_obs = counts / n

    # Divergencia KL (en bits) de la empírica respecto de la teórica:
    # D_KL(P_emp || P_teo) = sum_i p_emp(i) * log2(p_emp(i) / p_teo(i))
    mask = freq_obs > 0
    kl_bits = float(np.sum(freq_obs[mask] * np.log2(freq_obs[mask] / p_teo[mask])))

    # Gráfico de barras: frecuencias observadas vs probabilidades teóricas
    x = np.arange(len(tokens))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, freq_obs, width, color="#4C72B0",
           label="Frecuencia observada (10,000 muestras)")
    ax.bar(x + width / 2, p_teo, width, color="#DD8452",
           label="Probabilidad teórica (top-p renormalizada)")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.set_ylabel("Probabilidad / Frecuencia relativa")
    ax.set_xlabel("Token")
    ax.set_title("T3: Frecuencias observadas vs probabilidades teóricas\n"
                 "top-p (p=0.9) sobre T=1 · 10,000 muestras · default_rng(0)")
    for xi, fo, pt in zip(x, freq_obs, p_teo):
        ax.text(xi - width / 2, fo + 0.008, f"{fo:.4f}", ha="center", fontsize=8)
        ax.text(xi + width / 2, pt + 0.008, f"{pt:.4f}", ha="center", fontsize=8)
    ax.legend()
    ax.set_ylim(0, max(freq_obs.max(), p_teo.max()) * 1.15)
    plt.tight_layout()
    plt.savefig("T3_frecuencias_vs_teoricas.png", dpi=120)
    plt.close(fig)

    # Contrato de resultados
    resultados = {
        "subtarea": "T3_comprobacion_empirica_muestreo_top_p",
        "fuente_distribucion": fuente,
        "semilla": "numpy.random.default_rng(0)",
        "num_muestras": n,
        "tokens": tokens,
        "probabilidades_teoricas": p_teo.tolist(),
        "conteos_observados": counts.tolist(),
        "frecuencias_observadas": freq_obs.tolist(),
        "divergencia_KL_bits": kl_bits,
        "definicion_KL": "D_KL(P_emp || P_teo) = sum p_emp * log2(p_emp / p_teo), en bits",
        "figura": "T3_frecuencias_vs_teoricas.png",
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # Resumen
    print("T3 — Comprobación empírica (top-p p=0.9 sobre T=1)")
    print(f"Fuente de la distribución: {fuente}")
    print(f"Muestras: {n} · Semilla: numpy.random.default_rng(0)")
    print(f"{'token':<6}{'observado':>12}{'teórico':>12}{'conteo':>10}")
    for tok, fo, pt, c in zip(tokens, freq_obs, p_teo, counts):
        print(f"{tok:<6}{fo:>12.4f}{pt:>12.4f}{c:>10d}")
    print(f"Divergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
    print("Figura guardada: T3_frecuencias_vs_teoricas.png")
    print("Resultados guardados: resultados.json")


if __name__ == "__main__":
    main()
