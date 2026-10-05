#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T=1 (salida de T1).

Pasos:
1) Carga la distribución de T=1 desde entrada/T1/resultados.json (contrato de T1).
2) Ordena los tokens por probabilidad descendente.
3) Conserva el conjunto MÍNIMO cuya probabilidad acumulada alcanza p = 0.9.
4) Renormaliza las probabilidades de los tokens supervivientes (suma = 1.0).
5) Escribe resultados.json y guarda la figura T2_top_p_nucleo.png.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 0.9
RUTA_T1 = os.path.join("entrada", "T1", "resultados.json")
ARCHIVO_SALIDA = "resultados.json"
ARCHIVO_FIGURA = "T2_top_p_nucleo.png"


def softmax_con_temperatura(logits, T):
    """Softmax numéricamente estable con temperatura (respaldo si no está la salida de T1)."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def cargar_distribucion_T1():
    """Lee la distribución de T=1 de la salida de T1; si no existe, la recalcula."""
    if os.path.exists(RUTA_T1):
        with open(RUTA_T1, "r", encoding="utf-8") as f:
            res_t1 = json.load(f)
        bloque = res_t1.get("softmax_con_temperatura", {})
        if "T=1.0" in bloque:
            dist = bloque["T=1.0"]["distribucion"]
        elif "T=1" in bloque:
            dist = bloque["T=1"]["distribucion"]
        else:
            raise KeyError("No se encontró la distribución de T=1 en entrada/T1/resultados.json")
        tokens = sorted(dist.keys())  # t0..t5 en orden canónico
        probs = np.array([dist[t] for t in tokens], dtype=float)
        return tokens, probs, "entrada/T1/resultados.json (salida de T1)"
    # Respaldo: recalcular con la softmax con temperatura de la Parte 1
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
    return tokens, softmax_con_temperatura(logits, 1.0), "recalculada con softmax estable (T=1)"


def main():
    # --- 1) Distribución de T=1 proveniente de T1 -----------------------------
    tokens, probs, origen = cargar_distribucion_T1()

    # --- 2) Orden descendente por probabilidad --------------------------------
    orden = np.argsort(-probs, kind="stable")
    tokens_ord = [tokens[i] for i in orden]
    probs_ord = probs[orden]
    acumulada = np.cumsum(probs_ord)

    # --- 3) Conjunto mínimo con probabilidad acumulada >= p -------------------
    if acumulada[-1] < P:  # salvaguarda numérica: si nada alcanza p, se queda todo
        idx = len(acumulada) - 1
    else:
        # primer índice i con acumulada[i] >= p; el núcleo son los índices 0..i
        idx = int(np.searchsorted(acumulada, P, side="left"))
    k = idx + 1

    tokens_nucleo = tokens_ord[:k]
    probs_nucleo = probs_ord[:k]
    acumulada_nucleo = float(acumulada[idx])

    # --- 4) Renormalización ----------------------------------------------------
    q_nucleo = probs_nucleo / probs_nucleo.sum()
    suma_renorm = float(q_nucleo.sum())
    dist_renorm = {t: float(q) for t, q in zip(tokens_nucleo, q_nucleo)}
    tokens_descartados = tokens_ord[k:]

    # Entropía en bits de la distribución renormalizada (informativo)
    mask = q_nucleo > 0
    entropia_bits_renorm = float(-np.sum(q_nucleo[mask] * np.log2(q_nucleo[mask])))

    # Tabla completa (orden descendente, valores sin redondear)
    filas = []
    for i, t in enumerate(tokens_ord):
        filas.append({
            "token": t,
            "prob_T1": float(probs_ord[i]),
            "prob_acumulada": float(acumulada[i]),
            "en_nucleo": bool(i < k),
            "prob_renormalizada": float(q_nucleo[i]) if i < k else 0.0,
        })

    # Tabla de presentación con 4 decimales
    tabla_4d = {
        "token": tokens_ord,
        "prob_T1": [f"{v:.4f}" for v in probs_ord],
        "prob_acumulada": [f"{v:.4f}" for v in acumulada],
        "en_nucleo": ["si" if i < k else "no" for i in range(len(tokens_ord))],
        "prob_renormalizada": [f"{q_nucleo[i]:.4f}" if i < k else "-" for i in range(len(tokens_ord))],
    }

    # --- 5) resultados.json (contrato de la subtarea) --------------------------
    resultados = {
        "subtarea": "T2 - Muestreo de nucleo top-p (p=0.9) sobre la distribucion de T=1",
        "p": P,
        "fuente_distribucion_T1": origen,
        "distribucion_T1": {t: float(p) for t, p in zip(tokens, probs)},
        "orden_descendente": tokens_ord,
        "probabilidades_ordenadas": {t: float(p) for t, p in zip(tokens_ord, probs_ord)},
        "probabilidad_acumulada_ordenada": {t: float(a) for t, a in zip(tokens_ord, acumulada)},
        "tokens_supervivientes": tokens_nucleo,
        "num_tokens_supervivientes": k,
        "probabilidad_acumulada_nucleo": acumulada_nucleo,
        "distribucion_renormalizada": dist_renorm,
        "suma_distribucion_renormalizada": suma_renorm,
        "tokens_descartados": tokens_descartados,
        "entropia_bits_renormalizada": entropia_bits_renorm,
        "tabla_completa": filas,
        "tabla_4_decimales": tabla_4d,
    }
    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # --- Figura -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    colores = ["#2a9d8f" if i < k else "#e76f51" for i in range(len(tokens_ord))]
    ax.bar(tokens_ord, probs_ord, color=colores, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("P(token | T=1)")
    ax.set_ylim(0, max(probs_ord) * 1.15)
    ax2 = ax.twinx()
    ax2.plot(tokens_ord, acumulada, "o-", color="#264653", label="acumulada")
    ax2.axhline(P, color="gray", linestyle="--", linewidth=1, label=f"p = {P}")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Probabilidad acumulada")
    ax2.legend(loc="lower right", fontsize=8)
    ax.set_title("Distribución T=1 ordenada + acumulada\n(verde = núcleo top-p, rojo = descartado)")

    ax = axes[1]
    ax.bar(tokens_nucleo, q_nucleo, color="#2a9d8f", edgecolor="black", linewidth=0.5)
    for i, q in enumerate(q_nucleo):
        ax.text(i, q + 0.01, f"{q:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(q_nucleo) * 1.18)
    ax.set_ylabel("Probabilidad renormalizada")
    ax.set_title(f"Distribución renormalizada del núcleo\n{tokens_nucleo} (suma = {suma_renorm:.6f})")

    fig.tight_layout()
    fig.savefig(ARCHIVO_FIGURA, dpi=120)
    plt.close(fig)

    # --- Resumen ------------------------------------------------------------------
    print("=" * 66)
    print("T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T=1")
    print("=" * 66)
    print(f"Fuente de la distribución T=1: {origen}")
    print("Distribución T=1 (orden descendente):")
    for t, p_, a in zip(tokens_ord, probs_ord, acumulada):
        marca = "  <- núcleo" if t in tokens_nucleo else ""
        print(f"  {t}: p = {p_:.6f}   acumulada = {a:.6f}{marca}")
    print(f"\nTokens supervivientes (conjunto mínimo con acumulada >= {P}): {tokens_nucleo}")
    print(f"Probabilidad acumulada del núcleo: {acumulada_nucleo:.6f}")
    print("Distribución renormalizada:")
    for t, q in dist_renorm.items():
        print(f"  {t}: {q:.6f}")
    print(f"Suma de la distribución renormalizada: {suma_renorm:.6f}")
    print(f"Tokens descartados: {tokens_descartados}")
    print(f"Entropía (bits) de la distribución renormalizada: {entropia_bits_renorm:.6f}")
    print(f"\nArchivos escritos: {ARCHIVO_SALIDA}, {ARCHIVO_FIGURA}")


if __name__ == "__main__":
    main()
