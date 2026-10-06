#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Construir el resumen por servidor.

Agrupa el DataFrame enriquecido de T3 (con load_score y requires_review) por la
columna `server` y construye una tabla con una fila por servidor y EXACTAMENTE
estas columnas, en este orden:
    server, observations, mean_power_w, max_temperature_c, mean_load, review_count

- observations:      número de filas del servidor.
- mean_power_w:      media de power_w.
- max_temperature_c: máximo de temperature_c.
- mean_load:         media de load_score.
- review_count:      conteo de filas con requires_review=True.

Entradas:
  - entrada/T3/ (resultados.json y, si existen, archivos con el DataFrame enriquecido)
  - data/server_measurements.csv (fuente original, para recalcular si hace falta,
    usando los parámetros exactos de T2/T3: pesos (0.50, 0.30, 0.20), ddof=0)

Salidas:
  - resultados.json (contrato de la subtarea, con todas las cifras)
  - output/server_analysis.parquet (tabla resumen, ruta exigida por el enunciado)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

T3_DIR = Path("entrada/T3")
DATA_CSV = Path("data/server_measurements.csv")
OUT_PARQUET = Path("output/server_analysis.parquet")

FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
WEIGHTS = np.array([0.50, 0.30, 0.20])
TH_LOAD = 1.5
TH_TEMP = 80.0
EXPECTED_COLS = [
    "server",
    "observations",
    "mean_power_w",
    "max_temperature_c",
    "mean_load",
    "review_count",
]
EXPECTED_SRV01 = {
    "observations": 50,
    "mean_power_w": 363.4598,
    "max_temperature_c": 86.50,
    "mean_load": -0.614527,
    "review_count": 2,
}
REQUIRED_T3_COLS = {"server", "power_w", "temperature_c", "load_score", "requires_review"}


def load_t3_params():
    """Lee los parámetros/resultados de T3 si están disponibles."""
    pj = T3_DIR / "resultados.json"
    if pj.exists():
        try:
            return json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def try_load_t3_frame(params):
    """Intenta cargar el DataFrame enriquecido de T3 (parquet o csv) desde entrada/T3/."""
    if not T3_DIR.exists():
        return None
    candidates = sorted(T3_DIR.glob("*.parquet")) + sorted(T3_DIR.glob("*.csv"))
    for p in candidates:
        try:
            d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        except Exception:
            continue
        if not REQUIRED_T3_COLS.issubset(d.columns):
            continue
        if len(d) != 300:
            continue
        if d["requires_review"].dtype != bool:
            d["requires_review"] = (
                d["requires_review"].astype(str).str.strip().str.lower()
                .isin(["true", "1", "1.0", "yes"])
            )
        d["load_score"] = pd.to_numeric(d["load_score"], errors="coerce")
        if d["load_score"].isna().any():
            continue
        expected_rev = params.get("n_requires_review") if isinstance(params, dict) else None
        if expected_rev is not None and int(d["requires_review"].sum()) != int(expected_rev):
            continue
        return d
    return None


def recompute_frame(params):
    """Recalcula load_score y requires_review desde el CSV original con los
    parámetros exactos usados en T2/T3 (z-scores con ddof=0, pesos 0.50/0.30/0.20)."""
    d = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    if isinstance(params, dict) and "means_used" in params and "stds_used" in params:
        mu = np.asarray(params["means_used"], dtype=float)
        sd = np.asarray(params["stds_used"], dtype=float)
    else:
        mu = d[FEATURES].mean().to_numpy(dtype=float)
        sd = d[FEATURES].std(ddof=0).to_numpy(dtype=float)
    Z = (d[FEATURES].to_numpy(dtype=float) - mu) / sd
    d["load_score"] = Z @ WEIGHTS
    d["requires_review"] = (d["load_score"] > TH_LOAD) | (d["temperature_c"] > TH_TEMP)
    return d


def to_native(obj):
    """Convierte tipos de numpy a tipos nativos de Python para JSON."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main():
    params = load_t3_params()
    df = try_load_t3_frame(params)
    if df is not None:
        source = "entrada/T3 (DataFrame enriquecido de la subtarea T3)"
    else:
        df = recompute_frame(params)
        source = "recalculado desde data/server_measurements.csv (medias/stds de T3, ddof=0)"

    df["requires_review"] = df["requires_review"].astype(bool)
    df["load_score"] = df["load_score"].astype(float)

    # ---- Agrupación por servidor y tabla resumen con orden exacto de columnas ----
    summary = (
        df.groupby("server", as_index=False, sort=True)
        .agg(
            observations=("server", "size"),
            mean_power_w=("power_w", "mean"),
            max_temperature_c=("temperature_c", "max"),
            mean_load=("load_score", "mean"),
            review_count=("requires_review", "sum"),
        )[EXPECTED_COLS]
    )
    summary["observations"] = summary["observations"].astype(int)
    summary["review_count"] = summary["review_count"].astype(int)

    # ---- Impresión de la tabla completa (valores redondeados solo para lectura) ----
    disp = summary.copy()
    disp["mean_power_w"] = disp["mean_power_w"].round(4)
    disp["max_temperature_c"] = disp["max_temperature_c"].round(2)
    disp["mean_load"] = disp["mean_load"].round(6)
    print("Tabla resumen por servidor:")
    print(disp.to_string(index=False))

    # ---- Verificación contra el resultado esperado de AI-SRV-01 ----
    rows01 = summary.loc[summary["server"] == "AI-SRV-01"]
    if len(rows01) == 1:
        r = rows01.iloc[0]
        checks = {
            "observations": int(r["observations"]) == EXPECTED_SRV01["observations"],
            "mean_power_w": round(float(r["mean_power_w"]), 4) == EXPECTED_SRV01["mean_power_w"],
            "max_temperature_c": round(float(r["max_temperature_c"]), 2) == EXPECTED_SRV01["max_temperature_c"],
            "mean_load": round(float(r["mean_load"]), 6) == EXPECTED_SRV01["mean_load"],
            "review_count": int(r["review_count"]) == EXPECTED_SRV01["review_count"],
        }
    else:
        checks = {k: False for k in EXPECTED_SRV01}

    t3_total = params.get("n_requires_review") if isinstance(params, dict) else None
    total_rev = int(summary["review_count"].sum())

    results = {
        "task": "T4_resumen_por_servidor",
        "data_source": str(DATA_CSV),
        "t3_source": source,
        "groupby_column": "server",
        "review_rule": "load_score > 1.5 or temperature_c > 80",
        "column_order": EXPECTED_COLS,
        "n_rows": int(summary.shape[0]),
        "n_cols": int(summary.shape[1]),
        "n_servers": int(summary["server"].nunique()),
        "dtypes": {c: str(t) for c, t in summary.dtypes.items()},
        "table": summary.to_dict(orient="records"),
        "table_by_server": {
            rec["server"]: {k: to_native(v) for k, v in rec.items() if k != "server"}
            for rec in summary.to_dict(orient="records")
        },
        "total_observations": int(summary["observations"].sum()),
        "total_review_count": total_rev,
        "t3_n_requires_review": to_native(t3_total),
        "review_total_consistent_with_T3": (t3_total is None) or (int(t3_total) == total_rev),
        "expected_AI_SRV_01": EXPECTED_SRV01,
        "checks_AI_SRV_01": checks,
        "shape_and_order_ok": bool(
            summary.shape == (6, 6) and list(summary.columns) == EXPECTED_COLS
        ),
        "all_checks_pass": bool(all(checks.values()))
        and bool(summary.shape == (6, 6) and list(summary.columns) == EXPECTED_COLS),
        "parquet_path": str(OUT_PARQUET),
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(to_native(results), f, indent=2, ensure_ascii=False)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUT_PARQUET, index=False)

    # ---- Resumen breve ----
    print("\nResumen T4:")
    print(f"  Filas x columnas: {summary.shape[0]} x {summary.shape[1]} | Columnas: {list(summary.columns)}")
    print(f"  Observaciones totales: {int(summary['observations'].sum())} | Revisiones totales: {total_rev}")
    print(f"  Verificación AI-SRV-01: {'OK' if all(checks.values()) else 'FALLO'} -> {checks}")
    print(f"  Fuente: {source}")
    print("  Archivos escritos: resultados.json, output/server_analysis.parquet")


if __name__ == "__main__":
    main()
