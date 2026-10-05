import json
from pathlib import Path

import numpy as np
import pandas as pd


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names).

    Lanza ValueError si:
      - la matriz no es bidimensional,
      - los datos no son numéricos,
      - el número de columnas no coincide con la cantidad de nombres.
    """

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional
        if arr.ndim != 2:
            raise ValueError(
                f"values debe ser una matriz bidimensional; se recibió ndim={arr.ndim}."
            )

        # Validación 2: los datos deben ser numéricos
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"Los datos deben ser numéricos; se recibió dtype={arr.dtype}."
            )

        # Validación 3: número de columnas == cantidad de nombres
        feature_names = tuple(feature_names)
        if arr.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas ({arr.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(feature_names)})."
            )

        self.values = arr.astype(float)
        self.feature_names = feature_names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df):
    """Selecciona gpu_utilization, cpu_utilization y memory_gb (en ese orden),
    las convierte a una matriz NumPy de tipo flotante y retorna un objeto
    ServerMeasurements."""
    feature_names = ("gpu_utilization", "cpu_utilization", "memory_gb")

    faltantes = [c for c in feature_names if c not in df.columns]
    if faltantes:
        raise KeyError(f"Columnas faltantes en el DataFrame: {faltantes}")

    matrix = df.loc[:, list(feature_names)].to_numpy(dtype=float)
    return ServerMeasurements(matrix, feature_names)


def _lanza_valueerror(fn):
    """Retorna True si fn() lanza ValueError, False en caso contrario."""
    try:
        fn()
        return False
    except ValueError:
        return True


def main():
    # 1) Carga de datos con parse_dates en timestamp
    df = pd.read_csv(
        Path("data") / "server_measurements.csv",
        parse_dates=["timestamp"],
        encoding="utf-8-sig",
    )
    # Normalización defensiva de nombres de columna (espacios -> guion bajo)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # 2) Construcción del objeto y obtención del repr (salida exacta exigida)
    measurements = build_measurements(df)
    repr_str = repr(measurements)
    print(repr_str)

    # 3) Verificación de las tres validaciones (deben lanzar ValueError)
    validaciones = {
        "matriz_no_bidimensional_lanza_valueerror": _lanza_valueerror(
            lambda: ServerMeasurements(np.array([1.0, 2.0, 3.0]), ("a", "b", "c"))
        ),
        "columnas_vs_nombres_lanza_valueerror": _lanza_valueerror(
            lambda: ServerMeasurements(np.ones((4, 3)), ("a", "b"))
        ),
        "datos_no_numericos_lanza_valueerror": _lanza_valueerror(
            lambda: ServerMeasurements(
                np.array([["x", "y", "z"], ["p", "q", "r"]]), ("a", "b", "c")
            )
        ),
    }

    # 4) resultados.json (contrato de la subtarea)
    results = {
        "repr": repr_str,
        "forma": [int(measurements.values.shape[0]), int(measurements.values.shape[1])],
        "n_filas": int(measurements.values.shape[0]),
        "n_columnas": int(measurements.values.shape[1]),
        "feature_names": list(measurements.feature_names),
        "dtype_values": str(measurements.values.dtype),
        "validaciones_valueerror": validaciones,
        "todas_las_validaciones_lanzan_valueerror": bool(all(validaciones.values())),
    }

    out_dir = Path("entrada") / "T1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(Path("resultados.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 5) Resumen breve
    print("Resumen T1:")
    print(f"  Forma de values: {measurements.values.shape}")
    print(f"  Características: {measurements.feature_names}")
    print(f"  dtype de values: {measurements.values.dtype}")
    for nombre, ok in validaciones.items():
        estado = "OK (lanza ValueError)" if ok else "FALLO (no lanza ValueError)"
        print(f"  {nombre}: {estado}")
    print(f"  Resultados guardados en {out_dir / 'resultados.json'} y ./resultados.json")


if __name__ == "__main__":
    main()
