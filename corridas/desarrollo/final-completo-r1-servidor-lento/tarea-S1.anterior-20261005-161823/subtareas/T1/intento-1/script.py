# -*- coding: utf-8 -*-
"""
T1 - Representar las mediciones.

Carga data/server_measurements.csv (parse_dates en timestamp, sin modificar el CSV),
implementa la clase ServerMeasurements (values matriz NumPy 2D, feature_names tupla,
tres validaciones ValueError y __repr__ con forma y nombres) y build_measurements(df),
que selecciona gpu_utilization, cpu_utilization, memory_gb en ese orden, las convierte
a matriz NumPy flotante y retorna el objeto. Escribe resultados.json con el repr
capturado y la verificación de las tres validaciones.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data/server_measurements.csv")
RESULTS_PATH = Path("resultados.json")
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")


def _es_numerico(arr: np.ndarray) -> bool:
    """True si todos los elementos del arreglo son numéricos."""
    if np.issubdtype(arr.dtype, np.number):
        return True
    if arr.dtype == object:
        return all(
            isinstance(x, (int, float, np.integer, np.floating))
            and not isinstance(x, bool)
            for x in arr.flat
        )
    return False


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names)."""

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional
        if arr.ndim != 2:
            raise ValueError(
                f"values debe ser una matriz bidimensional; se recibió ndim={arr.ndim}."
            )

        feature_names = tuple(feature_names)

        # Validación 2: el número de columnas debe coincidir con la cantidad de nombres
        if arr.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas ({arr.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(feature_names)})."
            )

        # Validación 3: los datos deben ser numéricos
        if not _es_numerico(arr):
            raise ValueError("Los datos no son numéricos.")

        self.values = arr
        self.feature_names = feature_names

    @property
    def shape(self):
        return self.values.shape

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df: pd.DataFrame) -> ServerMeasurements:
    """Selecciona las tres características en el orden indicado, las convierte a
    una matriz NumPy de tipo flotante y retorna un objeto ServerMeasurements."""
    faltantes = [c for c in FEATURES if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el DataFrame: {faltantes}")
    values = df.loc[:, list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def _verifica_validaciones() -> dict:
    """Comprueba que las tres validaciones lanzan ValueError."""
    res = {}

    # 1) matriz no bidimensional
    try:
        ServerMeasurements(np.zeros(6), FEATURES)
        res["no_bidimensional"] = False
    except ValueError:
        res["no_bidimensional"] = True

    # 2) número de columnas != cantidad de nombres
    try:
        ServerMeasurements(np.zeros((4, 2)), FEATURES)
        res["columnas_vs_nombres"] = False
    except ValueError:
        res["columnas_vs_nombres"] = True

    # 3) datos no numéricos
    try:
        ServerMeasurements(
            np.array([["a", "b", "c"]] * 4, dtype=object), FEATURES
        )
        res["datos_no_numericos"] = False
    except ValueError:
        res["datos_no_numericos"] = True

    return res


def main():
    # Carga del CSV sin modificarlo; timestamp se lee como fecha-hora
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

    # Construcción del objeto y captura del repr
    measurements = build_measurements(df)
    repr_str = repr(measurements)
    print("repr ->", repr_str)

    esperado = (
        "ServerMeasurements(shape=(300, 3), "
        "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
    )
    coincide = repr_str == esperado

    # Verificación de las tres validaciones de ValueError
    validaciones = _verifica_validaciones()

    resultados = {
        "subtarea": "T1_representar_las_mediciones",
        "repr_capturado": repr_str,
        "repr_esperado": esperado,
        "repr_coincide": bool(coincide),
        "shape": [int(measurements.values.shape[0]), int(measurements.values.shape[1])],
        "feature_names": list(measurements.feature_names),
        "values_dtype": str(measurements.values.dtype),
        "n_filas_csv": int(len(df)),
        "columnas_csv": list(df.columns),
        "dtypes_csv": {c: str(t) for c, t in df.dtypes.items()},
        "timestamp_es_datetime": bool(
            np.issubdtype(df["timestamp"].dtype, np.datetime64)
        ),
        "validaciones_valueerror_implementadas": validaciones,
        "todas_las_validaciones_ok": bool(all(validaciones.values())),
        "primeras_5_filas_values": measurements.values[:5].tolist(),
    }

    RESULTS_PATH.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Validaciones ValueError:", validaciones)
    print("¿repr coincide con el esperado?", coincide)
    print(f"Resultados escritos en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
