# -*- coding: utf-8 -*-
"""T1 - Representar las mediciones.

Carga data/server_measurements.csv (parse_dates en timestamp), implementa la
clase ServerMeasurements (valida bidimensionalidad, coincidencia
columnas/nombres y tipo numérico, con __repr__ de forma y nombres) y la
función build_measurements(df) que selecciona gpu_utilization,
cpu_utilization, memory_gb en ese orden y retorna el objeto con matriz NumPy
flotante. Imprime el repr y escribe resultados.json.

Nota: output/server_analysis.parquet corresponde a la subtarea T4 (resumen
por servidor); esta subtarea T1 no lo produce, por lo que no se genera aquí.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
EXPECTED_REPR = (
    "ServerMeasurements(shape=(300, 3), features=('gpu_utilization', "
    "'cpu_utilization', 'memory_gb'))"
)


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names)."""

    def __init__(self, values, feature_names):
        values = np.asarray(values)
        # Validación 1: la matriz debe ser bidimensional
        if values.ndim != 2:
            raise ValueError(
                "Los valores deben formar una matriz bidimensional; "
                f"se recibió ndim={values.ndim}."
            )
        feature_names = tuple(feature_names)
        # Validación 2: número de columnas == cantidad de nombres
        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas de la matriz ({values.shape[1]}) no "
                f"coincide con la cantidad de nombres de características "
                f"({len(feature_names)})."
            )
        # Validación 3: los datos deben ser numéricos
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(
                f"Los datos deben ser numéricos; se recibió dtype={values.dtype}."
            )
        self.values = values
        self.feature_names = feature_names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df):
    """Selecciona las tres características en el orden exigido, las convierte
    a una matriz NumPy de tipo flotante y retorna un ServerMeasurements."""
    selected = df[list(FEATURES)]
    values = selected.to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def check_value_error(label, results, fn):
    """Registra True si fn lanza ValueError, False en caso contrario."""
    try:
        fn()
        results[label] = False
    except ValueError:
        results[label] = True


def main():
    # 1) Carga de datos con parse_dates en timestamp
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])

    # 2) Construcción del objeto de mediciones e impresión del repr
    measurements = build_measurements(df)
    print(repr(measurements))

    # 3) Verificación de las validaciones: cada caso inválido debe lanzar ValueError
    validations = {}

    def caso_no_bidimensional():
        ServerMeasurements(np.zeros(10), FEATURES)  # matriz 1-D

    def caso_columnas_vs_nombres():
        ServerMeasurements(np.zeros((10, 2)), FEATURES)  # 2 columnas vs 3 nombres

    def caso_no_numerico():
        ServerMeasurements(np.array([["a", "b", "c"]], dtype=object), FEATURES)

    check_value_error(
        "no_bidimensional_lanza_valueerror", validations, caso_no_bidimensional
    )
    check_value_error(
        "columnas_vs_nombres_lanza_valueerror", validations, caso_columnas_vs_nombres
    )
    check_value_error(
        "no_numerico_lanza_valueerror", validations, caso_no_numerico
    )

    # 4) Estadísticas descriptivas por característica (informativas)
    stats = {}
    for i, name in enumerate(measurements.feature_names):
        col = measurements.values[:, i]
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    # 5) Contrato de resultados
    repr_str = repr(measurements)
    results = {
        "repr": repr_str,
        "repr_coincide_con_esperado": bool(repr_str == EXPECTED_REPR),
        "shape": [
            int(measurements.values.shape[0]),
            int(measurements.values.shape[1]),
        ],
        "feature_names": list(measurements.feature_names),
        "dtype_values": str(measurements.values.dtype),
        "timestamp_dtype": str(df["timestamp"].dtype),
        "n_filas_csv": int(len(df)),
        "validaciones": validations,
        "estadisticas_por_caracteristica": stats,
    }
    Path("resultados.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6) Resumen breve
    print("Resumen T1:")
    print(f"  Filas cargadas: {len(df)}")
    print(f"  Objeto: {repr_str}")
    print(f"  dtype de values: {measurements.values.dtype}")
    print(f"  dtype de timestamp: {df['timestamp'].dtype}")
    print(f"  Validaciones (True = ValueError lanzado): {validations}")
    print("  resultados.json escrito.")


if __name__ == "__main__":
    main()
