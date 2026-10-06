"""T1: Cargar data/server_measurements.csv e implementar ServerMeasurements.

- Clase ServerMeasurements: almacena values (matriz NumPy 2D) y feature_names (tupla).
  Lanza ValueError si: la matriz no es bidimensional, el número de columnas no
  coincide con la cantidad de nombres, o los datos no son numéricos.
  __repr__ muestra forma y nombres.
- build_measurements(df): selecciona gpu_utilization, cpu_utilization, memory_gb
  (en ese orden), las convierte a matriz NumPy flotante y retorna el objeto.
"""

import json

import numpy as np
import pandas as pd

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
EXPECTED_REPR = (
    "ServerMeasurements(shape=(300, 3), "
    "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
)


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names)."""

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional.
        if arr.ndim != 2:
            raise ValueError(
                f"La matriz debe ser bidimensional; se recibió ndim={arr.ndim}."
            )

        # Validación 2: el número de columnas debe coincidir con los nombres.
        feature_names = tuple(feature_names)
        if arr.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas ({arr.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(feature_names)})."
            )

        # Validación 3: los datos deben ser numéricos.
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"Los datos deben ser numéricos; se recibió dtype={arr.dtype!r}."
            )

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


def build_measurements(df):
    """Selecciona las tres características, las convierte a una matriz NumPy
    de tipo flotante y retorna un objeto ServerMeasurements."""
    values = df.loc[:, list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def _raises_value_error(fn):
    """Retorna True si fn() lanza ValueError."""
    try:
        fn()
        return False
    except ValueError:
        return True


def main():
    # Carga de datos con parse_dates en timestamp.
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])

    # Construcción del objeto y repr esperado.
    measurements = build_measurements(df)
    repr_str = repr(measurements)
    print(repr_str)

    # Casos de prueba: las tres validaciones deben lanzar ValueError.
    test_not_2d = _raises_value_error(
        lambda: ServerMeasurements(np.array([1.0, 2.0, 3.0]), ("a", "b", "c"))
    )
    test_col_mismatch = _raises_value_error(
        lambda: ServerMeasurements(np.zeros((4, 2)), ("a", "b", "c"))
    )
    test_non_numeric = _raises_value_error(
        lambda: ServerMeasurements(
            np.array([["x", "y", "z"], ["1", "2", "3"]]), ("a", "b", "c")
        )
    )

    results = {
        "repr": repr_str,
        "expected_repr": EXPECTED_REPR,
        "repr_matches_expected": repr_str == EXPECTED_REPR,
        "shape": [int(measurements.shape[0]), int(measurements.shape[1])],
        "feature_names": list(measurements.feature_names),
        "values_dtype": str(measurements.values.dtype),
        "n_rows": int(measurements.shape[0]),
        "n_columns": int(measurements.shape[1]),
        "timestamp_dtype": str(df["timestamp"].dtype),
        "first_row_values": [float(v) for v in measurements.values[0]],
        "validation_tests": {
            "not_2d_raises_valueerror": bool(test_not_2d),
            "column_mismatch_raises_valueerror": bool(test_col_mismatch),
            "non_numeric_raises_valueerror": bool(test_non_numeric),
        },
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Resumen breve.
    print(f"Filas: {results['n_rows']} | Columnas: {results['n_columns']}")
    print(f"Dtype de values: {results['values_dtype']}")
    print(f"timestamp parseado como: {results['timestamp_dtype']}")
    print(
        "ValueError en pruebas (no-2D / columnas / no numérico): "
        f"{test_not_2d} / {test_col_mismatch} / {test_non_numeric}"
    )
    print(f"repr coincide con el esperado: {results['repr_matches_expected']}")


if __name__ == "__main__":
    main()
