"""T1 - Representar las mediciones.

Carga data/server_measurements.csv (con parse_dates en timestamp), implementa la
clase ServerMeasurements (values matriz NumPy 2D, feature_names tupla, tres
validaciones con ValueError y __repr__ con forma y nombres) y la función
build_measurements(df) que selecciona gpu_utilization, cpu_utilization y
memory_gb en ese orden, las convierte a matriz flotante y retorna el objeto.
Escribe resultados.json con el repr obtenido y el resultado de las pruebas de
las validaciones.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = Path("data") / "server_measurements.csv"
RUTA_RESULTADOS = Path("resultados.json")

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
REPR_ESPERADO = (
    "ServerMeasurements(shape=(300, 3), "
    "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
)


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional y los nombres de las características.

    Lanza ValueError si:
      - la matriz no es bidimensional,
      - el número de columnas no coincide con la cantidad de nombres,
      - los datos no son numéricos.
    """

    def __init__(self, values, feature_names):
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(
                f"Los valores deben ser una matriz bidimensional; se recibió "
                f"ndim={values.ndim}."
            )
        feature_names = tuple(feature_names)
        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas ({values.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(feature_names)})."
            )
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

    @staticmethod
    def build_measurements(df):
        """Selecciona las tres características, las convierte a flotante y retorna el objeto."""
        valores = df[list(FEATURES)].to_numpy(dtype=float)
        return ServerMeasurements(valores, FEATURES)


def build_measurements(df):
    """Función de nivel de módulo; delega en ServerMeasurements.build_measurements."""
    return ServerMeasurements.build_measurements(df)


def _lanza_value_error(funcion):
    """Retorna True si funcion() lanza ValueError, False en otro caso."""
    try:
        funcion()
    except ValueError:
        return True
    except Exception:
        return False
    return False


def main():
    # 1) Carga de datos con parse_dates en timestamp
    df = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
    timestamp_es_fecha = pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    # 2) Construcción del objeto e impresión del repr
    mediciones = build_measurements(df)
    repr_obtenido = repr(mediciones)
    print(repr_obtenido)

    # 3) Pruebas de las tres validaciones (deben lanzar ValueError)
    validaciones = {
        "matriz_1d_lanza_value_error": _lanza_value_error(
            lambda: ServerMeasurements(
                np.zeros(5, dtype=float), ("a", "b", "c", "d", "e")
            )
        ),
        "matriz_3d_lanza_value_error": _lanza_value_error(
            lambda: ServerMeasurements(np.zeros((2, 2, 2)), ("a", "b"))
        ),
        "mas_nombres_que_columnas_lanza_value_error": _lanza_value_error(
            lambda: ServerMeasurements(np.zeros((3, 2)), ("a", "b", "c"))
        ),
        "mas_columnas_que_nombres_lanza_value_error": _lanza_value_error(
            lambda: ServerMeasurements(np.zeros((3, 3)), ("a", "b"))
        ),
        "datos_texto_lanzan_value_error": _lanza_value_error(
            lambda: ServerMeasurements(
                np.array([["x", "y", "z"]]), ("a", "b", "c")
            )
        ),
        "datos_objeto_no_numericos_lanzan_value_error": _lanza_value_error(
            lambda: ServerMeasurements(
                np.array([[1.0, 2.0, "tres"]], dtype=object), ("a", "b", "c")
            )
        ),
    }

    # 4) Verificaciones del criterio de éxito
    repr_coincide = repr_obtenido == REPR_ESPERADO
    valores_flotantes = np.issubdtype(mediciones.values.dtype, np.floating)
    medias = {
        nombre: float(mediciones.values[:, i].mean())
        for i, nombre in enumerate(mediciones.feature_names)
    }

    resultados = {
        "repr": repr_obtenido,
        "repr_esperado": REPR_ESPERADO,
        "repr_coincide": bool(repr_coincide),
        "shape": [int(v) for v in mediciones.values.shape],
        "feature_names": list(mediciones.feature_names),
        "values_dtype": str(mediciones.values.dtype),
        "valores_son_flotantes": bool(valores_flotantes),
        "filas_csv": int(len(df)),
        "columnas_csv": [str(c) for c in df.columns],
        "timestamp_parseado_como_fecha": bool(timestamp_es_fecha),
        "media_por_feature": medias,
        "validaciones_value_error": validaciones,
        "todas_las_validaciones_pasan": bool(all(validaciones.values())),
    }
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # 5) Resumen breve
    print("-" * 60)
    print(f"Filas leídas del CSV: {len(df)}")
    print(f"Timestamp parseado como fecha: {timestamp_es_fecha}")
    print(f"Shape de values: {mediciones.values.shape}")
    print(f"Feature names: {mediciones.feature_names}")
    print(f"Tipo de values: {mediciones.values.dtype}")
    print(f"Repr coincide con el esperado: {repr_coincide}")
    print(f"Validaciones ValueError correctas: {all(validaciones.values())}")
    print(f"Resultados escritos en: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
