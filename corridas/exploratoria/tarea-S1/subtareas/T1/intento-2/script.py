# -*- coding: utf-8 -*-
"""
Subtarea T1 - Representar las mediciones.

Implementa:
  * ServerMeasurements: clase que almacena una matriz NumPy bidimensional
    (values) y una tupla con los nombres de las características
    (feature_names). Lanza ValueError si la matriz no es bidimensional, si
    el número de columnas no coincide con la cantidad de nombres o si los
    datos no son numéricos. __repr__ muestra forma y nombres.
  * build_measurements(df): selecciona gpu_utilization, cpu_utilization y
    memory_gb (en ese orden), las convierte a una matriz NumPy flotante y
    retorna un objeto ServerMeasurements.

Se ejecuta sobre data/server_measurements.csv, imprime el repr esperado,
verifica que las tres validaciones lanzan ValueError en casos de prueba y
escribe resultados.json.
"""

import json

import numpy as np
import pandas as pd

DATA_PATH = "data/server_measurements.csv"
RESULTS_PATH = "resultados.json"
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
EXPECTED_REPR = (
    "ServerMeasurements(shape=(300, 3), "
    "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
)


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional y los nombres de las características.

    Lanza ValueError si:
      * los datos no son numéricos,
      * la matriz no es bidimensional,
      * el número de columnas no coincide con la cantidad de nombres.
    """

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: los datos deben ser numéricos.
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"Los datos deben ser numéricos; se recibió dtype={arr.dtype}."
            )

        # Validación 2: la matriz debe ser bidimensional.
        if arr.ndim != 2:
            raise ValueError(
                f"La matriz debe ser bidimensional (ndim=2); "
                f"se recibió ndim={arr.ndim}."
            )

        # Validación 3: columnas y nombres deben coincidir.
        names = tuple(feature_names)
        if arr.shape[1] != len(names):
            raise ValueError(
                f"El número de columnas ({arr.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(names)})."
            )

        self.values = arr.astype(float)
        self.feature_names = names

    @property
    def shape(self):
        """Forma de la matriz de valores."""
        return self.values.shape

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df):
    """Selecciona las tres características (en el orden exigido), las convierte
    a una matriz NumPy flotante y retorna un objeto ServerMeasurements."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}")
    values = df[list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def run_validation_tests():
    """Verifica que las tres validaciones lanzan ValueError en casos de prueba
    y que un caso válido no lanza error. Devuelve una lista de resultados."""
    tests = []

    def run_case(name, fn, expect_error):
        try:
            fn()
            raised, message = False, ""
        except ValueError as exc:
            raised, message = True, str(exc)
        except Exception as exc:  # cualquier otra excepción cuenta como fallo
            raised, message = False, f"{type(exc).__name__}: {exc}"
        tests.append(
            {
                "test": name,
                "espera_valueerror": bool(expect_error),
                "lanza_valueerror": bool(raised),
                "passed": bool(raised == expect_error),
                "mensaje": message,
            }
        )

    # Validación 1: matriz no bidimensional
    run_case(
        "no_bidimensional_ndim1",
        lambda: ServerMeasurements(np.array([1.0, 2.0, 3.0]), ("a", "b", "c")),
        expect_error=True,
    )
    run_case(
        "no_bidimensional_ndim3",
        lambda: ServerMeasurements(np.zeros((2, 2, 2)), ("a", "b")),
        expect_error=True,
    )
    # Validación 2: desajuste entre columnas y nombres
    run_case(
        "desajuste_3_columnas_2_nombres",
        lambda: ServerMeasurements(np.zeros((5, 3)), ("a", "b")),
        expect_error=True,
    )
    run_case(
        "desajuste_2_columnas_3_nombres",
        lambda: ServerMeasurements(np.zeros((5, 2)), ("a", "b", "c")),
        expect_error=True,
    )
    # Validación 3: datos no numéricos
    run_case(
        "no_numericos_cadenas",
        lambda: ServerMeasurements(
            np.array([["alto", "bajo"], ["medio", "bajo"]]), ("nivel", "carga")
        ),
        expect_error=True,
    )
    run_case(
        "no_numericos_objeto_mixto",
        lambda: ServerMeasurements(
            np.array([[1.0, "x"], [2.0, "y"]], dtype=object), ("a", "b")
        ),
        expect_error=True,
    )
    # Caso válido: no debe lanzar excepción
    run_case(
        "caso_valido_sin_error",
        lambda: ServerMeasurements(np.array([[1.0, 2.0, 3.0]]), ("a", "b", "c")),
        expect_error=False,
    )
    # Defensivo: build_measurements con columna faltante
    run_case(
        "build_measurements_falta_columna",
        lambda: build_measurements(
            pd.DataFrame({"gpu_utilization": [1.0], "cpu_utilization": [2.0]})
        ),
        expect_error=True,
    )
    return tests


def main():
    # --- Carga de datos (ruta relativa) -----------------------------------
    # Las columnas del CSV se usan tal cual, con los nombres exactos
    # indicados en el enunciado (gpu_utilization, cpu_utilization, memory_gb).
    df = pd.read_csv(DATA_PATH)

    # --- Construcción del objeto pedido ------------------------------------
    measurements = build_measurements(df)
    repr_str = repr(measurements)
    print(repr_str)  # repr esperado por el criterio de éxito

    # --- Pruebas de las tres validaciones ----------------------------------
    tests = run_validation_tests()
    all_passed = all(t["passed"] for t in tests)

    # --- Estadísticas descriptivas informativas (sin redondear) ------------
    stats = {}
    for i, name in enumerate(measurements.feature_names):
        col = measurements.values[:, i]
        finite = col[np.isfinite(col)]
        stats[name] = {
            "n_valores": int(col.size),
            "n_no_finitos": int(col.size - finite.size),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "media": float(finite.mean()) if finite.size else None,
            "desv_std": float(finite.std()) if finite.size else None,
        }

    results = {
        "subtarea": "T1_representar_las_mediciones",
        "archivo_datos": DATA_PATH,
        "clase": "ServerMeasurements",
        "repr": repr_str,
        "repr_esperado": EXPECTED_REPR,
        "repr_coincide_con_esperado": bool(repr_str == EXPECTED_REPR),
        "shape": [
            int(measurements.values.shape[0]),
            int(measurements.values.shape[1]),
        ],
        "n_filas": int(measurements.values.shape[0]),
        "n_columnas": int(measurements.values.shape[1]),
        "dtype_values": str(measurements.values.dtype),
        "feature_names": list(measurements.feature_names),
        "orden_caracteristicas": list(FEATURES),
        "estadisticas_por_caracteristica": stats,
        "pruebas_validacion": tests,
        "todas_las_pruebas_pasaron": bool(all_passed),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --- Resumen breve ------------------------------------------------------
    print("-" * 70)
    print(
        f"Datos leidos: {DATA_PATH} -> {df.shape[0]} filas, "
        f"{df.shape[1]} columnas"
    )
    print(f"Objeto construido: {repr_str}")
    print(f"repr coincide con el esperado: {repr_str == EXPECTED_REPR}")
    print(f"dtype de values: {measurements.values.dtype}")
    print("Pruebas de validacion (ValueError esperado en los casos de error):")
    for t in tests:
        estado = "OK   " if t["passed"] else "FALLO"
        print(
            f"  [{estado}] {t['test']} "
            f"(lanza ValueError: {t['lanza_valueerror']})"
        )
    print(f"Todas las pruebas pasaron: {all_passed}")
    print(f"Resultados escritos en: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
