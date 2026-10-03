# Tareas reales — banco de prueba del solver

Enunciados de *Matemáticas y Programación para IA* (otra materia de la maestría), ya resueltos y
entregados por Miguel Álvarez. Se usan solo para evaluar el solver; uso a declarar en el informe
(Parte 5) y con permiso del profesor de esa materia.

| Carpeta | `enunciado.pdf` (versión solver) | Datos | Entregable |
|---|---|---|---|
| `semana1/` | Telemetría de servidores: NumPy, Pandas, Parquet | `data/server_measurements.csv` | scripts + `output/server_analysis.parquet` + `reporte.md` |
| `semana2/` | Comparación pareada de dos modelos | `data/model_errors.csv` (también como tabla en el PDF) | scripts + `output/store_differences.{csv,png}` + `informe.md` |
| `semana3/` | Tensores, autograd y CPU vs GPU en PyTorch | se generan con semilla | notebook ejecutado + `resultados_ejercicio3/` |

`enunciado_original.pdf` es el del curso, sin cambios. La versión solver conserva contenido,
constantes y resultados esperados; solo quita la logística (Git, uv, pytest, Colab, D2L) y, en la
semana 3, transcribe la configuración, las tareas A-D y las preguntas P1-P6 del notebook inicial.
Dependencias extra: `pyarrow` (semana 1) y el grupo opcional `torch` (`uv sync --extra torch`).
