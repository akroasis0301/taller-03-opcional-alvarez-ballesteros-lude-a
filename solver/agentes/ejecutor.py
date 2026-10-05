"""[6] Ejecutor — corre el script en el sandbox. Es código: no hay LLM. C3, segunda mitad.

Orden: guarda estática → (descarga: confirmación humana) → proceso confinado.
El proceso: entorno VACÍO (ninguna variable del .env llega al script), carpeta de trabajo
propia, grupo de procesos nuevo y tiempo máximo que mata al GRUPO (no solo al padre).

Límite declarado (D4): en macOS, desde Python, no hay aislamiento de red a nivel del sistema
operativo. La red la bloquea la guarda (imports) y la falta de variables de proxy; un script
que la construyera en tiempo de ejecución no estaría contenido. Se declara en el informe.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from solver.sandbox.guarda import revisar

COPIAR_DE_DEPENDENCIAS = {".json", ".csv", ".png", ".parquet", ".npy", ".txt"}


def huella(codigo: str) -> str:
    return hashlib.sha256(" ".join(codigo.split()).encode()).hexdigest()[:16]


def preparar(carpeta: Path, codigo: str, entradas: dict[str, Path], datos: str | None) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "script.py").write_text(codigo, encoding="utf-8")
    for dep, origen in entradas.items():                # resultados de las dependencias, solo lectura
        destino = carpeta / "entrada" / dep
        destino.mkdir(parents=True, exist_ok=True)
        for f in Path(origen).iterdir():
            if f.is_file() and f.suffix in COPIAR_DE_DEPENDENCIAS and f.name not in {"stdout.txt", "stderr.txt"}:
                shutil.copy2(f, destino / f.name)
                (destino / f.name).chmod(0o444)
    if datos and Path(datos).is_dir():                   # datos de la tarea, con su nombre: data/*.csv
        destino = carpeta / Path(datos).name
        shutil.copytree(datos, destino, dirs_exist_ok=True)
        for f in destino.rglob("*"):
            if f.is_file():
                f.chmod(0o444)


def entorno_vacio(carpeta: Path, tmp: Path | None = None) -> dict[str, str]:
    """Nada del .env. La carpeta temporal va FUERA de la de resultados: las cachés de
    matplotlib traen números que, si no, contarían como «escritos por una ejecución»."""
    tmp = tmp or carpeta / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "TMPDIR": str(tmp),
            "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp), "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "LANG": "C.UTF-8"}


def ejecutar(codigo: str, carpeta: Path, *, entradas: dict[str, Path] | None = None,
             datos: str | None = None, timeout_s: int = 120,
             confirmar: Callable[[str], bool] | None = None, subtarea: str = "",
             tmp: Path | None = None) -> dict:
    """Resultado de una ejecución. Nunca lanza: todo error vuelve como dato."""
    t0 = time.perf_counter()
    veredicto = revisar(codigo)
    base = {"huella": huella(codigo), "carpeta": str(carpeta), "violaciones": veredicto.violaciones,
            "descargas": veredicto.descargas, "returncode": None, "stdout": "", "stderr": "",
            "archivos": [], "timeout": False, "error": None, "confirmacion": None}
    preparar(carpeta, codigo, entradas or {}, datos)
    if not veredicto.permitido:
        base["error"] = "guarda: " + "; ".join(veredicto.violaciones)
        base["duracion_s"] = round(time.perf_counter() - t0, 2)
        return base
    if veredicto.descargas:
        motivo = f"{subtarea}: el script quiere descargar ({'; '.join(veredicto.descargas)})"
        aprobado = bool(confirmar and confirmar(motivo))
        base["confirmacion"] = {"motivo": motivo, "aprobado": aprobado}
        if not aprobado:
            base["error"] = "descarga no confirmada por una persona: " + "; ".join(veredicto.descargas)
            base["duracion_s"] = round(time.perf_counter() - t0, 2)
            return base
    antes = {p for p in carpeta.rglob("*") if p.is_file()}
    proc = subprocess.Popen([sys.executable, "script.py"], cwd=carpeta, env=entorno_vacio(carpeta, tmp),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)           # su propio grupo de procesos
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)               # mata al GRUPO: también a los hijos
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        base["timeout"] = True
        base["error"] = f"timeout: superó {timeout_s} s y se mató el grupo de procesos"
    base["returncode"] = proc.returncode
    base["stdout"], base["stderr"] = out[-20_000:], err[-20_000:]
    (carpeta / "stdout.txt").write_text(out, encoding="utf-8")
    (carpeta / "stderr.txt").write_text(err, encoding="utf-8")
    base["archivos"] = sorted(str(p.relative_to(carpeta)) for p in carpeta.rglob("*")
                              if p.is_file() and p not in antes and ".tmp" not in p.parts
                              and p.name not in {"stdout.txt", "stderr.txt"})
    base["duracion_s"] = round(time.perf_counter() - t0, 2)
    return base
