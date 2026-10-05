"""[7] Crítico — mira antes de opinar. C4.

Primero las comprobaciones de CÓDIGO (no se dejan convencer): la guarda, el código de salida,
el tiempo, que exista resultados.json y sea un objeto con cifras, que no haya NaN/inf, que la
figura exista si el criterio la pide, y las dos fugas de la Parte 0.c (estática sobre el AST y
de plausibilidad), más el escalador ajustado antes de dividir que la 0.c no detectaba.
Solo si todo pasa opina el LLM (M2). En el M1 decide solo el código.
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

METRICAS = ("acc", "exact", "f1", "precision", "recall", "auc", "r2")
PIDE_FIGURA = re.compile(r"figura|gr[aá]fic|curva|plot|png|dibuja|barras|histograma", re.I)
DIVISIONES = {"train_test_split", "KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit"}


def _numeros(valor, ruta="") -> list[tuple[str, float]]:
    if isinstance(valor, bool):
        return []
    if isinstance(valor, (int, float)):
        return [(ruta, float(valor))]
    if isinstance(valor, dict):
        return [x for k, v in valor.items() for x in _numeros(v, f"{ruta}.{k}" if ruta else str(k))]
    if isinstance(valor, list):
        return [x for i, v in enumerate(valor) for x in _numeros(v, f"{ruta}[{i}]")]
    return []


def evalua_sobre_entrenamiento(codigo: str) -> list[str]:
    """La de la 0.c: la misma variable en .fit(X…) y en .predict(X)/.score(X…)."""
    ajustadas, evaluadas = set(), []
    for n in ast.walk(ast.parse(codigo)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args \
                and isinstance(n.args[0], ast.Name):
            if n.func.attr in {"fit", "fit_transform"}:
                ajustadas.add(n.args[0].id)
            elif n.func.attr in {"predict", "predict_proba", "score", "decision_function"}:
                evaluadas.append(n.args[0].id)
    return sorted({v for v in evaluadas if v in ajustadas})


def ajuste_antes_de_dividir(codigo: str) -> list[str]:
    """Lo que la 0.c NO detectaba: un .fit/.fit_transform sobre X cuyo resultado (o X mismo)
    se divide después. Las estadísticas de prueba ya entraron al escalador."""
    sospechosas = []
    arbol = ast.parse(codigo)
    ajustes: dict[str, int] = {}          # variable ajustada o producida por un fit → línea
    for n in ast.walk(arbol):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "fit_transform":
            for t in n.targets:            # la SALIDA del ajuste; dividir la entrada después
                if isinstance(t, ast.Name):  # (submuestras del entrenamiento) no es fuga
                    ajustes[t.id] = n.lineno
    for n in ast.walk(arbol):
        if isinstance(n, ast.Call) and (getattr(n.func, "id", None) in DIVISIONES
                                        or getattr(n.func, "attr", None) in DIVISIONES | {"split"}):
            for a in n.args:
                if isinstance(a, ast.Name) and a.id in ajustes and ajustes[a.id] < n.lineno:
                    sospechosas.append(f"{a.id} (ajustado en la línea {ajustes[a.id]}, dividido en la {n.lineno})")
    return sorted(set(sospechosas))


def implausibles(resultados: dict, techo: float = 0.999) -> list[str]:
    return [f"{k}={v}" for k, v in _numeros(resultados)
            if k.split(".")[-1].split("[")[0].lower().startswith(METRICAS) and v >= techo]


def comprobar(subtarea: dict, codigo: str, ejecucion: dict) -> dict:
    """{aprobado, motivos, correccion}: la decisión del código."""
    motivos: list[str] = []
    if ejecucion.get("violaciones"):
        motivos += [f"guarda: {v}" for v in ejecucion["violaciones"]]
    if ejecucion.get("error") and not ejecucion.get("violaciones"):
        motivos.append(ejecucion["error"])
    if ejecucion.get("returncode") not in (0, None) and not ejecucion.get("timeout"):
        cola = "\n".join(ejecucion.get("stderr", "").strip().splitlines()[-12:])
        motivos.append(f"el script terminó con código {ejecucion['returncode']}:\n{cola}")
    carpeta = Path(ejecucion["carpeta"])
    resultados = None
    if ejecucion.get("returncode") == 0:
        ruta = carpeta / "resultados.json"
        if not ruta.exists():
            motivos.append("no escribió resultados.json (es el contrato)")
        else:
            try:
                resultados = json.loads(ruta.read_text(encoding="utf-8"))
            except json.JSONDecodeError as err:
                motivos.append(f"resultados.json no es JSON válido: {err.msg}")
            else:
                if not isinstance(resultados, dict) or not _numeros(resultados):
                    motivos.append("resultados.json no es un objeto con cifras")
                else:
                    malos = [k for k, v in _numeros(resultados) if not math.isfinite(v)]
                    if malos:
                        motivos.append(f"hay NaN o infinitos en resultados.json: {malos[:5]}")
                    if raras := implausibles(resultados):
                        motivos.append(f"plausibilidad: métricas ≥ 0,999 en un problema con ruido: {raras[:5]}")
        pide = PIDE_FIGURA.search(f"{subtarea.get('objetivo', '')} {subtarea.get('criterio', '')}")
        if pide and not any(a.endswith(".png") for a in ejecucion.get("archivos", [])):
            motivos.append("el criterio pide una figura y el script no guardó ningún PNG")
        try:
            if fugas := evalua_sobre_entrenamiento(codigo):
                motivos.append(f"fuga: se evalúa sobre las mismas variables con que se ajustó: {fugas}")
            if antes := ajuste_antes_de_dividir(codigo):
                motivos.append(f"fuga: se ajusta un transformador antes de dividir los datos: {antes}")
        except SyntaxError:
            pass
    correccion = "\n".join(f"- {m}" for m in motivos)
    return {"aprobado": not motivos, "motivos": motivos, "correccion": correccion,
            "resultados": resultados if not motivos else None}
