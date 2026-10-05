"""Guarda estática sobre el árbol sintáctico — C3, primera mitad.

Decide ANTES de que exista un proceso. No ejecuta nada y no usa modelo: lee el AST.
Bloquea: red (sockets, HTTP), procesos (subprocess, os.system…), borrado de archivos,
eval/exec/compile/__import__, y rutas que salen de la carpeta de trabajo (absolutas, ~, ..).
Aparte señala las DESCARGAS (fetch_openml, load_dataset, download=True, urlretrieve): no se
bloquean por regla, se detienen hasta que una persona las confirme (freno 4 de la Parte 3).

Límite declarado: una guarda estática no ve lo que se construye en tiempo de ejecución
(getattr(os, "sys"+"tem")). Por eso el proceso además corre con entorno vacío, carpeta propia
y tiempo máximo (ejecutor.py).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

MODULOS_PROHIBIDOS = {"socket", "ssl", "requests", "httpx", "urllib", "urllib3", "http", "aiohttp",
                      "ftplib", "smtplib", "telnetlib", "subprocess", "multiprocessing", "pty",
                      "ctypes", "webbrowser", "paramiko", "shutil"}
LLAMADAS_PROHIBIDAS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
ATRIBUTOS_PROHIBIDOS = {
    "system", "popen", "spawnl", "spawnv", "spawnve", "execv", "execve", "execl", "fork", "kill",
    "remove", "unlink", "rmdir", "removedirs", "rmtree", "rename", "replace", "chmod", "chown",
    "putenv", "unsetenv", "setuid", "symlink", "link", "truncate"}
DESCARGAS = {"fetch_openml", "load_dataset", "urlretrieve", "fetch_20newsgroups",
             "fetch_california_housing", "fetch_covtype", "fetch_lfw_people", "fetch_olivetti_faces",
             "fetch_kddcup99", "fetch_rcv1", "fetch_species_distributions", "hf_hub_download",
             "snapshot_download"}
FUNCIONES_DE_RUTA = {"open", "Path", "read_csv", "read_parquet", "read_json", "to_csv", "to_parquet",
                     "savefig", "load", "save", "loadtxt", "savetxt", "genfromtxt", "read_excel",
                     "to_json", "write_text", "read_text", "imread", "imsave", "chdir", "listdir",
                     "walk", "glob", "scandir", "mkdir", "makedirs"}


@dataclass
class Veredicto:
    violaciones: list[str] = field(default_factory=list)
    descargas: list[str] = field(default_factory=list)

    @property
    def permitido(self) -> bool:
        return not self.violaciones


def _nombre(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _ruta_fuera(valor: str) -> bool:
    v = valor.strip()
    return v.startswith(("/", "~", "\\")) or ".." in v.replace("\\", "/").split("/") or \
        (len(v) > 1 and v[1] == ":")                      # C:\…


def revisar(codigo: str) -> Veredicto:
    v = Veredicto()
    try:
        arbol = ast.parse(codigo)
    except SyntaxError as err:
        v.violaciones.append(f"el código no compila: {err.msg} (línea {err.lineno})")
        return v
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for a in nodo.names:
                if a.name.split(".")[0] in MODULOS_PROHIBIDOS:
                    v.violaciones.append(f"línea {nodo.lineno}: import {a.name} (red, procesos o borrado)")
        elif isinstance(nodo, ast.ImportFrom):
            if (nodo.module or "").split(".")[0] in MODULOS_PROHIBIDOS:
                v.violaciones.append(f"línea {nodo.lineno}: from {nodo.module} import … (red, procesos o borrado)")
            for a in nodo.names:
                if a.name in ATRIBUTOS_PROHIBIDOS | LLAMADAS_PROHIBIDAS:
                    v.violaciones.append(f"línea {nodo.lineno}: from {nodo.module} import {a.name}")
                if a.name in DESCARGAS:
                    v.descargas.append(f"línea {nodo.lineno}: import {a.name}")
        elif isinstance(nodo, ast.Call):
            nombre = _nombre(nodo.func)
            if isinstance(nodo.func, ast.Name) and nombre in LLAMADAS_PROHIBIDAS:
                v.violaciones.append(f"línea {nodo.lineno}: {nombre}()")
            if isinstance(nodo.func, ast.Attribute) and nombre in ATRIBUTOS_PROHIBIDOS:
                v.violaciones.append(f"línea {nodo.lineno}: .{nombre}() (procesos, borrado o permisos)")
            if nombre in DESCARGAS:
                v.descargas.append(f"línea {nodo.lineno}: {nombre}()")
            for kw in nodo.keywords:
                if kw.arg == "download" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    v.descargas.append(f"línea {nodo.lineno}: {nombre}(download=True)")
            if nombre in FUNCIONES_DE_RUTA:
                for arg in list(nodo.args[:1]) + [k.value for k in nodo.keywords if k.arg in {"path", "fname", "file"}]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _ruta_fuera(arg.value):
                        v.violaciones.append(f"línea {nodo.lineno}: {nombre}({arg.value!r}) sale de la carpeta de trabajo")
        elif isinstance(nodo, ast.Attribute) and nodo.attr == "environ":
            v.violaciones.append(f"línea {nodo.lineno}: os.environ (el script no lee el entorno)")
    return v
