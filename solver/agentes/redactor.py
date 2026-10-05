"""[8] Redactor — el entregable en el formato del enunciado.

El LLM redacta con los resultados APROBADOS (y solo con ellos); el código comprueba antes de
publicar: secciones exigidas y en orden, límite de palabras, y procedencia (C5). Lo que falla
vuelve al redactor por su nombre («estas cifras no salen de ninguna ejecución: …»).

M1: Markdown. M2: PDF (con límite de páginas) y notebook ejecutado.
"""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path

from solver.cliente_llm import ClienteLLM
from solver.procedencia import sin_codigo

SISTEMA = """Eres el redactor de un solver de tareas de una maestría en IA. Escribes el entregable
en Markdown a partir de resultados MEDIDOS. Reglas:
- Usa SOLO las cifras que aparecen en los resultados que se te entregan (puedes redondearlas a
  4 decimales). Nunca inventes, estimes ni completes una cifra. Si falta un resultado, dilo.
- Usa exactamente las secciones exigidas, en ese orden, como encabezados "## Nombre".
- Para tablas usa Markdown; para figuras, ![descripción](figuras/archivo.png) con los archivos dados.
- Las partes conceptuales se argumentan con las cifras medidas.
- Respeta el límite de palabras. Devuelve SOLO el documento Markdown."""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def posicion_seccion(texto: str, seccion: str) -> int:
    """La misma regla del evaluador: el renglón que ES el encabezado, sin acentos ni '#'."""
    objetivo, pos = _norm(seccion), 0
    for linea in texto.splitlines(keepends=True):
        h = re.sub(r"^\d+ ", "", _norm(linea.lstrip("# ")))
        if h and len(h) <= len(objetivo) + 40 and (h == objetivo or h.startswith(objetivo + " ")):
            return pos
        pos += len(linea)
    return -1


IMAGEN = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def comprobar_forma(texto: str, restricciones: dict, carpeta: Path | None = None) -> list[str]:
    problemas = []
    if carpeta is not None:                  # una figura citada que no existe es otra cifra sin origen
        if faltan := [r for r in IMAGEN.findall(texto) if not (Path(carpeta) / r).exists()]:
            problemas.append(f"cita figuras que no existen: {faltan}; usa solo las FIGURAS DISPONIBLES "
                             "o di que la figura no se pudo generar")
    exigidas = restricciones.get("secciones") or []
    posiciones = [posicion_seccion(texto, s) for s in exigidas]
    if faltan := [s for s, p in zip(exigidas, posiciones) if p < 0]:
        problemas.append(f"faltan las secciones {faltan}")
    elif posiciones != sorted(posiciones):
        problemas.append(f"las secciones deben ir en este orden: {exigidas}")
    if (maximo := restricciones.get("palabras_max")) and (n := len(sin_codigo(texto).split())) > maximo:
        problemas.append(f"tiene {n} palabras y el máximo es {maximo}: recorta")
    return problemas


def copiar_figuras(aprobadas: dict[str, Path], salida: Path) -> list[str]:
    destino = salida / "figuras"
    nombres = []
    for sid, carpeta in aprobadas.items():
        for png in sorted(Path(carpeta).glob("*.png")):
            destino.mkdir(parents=True, exist_ok=True)
            nombre = f"{sid}_{png.name}"
            shutil.copy2(png, destino / nombre)
            nombres.append(f"figuras/{nombre}")
    return nombres


def mensajes(documento: dict, plan: list[dict], figuras: list[str], problemas: list[str] | None,
             borrador: str | None, notas: list[str]) -> list[dict]:
    r = documento["restricciones"]
    resultados = []
    for s in plan:
        if s.get("tipo") == "calculo":
            estado = (json.dumps(s.get("resultados"), ensure_ascii=False)[:6000]
                      if s.get("status") == "aprobada" else f"SIN RESULTADO ({s.get('status')})")
            resultados.append(f"[{s['id']} · {', '.join(s.get('secciones') or [])}] {s.get('objetivo', '')}\n{estado}")
        else:
            resultados.append(f"[{s['id']} · conceptual · {', '.join(s.get('secciones') or [])}] "
                              f"{s.get('objetivo', '')} (depende de {s.get('depende_de') or []})")
    usuario = (f"ENUNCIADO:\n{documento['texto']}\n\n"
               f"SECCIONES EXIGIDAS, EN ORDEN: {r.get('secciones') or 'las que pida el enunciado'}\n"
               f"LÍMITE DE PALABRAS: {r.get('palabras_max') or 'sin límite'}\n"
               f"FIGURAS DISPONIBLES: {figuras or 'ninguna'}\n\n"
               "RESULTADOS MEDIDOS POR SUBTAREA:\n" + "\n\n".join(resultados))
    if notas:
        usuario += "\n\nDEBES DECLARAR EN EL DOCUMENTO:\n" + "\n".join(f"- {n}" for n in notas)
    msgs = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": usuario}]
    if problemas and borrador:
        msgs += [{"role": "assistant", "content": borrador},
                 {"role": "user", "content": "El documento NO se puede publicar. Corrige y devuelve "
                  "el documento completo:\n" + "\n".join(f"- {p}" for p in problemas)}]
    return msgs


def limpiar(texto: str) -> str:
    t = texto.strip()
    m = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*)\n```", t, re.S)
    return (m.group(1) if m else t).strip() + "\n"


def redactar(llm: ClienteLLM, documento: dict, plan: list[dict], figuras: list[str],
             problemas: list[str] | None = None, borrador: str | None = None,
             notas: list[str] | None = None) -> str:
    return limpiar(llm.pedir("redactor", mensajes(documento, plan, figuras, problemas, borrador, notas or [])))
