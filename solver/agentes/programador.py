"""[5] Programador — un script por subtarea de cálculo. C4 (el contrato de resultados).

El contrato es fijo: el script escribe sus cifras en `resultados.json` (nombre fijo, en su
carpeta), guarda las figuras como PNG y lee lo de sus dependencias de `entrada/<id>/`.
Si el crítico lo devolvió, recibe el script anterior y la corrección.
"""
from __future__ import annotations

import re

from solver.cliente_llm import ClienteLLM

SISTEMA = """Eres el programador de un solver de tareas de una maestría en IA. Escribes UN script
de Python 3.12 que resuelve una subtarea. Reglas obligatorias:
- Librerías disponibles: numpy, pandas, scipy, scikit-learn, matplotlib (backend Agg), pyarrow.
- Sin red ni descargas, sin subprocess, sin borrar archivos, sin os.environ, sin eval/exec.
- Solo rutas relativas a la carpeta actual. Los resultados de subtareas previas están en
  entrada/<id>/ (resultados.json y archivos); los archivos de datos de la tarea, si existen,
  están en las rutas relativas que se listan en el contexto (p. ej. data/archivo.csv).
- Escribe TODAS las cifras que la subtarea pide en resultados.json (un objeto JSON con claves
  descriptivas, valores numéricos sin redondear o listas/tablas). Ese archivo es el contrato.
- Guarda cada figura como PNG en la carpeta actual (plt.savefig("nombre.png", dpi=120)).
- Respeta al pie de la letra semillas, particiones y parámetros del enunciado. El conjunto de
  prueba no se usa para ajustar nada (tampoco escaladores ni selección de atributos).
- Si el contexto lista ARCHIVOS QUE EXIGE EL ENUNCIADO y tu subtarea produce alguno, escríbelo con
  esa ruta relativa EXACTA (crea la carpeta con Path(...).parent.mkdir(parents=True, exist_ok=True)).
- Imprime un resumen breve de los resultados.
Devuelve SOLO el script, dentro de un bloque ```python ... ```."""


def mensajes(subtarea: dict, contexto: dict, previo: str | None, correccion: str | None) -> list[dict]:
    usuario = (f"SUBTAREA {subtarea['id']}: {subtarea.get('objetivo', '')}\n"
               f"CRITERIO DE ÉXITO: {subtarea.get('criterio', '')}\n\n"
               f"CONTEXTO DEL ENUNCIADO (citado por sección):\n{contexto['texto']}\n")
    if contexto.get("previos"):
        usuario += f"\nRESULTADOS DE LAS SUBTAREAS PREVIAS:\n{contexto['previos']}\n"
    msgs = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": usuario}]
    if previo and correccion:
        msgs += [{"role": "assistant", "content": f"```python\n{previo}\n```"},
                 {"role": "user", "content": f"El crítico RECHAZÓ el script. Corrige esto y devuelve "
                  f"el script completo:\n{correccion}"}]
    return msgs


def extraer_codigo(texto: str) -> str:
    bloques = re.findall(r"```(?:python|py)?\s*\n(.*?)```", texto, re.S)
    if bloques:
        return max(bloques, key=len).strip() + "\n"
    return texto.strip() + "\n"


def programar(llm: ClienteLLM, subtarea: dict, contexto: dict, previo: str | None = None,
              correccion: str | None = None) -> str:
    return extraer_codigo(llm.pedir("programador", mensajes(subtarea, contexto, previo, correccion),
                                    subtarea=subtarea["id"]))
