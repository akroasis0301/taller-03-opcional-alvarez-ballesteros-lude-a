"""La evidencia de la traza para la Parte 2.c: qué recibió cada agente, qué produjo y qué decidió el código.

    uv run python scripts/evidencia.py corridas/completo-r2/tarea-C                 # panorama de la corrida
    uv run python scripts/evidencia.py corridas/completo-r2/tarea-C --subtarea T3   # la historia de una subtarea
    uv run python scripts/evidencia.py corridas/completo-r2/tarea-C --subtarea T3 --completo   # con mensajes
    uv run python scripts/evidencia.py corridas/completo-r2/tarea-B --redactor      # redacción y procedencia
    ... --salida informe/evidencia_C_T3.md                                           # a un archivo

No interpreta: ordena los eventos de la traza y señala los archivos que los respaldan (el
contexto que recibió el programador, cada script con su salida, los rechazados), para que la
2.c pueda decir «el fallo apunta a este agente» con la traza como evidencia.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DECISIONES_SUBTAREA = {"rechazado", "aprobado", "rechazo_descartado", "tope_intentos", "script_repetido",
                       "llm_fallo", "sin_script", "llm_no_disponible", "presupuesto_agotado", "omitida",
                       "tope_de_razonamiento", "json_invalido", "confirmacion_red", "busqueda_local_fallida",
                       "conceptual_al_redactor"}


def leer(ruta: Path) -> list[dict]:
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def recorte(texto, n: int) -> str:
    texto = texto if isinstance(texto, str) else json.dumps(texto, ensure_ascii=False)
    return texto if len(texto) <= n else texto[:n] + f" … [{len(texto) - n} car. más]"


def bloque(texto: str) -> str:
    return "```\n" + texto.replace("```", "ʼʼʼ") + "\n```"


def linea_llamada(f: dict) -> str:
    err = f" · **error: {f['error']}**" if f.get("error") else ""
    # «intento» de una llamada es el reintento del cliente (vacío por longitud, red), no el
    # intento de código de la subtarea: ese lo cuentan las ejecuciones.
    reintento = f" (reintento {f['intento']} del cliente)" if (f.get("intento") or 1) > 1 else ""
    return (f"- `#{f['seq']}` **{f['agente']}** llamada{reintento}: "
            f"{f.get('tokens_entrada')} → {f.get('tokens_salida')} tokens, max_tokens={f.get('max_tokens')}, "
            f"fin={f.get('fin')}, {f.get('latencia_s')} s{err}")


def linea_evento(f: dict) -> str:
    if f["tipo"] == "llamada":
        return linea_llamada(f)
    if f["tipo"] == "ejecucion":
        err = f" · **error: {f['error']}**" if f.get("error") else ""
        return (f"- `#{f['seq']}` **ejecutor** intento {f.get('intento')}: returncode={f.get('returncode')}, "
                f"{f.get('duracion_s')} s, archivos={f.get('archivos')}{err}")
    if f["tipo"] == "decision":
        extra = ""
        for k in ("por", "semillas", "anclas", "caracteres", "fraccion", "total"):
            if f.get(k) not in (None, "", []):
                extra += f" · {k}={f[k]}"
        return f"- `#{f['seq']}` **{f['agente']}** → `{f['decision']}`{extra}: {recorte(f.get('motivo', ''), 600)}"
    if f["tipo"] == "error":
        return f"- `#{f['seq']}` **ERROR en {f['agente']}**: {recorte(f.get('error', ''), 600)}"
    return f"- `#{f['seq']}` {f['tipo']} {f.get('agente', '')}"


def archivos_de(carpeta: Path, sid: str) -> list[str]:
    salida = []
    for base, etiqueta in [(carpeta / "subtareas" / sid, "intento"),
                           (carpeta / "cache_solver" / "rechazados" / sid, "RECHAZADO")]:
        for intento in sorted(base.glob("intento-*")):
            archivos = sorted(p.name for p in intento.iterdir() if p.is_file())
            salida.append(f"- {etiqueta} `{intento.relative_to(carpeta)}/`: {', '.join(archivos)}")
    ctx = carpeta / "cache_solver" / "contextos" / f"{sid}.md"
    if ctx.exists():
        salida.insert(0, f"- contexto del programador: `{ctx.relative_to(carpeta)}` ({len(ctx.read_text(encoding='utf-8'))} car.)")
    return salida


def panorama(carpeta: Path, filas: list[dict]) -> list[str]:
    inicio = next((f for f in filas if f["tipo"] == "inicio"), {})
    fin = next((f for f in reversed(filas) if f["tipo"] == "fin"), {})
    modelo = next((f.get("modelo") for f in filas if f["tipo"] == "llamada"), "")
    md = [f"# Evidencia — `{carpeta}`", "",
          f"variante **{inicio.get('variante')}** · modelo `{modelo}` · status **{fin.get('status')}** "
          f"({fin.get('motivo')}) · tokens {fin.get('tokens_entrada')} → {fin.get('tokens_salida')}", "",
          "## Decisiones de la corrida", ""]
    globales = [f for f in filas if f["tipo"] in {"decision", "error"} and not f.get("subtarea")
                and f.get("decision") not in {"conceptual_al_redactor"}]
    md += [linea_evento(f) for f in globales] + ["", "## Por subtarea", "",
           "| subtarea | status final | ejecuciones | rechazos (código / LLM) | llamadas al programador | tokens del programador (entrada → salida) |",
           "|---|---|---|---|---|---|"]
    for sid, status in (fin.get("subtareas") or {}).items():
        fs = [f for f in filas if f.get("subtarea") == sid]
        prog = [f for f in fs if f["tipo"] == "llamada" and f["agente"] == "programador"]
        rech = [f for f in fs if f.get("decision") == "rechazado"]
        md.append(f"| {sid} | {status} | {sum(f['tipo'] == 'ejecucion' for f in fs)} | "
                  f"{sum(r.get('por') == 'codigo' for r in rech)} / {sum(r.get('por') == 'llm' for r in rech)} | "
                  f"{len(prog)} | {sum(f['tokens_entrada'] for f in prog)} → {sum(f['tokens_salida'] for f in prog)} |")
    md += ["", "Detalle de una subtarea: `--subtarea T<n>`; redacción y procedencia: `--redactor`."]
    return md


def historia(carpeta: Path, filas: list[dict], sid: str, completo: bool) -> list[str]:
    plan = json.loads((carpeta / "plan.json").read_text(encoding="utf-8")) if (carpeta / "plan.json").exists() else {}
    s = next((x for x in plan.get("subtareas", []) if x["id"] == sid), {})
    fin = next((f for f in reversed(filas) if f["tipo"] == "fin"), {})
    md = [f"# Evidencia — `{carpeta}` · subtarea **{sid}**", "",
          f"- tipo: {s.get('tipo')} · secciones: {s.get('secciones')} · depende de: {s.get('depende_de')}",
          f"- objetivo: {s.get('objetivo')}", f"- criterio: {s.get('criterio')}",
          f"- status final: **{(fin.get('subtareas') or {}).get(sid, '?')}**", "", "## Archivos", ""]
    md += archivos_de(carpeta, sid) or ["- (ninguno)"]
    md += ["", "## Eventos, en orden", ""]
    for f in filas:
        if f.get("subtarea") != sid:
            continue
        if f["tipo"] == "decision" and f.get("decision") not in DECISIONES_SUBTAREA and f["agente"] != "investigador":
            continue
        md.append(linea_evento(f))
        if completo and f["tipo"] == "llamada":
            entrada = f.get("entrada") or []
            if entrada:
                md += ["", f"  Lo que recibió ({entrada[-1].get('role')}, último mensaje):", "",
                       bloque(recorte(entrada[-1].get("content", ""), 3000)), ""]
            md += ["  Lo que produjo:", "", bloque(recorte(f.get("salida") or "(vacío)", 2000)), ""]
    return md


def redaccion(carpeta: Path, filas: list[dict], completo: bool) -> list[str]:
    md = [f"# Evidencia — `{carpeta}` · redacción y procedencia", ""]
    for f in filas:
        if f.get("agente") in {"redactor", "procedencia"} or f.get("decision") in {"devuelto_al_redactor", "publicable"}:
            md.append(linea_evento(f))
            if completo and f["tipo"] == "llamada":
                md += ["", "  Lo que produjo:", "", bloque(recorte(f.get("salida") or "(vacío)", 3000)), ""]
    borrador = carpeta / "cache_solver" / "borrador.md"
    if borrador.exists():
        md += ["", f"Borrador del último intento: `{borrador.relative_to(carpeta)}`"]
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corrida", help="carpeta de una tarea, p. ej. corridas/completo-r2/tarea-C")
    ap.add_argument("--subtarea")
    ap.add_argument("--redactor", action="store_true")
    ap.add_argument("--completo", action="store_true", help="incluye lo que recibió y produjo cada llamada")
    ap.add_argument("--salida", help="escribe el Markdown en este archivo")
    a = ap.parse_args()

    carpeta = Path(a.corrida)
    traza = carpeta / "traza.jsonl"
    if not traza.exists():
        ap.error(f"no existe {traza}")
    filas = leer(traza)
    if a.subtarea:
        md = historia(carpeta, filas, a.subtarea, a.completo)
    elif a.redactor:
        md = redaccion(carpeta, filas, a.completo)
    else:
        md = panorama(carpeta, filas)
    texto = "\n".join(md) + "\n"
    if a.salida:
        Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
        Path(a.salida).write_text(texto, encoding="utf-8")
        print(f"→ {a.salida}")
    else:
        print(texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
