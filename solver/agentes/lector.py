"""[1] Lector — PDF (o carpeta) → documento: secciones, tablas, formato y restricciones.

Sin LLM. Lee con PyMuPDF, igual que evaluar_solver.py, para que el solver y el evaluador vean
el mismo texto. Un encabezado se reconoce por la tipografía (más grande que el cuerpo y en
negrita, o mucho más grande), con una regla de respaldo por patrón («Parte 2 — …») cuando el
PDF no trae información de fuentes.

Lo que el código comprueba (tabla del enunciado): ligaduras (ﬁ → fi, con NFKC), palabras
cortadas por guion al final de línea, y PDF escaneado (páginas sin texto: se marca, el OCR
es opcional y no está en el baseline).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from statistics import median

# «Parte 2», «Tarea 3», «Pregunta 1», «Ejercicio 4»: secciones de trabajo del enunciado.
PATRON_TRABAJO = re.compile(r"^(?:#+\s*)?(Parte|Tarea|Pregunta|Ejercicio|Problema)\s+(\d+)\b", re.I)
ENTREGA = re.compile(r"\b([\w\-]+\.(?:md|pdf|ipynb))\b", re.I)
NUMEROS_ESCRITOS = {"uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
                    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", texto)              # ligaduras: ﬁ → fi
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)             # guion de corte de línea
    return texto


def clave_de(titulo: str) -> str:
    """«Parte 3 — La curva…» → «Parte 3». Si no es de trabajo, el título mismo."""
    m = PATRON_TRABAJO.match(titulo.strip())
    return f"{m.group(1).capitalize()} {m.group(2)}" if m else titulo.strip()[:60]


# ------------------------------------------------------------------ extracción
def lineas_pdf(ruta: Path) -> tuple[list[dict], list[dict], list[int]]:
    """Líneas con tipografía, tablas y páginas sin texto (posible escaneo)."""
    import pymupdf

    lineas, tablas, sin_texto = [], [], []
    with pymupdf.open(ruta) as doc:
        for n, pagina in enumerate(doc, start=1):
            if not pagina.get_text().strip():
                sin_texto.append(n)
                continue
            for bloque in pagina.get_text("dict")["blocks"]:
                for linea in bloque.get("lines", []):
                    spans = [s for s in linea["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    lineas.append({
                        "texto": normalizar("".join(s["text"] for s in linea["spans"])).strip(),
                        "tam": round(max(s["size"] for s in spans), 1),
                        "negrita": all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans),
                        "pagina": n})
            try:
                for t in pagina.find_tables().tables:
                    filas = [[normalizar(c or "").strip() for c in fila] for fila in t.extract()]
                    if filas:
                        tablas.append({"pagina": n, "filas": filas})
            except Exception:  # find_tables es heurístico: sin tabla no se cae la lectura
                pass
    return lineas, tablas, sin_texto


def es_encabezado(linea: dict, cuerpo: float) -> bool:
    texto, tam = linea["texto"], linea.get("tam")
    if not texto or len(texto) > 100:
        return False
    if tam is None and re.match(r"^#{1,4}\s+\S", texto):      # Markdown (paquetes, Tarea D)
        return True
    if tam is not None and cuerpo:
        if tam >= cuerpo * 1.3 or (tam >= cuerpo * 1.1 and linea.get("negrita")):
            return True
        if tam < cuerpo * 1.1:                       # del tamaño del cuerpo: solo por patrón
            return bool(linea.get("negrita") and PATRON_TRABAJO.match(texto) and len(texto) < 80)
    return bool(PATRON_TRABAJO.match(texto)) and len(texto) < 80   # sin fuentes: respaldo


def segmentar(lineas: list[dict]) -> list[dict]:
    """Agrupa las líneas en secciones. La primera (antes del primer encabezado de trabajo)
    es el preámbulo: datos, tablas y reglas generales que todas las partes necesitan."""
    tams = [l["tam"] for l in lineas if l.get("tam") and len(l["texto"]) > 40]
    cuerpo = median(tams) if tams else 0
    secciones: list[dict] = []
    actual = {"titulo": "Preámbulo", "lineas": [], "pagina": 1, "tam": None}
    previo_encabezado = None
    for linea in lineas:
        if es_encabezado(linea, cuerpo):
            # Un título partido en dos renglones del mismo tamaño es un solo encabezado.
            if previo_encabezado and previo_encabezado.get("tam") == linea.get("tam") \
                    and not actual["lineas"]:
                actual["titulo"] += " " + linea["texto"]
                continue
            if actual["lineas"] or actual["titulo"] != "Preámbulo":
                secciones.append(actual)
            actual = {"titulo": linea["texto"].lstrip("#").strip(), "lineas": [], "pagina": linea.get("pagina", 1),
                      "tam": linea.get("tam")}
            previo_encabezado = linea
        else:
            actual["lineas"].append(linea["texto"])
            previo_encabezado = None
    secciones.append(actual)

    # El primer encabezado (el título del documento) y su texto forman el preámbulo.
    salida = []
    for i, s in enumerate(secciones):
        es_trabajo = bool(PATRON_TRABAJO.match(s["titulo"]))
        salida.append({"id": f"S{i}", "titulo": s["titulo"], "clave": clave_de(s["titulo"]),
                       "trabajo": es_trabajo, "pagina": s["pagina"],
                       "texto": "\n".join(s["lineas"]).strip()})
    if salida and not salida[0]["trabajo"]:
        salida[0]["clave"] = "Preámbulo"
    return salida


# ------------------------------------------------------------------ restricciones
def _numero(txt: str) -> int | None:
    txt = txt.strip().lower()
    if txt in NUMEROS_ESCRITOS:
        return NUMEROS_ESCRITOS[txt]
    digitos = re.sub(r"[\s.  ]", "", txt)
    return int(digitos) if digitos.isdigit() else None


def _limite(texto: str, unidad: str) -> int | None:
    num = r"(\d[\d   .]*\d|\d|" + "|".join(NUMEROS_ESCRITOS) + r")"
    patrones = [rf"(?:m[aá]ximo|como\s+m[aá]ximo|≤|hasta)\s*(?:de\s*)?{num}\s*{unidad}",
                rf"{num}\s*{unidad}\s*(?:como\s*)?m[aá]ximo"]
    for p in patrones:
        m = re.search(p, texto, re.I)
        if m and (n := _numero(m.group(1))):
            return n
    return None


def restricciones(texto: str, secciones: list[dict]) -> dict:
    """Formato del entregable, límites y secciones exigidas, leídos del enunciado."""
    plano = " ".join(texto.replace("*", "").split())
    entrega_txt = " ".join(s["texto"] + " " + s["titulo"] for s in secciones
                           if re.search(r"reporte|informe|entrega|formato", s["titulo"], re.I))
    candidatos = [n for n in ENTREGA.findall(entrega_txt + " " + plano)
                  if not re.search(r"enunciado|starter", n, re.I)]
    nombre = candidatos[0] if candidatos else None
    if nombre is None and re.search(r"\bnotebook\b|\.ipynb", plano, re.I):
        nombre = "solucion.ipynb"
    nombre = nombre or "reporte.md"

    exigidas: list[str] = []
    m = re.search(r"secciones(?:,?\s*en este orden)?\s*:?\s*(.+?)\.(?:\s|$)",
                  re.sub(r"\([^)]*\)", "", plano), re.I)
    if m:
        partes = re.split(r",\s*|\s+y\s+", m.group(1))
        exigidas = [p.strip() for p in partes
                    if p.strip() and p.strip()[0].isupper() and len(p.split()) <= 4]
    return {"entregable": nombre,
            "formato": Path(nombre).suffix.lstrip(".").lower(),
            "palabras_max": _limite(plano, "palabras"),
            "paginas_max": _limite(plano, r"p[aá]ginas?"),
            "secciones": exigidas}


# ------------------------------------------------------------------ entrada
def leer(ruta_entrada: str | Path) -> dict:
    """El documento completo. `ruta_entrada` es un PDF o una carpeta (paquete, Tarea D)."""
    ruta = Path(ruta_entrada)
    if not ruta.exists():
        raise FileNotFoundError(f"la entrada no existe: {ruta}")
    tablas, sin_texto = [], []
    if ruta.is_dir():
        lineas = []
        for f in sorted(ruta.rglob("*")):
            if f.suffix == ".pdf":
                l, t, s = lineas_pdf(f)
                lineas += l
                tablas += t
                sin_texto += s
            elif f.suffix in {".md", ".txt"}:
                lineas += [{"texto": x, "tam": None} for x in normalizar(f.read_text(encoding="utf-8")).splitlines()]
            elif f.suffix == ".ipynb":
                nb = json.loads(f.read_text(encoding="utf-8"))
                for c in nb.get("cells", []):
                    lineas += [{"texto": x, "tam": None} for x in "".join(c.get("source", "")).splitlines()]
        datos = ruta
    else:
        lineas, tablas, sin_texto = lineas_pdf(ruta)
        datos = ruta.parent / "data" if (ruta.parent / "data").is_dir() else None

    secciones = segmentar(lineas)
    texto = "\n".join(l["texto"] for l in lineas)
    return {"fuente": str(ruta), "texto": texto, "secciones": secciones, "tablas": tablas,
            "paginas_sin_texto": sin_texto, "escaneado": bool(sin_texto) and not texto.strip(),
            "datos": str(datos) if datos else None,
            "restricciones": restricciones(texto, secciones)}
