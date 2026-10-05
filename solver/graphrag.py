"""[2] Indexador (GraphRAG), capa 2 — entidades, notas del curso, embeddings y comunidades. C2.

La capa 1 (agentes/indexador.py) es el esqueleto determinístico: secciones y aristas
`depende_de` por regla. Esta capa va ENCIMA y nunca la reemplaza:

  1. El LLM extrae entidades (dataset, método, métrica, concepto, restricción) y relaciones
     de cada sección del enunciado y de cada fragmento de las notas del curso.
  2. Se fusionan por NOMBRE NORMALIZADO («Regresión logística (LR)» = «regresion logistica»
     = «LR»): así una entidad del enunciado queda unida a lo que dicen las notas de ella.
  3. Embeddings (bge-m3 en el Ollama de la H200) eligen las entidades SEMILLA de cada consulta.
  4. Comunidades (Louvain) sobre el grafo de las notas, cada una con un resumen del LLM:
     la búsqueda GLOBAL ordena esos resúmenes por similitud.
  5. Búsqueda LOCAL para el investigador: semillas → vecinos en el grafo → los fragmentos de
     las notas que las mencionan, cada uno con su cita «[s2-rag-y-vector-search § 4.2 …]».

Costo: el índice de las notas se construye UNA vez (≈6 min con la H200, en paralelo) y queda
en cache/graphrag/; tiene su propia traza y su propio presupuesto, porque se amortiza entre
todas las corridas. Lo que sí se cobra a cada tarea es extraer las entidades de SU enunciado.

Si algo de esta capa falla (sin embeddings, JSON inválido), el solver sigue con la capa 1: el
investigador siempre entrega la sección literal y sus dependencias. Eso queda en la traza.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np

from solver.cliente_llm import ClienteLLM, PresupuestoAgotado
from solver.traza import Traza

VERSION = "capa2-v1"          # cambia el prompt o el formato → la caché se reconstruye
TIPOS = ("dataset", "metodo", "metrica", "concepto", "restriccion")
MAX_CAR_FRAGMENTO = 2500      # fragmentos de las notas: se juntan subsecciones hasta este tamaño
MIN_MIEMBROS_COMUNIDAD = 3    # las comunidades más chicas no llevan resumen

SISTEMA_ENTIDADES = """Eres el indexador de un solver de tareas de una maestría en IA. Extraes del
fragmento las entidades y las relaciones entre ellas, para un grafo de conocimiento. Tipos:
- dataset: conjuntos de datos y archivos (p. ej. breast_cancer, data/ventas.csv)
- metodo: modelos, algoritmos y técnicas (regresión logística, HNSW, LoRA, top-p)
- metrica: medidas de evaluación (exactitud, F1, KL, perplejidad, recall@k)
- concepto: ideas teóricas (atención, entropía, fuga de datos, variable latente)
- restriccion: reglas que hay que cumplir (semilla 42, 30 % de prueba, no tocar el conjunto
  de prueba, estandarizar antes de entrenar, máximo de palabras)
Devuelve SOLO un objeto JSON:
{"entidades": [{"nombre": "...", "tipo": "...", "alias": ["sigla u otro nombre"],
                "descripcion": "una frase fiel al texto"}],
 "relaciones": [{"origen": "nombre", "destino": "nombre", "relacion": "verbo_corto"}]}
Solo lo que dice el texto, sin inventar. Como máximo 15 entidades y 15 relaciones.
Los nombres de las relaciones usan entidades de tu propia lista."""

SISTEMA_COMUNIDAD = """Eres el indexador de un solver de tareas de una maestría en IA. Te doy una
comunidad de entidades del material del curso y sus relaciones. Escribe un resumen de 2 o 3
frases: qué tema las une y para qué sirve ese conocimiento al resolver una tarea. Solo texto,
sin cifras que no estén en las descripciones."""

_ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "the", "a", "an"}
_GENERICAS = {"modelo", "dato", "resultado", "tarea", "parte", "pregunta", "figura", "tabla",
              "seccion", "enunciado", "codigo", "script", "archivo", "ejemplo", "valor", "metodo"}


# =========================================================================== nombres
def _sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalizar(nombre: str) -> str:
    """La clave de fusión. «Las Regresiones Logísticas» → «regresion logistica».

    El plural se quita con una regla burda, pero igual a los dos lados: lo que importa es que
    el enunciado y las notas lleguen a la MISMA clave, no que la clave sea bonita.
    """
    palabras = re.sub(r"[^a-z0-9]+", " ", _sin_acentos(nombre).lower()).split()
    while palabras and palabras[0] in _ARTICULOS:
        palabras.pop(0)
    return " ".join(_singular(p) for p in palabras)


def _singular(p: str) -> str:
    if len(p) > 4 and p.endswith("es") and p[-3] in "lnrdj":     # regresiones, redes, árboles
        return p[:-2]
    if len(p) > 3 and p.endswith("s") and not p.endswith("ss"):  # métricas, datos, embeddings
        return p[:-1]
    return p


def formas(entidad: dict) -> list[str]:
    """Todas las claves por las que se puede fusionar: nombre, alias y la sigla entre paréntesis."""
    nombre = str(entidad.get("nombre") or "")
    crudas = [nombre, re.sub(r"\([^)]*\)", "", nombre)]
    crudas += re.findall(r"\(([^)]{2,15})\)", nombre)
    crudas += [str(a) for a in entidad.get("alias") or [] if isinstance(a, (str, int, float))]
    vistas = []
    for c in crudas:
        n = normalizar(c)
        if len(n) >= 2 and n not in vistas:
            vistas.append(n)
    return vistas


# =========================================================================== notas del curso
def fragmentar_notas(carpeta: Path) -> list[dict]:
    """Los .md de las notas, partidos por «##»/«###» (ignorando los «#» dentro de bloques de
    código) y reagrupados hasta MAX_CAR_FRAGMENTO sin cruzar una sección «##»."""
    fragmentos = []
    for archivo in sorted(Path(carpeta).glob("*.md")):
        bloques, actual, en_codigo, seccion2 = [], None, False, ""
        for linea in archivo.read_text(encoding="utf-8").splitlines():
            if linea.strip().startswith("```"):
                en_codigo = not en_codigo
            m = None if en_codigo else re.match(r"^(#{2,3})\s+(.+?)\s*$", linea)
            if m:
                if m.group(1) == "##":
                    seccion2 = m.group(2)
                actual = {"titulo": m.group(2), "nivel": len(m.group(1)), "seccion": seccion2, "lineas": []}
                bloques.append(actual)
            elif actual is not None:
                actual["lineas"].append(linea)
        grupo: list[dict] = []

        def cerrar():
            if not grupo:
                return
            texto = "\n\n".join(f"{'#' * b['nivel']} {b['titulo']}\n" + "\n".join(b["lineas"]).strip()
                                for b in grupo).strip()
            titulos = [b["titulo"] for b in grupo]
            fragmentos.append({"id": f"{archivo.stem}#{len([f for f in fragmentos if f['fuente'] == archivo.stem]) + 1}",
                               "fuente": archivo.stem, "titulos": titulos,
                               "cita": f"{archivo.stem} § {titulos[0]}" + (f" … {titulos[-1]}" if len(titulos) > 1 else ""),
                               "texto": texto})
            grupo.clear()

        for b in bloques:
            largo = sum(len("\n".join(x["lineas"])) + len(x["titulo"]) for x in grupo)
            if grupo and (b["nivel"] == 2 or largo + len("\n".join(b["lineas"])) > MAX_CAR_FRAGMENTO):
                cerrar()
            grupo.append(b)
        cerrar()
    return fragmentos


def huella_notas(carpeta: Path | None, modelo: str) -> str:
    h = hashlib.sha256(f"{VERSION}|{modelo}|{MAX_CAR_FRAGMENTO}".encode())
    if carpeta and Path(carpeta).is_dir():
        for f in sorted(Path(carpeta).glob("*.md")):
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


# =========================================================================== extracción (LLM)
def _limpiar_extraccion(datos: dict) -> dict:
    entidades, nombres = [], set()
    for e in (datos.get("entidades") or [])[:20]:
        if not isinstance(e, dict) or not str(e.get("nombre") or "").strip():
            continue
        f = formas(e)
        if not f or f[0] in _GENERICAS or len(f[0]) < 2:
            continue
        tipo = _sin_acentos(str(e.get("tipo") or "concepto")).lower().strip()
        entidades.append({"nombre": str(e["nombre"]).strip()[:80], "tipo": tipo if tipo in TIPOS else "concepto",
                          "alias": [str(a)[:40] for a in e.get("alias") or [] if isinstance(a, (str, int, float))][:5],
                          "descripcion": str(e.get("descripcion") or "").strip()[:300]})
        nombres.update(f)
    relaciones = []
    for r in (datos.get("relaciones") or [])[:20]:
        if not isinstance(r, dict):
            continue
        o, d = normalizar(str(r.get("origen") or "")), normalizar(str(r.get("destino") or ""))
        if o in nombres and d in nombres and o != d:
            rel = re.sub(r"[^a-z0-9]+", "_", _sin_acentos(str(r.get("relacion") or "relacionado_con")).lower()).strip("_")
            relaciones.append({"origen": o, "destino": d, "relacion": rel[:40] or "relacionado_con"})
    return {"entidades": entidades, "relaciones": relaciones}


def extraer(llm: ClienteLLM, texto: str, cita: str, subtarea: str | None = None) -> dict:
    datos = llm.pedir_json("indexador", [
        {"role": "system", "content": SISTEMA_ENTIDADES},
        {"role": "user", "content": f"FRAGMENTO [{cita}]:\n{texto[:6000]}"}], subtarea=subtarea,
        max_tokens=llm.cfg.max_tokens_indexador, tope=llm.cfg.max_tokens_indexador)
    return {"cita": cita, **_limpiar_extraccion(datos)}


def extraer_en_paralelo(llm: ClienteLLM, piezas: list[tuple[str, str]], hilos: int,
                        subtarea: str | None = None) -> list[dict]:
    """[(texto, cita)] → extracciones. vLLM agrupa las peticiones concurrentes (enunciado del
    taller). Una pieza que falla se salta con su decisión en la traza; el presupuesto agotado
    sí se propaga: es un freno, no un error."""
    def una(pieza):
        texto, cita = pieza
        try:
            return extraer(llm, texto, cita, subtarea=subtarea)
        except PresupuestoAgotado:
            raise
        except Exception as err:
            llm.traza.decision("indexador", "extraccion_fallida", f"{cita}: {type(err).__name__}: {err}"[:300])
            return None
    with ThreadPoolExecutor(max_workers=max(1, hilos)) as pool:
        return [r for r in pool.map(una, piezas) if r is not None]


# =========================================================================== fusión
class Fusion:
    """Entidades fusionadas por cualquiera de sus formas normalizadas."""

    def __init__(self):
        self.entidades: dict[str, dict] = {}     # clave → entidad
        self.indice: dict[str, str] = {}         # forma → clave
        self.relaciones: list[dict] = []

    def clave_de(self, forma: str) -> str | None:
        return self.indice.get(forma)

    def agregar(self, extraccion: dict, origen: str, fragmento: str | None = None) -> list[str]:
        cita, claves_locales, locales = extraccion["cita"], [], {}
        for e in extraccion["entidades"]:
            fs = formas(e)
            clave = next((self.indice[f] for f in fs if f in self.indice), None)
            if clave is None:
                clave = fs[0]
                self.entidades[clave] = {"clave": clave, "nombre": e["nombre"], "tipo": e["tipo"],
                                         "alias": [], "descripciones": [], "citas": [], "origen": [],
                                         "fragmentos": []}
            ent = self.entidades[clave]
            if ent["tipo"] == "concepto" and e["tipo"] != "concepto":
                ent["tipo"] = e["tipo"]
            for a in [e["nombre"], *e["alias"]]:
                if a not in ent["alias"] and a != ent["nombre"]:
                    ent["alias"].append(a)
            if e["descripcion"] and e["descripcion"] not in ent["descripciones"]:
                ent["descripciones"].append(e["descripcion"])
            for lista, valor in ((ent["citas"], cita), (ent["origen"], origen)):
                if valor not in lista:
                    lista.append(valor)
            if fragmento and fragmento not in ent["fragmentos"]:
                ent["fragmentos"].append(fragmento)
            for f in fs:
                self.indice.setdefault(f, clave)
                locales[f] = clave
            claves_locales.append(clave)
        for r in extraccion["relaciones"]:
            o, d = locales.get(r["origen"]), locales.get(r["destino"])
            if not (o and d and o != d):
                continue
            igual = next((x for x in self.relaciones
                          if (x["origen"], x["destino"], x["relacion"]) == (o, d, r["relacion"])), None)
            if igual is None:                 # la misma relación en otro fragmento: una arista, más citas
                self.relaciones.append({"origen": o, "destino": d, "relacion": r["relacion"],
                                        "cita": cita, "citas": [cita], "origen_capa": [origen]})
            else:
                for lista, valor in ((igual["citas"], cita), (igual["origen_capa"], origen)):
                    if valor not in lista:
                        lista.append(valor)
        return claves_locales

    def a_dict(self) -> dict:
        return {"entidades": self.entidades, "indice": self.indice, "relaciones": self.relaciones}

    @classmethod
    def de_dict(cls, d: dict) -> "Fusion":
        f = cls()
        f.entidades = json.loads(json.dumps(d["entidades"]))
        f.indice = dict(d["indice"])
        f.relaciones = json.loads(json.dumps(d["relaciones"]))   # copia: la tarea no toca el índice
        return f


def texto_entidad(e: dict) -> str:
    return f"{e['nombre']} ({e['tipo']}): " + " ".join(e["descripciones"][:2])


# =========================================================================== embeddings
class EmbedLexico:
    """Respaldo sin red: n-gramas de caracteres con hashing (sin ajuste, determinístico).
    No es semántico como bge-m3, pero «regresión logística» y «regresiones logísticas»
    quedan cerca. Se usa en las pruebas y si el Ollama de embeddings no responde."""

    nombre = "lexico-char34-1024"

    def embed(self, textos: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer
        v = HashingVectorizer(n_features=1024, analyzer="char_wb", ngram_range=(3, 4),
                              alternate_sign=False, norm="l2", strip_accents="unicode", lowercase=True)
        return v.transform(list(textos)).toarray().astype(np.float32)


class EmbedH200:
    """bge-m3 en el Ollama de la H200, con el embed() de solver-v2/h200.py (ya normaliza)."""

    def __init__(self, h200):
        self.h = h200
        self.nombre = str(getattr(h200, "MODELO_EMB", "bge-m3"))

    def embed(self, textos: list[str]) -> np.ndarray:
        return np.asarray(self.h.embed(list(textos)), dtype=np.float32)


def elegir_embedder(cliente: ClienteLLM, preferido=None, traza: Traza | None = None):
    """El de Config si lo hay; si no, bge-m3 de la H200; si no responde, el léxico."""
    traza = traza or cliente.traza
    candidatos = [preferido] if preferido is not None else []
    if hasattr(cliente.llm, "embed"):
        candidatos.append(EmbedH200(cliente.llm))
    for c in candidatos:
        try:
            v = c.embed(["prueba de embeddings"])
            if len(v) == 1 and len(v[0]) > 0:
                return c
        except Exception as err:
            traza.decision("indexador", "embeddings_no_disponibles",
                           f"{getattr(c, 'nombre', c)}: {type(err).__name__}: {err}"[:300])
    if candidatos:
        traza.decision("indexador", "embeddings_lexicos", "se usan n-gramas de caracteres como respaldo")
    return EmbedLexico()


def embeber(embedder, textos: list[str], traza: Traza, que: str) -> np.ndarray:
    if not textos:
        return np.zeros((0, 1), dtype=np.float32)
    t0 = time.perf_counter()
    m = embedder.embed(textos)
    traza.evento("embeddings", agente="indexador", modelo=embedder.nombre, que=que,
                 textos=len(textos), dimension=int(m.shape[1]), latencia_s=round(time.perf_counter() - t0, 2))
    return m


# =========================================================================== comunidades
def grafo_entidades(fusion: Fusion, coapariciones: list[list[str]]) -> nx.Graph:
    """Grafo no dirigido para Louvain: relaciones (peso 2) y co-aparición en un fragmento (1)."""
    g = nx.Graph()
    g.add_nodes_from(fusion.entidades)
    for r in fusion.relaciones:
        w = g.get_edge_data(r["origen"], r["destino"], {}).get("weight", 0)
        g.add_edge(r["origen"], r["destino"], weight=w + 2 * len(r["citas"]))
    for claves in coapariciones:
        unicas = sorted(set(claves))
        for i, a in enumerate(unicas):
            for b in unicas[i + 1:]:
                w = g.get_edge_data(a, b, {}).get("weight", 0)
                g.add_edge(a, b, weight=w + 1)
    return g


def comunidades(g: nx.Graph) -> list[list[str]]:
    if g.number_of_edges() == 0:
        return []
    grupos = nx.community.louvain_communities(g, weight="weight", seed=0)
    return sorted((sorted(c) for c in grupos if len(c) >= MIN_MIEMBROS_COMUNIDAD), key=lambda c: (-len(c), c))


def resumir_comunidades(llm: ClienteLLM, fusion: Fusion, grupos: list[list[str]], hilos: int) -> list[dict]:
    def una(par):
        i, miembros = par
        ents = [fusion.entidades[c] for c in miembros][:25]
        rels = [r for r in fusion.relaciones if r["origen"] in miembros and r["destino"] in miembros][:20]
        usuario = ("ENTIDADES:\n" + "\n".join(f"- {texto_entidad(e)}" for e in ents)
                   + "\n\nRELACIONES:\n" + ("\n".join(f"- {fusion.entidades[r['origen']]['nombre']} —{r['relacion']}→ "
                                                     f"{fusion.entidades[r['destino']]['nombre']}" for r in rels) or "(ninguna)"))
        citas = sorted({c for e in ents for c in e["citas"]})
        try:
            resumen = llm.pedir("indexador", [{"role": "system", "content": SISTEMA_COMUNIDAD},
                                              {"role": "user", "content": usuario}], subtarea=f"comunidad-{i}",
                               max_tokens=llm.cfg.max_tokens_indexador, tope=llm.cfg.max_tokens_indexador).strip()
        except PresupuestoAgotado:
            raise
        except Exception as err:
            llm.traza.decision("indexador", "resumen_fallido", f"comunidad {i}: {type(err).__name__}: {err}"[:300])
            resumen = "Entidades: " + ", ".join(e["nombre"] for e in ents[:12])
        return {"id": i, "miembros": miembros, "resumen": resumen[:1200], "citas": citas}
    with ThreadPoolExecutor(max_workers=max(1, hilos)) as pool:
        return list(pool.map(una, enumerate(grupos)))


# =========================================================================== índice de las notas
def _escribir(ruta: Path, datos) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ruta)                     # atómico: una corrida cortada no deja un índice a medias


def indice_notas(cliente: ClienteLLM, cfg, traza_corrida: Traza) -> dict | None:
    """El índice de las notas del curso, de la caché o construido (una vez). Devuelve None si no
    hay notas. Su traza y su presupuesto son propios: es un costo que se amortiza."""
    carpeta = Path(cfg.notas) if cfg.notas else None
    if not carpeta or not carpeta.is_dir() or not any(carpeta.glob("*.md")):
        traza_corrida.decision("indexador", "sin_notas", f"no hay notas del curso en {carpeta}")
        return None
    clave = huella_notas(carpeta, cliente.modelo)
    destino = Path(cfg.cache_graphrag) / f"notas-{clave}"
    ruta = destino / "indice.json"
    if ruta.exists():
        indice = json.loads(ruta.read_text(encoding="utf-8"))
        traza_corrida.decision("indexador", "indice_notas", f"en caché: {ruta}", cache=True,
                               entidades=len(indice["fusion"]["entidades"]), comunidades=len(indice["comunidades"]))
        return indice

    t0 = time.perf_counter()
    traza = Traza(destino / "traza_notas.jsonl")
    cfg_notas = dataclasses.replace(cfg, llm=cliente.llm, presupuesto_tokens=cfg.presupuesto_notas,
                                    reserva_redactor=0)
    llm = ClienteLLM(cfg_notas, traza)
    fragmentos = fragmentar_notas(carpeta)
    traza.evento("inicio", agente="indexador", que="indice_notas", carpeta=str(carpeta),
                 fragmentos=len(fragmentos), modelo=llm.modelo)
    extracciones = extraer_en_paralelo(llm, [(f["texto"], f["cita"]) for f in fragmentos], cfg.hilos_indexador)
    por_cita = {f["cita"]: f["id"] for f in fragmentos}
    fusion, coap = Fusion(), []
    for ex in extracciones:
        coap.append(fusion.agregar(ex, "notas", fragmento=por_cita.get(ex["cita"])))
    grupos = comunidades(grafo_entidades(fusion, coap))
    resumenes = resumir_comunidades(llm, fusion, grupos, cfg.hilos_indexador)
    for c in resumenes:
        for m in c["miembros"]:
            fusion.entidades[m]["comunidad"] = c["id"]
    tokens = traza.tokens_totales
    indice = {"version": VERSION, "clave": clave, "modelo": llm.modelo, "creado": time.strftime("%Y-%m-%d %H:%M:%S"),
              "fragmentos": fragmentos, "fusion": fusion.a_dict(), "comunidades": resumenes,
              "tokens": tokens, "duracion_s": round(time.perf_counter() - t0, 1)}
    _escribir(ruta, indice)
    traza.evento("fin", agente="indexador", **tokens, duracion_s=indice["duracion_s"])
    traza_corrida.decision("indexador", "indice_notas",
                           f"construido en {indice['duracion_s']} s: {len(fragmentos)} fragmentos, "
                           f"{len(fusion.entidades)} entidades, {len(fusion.relaciones)} relaciones, "
                           f"{len(resumenes)} comunidades; {sum(tokens.values())} tokens que NO cuentan para el "
                           f"presupuesto de esta tarea (traza: {traza.ruta})", cache=False, **tokens)
    return indice


def _embeddings_notas(indice: dict, embedder, cfg, traza: Traza) -> dict[str, np.ndarray]:
    """Los vectores de las notas también van a caché, por embedder (bge-m3 o léxico)."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", embedder.nombre)
    ruta = Path(cfg.cache_graphrag) / f"notas-{indice['clave']}" / f"emb-{slug}.npz"
    claves = list(indice["fusion"]["entidades"])
    if ruta.exists():
        with np.load(ruta) as z:
            if list(z["claves"]) == claves:
                return {k: z[k] for k in ("entidades", "fragmentos", "comunidades")}
    ents = indice["fusion"]["entidades"]
    m = {"entidades": embeber(embedder, [texto_entidad(ents[c]) for c in claves], traza, "entidades de las notas"),
         "fragmentos": embeber(embedder, [f["texto"][:2000] for f in indice["fragmentos"]], traza, "fragmentos de las notas"),
         "comunidades": embeber(embedder, [c["resumen"] for c in indice["comunidades"]], traza, "resúmenes de comunidades")}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ruta, claves=np.array(claves, dtype=str), **m)
    return m


# =========================================================================== índice de la tarea
@dataclasses.dataclass
class IndiceTarea:
    fusion: Fusion
    fragmentos: list[dict]
    comunidades: list[dict]
    claves: list[str]                 # orden de las filas de E
    E: np.ndarray                     # entidades
    F: np.ndarray                     # fragmentos de las notas
    C: np.ndarray                     # resúmenes de comunidades
    embedder: object
    del_enunciado: dict[str, list[str]]   # sección → claves de entidades que menciona

    def resumen(self) -> dict:
        ents = self.fusion.entidades
        propias = {c for cs in self.del_enunciado.values() for c in cs}
        fusionadas = sorted(c for c in propias if "notas" in ents[c]["origen"])
        return {"entidades_enunciado": len(propias),
                "por_tipo": dict(Counter(ents[c]["tipo"] for c in propias)),
                "fusionadas_con_notas": len(fusionadas),
                "ejemplos_fusion": [ents[c]["nombre"] for c in fusionadas[:8]],
                "entidades_total": len(ents), "relaciones": len(self.fusion.relaciones),
                "comunidades": len(self.comunidades), "embeddings": self.embedder.nombre}


def indexar_tarea(cliente: ClienteLLM, cfg, documento: dict, traza: Traza,
                  notas: dict | None) -> IndiceTarea:
    """Extrae las entidades de CADA sección del enunciado (con cargo a la tarea) y las fusiona
    con las del índice de las notas."""
    embedder = elegir_embedder(cliente, cfg.embedder, traza)
    fusion = Fusion.de_dict(notas["fusion"]) if notas else Fusion()
    # Solo las secciones de TRABAJO: son las únicas que la búsqueda local usa como anclas (las de la
    # subtarea y sus dependencias). Semana 2 (exploratoria, 2026-10-05): extraer las 18 secciones,
    # anexos y escenario incluidos, costó 84 k tokens de 309 k. Sin secciones de trabajo, todas.
    candidatas = [s for s in documento["secciones"] if s["texto"].strip()]
    de_trabajo = [s for s in candidatas if s.get("trabajo")]
    piezas = [(f"{s['titulo']}\n{s['texto']}", s["clave"]) for s in (de_trabajo or candidatas)]
    extracciones = extraer_en_paralelo(cliente, piezas, cfg.hilos_indexador)
    del_enunciado = {ex["cita"]: fusion.agregar(ex, "enunciado") for ex in extracciones}
    comunidades_notas = notas["comunidades"] if notas else []
    # Las entidades nuevas del enunciado heredan la comunidad de sus vecinos de las notas.
    _asignar_comunidades(fusion)
    claves = list(fusion.entidades)
    if notas:
        mn = _embeddings_notas(notas, embedder, cfg, traza)
        de_notas = set(notas["fusion"]["entidades"])
        fila = {c: i for i, c in enumerate(notas["fusion"]["entidades"])}
        nuevas = [c for c in claves if c not in de_notas]
        En = embeber(embedder, [texto_entidad(fusion.entidades[c]) for c in nuevas], traza, "entidades del enunciado")
        dim = mn["entidades"].shape[1] if len(mn["entidades"]) else En.shape[1]
        E = np.zeros((len(claves), dim), dtype=np.float32)
        nueva = {c: i for i, c in enumerate(nuevas)}
        for i, c in enumerate(claves):
            # una entidad fusionada se re-embebe con su descripción ampliada solo si es nueva;
            # las de las notas conservan su vector de caché
            E[i] = mn["entidades"][fila[c]] if c in fila else En[nueva[c]]
        F, C = mn["fragmentos"], mn["comunidades"]
    else:
        E = embeber(embedder, [texto_entidad(fusion.entidades[c]) for c in claves], traza, "entidades del enunciado")
        F = C = np.zeros((0, E.shape[1] if len(E) else 1), dtype=np.float32)
    return IndiceTarea(fusion, notas["fragmentos"] if notas else [], comunidades_notas, claves, E, F, C,
                       embedder, del_enunciado)


def _asignar_comunidades(fusion: Fusion) -> None:
    vecinos = defaultdict(list)
    for r in fusion.relaciones:
        vecinos[r["origen"]].append(r["destino"])
        vecinos[r["destino"]].append(r["origen"])
    for c, e in fusion.entidades.items():
        if e.get("comunidad") is None:
            votos = Counter(fusion.entidades[v].get("comunidad") for v in vecinos[c]
                            if fusion.entidades[v].get("comunidad") is not None)
            if votos:
                e["comunidad"] = votos.most_common(1)[0][0]


def al_grafo(g: nx.DiGraph, indice: IndiceTarea) -> nx.DiGraph:
    """Agrega la capa 2 al grafo del enunciado: nodos «ent:…», aristas «menciona» desde cada
    sección y las relaciones extraídas. Las aristas depende_de de la capa 1 no se tocan."""
    ents = indice.fusion.entidades
    usadas = {c for cs in indice.del_enunciado.values() for c in cs}
    usadas |= {r[k] for r in indice.fusion.relaciones for k in ("origen", "destino")
               if r["origen"] in usadas or r["destino"] in usadas}
    for c in usadas:
        e = ents[c]
        g.add_node(f"ent:{c}", capa="entidad", nombre=e["nombre"], tipo=e["tipo"], origen=e["origen"],
                   citas=e["citas"], comunidad=e.get("comunidad"))
    for seccion, claves in indice.del_enunciado.items():
        for c in claves:
            if seccion in g:
                g.add_edge(seccion, f"ent:{c}", tipo="menciona")
    for r in indice.fusion.relaciones:
        if r["origen"] in usadas and r["destino"] in usadas:
            g.add_edge(f"ent:{r['origen']}", f"ent:{r['destino']}", tipo="relacion",
                       relacion=r["relacion"], citas=r["citas"])
    return g


# =========================================================================== búsquedas
def _top(m: np.ndarray, q: np.ndarray, k: int, candidatos: list[int] | None = None) -> list[tuple[int, float]]:
    if len(m) == 0:
        return []
    sim = m @ q
    idx = candidatos if candidatos is not None else range(len(m))
    return sorted(((i, float(sim[i])) for i in idx), key=lambda t: -t[1])[:k]


def busqueda_global(indice: IndiceTarea, q: np.ndarray, k: int = 2,
                    preferidas: set | None = None) -> list[tuple[dict, float]]:
    """Los resúmenes de comunidad más parecidos a la consulta (las de las semillas primero)."""
    pares = _top(indice.C, q, len(indice.comunidades))
    pares.sort(key=lambda t: (indice.comunidades[t[0]]["id"] not in (preferidas or set()), -t[1]))
    return [(indice.comunidades[i], s) for i, s in pares[:k]]


MAX_FRAGMENTOS_ANCLA = 4   # una entidad citada en más fragmentos («LLM», «embeddings») no ancla nada
MARGEN_SIMILITUD = 0.05    # solo los fragmentos a esta distancia del mejor


def anclas(indice: IndiceTarea, secciones: list[str]) -> list[str]:
    """Las entidades que el enunciado menciona en esas secciones Y que también están en las notas:
    son el único puente legítimo hacia el material del curso. Calibrado con r2 (2026-10-04): sin
    este filtro, la T1 de la Tarea A (cargar breast_cancer y dividir) recibía «§ 6.3 GraphRAG» y
    «§ 5.3 Structured outputs» con similitud 0.52, más alta que la de fragmentos pertinentes de
    otras tareas (0.508): ningún umbral de similitud separaba lo útil del ruido."""
    ents = indice.fusion.entidades
    vistas = [c for s in secciones for c in indice.del_enunciado.get(s, [])]
    return [c for c in dict.fromkeys(vistas)
            if "notas" in ents[c]["origen"] and 0 < len(ents[c].get("fragmentos", [])) <= MAX_FRAGMENTOS_ANCLA]


def busqueda_local(indice: IndiceTarea, subtarea: dict, propias: list[str], titulos: list[str],
                   dependencias: list[str] | None = None, k_semillas: int = 6, k_vecinos: int = 6,
                   k_fragmentos: int = 3, max_car: int = 6000) -> dict:
    """Semillas (las entidades de las secciones propias + las k más parecidas) → vecinos; y del
    material del curso, SOLO lo que alcanzan las anclas (entidades del enunciado que también están
    en las notas, en las secciones propias o en sus dependencias): sus fragmentos más parecidos a la
    subtarea y los resúmenes de sus comunidades. Sin anclas no se agregan notas. Todo citado."""
    ents, fila = indice.fusion.entidades, {c: i for i, c in enumerate(indice.claves)}
    consulta = " ".join([subtarea.get("objetivo", ""), subtarea.get("criterio", ""), *titulos])
    q = indice.embedder.embed([consulta])[0]
    sim = (indice.E @ q) if len(indice.E) else np.zeros(0)
    mencionadas = [c for s in propias for c in indice.del_enunciado.get(s, [])]
    semillas = list(dict.fromkeys(sorted(mencionadas, key=lambda c: -sim[fila[c]])[:k_semillas]
                                  + [indice.claves[i] for i, _ in _top(indice.E, q, k_semillas)]))
    vecinos: dict[str, None] = {}
    rels = []
    for r in indice.fusion.relaciones:
        if r["origen"] in semillas or r["destino"] in semillas:
            rels.append(r)
            for c in (r["origen"], r["destino"]):
                if c not in semillas:
                    vecinos[c] = None
    vecinos_top = sorted(vecinos, key=lambda c: -sim[fila[c]])[:k_vecinos]
    seleccion = semillas + vecinos_top
    rels = [r for r in rels if r["origen"] in seleccion and r["destino"] in seleccion][:15]

    ancladas = anclas(indice, list(propias) + list(dependencias or []))
    ids_frag = {f["id"]: i for i, f in enumerate(indice.fragmentos)}
    candidatos = sorted({ids_frag[f] for c in ancladas for f in ents[c]["fragmentos"] if f in ids_frag})
    frags = _top(indice.F, q, k_fragmentos, candidatos) if candidatos else []
    if frags:
        frags = [(i, s) for i, s in frags if s >= frags[0][1] - MARGEN_SIMILITUD]
    de_anclas = {ents[c].get("comunidad") for c in ancladas} - {None}
    globales = [(c, s) for c, s in busqueda_global(indice, q, len(indice.comunidades), de_anclas)
                if c["id"] in de_anclas][:2]

    def fuentes(e):
        return "; ".join(e["citas"][:3])
    partes = ["[GraphRAG — búsqueda local. Material de REFERENCIA del curso: si algo contradice al "
              "enunciado, manda el enunciado]",
              "Entidades (semillas y vecinos):\n" + "\n".join(
                  f"- {ents[c]['nombre']} ({ents[c]['tipo']}): {(ents[c]['descripciones'] or [''])[0]} "
                  f"[{fuentes(ents[c])}]" for c in seleccion)]
    if rels:
        partes.append("Relaciones:\n" + "\n".join(
            f"- {ents[r['origen']]['nombre']} —{r['relacion']}→ {ents[r['destino']]['nombre']} "
            f"[{'; '.join(r['citas'][:2])}]"
            for r in rels))
    citas = []
    if not ancladas:
        partes.append("[material del curso: ninguna entidad de esta subtarea aparece en las notas; "
                      "no se agregan fragmentos]")
    for i, s in frags:
        f = indice.fragmentos[i]
        partes.append(f"[notas del curso — {f['cita']} · similitud {s:.3f}]\n{f['texto'][:1200]}")
        citas.append(f["cita"])
    for c, s in globales:
        partes.append(f"[GraphRAG — búsqueda global: comunidad {c['id']} · similitud {s:.3f}]\n{c['resumen']}")
        citas.append(f"comunidad {c['id']}")
    texto = "\n\n".join(partes)
    if len(texto) > max_car:
        texto = texto[:max_car] + "\n[… recortado]"
    return {"texto": texto, "citas": citas, "semillas": [ents[c]["nombre"] for c in semillas],
            "vecinos": [ents[c]["nombre"] for c in vecinos_top],
            "anclas": [ents[c]["nombre"] for c in ancladas]}


# =========================================================================== figura
def dibujar_entidades(g: nx.DiGraph, destino: Path, max_nodos: int = 45) -> Path | None:
    """Secciones de trabajo + sus entidades; color por origen (enunciado / fusionada / notas)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    secciones = [n for n, d in g.nodes(data=True) if d.get("capa") == "enunciado" and d.get("trabajo")]
    ents = [v for s in secciones for _, v, d in g.out_edges(s, data=True) if d.get("tipo") == "menciona"]
    ents = list(dict.fromkeys(ents))[:max_nodos]
    if not ents:
        return None
    sub = g.subgraph(secciones + ents)
    color = {"enunciado": "#4C78A8", "ambos": "#B279A2", "notas": "#BBBBBB"}

    def origen(n):
        o = set(sub.nodes[n].get("origen") or [])
        return "ambos" if o == {"enunciado", "notas"} else ("notas" if o == {"notas"} else "enunciado")
    pos = nx.spring_layout(sub.to_undirected(), seed=0, k=0.9)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(sub, pos, nodelist=secciones, node_color="#F58518", node_size=1300, ax=ax)
    nx.draw_networkx_nodes(sub, pos, nodelist=ents, node_color=[color[origen(n)] for n in ents], node_size=420, ax=ax)
    nx.draw_networkx_edges(sub, pos, edge_color="#999999", width=0.8, arrows=False, ax=ax)
    etiquetas = {n: n for n in secciones} | {n: sub.nodes[n].get("nombre", n)[:22] for n in ents}
    nx.draw_networkx_labels(sub, pos, labels=etiquetas, font_size=7, ax=ax)
    for nombre, c in [("sección de trabajo", "#F58518"), ("entidad solo del enunciado", color["enunciado"]),
                      ("fusionada con las notas del curso", color["ambos"])]:
        ax.scatter([], [], c=c, label=nombre)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Capa 2 del GraphRAG — entidades del enunciado y su fusión con las notas", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return destino
