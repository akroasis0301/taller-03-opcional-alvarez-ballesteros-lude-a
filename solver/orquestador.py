"""El orquestador: un grafo de estados en LangGraph que conecta los ocho agentes.

    leer → indexar → planificar ⇄ (plan inválido y < 3 intentos)
         → siguiente → investigar → programar → ejecutar → criticar
                ↑                      ↑______ rechazado y < tope ______|
                |_________________ aprobado / tope / omitida ___________|
         → (cola vacía | presupuesto | error) → redactar ⇄ verificar → cerrar

Los agentes viven en solver/agentes/ como funciones puras; aquí solo se decide quién va
después de quién. El LLM y la traza no viajan en el estado (no son serializables): los tiene
la instancia del Solver.

Contrato (Parte 1): Solver().solve(ruta_pdf, carpeta_salida) -> dict
    status, entregables, subtareas [{id, tipo, status, intentos}], usage, model, trace
y Solver().run(pregunta) -> dict con answer, trace, status, model, usage (Taller 4).
"""
from __future__ import annotations

import functools
import json
import shutil
import time
import uuid
from pathlib import Path

from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover  (nombres anteriores de LangGraph)
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from solver import formatos, graphrag
from solver.agentes import critico, ejecutor, indexador, investigador, lector, planificador, programador, redactor
from solver.cliente_llm import ClienteLLM, PresupuestoAgotado
from solver.config import RAIZ, Config
from solver.estado import Estado
from solver.procedencia import verificar as verificar_procedencia
from solver.traza import Traza

MAX_INTENTOS_PLAN = 3
MAX_REDACCIONES = 3          # la primera más dos correcciones
INTERNO = "cache_solver"     # lo que no escribió una ejecución: el evaluador no lo cuenta como respaldo


def _cargar(ruta: str) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _guardar(ruta: Path, datos) -> str:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return str(ruta)


def _misma(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return False


def _libre(ruta: Path) -> Path:
    """Dos corridas en el mismo segundo no deben chocar: -2, -3… si el nombre ya existe."""
    candidata, n = ruta, 2
    while candidata.exists():
        candidata = ruta.with_name(f"{ruta.name}-{n}")
        n += 1
    return candidata


class Solver:
    """El solver completo (baseline)."""

    variante = "completo"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.traza: Traza | None = None
        self.llm: ClienteLLM | None = None
        self.capa2: graphrag.IndiceTarea | None = None   # en memoria, como el LLM: no es serializable
        self.app = self._grafo().compile(checkpointer=InMemorySaver())

    # ================================================================ grafo
    def _grafo(self) -> StateGraph:
        g = StateGraph(Estado)
        for nombre in ["leer", "indexar", "planificar", "siguiente", "investigar", "programar",
                       "ejecutar", "criticar", "redactar", "verificar", "cerrar"]:
            g.add_node(nombre, self._nodo(nombre, getattr(self, f"_{nombre}")))
        g.add_edge(START, "leer")
        g.add_conditional_edges("leer", self._seguir_o_cerrar("indexar"), {"indexar": "indexar", "cerrar": "cerrar"})
        g.add_conditional_edges("indexar", self._seguir_o_cerrar("planificar"),
                                {"planificar": "planificar", "cerrar": "cerrar"})
        g.add_conditional_edges("planificar", self._tras_planificar,
                                {"planificar": "planificar", "siguiente": "siguiente",
                                 "redactar": "redactar", "cerrar": "cerrar"})
        g.add_conditional_edges("siguiente", self._tras_siguiente,
                                {"investigar": "investigar", "redactar": "redactar"})
        g.add_conditional_edges("investigar", self._seguir_o_redactar("programar"),
                                {"programar": "programar", "redactar": "redactar"})
        g.add_conditional_edges("programar", self._seguir_o_redactar("ejecutar"),
                                {"ejecutar": "ejecutar", "redactar": "redactar"})
        g.add_edge("ejecutar", "criticar")
        g.add_conditional_edges("criticar", self._tras_criticar,
                                {"programar": "programar", "siguiente": "siguiente", "redactar": "redactar"})
        g.add_conditional_edges("redactar", self._tras_redactar, {"verificar": "verificar", "cerrar": "cerrar"})
        g.add_conditional_edges("verificar", self._tras_verificar, {"redactar": "redactar", "cerrar": "cerrar"})
        g.add_edge("cerrar", END)
        return g

    def diagrama(self) -> str:
        """El diagrama del orquestador en Mermaid (entregable de la Parte 1)."""
        return self.app.get_graph().draw_mermaid()

    def _nodo(self, nombre: str, fn):
        """Ningún error mata la corrida: el presupuesto agotado o una excepción quedan en la
        traza y en motivo_parada, y los enrutadores deciden a dónde ir (redactar o cerrar)."""
        @functools.wraps(fn)
        def envuelto(estado: Estado) -> dict:
            try:
                return fn(estado) or {}
            except PresupuestoAgotado as err:
                self.traza.decision("orquestador", "presupuesto_agotado", str(err), nodo=nombre)
                return {"motivo_parada": "presupuesto"}
            except Exception as err:
                self.traza.error(nombre, err, subtarea=estado.get("actual"))
                return {"motivo_parada": f"error en {nombre}: {type(err).__name__}: {err}"}
        return envuelto

    # ================================================================ enrutadores
    @staticmethod
    def _parado(estado: Estado) -> bool:
        return bool(estado.get("motivo_parada"))

    def _seguir_o_cerrar(self, siguiente: str):
        return lambda estado: "cerrar" if self._parado(estado) else siguiente

    @staticmethod
    def _fallo_al_redactar(estado: Estado) -> bool:
        return (estado.get("motivo_parada") or "").startswith(("error en redactar", "error en verificar"))

    def _tras_planificar(self, estado: Estado) -> str:
        if self._parado(estado):
            return "redactar" if estado.get("motivo_parada") == "presupuesto" else "cerrar"
        if estado.get("problemas_plan"):
            return "planificar" if estado.get("intentos_plan", 0) < MAX_INTENTOS_PLAN else "cerrar"
        return "siguiente"

    def _tras_siguiente(self, estado: Estado) -> str:
        return "redactar" if self._parado(estado) or not estado.get("actual") else "investigar"

    def _seguir_o_redactar(self, siguiente: str):
        return lambda estado: "redactar" if self._parado(estado) else siguiente

    def _tras_criticar(self, estado: Estado) -> str:
        if self._parado(estado):
            return "redactar"
        actual = self._subtarea(estado, estado["actual"])
        return "siguiente" if actual["status"] in {"aprobada", "fallida"} else "programar"

    def _tras_redactar(self, estado: Estado) -> str:
        if self._fallo_al_redactar(estado) or estado.get("motivo_parada") == "presupuesto" \
                and estado.get("intentos_redaccion", 0) == 0:
            return "cerrar"
        return "verificar" if estado.get("borrador") else "cerrar"

    def _tras_verificar(self, estado: Estado) -> str:
        if self._fallo_al_redactar(estado):
            return "cerrar"
        if estado.get("problemas_redaccion") and estado.get("intentos_redaccion", 0) < MAX_REDACCIONES \
                and estado.get("motivo_parada") != "presupuesto":
            return "redactar"
        return "cerrar"

    # ================================================================ utilidades
    @staticmethod
    def _subtarea(estado: Estado, sid: str) -> dict:
        return next(s for s in estado["plan"] if s["id"] == sid)

    @staticmethod
    def _con(estado: Estado, sid: str, **cambios) -> list[dict]:
        """Una copia del plan con una subtarea cambiada (el estado no se muta en su sitio)."""
        return [{**s, **cambios} if s["id"] == sid else s for s in estado["plan"]]

    def _salida(self, estado: Estado) -> Path:
        return Path(estado["carpeta_salida"])

    # ================================================================ nodos
    def _leer(self, estado: Estado) -> dict:
        doc = lector.leer(estado["ruta_entrada"])
        ruta = _guardar(self._salida(estado) / INTERNO / "enunciado_leido.json", doc)
        trabajo = [s["clave"] for s in doc["secciones"] if s["trabajo"]]
        self.traza.decision("lector", "leido", f"{len(doc['secciones'])} secciones; de trabajo: {trabajo}; "
                            f"{len(doc['tablas'])} tablas; restricciones: {doc['restricciones']}",
                            paginas_sin_texto=doc["paginas_sin_texto"])
        if doc["escaneado"]:
            return {"documento": ruta, "motivo_parada": "error en leer: PDF escaneado sin texto (OCR no está en el baseline)"}
        if not trabajo:
            self.traza.decision("lector", "sin_secciones_de_trabajo",
                                "no se reconoció ninguna «Parte N»: el plan cubrirá el documento entero")
        return {"documento": ruta, "restricciones": doc["restricciones"]}

    def _indexar(self, estado: Estado) -> dict:
        doc = _cargar(estado["documento"])
        g = indexador.esqueleto(doc["secciones"])                     # capa 1: regla, no modelo
        aristas = [f"{u} → {v} ({d['regla']}: «{d['evidencia']}»)" for u, v, d in g.edges(data=True)]
        self.traza.decision("indexador", "esqueleto", "; ".join(aristas) or "sin aristas depende_de",
                            nodos=g.number_of_nodes(), aristas=g.number_of_edges())
        if self.cfg.usar_grafo and self.cfg.capa2:
            self._capa2(doc, g, self._salida(estado))
        ruta, png = indexador.guardar(g, self._salida(estado))
        self.traza.decision("indexador", "grafo_guardado", str(ruta), figura=str(png),
                            nodos=g.number_of_nodes(), aristas=g.number_of_edges())
        return {"grafo": str(ruta)}

    def _capa2(self, doc: dict, g, salida: Path) -> None:
        """Capa 2 del GraphRAG (C2): entidades del LLM fusionadas con las notas del curso,
        embeddings y comunidades. Si falla, la corrida sigue con la capa 1 y queda en la traza;
        el presupuesto agotado sí se propaga (es un freno)."""
        try:
            notas = graphrag.indice_notas(self.llm, self.cfg, self.traza)
            self.capa2 = graphrag.indexar_tarea(self.llm, self.cfg, doc, self.traza, notas)
            graphrag.al_grafo(g, self.capa2)
            ents = self.capa2.fusion.entidades
            _guardar(salida / INTERNO / "graphrag" / "entidades_enunciado.json",
                     {s: [ents[c] for c in cs] for s, cs in self.capa2.del_enunciado.items()})
            png = graphrag.dibujar_entidades(g, salida / "grafo_entidades.png")
            r = self.capa2.resumen()
            self.traza.decision("indexador", "capa2",
                                f"{r['entidades_enunciado']} entidades del enunciado {r['por_tipo']}; "
                                f"{r['fusionadas_con_notas']} fusionadas con las notas (p. ej. {r['ejemplos_fusion']}); "
                                f"{r['relaciones']} relaciones; {r['comunidades']} comunidades; "
                                f"embeddings: {r['embeddings']}", figura=str(png), **r)
        except PresupuestoAgotado:
            raise
        except Exception as err:
            self.capa2 = None
            self.traza.decision("indexador", "capa2_no_disponible",
                                f"{type(err).__name__}: {err}"[:500] + " — se sigue con el esqueleto (capa 1)")

    def _planificar(self, estado: Estado) -> dict:
        doc, g = _cargar(estado["documento"]), indexador.cargar(estado["grafo"])
        intento = estado.get("intentos_plan", 0) + 1
        plan, problemas = planificador.planificar(self.llm, doc, g, estado.get("problemas_plan"),
                                                  estado.get("plan_previo"))
        if problemas:
            self.traza.decision("planificador", "plan_invalido", "; ".join(problemas), intento=intento)
            return {"plan_previo": plan, "problemas_plan": problemas, "intentos_plan": intento}
        orden = planificador.ordenar(plan)
        _guardar(self._salida(estado) / "plan.json", {"subtareas": orden})
        self.traza.decision("planificador", "plan_valido", f"{len(orden)} subtareas: "
                            + ", ".join(f"{s['id']}({s['tipo']})" for s in orden), intento=intento)
        return {"plan": orden, "cola": [s["id"] for s in orden], "problemas_plan": [],
                "intentos_plan": intento, "rechazadas": {}}

    def _siguiente(self, estado: Estado) -> dict:
        plan, cola = estado["plan"], list(estado.get("cola") or [])
        while cola:
            sid = cola.pop(0)
            s = next(x for x in plan if x["id"] == sid)
            caidas = [d for d in s.get("depende_de") or []
                      if next(x for x in plan if x["id"] == d)["status"] in {"fallida", "omitida"}]
            if s["tipo"] == "conceptual":
                plan = [{**x, "status": "aprobada"} if x["id"] == sid else x for x in plan]
                self.traza.decision("orquestador", "conceptual_al_redactor", subtarea=sid)
                continue
            if caidas:
                plan = [{**x, "status": "omitida"} if x["id"] == sid else x for x in plan]
                self.traza.decision("orquestador", "omitida", f"depende de {caidas}, que no tienen resultado",
                                    subtarea=sid)
                continue
            return {"plan": plan, "cola": cola, "actual": sid, "contexto": {}, "codigo": "", "ejecucion": {}}
        return {"plan": plan, "cola": [], "actual": None}

    def _investigar(self, estado: Estado) -> dict:
        s = self._subtarea(estado, estado["actual"])
        doc = _cargar(estado["documento"])
        g = indexador.cargar(estado["grafo"]) if self.cfg.usar_grafo else None
        ctx = investigador.investigar(s, doc, g, estado["plan"], usar_grafo=self.cfg.usar_grafo)
        local = {}
        if self.cfg.usar_grafo and self.capa2 is not None:
            try:                                 # búsqueda local + global de la capa 2, con citas
                titulos = [x["titulo"] for x in doc["secciones"] if x["clave"] in (s.get("secciones") or [])]
                propias = s.get("secciones") or []
                local = graphrag.busqueda_local(self.capa2, s, propias, titulos,
                                                dependencias=indexador.dependencias(g, propias))
                ctx["texto"] += "\n\n" + local["texto"]
                ctx["citas"] = ctx["citas"] + local["citas"]
                ctx["modo"] = "grafo+capa2"
            except Exception as err:
                self.traza.decision("investigador", "busqueda_local_fallida", f"{type(err).__name__}: {err}"[:300],
                                    subtarea=s["id"])
        if doc.get("datos"):                    # los archivos de la tarea, con la ruta que verá el script
            base = Path(doc["datos"])
            archivos = [f"{base.name}/{f.relative_to(base)}" for f in sorted(base.rglob("*")) if f.is_file()][:30]
            ctx["texto"] += "\n\n[archivos de datos disponibles, rutas relativas]\n" + "\n".join(archivos)
        if archivos := doc["restricciones"].get("archivos"):
            ctx["texto"] += ("\n\n[ARCHIVOS QUE EXIGE EL ENUNCIADO, con su ruta relativa exacta]\n"
                             + "\n".join(f"- {a}" for a in archivos))
        ruta = self._salida(estado) / INTERNO / "contextos" / f"{s['id']}.md"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(ctx["texto"] + "\n\n" + ctx["previos"], encoding="utf-8")
        self.traza.decision("investigador", ctx["modo"], f"citas: {ctx['citas']}", subtarea=s["id"],
                            caracteres=len(ctx["texto"]), semillas=local.get("semillas"),
                            vecinos=local.get("vecinos"), anclas=local.get("anclas"))
        return {"contexto": ctx}

    def _programar(self, estado: Estado) -> dict:
        s = self._subtarea(estado, estado["actual"])
        intento = s.get("intentos", 0) + 1
        try:
            codigo = programador.programar(self.llm, s, estado["contexto"],
                                           previo=estado.get("codigo") or None, correccion=s.get("correccion"))
        except PresupuestoAgotado:
            raise                                   # el freno de presupuesto sí detiene la corrida
        except Exception as err:                    # timeout, vacío, red: falla ESTA subtarea, no la corrida
            self.traza.error("programador", err, subtarea=s["id"])
            self.traza.decision("programador", "llm_fallo", f"{type(err).__name__}: {err}",
                                subtarea=s["id"], intento=intento)
            return {"codigo": "", "plan": self._con(estado, s["id"], intentos=intento)}
        return {"codigo": codigo, "plan": self._con(estado, s["id"], intentos=intento)}

    def _ejecutar(self, estado: Estado) -> dict:
        s = self._subtarea(estado, estado["actual"])
        carpeta = self._salida(estado) / "subtareas" / s["id"] / f"intento-{s['intentos']}"
        if not estado.get("codigo"):                # el programador no entregó script
            return {"ejecucion": {"huella": "", "carpeta": str(carpeta), "sin_script": True,
                                  "error": "sin script: el modelo no respondió en este intento",
                                  "returncode": None, "archivos": [], "violaciones": []}}
        huella = ejecutor.huella(estado["codigo"])
        if huella in (estado.get("rechazadas") or {}).get(s["id"], []):      # freno: repetición
            self.traza.decision("ejecutor", "script_repetido",
                                f"idéntico ({huella}) a uno ya rechazado: no se ejecuta", subtarea=s["id"])
            return {"ejecucion": {"huella": huella, "carpeta": str(carpeta), "repetido": True,
                                  "error": "script idéntico a uno ya rechazado: no se ejecutó",
                                  "returncode": None, "archivos": [], "violaciones": []}}
        entradas = {d: Path(self._subtarea(estado, d)["carpeta"]) for d in s.get("depende_de") or []
                    if self._subtarea(estado, d).get("carpeta")}
        r = ejecutor.ejecutar(estado["codigo"], carpeta, entradas=entradas,
                              datos=_cargar(estado["documento"]).get("datos"),
                              timeout_s=self.cfg.timeout_s, confirmar=self.cfg.confirmar, subtarea=s["id"],
                              tmp=self._salida(estado) / INTERNO / "tmp" / s["id"])
        if r.get("confirmacion"):
            self.traza.decision("ejecutor", "confirmacion_red", r["confirmacion"]["motivo"], subtarea=s["id"],
                                aprobado=r["confirmacion"]["aprobado"])
        self.traza.ejecucion("ejecutor", subtarea=s["id"], intento=s["intentos"], returncode=r["returncode"],
                             duracion_s=r["duracion_s"], archivos=r["archivos"], error=r["error"],
                             huella=huella, carpeta=str(carpeta), violaciones=r["violaciones"],
                             timeout=r["timeout"], stderr=r["stderr"][-2000:])
        return {"ejecucion": r}

    def _criticar(self, estado: Estado) -> dict:
        s = self._subtarea(estado, estado["actual"])
        r = estado["ejecucion"]
        if r.get("sin_script"):                     # ni la ablación aprueba un script que no existe
            v = {"aprobado": False, "motivos": [r["error"]], "correccion": "- " + r["error"], "resultados": None}
        elif self.cfg.usar_critico:
            v = critico.comprobar(s, estado["codigo"], r)
            if v["aprobado"]:                       # el código no objetó: ahora opina el LLM
                v = self._critica_llm(estado, s, r, v)
        else:                                   # ablación: un intento y ninguna comprobación
            ruta = Path(r["carpeta"]) / "resultados.json"
            try:
                res = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
            except json.JSONDecodeError:
                res = None
            v = {"aprobado": True, "motivos": [], "correccion": "", "resultados": res}
        if v["aprobado"]:
            self.traza.decision("critico", "aprobado", subtarea=s["id"], intento=s["intentos"])
            return {"plan": self._con(estado, s["id"], status="aprobada", carpeta=r["carpeta"],
                                      resultados=v["resultados"], correccion="")}
        rechazadas = {k: list(v_) for k, v_ in (estado.get("rechazadas") or {}).items()}
        if r.get("huella"):
            rechazadas.setdefault(s["id"], []).append(r["huella"])
        movido = None
        if Path(r["carpeta"]).exists():        # lo rechazado no respalda cifras (procedencia)
            movido = self._salida(estado) / INTERNO / "rechazados" / s["id"] / Path(r["carpeta"]).name
            movido.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(r["carpeta"], movido)
        self.traza.decision("critico", "rechazado", v["correccion"], subtarea=s["id"], intento=s["intentos"],
                            movido_a=str(movido) if movido else None,
                            por="llm" if "descartados" in v else "codigo")
        if s["intentos"] >= self.cfg.max_intentos:          # freno: tope de intentos
            self.traza.decision("orquestador", "tope_intentos",
                                f"{s['intentos']} intentos: la subtarea queda fallida y la cola sigue",
                                subtarea=s["id"])
            return {"rechazadas": rechazadas,
                    "plan": self._con(estado, s["id"], status="fallida", correccion=v["correccion"])}
        return {"rechazadas": rechazadas, "plan": self._con(estado, s["id"], correccion=v["correccion"])}

    def _critica_llm(self, estado: Estado, s: dict, r: dict, v_codigo: dict) -> dict:
        """El LLM revisa contra el enunciado completo. Si no responde, vale la decisión del
        código (y queda en la traza): un crítico caído no debe detener la cola."""
        try:
            v = critico.revisar_con_llm(self.llm, _cargar(estado["documento"])["texto"], s,
                                        estado["codigo"], r, v_codigo["resultados"])
        except PresupuestoAgotado:
            raise
        except Exception as err:
            self.traza.decision("critico", "llm_no_disponible", f"{type(err).__name__}: {err}; "
                                "vale la decisión del código", subtarea=s["id"])
            return v_codigo
        if v["descartados"]:
            self.traza.decision("critico", "rechazo_descartado",
                                "el LLM citó frases que no están en el enunciado: "
                                + "; ".join(str(p.get("cita"))[:80] for p in v["descartados"]),
                                subtarea=s["id"])
        if v["aprobado"]:
            return v_codigo
        return {**v, "resultados": None}

    def _notas(self, estado: Estado) -> list[str]:
        notas = []
        for s in estado.get("plan") or []:
            # «pendiente»: la cola no llegó a ella (presupuesto o error). El enunciado exige que el
            # entregable diga QUÉ no se hizo, no solo que algo faltó.
            if s["status"] in {"fallida", "omitida", "pendiente"} and s.get("tipo") == "calculo":
                notas.append(f"La subtarea {s['id']} ({', '.join(s.get('secciones') or [])}) no tiene "
                             f"resultado ({s['status']}): no se reportan sus cifras.")
        if estado.get("motivo_parada") == "presupuesto":
            notas.append("El presupuesto de tokens se agotó: se entrega lo que se alcanzó a medir.")
        elif (m := estado.get("motivo_parada")):
            notas.append(f"La corrida se detuvo: {m}.")
        return notas

    def _redactar(self, estado: Estado) -> dict:
        if not estado.get("documento"):
            return {}
        doc = _cargar(estado["documento"])
        plan = estado.get("plan") or []
        aprobadas = {s["id"]: Path(s["carpeta"]) for s in plan if s.get("status") == "aprobada" and s.get("carpeta")}
        figuras = redactor.copiar_figuras(aprobadas, self._salida(estado))
        publicados = self._publicar_exigidos(estado, doc, plan, aprobadas)
        texto = redactor.redactar(self.llm, doc, plan, figuras, estado.get("problemas_redaccion"),
                                  estado.get("borrador"), self._notas(estado))
        salida, r = self._salida(estado), doc["restricciones"]
        nombre, fmt = Path(r["entregable"]).name, r.get("formato", "md")
        if fmt == "pdf":
            (salida / INTERNO).mkdir(parents=True, exist_ok=True)
            (salida / INTERNO / "borrador.md").write_text(texto, encoding="utf-8")
            destino = salida / nombre
            paginas = formatos.md_a_pdf(texto, destino, salida)
            self.traza.decision("redactor", "pdf_generado", f"{paginas} páginas", paginas=paginas)
        elif fmt == "ipynb":
            # En la RAÍZ de la corrida: el golden busca «*.ipynb» sin recursión (B01 falló el
            # 2026-10-04 con el notebook en notebook/). Las celdas leen entrada/<id>/ desde aquí.
            carpeta = salida
            formatos.preparar_carpeta(carpeta, aprobadas, doc.get("datos"))
            scripts = {sid: (c / "script.py").read_text(encoding="utf-8") for sid, c in aprobadas.items()
                       if (c / "script.py").exists()}
            destino = carpeta / nombre
            destino.write_text(json.dumps(formatos.construir_notebook(texto, plan, scripts), ensure_ascii=False,
                                          indent=1), encoding="utf-8")
            ej = formatos.ejecutar_notebook(destino, max(self.cfg.timeout_s, 120),
                                            salida / INTERNO / "tmp" / "notebook")
            self.traza.ejecucion("ejecutor", subtarea="notebook", intento=estado.get("intentos_redaccion", 0) + 1,
                                 returncode=ej["returncode"], duracion_s=ej["duracion_s"], archivos=[nombre],
                                 error="; ".join(ej["errores"]) or None, stderr=ej["stderr"][-1500:])
        else:
            destino = salida / (nombre if nombre.endswith(".md") else Path(nombre).stem + ".md")
            destino.write_text(texto, encoding="utf-8")
        return {"borrador": texto, "entregables": [str(destino)] + publicados,
                "intentos_redaccion": estado.get("intentos_redaccion", 0) + 1}

    def _publicar_exigidos(self, estado: Estado, doc: dict, plan: list[dict],
                           aprobadas: dict[str, Path]) -> list[str]:
        """Los archivos que el enunciado exige con su ruta (output/x.parquet) se copian, desde la
        ejecución APROBADA que los escribió, a esa ruta en la raíz de la corrida. Antes quedaban
        dentro de subtareas/ y el entregable estaba incompleto (S102, S202, S203 el 2026-10-05).
        Nunca se fabrican: si ninguna ejecución aprobada los escribió, la traza lo dice."""
        salida, publicados, faltan = self._salida(estado), [], []
        orden = [s["id"] for s in plan if s["id"] in aprobadas]
        for rel in doc["restricciones"].get("archivos") or []:
            origen = None
            for sid in reversed(orden):                          # la última subtarea que lo escribió
                carpeta = aprobadas[sid]
                exacto = carpeta / rel
                if exacto.is_file():
                    origen = exacto
                    break
                por_nombre = [f for f in carpeta.rglob(Path(rel).name)
                              if f.is_file() and "entrada" not in f.relative_to(carpeta).parts]
                if por_nombre:
                    origen = por_nombre[0]
                    break
            if origen is None:
                faltan.append(rel)
                continue
            destino = salida / rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origen, destino)
            publicados.append(str(destino))
        if publicados or faltan:
            self.traza.decision("redactor", "archivos_exigidos",
                                f"publicados: {[str(Path(p).relative_to(salida)) for p in publicados]}; "
                                f"ninguna ejecución aprobada escribió: {faltan}", faltan=faltan)
        return publicados

    def _verificar(self, estado: Estado) -> dict:
        doc = _cargar(estado["documento"])
        plan = estado.get("plan") or []
        aprobadas = [Path(s["carpeta"]) for s in plan if s.get("status") == "aprobada" and s.get("carpeta")]
        r = dict(doc["restricciones"])
        fmt, entregable = r.get("formato", "md"), Path(estado["entregables"][0])
        if fmt == "ipynb" and not r.get("secciones"):     # una celda Markdown que encabece cada parte
            r["secciones"] = [s["clave"] for s in doc["secciones"] if s["trabajo"]]
        problemas = redactor.comprobar_forma(estado["borrador"], r, self._salida(estado))
        if fmt == "pdf" and r.get("paginas_max"):
            import pymupdf
            with pymupdf.open(entregable) as pdf:
                if pdf.page_count > r["paginas_max"]:
                    problemas.append(f"el PDF tiene {pdf.page_count} páginas y el máximo es {r['paginas_max']}: "
                                     "recorta texto, une párrafos o quita una figura")
        if fmt == "ipynb":
            if errores := formatos.errores_notebook(entregable):
                problemas.append(f"el notebook no se ejecutó limpio: {errores[:3]}")
            aprobadas = aprobadas + [entregable.parent]      # sus salidas de celda también son ejecución
        proc = verificar_procedencia(estado["borrador"], doc["texto"], aprobadas)
        if proc["sin_origen"]:
            problemas.append("estas cifras no salen de ninguna ejecución aprobada ni del enunciado; "
                             f"quítalas o usa las medidas: {proc['sin_origen']}")
        self.traza.decision("procedencia", "publicable" if not problemas else "devuelto_al_redactor",
                            "; ".join(problemas), fraccion=round(proc["fraccion"], 3), total=proc["total"],
                            intento=estado.get("intentos_redaccion"))
        return {"problemas_redaccion": problemas, "procedencia": proc}

    def _cerrar(self, estado: Estado) -> dict:
        plan = estado.get("plan") or []
        calculo = [s for s in plan if s["tipo"] == "calculo"]
        if not estado.get("entregables"):
            status = "fallido"
        elif all(s["status"] == "aprobada" for s in calculo) and not estado.get("problemas_redaccion") \
                and not estado.get("motivo_parada"):
            status = "completado"
        else:
            status = "parcial"
        motivo = estado.get("motivo_parada") or ("plan inválido" if estado.get("problemas_plan") else "cola vacía")
        self.traza.evento("fin", agente="orquestador", status=status, motivo=motivo,
                          subtareas={s["id"]: s["status"] for s in plan}, **self.traza.tokens_totales)
        return {"status": status, "motivo_parada": estado.get("motivo_parada") or ""}

    # ================================================================ contrato
    def solve(self, ruta_pdf: str, salida: str) -> dict:
        carpeta = Path(salida).resolve()
        entrada = Path(ruta_pdf).resolve()
        if carpeta.exists() and (entrada == carpeta or entrada.is_relative_to(carpeta)
                                 or (entrada.exists() and _misma(entrada, carpeta))):
            # En macOS «tarea-x» y «tarea-X» son la MISMA carpeta: apartarla movería el enunciado.
            raise ValueError(f"la carpeta de salida {carpeta} contiene la entrada {entrada}")
        apartada = None
        if carpeta.exists() and any(carpeta.iterdir()):
            # Nunca mezclar corridas: los artefactos aprobados de la anterior respaldarían cifras
            # de la nueva en la procedencia. La anterior se aparta, no se borra.
            apartada = _libre(carpeta.with_name(f"{carpeta.name}.anterior-{time.strftime('%Y%m%d-%H%M%S')}"))
            carpeta.rename(apartada)
        carpeta.mkdir(parents=True, exist_ok=True)
        traza_ruta = carpeta / "traza.jsonl"
        self.traza = Traza(traza_ruta)
        self.capa2 = None
        if apartada:
            self.traza.decision("orquestador", "corrida_anterior_apartada", str(apartada))
        self.traza.evento("inicio", agente="orquestador", variante=self.variante, entrada=str(ruta_pdf),
                          config={k: v for k, v in vars(self.cfg).items() if k not in {"llm", "confirmar", "embedder"}})
        t0 = time.perf_counter()
        try:
            self.llm = ClienteLLM(self.cfg, self.traza)
        except Exception as err:                        # sin VPN o sin modelo: fallido, con traza
            self.traza.error("orquestador", err)
            return {"status": "fallido", "entregables": [], "subtareas": [],
                    "usage": {"tokens_entrada": 0, "tokens_salida": 0}, "model": "",
                    "trace": str(traza_ruta), "error": f"{type(err).__name__}: {err}"}
        final = self.app.invoke(
            {"ruta_entrada": str(Path(ruta_pdf).resolve()), "carpeta_salida": str(carpeta), "motivo_parada": ""},
            config={"configurable": {"thread_id": f"{self.variante}-{uuid.uuid4().hex[:8]}"},
                    "recursion_limit": 2000})
        return {"status": final.get("status", "fallido"),
                "entregables": final.get("entregables") or [],
                "subtareas": [{"id": s["id"], "tipo": s["tipo"], "status": s["status"],
                               "intentos": s.get("intentos", 0)} for s in final.get("plan") or []],
                "usage": self.traza.tokens_totales, "model": self.llm.modelo, "trace": str(traza_ruta),
                "motivo_parada": final.get("motivo_parada", ""),
                "procedencia": final.get("procedencia"), "duracion_s": round(time.perf_counter() - t0, 1)}

    def run(self, pregunta: str) -> dict:
        """Para el Taller 4: la pregunta es la ruta a un enunciado."""
        salida = RAIZ / "corridas" / "run" / f"{Path(pregunta).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
        r = self.solve(pregunta, str(salida))
        answer = ""
        if r["entregables"]:
            answer = Path(r["entregables"][0]).read_text(encoding="utf-8")
        return {"answer": answer, "trace": r["trace"], "status": r["status"], "model": r["model"],
                "usage": r["usage"]}


class SolverSinGrafo(Solver):
    """Ablación 2.b: el investigador recibe los k fragmentos más similares y nada más."""

    variante = "sin_grafo"

    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        cfg.usar_grafo = False
        super().__init__(cfg)


class SolverSinCritico(Solver):
    """Ablación 2.b: un solo intento por script y ninguna comprobación."""

    variante = "sin_critico"

    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        cfg.usar_critico = False
        cfg.max_intentos = 1
        super().__init__(cfg)
