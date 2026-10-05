"""M2c — GraphRAG capa 2: entidades del LLM, fusión con las notas, embeddings y comunidades.

Ninguna usa red: el LLM es un guion y los embeddings son los léxicos (n-gramas de caracteres).
"""
from __future__ import annotations

import dataclasses
import json
import threading
import textwrap
from pathlib import Path

import pytest

from solver import graphrag as G
from solver.cliente_llm import ClienteLLM, LLMGuion
from solver.config import Config
from solver.traza import Traza, leer

from tests.test_m1 import ENUNCIADO, guion

NOTA = textwrap.dedent("""\
    # Notas — Semana 9: evaluación
    ## 1. Evaluación de clasificadores
    ### 1.1 Exactitud
    La exactitud se mide sobre el conjunto de prueba. Medirla sobre entrenamiento es una fuga de datos.
    ```python
    # Esto es un comentario, no un encabezado
    ## tampoco esto
    ```
    ### 1.2 Regresión logística
    Las regresiones logísticas necesitan atributos estandarizados.
    ## 2. Otra sección
    Texto de la otra sección.
    """)


def _notas(tmp_path: Path) -> Path:
    carpeta = tmp_path / "notas"
    carpeta.mkdir()
    (carpeta / "s9-evaluacion.md").write_text(NOTA, encoding="utf-8")
    return carpeta


ENTIDADES_NOTAS = {"entidades": [
    {"nombre": "Exactitud", "tipo": "métrica", "alias": ["accuracy"], "descripcion": "aciertos sobre el total"},
    {"nombre": "Conjunto de prueba", "tipo": "restriccion", "descripcion": "no se usa para ajustar"},
    {"nombre": "Fuga de datos", "tipo": "concepto", "descripcion": "medir sobre lo que se entrenó"},
    {"nombre": "Regresiones logísticas", "tipo": "metodo", "descripcion": "clasificador lineal"},
    {"nombre": "Modelo", "tipo": "concepto", "descripcion": "genérica: se descarta"}],
    "relaciones": [{"origen": "Exactitud", "destino": "Conjunto de prueba", "relacion": "se mide en"},
                   {"origen": "Fuga de datos", "destino": "Exactitud", "relacion": "infla"},
                   {"origen": "Exactitud", "destino": "No existe", "relacion": "x"}]}
ENTIDADES_ENUNCIADO = {"entidades": [
    {"nombre": "accuracy", "tipo": "metrica", "descripcion": "la exactitud que pide la Parte 2"},
    {"nombre": "Semilla 0", "tipo": "restriccion", "descripcion": "reproducibilidad"}],
    "relaciones": [{"origen": "accuracy", "destino": "Semilla 0", "relacion": "depende de"}]}
RESUMEN = "Evaluación honesta de clasificadores: la exactitud se mide en prueba para evitar la fuga."


def con_capa2(base=None):
    """Envuelve un guion: responde la extracción y los resúmenes; lo demás va al guion base."""
    def responder(m):
        sistema, usuario = m[0]["content"], m[-1]["content"]
        if sistema == G.SISTEMA_ENTIDADES:
            return json.dumps(ENTIDADES_NOTAS if "s9-evaluacion" in usuario else ENTIDADES_ENUNCIADO)
        if sistema == G.SISTEMA_COMUNIDAD:
            return RESUMEN
        if base is None:
            raise AssertionError(f"agente inesperado: {sistema[:40]}")
        return base(m)
    return responder


def _cfg(tmp_path, llm, **kw):
    return Config(llm=llm, capa2=True, notas=_notas(tmp_path), cache_graphrag=tmp_path / "cache",
                  embedder=G.EmbedLexico(), **kw)


# =========================================================================== nombres y fusión
def test_normalizar_une_plurales_acentos_y_articulos():
    assert G.normalizar("Las Regresiones Logísticas") == G.normalizar("regresión logística")
    assert G.normalizar("redes") == G.normalizar("red")
    assert "lr" in G.formas({"nombre": "Regresión logística (LR)"})


def test_limpiar_extraccion_descarta_genericas_y_relaciones_sueltas():
    ex = G._limpiar_extraccion(ENTIDADES_NOTAS)
    nombres = [e["nombre"] for e in ex["entidades"]]
    assert "Modelo" not in nombres and len(nombres) == 4
    assert {e["tipo"] for e in ex["entidades"]} <= set(G.TIPOS)        # «métrica» → «metrica»
    assert len(ex["relaciones"]) == 2                                    # «No existe» se descarta
    assert ex["relaciones"][0]["relacion"] == "se_mide_en"


def test_fusion_por_nombre_normalizado_y_alias():
    f = G.Fusion()
    f.agregar({"cita": "s9 § 1.1", **G._limpiar_extraccion(ENTIDADES_NOTAS)}, "notas", fragmento="s9#1")
    f.agregar({"cita": "Parte 2", **G._limpiar_extraccion(ENTIDADES_ENUNCIADO)}, "enunciado")
    exactitud = f.entidades[f.clave_de("accuracy")]
    assert exactitud["nombre"] == "Exactitud"                            # «accuracy» era su alias
    assert exactitud["origen"] == ["notas", "enunciado"] and exactitud["citas"] == ["s9 § 1.1", "Parte 2"]
    assert exactitud["fragmentos"] == ["s9#1"]


def test_fragmentar_ignora_los_numerales_dentro_del_codigo(tmp_path):
    fr = G.fragmentar_notas(_notas(tmp_path))
    assert [f["titulos"] for f in fr] == [["1. Evaluación de clasificadores", "1.1 Exactitud", "1.2 Regresión logística"],
                                          ["2. Otra sección"]]
    assert "# Esto es un comentario" in fr[0]["texto"]
    assert fr[0]["cita"] == "s9-evaluacion § 1. Evaluación de clasificadores … 1.2 Regresión logística"


# =========================================================================== índice de las notas
def test_indice_de_notas_se_construye_una_vez_con_traza_y_presupuesto_propios(tmp_path):
    llm = LLMGuion([con_capa2()])
    cfg = _cfg(tmp_path, llm, presupuesto_tokens=100)          # la tarea casi no tiene presupuesto
    traza = Traza(tmp_path / "corrida" / "traza.jsonl")
    cliente = ClienteLLM(cfg, traza)
    indice = G.indice_notas(cliente, cfg, traza)
    assert len(indice["fragmentos"]) == 2 and indice["comunidades"]
    assert indice["comunidades"][0]["resumen"] == RESUMEN
    assert traza.tokens_totales == {"tokens_entrada": 0, "tokens_salida": 0}   # no se cobra a la tarea
    propia = tmp_path / "cache" / f"notas-{indice['clave']}" / "traza_notas.jsonl"
    assert any(f["tipo"] == "llamada" for f in leer(propia))
    llamadas = len(llm.llamadas)
    G.indice_notas(cliente, cfg, traza)                          # segunda vez: de la caché
    assert len(llm.llamadas) == llamadas
    assert [f.get("cache") for f in leer(traza.ruta) if f.get("decision") == "indice_notas"] == [False, True]


def test_sin_notas_no_falla(tmp_path):
    cfg = Config(llm=LLMGuion(["{}"]), capa2=True, notas=tmp_path / "no-existe", cache_graphrag=tmp_path)
    traza = Traza(tmp_path / "t.jsonl")
    assert G.indice_notas(ClienteLLM(cfg, traza), cfg, traza) is None
    assert leer(traza.ruta)[-1]["decision"] == "sin_notas"


def test_embeddings_caen_al_respaldo_lexico_con_traza(tmp_path):
    class SinOllama:
        modelo = "x"
        MODELO_EMB = "bge-m3"

        def chat(self, *a, **k):
            return {"contenido": "ok"}

        def embed(self, textos):
            raise ConnectionError("11434 no responde")
    traza = Traza(tmp_path / "t.jsonl")
    e = G.elegir_embedder(ClienteLLM(Config(llm=SinOllama()), traza))
    assert isinstance(e, G.EmbedLexico)
    assert [f["decision"] for f in leer(traza.ruta)] == ["embeddings_no_disponibles", "embeddings_lexicos"]
    v = e.embed(["regresión logística", "regresiones logísticas", "temperatura del muestreo"])
    assert v[0] @ v[1] > v[0] @ v[2]


def test_traza_concurrente_no_pierde_ni_repite_eventos(tmp_path):
    traza = Traza(tmp_path / "t.jsonl")
    hilos = [threading.Thread(target=lambda: [traza.llamada("indexador", modelo="m", tokens_entrada=1,
                                                            tokens_salida=1, latencia_s=0, fin="stop")
                                              for _ in range(50)]) for _ in range(8)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    filas = leer(traza.ruta)
    assert sorted(f["seq"] for f in filas) == list(range(1, 401))
    assert traza.tokens_totales == {"tokens_entrada": 400, "tokens_salida": 400}


# =========================================================================== de punta a punta
@pytest.fixture
def enunciado(tmp_path):
    carpeta = tmp_path / "tarea-x"
    carpeta.mkdir()
    (carpeta / "enunciado.md").write_text(ENUNCIADO, encoding="utf-8")
    return carpeta


def test_capa2_de_punta_a_punta(enunciado, tmp_path):
    from solver.orquestador import Solver

    salida = tmp_path / "corrida"
    r = Solver(_cfg(tmp_path, LLMGuion([con_capa2(guion())]))).solve(str(enunciado), str(salida))
    assert r["status"] == "completado", r
    filas = leer(r["trace"])
    capa2 = next(f for f in filas if f.get("decision") == "capa2")
    assert capa2["fusionadas_con_notas"] >= 1 and "Exactitud" in capa2["ejemplos_fusion"]
    # el investigador SUMA la capa 2: la sección literal sigue y llegan las notas con su cita
    ctx = (salida / "cache_solver" / "contextos" / "T2.md").read_text()
    assert "Sobre los mismos datos de la Parte 1" in ctx
    assert "[notas del curso — s9-evaluacion § 1. Evaluación de clasificadores" in ctx
    assert "búsqueda global: comunidad 0" in ctx
    inv = next(f for f in filas if f.get("decision") == "grafo+capa2" and f["subtarea"] == "T2")
    assert "Exactitud" in inv["semillas"] and "comunidad 0" in inv["motivo"]
    g = json.loads((salida / "grafo.json").read_text())
    tipos = {a["tipo"] for a in g["aristas"]}
    assert {"depende_de", "menciona", "relacion"} <= tipos
    # la extracción del enunciado sí se cobra a la tarea; el índice de las notas no
    del_indexador = [f for f in filas if f["tipo"] == "llamada" and f["agente"] == "indexador"]
    assert len(del_indexador) == 3        # una por sección de TRABAJO (Partes 1-3): las únicas que anclan
    pytest.importorskip("matplotlib")
    assert (salida / "grafo_entidades.png").exists()


def test_sin_grafo_no_corre_la_capa2(enunciado, tmp_path):
    from solver.orquestador import SolverSinGrafo

    r = SolverSinGrafo(_cfg(tmp_path, LLMGuion([con_capa2(guion())]))).solve(str(enunciado), str(tmp_path / "c"))
    filas = leer(r["trace"])
    assert not any(f["tipo"] == "llamada" and f["agente"] == "indexador" for f in filas)
    assert not any(f.get("decision", "").startswith("capa2") for f in filas)


def test_si_la_capa2_falla_sigue_con_el_esqueleto(enunciado, tmp_path):
    from solver.orquestador import Solver

    class Rompe(G.EmbedLexico):
        nombre = "rompe"
        n = 0

        def embed(self, textos):
            Rompe.n += 1
            if Rompe.n > 1:                       # pasa la prueba de elegir_embedder y luego falla
                raise RuntimeError("embeddings caídos a mitad de corrida")
            return super().embed(textos)
    cfg = _cfg(tmp_path, LLMGuion([con_capa2(guion())]))
    cfg.embedder = Rompe()
    r = Solver(cfg).solve(str(enunciado), str(tmp_path / "corrida"))
    assert r["status"] == "completado"
    filas = leer(r["trace"])
    assert any(f.get("decision") == "capa2_no_disponible" for f in filas)
    assert not any(f.get("decision") == "grafo+capa2" for f in filas)


# =========================================================================== F1 y F2 (tras r2, 2026-10-04)
def test_vacio_por_longitud_en_el_tope_no_se_repite(tmp_path):
    """r2, Tarea C: repetir una llamada que ya razonó hasta el tope cobró otros 32 768 tokens."""
    from solver.cliente_llm import LLMVacio

    llm = LLMGuion([{"contenido": "", "fin": "length"}])
    traza = Traza(tmp_path / "t.jsonl")
    c = ClienteLLM(Config(llm=llm, max_tokens=1000, max_tokens_tope=2000), traza)
    with pytest.raises(LLMVacio):
        c.chat("programador", [{"role": "user", "content": "x"}])
    assert [ll["max_tokens"] for ll in llm.llamadas] == [1000, 2000]        # duplica una vez y para
    assert leer(traza.ruta)[-1]["decision"] == "tope_de_razonamiento"


def test_el_indexador_usa_su_propio_tope_de_tokens(tmp_path):
    llm = LLMGuion([{"contenido": "", "fin": "length"}])
    traza = Traza(tmp_path / "t.jsonl")
    c = ClienteLLM(Config(llm=llm, max_tokens_indexador=4096), traza)
    assert G.extraer_en_paralelo(c, [("texto", "Parte 1")], hilos=1) == []   # se omite la sección
    assert [ll["max_tokens"] for ll in llm.llamadas] == [4096]               # sin duplicar
    assert leer(traza.ruta)[-1]["decision"] == "extraccion_fallida"


def _indice_de_prueba(tmp_path):
    llm = LLMGuion([con_capa2()])
    cfg = _cfg(tmp_path, llm)
    traza = Traza(tmp_path / "t.jsonl")
    cliente = ClienteLLM(cfg, traza)
    notas = G.indice_notas(cliente, cfg, traza)
    doc = {"secciones": [
        {"clave": "Parte 1", "titulo": "Parte 1 — Datos", "texto": "Carga los datos."},
        {"clave": "Parte 2", "titulo": "Parte 2 — Modelo", "texto": "Mide la exactitud."}]}
    sin_entidades = LLMGuion([lambda m: json.dumps({"entidades": [], "relaciones": []})
                              if "[Parte 1]" in m[-1]["content"] else json.dumps(ENTIDADES_ENUNCIADO)])
    cliente2 = ClienteLLM(dataclasses.replace(cfg, llm=sin_entidades), traza)
    return G.indexar_tarea(cliente2, cfg, doc, traza, notas)


def test_sin_anclas_no_se_agregan_notas(tmp_path):
    """La T1 de A en r2 recibió «§ 6.3 GraphRAG» para cargar breast_cancer: ninguna entidad
    de su sección estaba en las notas. Ahora no recibe fragmentos y lo dice."""
    indice = _indice_de_prueba(tmp_path)
    r = G.busqueda_local(indice, {"objetivo": "cargar los datos"}, ["Parte 1"], ["Parte 1 — Datos"])
    assert r["anclas"] == [] and r["citas"] == []
    assert "ninguna entidad de esta subtarea aparece en las notas" in r["texto"]
    # con la dependencia (Parte 2 menciona «accuracy» = «Exactitud» de las notas) sí hay notas
    r = G.busqueda_local(indice, {"objetivo": "cargar los datos"}, ["Parte 1"], ["Parte 1 — Datos"],
                         dependencias=["Parte 2"])
    assert r["anclas"] == ["Exactitud"] and any(c.startswith("s9-evaluacion") for c in r["citas"])


def test_una_entidad_en_demasiados_fragmentos_no_ancla(tmp_path):
    indice = _indice_de_prueba(tmp_path)
    clave = indice.fusion.clave_de("accuracy")
    indice.fusion.entidades[clave]["fragmentos"] = [f"x#{i}" for i in range(G.MAX_FRAGMENTOS_ANCLA + 1)]
    assert G.anclas(indice, ["Parte 2"]) == []
