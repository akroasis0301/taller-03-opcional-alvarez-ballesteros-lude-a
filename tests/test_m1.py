"""Pruebas del M1: cada agente por separado y el solver de punta a punta con un modelo de guion.

Ninguna usa red ni la H200. La de punta a punta recorre todas las aristas que importan:
plan inválido → corregido, script rechazado por el crítico → corregido, cifra inventada
devuelta al redactor → corregida, y el freno de presupuesto.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from solver.agentes import critico, ejecutor, indexador, lector, planificador
from solver.cliente_llm import LLMGuion
from solver.config import RAIZ, Config
from solver.procedencia import verificar
from solver.sandbox.guarda import revisar
from solver.traza import leer

# ======================================================================== lector
TAREA_A = [  # líneas como las entrega PyMuPDF para la Tarea A (tamaño, negrita)
    ("Tarea A — Generativo contra discriminativo con pocos", 17, True),
    ("datos", 17, True),
    ("MMIA 6013 · Tarea de práctica para el solver del Taller 03 v2 Entrega: un reporte en Markdown", 10.5, False),
    ("(reporte.md), con el código que lo produce. Ng y Jordan mostraron que un clasificador generativo", 10.5, False),
    ("Parte 1 — Datos", 13, True),
    ("Carga el conjunto Breast Cancer Wisconsin que trae scikit-learn y divide 70 % y 30 %.", 10.5, False),
    ("Parte 2 — Dos clasificadores", 13, True),
    ("Entrena un Naive Bayes gaussiano y una regresión logística con los atributos estandarizados.", 10.5, False),
    ("Parte 3 — La curva de aprendizaje", 13, True),
    ("Sobre la misma división de la Parte 1, entrena los dos modelos con el 5 %, 10 % y 100 %.", 10.5, False),
    ("Parte 4 — Discusión", 13, True),
    ("Explica, con tus cifras, si se observa el cruce que predicen Ng y Jordan en cada tamaño.", 10.5, False),
    ("El reporte", 13, True),
    ("Un reporte.md con estas secciones, en este orden: Introducción, Metodología, Resultados", 10.5, False),
    ("(con la tabla de la Parte 2 y la figura de la Parte 3), Discusión y Conclusiones. Máximo 1 200", 10.5, False),
    ("palabras. Toda cifra del reporte tiene que salir de la ejecución del código entregado.", 10.5, False),
]


def lineas(datos):
    return [{"texto": t, "tam": s, "negrita": b, "pagina": 1} for t, s, b in datos]


def test_segmenta_por_tipografia_y_une_titulo_partido():
    secs = lector.segmentar(lineas(TAREA_A))
    assert [s["clave"] for s in secs] == ["Preámbulo", "Parte 1", "Parte 2", "Parte 3", "Parte 4", "El reporte"]
    assert "pocos datos" in secs[0]["titulo"]
    assert [s["trabajo"] for s in secs] == [False, True, True, True, True, False]


def test_restricciones_de_la_tarea_a():
    secs = lector.segmentar(lineas(TAREA_A))
    r = lector.restricciones("\n".join(t for t, _, _ in TAREA_A), secs)
    assert r["entregable"] == "reporte.md" and r["formato"] == "md"
    assert r["palabras_max"] == 1200
    assert r["secciones"] == ["Introducción", "Metodología", "Resultados", "Discusión", "Conclusiones"]


def test_restricciones_de_las_tareas_b_y_c():
    c = ("Entrega: un reporte en PDF (reporte.pdf) de dos páginas como máximo, con el código. "
         "El reporte En PDF, dos páginas como máximo, con las secciones Objetivo, Método, Resultados "
         "(la tabla de la Parte 2 y la de la Parte 3) y Discusión. Toda cifra sale de la ejecución.")
    r = lector.restricciones(c, [])
    assert (r["entregable"], r["paginas_max"]) == ("reporte.pdf", 2)
    assert r["secciones"] == ["Objetivo", "Método", "Resultados", "Discusión"]
    b = lector.restricciones("Entrega: un notebook de Jupyter (.ipynb) ejecutado, con las salidas visibles.", [])
    assert b["formato"] == "ipynb"


def test_ligaduras_y_guiones():
    assert lector.normalizar("clasiﬁcador estratiﬁ-\ncando") == "clasificador estratificando"


def test_lee_el_pdf_real_de_la_tarea_a():
    pytest.importorskip("pymupdf")
    pdf = RAIZ / "solver-v2" / "enunciados" / "tarea-a-generativo-discriminativo.pdf"
    doc = lector.leer(pdf)
    assert [s["clave"] for s in doc["secciones"] if s["trabajo"]] == ["Parte 1", "Parte 2", "Parte 3", "Parte 4"]
    assert doc["restricciones"]["palabras_max"] == 1200
    g = indexador.esqueleto(doc["secciones"])
    assert g.has_edge("Parte 3", "Parte 1")             # la arista de la 0.b, por regla


# ======================================================================== indexador
def test_aristas_depende_de_por_regla():
    g = indexador.esqueleto(lector.segmentar(lineas(TAREA_A)))
    assert g.has_edge("Parte 3", "Parte 1") and not g.has_edge("Parte 1", "Parte 3")
    assert g.has_edge("El reporte", "Parte 2") and g.has_edge("El reporte", "Parte 3")
    assert indexador.dependencias(g, ["Parte 3"]) == ["Parte 1", "Parte 2"]
    assert g.edges["Parte 3", "Parte 2"]["regla"] == "anafora"          # «entrena los dos modelos»


def test_la_pregunta_anterior():
    secs = [{"clave": "Preámbulo", "titulo": "x", "texto": "", "trabajo": False, "id": "S0"},
            {"clave": "Pregunta 1", "titulo": "Pregunta 1", "texto": "a", "trabajo": True, "id": "S1"},
            {"clave": "Pregunta 2", "titulo": "Pregunta 2", "texto": "Con la pregunta anterior…", "trabajo": True, "id": "S2"}]
    assert indexador.esqueleto(secs).has_edge("Pregunta 2", "Pregunta 1")


# ======================================================================== planificador (C1)
def _doc_y_grafo():
    secs = lector.segmentar(lineas(TAREA_A))
    return {"secciones": secs}, indexador.esqueleto(secs)


def _plan(**cambios_t3):
    t3 = {"id": "T3", "tipo": "calculo", "secciones": ["Parte 3"], "depende_de": ["T1", "T2"], "objetivo": "o", "criterio": "c"}
    t3.update(cambios_t3)
    return {"subtareas": [
        {"id": "T1", "tipo": "calculo", "secciones": ["Parte 1"], "depende_de": [], "objetivo": "o", "criterio": "c"},
        {"id": "T2", "tipo": "calculo", "secciones": ["Parte 2"], "depende_de": ["T1"], "objetivo": "o", "criterio": "c"},
        t3,
        {"id": "T4", "tipo": "conceptual", "secciones": ["Parte 4"], "depende_de": ["T2", "T3"], "objetivo": "o", "criterio": "c"}]}


def test_plan_valido():
    doc, g = _doc_y_grafo()
    assert planificador.validar(_plan(), doc, g) == []
    assert [s["id"] for s in planificador.ordenar(_plan())] == ["T1", "T2", "T3", "T4"]


@pytest.mark.parametrize("cambio, esperado", [
    ({"depende_de": []}, "T3 cubre Parte 3, que depende de Parte 1"),      # coherencia con el grafo
    ({"depende_de": ["T9"]}, "depende de T9, que no existe"),
    ({"depende_de": ["T3"]}, "depende de sí misma"),
    ({"depende_de": ["T4"]}, "ciclo"),
    ({"secciones": ["Parte 7"]}, "no existe en el enunciado"),
    ({"tipo": "redaccion"}, "tipo"),
])
def test_plan_invalido(cambio, esperado):
    doc, g = _doc_y_grafo()
    assert any(esperado in p for p in planificador.validar(_plan(**cambio), doc, g))


def test_seccion_sin_cubrir():
    doc, g = _doc_y_grafo()
    plan = _plan()
    plan["subtareas"] = plan["subtareas"][:3]
    assert any("Parte 4" in p for p in planificador.validar(plan, doc, g))


# ======================================================================== guarda y ejecutor (C3)
@pytest.mark.parametrize("codigo", [
    "import socket", "import subprocess", "from urllib.request import urlopen", "import shutil",
    "import os\nos.system('ls')", "import os\nos.remove('a')", "from pathlib import Path\nPath('a').unlink()",
    "eval('1')", "open('/etc/passwd')", "open('../../.env')", "import os\nprint(os.environ)",
    "x = __import__('os')", "def f(:\n  pass"])
def test_guarda_bloquea(codigo):
    assert not revisar(codigo).permitido


def test_guarda_deja_pasar_lo_legitimo():
    codigo = textwrap.dedent("""
        import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        X, y = load_breast_cancer(return_X_y=True)
        df = pd.read_csv('data/x.csv')
        json.dump({'n': len(y)}, open('resultados.json', 'w'))
        plt.savefig('curva.png')""")
    v = revisar(codigo)
    assert v.permitido and not v.descargas


def test_descarga_pide_confirmacion_y_por_defecto_se_niega(tmp_path):
    codigo = "from sklearn.datasets import fetch_openml\nX = fetch_openml('mnist_784')\n"
    assert revisar(codigo).descargas
    r = ejecutor.ejecutar(codigo, tmp_path / "t", confirmar=lambda m: False)
    assert r["returncode"] is None and "no confirmada" in r["error"]


def test_ejecuta_con_entorno_vacio_y_entradas(tmp_path):
    previo = tmp_path / "previo"
    previo.mkdir()
    (previo / "resultados.json").write_text('{"media": 2.5}')
    codigo = textwrap.dedent("""
        import json
        m = json.load(open('entrada/T1/resultados.json'))['media']
        json.dump({'doble': m * 2}, open('resultados.json', 'w'))""")
    r = ejecutor.ejecutar(codigo, tmp_path / "t2", entradas={"T1": previo})
    assert r["returncode"] == 0 and "resultados.json" in r["archivos"]
    assert json.loads((tmp_path / "t2" / "resultados.json").read_text()) == {"doble": 5.0}
    env = ejecutor.entorno_vacio(tmp_path)
    assert not any(k.endswith(("KEY", "TOKEN")) or k.startswith("H200") for k in env)


def test_timeout_mata_el_proceso(tmp_path):
    r = ejecutor.ejecutar("while True:\n    pass\n", tmp_path / "t", timeout_s=1)
    assert r["timeout"] and "timeout" in r["error"]


# ======================================================================== crítico (C4)
FUGA_0C = textwrap.dedent("""
    import json
    from sklearn.datasets import load_breast_cancer
    from sklearn.tree import DecisionTreeClassifier
    X, y = load_breast_cancer(return_X_y=True)
    m = DecisionTreeClassifier(random_state=42).fit(X, y)
    acc = float((m.predict(X) == y).mean())
    json.dump({'accuracy': acc}, open('resultados.json', 'w'))""")


def test_critico_rechaza_la_fuga_de_la_0c(tmp_path):
    r = ejecutor.ejecutar(FUGA_0C, tmp_path / "t")
    v = critico.comprobar({"objetivo": "exactitud", "criterio": "accuracy"}, FUGA_0C, r)
    assert not v["aprobado"]
    assert any("fuga" in m for m in v["motivos"]) and any("plausibilidad" in m for m in v["motivos"])


def test_critico_detecta_escalador_antes_de_dividir():
    codigo = textwrap.dedent("""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        Xs = StandardScaler().fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3)""")
    assert critico.ajuste_antes_de_dividir(codigo)
    bien = textwrap.dedent("""
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)
        Xtr_s = StandardScaler().fit_transform(Xtr)
        sub, _, ysub, _ = train_test_split(Xtr, ytr, train_size=0.1)""")
    assert not critico.ajuste_antes_de_dividir(bien)


def test_critico_exige_figura_contrato_y_cifras_finitas(tmp_path):
    codigo = "import json\njson.dump({'kl': float('nan')}, open('resultados.json', 'w'))\n"
    r = ejecutor.ejecutar(codigo, tmp_path / "t")
    v = critico.comprobar({"objetivo": "dibuja la curva", "criterio": "una figura PNG"}, codigo, r)
    assert any("NaN" in m for m in v["motivos"]) and any("figura" in m for m in v["motivos"])
    sin = ejecutor.ejecutar("print('hola')\n", tmp_path / "t2")
    assert any("resultados.json" in m for m in critico.comprobar({}, "print('hola')", sin)["motivos"])


# ======================================================================== procedencia (C5)
def test_procedencia_con_redondeo_y_enunciado(tmp_path):
    (tmp_path / "resultados.json").write_text('{"acc": 0.947368421, "f1": 0.9428}')
    texto = "NB 0.9474, F1 0.94, inventada 0.9415, del enunciado 0.25, y en código:\n```\n0.5555\n```"
    p = verificar(texto, "con el 25 % y 0.25", [tmp_path])
    assert p["sin_origen"] == ["0.9415"] and p["total"] == 3


# ======================================================================== de punta a punta
ENUNCIADO = textwrap.dedent("""\
    # Tarea X — prueba del solver
    **Entrega:** un reporte en Markdown (`reporte.md`).
    ## Parte 1 — Datos
    Calcula la media de los datos con semilla 0.
    ## Parte 2 — Modelo
    Sobre los mismos datos de la Parte 1, calcula una exactitud y dibuja una figura.
    ## Parte 3 — Discusión
    Explica los resultados.
    ## El reporte
    Un reporte.md con estas secciones, en este orden: Resultados y Discusión. Máximo 300 palabras.
    """)

PLAN_MALO = {"subtareas": [
    {"id": "T1", "tipo": "calculo", "secciones": ["Parte 1"], "depende_de": [], "objetivo": "media", "criterio": "media en resultados.json"},
    {"id": "T2", "tipo": "calculo", "secciones": ["Parte 2"], "depende_de": [], "objetivo": "exactitud y figura", "criterio": "accuracy y una figura PNG"},
    {"id": "T3", "tipo": "conceptual", "secciones": ["Parte 3"], "depende_de": ["T2"], "objetivo": "discusión", "criterio": "argumentada"}]}
PLAN_BUENO = json.loads(json.dumps(PLAN_MALO))
PLAN_BUENO["subtareas"][1]["depende_de"] = ["T1"]

T1 = "import json\njson.dump({'media': 0.1234}, open('resultados.json', 'w'))\nprint('media 0.1234')\n"
T2_FUGA = FUGA_0C
T2_BIEN = textwrap.dedent("""
    import json
    import matplotlib.pyplot as plt
    media = json.load(open('entrada/T1/resultados.json'))['media']
    plt.plot([1, 2], [media, 0.8765]); plt.savefig('figura.png')
    json.dump({'accuracy': 0.8765, 'media_previa': media}, open('resultados.json', 'w'))""")
REPORTE_INVENTADO = "## Resultados\nMedia 0.1234, exactitud 0.8765 y un error de 0.4321.\n\n## Discusión\nBien.\n"
REPORTE_BIEN = "## Resultados\nMedia 0.1234 y exactitud 0.8765.\n\n![figura](figuras/T2_figura.png)\n\n## Discusión\nBien.\n"


def aprueba(usuario):
    return {"aprobado": True, "problemas": []}


CRITICA = aprueba


def guion():
    estado = {"plan": 0, "T2": 0, "redactor": 0}

    def responder(mensajes):
        sistema, usuario = mensajes[0]["content"], mensajes[-1]["content"]
        if "planificador" in sistema:
            estado["plan"] += 1
            return json.dumps(PLAN_MALO if estado["plan"] == 1 else PLAN_BUENO)
        if "programador" in sistema:
            if "SUBTAREA T1" in mensajes[1]["content"]:
                return f"```python\n{T1}```"
            estado["T2"] += 1
            return f"```python\n{T2_FUGA if estado['T2'] == 1 else T2_BIEN}```"
        if "redactor" in sistema:
            estado["redactor"] += 1
            return REPORTE_INVENTADO if estado["redactor"] == 1 else REPORTE_BIEN
        if "crítico" in sistema:
            return json.dumps(CRITICA(usuario))
        raise AssertionError(f"agente inesperado: {sistema[:40]}")
    return responder


@pytest.fixture
def enunciado(tmp_path):
    carpeta = tmp_path / "tarea-x"
    carpeta.mkdir()
    (carpeta / "enunciado.md").write_text(ENUNCIADO, encoding="utf-8")
    return carpeta


def test_de_punta_a_punta(enunciado, tmp_path):
    from solver.orquestador import Solver

    salida = tmp_path / "corrida"
    r = Solver(Config(llm=LLMGuion([guion()]))).solve(str(enunciado), str(salida))
    assert r["status"] == "completado", r
    assert {s["id"]: s["status"] for s in r["subtareas"]} == {"T1": "aprobada", "T2": "aprobada", "T3": "aprobada"}
    assert {s["id"]: s["intentos"] for s in r["subtareas"]}["T2"] == 2
    filas = leer(r["trace"])
    decisiones = [f.get("decision") for f in filas if f["tipo"] == "decision"]
    for d in ["plan_invalido", "plan_valido", "rechazado", "aprobado", "devuelto_al_redactor", "publicable"]:
        assert d in decisiones, d
    assert any(f["tipo"] == "ejecucion" for f in filas) and filas[-1]["tipo"] == "fin"
    reporte = (salida / "reporte.md").read_text()
    assert "0.4321" not in reporte and (salida / "figuras" / "T2_figura.png").exists()
    assert (salida / "plan.json").exists() and (salida / "grafo.json").exists()
    assert (salida / "cache_solver" / "rechazados" / "T2" / "intento-1" / "script.py").exists()
    assert r["usage"]["tokens_entrada"] > 0


def test_freno_de_presupuesto_redacta_con_la_reserva(enunciado, tmp_path):
    from solver.orquestador import Solver

    cfg = Config(llm=LLMGuion([lambda m: json.dumps(PLAN_BUENO) if "planificador" in m[0]["content"]
                                else REPORTE_BIEN if "redactor" in m[0]["content"] else f"```python\n{T1}```"]),
                 presupuesto_tokens=4000, reserva_redactor=2000)
    r = Solver(cfg).solve(str(enunciado), str(tmp_path / "corrida"))
    assert r["status"] == "parcial" and r["motivo_parada"] == "presupuesto"
    assert r["entregables"], "con el presupuesto agotado igual se entrega lo que hay"
    assert any(f.get("decision") == "presupuesto_agotado" for f in leer(r["trace"]))


def test_ablacion_sin_critico_aprueba_la_fuga(enunciado, tmp_path):
    from solver.orquestador import SolverSinCritico

    r = SolverSinCritico(Config(llm=LLMGuion([guion()]))).solve(str(enunciado), str(tmp_path / "corrida"))
    t2 = next(s for s in r["subtareas"] if s["id"] == "T2")
    assert t2["status"] == "aprobada" and t2["intentos"] == 1      # la fuga pasó: eso mide la ablación


def test_sin_modelo_disponible_es_fallido_con_traza(enunciado, tmp_path, monkeypatch):
    from solver import cliente_llm
    from solver.orquestador import Solver

    def sin_vpn(*args, **kwargs):
        raise RuntimeError("Sin respuesta de la H200. ¿Está GlobalProtect conectada?")
    monkeypatch.setattr(cliente_llm, "cargar_h200", sin_vpn)
    r = Solver(Config()).solve(str(enunciado), str(tmp_path / "corrida"))
    assert r["status"] == "fallido" and "GlobalProtect" in r["error"]
    assert any(f["tipo"] == "error" for f in leer(r["trace"]))


# ======================================================================== correcciones tras la corrida real
def test_un_fallo_del_llm_en_una_subtarea_no_detiene_la_cola(enunciado, tmp_path):
    """Corrida del 2026-10-04: un timeout del programador en la T2 detuvo todo. Ahora la T2
    agota sus intentos, queda fallida, y la corrida sigue hasta el redactor."""
    from solver.orquestador import Solver

    def responder(m):
        sistema = m[0]["content"]
        if "planificador" in sistema:
            return json.dumps(PLAN_BUENO)
        if "programador" in sistema:
            if "SUBTAREA T2" in m[1]["content"]:
                raise TimeoutError("timed out")
            return f"```python\n{T1}```"
        return "## Resultados\nMedia 0.1234. La T2 no tiene resultado.\n\n## Discusión\nSin T2.\n"
    r = Solver(Config(llm=LLMGuion([responder]))).solve(str(enunciado), str(tmp_path / "corrida"))
    estados = {s["id"]: (s["status"], s["intentos"]) for s in r["subtareas"]}
    assert estados["T1"] == ("aprobada", 1) and estados["T2"] == ("fallida", 3)
    assert r["status"] == "parcial" and r["motivo_parada"] == "" and r["entregables"]
    filas = leer(r["trace"])
    assert sum(f.get("decision") == "llm_fallo" for f in filas) == 3
    assert any(f.get("decision") == "tope_intentos" for f in filas)
    # un timeout no se reintenta a ciegas: una llamada por intento
    assert sum(f["tipo"] == "llamada" and f.get("subtarea") == "T2" for f in filas) == 3


def test_timeout_no_se_reintenta(tmp_path):
    from solver.cliente_llm import ClienteLLM
    from solver.traza import Traza
    llm = LLMGuion([TimeoutError("timed out"), "ok"])
    c = ClienteLLM(Config(llm=llm), Traza(tmp_path / "t.jsonl"))
    with pytest.raises(TimeoutError):
        c.pedir("programador", [{"role": "user", "content": "x"}])
    assert len(llm.llamadas) == 1


def test_figura_citada_que_no_existe(tmp_path):
    from solver.agentes.redactor import comprobar_forma
    (tmp_path / "figuras").mkdir()
    (tmp_path / "figuras" / "T2_a.png").write_bytes(b"png")
    texto = "![ok](figuras/T2_a.png)\n![inventada](figuras/curvas_aprendizaje.png)"
    problemas = comprobar_forma(texto, {}, tmp_path)
    assert len(problemas) == 1 and "curvas_aprendizaje.png" in problemas[0]


# ======================================================================== M2-A: crítico LLM, anáfora, corridas separadas
def test_anafora_no_duplica_ni_inventa():
    secs = [{"clave": "Preámbulo", "titulo": "x", "texto": "", "trabajo": False, "id": "S0"},
            {"clave": "Parte 1", "titulo": "Parte 1", "texto": "Implementa TF-IDF.", "trabajo": True, "id": "S1"},
            {"clave": "Parte 2", "titulo": "Parte 2", "texto": "Con las mismas consultas y los mismos juicios, usa LSA.",
             "trabajo": True, "id": "S2"},
            {"clave": "Parte 3", "titulo": "Parte 3", "texto": "Evalúa sobre el mismo conjunto de prueba.",
             "trabajo": True, "id": "S3"}]
    g = indexador.esqueleto(secs)
    assert g.has_edge("Parte 2", "Parte 1") and g.edges["Parte 2", "Parte 1"]["regla"] == "anafora"
    assert not g.has_edge("Parte 3", "Parte 2")          # «el mismo conjunto» no es anáfora de una parte


def test_critico_llm_exige_cita_literal(tmp_path):
    from solver.agentes.critico import revisar_con_llm
    from solver.cliente_llm import ClienteLLM
    from solver.traza import Traza
    enunciado = "Entrena una regresión logística (con los atributos estandarizados; el escalador se ajusta solo con el entrenamiento)."
    respuesta = {"aprobado": False, "problemas": [
        {"cita": "con los atributos estandarizados", "problema": "no estandariza", "correccion": "usa StandardScaler"},
        {"cita": "usa siempre 5 pliegues", "problema": "inventado", "correccion": "x"}]}
    c = ClienteLLM(Config(llm=LLMGuion([json.dumps(respuesta)])), Traza(tmp_path / "t.jsonl"))
    v = revisar_con_llm(c, enunciado, {"id": "T3"}, "codigo", {"stdout": ""}, {"acc": 0.9})
    assert not v["aprobado"] and len(v["motivos"]) == 1 and "estandarizados" in v["motivos"][0]
    assert len(v["descartados"]) == 1                    # la cita inventada no rechaza nada
    solo_inventada = {"aprobado": False, "problemas": [respuesta["problemas"][1]]}
    c2 = ClienteLLM(Config(llm=LLMGuion([json.dumps(solo_inventada)])), Traza(tmp_path / "t2.jsonl"))
    assert revisar_con_llm(c2, enunciado, {"id": "T3"}, "codigo", {"stdout": ""}, {})["aprobado"]


def test_critico_llm_rechaza_y_el_programador_corrige(enunciado, tmp_path, monkeypatch):
    """El rechazo que la Parte 1 pide ver en una traza: el código aprueba, el LLM rechaza
    citando el enunciado, el programador corrige y el segundo intento se aprueba."""
    from solver.orquestador import Solver
    vistos = {"T2": 0}

    def critica(usuario):
        if "SUBTAREA T2" in usuario:
            vistos["T2"] += 1
            if vistos["T2"] == 1:
                return {"aprobado": False, "problemas": [{"cita": "calcula una exactitud y dibuja una figura",
                        "problema": "la figura no muestra la exactitud", "correccion": "graficar accuracy"}]}
        return {"aprobado": True, "problemas": []}
    monkeypatch.setattr(sys.modules[__name__], "CRITICA", critica)
    estado_guion = guion()

    def sin_fuga(m):        # T2 sin fuga; el segundo intento cambia (si no, sería un script repetido)
        r = estado_guion(m)
        if "programador" in m[0]["content"] and "SUBTAREA T2" in m[1]["content"]:
            corregido = "rechazó" in m[-1]["content"].lower()
            return f"```python\n{T2_BIEN}{'# graficada la exactitud' if corregido else ''}\n```"
        return r
    r = Solver(Config(llm=LLMGuion([sin_fuga]))).solve(str(enunciado), str(tmp_path / "corrida"))
    filas = leer(r["trace"])
    rechazos = [f for f in filas if f.get("decision") == "rechazado"]
    assert len(rechazos) == 1 and rechazos[0]["por"] == "llm" and "dibuja una figura" in rechazos[0]["motivo"]
    assert {s["id"]: s["intentos"] for s in r["subtareas"]}["T2"] == 2 and r["status"] == "completado"


def test_corrida_anterior_se_aparta_no_se_mezcla(enunciado, tmp_path):
    from solver.orquestador import Solver
    salida = tmp_path / "corrida-X"       # ojo: en macOS «tarea-X» sería la carpeta «tarea-x» del enunciado
    for _ in range(3):                      # tres corridas en el mismo segundo: no chocan
        Solver(Config(llm=LLMGuion([guion()]))).solve(str(enunciado), str(salida))
    apartadas = sorted(tmp_path.glob("corrida-X.anterior-*"))
    assert len(apartadas) == 2 and all((a / "traza.jsonl").exists() for a in apartadas)
    assert any(f.get("decision") == "corrida_anterior_apartada" for f in leer(salida / "traza.jsonl"))
    assert (enunciado / "enunciado.md").exists()


def test_la_salida_no_puede_contener_la_entrada(enunciado):
    from solver.orquestador import Solver
    with pytest.raises(ValueError):
        Solver(Config(llm=LLMGuion([guion()]))).solve(str(enunciado), str(enunciado))
    assert (enunciado / "enunciado.md").exists()


# ======================================================================== M2-B: PDF y notebook
MD_B = """# Tarea B
## Parte 1 — Softmax
La entropía con T=1 es 1.9873 bits.
## Parte 2 — Top-p
Sobreviven t0, t1, t2 y t3.
## Parte 4 — Pregunta conceptual
Con T → 0 la distribución se concentra.
"""


def test_notebook_intercala_codigo_aprobado_bajo_cada_parte():
    from solver.formatos import construir_notebook
    plan = [{"id": "T1", "tipo": "calculo", "secciones": ["Parte 1"]},
            {"id": "T2", "tipo": "calculo", "secciones": ["Parte 2", "Parte 3"]},
            {"id": "T3", "tipo": "calculo", "secciones": ["Parte 9"]},          # sin encabezado: al final
            {"id": "T4", "tipo": "conceptual", "secciones": ["Parte 4"]}]
    nb = construir_notebook(MD_B, plan, {"T1": "print(1)", "T2": "print(2)", "T3": "print(3)"})
    tipos = [(c["cell_type"], c["source"].splitlines()[0]) for c in nb["cells"]]
    assert tipos == [("markdown", "# Tarea B"), ("markdown", "## Parte 1 — Softmax"),
                     ("code", "# T1: código aprobado por el crítico"), ("markdown", "## Parte 2 — Top-p"),
                     ("code", "# T2: código aprobado por el crítico"),
                     ("markdown", "## Parte 4 — Pregunta conceptual"),
                     ("code", "# T3: código aprobado por el crítico")]
    assert nb["nbformat"] == 4


def test_errores_del_notebook_como_los_ve_el_evaluador(tmp_path):
    from solver.formatos import errores_notebook
    nb = {"cells": [{"cell_type": "code", "execution_count": 1, "outputs": [], "source": "x"},
                    {"cell_type": "code", "execution_count": None, "outputs": [], "source": "y"},
                    {"cell_type": "code", "execution_count": 3, "source": "z",
                     "outputs": [{"output_type": "error", "ename": "ValueError", "evalue": "malo"}]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (tmp_path / "n.ipynb").write_text(json.dumps(nb))
    errores = errores_notebook(tmp_path / "n.ipynb")
    assert any("ValueError" in e for e in errores) and any("sin ejecutar" in e for e in errores)


def test_notebook_se_ejecuta_de_verdad(tmp_path):
    pytest.importorskip("nbconvert")
    from solver.formatos import construir_notebook, ejecutar_notebook, errores_notebook, texto_notebook
    nb = construir_notebook("## Parte 1 — x\nTexto.", [{"id": "T1", "tipo": "calculo", "secciones": ["Parte 1"]}],
                            {"T1": "import json\nv = 0.12345\nprint(f'entropia {v}')\n"})
    ruta = tmp_path / "nb" / "solucion.ipynb"
    ruta.parent.mkdir()
    ruta.write_text(json.dumps(nb))
    r = ejecutar_notebook(ruta, 120, tmp_path / "tmp")
    assert r["returncode"] == 0, r["stderr"]
    assert errores_notebook(ruta) == [] and "entropia 0.12345" in texto_notebook(ruta)


def test_pdf_respeta_paginas(tmp_path):
    pytest.importorskip("pymupdf")
    from solver.formatos import md_a_pdf
    corto = "## Objetivo\nUno.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert md_a_pdf(corto, tmp_path / "c.pdf", tmp_path) == 1
    largo = "## Resultados\n" + "\n\n".join("Párrafo largo de prueba. " * 40 for _ in range(30))
    assert md_a_pdf(largo, tmp_path / "l.pdf", tmp_path) > 2


def test_de_punta_a_punta_con_notebook(tmp_path):
    pytest.importorskip("nbconvert")
    from solver.orquestador import Solver
    carpeta = tmp_path / "tarea-b"
    carpeta.mkdir()
    (carpeta / "enunciado.md").write_text(
        "# Tarea B\n**Entrega:** un notebook de Jupyter (`.ipynb`) ejecutado.\n"
        "## Parte 1 — Entropía\nCalcula la entropía en bits.\n## Parte 2 — Pregunta\nExplica la Parte 1.\n",
        encoding="utf-8")
    plan = {"subtareas": [
        {"id": "T1", "tipo": "calculo", "secciones": ["Parte 1"], "depende_de": [], "objetivo": "entropía", "criterio": "entropia"},
        {"id": "T2", "tipo": "conceptual", "secciones": ["Parte 2"], "depende_de": ["T1"], "objetivo": "explicar", "criterio": "c"}]}
    script = "import json\nh = 1.9873\nprint(f'entropia {h}')\njson.dump({'entropia': h}, open('resultados.json', 'w'))\n"

    def responder(m):
        s = m[0]["content"]
        if "planificador" in s:
            return json.dumps(plan)
        if "programador" in s:
            return f"```python\n{script}```"
        if "crítico" in s:
            return '{"aprobado": true, "problemas": []}'
        return "## Parte 1 — Entropía\nLa entropía es 1.9873 bits.\n\n## Parte 2 — Pregunta\nCon la Parte 1.\n"
    r = Solver(Config(llm=LLMGuion([responder]))).solve(str(carpeta), str(tmp_path / "corrida"))
    assert r["status"] == "completado", r
    nb = Path(r["entregables"][0])
    assert nb.suffix == ".ipynb" and nb.parent == (tmp_path / "corrida").resolve()   # en la raíz, como lo busca B01
    from solver.formatos import errores_notebook, texto_notebook
    assert errores_notebook(nb) == [] and "entropia 1.9873" in texto_notebook(nb)


# ======================================================================== tras la exploratoria (2026-10-05)
def test_limite_de_una_respuesta_no_es_el_del_documento():
    """Semana 2: «La respuesta 18 no supera las 150 palabras» se tomó como límite del informe."""
    texto = ("Completar las preguntas 11 a 17 y redactar el informe ejecutivo de máximo 150 palabras de la "
             "pregunta 18. Entregable • informe.md con las 18 preguntas. La respuesta 18 no supera las 150 "
             "palabras. 18. Redacte un máximo de 150 palabras que incluya unidad.")
    r = lector.restricciones(texto, [])
    assert r["palabras_max"] is None and len(r["limites_parciales"]) == 3
    r = lector.restricciones("Un reporte.md con estas secciones, en este orden: Resultados, Discusión y "
                             "Conclusiones. Máximo 1 200 palabras.", [])
    assert r["palabras_max"] == 1200 and r["limites_parciales"] == []


def test_archivos_exigidos_por_el_enunciado():
    r = lector.restricciones("Guardar en output/server_analysis.parquet. Leer data/x.csv. Guardar "
                             "output/store_differences.csv y output/store_differences.png.", [])
    assert r["archivos"] == ["output/server_analysis.parquet", "output/store_differences.csv",
                             "output/store_differences.png"]


ENUNCIADO_CON_ARCHIVO = ENUNCIADO.replace("Calcula la media de los datos con semilla 0.",
                                          "Calcula la media de los datos con semilla 0 y guárdala en output/media.csv.")
T1_CON_ARCHIVO = ("import json\nfrom pathlib import Path\nPath('output').mkdir(exist_ok=True)\n"
                  "Path('output/media.csv').write_text('media\\n0.1234\\n')\n" + T1)


def test_los_archivos_exigidos_se_publican_en_la_raiz(tmp_path):
    from solver.orquestador import Solver

    carpeta = tmp_path / "tarea-x"
    carpeta.mkdir()
    (carpeta / "enunciado.md").write_text(ENUNCIADO_CON_ARCHIVO, encoding="utf-8")
    base = guion()

    def responder(m):
        if "programador" in m[0]["content"] and "SUBTAREA T1" in m[1]["content"]:
            assert "output/media.csv" in m[1]["content"]          # el programador sabe qué se exige
            return f"```python\n{T1_CON_ARCHIVO}```"
        return base(m)
    salida = tmp_path / "corrida"
    r = Solver(Config(llm=LLMGuion([responder]))).solve(str(carpeta), str(salida))
    assert (salida / "output" / "media.csv").read_text() == "media\n0.1234\n"
    assert str(salida / "output" / "media.csv") in r["entregables"]
    d = next(f for f in leer(r["trace"]) if f.get("decision") == "archivos_exigidos")
    assert d["faltan"] == []


def test_un_archivo_exigido_que_nadie_escribio_no_se_fabrica(tmp_path):
    from solver.orquestador import Solver

    carpeta = tmp_path / "tarea-x"
    carpeta.mkdir()
    (carpeta / "enunciado.md").write_text(ENUNCIADO_CON_ARCHIVO, encoding="utf-8")
    salida = tmp_path / "corrida"
    r = Solver(Config(llm=LLMGuion([guion()]))).solve(str(carpeta), str(salida))
    assert not (salida / "output").exists()
    d = next(f for f in leer(r["trace"]) if f.get("decision") == "archivos_exigidos")
    assert d["faltan"] == ["output/media.csv"]


@pytest.mark.parametrize("codigo", ["import pandas as pd\npd.read_csv('https://example.org/datos.csv')",
                                    "import numpy as np\nnp.loadtxt(fname='ftp://example.org/x.txt')"])
def test_una_url_como_argumento_es_una_descarga(codigo):
    v = revisar(codigo)
    assert v.descargas and not v.violaciones          # se detiene hasta que una persona confirme


def test_importlib_esta_bloqueado():
    assert revisar("import importlib\nimportlib.import_module('soc' + 'ket')").violaciones


# ======================================================================== F3 y F4 (completo-r1, 2026-10-05)
def test_procedencia_sugiere_el_valor_medido(tmp_path):
    """Tarea B: el redactor escribió 5.0069 por 5.0069e-05 y 0.6140 por 0.61405 (mal redondeado)."""
    (tmp_path / "resultados.json").write_text(json.dumps({"kl": 5.006898814879229e-05, "sim": 0.6140509840251385,
                                                          "var": 0.3953421709086953}))
    r = verificar("KL 5.0069, similitud 0.6140, varianza 39.53 %, bien: 0.6141 y 5.0069e-05", "", [tmp_path])
    assert r["sin_origen"] == ["0.6140", "5.0069"]                 # 39.53 % vale por su fracción 0.3953
    assert "10^-5" in r["sugerencias"]["5.0069"] and "0.6141" in r["sugerencias"]["0.6140"]


def test_lo_que_el_redactor_no_corrige_no_se_publica(enunciado, tmp_path):
    from solver.orquestador import Solver
    from solver.procedencia import MARCA

    base = guion()

    def responder(m):
        if m[0]["content"].startswith("Eres el redactor"):
            return REPORTE_INVENTADO                        # nunca corrige el 0.4321
        return base(m)
    salida = tmp_path / "corrida"
    r = Solver(Config(llm=LLMGuion([responder]))).solve(str(enunciado), str(salida))
    reporte = (salida / "reporte.md").read_text()
    assert "0.4321" not in reporte and MARCA in reporte and "0.8765" in reporte
    filas = leer(r["trace"])
    assert sum(f.get("decision") == "devuelto_al_redactor" for f in filas) == 3
    assert next(f for f in filas if f.get("decision") == "cifras_marcadas")["cifras"] == ["0.4321"]
    assert r["status"] == "parcial"
