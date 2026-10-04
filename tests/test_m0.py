"""Pruebas del M0: traza, enmascarado, presupuesto, reintentos y JSON. Sin red: LLMGuion."""
from __future__ import annotations

import json

import pytest

from solver.cliente_llm import ClienteLLM, LLMGuion, LLMVacio, PresupuestoAgotado, extraer_json
from solver.config import Config
from solver.traza import Traza, enmascarar, leer


def cliente(tmp_path, respuestas, **cfg):
    traza = Traza(tmp_path / "traza.jsonl")
    llm = LLMGuion(respuestas)
    return ClienteLLM(Config(llm=llm, **cfg), traza), traza, llm


# ---------------------------------------------------------------- traza
def test_traza_escribe_cada_evento_al_momento(tmp_path):
    t = Traza(tmp_path / "t.jsonl")
    t.decision("critico", "rechazado", "fuga", subtarea="T2")
    filas = leer(tmp_path / "t.jsonl")          # ya está en disco, sin cerrar nada
    assert filas[0]["tipo"] == "decision" and filas[0]["subtarea"] == "T2"


def test_traza_sobrevive_a_una_excepcion(tmp_path):
    t = Traza(tmp_path / "t.jsonl")
    with pytest.raises(ZeroDivisionError):
        t.evento("inicio")
        1 / 0
    assert len(leer(tmp_path / "t.jsonl")) == 1


def test_enmascara_claves(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "valor-secreto-del-entorno-123")
    texto = ("clave sk-proj-abcdefghijklmnopqrstuvwx y sk-ant-api03-zzzzzzzzzzzz "
             "Bearer abcdefghijklmnopqrstu api_key=supersecreta99 valor-secreto-del-entorno-123")
    limpio = enmascarar(texto)
    for secreto in ["sk-proj-abc", "sk-ant-api03", "abcdefghijklmnopqrstu", "supersecreta99",
                    "valor-secreto-del-entorno-123"]:
        assert secreto not in limpio
    assert "api_key=***" in limpio


# ---------------------------------------------------------------- cliente
def test_llamada_queda_en_la_traza_por_agente(tmp_path):
    c, traza, _ = cliente(tmp_path, ["hola"])
    assert c.pedir("planificador", [{"role": "user", "content": "x"}], subtarea="T1") == "hola"
    fila = leer(traza.ruta)[0]
    assert fila["tipo"] == "llamada" and fila["agente"] == "planificador"
    assert fila["tokens_entrada"] == 1000 and fila["tokens_salida"] == 200
    assert traza.tokens_por_agente["planificador"] == {"tokens_entrada": 1000, "tokens_salida": 200}


def test_vacio_por_longitud_se_reintenta_con_mas_tokens(tmp_path):
    c, traza, llm = cliente(tmp_path, [{"contenido": "", "fin": "length"}, "ya"], max_tokens=1000)
    assert c.pedir("programador", [{"role": "user", "content": "x"}]) == "ya"
    assert [l["max_tokens"] for l in llm.llamadas] == [1000, 2000]
    assert leer(traza.ruta)[0]["error"].startswith("contenido vacío")


def test_vacio_persistente_es_excepcion_no_cadena_vacia(tmp_path):
    c, _, _ = cliente(tmp_path, [{"contenido": "", "fin": "length"}])
    with pytest.raises(LLMVacio):
        c.pedir("programador", [{"role": "user", "content": "x"}])


def test_error_de_red_se_reintenta_una_vez_y_se_traza(tmp_path):
    c, traza, _ = cliente(tmp_path, [RuntimeError("sin VPN"), "ok"])
    assert c.pedir("lector", [{"role": "user", "content": "x"}]) == "ok"
    assert "sin VPN" in leer(traza.ruta)[0]["error"]


def test_presupuesto_se_comprueba_antes_y_respeta_la_reserva(tmp_path):
    # cada llamada cuesta 1200; presupuesto 3000 con reserva 1000 → trabajo hasta 2000
    c, traza, llm = cliente(tmp_path, ["r"], presupuesto_tokens=3000, reserva_redactor=1000)
    msg = [{"role": "user", "content": "x"}]
    c.pedir("programador", msg)                     # 1200
    c.pedir("programador", msg)                     # 2400 (se comprobó con 1200 < 2000)
    with pytest.raises(PresupuestoAgotado):
        c.pedir("programador", msg)                 # 2400 ≥ 2000: no llama
    assert len(llm.llamadas) == 2
    assert c.pedir("redactor", msg) == "r"          # el redactor usa la reserva
    assert leer(traza.ruta)[2]["decision"] == "presupuesto_agotado"


def test_pedir_json_corrige_con_el_error(tmp_path):
    c, traza, llm = cliente(tmp_path, ["esto no es json", '```json\n{"ok": true}\n```'])
    assert c.pedir_json("planificador", [{"role": "user", "content": "x"}]) == {"ok": True}
    assert "no es JSON válido" in llm.llamadas[1]["mensajes"][-1]["content"]
    assert any(f.get("decision") == "json_invalido" for f in leer(traza.ruta))


def test_extraer_json_tolera_texto_alrededor():
    assert extraer_json('Aquí va: {"a": [1, 2]} listo') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extraer_json("[1, 2]")


def test_la_traza_es_json_valido_linea_a_linea(tmp_path):
    c, traza, _ = cliente(tmp_path, ["á é ñ"])
    c.pedir("redactor", [{"role": "user", "content": "x"}])
    for linea in traza.ruta.read_text(encoding="utf-8").splitlines():
        json.loads(linea)


def test_quita_el_razonamiento_de_ollama(tmp_path):
    c, _, _ = cliente(tmp_path, ['<think>pienso {"x": 0}</think>\n{"ok": 1}'])
    assert c.pedir_json("planificador", [{"role": "user", "content": "x"}]) == {"ok": 1}


def test_solo_razonamiento_cuenta_como_vacio(tmp_path):
    c, _, _ = cliente(tmp_path, [{"contenido": "<think>mucho</think>", "fin": "length"}])
    with pytest.raises(LLMVacio):
        c.pedir("programador", [{"role": "user", "content": "x"}])
