"""Configuración común de las pruebas.

La capa 2 del GraphRAG llama al LLM para extraer entidades y construye el índice de las notas
en cache/graphrag/. Las pruebas que no son de la capa 2 la apagan (SOLVER_CAPA2=0) para que sus
guiones de LLM sigan siendo exactos; las de tests/test_m2c.py la encienden con Config(capa2=True)
y una caché en tmp_path. Así ninguna prueba escribe en la caché real del repositorio.
"""
import pytest


@pytest.fixture(autouse=True)
def _sin_capa2_por_defecto(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("SOLVER_CAPA2", "0")
    monkeypatch.setenv("SOLVER_CACHE_GRAPHRAG", str(tmp_path_factory.mktemp("cache_graphrag")))
