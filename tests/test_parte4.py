"""Parte 4, opción E: el juez (sin red: el Ollama se simula)."""
from __future__ import annotations

import csv
import json
import sys

from solver.config import RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))


def test_kappa():
    import juez

    assert juez.kappa([(1, 1), (0, 0), (1, 1), (0, 0)]) == 1.0
    assert juez.kappa([(1, 0), (0, 1)]) == -1.0
    assert abs(juez.kappa([(1, 1), (1, 0), (0, 0), (0, 0)]) - 0.5) < 1e-9
    assert juez.kappa([]) is None


def test_juez_contra_el_golden(tmp_path, monkeypatch):
    import juez

    (tmp_path / "enunciado.md").write_text("Reporta la media. Máximo 50 palabras.", encoding="utf-8")
    golden = {"tareas": [{"id": "X", "pdf": "enunciado.md", "entregable": "reporte.md", "checks": [
        {"id": "X1", "tipo": "palabras_max", "valor": 50, "juez": "¿Tiene como máximo 50 palabras?"},
        {"id": "X2", "tipo": "cifra_presente", "verdad_py": "verdad = 0.5", "juez": "¿La media es correcta?"},
        {"id": "X3", "tipo": "archivo", "patron": "reporte.md"}]}]}
    (tmp_path / "golden.json").write_text(json.dumps(golden), encoding="utf-8")
    (tmp_path / "corrida" / "tarea-X").mkdir(parents=True)
    (tmp_path / "corrida" / "tarea-X" / "reporte.md").write_text("La media es 0.7.", encoding="utf-8")
    with (tmp_path / "res.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tarea", "check", "tipo", "ok", "detalle"])
        w.writeheader()
        w.writerows([{"tarea": "X", "check": "X1", "tipo": "palabras_max", "ok": "1", "detalle": "4 palabras"},
                     {"tarea": "X", "check": "X2", "tipo": "cifra_presente", "ok": "0", "detalle": "no hallada"},
                     {"tarea": "X", "check": "X3", "tipo": "archivo", "ok": "1", "detalle": "1"}])
    llamadas = []

    def falso(ruta, cuerpo=None, timeout=600):
        if ruta == "api/tags":
            return {"models": [{"name": "gemma3:27b"}, {"name": "bge-m3:latest"}]}
        llamadas.append(cuerpo)
        assert "0.5" not in cuerpo["messages"][1]["content"]          # el juez NO ve la verdad
        return {"choices": [{"message": {"content": json.dumps({"veredictos": {
            "X1": {"ok": True, "razon": "cuatro palabras"}, "X2": {"ok": True, "razon": "parece bien"}}})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    monkeypatch.setattr(juez, "_pedir", falso)
    monkeypatch.setattr(sys, "argv", ["juez.py", "--golden", str(tmp_path / "golden.json"), "--corrida", "v",
                                      f"{tmp_path / 'corrida'}:{tmp_path / 'res.csv'}", "--salida", str(tmp_path / "out")])
    assert juez.main() == 0
    assert llamadas[0]["model"] == "gemma3:27b"
    filas = list(csv.DictReader((tmp_path / "out" / "juez.csv").open()))
    assert [(f["check"], f["golden"], f["juez"]) for f in filas] == [("X1", "1", "1"), ("X2", "0", "1")]
    md = (tmp_path / "out" / "juez.md").read_text()
    assert "falso positivo del juez" in md and "X2" in md
