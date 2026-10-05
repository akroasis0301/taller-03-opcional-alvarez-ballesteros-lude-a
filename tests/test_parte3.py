"""Parte 3: los cuatro frenos se disparan con el script que genera la evidencia del informe."""
from __future__ import annotations

import sys

from solver.config import RAIZ


def test_los_cuatro_frenos_se_disparan(tmp_path, monkeypatch):
    sys.path.insert(0, str(RAIZ / "scripts"))
    import forzar_frenos

    monkeypatch.setattr(sys, "argv", ["forzar_frenos.py", "--salida", str(tmp_path)])
    assert forzar_frenos.main() == 0
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    for freno in ["presupuesto", "repeticion", "timeout", "red"]:
        assert f"## {freno} — disparado" in readme
        assert (tmp_path / freno / "traza.jsonl").exists()
    reporte = (tmp_path / "presupuesto" / "reporte.md").read_text(encoding="utf-8")
    assert "T2" in reporte and "presupuesto" in reporte          # dice QUÉ no se hizo
