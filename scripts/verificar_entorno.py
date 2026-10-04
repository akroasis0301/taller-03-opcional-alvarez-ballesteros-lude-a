"""Comprueba que el entorno del solver está listo: Python, dependencias y la H200.

    uv run python scripts/verificar_entorno.py            # todo (necesita VPN)
    uv run python scripts/verificar_entorno.py --sin-red  # solo Python y dependencias
"""
import importlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

ok = True
print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info[:2] == (3, 12) else "-> se esperaba 3.12")
for mod in ["langgraph", "dotenv", "pydantic", "pymupdf", "pypdf", "networkx", "numpy", "pandas",
            "scipy", "sklearn", "matplotlib", "pyarrow", "nbformat", "nbclient", "nbconvert",
            "markdown", "markdown_it", "fpdf"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:12s} {getattr(m, '__version__', getattr(m, 'VersionBind', ''))}")
    except ImportError as e:
        ok = False
        print(f"  {mod:12s} FALTA ({e})")

if "--sin-red" not in sys.argv:
    from solver.cliente_llm import cargar_h200
    try:
        h = cargar_h200()
        print(f"LLM  {h.HOST}:{h.PUERTO} -> {h.modelo}")
        v = h.embed(["prueba"])[0]
        print(f"Emb  {h.HOST}:{h.PUERTO_EMB} -> {h.MODELO_EMB}, dimensión {len(v)}")
    except Exception as e:
        ok = False
        print(f"H200: {e}")

sys.exit(0 if ok else 1)
