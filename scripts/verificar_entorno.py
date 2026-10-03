"""Comprueba que el entorno del solver esta listo: Python, dependencias, .env y servidores."""
import importlib
import os
import sys
from pathlib import Path

ok = True
print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info[:2] == (3, 12) else "-> se esperaba 3.12")

for mod in ["openai", "langgraph", "httpx", "dotenv", "pydantic", "pdfplumber", "pypdf",
            "networkx", "numpy", "pandas", "scipy", "sklearn", "matplotlib",
            "nbformat", "nbclient", "markdown", "fpdf"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:12s} {getattr(m, '__version__', '')}")
    except ImportError as e:
        ok = False
        print(f"  {mod:12s} FALTA ({e})")

raiz = Path(__file__).resolve().parents[1]
if not (raiz / ".env").exists():
    print(".env no existe: copia .env.example a .env")
else:
    from dotenv import load_dotenv
    import httpx
    load_dotenv(raiz / ".env")
    base = os.getenv("OPENAI_BASE_URL", "")
    if base:
        try:
            r = httpx.get(f"{base}/models", timeout=5,
                          headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'EMPTY')}"})
            ids = [m["id"] for m in r.json().get("data", [])]
            print(f"LLM {base}: {ids}")
        except Exception as e:
            print(f"LLM {base}: sin respuesta ({type(e).__name__}). ¿VPN GlobalProtect conectada?")
    emb = os.getenv("EMBED_URL", "")
    if emb:
        try:
            r = httpx.get(f"{emb}/api/tags", timeout=5)
            print(f"Embeddings {emb}: {[m['name'] for m in r.json().get('models', [])]}")
        except Exception as e:
            print(f"Embeddings {emb}: sin respuesta ({type(e).__name__})")

sys.exit(0 if ok else 1)
