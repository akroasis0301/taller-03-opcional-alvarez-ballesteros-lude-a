"""Las verdades de las tareas reales (Parte 2.a), calculadas desde sus datos, no escritas a mano.

Cada función reproduce el enunciado adaptado (tareas/reales/semanaN/enunciado.pdf) con los
parámetros que fija, y devuelve un diccionario. El golden set las llama desde `verdad_py`:

    import sys; sys.path.insert(0, 'golden'); from verdades_reales import s2
    verdad = s2()['sd']

Las rutas a los datos se resuelven desde este archivo, así que no dependen de la carpeta en la
que se corra el evaluador; el `sys.path.insert(0, 'golden')` sí: se corre desde la raíz del repo.

    python golden/verdades_reales.py      # imprime todas las verdades
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path

REALES = Path(__file__).resolve().parents[1] / "tareas" / "reales"
PESOS = (0.50, 0.30, 0.20)
CARACTERISTICAS = ["gpu_utilization", "cpu_utilization", "memory_gb"]
SEMILLA_S2 = 20260829
REMUESTRAS_S2 = 10_000


@lru_cache(maxsize=None)
def s1(ddof: int = 0) -> dict:
    """Semana 1, telemetría. El z-score usa la desviación de NumPy (ddof=0): es la que reproduce
    el «Resultado esperado» del enunciado (fila 23 → 3.136218). Con ddof=1 (la `.std()` de pandas)
    sale 3.130986: `s1(ddof=1)` existe para documentar esa trampa en el golden."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(REALES / "semana1" / "data" / "server_measurements.csv", parse_dates=["timestamp"])
    X = df[CARACTERISTICAS].to_numpy(float)
    media, desv = X.mean(axis=0), X.std(axis=0, ddof=ddof)
    z = (X - media) / np.where(desv == 0, 1.0, desv)
    carga = z @ np.array(PESOS)
    a = df.copy()
    a["load_score"] = carga
    a["requires_review"] = (carga > 1.5) | (a["temperature_c"] > 80)
    resumen = (a.groupby("server")
                .agg(observations=("server", "size"), mean_power_w=("power_w", "mean"),
                     max_temperature_c=("temperature_c", "max"), mean_load=("load_score", "mean"),
                     review_count=("requires_review", "sum"))
                .reset_index())
    por_servidor = {r.server: r._asdict() for r in resumen.itertuples(index=False)}
    return {"medias": media.tolist(), "load_score_23": float(carga[23]),
            "revisiones": int(a["requires_review"].sum()), "resumen": por_servidor}


@lru_cache(maxsize=None)
def s2() -> dict:
    """Semana 2, comparación pareada. d = error_a − error_b por tienda."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(REALES / "semana2" / "data" / "model_errors.csv")
    d = (df["error_a"] - df["error_b"]).to_numpy(float)
    n = len(d)
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = stats.ttest_1samp(d, 0.0)
    ic_t = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    # Prueba exacta por cambios de signo: las 2^16 configuraciones.
    observada = abs(d.mean())
    extremas = sum(abs((d * np.array(s)).mean()) >= observada - 1e-12 for s in product((1, -1), repeat=n))
    # Bootstrap pareado: tiendas completas (el vector de diferencias), default_rng con la semilla.
    rng = np.random.default_rng(SEMILLA_S2)
    medias = d[rng.integers(0, n, size=(REMUESTRAS_S2, n))].mean(axis=1)
    ic_boot = np.percentile(medias, [2.5, 97.5])
    return {"n": n, "media": float(d.mean()), "sd": sd, "se": float(se),
            "t": float(t.statistic), "gl": n - 1, "p_t": float(t.pvalue),
            "ic_t": [float(x) for x in ic_t],
            "configuraciones": 2 ** n, "extremas": int(extremas), "p_signos": extremas / 2 ** n,
            "ic_boot": [float(x) for x in ic_boot],
            "favorecen_b": int((d > 0).sum()), "favorecen_a": int((d < 0).sum())}


def s2_trampas() -> dict:
    """Lo que obtiene un solver que viola las reglas de la Tarea 6 (para comentar el golden)."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(REALES / "semana2" / "data" / "model_errors.csv")
    a, b = df["error_a"].to_numpy(float), df["error_b"].to_numpy(float)
    rng = np.random.default_rng(SEMILLA_S2)
    separado = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(REMUESTRAS_S2)]
    np.random.seed(SEMILLA_S2)
    d = a - b
    antiguo = [np.random.choice(d, len(d)).mean() for _ in range(REMUESTRAS_S2)]
    return {"remuestreo_separado": np.percentile(separado, [2.5, 97.5]).tolist(),
            "semilla_global_antigua": np.percentile(antiguo, [2.5, 97.5]).tolist()}


if __name__ == "__main__":
    import json
    print(json.dumps({"s1": s1(), "s1_ddof1": {k: v for k, v in s1(ddof=1).items() if k != "resumen"},
                      "s2": s2(), "s2_trampas": s2_trampas()}, indent=1, default=float, ensure_ascii=False))
