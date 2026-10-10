# -*- coding: utf-8 -*-
"""
T2 — Parte 2 (Tarea A): dos clasificadores sobre la división de la Parte 1.

Entrena:
  * Naive Bayes gaussiano (modelo generativo) sobre los atributos crudos.
  * Regresión logística (modelo discriminativo) con atributos estandarizados,
    donde el StandardScaler se ajusta SOLO con el conjunto de entrenamiento.

Evalúa ambos sobre el conjunto de prueba (30 %) con accuracy y F1 macro y
presenta los resultados en una tabla (consola, PNG y resultados.json).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
RUTA_T1 = Path("entrada") / "T1"

# ---------------------------------------------------------------------------
# 1) Recuperar la división única de la Parte 1 (70/30 estratificada, rs=42)
# ---------------------------------------------------------------------------
with open(RUTA_T1 / "resultados.json", encoding="utf-8") as f:
    t1 = json.load(f)

feature_names = list(t1["nombres_atributos"])
codificacion = {k: int(v) for k, v in t1["codificacion_clases"].items()}
train_path, test_path = RUTA_T1 / "t1_train.csv", RUTA_T1 / "t1_test.csv"


def _codificar_objetivo(serie):
    """Pasa la columna objetivo a enteros usando la codificación de la T1."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(int).to_numpy()
    s = serie.astype(str).str.strip()
    y = s.map(codificacion)
    if y.isna().any():  # respaldo: orden alfabético de las clases reportadas
        clases = sorted(t1.get("nombres_clases", sorted(s.unique())))
        y = s.map({c: i for i, c in enumerate(clases)})
    return y.astype(int).to_numpy()


def _cargar_desde_t1():
    """Lee los CSV de la Parte 1; devuelve None si no son utilizables."""
    if not (train_path.exists() and test_path.exists()):
        return None
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    extra = [c for c in train_df.columns if c not in feature_names]
    if not extra:
        return None
    if len(extra) == 1:
        target_col = extra[0]
    else:
        claves = ("target", "diagnosis", "class", "clase", "label", "y")
        target_col = next(
            (c for c in extra if any(k in c.lower() for k in claves)),
            train_df.columns[-1],
        )
    X_tr = train_df[feature_names].to_numpy(dtype=float)
    X_te = test_df[feature_names].to_numpy(dtype=float)
    y_tr = _codificar_objetivo(train_df[target_col])
    y_te = _codificar_objetivo(test_df[target_col])
    fuente = f"archivos de la Parte 1: {train_path.as_posix()} y {test_path.as_posix()}"
    return X_tr, X_te, y_tr, y_te, fuente


cargado = _cargar_desde_t1()
if cargado is not None:
    X_train, X_test, y_train, y_test, fuente = cargado
else:
    # Respaldo determinista: regenerar exactamente la misma división de la Parte 1
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3,
        random_state=RANDOM_STATE, stratify=data.target,
    )
    fuente = ("regenerada: train_test_split(test_size=0.3, random_state=42, "
              "stratify=y) sobre load_breast_cancer")

# ---------------------------------------------------------------------------
# 2) Modelo generativo: Naive Bayes gaussiano (atributos crudos)
# ---------------------------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred_gnb = gnb.predict(X_test)

# ---------------------------------------------------------------------------
# 3) Modelo discriminativo: regresión logística con estandarización
#    (el escalador se ajusta SOLO con el conjunto de entrenamiento)
# ---------------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)  # misma transformación, sin reajustar

logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
logreg.fit(X_train_std, y_train)
pred_lr = logreg.predict(X_test_std)

# ---------------------------------------------------------------------------
# 4) Métricas sobre el conjunto de prueba y tabla
# ---------------------------------------------------------------------------
acc_gnb = float(accuracy_score(y_test, pred_gnb))
f1_gnb = float(f1_score(y_test, pred_gnb, average="macro"))
acc_lr = float(accuracy_score(y_test, pred_lr))
f1_lr = float(f1_score(y_test, pred_lr, average="macro"))

tabla = pd.DataFrame(
    {
        "Modelo": ["Naive Bayes gaussiano", "Regresión logística"],
        "Accuracy (prueba)": [acc_gnb, acc_lr],
        "F1 macro (prueba)": [f1_gnb, f1_lr],
    }
)

# ---------------------------------------------------------------------------
# 5) Figura: la tabla de métricas como PNG
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.axis("off")
celdas = [
    ["Naive Bayes gaussiano", f"{acc_gnb:.4f}", f"{f1_gnb:.4f}"],
    ["Regresión logística", f"{acc_lr:.4f}", f"{f1_lr:.4f}"],
]
tbl = ax.table(
    cellText=celdas,
    colLabels=["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
    cellLoc="center",
    bbox=[0.0, 0.05, 1.0, 0.72],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for j in range(3):
    tbl[0, j].set_facecolor("#d9e2f3")
    tbl[0, j].set_text_props(weight="bold")
ax.set_title(
    "Parte 2 — Accuracy y F1 macro en el conjunto de prueba\n"
    f"Breast Cancer · división 70/30 estratificada · random_state={RANDOM_STATE}",
    fontsize=11,
)
plt.savefig("t2_tabla_metricas.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "dataset": t1.get("dataset", "Breast Cancer Wisconsin (Diagnostic)"),
    "division_usada": {
        "origen": fuente,
        "test_size": 0.3,
        "estratificada": True,
        "random_state": RANDOM_STATE,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    },
    "preprocesamiento": {
        "naive_bayes_gaussiano": "atributos crudos (sin estandarizar)",
        "regresion_logistica": "StandardScaler ajustado unicamente con el entrenamiento",
    },
    "metricas_prueba": {
        "naive_bayes_gaussiano": {"accuracy": acc_gnb, "f1_macro": f1_gnb},
        "regresion_logistica": {"accuracy": acc_lr, "f1_macro": f1_lr},
    },
    "tabla": tabla.to_dict(orient="records"),
    "archivos_generados": {"figura_tabla": "t2_tabla_metricas.png"},
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 7) Resumen
# ---------------------------------------------------------------------------
print("=" * 64)
print("T2 - Parte 2: dos clasificadores (generativo vs. discriminativo)")
print("=" * 64)
print(f"Division usada : {fuente}")
print(f"n_train = {len(y_train)} | n_test = {len(y_test)}")
print("-" * 64)
print(tabla.to_string(index=False))
print("-" * 64)
print(f"NB gaussiano   -> accuracy = {acc_gnb:.4f} | F1 macro = {f1_gnb:.4f}")
print(f"Reg. logistica -> accuracy = {acc_lr:.4f} | F1 macro = {f1_lr:.4f}")
print("Salidas: resultados.json, t2_tabla_metricas.png")
