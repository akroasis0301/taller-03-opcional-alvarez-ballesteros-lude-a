import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import json

# Cargar dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target').map({0: 'malignant', 1: 'benign'})

# Estadísticas básicas
total_samples = len(y)
num_features = X.shape[1]
class_distribution = y.value_counts().to_dict()

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Preparar resultados
results = {
    "num_examples": total_samples,
    "num_attributes": num_features,
    "class_distribution": class_distribution,
    "split": {
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
}

# Guardar resultados en JSON
with open("resultados.json", "w") as f:
    json.dump(results, f)

# Imprimir resumen
print(f"Datos cargados: {total_samples} ejemplos, {num_features} atributos")
print(f"Distribución de clases: {class_distribution}")
print(f"División estratificada: {len(X_train)} entrenamiento, {len(X_test)} prueba")
