import json
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
json.dump({"n_ejemplos": int(X.shape[0])}, open("resultados.json", "w"))
print("n_ejemplos", X.shape[0])
