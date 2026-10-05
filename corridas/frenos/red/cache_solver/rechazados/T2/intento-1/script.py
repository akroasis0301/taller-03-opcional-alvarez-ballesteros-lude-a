import json
from sklearn.datasets import fetch_openml
X, y = fetch_openml("breast-w", version=1, return_X_y=True, as_frame=False)
json.dump({"n": int(X.shape[0])}, open("resultados.json", "w"))
