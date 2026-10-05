import json
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
m = DecisionTreeClassifier(random_state=0).fit(X, y)
json.dump({"accuracy": float(m.score(X, y))}, open("resultados.json", "w"))
