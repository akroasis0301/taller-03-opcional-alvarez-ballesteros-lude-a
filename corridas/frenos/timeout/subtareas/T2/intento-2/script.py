import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
m = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
json.dump({"accuracy": float(m.score(X_te, y_te))}, open("resultados.json", "w"))
print("accuracy", m.score(X_te, y_te))
