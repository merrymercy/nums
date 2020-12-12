from nums import numpy as nps
from nums.models.glms import LogisticRegression

N = 1000
d = 28

# Make dataset.
X1 = nps.random.randn(N // 2, d) + 5.0
y1 = nps.zeros(shape=(N // 2,), dtype=bool)
X2 = nps.random.randn(N // 2, d) + 10.0
y2 = nps.ones(shape=(N // 2,), dtype=bool)
#X = nps.concatenate([X1, X2], axis=0)
#y = nps.concatenate([y1, y2], axis=0)

X = X1
y = y1

# Train Logistic Regression Model.
model = LogisticRegression(solver="newton-cg", tol=1e-8, max_iter=1)
model.fit(X, y)
y_pred = model.predict(X)
print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())
