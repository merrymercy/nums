"""
log:
BlockArray::tensordot (11000000, 29) (29,) block shape (76389, 28)
BlockArray::tensordot (29, 11000000) (11000000,) block shape (28, 76389)
BlockArray::tensordot (29, 11000000) (11000000, 29) block shape (28, 76389)
BlockArray::tensordot (29, 29) (29,) block shape (28, 28)
"""
import time

import nums
import nums.numpy as nps
from nums.models.glms import LogisticRegression


t = time.time()
filename = "/root/HIGGS.csv"
higgs_dataset = nums.read_csv(filename)
higgs_dataset.touch()
t = time.time() - t
print("load time %s" % str(t))


t = time.time()
y, X = higgs_dataset[:, 0].astype(nps.int), higgs_dataset[:, 1:]
y.touch()
X.touch()
t = time.time() - t
print("partition time %s" % str(t))


t = time.time()
model = LogisticRegression(solver="newton-cg")
model.fit(X, y)
model._beta.touch()
t = time.time() - t
print("fit time %s" % str(t))

t = time.time()
y_pred = model.predict(X)
print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())
t = time.time() - t
print("predict time %s" % str(t))

