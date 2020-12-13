from nums.core.array.application import ArrayApplication
from nums.core.systems import numpy_compute
from nums.core.systems.systems import SerialSystem, GPUSystem
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.gpu_engine import (
    NumpySerialEngine, CupySerialEngine, NumpyRayEngine, CupyRayEngine,
    TorchCPURayEngine, TorchGPURayEngine, CupyOsActorEngine, CupyNcclActorEngine
)
from nums.core.application_manager import set_instance
from nums.models.glms import LogisticRegression
from nums import numpy as nps

# Set instance
if True:
    num_gpus = 1
    system = GPUSystem(engine=CupySerialEngine(num_gpus))
    system.init()
    app_inst = ArrayApplication(system=system, filesystem=FileSystem(system))
    set_instance(app_inst)

# Make dataset.
N = 1000
d = 28

X1 = nps.random.randn(N // 2, d) + 5.0
y1 = nps.zeros(shape=(N // 2,), dtype=bool)
if False:
    X2 = nps.random.randn(N // 2, d) + 10.0
    y2 = nps.ones(shape=(N // 2,), dtype=bool)
    X = nps.concatenate([X1, X2], axis=0)
    y = nps.concatenate([y1, y2], axis=0)
else:
    X = X1
    y = y1

# Train Logistic Regression Model.
model = LogisticRegression(solver="newton-cg", tol=1e-8, max_iter=1)
model.fit(X, y)
y_pred = model.predict(X)
print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())
