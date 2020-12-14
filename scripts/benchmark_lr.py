import time

import numpy as np
import ray

from nums import numpy as nps
from nums.core.array.application import ArrayApplication
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.gpu_systems import (
    NumpySerialSystem, CupySerialSystem, NumpyRaySystem, CupyRaySystem,
    TorchCPURaySystem, TorchGPURaySystem, CupyOsActorSystem, CupyNcclActorSystem
)
from nums.core.application_manager import set_instance
from nums.models.glms import LogisticRegression

from utils import benchmark_func, get_number_of_gpus

def benchmark_lr(num_gpus, system_class_list):
    N = 10000
    d = 1000

    for system_class in system_class_list:
        # Init system
        system = system_class(num_gpus)
        system.init()
        app_inst = ArrayApplication(system=system, filesystem=FileSystem(system))
        set_instance(app_inst)

        # Make dataset
        nps.random.seed(0)
        X = nps.random.randn(N, d)
        y = nps.zeros(shape=(N,), dtype=bool)

        # Train Logistic Regression Model.
        model = LogisticRegression(solver="newton-cg", tol=1e-8, max_iter=1)
        def func():
            tic = time.time()
            model.fit(X, y)
            toc = time.time()

            return toc - tic, None

        costs = benchmark_func(func)
        
        y_pred = model.predict(X)
        print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())

        del (X, y, y_pred, model)

        print(
            "Lib: %s\tCost: %.4f  (CV: %.2f)"
            % (system_class.__name__ if system_class else "None",
              np.mean(costs), np.std(costs) / np.mean(costs))
        )


if __name__ == "__main__":
    num_gpus = get_number_of_gpus()
    ray.init(num_gpus=num_gpus)

    benchmark_lr(num_gpus, [
        # NumpySerialSystem,
        CupySerialSystem,
        # NumpyRaySystem,
        # CupyRaySystem,
        # TorchGPURaySystem,
        CupyOsActorSystem,
        # CupyNcclActorSystem,
    ])

