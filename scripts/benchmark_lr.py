import time

import numpy as np
import ray

from nums import numpy as nps
from nums.core.array.application import ArrayApplication
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.gpu_systems import (
    NumpySerialSystem,
    CupySerialSystem,
    NumpyRaySystem,
    CupyRaySystem,
    TorchCPURaySystem,
    TorchGPURaySystem,
    CupyOsActorSystem,
    CupyNcclActorSystem,
    CupyParallelSystem,
)
from nums.models.glms import LogisticRegression

from utils import benchmark_func, get_number_of_gpus


def cupy_used_bytes():
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes()


global app


def forward(X, theta, one):
    Z = X @ theta
    mu = one / (one + app.exp(-Z))
    return mu


def grad(X, y, mu):
    return X.T @ (mu - y)


def hessian(X, one, mu):
    s = mu * (one - mu)
    t = X.T * s
    return t @ X


def update_theta(g, hess, local_theta):
    return local_theta - app.inv(hess) @ g


def one_step_fit(app, X, y):
    theta = app.zeros((X.shape[1],), (X.block_shape[1],), dtype=X.dtype)
    one = app.one
    mu = forward(X, theta, one)
    grad_ = grad(X, y, mu)
    hess_ = hessian(X, one, mu)
    theta = update_theta(grad_, hess_, theta)
    theta.touch()


def one_step_fit_np(np, X, y):
    theta = np.zeros((X.shape[1],), dtype=X.dtype)
    one = 1
    mu = forward(X, theta, one)
    grad_ = grad(X, y, mu)
    hess_ = hessian(X, one, mu)
    theta = update_theta(grad_, hess_, theta)


def benchmark_lr(num_gpus, N_list, system_class_list, d=1000, dtype=np.float32):
    format_string = "%20s,%10s,%10s,%10s"
    print(format_string % ("Library", "N", "Cost", "CV"))
    global app

    for N in N_list:
        N = int(N)
        N_block = N // num_gpus
        d_block = d // 1

        for system_class in system_class_list:
            try:
                if system_class in ["Cupy", "Numpy"]:
                    name = system_class
                    import cupy as cp

                    arr_lib = cp if system_class == "Cupy" else np
                    arr_lib.inv = arr_lib.linalg.inv
                    app = arr_lib

                    X = arr_lib.zeros((N, d), dtype=dtype)
                    y = arr_lib.ones((N,), dtype=dtype)

                    # Prevent the Singular matrix Error in np.linalg.inv
                    arange = arr_lib.arange(N)
                    X[arange, arange % d] = 1
                    cp.cuda.Device(0).synchronize()

                    # Benchmark one step LR
                    def func():
                        tic = time.time()
                        one_step_fit_np(arr_lib, X, y)
                        cp.cuda.Device(0).synchronize()
                        toc = time.time()
                        return toc - tic, None

                    # func()
                    # exit()

                    costs = benchmark_func(func)
                    del (X, y)
                else:
                    # Init system
                    name = system_class.__name__
                    system = system_class(num_gpus)
                    system.init()
                    app = ArrayApplication(system=system, filesystem=FileSystem(system))

                    # Make dataset
                    nps.random.seed(0)
                    X = app.ones((N, d), block_shape=(N_block, d_block), dtype=dtype)
                    y = app.ones((N,), block_shape=(N_block,), dtype=dtype)

                    # Benchmark one step LR
                    def func():
                        tic = time.time()
                        one_step_fit(app, X, y)
                        toc = time.time()
                        return toc - tic, None

                    costs = benchmark_func(func)

                    system.shutdown()
                    del (X, y, app)
            except Exception:
                costs = [-1]

            log_str = format_string % (
                name,
                "%d" % N,
                "%.4f" % np.mean(costs),
                "%.2f" % (np.std(costs) / np.mean(costs)),
            )
            print(log_str)
            with open("result_lr.csv", "a") as f:
                f.write(log_str + "\n")


if __name__ == "__main__":
    num_gpus = get_number_of_gpus()
    ray.init(num_gpus=num_gpus)

    benchmark_lr(
        num_gpus,
        N_list=[
            0.5e6 / 4,
            1e6 / 4,
            2e6 / 4,
            3e6 / 4,
            5e6 / 4,
            10e6 / 4,
            20e6 / 4,
            40e6 / 4,
        ],
        system_class_list=[
            # NumpySerialSystem,
            # CupySerialSystem,
            # NumpyRaySystem,
            # CupyRaySystem,
            # TorchGPURaySystem,
            # CupyOsActorSystem,
            # CupyNcclActorSystem,
            CupyParallelSystem,
            "Cupy",
            # "Numpy",
        ],
    )
