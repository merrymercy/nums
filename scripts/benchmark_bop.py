import argparse
import time
import gc

import numpy as np
import ray

import nums
from nums.core.array.application import ArrayApplication
from nums.core.systems import numpy_compute
from nums.core.systems.systems import SerialSystem
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

from utils import benchmark_func, get_number_of_gpus


def benchmark_tensordot(num_gpus, system_class_list):
    # Compute answer with numpy
    if True:
        N, d = 500000, 1000
        N_block = N // 2
        d_block = d // 2
    else:
        N, d = 1000, 1000
        N_block = N // 2
        d_block = d // 2
    dtype = np.float32

    a_np = np.random.uniform(size=(d, N)).astype(dtype)
    b_np = np.random.uniform(size=(N, d)).astype(dtype)
    # c_np = np.tensordot(a_np, b_np, axes=1)
    print("np init array generated")

    # Benchmark engines
    for system_class in system_class_list:
        if system_class:
            system = system_class(num_gpus)
            system.init()
            app = ArrayApplication(system=system, filesystem=FileSystem(system))

            block_a = app.array(a_np, block_shape=(d_block, N_block))
            block_b = app.array(b_np, block_shape=(N_block, d_block))

            def func():
                tic = time.time()
                block_c = block_a.tensordot(block_b, axes=1)
                block_c.touch()
                toc = time.time()
                del block_c
                return toc - tic, None

            costs = benchmark_func(func)

            del (block_a, block_b, app)
            system.shutdown()
        else:
            import cupy as cp

            a_cp = cp.array(a_np)
            b_cp = cp.array(b_np)
            cp.cuda.Device(0).synchronize()

            def func():
                tic = time.time()
                c_cp = cp.tensordot(a_cp, b_cp, axes=1)
                cp.cuda.Device(0).synchronize()
                toc = time.time()

                del c_cp
                return toc - tic, None

            costs = benchmark_func(func)

            del (a_cp, b_cp)

        print(
            "Lib: %s\tCost: %.4f  (CV: %.2f)"
            % (
                system_class.__name__ if system_class else "None",
                np.mean(costs),
                np.std(costs) / np.mean(costs),
            )
        )


def benchmark_x_T_x(num_gpus, N_list, system_class_list, d=1000, dtype=np.float32):
    format_string = "%20s,%10s,%10s,%10s"
    print(format_string % ("Library", "N", "Cost", "CV"))

    # Benchmark engines
    for N in N_list:
        N = int(N)
        N_block = N // num_gpus
        d_block = d
        for system_class in system_class_list:
            try:
                if system_class in ["Cupy", "Numpy"]:
                    name = system_class
                    import cupy as cp

                    arr_lib = cp if system_class == "Cupy" else np

                    x = arr_lib.ones((N, d), dtype=dtype)
                    cp.cuda.Device(0).synchronize()

                    def func():
                        tic = time.time()
                        c = x.T @ x
                        cp.cuda.Device(0).synchronize()
                        toc = time.time()

                        del c
                        return toc - tic, None

                    costs = benchmark_func(func)

                    del x
                else:
                    name = system_class.__name__
                    system = system_class(num_gpus)
                    system.init()
                    app = ArrayApplication(system=system, filesystem=FileSystem(system))

                    x = app.ones((N, d), block_shape=(N_block, d_block), dtype=dtype)
                    x.touch()

                    def func():
                        tic = time.time()
                        c = x.T @ x
                        c.touch()
                        toc = time.time()
                        del c
                        return toc - tic, None

                    costs = benchmark_func(func)

                    system.shutdown()
                    del (x, app)
            except Exception:
                costs = [-1]

            log_str = format_string % (
                name,
                "%d" % N,
                "%.4f" % np.mean(costs),
                "%.2f" % (np.std(costs) / np.mean(costs)),
            )
            print(log_str)
            with open("result_bop.csv", "a") as f:
                f.write(log_str + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int)

    args = parser.parse_args()
    num_gpus = args.num_gpus or get_number_of_gpus()

    try:
        ray.init(address="auto")
    except ConnectionError:
        ray.init()

    benchmark_x_T_x(
        num_gpus,
        N_list=[
            0.5e6 / 4,
            1e6 / 4,
            5e6 / 4,
            10e6 / 4,
            20e6 / 4,
            40e6 / 4,
            80e6 / 4,
        ],
        system_class_list=[
            # NumpySerialSystem,
            # CupySerialSystem,
            # NumpyRaySystem,
            CupyRaySystem,
            # TorchGPURaySystem,
            # CupyOsActorSystem,
            # CupyNcclActorSystem,
            CupyParallelSystem,
            "Cupy",
            "Numpy",
        ],
    )
