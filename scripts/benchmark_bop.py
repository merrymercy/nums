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
    NumpySerialSystem, CupySerialSystem, NumpyRaySystem, CupyRaySystem,
    TorchCPURaySystem, TorchGPURaySystem, CupyOsActorSystem, CupyNcclActorSystem,
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
            app_inst = ArrayApplication(system=system, filesystem=FileSystem(system))

            block_a = app_inst.array(a_np, block_shape=(d_block, N_block))
            block_b = app_inst.array(b_np, block_shape=(N_block, d_block))

            def func():
                tic = time.time()
                block_c = block_a.tensordot(block_b, axes=1)
                block_c.touch()
                toc = time.time()
                del block_c
                return toc - tic, None

            costs = benchmark_func(func)

            del (block_a, block_b, app_inst)
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
            % (system_class.__name__ if system_class else "None",
              np.mean(costs), np.std(costs) / np.mean(costs))
        )


def benchmark_x_T_x(num_gpus, system_class_list):
    # Compute answer with numpy
    if True:
        N, d = 4000000, 1000
        N_block = N // 4
        d_block = d // 1
    else:
        N, d = 1000, 1000
        N_block = N // 2
        d_block = d // 2
    dtype = np.float32

    # Benchmark engines
    for system_class in system_class_list:
        if system_class:
            system = system_class(num_gpus)
            system.init()
            app_inst = ArrayApplication(system=system, filesystem=FileSystem(system))

            block_x = app_inst.ones((N, d), block_shape=(N_block, d_block), dtype=dtype)
            block_x.touch()

            def func():
                tic = time.time()
                block_c = block_x.T @ block_x
                block_c.touch()
                toc = time.time()
                del block_c
                return toc - tic, None

            costs = benchmark_func(func)

            del (block_x, app_inst)
            system.shutdown()
        else:
            import cupy as cp

            x_cp = cp.ones((N, d), dtype=dtype)
            cp.cuda.Device(0).synchronize()

            def func():
                tic = time.time()
                c_cp = x_cp.T @ x_cp
                cp.cuda.Device(0).synchronize()
                toc = time.time()

                del c_cp
                return toc - tic, None

            costs = benchmark_func(func)

            del x_cp

        print(
            "Lib: %s\tCost: %.4f  (CV: %.2f)"
            % (system_class.__name__ if system_class else "None",
              np.mean(costs), np.std(costs) / np.mean(costs))
        )


if __name__ == "__main__":
    num_gpus = get_number_of_gpus()
    ray.init(num_gpus=num_gpus)

    benchmark_x_T_x(num_gpus, [
        # NumpySerialSystem,
        CupySerialSystem,
        # NumpyRaySystem,
        # CupyRaySystem,
        # TorchGPURaySystem,
        # CupyOsActorSystem,
        # CupyNcclActorSystem,
        CupyParallelSystem,
        None,
    ])

