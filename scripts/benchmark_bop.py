import time
import gc

import numpy as np
import ray

import nums
from nums.core.array.application import ArrayApplication
from nums.core.systems import numpy_compute
from nums.core.systems.systems import SerialSystem, GPUSystem
from nums.core.systems.filesystem import FileSystem
from nums.core.systems.gpu_engine import (
    NumpySerialEngine, CupySerialEngine, NumpyRayEngine, CupyRayEngine,
    TorchCPURayEngine, TorchGPURayEngine, CupyOsActorEngine, CupyNcclActorEngine
)

def check_block_integrity(arr):
    for grid_entry in arr.grid.get_entry_iterator():
        assert arr.blocks[grid_entry].grid_entry == grid_entry
        assert arr.blocks[grid_entry].rect == arr.grid.get_slice_tuples(grid_entry)
        assert arr.blocks[grid_entry].shape == arr.grid.get_block_shape(grid_entry)

def benchmark_func(func, repeat=1, warmup=1):
    for i in range(warmup):
        func()

    costs = []
    for i in range(repeat):
        cost = func()
        costs.append(cost)

    return costs


def benchmark_tensordot(num_gpus, engine_class_list):
    # Compute answer with numpy
    array_shape = (20000, 20000)
    block_shape = (10000, 10000)
    #array_shape = (8000, 8000)
    #block_shape = (4000, 4000)
    dtype = np.float32

    a = np.random.uniform(size=array_shape).astype(dtype)
    b = np.random.uniform(size=array_shape).astype(dtype)
    # c = np.tensordot(a, b, axes=1)
    print("np init array generated")

    # Benchmark engines
    for engine_class in engine_class_list:
        def func():
            system = GPUSystem(engine=engine_class(num_gpus))
            system.init()
            app_inst = ArrayApplication(system=system, filesystem=FileSystem(system))

            block_a = app_inst.array(a, block_shape=block_shape)
            block_b = app_inst.array(b, block_shape=block_shape)
            block_a.touch()
            block_b.touch()

            tic = time.time()
            block_c = block_a.tensordot(block_b, axes=1)
            block_c = block_c.tensordot(block_c, axes=1)
            block_c.touch()
            toc = time.time()

            del (block_a, block_b, block_c, app_inst)
            system.shutdown()
            return toc - tic

        # check correctness
        # res = func()
        # exit()
        # assert np.allclose(res, c)

        costs = benchmark_func(func)
        print(
            "Lib: %s\tCost: %.2f  (CV: %.2f)"
            % (engine_class.__name__, np.mean(costs), np.std(costs) / np.mean(costs))
        )


if __name__ == "__main__":
    num_gpus = 4

    ray.init(num_gpus=num_gpus)

    benchmark_tensordot(num_gpus, [
        # NumpySerialEngine,
        CupySerialEngine,
        # NumpyRayEngine,
        # CupyRayEngine,
        # TorchGPURayEngine,
        # CupyOsActorEngine,
        CupyNcclActorEngine,
    ])

