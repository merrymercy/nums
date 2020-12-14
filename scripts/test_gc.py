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

def test_gc(num_gpus):
    # Compute answer with numpy
    if True:
        N, d = 20000, 20000
        N_block = N // 1
        d_block = d // 1
    else:
        N, d = 1000, 1000
        N_block = N // 1
        d_block = d // 1
    dtype = np.float32

    a_np = np.zeros((d, N)).astype(dtype)
    b_np = np.zeros((N, d)).astype(dtype)
    print("np array generated")

    #system = CupyOsActorSystem(num_gpus)
    system = CupyParallelSystem(num_gpus, local_cache=False)
    system.init()
    app_inst = ArrayApplication(system=system, filesystem=FileSystem(system))

    block_a = app_inst.array(a_np, block_shape=(d_block, N_block))
    block_b = app_inst.array(b_np, block_shape=(N_block, d_block))

    for i in range(10000):
        print(i)
        block_c = (block_a + block_b).get()


if __name__ == "__main__":
    num_gpus = 1  # get_number_of_gpus()
    ray.init(num_gpus=num_gpus)

    test_gc(num_gpus)
