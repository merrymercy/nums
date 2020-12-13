import sys
import functools

import numpy as np
import ray

from nums.core.systems import numpy_compute
from nums.core.settings import np_ufunc_map
from nums.core.systems.interfaces import RNGInterface
from nums.core.systems.utils import extract_functions

class BaseGPUSystem(object):
    def __init__(self):
        for name in ['random_block', 'new_block', 'update_block', 'create_block',
            'sum_reduce', 'map_uop', 'reshape', 'inv', 'empty', 'reduce_axis',
            'astype', 'bop']:
            setattr(self, name, functools.partial(self.call_compute_interface, name))

    def get_rng(self, seed) -> RNGInterface:
        from nums.core.systems import numpy_compute
        self.rng_cls = numpy_compute.RNG
        return self.rng_cls(seed)

    def init(self):
        pass

    def shutdown(self):
        pass

    def register(self, name: str, func: callable, remote_params: dict = None):
        pass

    def call_compute_interface(self, name, *args, **kwargs):
        raise NotImplementedError


##############################################################
############ SerialSystem: Serial implementation #############
##############################################################
class SerialSystem(BaseGPUSystem):
    def __init__(self, compute_module):
        # Init ComputeInterface
        self.compute_imp = compute_module.ComputeCls()
        super().__init__()

    def touch(self, object_id, syskwargs):
        self.get(object_id)
        return object_id

    def call_compute_interface(self, name, *args, **kwargs):
        del kwargs['syskwargs']
        return getattr(self.compute_imp, name)(*args, **kwargs)


class NumpySerialSystem(SerialSystem):
    def __init__(self, num_gpus):
        super().__init__(numpy_compute)

    def put(self, x):
        return x

    def get(self, x):
        return x


class CupySerialSystem(SerialSystem):
    def __init__(self, num_gpus):
        import cupy as cp
        from nums.core.systems import cupy_compute

        self.cp = cp
        super().__init__(cupy_compute)

    def put(self, x):
        return self.cp.array(x)

    def get(self, x):
        self.cp.cuda.Device(0).synchronize()
        if isinstance(x, list):
            return [a.get() for a in x]
        else:
            return x.get()

    def shutdown(self):
        mempool = self.cp.get_default_memory_pool()
        mempool.free_all_blocks()


##############################################################
##### RaySystem: Use the scheduler + object store in Ray #####
##############################################################
class RaySystem(BaseGPUSystem):
    def __init__(self):
        super().__init__()

    def put(self, x):
        return ray.put(x)

    def get(self, x):
        return ray.get(x)

    def touch(self, object_id, syskwargs):
        ray.wait([object_id])
        return object_id

    def shutdown(self):
        pass

    def call_compute_interface(self, name, *args, **kwargs):
        del kwargs['syskwargs']
        return self.compute_imp_funcs[name].remote(*args, **kwargs)


class NumpyRaySystem(RaySystem):
    def __init__(self, num_gpus=None):
        self.compute_imp_funcs = extract_functions(numpy_compute.ComputeCls)
        for name in self.compute_imp_funcs:
            self.compute_imp_funcs[name] = ray.remote(self.compute_imp_funcs[name])
        super().__init__()


class CupyRaySystem(RaySystem):
    def __init__(self, num_gpus=None):
        import cupy as cp
        from nums.core.systems import cupy_compute
        self.compute_imp_funcs = extract_functions(cupy_compute.ComputeCls)

        for name in self.compute_imp_funcs:
            raw_func = self.compute_imp_funcs[name]
            def _func(raw_func):
                @ray.remote(num_gpus=1)
                def local_func(*args, **kwargs):
                    args = [cp.array(v) if isinstance(v, np.ndarray) else v for v in args]
                    kwargs = {k: cp.array(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
                    return raw_func(*args, **kwargs).get()

                self.compute_imp_funcs[name] = local_func
            _func(raw_func)

        super().__init__()


class TorchCPURaySystem(RaySystem):
    def __int__(self):
        raise NotImplementedError


class TorchGPURaySystem(RaySystem):
    def __int__(self):
        raise NotImplementedError

##############################################################
######### RayActorEngien: Use actors to manage GPUs ##########
##############################################################
from typing import Union, List
from collections import namedtuple
import uuid

import numpy as np
import ray
from ray._raylet import ObjectRef
from ray.actor import ActorHandle


ArrayRef = namedtuple("ArrayRef", ["uid", "shape", "dtype"])
UID_MAX_LEN = 3


@ray.remote(num_gpus=1)
class GPUActor:
    def __init__(
        self,
        world_size,
        world_rank,
        arr_lib="torch",
        cupy_nccl_uid=None,
        torch_init_method=None,
    ):
        self.world_size = world_size
        self.world_rank = world_rank
        self.arr_lib = arr_lib
        self.torch_init_method = torch_init_method
        self.cupy_nccl_uid = cupy_nccl_uid
        self.arrays = {}
        self.gc_list = []
        self.gc_nbytes = 0

        if self.arr_lib == "torch":
            import torch  # move import here to avoid the conflict between cupy and torch
            import torch.distributed as dist

            self.torch = torch
            self.torch_dist = dist
            self.cuda_sync = self.cuda_sync_torch
            self.compute_imp = None
        elif self.arr_lib == "cupy":
            import cupy as cp
            import cupy.cuda.nccl as nccl
            from nums.core.systems import cupy_compute

            self.cp = cp
            self.cp_nccl = nccl
            self.cuda_sync = self.cuda_sync_cupy
            self.compute_imp = cupy_compute.ComputeCls()

            self.ARR_DTYPE_TO_NCCL_DTYPE = {
                cp.int32: nccl.NCCL_INT32,
                cp.float32: nccl.NCCL_FLOAT32,
                cp.float64: nccl.NCCL_FLOAT64,
            }
        else:
            raise ValueError("Invalid arr_lib: " + self.arr_lib)

    def setup(self):
        if self.arr_lib == "torch":
            self.torch_dist.init_process_group(
                backend="nccl",
                init_method=self.torch_init_method,
                world_size=self.world_size,
                rank=self.world_rank,
            )
        elif self.arr_lib == "cupy":
            self.comm = self.cp_nccl.NcclCommunicator(
                self.world_size, self.cupy_nccl_uid, self.world_rank
            )
        self.cuda_sync()

    def put(self, data) -> ArrayRef:
        uid = str(uuid.uuid4())[:UID_MAX_LEN]
        if self.arr_lib == "torch":
            data = self.torch.tensor(data, device="cuda:0")
        elif self.arr_lib == "cupy":
            data = self.cp.array(data)

        self._register_new_array(uid, data)
        self.cuda_sync()
        return ArrayRef(uid, data.shape, data.dtype)

    def get(self, arr_ref: ArrayRef):
        if self.arr_lib == "torch":
            data = self.arrays[arr_ref.uid].cpu().numpy()
        elif self.arr_lib == "cupy":
            data = self.arrays[arr_ref.uid].get()
        return data

    def touch(self, arr_ref: ArrayRef):
        self.cuda_sync()

    def call_compute_interface(self, name, *args, **kwargs):
        uid = str(uuid.uuid4())[:UID_MAX_LEN]

        args = [self.arrays[v.uid]
                if isinstance(v, ArrayRef) else v for v in args]
        kwargs = {k: self.arrays[v.uid]
                if isinstance(v, ArrayRef) else v for k, v in kwargs.items()}

        ret = getattr(self.compute_imp, name)(*args, **kwargs)

        self._register_new_array(uid, ret)
        self.cuda_sync()
        return ArrayRef(uid, ret.shape, ret.dtype)

    def send_nccl(self, arr_ref: ArrayRef, dst_rank):
        data = self.arrays[arr_ref.uid]
        if self.arr_lib == "torch":
            self.torch_dist.send(data, dst_rank)
        elif self.arr_lib == "cupy":
            self.comm.send(
                data.data.ptr,
                data.size,
                self.ARR_DTYPE_TO_NCCL_DTYPE[data.dtype.type],
                dst_rank,
                self.cp.cuda.Stream.null.ptr,
            )
        self.cuda_sync()

    def recv_nccl(self, arr_ref: ArrayRef, src_rank):
        if self.arr_lib == "torch":
            data = self.torch.empty(
                list(arr_ref.shape), dtype=arr_ref.dtype, device="cuda:0"
            )
            self.torch_dist.recv(data, src_rank)
        elif self.arr_lib == "cupy":
            data = self.cp.empty(arr_ref.shape, arr_ref.dtype)
            self.comm.recv(
                data.data.ptr,
                data.size,
                self.ARR_DTYPE_TO_NCCL_DTYPE[data.dtype.type],
                src_rank,
                self.cp.cuda.Stream.null.ptr,
            )
        self._register_new_array(arr_ref.uid, data)
        self.cuda_sync()

    def send_obj_store(self, arr_ref: ArrayRef):
        return self.arrays[arr_ref.uid]

    def recv_obj_store(self, arr_ref: ArrayRef, data):
        self._register_new_array(arr_ref.uid, data)

    def cuda_sync_torch(self):
        self.torch.cuda.synchronize()

    def cuda_sync_cupy(self):
        self.cp.cuda.Device(0).synchronize()

    def _get_bytes(self, data):
        if self.arr_lib == "torch":
            return data.element_size() * data.nelement()
        elif self.arr_lib == "cupy":
            return data.nbytes

    def _register_new_array(self, uid, data):
        nbytes = self._get_bytes(data)

        self.arrays[uid] = data
        self.gc_nbytes += nbytes
        self.gc_list.append(uid)

        # print("actor %d : register %s (%s)" % (self.world_rank, uid, str(data.shape)))

        # print("actor %d : mem size %.1f MB" % (self.world_rank, self.gc_nbytes / 1024/1024))

        if self.gc_nbytes >= 8 * (1 << 30):
            split = int(len(self.gc_list) * 0.2)
            to_del = self.gc_list[:split]
            self.gc_list = self.gc_list[split:]

            for uid in to_del:
                self.gc_nbytes -= self._get_bytes(self.arrays[uid])
                del self.arrays[uid]


def get_flatten_id(grid_entry, grid_shape):
    ret = 0
    for i in range(len(grid_entry)):
        dim = 1 if i == len(grid_entry) - 1 else grid_shape[i+1]
        ret = (ret + grid_entry[i]) * dim

    return ret


class GPUActorSystem(BaseGPUSystem):
    def __init__(self, num_gpus, arr_lib="torch", comm_lib="nccl"):
        self.arr_lib = arr_lib

        # Launch actors
        if self.arr_lib == "torch":
            cluster_file = "tcp://localhost:%d" % np.random.randint(10000, 20000)
            self.gpu_actors = [
                GPUActor.remote(
                    num_gpus, i, arr_lib=arr_lib, torch_init_method=cluster_file
                )
                for i in range(num_gpus)
            ]
        elif self.arr_lib == "cupy":
            import cupy as cp

            uid = cp.cuda.nccl.get_unique_id()
            self.gpu_actors = [
                GPUActor.remote(num_gpus, i, arr_lib=arr_lib, cupy_nccl_uid=uid)
                for i in range(num_gpus)
            ]
        else:
            raise ValueError("Invalid arr_lib: " + self.arr_lib)

        # Meta info about actors and communication lock
        self.actor_to_rank = {self.gpu_actors[i]: i for i in range(num_gpus)}
        self.obj_ref_to_owner = {}  # ObjectRef -> ActorHandle
        self.distribution_dict = {}  # (ObjectRef, ActorHandle) -> ObjectRef
        self.comm_pair_lock = {}  # (rank, rank) -> ObjectRef

        # Setup communication library
        if comm_lib == "nccl":
            self.copy_task = GPUActorSystem._copy_task_nccl
        else:
            self.copy_task = GPUActorSystem._copy_task_obj_store

        ray.get([actor.setup.remote() for actor in self.gpu_actors])
        self.actor_ct = 0

        # Init ComputeInterface
        super().__init__()

    def put(self, data) -> ObjectRef:
        #self.actor_ct = (self.actor_ct + 1) % len(self.gpu_actors)
        #dst_actor = self.gpu_actors[self.actor_ct]
        #obj_ref = dst_actor.put.remote(data)
        #return self._register_new_array(obj_ref, dst_actor)

        actor0 = self.gpu_actors[0]
        obj_ref = actor0.put.remote(data)
        self._register_new_array(obj_ref, actor0)

        dist_tasks = []
        for i in range(1, len(self.gpu_actors)):
            ref = self._distribute_to(obj_ref, self.gpu_actors[i])
            dist_tasks.append(ref)
        ray.get(dist_tasks)

        return obj_ref

    def get(self, obj_ref: Union[ObjectRef, List[ObjectRef]]):
        if isinstance(obj_ref, ObjectRef):
            actor = self.obj_ref_to_owner[obj_ref]
            obj_ref = actor.get.remote(obj_ref)
            return ray.get(obj_ref)
        else:
            obj_refs = obj_ref
            actors = [self.obj_ref_to_owner[obj_ref] for obj_ref in obj_refs]
            obj_refs = [
                actor.get.remote(obj_ref) for actor, obj_ref in zip(actors, obj_refs)
            ]
            return ray.get(obj_refs)

    def touch(self, obj_ref, syskwargs):
        actor = self.obj_ref_to_owner[obj_ref]
        ray.wait([actor.touch.remote(obj_ref)])
        return obj_ref

    def call_compute_interface(self, name, *args, **kwargs) -> ObjectRef:
        syskwargs = kwargs.pop('syskwargs')
        gid = get_flatten_id(syskwargs['grid_entry'], syskwargs['grid_shape'])
        dst_actor = self.gpu_actors[gid % len(self.gpu_actors)]

        args = [self._distribute_to(v, dst_actor)
                if isinstance(v, np.ndarray) else v for v in args]
        kwargs = {k: self._distribute_to(v, dst_actor)
                if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}

        obj_ref = dst_actor.call_compute_interface.remote(name, *args, **kwargs)

        return self._register_new_array(obj_ref, dst_actor)

    def _distribute_to(self, obj_ref: ObjectRef, dst: ActorHandle) -> ObjectRef:
        ret = self.distribution_dict.get((obj_ref, dst), None)
        if ret is None:
            src = self.obj_ref_to_owner[obj_ref]
            src_rank = self.actor_to_rank[src]
            dst_rank = self.actor_to_rank[dst]
            assert src_rank != dst_rank
            lock_key = (min(src_rank, dst_rank), max(src_rank, dst_rank))

            ret = self.copy_task.remote(
                obj_ref,
                src,
                src_rank,
                dst,
                dst_rank,
                self.comm_pair_lock.get(lock_key, None),
            )
            self.distribution_dict[(obj_ref, dst)] = ret
            self.comm_pair_lock[lock_key] = ret
        return ret

    @ray.remote
    def _copy_task_nccl(
        arr_ref: ArrayRef,
        src_actor: ActorHandle,
        src_rank,
        dst_actor: ActorHandle,
        dst_rank,
        lock,
    ) -> ArrayRef:
        a = src_actor.send_nccl.remote(arr_ref, dst_rank)
        b = dst_actor.recv_nccl.remote(arr_ref, src_rank)
        ray.get([a, b])
        return arr_ref

    @ray.remote
    def _copy_task_obj_store(
        arr_ref: ArrayRef,
        src_actor: ActorHandle,
        src_rank,
        dst_actor: ActorHandle,
        dst_rank,
        lock,
    ) -> ArrayRef:
        print("GPUSystem send %s from %d to %d" % (arr_ref.uid, src_rank, dst_rank))
        ray.get(
            dst_actor.recv_obj_store.remote(
                arr_ref, src_actor.send_obj_store.remote(arr_ref)
            )
        )
        return arr_ref

    def _register_new_array(self, obj_ref: ObjectRef, owner: ActorHandle):
        self.obj_ref_to_owner[obj_ref] = owner
        self.distribution_dict[(obj_ref, owner)] = obj_ref
        return obj_ref

    def shutdown(self):
        for actor in self.gpu_actors:
            ray.kill(actor)
        del self.gpu_actors
        del self.actor_to_rank
        del self.obj_ref_to_owner
        del self.distribution_dict
        del self.comm_pair_lock


class CupyNcclActorSystem(GPUActorSystem):
    def __init__(self, num_gpus):
        super().__init__(num_gpus, arr_lib="cupy", comm_lib="nccl")


class CupyOsActorSystem(GPUActorSystem):
    def __init__(self, num_gpus):
        super().__init__(num_gpus, arr_lib="cupy", comm_lib="object_store")


class TorchNcclActorSystem(GPUActorSystem):
    def __init__(self, num_gpus):
        super().__init__(num_gpus, arr_lib="torch", comm_lib="nccl")


class TorchOsActorSystem(GPUActorSystem):
    def __init__(self, num_gpus):
        super().__init__(num_gpus, arr_lib="torch", comm_lib="object_store")

