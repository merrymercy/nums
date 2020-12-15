import sys
import functools
import time
import itertools
import gc
from collections import defaultdict

import numpy as np
import ray

from nums.core.systems import numpy_compute
from nums.core.settings import np_ufunc_map
from nums.core.systems.interfaces import RNGInterface
from nums.core.systems.utils import extract_functions

def cupy_used_bytes():
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes()

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

    def call_compute_interface(self, name, *args, **kwargs):
        del kwargs['syskwargs']
        #if name in ['bop', 'map_uop']:
        #    print(f"SerialSystem::call compute {name} {args[0]}")
        #else:
        #    print(f"SerialSystem::call compute {name}")
        ret =  getattr(self.compute_imp, name)(*args, **kwargs)
        #print(f"SerialSystem::result {ret.shape} {cupy_used_bytes()/1e9} {ret.dtype}")
        return ret


class NumpySerialSystem(SerialSystem):
    def __init__(self, num_gpus):
        super().__init__(numpy_compute)

    def put(self, x):
        return x

    def get(self, x):
        return x

    def touch(self, object_id, syskwargs):
        return object_id


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

    def touch(self, object_id, syskwargs):
        self.cp.cuda.Device(0).synchronize()
        return object_id

    def shutdown(self):
        mempool = self.cp.get_default_memory_pool()
        mempool.free_all_blocks()

##############################################################
########## ParallelSystem: Parallel implementation ###########
##############################################################
class CupySystemArrRef:
    def __init__(self, cp_arr, system):
        self.cp_arr = cp_arr
        self.system = system

    def __del__(self):
        self.system.delete(self.cp_arr)


class CupyParallelSystem(BaseGPUSystem):
    def __init__(self, num_gpus, local_cache=True, immediate_gc=True):
        import cupy as cp
        from nums.core.systems import cupy_compute

        self.cp = cp
        self.num_gpus = num_gpus
        self.local_cache = local_cache
        self.immediate_gc = immediate_gc

        self.compute_imp = cupy_compute.ComputeCls()
        self.dist_dict = defaultdict(dict)   # Dict[array_id -> Dict[actor_id -> array]]
        super().__init__()

    def put(self, x):
        with self.cp.cuda.Device(0):
            ret = self.cp.array(x)
        ret = self._register_new_array(ret, 0)

        for actor_id in range(1, self.num_gpus):
            self._distribute_to(ret, actor_id)

        return CupySystemArrRef(ret, self)

    def get(self, x):
        if isinstance(x, list):
            return [a.cp_arr.get() for a in x]
        else:
            return x.cp_arr.get()

    def touch(self, x, syskwargs):
        x = x.cp_arr
        x.device.synchronize()
        return x

    def call_compute_interface(self, name, *args, **kwargs):
        # Make device placement decisions
        syskwargs = kwargs.pop('syskwargs')
        if name == 'bop':
            dst_actor = None
            for arg in itertools.chain(args, kwargs.values()):
                if isinstance(arg, CupySystemArrRef):
                    dst_actor = arg.cp_arr.data.device_id
                    break
        else:
            gid = get_flatten_id(syskwargs['grid_entry'], syskwargs['grid_shape'])
            dst_actor = gid % self.num_gpus

        #print(f"CupyParallelSystem::call compute {name} on {dst_actor}")

        args = [self._distribute_to(v.cp_arr, dst_actor)
                if isinstance(v, CupySystemArrRef) else v for v in args]
        kwargs = {k: self._distribute_to(v.cp_arr, dst_actor)
                if isinstance(v, CupySystemArrRef) else v for k, v in kwargs.items()}

        with self.cp.cuda.Device(dst_actor):
            ret = getattr(self.compute_imp, name)(*args, **kwargs)

        if self.immediate_gc:
            self.dist_dict = defaultdict(dict)
        else:
            ret = self._register_new_array(ret, dst_actor)

        return CupySystemArrRef(ret, self)

    def _distribute_to(self, arr, dst_actor):
        if self.local_cache:
            arr_hash = self._get_array_hash(arr)
            ret = self.dist_dict[arr_hash].get(dst_actor, None)
            if ret is None:
                if arr.data.device_id == dst_actor:
                    ret = arr
                else:
                    # print(f"Copy {arr.shape} from {arr.data.device_id} to {dst_actor}")
                    with self.cp.cuda.Device(dst_actor):
                        ret = self.cp.asarray(arr)
                    self.dist_dict[arr_hash][dst_actor] = ret
        else:
            if arr.data.device_id == dst_actor:
                ret = arr
            else:
                with self.cp.cuda.Device(dst_actor):
                    ret = self.cp.asarray(arr)

        return ret

    def _get_array_hash(self, arr):
        return (arr.data.device_id, arr.data.mem, arr.data.ptr)

    def _register_new_array(self, arr, dst_actor):
        if self.local_cache:
            self.dist_dict[self._get_array_hash(arr)][dst_actor] = arr
            return arr
        else:
            return arr

    def delete(self, arr):
        if not self.immediate_gc:
            if self.dist_dict is not None:
                del self.dist_dict[self._get_array_hash(arr)]

    def shutdown(self):
        self.dist_dict = None
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
######### RayActorSystem: Use actors to manage GPUs ##########
##############################################################
from typing import Union, List
from collections import namedtuple
import uuid

import numpy as np
import ray
from ray._raylet import ObjectRef
from ray.actor import ActorHandle


ArrayRef = namedtuple("ArrayRef", ["uid", "shape", "dtype"])
UID_MAX_LEN = 30


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
        self.gc_lru_ct = 0
        self.gc_lru_map = {}
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
        self.gc_lru_ct += 1
        self.gc_lru_map[uid] = self.gc_lru_ct

        # print("actor %d : register %s (%s)" % (self.world_rank, uid, str(data.shape)))
        # print("actor %d : mem size %.1f MB" % (self.world_rank, self.gc_nbytes / 1024/1024))

        gc_begin_threshold = 100 * (1 << 30)
        gc_finish_threshold = 8 * (1 << 30)

        # print(f"gc bytes: {self.gc_nbytes // 1024 ** 3} GB")

        if self.gc_nbytes >= gc_begin_threshold:
            uids = list(self.gc_lru_map.keys())
            uids.sort(key=lambda x: self.gc_lru_map[x])
            for uid in uids:
                if self._get_bytes(self.arrays[uid]) < 100:
                    continue
                print(f"gc del {uid}, shape: {self.arrays[uid].shape}, lru_ct: {self.gc_lru_map[uid]}")

                self.gc_nbytes -= self._get_bytes(self.arrays[uid])
                del self.arrays[uid]
                del self.gc_lru_map[uid]

                if self.gc_nbytes <= gc_finish_threshold:
                    print("gc break")
                    break


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
        # Make device placement decisions
        syskwargs = kwargs.pop('syskwargs')
        if name == 'bop':
            dst_actor = None
            for arg in itertools.chain(args, kwargs.values()):
                if isinstance(arg, ObjectRef):
                    dst_actor = self.obj_ref_to_owner[arg]
                    break
        else:
            gid = get_flatten_id(syskwargs['grid_entry'], syskwargs['grid_shape'])
            actor_id = gid % len(self.gpu_actors)
            dst_actor = self.gpu_actors[actor_id]

        print(f"CupyActorSystem::call compute {name} on {self.gpu_actors.index(dst_actor)}")

        args = [self._distribute_to(v, dst_actor)
                if isinstance(v, ObjectRef) else v for v in args]
        kwargs = {k: self._distribute_to(v, dst_actor)
                if isinstance(v, ObjectRef) else v for k, v in kwargs.items()}

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

    def _remove(self, obj_ref: ObjectRef):
        pass

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
        # print("GPUSystem send %s from %d to %d" % (arr_ref.uid, src_rank, dst_rank), flush=True)
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

