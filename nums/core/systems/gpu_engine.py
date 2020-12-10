import sys

import numpy as np
import ray

def bop_common(op, a1, a2, a1_shape, a2_shape, a1_T, a2_T, axes, syskwargs):
    # print(op, a1.shape, a2.shape)

    if a1_T:
        a1 = a1.T
    if a2_T:
        a2 = a2.T
    if a1.shape != a1_shape:
        a1 = a1.reshape(a1_shape)
    if a2.shape != a2_shape:
        a2 = a2.reshape(a2_shape)

    arr_lib = sys.modules[str(a1.__class__.__module__).split('.')[0]]

    if op == "tensordot":
        ret = arr_lib.tensordot(a1, a2, axes=axes)
    elif op == "add":
        ret = arr_lib.add(a1, a2)
    return ret


##############################################################
##### RayEngine: Use the scheduler + object store in Ray #####
##############################################################
class RayEngine(object):
    def __init__(self, num_gpus):
        pass

    def put(self, x):
        return ray.put(x)

    def get(self, x):
        return ray.get(x)

    def bop(self, *args, **kwargs):
        return self.bop_task.remote(*args, **kwargs)

    def touch(self, object_id, syskwargs):
        ray.wait([object_id])
        return object_id

    def shutdown(self):
        pass


class NumpyRayEngine(RayEngine):
    @ray.remote
    def bop_task(*args, **kwargs):
        return bop_common(*args, **kwargs)


class CupyRayEngine(RayEngine):
    @ray.remote(num_gpus=1)
    def bop_task(*args, **kwargs):
        import cupy as cp

        args = [cp.array(x) if isinstance(x, np.ndarray) else x for x in args]
        return bop_common(*args, **kwargs).get()


class TorchCPURayEngine(RayEngine):
    @ray.remote
    def bop_task(*args, **kwargs):
        import torch

        args = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in args]
        return bop_common(*args, **kwargs).numpy()


class TorchGPURayEngine(RayEngine):
    @ray.remote(num_gpus=1)
    def bop_task(*args, **kwargs):
        import torch

        args = [torch.tensor(x, device='cuda:0') if isinstance(x, np.ndarray) else x for x in args]
        return bop_common(*args, **kwargs).cpu().numpy()

##############################################################
############ SerialEngine: Serial implementation #############
##############################################################
class SerialEngine(object):
    def bop(self, *args, **kwargs):
        return bop_common(*args, **kwargs)

    def touch(self, object_id, syskwargs):
        self.get(object_id)
        return object_id

    def shutdown(self):
        pass


class NumpySerialEngine(SerialEngine):
    def __init__(self, num_gpus):
        pass

    def put(self, x):
        return x

    def get(self, x):
        return x


class CupySerialEngine(SerialEngine):
    def __init__(self, num_gpus):
        import cupy as cp

        self.cp = cp

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
########### RayActorEngien: Serial implementation ############
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
        elif self.arr_lib == "cupy":
            import cupy as cp
            import cupy.cuda.nccl as nccl

            self.cp = cp
            self.cp_nccl = nccl
            self.cuda_sync = self.cuda_sync_cupy

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

    def bop(self, *args, **kwargs):
        uid = str(uuid.uuid4())[:UID_MAX_LEN]

        new_args = []
        for arg in args:
            if isinstance(arg, ArrayRef):
                new_args.append(self.arrays[arg.uid])
            else:
                new_args.append(arg)


        ret = bop_common(*new_args, **kwargs)
        # print("actor %d: %s = %s %s %s" % (self.world_rank, uid, args[0], args[1].uid, args[2].uid))

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


class GPUActorEngine(object):
    def __init__(self, num_gpus, arr_lib="torch", comm_lib="nccl"):
        self.arr_lib = arr_lib

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

        self.actor_to_rank = {self.gpu_actors[i]: i for i in range(num_gpus)}

        self.obj_ref_to_owner = {}  # ObjectRef -> ActorHandle
        self.distribution_dict = {}  # (ObjectRef, ActorHandle) -> ObjectRef
        self.comm_pair_lock = {}  # (rank, rank) -> ObjectRef

        if comm_lib == "nccl":
            self.copy_task = GPUActorEngine._copy_task_nccl
        else:
            self.copy_task = GPUActorEngine._copy_task_obj_store

        # setup communication library
        ray.get([actor.setup.remote() for actor in self.gpu_actors])
        self.actor_ct = 0

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
        obj_ref = actor.get.remote(obj_ref)
        return ray.wait([obj_ref])

    def bop(self, op, a1, a2, a1_shape, a2_shape, a1_T, a2_T, axes, syskwargs) -> ObjectRef:
        gid = get_flatten_id(syskwargs['grid_entry'], syskwargs['grid_shape'])
        dst_actor = self.gpu_actors[gid % len(self.gpu_actors)]

        a1 = self._distribute_to(a1, dst_actor)
        a2 = self._distribute_to(a2, dst_actor)

        obj_ref = dst_actor.bop.remote(op, a1, a2, a1_shape, a2_shape, a1_T, a2_T, axes, None)
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
        # print("GPUEngine send %s from %d to %d" % (arr_ref.uid, src_rank, dst_rank))
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


class CupyNcclActorEngine(GPUActorEngine):
    def __init__(self, num_gpus):
        super().__init__(num_gpus, arr_lib="cupy", comm_lib="nccl")


class CupyOsActorEngine(GPUActorEngine):
    def __init__(self, num_gpus):
        super().__init__(num_gpus, arr_lib="cupy", comm_lib="object_store")


class TorchNcclActorEngine(GPUActorEngine):
    def __init__(self, num_gpus):
        super().__init__(num_gpus, arr_lib="torch", comm_lib="nccl")


class TorchOsActorEngine(GPUActorEngine):
    def __init__(self, num_gpus):
        super().__init__(num_gpus, arr_lib="torch", comm_lib="object_store")

