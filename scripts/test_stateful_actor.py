import ray
import time

@ray.remote
class Actor(object):
    def __init__(self):
        pass

    def sleep(self, ct, t):
        print("Sleep", ct, flush=True)
        time.sleep(t)
        return ct

    def wait(self, ctx=None):
        pass


@ray.remote
def another_task(actor, ct, wait):
    print(f"another_task {ct} begin", flush=True)
    time.sleep(5)
    print(f"another_task {ct} end", flush=True)


if __name__ == "__main__":
    ray.init()

    actor = Actor.remote()

    actor.wait.remote(another_task.remote(actor, 0, actor.wait.remote()))
    ret = []
    for i in range(10):
        ret.append(actor.sleep.remote(i, 2))
    another_task.remote(actor, 1, actor.wait.remote())

    ray.get(ret)

