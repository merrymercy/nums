import os

def check_block_integrity(arr):
    for grid_entry in arr.grid.get_entry_iterator():
        assert arr.blocks[grid_entry].grid_entry == grid_entry
        assert arr.blocks[grid_entry].rect == arr.grid.get_slice_tuples(grid_entry)
        assert arr.blocks[grid_entry].shape == arr.grid.get_block_shape(grid_entry)


def benchmark_func(func, repeat=2, warmup=1):
    for i in range(warmup):
        func()

    costs = []
    for i in range(repeat):
        cost = func()[0]
        costs.append(cost)

    return costs


def get_number_of_gpus():
    val = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader | wc -l').read()
    return int(val)


if __name__ == "__main__":
    print(f"Number of GPUS: {get_number_of_gpus()}")

