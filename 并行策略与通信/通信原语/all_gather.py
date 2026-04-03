import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
------------------------------------------
row_wise_all_gather
------------------------------------------
"""
def row_wise_all_gather_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    print(f"Initializing process group: {rank}, {type(rank)}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    tensor = torch.ones(1, 5, device=rank) * rank

    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    dist.all_gather(gather_list, tensor)

    gathered_tensor = torch.cat(gather_list, dim=0)
    print(gathered_tensor.shape)
    print(f"Rank {rank} gathered tensor:\n", gathered_tensor)

"""
------------------------------------------
column_wise_all_gather
------------------------------------------
"""
def column_wise_all_gather_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    print(f"Initializing process group: {rank}, {type(rank)}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cols_per_rank = 2
    tensor = torch.zeros(5, cols_per_rank, device=rank, dtype=torch.float)
    for col in range(cols_per_rank):
        tensor[:, col] = rank * 10 + col

    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    dist.all_gather(gather_list, tensor)

    gathered_tensor = torch.cat(gather_list, dim=1)
    print(gathered_tensor.shape)
    print(f"Rank {rank} gathered tensor:\n", gathered_tensor)


if __name__ == "__main__":
    try:
        # mp.spawn(row_wise_all_gather_mp, args=(3,), nprocs=3, join=True)

        mp.spawn(column_wise_all_gather_mp, args=(3,), nprocs=3, join=True)
    finally:
        # 解决资源泄漏问题
        if dist.is_initialized():
            dist.destroy_process_group()