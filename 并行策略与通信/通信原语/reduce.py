import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
------------------------------------------
reduce
------------------------------------------
"""
def reduce_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    print(f"Initializing process group: {rank}, {type(rank)}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    tensor = torch.ones(1, 5, device=rank) * rank
    print(f"Rank {rank} before reduce: {tensor.cpu().tolist()}")


    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} after reduce (sum): {tensor.cpu().tolist()}")


if __name__ == "__main__":
    try:
        mp.spawn(reduce_mp, args=(3,), nprocs=3, join=True)
    finally:
        # 解决资源泄漏问题
        if dist.is_initialized():
            dist.destroy_process_group()