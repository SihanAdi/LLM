import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
------------------------------------------
broadcast
------------------------------------------
"""
def broadcast_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    shape = (5, 2)
    if rank == 0:
        tensor = torch.ones(shape, device=rank, dtype=torch.float)
        print(f"Rank {rank} after broadcast:\n{tensor.cpu().numpy()}")
    else:
        tensor = torch.empty(shape, device=rank, dtype=torch.float)
        print(f"Rank {rank} before broadcast:\n{tensor.cpu().numpy()}")

    dist.broadcast(tensor, src=0)
    print(f"Rank {rank} after broadcast:\n{tensor.cpu().numpy()}")



if __name__ == "__main__":
    try:
        mp.spawn(broadcast_mp, args=(3,), nprocs=3, join=True)
    finally:
        # 解决资源泄漏问题
        if dist.is_initialized():
            dist.destroy_process_group()