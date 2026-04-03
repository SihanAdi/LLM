import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
------------------------------------------
scatter
------------------------------------------
"""
def scatter_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    recv_tensor = torch.empty(5, device=rank, dtype=torch.float)

    if rank == 0:
        send_tensor = torch.arange(world_size, device=rank, dtype=torch.float).unsqueeze(1).repeat(1, 5)
        print(f"Rank {rank}: Original data to scatter:\n{send_tensor.cpu().numpy()}")
        scatter_list = [send_tensor[i] for i in range(world_size)] # 列表中的第 i 个张量，就会被发送给 Rank 为 i 的进程
        print(f"Rank {rank}: Scatter list shapes: {[t.shape for t in scatter_list]}")
    else:
        scatter_list = None

    dist.scatter(recv_tensor, scatter_list=scatter_list, src=0)

    print(f"Rank {rank} received tensor of shape {recv_tensor.shape}:\n{recv_tensor.cpu().numpy()}")


if __name__ == "__main__":
    try:
        mp.spawn(scatter_mp, args=(3,), nprocs=3, join=True)
    finally:
        # 解决资源泄漏问题
        if dist.is_initialized():
            dist.destroy_process_group()