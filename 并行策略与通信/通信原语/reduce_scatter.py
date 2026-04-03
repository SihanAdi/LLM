import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
------------------------------------------
reduce_scatter
------------------------------------------
"""
def reduce_scatter_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    data = torch.zeros(world_size, 5, device=rank, dtype=torch.float)
    for i in range(world_size):
        data[i] = rank * 10 + i
    print(f"Rank {rank} original data (shape {data.shape}):\n{data.cpu().numpy()}")
    input_list = [data[i].unsqueeze(0) for i in range(world_size)]

    output = torch.empty(1, 5, device=rank, dtype=torch.float)
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} after reduce_scatter (sum), output shape {output.shape}:\n{output.cpu().numpy()}")


if __name__ == "__main__":
    try:
        mp.spawn(reduce_scatter_mp, args=(4,), nprocs=4, join=True)
    finally:
        # 解决资源泄漏问题
        if dist.is_initialized():
            dist.destroy_process_group()