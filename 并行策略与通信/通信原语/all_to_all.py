import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
------------------------------------------
all_to_all
------------------------------------------
"""
def all_to_all_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    send_tensor = torch.zeros(world_size, 2, device=rank, dtype=torch.float)
    for i in range(world_size):
        send_tensor[i] = rank * 10 + i
    print(f"Rank {rank} original data (shape {send_tensor.shape}):\n{send_tensor.cpu().numpy()}")

    sender_list = [send_tensor[i].unsqueeze(0) for i in range(world_size)]
    
    output = torch.empty(world_size, 2, device=rank, dtype=torch.float)

    output_list = [output[i].unsqueeze(0) for i in range(world_size)]

    dist.all_to_all(output_list, sender_list)

    output_tensor = torch.cat(output_list, dim=0)
    print(f"\nRank {rank} output (after alltoall):\n{output_tensor.cpu().numpy()}")


if __name__ == "__main__":
    try:
        mp.spawn(all_to_all_mp, args=(4,), nprocs=4, join=True)
    finally:
        # 解决资源泄漏问题
        if dist.is_initialized():
            dist.destroy_process_group()