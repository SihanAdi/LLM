import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
------------------------------------------
all_reduce: allreduce操作可拆成两个集合通信操作：reduce scatter + all gather
------------------------------------------
"""
def all_reduce_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    print(f"Initializing process group: {rank}, {type(rank)}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    tensor = torch.ones(world_size, 5, device=rank) * rank
    print(f"Rank {rank} before reduce: {tensor.cpu().tolist()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} after reduce (sum): {tensor.cpu().tolist()}")

def all_reduce_impl_by_reduce_scatter_all_gather(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    tensor = torch.ones(world_size, 5, device=rank) * rank

    input_list = [tensor[i].unsqueeze(0) for i in range(world_size)]
    output = torch.empty(1, 5, device=rank)
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} after reduce (sum): {output.cpu().tolist()}")

    gather_list = [torch.empty_like(output) for _ in range(world_size)]
    dist.all_gather(gather_list, output)

    gathered_tensor = torch.cat(gather_list, dim=0)
    print(gathered_tensor.shape)
    print(f"Rank {rank} gathered tensor:\n", gathered_tensor)


if __name__ == "__main__":
    try:
        # mp.spawn(all_reduce_mp, args=(3,), nprocs=3, join=True)
        mp.spawn(all_reduce_impl_by_reduce_scatter_all_gather, args=(3,), nprocs=3, join=True)
    finally:
        # 解决资源泄漏问题
        if dist.is_initialized():
            dist.destroy_process_group()