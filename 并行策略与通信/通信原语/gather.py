import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
------------------------------------------
row_wise_gather
------------------------------------------
"""
def row_wise_gather_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    print(f"Initializing process group: {rank}, {type(rank)}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    tensor = torch.ones(1, 5, device=rank) * rank

    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)] # 第 i 个进程的数据，会被放入接收列表的第 i 个位置
        """
        在这里执行gather程序卡住的原因：
        破坏了分布式通信的“集体性”原则，导致了死锁
        dist.gather 是一个集体通信算子。这意味着：通信组内的所有进程（Rank 0, 1, 2...）必须同时执行到这行代码，程序才能继续向下运行

        Rank 0: 进入 if 分支，执行 dist.gather。
        Rank 0 的行为: 它作为接收方（dst=0），准备好了接收数据的缓冲区，然后它开始等待 Rank 1 和 Rank 2 发送数据过来。
        Rank 1 & Rank 2 的行为：它们进入 else 分支，完全跳过了 dist.gather 这一行代码，直接去执行后面的 print("non root") 甚至可能已经退出了程序。
        结果: Rank 0 在苦苦等待 Rank 1 和 Rank 2 的“加入”，但 Rank 1 和 Rank 2 根本不知道要发数据，直接跑远了。
              Rank 0 永远等不到伙伴，程序就永久挂起（Hang）了。
        """
        # dist.gather(tensor, gather_list, dst=0)
    else:
        gather_list = None

    dist.gather(tensor, gather_list, dst=0)

    if rank == 0:
        gathered = torch.cat(gather_list, dim=0)
        print(gathered)
    else:
        print("non root")

def row_wise_gather_torchrun():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Initializing process group: {rank}, {type(rank)}")
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    tensor = torch.ones(1, 5, device=local_rank) * rank

    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None

    dist.gather(tensor, gather_list, dst=0)

    if rank == 0:
        gathered = torch.cat(gather_list, dim=0)
        print(gathered)
    else:
        print("non root")

"""
------------------------------------------
column_wise_gather
------------------------------------------
"""
def column_wise_gather_mp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6231"
    
    print(f"Initializing process group: {rank}, {type(rank)}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cols_per_rank = 2
    tensor = torch.zeros(5, cols_per_rank, device=rank, dtype=torch.float)
    for col in range(cols_per_rank):
        tensor[:, col] = rank * 10 + col

    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None

    dist.gather(tensor, gather_list, dst=0)

    if rank == 0:
        gathered = torch.cat(gather_list, dim=1)
        print(gathered)
    else:
        print("non root")


if __name__ == "__main__":
    try:
        # mp.spawn(row_wise_gather_mp, args=(3,), nprocs=3, join=True)
        
        # row_wise_gather_torchrun() # torchrun --nproc-per-node=3 gather.py
        mp.spawn(column_wise_gather_mp, args=(3,), nprocs=3, join=True)
    finally:
        # 解决资源泄漏问题
        if dist.is_initialized():
            dist.destroy_process_group()