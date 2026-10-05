"""Small helpers for distributed SCRO execution."""

from typing import Tuple

import torch
import torch.distributed as dist


def distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    return dist.get_rank() if distributed_enabled() else 0


def distributed_world_size() -> int:
    return dist.get_world_size() if distributed_enabled() else 1


def is_main_process() -> bool:
    return distributed_rank() == 0


def request_partition(length: int) -> Tuple[int, int]:
    rank = distributed_rank()
    world_size = distributed_world_size()
    return length * rank // world_size, length * (rank + 1) // world_size


def all_gather_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Gather differently sized column shards in rank order."""
    if not distributed_enabled():
        return matrix
    if matrix.ndim != 2:
        raise ValueError("all_gather_columns expects a matrix")

    world_size = distributed_world_size()
    local_size = torch.tensor(
        [matrix.size(1)], device=matrix.device, dtype=torch.long
    )
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    column_counts = [int(size.item()) for size in gathered_sizes]
    max_columns = max(column_counts)

    padded = matrix.new_zeros((matrix.size(0), max_columns))
    if matrix.size(1) > 0:
        padded[:, : matrix.size(1)] = matrix
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded.contiguous())
    return torch.cat(
        [part[:, :count] for part, count in zip(gathered, column_counts)],
        dim=1,
    )


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if distributed_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_from_main_(tensor: torch.Tensor) -> torch.Tensor:
    if distributed_enabled():
        dist.broadcast(tensor, src=0)
    return tensor


def distributed_barrier() -> None:
    if distributed_enabled():
        dist.barrier()
