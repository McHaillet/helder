"""
Test for the multi-GPU (DDP) fix in get_avg_model_input_mean_and_std_from_dataloader: under
DDP, each rank only sees its own shard of the data, so the mean/std it computes locally must
be averaged across all ranks (via all_reduce) - otherwise different ranks would end up
applying different normalization in their forward pass, silently breaking the
data-parallel-training invariant that every replica computes the same function. Uses a real
2-process gloo process group (CPU-only, no GPU needed) rather than a full Lightning
Trainer/DDPStrategy, to test the fix in isolation.
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helder.utils.normalization import get_avg_model_input_mean_and_std_from_dataloader


def _worker(rank, world_size, port, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # each rank's local shard has a deliberately different mean, mimicking a
    # DistributedSampler split of a real dataset
    torch.manual_seed(rank)
    data = torch.randn(4, 6, 6, 6) + rank * 10.0
    loader = [(data,)]  # a dataloader-like iterable yielding one non-dict batch
    mean, std = get_avg_model_input_mean_and_std_from_dataloader(loader, batches=1)
    results[rank] = (mean, std)
    dist.destroy_process_group()


def test_normalization_stats_are_synced_across_ranks():
    world_size = 2
    port = 29511
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(world_size, port, results), nprocs=world_size, join=True)

    mean0, std0 = results[0]
    mean1, std1 = results[1]
    # both ranks must agree exactly on the synced statistic
    assert mean0 == pytest.approx(mean1)
    assert std0 == pytest.approx(std1)

    # and it must actually be the cross-rank average, not just rank 0's local value: rank 0's
    # own local data is centered near 0, rank 1's near +10, so the synced mean must land
    # roughly halfway between - nowhere near either rank's local-only mean
    assert 3 < mean0 < 7
