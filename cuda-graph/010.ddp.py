#!/usr/bin/env python

# DistributedDataParallel must be initialized on non-blocking stream.
#   cudaErrorStreamCaptureImplicit(906): operation would make the legacy stream depend on a capturing blocking stream

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz
#   torchrun --nproc_per_node=1 010.ddp.py

import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def test():
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.barrier()
    model = torch.nn.Linear(128, 128).cuda()
    x = torch.randn((32, 128), device="cuda")
    stream = torch.cuda.Stream()

    side_stream = False
    if side_stream:  # success
        with torch.cuda.stream(stream):
            model = DDP(model)
    else:  # failure
        model = DDP(model)

    # warmup
    for _ in range(15):
        model.zero_grad()
        y = model(x)
        y.sum().backward()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="global"):
        model.zero_grad()
        y = model(x)
        y.sum().backward()

if __name__ == "__main__":
    test()
