#!/usr/bin/env python

# All inputs of captured CUDA graph must be static CUDA tensor, otherwise it
# leads to silent error.

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    graph = torch.cuda.CUDAGraph()
    base_lr = torch.zeros(1, device="cuda", dtype=torch.int32)

    # warmup
    for i in range(2):
        lr = base_lr + i  # dynamic input
        out = 1 * lr      # capturing range
        print(f"{i=}, {out.item()=}")

    # capture
    with torch.cuda.graph(graph):
        out = 1 * lr

    # replay
    for i in range(2, 6):
        lr = base_lr + i
        graph.replay()
        print(f"{i=}, {out.item()=}")

if __name__ == "__main__":
    test()
