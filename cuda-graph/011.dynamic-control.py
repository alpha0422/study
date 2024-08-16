#!/usr/bin/env python

# Dynamic control flow.

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    def func(idx, x, y):
        if idx % 2 == 0:
            z = x + y
        else:
            z = x - y
        return z * x

    x = torch.ones(1, device="cuda", dtype=torch.int32)
    y = torch.ones(1, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()

    for i in range(6):
        if i < 3:  # warmup
            z = func(i, x, y)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                z = func(i, x, y)
            graph.replay()
        else:  # replay
            graph.replay()
        print(f"{i=}, {z.item()=}")

if __name__ == "__main__":
    test()
