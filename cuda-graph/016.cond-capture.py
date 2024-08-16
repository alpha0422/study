#!/usr/bin/env python

# Stream in library and extension must be maintained correctly.

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    def func(x: torch.Tensor, y: torch.Tensor):
        z = x + y
        if torch.cuda.is_current_stream_capturing():
            z *= -1
        return z

    x = torch.zeros(1, device='cuda', dtype=torch.int32)
    y = torch.zeros(1, device='cuda', dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    for i in range(6):
        y.fill_(i)
        if i < 3:  # warmup
            z = func(x, y)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                z = func(x, y)
            graph.replay()
        else:  # replay
            graph.replay()
        print(f"{i=}, {z.item()=}")

if __name__ == "__main__":
    test()
