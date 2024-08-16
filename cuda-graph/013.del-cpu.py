#!/usr/bin/env python

# Deconstruct CPU tensor.

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    def func(i: int):
        t_cpu = torch.tensor(i)
        t_gpu = t_cpu.to(device='cuda', non_blocking=True)
        return t_gpu  # t_cpu deconstructed

    graph = torch.cuda.CUDAGraph()
    for i in range(6):
        if i < 3:  # warmup
            y = func(i)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                y = func(i)
            graph.replay()
        else:  # replay
            graph.replay()
        print(f"{i=}, {y.item()=}")

if __name__ == "__main__":
    test()
