#!/usr/bin/env python

# Dynamic scalar variable.

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    def func(tensor: torch.Tensor, exponent: int):
        return torch.float_power(tensor, exponent)

    x = torch.tensor(2, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()

    for i in range(6):
        if i < 3:  # warmup
            y = func(x, i)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                y = func(x, i)
            graph.replay()
        else:  # replay
            graph.replay()  # always `power(x, 3)`
        print(f"{i=}, {y.item()=}")

if __name__ == "__main__":
    test()
