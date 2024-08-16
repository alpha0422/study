#!/usr/bin/env python

# Host state mutated inside the graph capturing range.

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    def capturable_func(state: list, x: torch.Tensor):
        y = x * 1
        state[0] = state[0] + 1  # mutate CPU state
        return y

    def non_capturable_func(state: list, x: torch.Tensor):
        if state[0] % 2 == 0:
            x *= -1
        return x

    state = [0]
    x = torch.ones(1, device='cuda', dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    for i in range(6):
        if i < 3:  # warmup
            y = capturable_func(state, x)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                y = capturable_func(state, x)
            graph.replay()
        else:  # replay
            graph.replay()
        y = non_capturable_func(state, y)
        print(f"{i=}, {state=}, {y.item()=}")

if __name__ == "__main__":
    test()
