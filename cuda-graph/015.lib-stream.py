#!/usr/bin/env python

# Stream in library and extension must be maintained correctly.

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    def library_call(x):
        # Simulate a library that doesn't aware of PyTorch
        # current stream.
        stream = torch.cuda.default_stream()
        with torch.cuda.stream(stream):
            x = x * 2
        return x

    x = torch.ones(1, device='cuda', dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    for i in range(6):
        if i < 3:  # warmup
            y = x + i
            y = library_call(y)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                y = x + i
                y = library_call(y)
            graph.replay()
        else:  # replay
            graph.replay()
        print(f"{i=}, {y.item()=}")

if __name__ == "__main__":
    test()
