#!/usr/bin/env python

# cudaMalloc is now supported

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, capture_error_mode="global"):
        data = torch.randn(8, device="cuda")

if __name__ == "__main__":
    test()
