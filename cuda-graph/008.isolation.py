#!/usr/bin/env python

# All stream in the capturing range should only depend on the graph capturing stream.
#   cudaErrorStreamCaptureIsolation(905): dependency created on uncaptured work in another stream

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    x = torch.randn(128, device="cuda")

    with torch.cuda.stream(stream):
        x.mean()
    with torch.cuda.graph(graph, capture_error_mode="global"):
        x.norm()
        torch.cuda.current_stream().wait_stream(stream)
        x.norm()

if __name__ == "__main__":
    test()
