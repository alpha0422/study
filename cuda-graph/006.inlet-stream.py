#!/usr/bin/env python

# All streams used inside capturing range must branch out from a single stream.
#   CUDA error cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    x = torch.randn(128, device="cuda")

    def func():
        # missing this line
        # stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            y = x * 2
        torch.cuda.current_stream().wait_stream(stream)
        return x + y

    # warmup
    for _ in range(2):
        func()

    with torch.cuda.graph(graph, capture_error_mode="global"):
        func()

if __name__ == "__main__":
    test()
