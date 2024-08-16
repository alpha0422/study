#!/usr/bin/env python

# Graph input tensor with gradient tape must run on a side stream.
#   cudaErrorStreamCaptureImplicit(906): operation would make the legacy stream depend on a capturing blocking stream

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = x + y
        return z * x

    x = torch.randn([2, 2], device="cuda", requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)

    side_stream = False
    if side_stream:  # success
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = y + 1
        torch.cuda.current_stream().wait_stream(s)
    else:  # failure
        y = y + 1

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="global"):
        z = func(x, y)
        z.sum().backward()

if __name__ == "__main__":
    test()
