#!/usr/bin/env python

# host & device sync is not allowed
#   cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    graph = torch.cuda.CUDAGraph()
    data = torch.randn(1, device="cuda")

    with torch.cuda.graph(graph, capture_error_mode="global"):
        data.item()

if __name__ == "__main__":
    test()
