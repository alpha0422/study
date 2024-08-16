#!/usr/bin/env python

# All streams used inside capturing range must join the graph stream
#   CUDA error cudaErrorStreamCaptureUnjoined(904): capturing stream has unjoined work

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import torch

def test():
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    x = torch.randn(128, device="cuda")

    with torch.cuda.graph(graph, capture_error_mode="global"):
        x.norm()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x.mean()
        # missing this line
        # torch.cuda.current_stream().wait_stream(stream)

if __name__ == "__main__":
    test()
