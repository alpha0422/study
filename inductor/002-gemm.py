#!/usr/bin/env python

# docker image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`

from common import trace_func_start, trace_func_stop

import torch
import torch.nn as nn
from collections import OrderedDict

def test():
    n, h = 32, 128
    repeats = 3
    layers = OrderedDict()
    for i in range(repeats):
        layers[f"fc_{i}"] = nn.Linear(h, h)
        layers[f"ln_{i}"] = nn.LayerNorm(h)
        layers[f"silu_{i}"] = nn.SiLU()
    model = nn.Sequential(layers).cuda().half()
    x = torch.randn((n, h), device="cuda", dtype=torch.float16, requires_grad=True)
    dy = torch.randn_like(x)

    trace_func_start()
    compiled = torch.compile(model)
    y = compiled(x)
    y.backward(dy)
    trace_func_stop()

if __name__ == "__main__":
    test()
