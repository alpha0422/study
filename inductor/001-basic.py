#!/usr/bin/env python

# docker image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`

from common import trace_func_start, trace_func_stop

import torch
import torch.nn as nn
from collections import OrderedDict

def test():
    n, h_in, h = 32, 128, 64
    model = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(h_in, h)),
        ("ln", nn.LayerNorm(h)),
        ("silu", nn.SiLU()),
        ("fc2", nn.Linear(h, h_in)),
    ])).cuda()
    x = torch.randn((n, h_in), device="cuda")

    trace_func_start()
    compiled = torch.compile(model)
    y = compiled(x)
    trace_func_stop()

if __name__ == "__main__":
    test()
