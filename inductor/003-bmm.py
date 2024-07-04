#!/usr/bin/env python

# docker image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`

# pytorch/torch/_inductor/fx_passes/group_batch_fusion.py
# pytorch/test/inductor/test_group_batch_fusion.py

from common import trace_func_start, trace_func_stop

from typing import DefaultDict, Dict, Any, Counter
import collections
import torch
from torch._dynamo.test_case import run_tests, TestCase

counters: DefaultDict[str, Counter[str]] = collections.defaultdict(collections.Counter)
optimus_scuba_log: Dict[str, Any] = {}

class TestBMMFusionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.my_modules = torch.nn.ModuleList()
        for _ in range(10):
            self.my_modules.append(torch.nn.Linear(10, 10))

    def forward(self, inputs):
        output = None
        for linear, input in zip(self.my_modules, inputs):
            if output is None:
                output = linear(input)
            else:
                output += linear(input)
        return output


@torch._inductor.config.patch(
    post_grad_fusion_options={"batch_linear_post_grad": {"require_fbgemm": False}}
)
class TestPostGradBatchLinearFusion(TestCase):
    def test_batch_linear_post_grad_fusion(self):
        pt1_module = TestBMMFusionModule().cuda()
        inputs = []
        for _ in range(10):
            inputs.append(torch.randn(10, 10).cuda())
        eager_output = pt1_module(inputs)
        trace_func_start()
        pt2_module = torch.compile(pt1_module)
        pt2_output = pt2_module(inputs)
        trace_func_stop()
        self.assertTrue(torch.allclose(eager_output, pt2_output))
        self.assertEqual(
            counters["inductor"]["batch_fusion"],
            2,
        )
        self.assertNotIn("group_batch_fusion_pre_grad", optimus_scuba_log)
        self.assertIn("group_batch_fusion_post_grad", optimus_scuba_log)

if __name__ == "__main__":
    run_tests()
