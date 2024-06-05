#!/usr/bin/env python

__all__ = ["trace_func_start", "trace_func_stop"]

##############################################################################
###  User Options
##############################################################################

import os

DEBUG = os.getenv("DEBUG", "1") == "1"

##############################################################################
###  tracefunc
##############################################################################

try:
    import tracefunc
except ImportError:
    HAS_TRACE_FUNC = False
else:
    HAS_TRACE_FUNC = True

if HAS_TRACE_FUNC:
    trace_func_start = tracefunc.trace_func_start
    trace_func_stop = tracefunc.trace_func_stop
else:
    trace_func_start = trace_func_stop = lambda: None

##############################################################################
###  torch.compile options
##############################################################################

if DEBUG:
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["INDUCTOR_WRITE_SCHEDULER_GRAPH"] = "1"
    #os.environ["TORCH_LOGS"] = "+fusion"  # suppress default logs

import torch
import torch._logging
import torch._inductor.config as inductor_config

if DEBUG:
    torch._logging.set_logs(fusion=True)  # doesn't work
    inductor_config.debug = True
    inductor_config.verbose_progress = True
    inductor_config.compile_threads = 1
    inductor_config.trace.enabled = True
    inductor_config.trace.debug_log = True
    inductor_config.trace.info_log = True
    inductor_config.trace.graph_diagram = True  # INDUCTOR_POST_FUSION_SVG=1
    inductor_config.trace.draw_orig_fx_graph = True  # INDUCTOR_ORIG_FX_SVG=1

