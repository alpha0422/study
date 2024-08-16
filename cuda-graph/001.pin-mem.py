#!/usr/bin/env python

# Query event while capturing graph leads to failure:
#   cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing
# Pin memory leads to such event query.

# Tested with:
#   Image: nvcr.io/nvidia/pytorch:23.11-py3
#   GPU: GH100-80GB-HBM3-700W-132SM-1980/2619MHz

import time
import threading
import queue
import torch

def worker(q):
    while True:
        data = q.get()
        data.pin_memory()
        q.task_done()

def test():
    q = queue.Queue()
    event = torch.cuda.Event()
    event.record()
    threading.Thread(target=worker, args=(q,), daemon=True).start()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    x = torch.randn(8, device="cuda")
    y = torch.randn(8, device="cuda")

    t1 = torch.randn(8).pin_memory()
    # non-blocking copy records t1 on side stream
    t2 = t1.to(device="cuda", non_blocking=True)
    # t1 will be freed the next time CachingHostAllocator finds the event finishes
    del t1
    data = torch.randn(8)

    # `capture_error_mode="thread_local"` works for this case,
    # but may not work for bprop which is another thread.
    with torch.cuda.graph(graph, capture_error_mode="global"):
        z = x + y
        q.put(data)
        time.sleep(1)
        z = z / y
    q.join()

if __name__ == "__main__":
    test()
