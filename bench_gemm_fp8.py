import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from sgl_kernel import fp8_scaled_mm

def run(x, w):

    scale_x = torch.ones((x.shape[0],), device="cuda")
    scale_w = torch.ones((w.shape[1],), device="cuda")
    
    time = 0.

    for i in range(100):

        mask = (x == 0)

        while mask.sum():
            x[mask] = torch.randn(mask.sum(), device="cuda")
            mask = (x == 0)

        mask = (w == 0)

        while mask.sum():
            w[mask] = torch.randn(mask.sum(), device="cuda")
            mask = (w == 0)

        x *= 0.1
        w *= 0.1

        # w needs to be column-major
        w = w.t().contiguous().t()

        x_fp8 = x.to(torch.float8_e4m3fn)
        w_fp8 = w.to(torch.float8_e4m3fn)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            acc_events=True
        ) as prof:
                out = fp8_scaled_mm(x_fp8, w_fp8, scale_x, scale_w, torch.bfloat16, None)

        for evt in prof.events():
            if evt.cpu_time_total == 0:
                time += evt.device_time_total
                # print(evt.name, evt.device_time_total)


    print(f'average run time: {(time/100):.1f}us')

# print(prof.events().table())

def start(B = 1):
    # Qwen3-32B QKV
    x = torch.randn(B, 5120, device="cuda")
    w = torch.randn(5120, 10240, device="cuda")
    run(x, w)

    # Qwen3-32B O
    x = torch.randn(B, 8192, device="cuda")
    w = torch.randn(8192, 8192, device="cuda")
    run(x, w)

    # Qwen3-32B UpGate
    x = torch.randn(B, 5120, device="cuda")
    w = torch.randn(5120, 51200, device="cuda")
    run(x, w)

    # Qwen3-32B Down
    x = torch.randn(B, 25600, device="cuda")
    w = torch.randn(25600, 5120, device="cuda")
    run(x, w)

    # Qwen3-32B Logits
    x = torch.randn(B, 5120, device="cuda")
    w = torch.randn(5120, 151936, device="cuda")
    run(x, w)

    print('')
    print('')

    # Qwen3-30B-A3B QKV
    x = torch.randn(B, 2048, device="cuda")
    w = torch.randn(2048, 5120, device="cuda")
    run(x, w)

    # Qwen3-30B-A3B O
    x = torch.randn(B, 4096, device="cuda")
    w = torch.randn(4096, 2048, device="cuda")
    run(x, w)

    # Qwen3-30B-A3B MoE Logits
    x = torch.randn(B, 2048, device="cuda")
    w = torch.randn(2048, 128, device="cuda")
    run(x, w)

    # Qwen3-30B-A3B Logits
    x = torch.randn(B, 2048, device="cuda")
    w = torch.randn(2048, 151936, device="cuda")
    run(x, w)

for B in [1, 2, 4, 8, 16, 32]:
    print('')
    print('')
    print(f'B = {B}')
    start(B)

