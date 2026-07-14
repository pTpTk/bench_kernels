import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

def flush_gpu_cache(size_mb=256):
    x = torch.empty(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    x += 1
    torch.cuda.synchronize()

def run(m, n, k):
    
    time = 0.

    for i in range(100):

        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        mask = (x == 0)

        while mask.sum():
            x[mask] = torch.randn(mask.sum(), device="cuda", dtype=torch.bfloat16)
            mask = (x == 0)

        mask = (w == 0)

        while mask.sum():
            w[mask] = torch.randn(mask.sum(), device="cuda", dtype=torch.bfloat16)
            mask = (w == 0)

        x *= 0.1
        w *= 0.1

        flush_gpu_cache()
        flush_gpu_cache()
        flush_gpu_cache()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            acc_events=True
        ) as prof:
                out = F.linear(x, w, None)

        for evt in prof.events():
            if evt.cpu_time_total == 0:
                time += evt.device_time_total
                # print(evt.name, evt.device_time_total)


    print(f'average run time: {(time/100):.1f}us')

# print(prof.events().table())

def start(B = 1):
    # Qwen3-32B QKV
    run(B, 10240, 5120)

    # Qwen3-32B O
    run(B, 5120, 8192)

    # Qwen3-32B UpGate
    run(B, 51200, 5120)

    # Qwen3-32B Down
    run(B, 5120, 25600)

    # Qwen3-32B Logits
    run(B, 151936, 5120)

    print('')
    print('')

    # Qwen3-30B-A3B QKV
    run(B, 5120, 2048)

    # Qwen3-30B-A3B O
    run(B, 2048, 4096)

    # Qwen3-30B-A3B MoE Logits
    run(B, 128, 2048)

    # Qwen3-30B-A3B Logits
    run(B, 151936, 2048)

for B in [1, 2, 4, 8, 16]:
    print('')
    print('')
    print(f'B = {B}')
    start(B)

