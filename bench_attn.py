import torch
from torch.profiler import profile, ProfilerActivity
import flashinfer

head_dim = 128
data_type = torch.bfloat16

workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace_buffer,
                    "NHD",
                    backend='fa2',
                    use_tensor_cores=True,
                )

def flush_gpu_cache(size_mb=256):
    x = torch.empty(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    x += 1
    torch.cuda.synchronize()

def run(B, num_qo_heads, num_kv_heads):
    
    time = 0.

    kv_indptr = torch.tensor([4098*i for i in range(B+1)],
                             device="cuda", dtype=torch.int32)
    
    kv_indice = []
    for i in range(B):
        base = 4096*i
        row = list(range(base + 1, base + 4097))
        row.append(B*4096 + i)
        row.append(B*4096 + B + i)
        kv_indice.append(row)

    flat = [x for row in kv_indice for x in row]
    kv_indices = torch.tensor(flat, device="cuda", dtype=torch.int32)

    kv_last_page_len = torch.ones(B, device='cuda', dtype=torch.int32)

    decode_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        data_type=data_type,
        q_data_type=data_type,
        non_blocking=True,
        fixed_split_size=None,
        disable_split_kv=False
    )

    for i in range(100):

        kv_cache = torch.randn((5000*B, 2, 1, num_kv_heads, head_dim),
                           dtype=data_type, device="cuda")
        
        q = torch.randn((B, num_qo_heads, head_dim), dtype=data_type, device="cuda")
        
        mask = (kv_cache == 0)

        while mask.sum():
            kv_cache[mask] = torch.randn(mask.sum(), device="cuda", dtype=torch.bfloat16)
            mask = (kv_cache == 0)

        mask = (q == 0)

        while mask.sum():
            q[mask] = torch.randn(mask.sum(), device="cuda", dtype=torch.bfloat16)
            mask = (q == 0)

        kv_cache *= 0.1
        q *= 0.1

        flush_gpu_cache()
        flush_gpu_cache()
        flush_gpu_cache()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            acc_events=True
        ) as prof:
                o = decode_wrapper.run(q, kv_cache)

        for evt in prof.events():
            if evt.cpu_time_total == 0:
                time += evt.device_time_total
                # print(evt.name, evt.device_time_total)


    print(f'average run time: {(time/100):.1f}us')

# print(prof.events().table())

def start(B = 1):
    run(B, 64, 8)
    run(B, 32, 4)
    
for B in [1, 2, 4, 8, 16]:
    print('')
    print('')
    print(f'B = {B}')
    start(B)

