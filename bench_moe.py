import torch
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
import triton.language as tl
from torch.profiler import profile, ProfilerActivity

def run_up(B):
    
    time = 0.

    for i in range(100):

        hidden_states = torch.randn(B, 2048, device="cuda", dtype=torch.bfloat16)
        w1 = 0.1 * torch.randn(128, 1536, 2048, device="cuda", dtype=torch.bfloat16)
        intermediate_cache1 = torch.randn(8*B, 1536, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.randn(B, 8, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.sort(topk_weights, descending=True).values
        topk_weights /= topk_weights.sum(dim=1, keepdim=True)
        rows = []
        for _ in range(B):
            ids = torch.randperm(128, device="cuda", dtype=torch.int32)[:8]
            rows.append(ids)

        topk_ids = torch.stack(rows, dim=0)   # Bx8
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, 16, 128
        )

        config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}


        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            acc_events=True
        ) as prof:
            invoke_fused_moe_kernel(
                hidden_states,
                w1,
                None,
                intermediate_cache1,
                None,
                None,
                None,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                8,
                config,
                compute_type   = tl.bfloat16,
                use_fp8_w8a8   = False,
                use_int8_w8a8  = False,
                use_int8_w8a16 = False,
                use_int4_w4a16 = False,
                per_channel_quant = False,
                block_shape   = None,
                c_sorted      = False,
                filter_expert = False,
            )

        for evt in prof.events():
            if evt.cpu_time_total == 0:
                time += evt.device_time_total
                # print(evt.name, evt.device_time_total)


    print(f'average run time: {(time/100):.1f}us')

def run_down(B):
    
    time = 0.

    for i in range(100):

        intermediate_cache2 = torch.randn(B*8, 768, device="cuda", dtype=torch.bfloat16)
        w2 = 0.1 * torch.randn(128, 2048, 768, device="cuda", dtype=torch.bfloat16)
        out_hidden_states = torch.randn(B, 8, 2048, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.randn(B, 8, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.sort(topk_weights, descending=True).values
        topk_weights /= topk_weights.sum(dim=1, keepdim=True)
        rows = []
        for _ in range(B):
            ids = torch.randperm(128, device="cuda", dtype=torch.int32)[:8]
            rows.append(ids)

        topk_ids = torch.stack(rows, dim=0)   # Bx8
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, 16, 128
        )

        config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}


        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            acc_events=True
        ) as prof:
            invoke_fused_moe_kernel(
                intermediate_cache2,
                w2,
                None,
                out_hidden_states,
                None,
                None,
                None,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                True,
                1,
                config,
                compute_type   = tl.bfloat16,
                use_fp8_w8a8   = False,
                use_int8_w8a8  = False,
                use_int8_w8a16 = False,
                use_int4_w4a16 = False,
                per_channel_quant = False,
                block_shape   = None,
                a_use_tma     = False,
                b_use_tma     = False,
                filter_expert = False,
                fuse_sum_all_reduce = False,
                router_topk = 8,
            )

        for evt in prof.events():
            if evt.cpu_time_total == 0:
                time += evt.device_time_total
                # print(evt.name, evt.device_time_total)


    print(f'average run time: {(time/100):.1f}us')

# run_up(1)
# run_down(1)

for B in [1, 2, 4, 8, 16]:
    print('')
    print('')
    print(f'B = {B}')
    run_up(B)
    run_down(B)

