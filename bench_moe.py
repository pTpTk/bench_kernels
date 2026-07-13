import torch
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
import triton.language as tl
from torch.profiler import profile, ProfilerActivity

counts = {
    0: 2536030,
    1: 1269044,
    2: 1145211,
    3: 1303537,
    4: 1769855,
    5: 1482301,
    6: 1379708,
    7: 691068,
    8: 874945,
    9: 1855368,
    10: 177175,
    11: 509836,
    12: 209296,
    13: 1253292,
    14: 1989054,
    15: 228386,
    16: 1254296,
    17: 859246,
    18: 1622501,
    19: 107118,
    20: 684840,
    21: 1956381,
    22: 1865565,
    23: 1360430,
    24: 1982350,
    25: 2778418,
    26: 2464395,
    27: 175643,
    28: 797283,
    29: 1483019,
    30: 1046157,
    31: 2558508,
    32: 1277543,
    33: 2438476,
    34: 344019,
    35: 1388227,
    36: 1223716,
    37: 751960,
    38: 1254537,
    39: 363523,
    40: 186833,
    41: 1277978,
    42: 1239905,
    43: 1564964,
    44: 941824,
    45: 1783557,
    46: 1065052,
    47: 1863130,
    48: 2706480,
    49: 1683189,
    50: 1595840,
    51: 779853,
    52: 1625585,
    53: 1140957,
    54: 1306836,
    55: 2211274,
    56: 1852204,
    57: 1279701,
    58: 735174,
    59: 1873723,
    60: 281327,
    61: 1666930,
    62: 1159254,
    63: 409851,
    64: 998223,
    65: 827311,
    66: 1777343,
    67: 636445,
    68: 2610014,
    69: 2348001,
    70: 1130236,
    71: 2339912,
    72: 891571,
    73: 2492512,
    74: 736309,
    75: 1186408,
    76: 840127,
    77: 666799,
    78: 1855381,
    79: 418506,
    80: 1739810,
    81: 1688273,
    82: 2876332,
    83: 314331,
    84: 679418,
    85: 752119,
    86: 1141855,
    87: 908733,
    88: 406574,
    89: 1850352,
    90: 346546,
    91: 1910925,
    92: 3198058,
    93: 1210973,
    94: 1230749,
    95: 240976,
    96: 466747,
    97: 1323877,
    98: 800472,
    99: 736400,
    100: 188357,
    101: 635777,
    102: 689978,
    103: 679392,
    104: 575580,
    105: 877502,
    106: 640520,
    107: 1280682,
    108: 274176,
    109: 382866,
    110: 1461360,
    111: 918474,
    112: 1112999,
    113: 559641,
    114: 1359030,
    115: 1366630,
    116: 383489,
    117: 393473,
    118: 832774,
    119: 661176,
    120: 735510,
    121: 306427,
    122: 1323598,
    123: 504015,
    124: 734045,
    125: 1041385,
    126: 1622084,
    127: 1079164,
}

total = sum(counts.values())

probs = torch.tensor([v / total for k, v in counts.items()], device="cuda")

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
            ids = torch.multinomial(probs, num_samples=8)
            ids.to(torch.int32)
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
            ids = torch.multinomial(probs, num_samples=8)
            ids.to(torch.int32)
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


