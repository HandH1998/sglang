import torch

import itertools
import unittest

import torch
import triton
import matplotlib.pyplot as plt

from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8, native_per_token_group_quant_fp8, w8a8_block_fp8_matmul, native_w8a8_block_fp8_matmul
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.quantization.fp8_kernel_old import w8a8_block_fp8_matmul as w8a8_block_fp8_matmul_old


# for i in [1, 8, 32, 128, 512, 2048, 4096] 
# ()
# fused moe
# (7168, 2048) (4096, 7168)
# cannot TP
total = [(512 + 64, 7168), ((128 + 64) * 128, 7168), (128 * (128 + 128), 512), (7168, 16384), (7168, 18432), (18432 * 2, 7168)]
# N can TP
n_tp = [(18432 * 2, 7168), ((128 + 64) * 128, 7168), (128 * (128 + 128), 512)]

# K can TP
k_tp = [(7168, 18432), (7168, 16384)]
# N: 18432 * 2, 7168, 4096, 7168, (128 + 64) * 128, 512 + 64, 128 * (128 + 128), 7168,
# K: 7168, 18432, 7168, 2048, 7168, 7168, 512, 128 * 128,
x_vals = []
for i in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]:
    for t in total:
        x_vals.append((i,) + t)
    for tp in [2, 4, 8, 16]:
        for n_t in n_tp:
            new_t = (i, n_t[0] // tp, n_t[1])
            x_vals.append(new_t)
        for k_t in k_tp:
            new_t = (i, k_t[0], k_t[1] // tp)
            x_vals.append(new_t)

print(x_vals)
print(len(x_vals))

    


configs = []
# for dtype in [torch.float16, torch.bfloat16, torch.float32]:
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot 作为绘图 x 轴的参数名
        x_vals=x_vals,  # Different possible values for `x_name` `x_names` 参数的不同可能值
        # x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name` `x_names` 参数的不同可能值
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot 对应绘图中不同线的参数名
        # Possible values for `line_arg` `line_arg` 的可能值
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment. 在 fp8 情况下不与 cuBLAS 比较，因为 torch.matmul 目前不支持 fp8。
        # line_vals=["native", "triton"],  # Label name for the lines
        # line_names=["Native", "Triton"],  # Line styles
        line_vals=["triton"],  # Label name for the lines
        line_names=["Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis y 轴的标签名称
        plot_name="matmul-performance-w8a8-block-fp8",  # Name for the plot, used also as a file 绘图名称，也用作保存绘图的文件名 name for saving the plot.
        # args={"fp8_inputs": fp8_inputs},
        # args={"out_dtype": dtype},
        args={},
    ))




@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    # if provider == "triton":
    print()
    print(f"M: {M}, N: {N}, K: {K}")
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    block_size = [128, 128]
    out_dtype = torch.float16

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device='cuda') - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device='cuda') - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k -1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device='cuda') * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device='cuda') * factor_for_scale

    quantiles = [0.5, 0.2, 0.8]
    if provider == "native":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: native_w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype), quantiles=quantiles)
    if provider == "triton-old":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: w8a8_block_fp8_matmul_old(A_fp8, B_fp8, As, Bs, block_size, out_dtype), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)



benchmark.run(show_plots=True, print_data=True)
# fig, ax = plt.subplots()
# benchmark.plot(ax)
# plt.savefig('performance_plot.png')  # 保存图像到文件
# print()