import torch
import torch.nn.functional as F
import triton

from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
from vllm._custom_ops import scaled_fp8_quant as vllm_scaled_fp8_quant
from sgl_kernel import fp8_scaled_mm as sgl_scaled_mm
from sgl_kernel import fp8_scaled_mm_profile as sgl_scaled_mm_profile
import time
import vllm
import triton

def get_sm_version():
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def get_device_name():
    return torch.cuda.get_device_name(torch.cuda.current_device())

def get_config_filename(dtype="bf16"):
    sm_version = get_sm_version()
    return f"sm{sm_version}_fp8_{dtype}.json"

def do_profile(dtype="bf16", n=4096, k=8192):
    M = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
    for m in M:
        a = torch.ones((m, k), device="cuda") * 5.0
        b = torch.ones((n, k), device="cuda") * 5.0
        scale_a = torch.randn((m,), device="cuda", dtype=torch.float32)
        scale_b = torch.randn((n,), device="cuda", dtype=torch.float32)
        a_fp8, scale_a_fp8 = vllm_scaled_fp8_quant(a, scale_a)
        b_fp8, scale_b_fp8 = vllm_scaled_fp8_quant(b, scale_b)
        b_fp8 = b_fp8.t()
        sgl_scaled_mm_profile(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype, bias=None)

@triton.testing.perf_report(
        triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=["vllm-fp8-fp16", "vllm-fp8-bf16", "sglang-fp8-fp16", "sglang-fp8-bf16", 
                  "sglang-fp8-profile-fp16", "sglang-fp8-profile-bf16", "torch-fp8"],
        line_names=["vllm-fp8-fp16", "vllm-fp8-bf16", "sglang-fp8-fp16", "sglang-fp8-bf16", 
                   "sglang-fp8-profile-fp16", "sglang-fp8-profile-bf16", "torch-fp8"],
        styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--"), 
               ("red", "-"), ("red", "--"), ("purple", "-")],
        ylabel="GB/s",
        plot_name="int8 scaled matmul",
        args={},
    )
)

def benchmark(batch_size, provider):
    M, N, K = batch_size, 4096, 8192
    a = torch.ones((M, K), device="cuda") * 5.0
    b = torch.ones((N, K), device="cuda") * 5.0
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    a_fp8, scale_a_fp8 = vllm_scaled_fp8_quant(a, scale_a)
    b_fp8, scale_b_fp8 = vllm_scaled_fp8_quant(b, scale_b)
    b_fp8 = b_fp8.t()
    quantiles = [0.5, 0.2, 0.8]

    dtype = torch.float16 if "fp16" in provider else torch.bfloat16
    bias = torch.randn((N,), device="cuda", dtype=dtype)
    if "vllm-fp8" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vllm_scaled_mm(
                a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype, bias=bias
            ),
            quantiles=quantiles,
        )
    elif "sglang-fp8-profile" in provider:
        do_profile(dtype, N, K)
        try:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: sgl_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype, bias=bias, is_profile=True),
                quantiles=quantiles,
            )
        except RuntimeError as e:
            print("Error details:", e)
            ms, min_ms, max_ms = 1, 1, 1
    elif "sglang-fp8" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sgl_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype, bias=bias, is_profile=False),
            quantiles=quantiles,
        )
    elif provider == "torch-fp8":
        scale_a_2d = scale_a_fp8.float().unsqueeze(1)  # [M, 1]
        scale_b_2d = scale_b_fp8.float().unsqueeze(0)  # [1, N]
        try:
            out = torch.empty(
                (a_fp8.shape[0], b_fp8.shape[0]), device="cuda", dtype=torch.bfloat16
            )
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch._scaled_mm(
                    a_fp8,
                    b_fp8,
                    out=out,
                    out_dtype=torch.bfloat16,
                    scale_a=scale_a_2d,
                    scale_b=scale_b_2d,
                    use_fast_accum=True,
                    bias=bias,
                ),
                quantiles=quantiles,
            )
        except RuntimeError as e:
            print("Error details:", e)
            raise
    gbps = lambda ms: (2 * M * N * K + M * N) * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path="bench_fp8_res")