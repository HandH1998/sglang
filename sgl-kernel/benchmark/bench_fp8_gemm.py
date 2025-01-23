import torch
import torch.nn.functional as F
import triton
import argparse
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
from vllm._custom_ops import scaled_fp8_quant as vllm_scaled_fp8_quant
from sgl_kernel import fp8_scaled_mm as sgl_scaled_mm
from weights_shape import WEIGHT_SHAPES
from typing import Callable, Iterable, List, Tuple
from torch.utils.benchmark import Measurement as TMeasurement
import itertools
import copy
import time
import pickle as pkl

def bench(dtype, M, N, K, provider):
    a = torch.ones((M, K), device="cuda") * 5.0
    b = torch.ones((N, K), device="cuda") * 5.0
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    a_fp8, scale_a_fp8 = vllm_scaled_fp8_quant(a, scale_a)
    b_fp8, scale_b_fp8 = vllm_scaled_fp8_quant(b, scale_b)
    b_fp8 = b_fp8.t()
    quantiles = [0.5, 0.2, 0.8]

    if "vllm-fp8" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vllm_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype),
            quantiles=quantiles,
        )
    elif "sglang-fp8" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sgl_scaled_mm(
                a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype, bias=None
            ),
            quantiles=quantiles,
        )

    gbps = lambda ms: (2 * M * N * K + M * N) * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def run(dtype: torch.dtype,
        MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:
    results = []
    for m, k, n in MKNs:
        for provider in ["vllm-fp8", "sglang-fp8"]:
            gbps = bench(dtype, m, k, n, provider) 
            results.append((provider, m, k, n, *gbps))

    return results

def run_model_bench(args):
    print("Benchmarking models:")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        data = run(args.dtype, MKNs)
        model_bench_data.append(data)

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {args.dtype} {model}-TP{tp_size} ====")
        print(f"{'Provider':<15} {'M':<10} {'K':<10} {'N':<10} {'GB/s':<10} {'Max GB/s':<10} {'Min GB/s':<10}")
        print("=" * 70)
        
        for provider, m, k, n, gbps, max_gbps, min_gbps in data:
            print(f"{provider:<15} {m:<10} {k:<10} {n:<10} {gbps:<10.2f} {max_gbps:<10.2f} {min_gbps:<10.2f}")
    timestamp = int(time.time())

    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{args.dtype}-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)
        
def to_torch_dtype(dt):
    if dt == "int8":
        return torch.int8
    if dt == "fp8":
        return torch.float16
    raise ValueError(f"unsupported dtype: {dt}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
Benchmark Cutlass GEMM.

To run square GEMMs:
    python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 square_bench --dim-start 128 --dim-end 512 --dim-increment 64

To run constant N and K and sweep M:
    python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384

To run dimensions from a model:
    python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1

Output:
    - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
        """
    )

    parser.add_argument("--dtype",
                        type=to_torch_dtype,
                        required=True,
                        help="Available options are ['fp8']")

    subparsers = parser.add_subparsers(dest="cmd")

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument("--models",
                              nargs="+",
                              type=str,
                              default=["meta-llama/Llama-3.1-8B-Instruct"],
                              help="List of models to benchmark")
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=[1],
                              help="List of tensor parallel sizes")
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=[16],
                              help="List of batch sizes")

    args = parser.parse_args()
    
    if args.cmd == "model_bench":
        run_model_bench(args)
