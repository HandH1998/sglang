import itertools
import unittest

import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8


def native_per_column_quant_int8(a, eps=1e-10):
    """Function to perform per-column quantization on an input tensor `x` using native torch.

    Args:
        x: 输入tensor
        eps: 防止除零的小值
    Returns:
        x_q: 量化后的int8 tensor
        x_s: 每列的量化scale
    """

    a_max = a.abs().max(dim=1, keepdim=True)[0].clamp(min=eps)  # [B, 1]
    a_q = (a / a_max * 127).clamp(min=-128, max=127).to(torch.int8)  # [B, D]
    a_s = a_max / 127  # [B, 1]
    # a_q, a_s = per_token_quant_int8(a)
    # print("a_q: ", a_q)
    # print("a_q2: ", a_q2)
    # print("a_s: ", a_s)
    # print("a_s2: ", a_s2)

    return a_q, a_s


def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
    """修改后的矩阵乘法函数，支持per-token输入量化和per-column权重量化"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "维度不匹配"
    assert B.ndim == 2 and B.is_contiguous(), "B必须是2D连续tensor"

    # 重塑输入
    M = A.numel() // A.shape[-1]
    B = B.t()  # 转置权重矩阵
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As是per-token的 [M, 1]，Bs是per-column的 [1, K]
    # A = A * As  # 广播到 [M, N]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # 广播per-column scale

    return C.reshape(origin_C_shape).to(output_dtype)


def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk):
    """This function performs fused moe with per-column quantization using native torch."""

    B, D = a.shape
    # 修改为per-token量化

    a_q, a_s = native_per_column_quant_int8(a)
    # 重复token以匹配topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    # 同样需要重复scale
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]

    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    # 计算路由
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # 对每个专家进行计算
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # 第一层MLP: 注意a_s现在是per-token的
            inter_out = native_w8a8_per_token_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], output_dtype=a.dtype
            )
            print("inter_out: ", inter_out)
            # 激活函数
            print("inter_out: ", inter_out)
            act_out = SiluAndMul().forward_native(inter_out)
            # 对激活输出进行per-token量化
            act_out_q, act_out_s = native_per_column_quant_int8(act_out)

            # 第二层MLP
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=a.dtype
            )
            print("out: ", out)

    # 应用路由权重并求和
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


class TestW8A8BlockFP8FusedMoE(unittest.TestCase):
    DTYPES = [torch.half]
    M = [1, 33]
    N = [128, 1024]
    K = [256, 4096]
    E = [8]
    TOP_KS = [2, 6]
    BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
    BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _w8a8_block_fp8_fused_moe(self, M, N, K, E, topk, block_size, dtype, seed):
        torch.manual_seed(seed)
        # 修改为int8量化
        factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        # 输入tensor
        # M * K
        a = torch.randn((M, K), dtype=dtype) / 10

        # 生成int8权重
        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
        w1 = (w1_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
        w2 = (w2_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

        # 为每一列生成scale (per_column量化)
        w1_s = torch.max(torch.abs(w1_fp32), dim=2)[0] * factor_for_scale  # [E, 2*N]
        w2_s = torch.max(torch.abs(w2_fp32), dim=2)[0] * factor_for_scale  # [E, N]
        score = torch.randn((M, E), dtype=dtype)

        with torch.inference_mode():
            ref_out = torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk)
            out = fused_moe(
                a,
                w1,
                w2,
                score,
                topk,
                renormalize=False,
                use_fp8_w8a8=False,  # 关闭fp8
                use_int8_w8a16=False,  # 启用int8
                use_int8_w8a8=True,  # 启用int8
                w1_scale=w1_s,
                w2_scale=w2_s,
                block_shape=None,  # 不使用block量化
                per_column=True,  # 启用per_column
            )

        # 检查结果
        print("out: ", out[0])
        print("ref_out: ", ref_out[0])
        print(
            "diff: ",
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / torch.mean(torch.abs(ref_out.to(torch.float32))),
        )
        self.assertTrue(
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / torch.mean(torch.abs(ref_out.to(torch.float32)))
            < 0.07
        )

    def test_w8a8_block_fp8_fused_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.TOP_KS,
            self.BLOCK_SIZE,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
                block_size=params[5],
                dtype=params[6],
                seed=params[7],
            ):
                self._w8a8_block_fp8_fused_moe(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
