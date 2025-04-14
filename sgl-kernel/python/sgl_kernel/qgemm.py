import torch


def gemm_forward_cuda(_in_feats, _kernel, _wscales, _ascales, _w_szs, _a_ssums, _out_feats):
    return torch.ops.sgl_kernel.gemm_forward_cuda.default(
        _in_feats,
        _kernel,
        _wscales,
        _ascales,
        _w_szs,
        _a_ssums,
        _out_feats,
    )
