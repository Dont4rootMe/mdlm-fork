# gemm_repro.py
# PyTorch 2.2.2+cu121, GPU: H100 (SM90)

import os
import time
import torch
import math

def info():
    print(f"GPU: {torch.cuda.get_device_name(0)}  CC: {torch.cuda.get_device_capability(0)}")
    print(f"Torch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print("TF32 allowed:", torch.backends.cuda.matmul.allow_tf32)

def benchmark(fn, name, ref=None):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C = fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    shape_ok = (C.shape == (M, N))
    print(f"{name:<18}  time: {dt:7.2f} ms   shape_ok: {shape_ok}")
    if ref is not None:
        max_ad = (C - ref).abs().max().item()
        print(f"{'':<18}  max|Δ| vs ref: {max_ad:.3e}")
    return C

# -------------------- problem sizes from your cuBLASLt log --------------------
M, N, K = 3072, 100, 768          # m, n, k
torch.manual_seed(0)

# A in R^{K x M}, B in R^{K x N}
A = torch.randn(K, M, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

def main():
    info()
    print("\nShapes:")
    print(" A:", tuple(A.shape), " B:", tuple(B.shape))
    print(" A.stride:", A.stride(), " A^T.stride:", A.mT.stride())
    print(" B.stride:", B.stride(), " B^T.stride:", B.mT.stride())
    print()

    # Case 0: baseline eager FP32 (оставляем TF32 включённым по умолчанию на H100)
    # !!! Эта строка может падать NOT_SUPPORTED в некоторых стеках из-за strides вида A^T
    def case_lazy_view():
        return A.mT @ B                      # (3072,768) @ (768,100) -> (3072,100)

    # Case 1: фикс — материализация транспонированного операнда
    def case_materialize():
        AT = A.mT.contiguous()               # превратить вид в плотный
        return AT @ B

    # Case 2: эквивалентная перестановка множителей: (B^T A)^T
    def case_layout_trick():
        return (B.mT @ A).mT

    # Case 3: канонически через F.linear (вес = A^T в виде (out=in M, in=K))
    def case_flinear():
        W = A.mT.contiguous()                # (M,K)
        # F.linear: out = X @ W^T + b, где X shape (*, in_features)
        # Хотим A^T B = W B, значит подадим X = B^T (N,K) и потом транспонируем результат.
        Y = torch.nn.functional.linear(B.mT, W)    # (N,M)
        return Y.mT                                 # (M,N)

    # Пытаемся выполнить все варианты, используя materialize как численный эталон.
    print("Running:")
    C_ref = benchmark(case_materialize, "materialize (ref)")
    try:
        C0 = benchmark(case_lazy_view, "lazy view", ref=C_ref)
    except RuntimeError as e:
        print("lazy view           FAIL:", e)
    C2 = benchmark(case_layout_trick, "layout trick", ref=C_ref)
    C3 = benchmark(case_flinear, "F.linear", ref=C_ref)

    # Дополнительно: демонстрация того, что материализация снимает «неудобные strides»
    ATv, ATc = A.mT, A.mT.contiguous()
    print("\nA^T (view)   stride:", ATv.stride(), "is_contiguous:", ATv.is_contiguous())
    print("A^T (contig) stride:", ATc.stride(), "is_contiguous:", ATc.is_contiguous())

if __name__ == "__main__":
    # На H100 TF32 уместен и ускоряет FP32-матмулы; при желании можно жёстко запретить:
    # torch.backends.cuda.matmul.allow_tf32 = False
    main()
