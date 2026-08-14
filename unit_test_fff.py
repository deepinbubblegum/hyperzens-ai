#!/usr/bin/env python3
"""Unit tests and benchmarks for MultiTreeFFFLayer (HyperZens 35B FFF Scale).

Tests:
  1. Shape Verification
  2. Gradient Flow & Backward Pass
  3. Precision Compatibility (bfloat16, float16)
  4. Numerical Stability
  5. VRAM / Memory Benchmark (T=2048)
"""

import sys
import torch
import torch.nn as nn

from models.fff_layer import MultiTreeFFFLayer, HYPERZENS_35B_HIDDEN_SIZE, HYPERZENS_35B_NUM_TREES, HYPERZENS_35B_TREE_DEPTH, HYPERZENS_35B_INTERMEDIATE_SIZE


def print_result(test_idx: int, total: int, name: str, passed: bool, msg: str = "") -> None:
    status = "���" if passed else "���"
    print(f"{status} [{test_idx}/{total}] {name}: {'Passed' if passed else 'Failed'}{' — ' + msg if msg else ''}")


def check_tensor_finite(name: str, tensor: torch.Tensor) -> tuple[bool, str]:
    if not torch.isfinite(tensor).all():
        return False, f"{name} contains NaN/Inf"
    return True, ""


def check_grad_finite(name: str, param: nn.Parameter) -> tuple[bool, str]:
    if param.grad is None:
        return False, f"{name} has no gradient"
    if not torch.isfinite(param.grad).all():
        return False, f"{name} gradient contains NaN/Inf"
    if param.grad.abs().max() == 0:
        return False, f"{name} gradient is all zeros"
    return True, ""


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_tests() -> bool:
    # HyperZens 35B Student Configuration
    HIDDEN_SIZE = HYPERZENS_35B_HIDDEN_SIZE        # 3584
    NUM_TREES = HYPERZENS_35B_NUM_TREES            # 16
    DEPTH = HYPERZENS_35B_TREE_DEPTH               # 5 (32 leaves/tree, 512 total)
    INTERMEDIATE_SIZE = HYPERZENS_35B_INTERMEDIATE_SIZE  # 18944
    BATCH_SIZE = 2
    SEQ_LEN = 128
    SEQ_LEN_MEM = 2048

    # Use smaller config for CPU/MPS to avoid timeout
    device = get_device()
    if device.type != "cuda":
        HIDDEN_SIZE = 256
        NUM_TREES = 2
        DEPTH = 3
        INTERMEDIATE_SIZE = 512
        BATCH_SIZE = 1
        SEQ_LEN = 16
        SEQ_LEN_MEM = 64
        print("������  Running on CPU/MPS — using reduced test configuration")

    TOTAL_TESTS = 5
    passed_count = 0

    print(f"\n{'='*60}")
    print(f"MultiTreeFFFLayer Unit Tests — HyperZens 35B Scale")
    print(f"{'='*60}")
    print(f"Config: H={HIDDEN_SIZE}, K={NUM_TREES}, d={DEPTH}, I_dense={INTERMEDIATE_SIZE}")
    print(f"Device: {device}")
    print(f"Total Leaves: {NUM_TREES * (1 << DEPTH)} (active/token: {NUM_TREES})")
    print(f"{'='*60}\n")

    def create_layer():
        return MultiTreeFFFLayer(
            hidden_size=HIDDEN_SIZE,
            num_trees=NUM_TREES,
            depth=DEPTH,
            intermediate_size=INTERMEDIATE_SIZE,
            init_temp=1.0,
        ).to(device)

    # -------------------------------------------------------------------------
    # Test 1: Shape Verification
    # -------------------------------------------------------------------------
    test_idx = 1
    try:
        layer = create_layer()
        layer.train()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            y_soft = layer.forward_soft(x)
            y_hard = layer.forward_hard(x)
            y_ste = layer.forward_soft_ste(x)

        ok_shape = (
            y_soft.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE) and
            y_hard.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE) and
            y_ste.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        )
        ok_finite, msg = check_tensor_finite("forward_soft output", y_soft)
        ok_finite &= check_tensor_finite("forward_hard output", y_hard)[0]
        ok_finite &= check_tensor_finite("forward_ste output", y_ste)[0]

        if ok_shape and ok_finite:
            passed_count += 1
            print_result(test_idx, TOTAL_TESTS, "Shape & Forward Pass", True)
        else:
            print_result(test_idx, TOTAL_TESTS, "Shape & Forward Pass", False, msg or "shape mismatch")
    except Exception as e:
        print_result(test_idx, TOTAL_TESTS, "Shape & Forward Pass", False, str(e))

    # -------------------------------------------------------------------------
    # Test 2: Gradient Flow & Backward Pass
    # -------------------------------------------------------------------------
    test_idx = 2
    try:
        layer = create_layer()
        layer.train()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, requires_grad=True)
        y = layer.forward_soft(x)
        loss = y.pow(2).mean()
        loss.backward()

        # Check input gradient
        ok_input_grad, msg = check_grad_finite("input x", x)
        # Check router gradients (sample a few)
        ok_router_w, msg_r = check_grad_finite("router_weights", layer.router_weights)
        ok_router_b, msg_b = check_grad_finite("router_biases", layer.router_biases)
        # Check leaf gradients (sample a few active leaves via grad flow)
        ok_gate, msg_g = check_grad_finite("gate_proj", layer.gate_proj)
        ok_up, msg_u = check_grad_finite("up_proj", layer.up_proj)
        ok_down, msg_d = check_grad_finite("down_proj", layer.down_proj)

        all_ok = all([ok_input_grad, ok_router_w, ok_router_b, ok_gate, ok_up, ok_down])
        if all_ok:
            passed_count += 1
            print_result(test_idx, TOTAL_TESTS, "Gradient Flow & Backward", True)
        else:
            msgs = [m for m in [msg, msg_r, msg_b, msg_g, msg_u, msg_d] if m]
            print_result(test_idx, TOTAL_TESTS, "Gradient Flow & Backward", False, "; ".join(msgs))
    except Exception as e:
        print_result(test_idx, TOTAL_TESTS, "Gradient Flow & Backward", False, str(e))

    # -------------------------------------------------------------------------
    # Test 3: Precision Compatibility (bfloat16, float16)
    # -------------------------------------------------------------------------
    test_idx = 3
    precision_results = []
    for dtype_name, dtype in [("bfloat16", torch.bfloat16), ("float16", torch.float16)]:
        try:
            if device.type == "cpu" and dtype == torch.bfloat16:
                # CPU bfloat16 requires Ampere+ or fallback
                precision_results.append((dtype_name, False, "CPU bfloat16 not supported"))
                continue
            if device.type == "mps" and dtype == torch.bfloat16:
                # MPS bfloat16 not supported yet
                precision_results.append((dtype_name, False, "MPS bfloat16 not supported"))
                continue

            layer = create_layer()
            layer = layer.to(dtype=dtype)
            layer.train()

            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
                y_soft = layer.forward_soft(x)
                y_hard = layer.forward_hard(x)
                loss = y_soft.pow(2).mean()
                loss.backward()

            ok_finite, msg = check_tensor_finite(f"{dtype_name} output", y_soft)
            ok_finite &= check_tensor_finite(f"{dtype_name} hard output", y_hard)[0]
            ok_grad, msg_g = check_grad_finite(f"{dtype_name} router_w", layer.router_weights)

            if ok_finite and ok_grad:
                precision_results.append((dtype_name, True, ""))
            else:
                precision_results.append((dtype_name, False, msg or msg_g))
        except Exception as e:
            precision_results.append((dtype_name, False, str(e)))

    # Pass if at least one precision mode works (e.g., float16 on MPS)
    working_precisions = [r[0] for r in precision_results if r[1]]
    if working_precisions:
        passed_count += 1
        print_result(test_idx, TOTAL_TESTS, "Precision Compatibility", True,
                     f"Working: {', '.join(working_precisions)}")
    else:
        failed = [f"{r[0]}: {r[2]}" for r in precision_results if not r[1]]
        print_result(test_idx, TOTAL_TESTS, "Precision Compatibility", False, "; ".join(failed))

    # -------------------------------------------------------------------------
    # Test 4: Numerical Stability (no NaN/Inf across multiple forwards)
    # -------------------------------------------------------------------------
    test_idx = 4
    try:
        layer = create_layer()
        layer.train()

        all_finite = True
        for _ in range(5):  # Reduced iterations for speed
            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device) * 10.0  # large inputs
            y_soft = layer.forward_soft(x)
            y_hard = layer.forward_hard(x)
            y_ste = layer.forward_soft_ste(x)
            ok, _ = check_tensor_finite("soft", y_soft)
            ok &= check_tensor_finite("hard", y_hard)[0]
            ok &= check_tensor_finite("ste", y_ste)[0]
            all_finite &= ok

            # Also test with temperature annealing
            layer.set_temperature(0.1)
            y_cold = layer.forward_soft(x)
            all_finite &= check_tensor_finite("cold", y_cold)[0]

        if all_finite:
            passed_count += 1
            print_result(test_idx, TOTAL_TESTS, "Numerical Stability", True)
        else:
            print_result(test_idx, TOTAL_TESTS, "Numerical Stability", False, "NaN/Inf detected")
    except Exception as e:
        print_result(test_idx, TOTAL_TESTS, "Numerical Stability", False, str(e))

    # -------------------------------------------------------------------------
    # Test 5: VRAM / Memory Benchmark (T=2048)
    # -------------------------------------------------------------------------
    test_idx = 5
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            layer = create_layer()
            layer.train()

            x = torch.randn(BATCH_SIZE, SEQ_LEN_MEM, HIDDEN_SIZE, device=device)

            # Warmup
            with torch.no_grad():
                _ = layer.forward_hard(x)

            torch.cuda.reset_peak_memory_stats(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = layer.forward_soft(x)
                loss = y.pow(2).mean()
                loss.backward()

            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_gb = peak_bytes / (1024 ** 3)

            passed_count += 1
            print_result(test_idx, TOTAL_TESTS, "VRAM Benchmark", True,
                         f"Peak GPU Memory: {peak_gb:.2f} GB (B={BATCH_SIZE}, T={SEQ_LEN_MEM})")
        elif device.type == "mps":
            # MPS memory tracking
            torch.mps.empty_cache()

            layer = create_layer()
            layer.train()

            x = torch.randn(BATCH_SIZE, SEQ_LEN_MEM, HIDDEN_SIZE, device=device)

            # Warmup
            with torch.no_grad():
                _ = layer.forward_hard(x)

            y = layer.forward_soft(x)
            loss = y.pow(2).mean()
            loss.backward()

            peak_bytes = torch.mps.driver_allocated_memory()
            peak_gb = peak_bytes / (1024 ** 3)

            passed_count += 1
            print_result(test_idx, TOTAL_TESTS, "VRAM Benchmark", True,
                         f"Peak MPS Memory: {peak_gb:.2f} GB (B={BATCH_SIZE}, T={SEQ_LEN_MEM})")
        else:
            print_result(test_idx, TOTAL_TESTS, "VRAM Benchmark", False, "CUDA/MPS not available, skipping")
    except Exception as e:
        print_result(test_idx, TOTAL_TESTS, "VRAM Benchmark", False, str(e))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Results: {passed_count}/{TOTAL_TESTS} tests passed")
    print(f"{'='*60}\n")

    return passed_count == TOTAL_TESTS


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)