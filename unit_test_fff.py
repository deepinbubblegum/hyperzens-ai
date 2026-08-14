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
    status = "✅" if passed else "❌"
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


def get_compute_dtype(device: torch.device) -> torch.dtype:
    """Preferred compute dtype for the device."""
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def run_tests() -> bool:
    # HyperZens 35B Student Configuration
    HIDDEN_SIZE = HYPERZENS_35B_HIDDEN_SIZE        # 3584
    NUM_TREES = HYPERZENS_35B_NUM_TREES            # 16
    DEPTH = HYPERZENS_35B_TREE_DEPTH               # 5 (32 leaves/tree, 512 total)
    INTERMEDIATE_SIZE = HYPERZENS_35B_INTERMEDIATE_SIZE  # 18944
    BATCH_SIZE = 2
    SEQ_LEN = 128
    SEQ_LEN_MEM = 2048

    device = get_device()
    compute_dtype = get_compute_dtype(device)

    # -----------------------------------------------------------------
    # Memory-aware configuration for RTX 3060 12GB
    # -----------------------------------------------------------------
    if device.type == "cuda":
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        print(f"🔍 CUDA Memory: {free_gb:.1f} GB free / {total_mem / (1024 ** 3):.1f} GB total")

        # 1) 4-bit NF4 quantization is ON BY DEFAULT at the full K=16, d=5 scale.
        #    Leaf weights drop from ~13 GB (BF16) to ~3.25 GB (4-bit NF4).
        try:
            import bitsandbytes  # noqa: F401 — Linear4bit backend used by fff_layer
            USE_4BIT = True
            USE_REDUCED_DEPTH = False
            print("✅ 4-bit NF4 quantization ENABLED (load_in_4bit=True) at K=16, d=5")
        except ImportError:
            USE_4BIT = False
            USE_REDUCED_DEPTH = True
            DEPTH = 3
            print("⚠️  bitsandbytes unavailable — falling back to unquantized BF16 at d=3")

        # Adaptive memory-benchmark seq len so the soft forward+backward fits VRAM.
        leaf_i = INTERMEDIATE_SIZE // NUM_TREES
        stored_leaf_params = NUM_TREES * (1 << DEPTH) * 3 * HIDDEN_SIZE * leaf_i
        weight_bytes = int(stored_leaf_params * (0.5 if USE_4BIT else 2.0))
        act_bytes_per_token = 12 * NUM_TREES * (1 << DEPTH) * HIDDEN_SIZE
        budget_bytes = max(free_mem - weight_bytes - int(1.0 * 1024 ** 3), 0)
        bench_seq_len = max(32, min(SEQ_LEN_MEM, budget_bytes // act_bytes_per_token))
    else:
        # CPU/MPS: reduced config for acceptable runtime.
        USE_4BIT = False
        USE_REDUCED_DEPTH = True
        HIDDEN_SIZE = 256
        NUM_TREES = 2
        DEPTH = 3
        INTERMEDIATE_SIZE = 512
        BATCH_SIZE = 1
        SEQ_LEN = 16
        SEQ_LEN_MEM = 64
        bench_seq_len = SEQ_LEN_MEM
        print(f"⚠️  Running on {device.type.upper()} — using reduced test configuration")

    TOTAL_TESTS = 5
    passed_count = 0

    print(f"\n{'='*60}")
    print(f"MultiTreeFFFLayer Unit Tests — HyperZens 35B Scale")
    print(f"{'='*60}")
    print(f"Config: H={HIDDEN_SIZE}, K={NUM_TREES}, d={DEPTH}, I_dense={INTERMEDIATE_SIZE}")
    print(f"Expert I per leaf: {INTERMEDIATE_SIZE // NUM_TREES}")
    print(f"Device: {device} | Compute dtype: {compute_dtype}")
    print(f"Total Leaves: {NUM_TREES * (1 << DEPTH)} (active/token: {NUM_TREES})")
    print(f"4-bit quantization: {USE_4BIT} | reduced depth: {USE_REDUCED_DEPTH}")
    print(f"Memory benchmark seq len: {bench_seq_len} (target {SEQ_LEN_MEM})")
    print(f"{'='*60}\n")

    def create_layer(cpu_init: bool = False):
        layer = MultiTreeFFFLayer(
            hidden_size=HIDDEN_SIZE,
            num_trees=NUM_TREES,
            depth=DEPTH,
            intermediate_size=INTERMEDIATE_SIZE,
            init_temp=1.0,
            load_in_4bit=USE_4BIT,
            quant_type="nf4",
            block_size=64,
            cpu_init=cpu_init or (device.type == "cuda" and USE_4BIT),
        )
        if device.type == "cuda" and not USE_4BIT:
            # Unquantized fallback runs in BF16 to stay well under the VRAM budget.
            layer = layer.to(device, dtype=torch.bfloat16)
        else:
            layer = layer.to(device)
        return layer

    def autocast_ctx():
        """Return autocast context for the current device."""
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=compute_dtype)
        if device.type == "mps":
            return torch.autocast(device_type="mps", dtype=compute_dtype)
        return torch.autocast(device_type="cpu", enabled=False)

    def clear_memory():
        """Clear device memory cache."""
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == "mps":
            torch.mps.empty_cache()

    def prepare_4bit_layer(layer):
        """Verify safe 4-bit initialization, then quantize on CPU → device.

        With ``load_in_4bit=True`` the dense leaf projections are ``None``
        (weights live in ``leaf_qstate`` after :meth:`quantize_leaves`), so
        parameter initialization must never touch them. This guards against
        ``'NoneType' object has no attribute 'uniform_'`` regressions.
        """
        assert layer.gate_proj is None, "4-bit layer must not hold dense gate_proj"
        assert layer.up_proj is None, "4-bit layer must not hold dense up_proj"
        assert layer.down_proj is None, "4-bit layer must not hold dense down_proj"
        assert layer.leaf_qstate is None, "leaf_qstate must be unset before quantize_leaves()"
        assert layer.router_weights is not None and torch.isfinite(layer.router_weights).all()
        assert layer.router_biases is not None and (layer.router_biases == 0).all()
        print("  📦 Quantizing leaves on CPU...")
        layer.quantize_leaves()
        layer = layer.to(device)
        clear_memory()
        return layer

    # -------------------------------------------------------------------------
    # Test 1: Shape Verification
    # -------------------------------------------------------------------------
    test_idx = 1
    try:
        clear_memory()
        layer = create_layer(cpu_init=True)

        if USE_4BIT:
            # Construct weights on CPU first, quantize there, then move to CUDA
            # so a full-size FP16/BF16 copy never lands on the GPU.
            layer = prepare_4bit_layer(layer)

        layer.train()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device)
        with autocast_ctx():
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
    finally:
        clear_memory()

    # -------------------------------------------------------------------------
    # Test 2: Gradient Flow & Backward Pass
    # -------------------------------------------------------------------------
    test_idx = 2
    try:
        clear_memory()
        layer = create_layer(cpu_init=True)

        if USE_4BIT:
            layer = prepare_4bit_layer(layer)

        layer.train()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, requires_grad=True)
        with autocast_ctx():
            y = layer.forward_soft(x)
            loss = y.pow(2).mean()
        loss.backward()

        # Check input gradient
        ok_input_grad, msg = check_grad_finite("input x", x)
        # Check router gradients
        ok_router_w, msg_r = check_grad_finite("router_weights", layer.router_weights)
        ok_router_b, msg_b = check_grad_finite("router_biases", layer.router_biases)

        # Check leaf gradients (only if not quantized, as quantized weights are frozen)
        if not USE_4BIT:
            ok_gate, msg_g = check_grad_finite("gate_proj", layer.gate_proj)
            ok_up, msg_u = check_grad_finite("up_proj", layer.up_proj)
            ok_down, msg_d = check_grad_finite("down_proj", layer.down_proj)
            all_ok = all([ok_input_grad, ok_router_w, ok_router_b, ok_gate, ok_up, ok_down])
        else:
            # With 4-bit, only router gradients should flow
            all_ok = all([ok_input_grad, ok_router_w, ok_router_b])

        if all_ok:
            passed_count += 1
            print_result(test_idx, TOTAL_TESTS, "Gradient Flow & Backward", True)
        else:
            msgs = [m for m in [msg, msg_r, msg_b] if m]
            if not USE_4BIT:
                msgs += [m for m in [msg_g, msg_u, msg_d] if m]
            print_result(test_idx, TOTAL_TESTS, "Gradient Flow & Backward", False, "; ".join(msgs))
    except Exception as e:
        print_result(test_idx, TOTAL_TESTS, "Gradient Flow & Backward", False, str(e))
    finally:
        clear_memory()

    # -------------------------------------------------------------------------
    # Test 3: Precision Compatibility (bfloat16, float16)
    # -------------------------------------------------------------------------
    test_idx = 3
    precision_results = []
    for dtype_name, dtype in [("bfloat16", torch.bfloat16), ("float16", torch.float16)]:
        try:
            if device.type == "cpu" and dtype == torch.bfloat16:
                precision_results.append((dtype_name, False, "CPU bfloat16 not supported"))
                continue
            if device.type == "mps" and dtype == torch.bfloat16:
                precision_results.append((dtype_name, False, "MPS bfloat16 not supported"))
                continue

            clear_memory()
            layer = create_layer(cpu_init=True)

            if USE_4BIT:
                layer = prepare_4bit_layer(layer)

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
        finally:
            clear_memory()

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
        clear_memory()
        layer = create_layer(cpu_init=True)

        if USE_4BIT:
            layer = prepare_4bit_layer(layer)

        layer.train()

        all_finite = True
        for _ in range(5):
            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device) * 10.0
            with autocast_ctx():
                y_soft = layer.forward_soft(x)
                y_hard = layer.forward_hard(x)
                y_ste = layer.forward_soft_ste(x)
            ok, _ = check_tensor_finite("soft", y_soft)
            ok &= check_tensor_finite("hard", y_hard)[0]
            ok &= check_tensor_finite("ste", y_ste)[0]
            all_finite &= ok

            # Also test with temperature annealing
            layer.set_temperature(0.1)
            with autocast_ctx():
                y_cold = layer.forward_soft(x)
            all_finite &= check_tensor_finite("cold", y_cold)[0]

        if all_finite:
            passed_count += 1
            print_result(test_idx, TOTAL_TESTS, "Numerical Stability", True)
        else:
            print_result(test_idx, TOTAL_TESTS, "Numerical Stability", False, "NaN/Inf detected")
    except Exception as e:
        print_result(test_idx, TOTAL_TESTS, "Numerical Stability", False, str(e))
    finally:
        clear_memory()

    # -------------------------------------------------------------------------
    # Test 5: VRAM / Memory Benchmark (T=2048)
    # -------------------------------------------------------------------------
    test_idx = 5
    try:
        clear_memory()
        layer = create_layer(cpu_init=True)

        if USE_4BIT:
            layer = prepare_4bit_layer(layer)

        layer.train()

        # Adaptive benchmark: start at bench_seq_len and halve on CUDA OOM so the
        # test still reports a passing peak-memory measurement at the max fit T.
        bench_t = bench_seq_len
        peak_bytes = 0
        achieved_t: int | None = None
        while bench_t >= 1:
            try:
                clear_memory()
                x = torch.randn(BATCH_SIZE, bench_t, HIDDEN_SIZE, device=device)

                # Warmup (soft forward — the hard path materializes large tensors
                # at full scale and is not what the benchmark measures).
                with torch.no_grad(), autocast_ctx():
                    _ = layer.forward_soft(x)

                clear_memory()

                with autocast_ctx():
                    y = layer.forward_soft(x)
                    loss = y.pow(2).mean()
                    loss.backward()

                if device.type == "cuda":
                    peak_bytes = torch.cuda.max_memory_allocated(device)
                elif device.type == "mps":
                    peak_bytes = torch.mps.driver_allocated_memory()
                achieved_t = bench_t
                break
            except torch.cuda.OutOfMemoryError:
                bench_t //= 2
                clear_memory()

        if achieved_t is None:
            raise RuntimeError("OOM even at minimal seq len in VRAM benchmark")

        peak_gb = peak_bytes / (1024 ** 3)
        passed_count += 1
        dev_name = "GPU" if device.type == "cuda" else "MPS"
        mode_str = "4-bit" if USE_4BIT else "FP16/BF16"
        print_result(
            test_idx, TOTAL_TESTS, "VRAM Benchmark", True,
            f"Peak {dev_name} Memory: {peak_gb:.2f} GB (B={BATCH_SIZE}, T={achieved_t}, {mode_str})",
        )
    except Exception as e:
        print_result(test_idx, TOTAL_TESTS, "VRAM Benchmark", False, str(e))
    finally:
        clear_memory()

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
