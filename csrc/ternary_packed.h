/* Shared declarations for the packed-ternary matmul extension.
 *
 * Weights are stored packed 2-bits-per-value:
 *   bits == 0b00 -> weight 0
 *   bits == 0b01 -> weight +1
 *   bits == 0b10 -> weight -1
 *   0b11 is unused
 * Four weights share one uint8: lane k of a byte holds weights at columns
 * (4*byte + 0..3), lane k occupies bits [2k, 2k+1).
 */
#pragma once

#include <ATen/ATen.h>

namespace ternary {

// CPU (ARM64 NEON) grouped ternary matmul: out[b] = x[b] @ W[leaf[b]]^T
at::Tensor ternary_mm_cpu(const at::Tensor& x,
                          const at::Tensor& packed_w,
                          const at::Tensor& leaf_idx);

// Fused single-pass routing + packed-ternary leaf matmul (CPU, ARM64 NEON).
// Routes each row through the decision tree, AbsMax-quantizes activations and
// evaluates only the routed leaf's packed rows.
at::Tensor route_and_matmul_cpu(const at::Tensor& x,
                                const at::Tensor& router_w,
                                const at::Tensor& router_b,
                                const at::Tensor& packed_w,
                                int64_t depth,
                                int64_t activation_bits,
                                const c10::optional<at::Tensor>& leaf_bias,
                                double eps);

// Metal (MPS) grouped ternary matmul; implemented in ternary_packed_mps.mm
at::Tensor ternary_mm_mps(const at::Tensor& x,
                          const at::Tensor& packed_w,
                          const at::Tensor& leaf_idx);

// Fused single-pass routing + packed-ternary leaf matmul (Metal).
at::Tensor route_and_matmul_mps(const at::Tensor& x,
                                const at::Tensor& router_w,
                                const at::Tensor& router_b,
                                const at::Tensor& packed_w,
                                int64_t depth,
                                int64_t activation_bits,
                                const c10::optional<at::Tensor>& leaf_bias,
                                double eps);

// True when an MTLDevice is available (even if torch MPS tensors are not built).
bool mps_supported();

}  // namespace ternary
