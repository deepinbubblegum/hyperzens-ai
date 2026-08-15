// Packed-ternary grouped matmul compute kernel (Apple Metal).
//
// Out[b, o] = sum_k sign(w[leaf[b], o, k]) * x[b, k]
//
// Each threadgroup owns one batch row `b`. It cooperatively loads the full
// input row into threadgroup memory once, then every thread computes one or
// more output columns of the leaf selected by `leaf[b]`. Multiplication by the
// ternary weights {-1, 0, +1} is expanded into branchless additions, so the
// leaf evaluation is pure matrix-addition with no FP multiply units involved.
//
// Weight encoding (2 bits per value, 4 values per uint8):
//   bits == 0b00 -> 0, bits == 0b01 -> +1, bits == 0b10 -> -1
// Column k of a row lives in byte k/4 at bits [2*(k%4), 2*(k%4)+1).

#include <metal_stdlib>
using namespace metal;

constant uint TERNARY_MAX_DIN [[function_constant(0)]];
// Keep the shared-memory row sized at compile time; 1024 floats = 4 KB.
constant uint TERNARY_TG_CAP = 1024;

kernel void ternary_mm(
    device const float* x        [[buffer(0)]],  // (B, d_in)
    device const uchar*  packed_w[[buffer(1)]],  // (L, d_out, K) K = d_in/4
    device const uint*   leaf    [[buffer(2)]],  // (B)
    device float*        out     [[buffer(3)]],  // (B, d_out)
    constant uint&       d_in    [[buffer(4)]],
    constant uint&       d_out   [[buffer(5)]],
    constant uint&       K       [[buffer(6)]],
    uint   g           [[thread_position_in_grid]],      // global thread id
    uint   tid         [[thread_position_in_threadgroup]],  // load lane
    uint   tg_w        [[threads_per_threadgroup]]) {

  threadgroup float xs[TERNARY_TG_CAP];

  const uint b = g / d_out;   // batch index  (grid is (B, d_out) threads)
  const uint o = g % d_out;   // output column
  const uint l = leaf[b];
  const device float* xrow = x + (size_t)b * d_in;

  // Cooperative load of the input row into threadgroup memory.
  for (uint k = tid; k < d_in; k += tg_w) {
    xs[k] = xrow[k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device uchar* wrow = packed_w + ((size_t)(l * d_out + o)) * K;
  float acc = 0.0f;
  const uint nbyte = (d_in + 3) >> 2;
  for (uint kb = 0; kb < nbyte; ++kb) {
    const uint byte = wrow[kb];
    const uint base = kb << 2;
    const float x0 = xs[base + 0];
    const float x1 = xs[base + 1];
    const float x2 = xs[base + 2];
    const float x3 = xs[base + 3];
    const uint s0 = (byte >> 0) & 3u;
    const uint s1 = (byte >> 2) & 3u;
    const uint s2 = (byte >> 4) & 3u;
    const uint s3 = (byte >> 6) & 3u;
    acc += select(0.0f, x0, s0 == 1u) - select(0.0f, x0, s0 == 2u);
    acc += select(0.0f, x1, s1 == 1u) - select(0.0f, x1, s1 == 2u);
    acc += select(0.0f, x2, s2 == 1u) - select(0.0f, x2, s2 == 2u);
    acc += select(0.0f, x3, s3 == 1u) - select(0.0f, x3, s3 == 2u);
  }
  out[(size_t)b * d_out + o] = acc;
}

// Fused single-pass routing + packed-ternary leaf matmul.
//
// Routes every batch row through the full binary decision tree and evaluates
// only the resulting leaf's packed rows, in one kernel dispatch and with zero
// host round-trips or per-branch buffers. Routing matches
// FastFeedForwardBitNet._routing_forward exactly:
//     node = 0
//     for level in depth:
//       logit = <x, router_w[node]> + router_b[node]
//       node  = 2*node + 1 + (logit >= 0)     // sigmoid(logit) >= 0.5
//     leaf = node - (2^depth - 1)
// (sigmoid is monotonic, so `sigmoid(logit) >= 0.5` iff `logit >= 0`.)
//
// Activations are AbsMax-quantized in-kernel (matching bitlinear.absmax_quantize
// per-row scaling + round-half-to-even) when `activation_max_q != 0`, i.e. when
// activation_bits < 32; at >= 32 bits the leaves consume the raw activations.
//
// Buffer layout (0-12):
//   0 x, 1 router_w, 2 router_b, 3 packed_w, 4 out, 5 d_in, 6 d_out, 7 K,
//   8 depth, 9 activation_max_q, 10 leaf_bias (optional), 11 has_bias, 12 eps.
//
// Routing is computed once per threadgroup (cooperatively: partial dots are
// reduced with simd_sum, then across SIMD groups through `red`, and the current
// node is broadcast through `nd`), so the leaf GEMM below reads only the routed
// leaf's packed rows - the leaf is never materialized in global memory.

kernel void fused_ternary_fff_mm(
    device const float* x         [[buffer(0)]],  // (B, d_in) raw (padded) input
    device const float* router_w  [[buffer(1)]],  // (N, d_in) N = 2^depth - 1
    device const float* router_b  [[buffer(2)]],  // (N)
    device const uchar* packed_w  [[buffer(3)]],  // (L, d_out, K) L = 2^depth
    device float*        out      [[buffer(4)]],  // (B, d_out)
    constant uint&       d_in     [[buffer(5)]],
    constant uint&       d_out    [[buffer(6)]],
    constant uint&       K        [[buffer(7)]],
    constant uint&       depth    [[buffer(8)]],
    constant int&        activation_max_q [[buffer(9)]],
    device const float*  leaf_bias [[buffer(10)]],  // (L, d_out), optional
    constant uint&       has_bias [[buffer(11)]],
    constant float&      eps      [[buffer(12)]],
    uint   tid [[thread_position_in_threadgroup]],
    uint   tg_w [[threads_per_threadgroup]],
    uint   b   [[threadgroup_position_in_grid]],
    uint   sg   [[simdgroup_index_in_threadgroup]],
    uint   lane [[thread_index_in_simdgroup]],
    uint   nsg  [[simdgroups_per_threadgroup]]) {

  threadgroup float xs[TERNARY_TG_CAP];  // raw row, then AbsMax-quantized
  threadgroup float red[32];             // cross-SIMD reduction scratch
  threadgroup uint  nd;                  // current heap node / final leaf

  const device float* xrow = x + (size_t)b * d_in;

  // Cooperative load of the input row; zero the reduction scratch.
  for (uint k = tid; k < d_in; k += tg_w) xs[k] = xrow[k];
  for (uint i = tid; i < 32u; i += tg_w) red[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Cooperative routing: each level is one cross-threadgroup dot product.
  nd = 0u;
  for (uint level = 0u; level < depth; ++level) {
    const device float* rw = router_w + (size_t)nd * d_in;
    float partial = 0.0f;
    if (tid < tg_w) {
      for (uint k = tid; k < d_in; k += tg_w) partial += xs[k] * rw[k];
    }
    partial = simd_sum(partial);
    if (lane == 0u) red[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float logit = router_b[nd];
      for (uint i = 0u; i < nsg; ++i) logit += red[i];
      const bool right = logit >= 0.0f;  // sigmoid(logit) >= 0.5
      nd = (nd << 1u) + 1u + (right ? 1u : 0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // AbsMax activation quantization, in place (matches absmax_quantize).
  if (activation_max_q != 0) {
    const float mq = (float)activation_max_q;
    float mx = 0.0f;
    for (uint k = tid; k < d_in; k += tg_w) mx = max(mx, abs(xs[k]));
    mx = simd_max(mx);
    if (lane == 0u) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float m = eps;  // scale = max(|x|, eps)
      for (uint i = 0u; i < nsg; ++i) m = max(m, red[i]);
      red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = red[0];
    for (uint k = tid; k < d_in; k += tg_w) {
      float v = xs[k];
      float q = rint(v / scale * mq);            // round-half-to-even
      q = clamp(q, -mq - 1.0f, mq);
      xs[k] = q / mq * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Leaf matmul: each thread owns one output column of the routed leaf.
  const uint leaf = nd - ((1u << depth) - 1u);
  if (tid < tg_w) {
    const device uchar* wrow = packed_w + ((size_t)(leaf * d_out + tid)) * K;
    float acc = 0.0f;
    for (uint kb = 0u; kb < K; ++kb) {
      const uint byte = wrow[kb];
      const uint base = kb << 2u;
      const float x0 = xs[base + 0u];
      const float x1 = xs[base + 1u];
      const float x2 = xs[base + 2u];
      const float x3 = xs[base + 3u];
      const uint s0 = (byte >> 0u) & 3u;
      const uint s1 = (byte >> 2u) & 3u;
      const uint s2 = (byte >> 4u) & 3u;
      const uint s3 = (byte >> 6u) & 3u;
      acc += select(0.0f, x0, s0 == 1u) - select(0.0f, x0, s0 == 2u);
      acc += select(0.0f, x1, s1 == 1u) - select(0.0f, x1, s1 == 2u);
      acc += select(0.0f, x2, s2 == 1u) - select(0.0f, x2, s2 == 2u);
      acc += select(0.0f, x3, s3 == 1u) - select(0.0f, x3, s3 == 2u);
    }
    if (has_bias != 0u) acc += leaf_bias[(size_t)leaf * d_out + tid];
    out[(size_t)b * d_out + tid] = acc;
  }
}
