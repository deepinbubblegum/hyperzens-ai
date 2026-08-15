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
