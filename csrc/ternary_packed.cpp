/* CPU implementation of the packed-ternary grouped matmul.
 *
 * Out[b, o] = sum_k  sign(w[leaf[b], o, k]) * x[b, k]
 *
 * The leaf lookup is done without allocating any per-branch buffers: tokens
 * are bucketed by leaf with a histogram + exclusive scan (O(B)), then each
 * bucket is processed against only its leaf's packed rows.
 *
 * SIMD (ARM64 NEON): multiplication by {-1, 0, +1} is replaced by two masked
 * additions (vaddq / vsubq with vbslq selects), i.e. pure matrix-addition.
 */

#include <arm_neon.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <atomic>
#include <thread>
#include <vector>

#include "ternary_packed.h"

namespace ternary {
namespace {

// per-lane select masks: lane set to 0xFFFFFFFF when the 2-bit code encodes
// +1 (POS) or -1 (NEG), zero otherwise. Indexed by the packed byte.
uint32_t POS_MASK[256][4];
uint32_t NEG_MASK[256][4];
bool g_tables_built = false;

void build_decode_tables() {
  if (g_tables_built) return;
  for (int byte = 0; byte < 256; ++byte) {
    for (int lane = 0; lane < 4; ++lane) {
      const int code = (byte >> (2 * lane)) & 3;
      POS_MASK[byte][lane] = (code == 1) ? 0xFFFFFFFFu : 0u;
      NEG_MASK[byte][lane] = (code == 2) ? 0xFFFFFFFFu : 0u;
    }
  }
  g_tables_built = true;
}

// dot product of one x row against one packed leaf row (d_in multiple of 4)
inline float packed_dot(const float* __restrict xb, const uint8_t* __restrict prow,
                        int64_t d_in) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  float32x4_t acc = zero;
  int64_t k = 0;
  for (; k + 4 <= d_in; k += 4) {
    const uint8_t byte = prow[k >> 2];
    const float32x4_t xv = vld1q_f32(xb + k);
    const float32x4_t pos =
        vbslq_f32(vld1q_u32(POS_MASK[byte]), xv, zero);
    const float32x4_t neg =
        vbslq_f32(vld1q_u32(NEG_MASK[byte]), xv, zero);
    acc = vaddq_f32(acc, pos);
    acc = vsubq_f32(acc, neg);
  }
  float32x2_t lo = vget_low_f32(acc);
  float32x2_t hi = vget_high_f32(acc);
  return vaddv_f32(vadd_f32(lo, hi));
}

void check_args(const at::Tensor& x, const at::Tensor& packed_w,
                const at::Tensor& leaf_idx) {
  TORCH_CHECK(x.is_cpu(), "ternary_mm_cpu expects a CPU tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(packed_w.is_contiguous(), "packed_w must be contiguous");
  TORCH_CHECK(packed_w.scalar_type() == at::kByte, "packed_w must be uint8");
  TORCH_CHECK(leaf_idx.scalar_type() == at::kLong ||
                  leaf_idx.scalar_type() == at::kInt,
              "leaf_idx must be int64 or int32");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be (B, d_in)");
  TORCH_CHECK(packed_w.dim() == 3, "packed_w must be (L, d_out, K)");
  TORCH_CHECK(leaf_idx.dim() == 1 && leaf_idx.size(0) == x.size(0),
              "leaf_idx must be (B,)");
  TORCH_CHECK(x.size(1) % 4 == 0, "d_in must be a multiple of 4");
  TORCH_CHECK(packed_w.size(2) == x.size(1) / 4,
              "packed_w columns must equal ceil(d_in / 4)");
}

}  // namespace

at::Tensor ternary_mm_cpu(const at::Tensor& x, const at::Tensor& packed_w,
                          const at::Tensor& leaf_idx) {
  check_args(x, packed_w, leaf_idx);
  build_decode_tables();

  const at::Tensor leaf64 =
      leaf_idx.scalar_type() == at::kLong ? leaf_idx : leaf_idx.to(at::kLong);

  const int64_t B = x.size(0);
  const int64_t d_in = x.size(1);
  const int64_t L = packed_w.size(0);
  const int64_t d_out = packed_w.size(1);
  const int64_t K = packed_w.size(2);

  auto out = at::empty({B, d_out}, x.options());
  const float* xp = x.data_ptr<float>();
  const uint8_t* wp = packed_w.data_ptr<uint8_t>();
  const int64_t* lp = leaf64.data_ptr<int64_t>();
  float* op = out.data_ptr<float>();

  // Bucket tokens by leaf (O(B), no per-branch buffers allocated).
  std::vector<int64_t> count(L, 0);
  for (int64_t b = 0; b < B; ++b) ++count[lp[b]];
  std::vector<int64_t> start(L, 0);
  for (int64_t l = 1; l < L; ++l) start[l] = start[l - 1] + count[l - 1];
  std::vector<int64_t> order(B);
  {
    std::vector<int64_t> cursor(start);
    for (int64_t b = 0; b < B; ++b) order[cursor[lp[b]]++] = b;
  }

  auto leaf_base = [&](int64_t l) -> const uint8_t* {
    return wp + l * d_out * K;
  };

  const unsigned nthr =
      std::min<unsigned>(std::thread::hardware_concurrency(), 16);
  std::atomic<int64_t> next{0};
  auto worker = [&]() {
    for (;;) {
      const int64_t i = next.fetch_add(1);
      if (i >= B) return;
      const int64_t b = order[i];
      const int64_t l = lp[b];
      const float* xb = xp + b * d_in;
      const uint8_t* lbase = leaf_base(l);
      float* ob = op + b * d_out;
      for (int64_t o = 0; o < d_out; ++o) {
        ob[o] = packed_dot(xb, lbase + o * K, d_in);
      }
    }
  };

  if (nthr <= 1 || B <= 1) {
    worker();
  } else {
    std::vector<std::thread> threads;
    threads.reserve(nthr);
    for (unsigned t = 0; t < nthr; ++t) threads.emplace_back(worker);
    for (auto& t : threads) t.join();
  }
  return out;
}

}  // namespace ternary

TORCH_LIBRARY(ternary_packed, m) {
  m.def("ternary_mm(Tensor x, Tensor packed_w, Tensor leaf_idx) -> Tensor");
}

TORCH_LIBRARY_IMPL(ternary_packed, CPU, m) {
  m.impl("ternary_mm", &ternary::ternary_mm_cpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mps_supported", &ternary::mps_supported,
        "True when a Metal device is available");
}
