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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

at::Tensor route_and_matmul_cpu(const at::Tensor& x, const at::Tensor& router_w,
                                const at::Tensor& router_b,
                                const at::Tensor& packed_w, int64_t depth,
                                int64_t activation_bits,
                                const c10::optional<at::Tensor>& leaf_bias,
                                double eps) {
  TORCH_CHECK(x.is_cpu(), "fused_ternary_fff expects a CPU tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be (B, d_in)");
  TORCH_CHECK(router_w.is_cpu() && router_w.is_contiguous(),
              "router_w must be a contiguous CPU tensor");
  TORCH_CHECK(router_w.scalar_type() == at::kFloat, "router_w must be float32");
  TORCH_CHECK(router_b.is_cpu() && router_b.is_contiguous(),
              "router_b must be a contiguous CPU tensor");
  TORCH_CHECK(router_b.scalar_type() == at::kFloat, "router_b must be float32");
  TORCH_CHECK(packed_w.is_cpu() && packed_w.is_contiguous(),
              "packed_w must be a contiguous CPU tensor");
  TORCH_CHECK(packed_w.scalar_type() == at::kByte, "packed_w must be uint8");
  TORCH_CHECK(packed_w.dim() == 3, "packed_w must be (L, d_out, K)");
  TORCH_CHECK(depth >= 0 && depth < 31, "depth must be in [0, 31)");
  TORCH_CHECK(router_w.dim() == 2 && router_w.size(1) == x.size(1),
              "router_w must be (2^depth - 1, d_in)");
  TORCH_CHECK(router_b.dim() == 1, "router_b must be (2^depth - 1,)");

  const int64_t d_in = x.size(1);
  TORCH_CHECK(d_in % 4 == 0, "d_in must be a multiple of 4");
  const int64_t L = packed_w.size(0);
  TORCH_CHECK(L == (int64_t{1} << depth), "packed_w leaves must equal 2^depth");
  const int64_t N = L - 1;
  TORCH_CHECK(router_w.size(0) == N && router_b.size(0) == N,
              "router must have 2^depth - 1 nodes");
  const int64_t d_out = packed_w.size(1);
  TORCH_CHECK(packed_w.size(2) == d_in / 4, "packed_w K must equal d_in/4");
  if (leaf_bias.has_value()) {
    TORCH_CHECK(leaf_bias->is_cpu() && leaf_bias->is_contiguous(),
                "leaf_bias must be a contiguous CPU tensor");
    TORCH_CHECK(leaf_bias->scalar_type() == at::kFloat,
                "leaf_bias must be float32");
    TORCH_CHECK(leaf_bias->dim() == 2 && leaf_bias->size(0) == L &&
                    leaf_bias->size(1) == d_out,
                "leaf_bias must be (L, d_out)");
  }

  build_decode_tables();

  const int64_t B = x.size(0);
  const int64_t K = packed_w.size(2);
  const int64_t max_q =
      (activation_bits > 0 && activation_bits < 32)
          ? (int64_t{1} << (activation_bits - 1)) - 1
          : 0;
  const float epsf = static_cast<float>(eps);

  auto out = at::empty({B, d_out}, x.options());
  const float* xp = x.data_ptr<float>();
  const float* rwp = router_w.data_ptr<float>();
  const float* rbp = router_b.data_ptr<float>();
  const uint8_t* wp = packed_w.data_ptr<uint8_t>();
  const float* lbp = leaf_bias.has_value() ? leaf_bias->data_ptr<float>() : nullptr;
  float* op = out.data_ptr<float>();

  // Aligned scratch holding each row's AbsMax-quantized activations.
  float* qscratch = nullptr;
  TORCH_CHECK(posix_memalign(reinterpret_cast<void**>(&qscratch), 64,
                             sizeof(float) * B * d_in) == 0,
              "posix_memalign failed");

  std::atomic<int64_t> next{0};
  const unsigned nthr =
      std::min<unsigned>(std::thread::hardware_concurrency(), 16);
  auto worker = [&]() {
    for (;;) {
      const int64_t i = next.fetch_add(1);
      if (i >= B) return;
      const float* xb = xp + i * d_in;

      // --- cooperative-free per-row routing (matches _routing_forward) ---
      int64_t node = 0;
      for (int64_t lvl = 0; lvl < depth; ++lvl) {
        const float* rw = rwp + node * d_in;
        float32x4_t acc4 = vdupq_n_f32(0.0f);
        int64_t k = 0;
        for (; k + 4 <= d_in; k += 4) {
          acc4 = vmlaq_f32(acc4, vld1q_f32(xb + k), vld1q_f32(rw + k));
        }
        float logit = vaddvq_f32(acc4) + rbp[node];
        for (; k < d_in; ++k) logit += xb[k] * rw[k];
        node = 2 * node + 1 + ((logit >= 0.0f) ? 1 : 0);
      }
      const int64_t leaf = node - ((int64_t{1} << depth) - 1);

      // --- AbsMax activation quantization into scratch (matches absmax_quantize)
      float* qb = qscratch + i * d_in;
      if (max_q != 0) {
        float maxabs = 0.0f;
        for (int64_t k = 0; k < d_in; ++k)
          maxabs = std::max(maxabs, std::fabs(xb[k]));
        const float scale = std::max(maxabs, epsf);
        const float mqf = static_cast<float>(max_q);
        for (int64_t k = 0; k < d_in; ++k) {
          float q = std::rint(xb[k] / scale * mqf);  // round-half-to-even
          q = std::clamp(q, -mqf - 1.0f, mqf);
          qb[k] = q / mqf * scale;
        }
      } else {
        std::memcpy(qb, xb, sizeof(float) * d_in);
      }

      // --- leaf matmul (packed ternary, pure add/sub) ---
      const uint8_t* lbase = wp + leaf * d_out * K;
      const float* lbrow = lbp ? lbp + leaf * d_out : nullptr;
      float* ob = op + i * d_out;
      for (int64_t o = 0; o < d_out; ++o) {
        float v = packed_dot(qb, lbase + o * K, d_in);
        if (lbrow) v += lbrow[o];
        ob[o] = v;
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
  std::free(qscratch);
  return out;
}

}  // namespace ternary

TORCH_LIBRARY(ternary_packed, m) {
  m.def("ternary_mm(Tensor x, Tensor packed_w, Tensor leaf_idx) -> Tensor");
  m.def("fused_ternary_fff(Tensor x, Tensor router_w, Tensor router_b, "
        "Tensor packed_leaf_w, int depth, int activation_bits, "
        "Tensor? leaf_bias, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(ternary_packed, CPU, m) {
  m.impl("ternary_mm", &ternary::ternary_mm_cpu);
  m.impl("fused_ternary_fff", &ternary::route_and_matmul_cpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mps_supported", &ternary::mps_supported,
        "True when a Metal device is available");
}
