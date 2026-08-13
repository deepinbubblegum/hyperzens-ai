// Multi-Tree Conditional Matrix Multiplication (CMM) — CPU hard-routing kernel.
//
// UltraFastBERT / Multi-Tree CMM (Belcak & Wattenhofer lineage)
// ------------------------------------------------------------
// Inference evaluates K independent shallow binary trees of depth ``d``
// instead of one deep tree. Per token:
//
//   for k in 0..K-1:
//       ℓ_k = hard_walk(tree_k, x)          // d router dots, branchless
//       y_k = x @ W[k, ℓ_k] + b[k, ℓ_k]     // one leaf GEMV
//   y    = Σ_k y_k            (reduce = sum)
//     or   concat_k(y_k)      (reduce = concat)
//
// Layout (tree-major, heap-order nodes — L1/L2 friendly):
//   w_router : (K, R, D_in)     R = 2^d - 1
//   b_router : (K, R)
//   w_leaf   : (K, L, D_in, D_out)   L = 2^d, row-major (D_in, D_out)
//   b_leaf   : (K, L, D_out)
//
// Why this layout: for a fixed token, x stays hot in L1 while we walk
// tree 0..K-1. Each tree's routers occupy a contiguous (R, D_in) slab;
// the selected leaf's (D_in, D_out) block is one contiguous stream.
//
// SIMD: AVX2 (+FMA) on x86_64, NEON on Apple Silicon / AArch64.
// Threading: at::parallel_for over tokens (no OpenMP requirement).
// GIL: released via py::gil_scoped_release.
//
// Backward compatible: ``fff_hard_forward_cpu`` is K=1, 2-D tensors.

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cstdint>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// Portable auto-vectorization hints (scalar tails / non-SIMD builds).
#if defined(_OPENMP)
#define FFF_PRAGMA_SIMD _Pragma("omp simd")
#elif defined(__clang__)
#define FFF_PRAGMA_SIMD _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define FFF_PRAGMA_SIMD _Pragma("GCC ivdep")
#else
#define FFF_PRAGMA_SIMD
#endif

namespace {

constexpr int64_t kMaxDepth = 16;  // 2^16 leaves/tree is already pathological.

inline int64_t checked_depth(int64_t depth) {
  if (depth < 1 || depth > kMaxDepth) {
    throw std::invalid_argument(
        "cmm_hard_forward_cpu: depth must be in [1, 16]");
  }
  return depth;
}

inline int64_t checked_num_trees(int64_t num_trees) {
  if (num_trees < 1 || num_trees > 256) {
    throw std::invalid_argument(
        "cmm_hard_forward_cpu: num_trees must be in [1, 256]");
  }
  return num_trees;
}

// ---------------------------------------------------------------------------
// SIMD primitives (float32)
// ---------------------------------------------------------------------------

#if defined(__AVX2__)
inline float hsum256(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  const __m128 shuf = _mm_movehdup_ps(s);
  s = _mm_add_ps(s, shuf);
  const __m128 shuf2 = _mm_movehl_ps(shuf, s);
  s = _mm_add_ss(s, shuf2);
  return _mm_cvtss_f32(s);
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
inline float hsum128(float32x4_t v) {
  const float32x2_t pair = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  const float32x2_t sum = vpadd_f32(pair, pair);
  return vget_lane_f32(sum, 0);
}
#endif

/// Dot product ``a · b`` of length ``n`` (router score body).
inline float dot_f32(const float* __restrict__ a, const float* __restrict__ b,
                     int64_t n) {
  int64_t i = 0;
  float acc = 0.0f;

#if defined(__AVX2__)
  __m256 vacc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
    vacc = _mm256_fmadd_ps(va, vb, vacc);
#else
    vacc = _mm256_add_ps(vacc, _mm256_mul_ps(va, vb));
#endif
  }
  acc += hsum256(vacc);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t vacc = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    const float32x4_t va = vld1q_f32(a + i);
    const float32x4_t vb = vld1q_f32(b + i);
    vacc = vfmaq_f32(vacc, va, vb);
  }
  acc += hsum128(vacc);
#endif

  for (; i < n; ++i) {
    acc += a[i] * b[i];
  }
  return acc;
}

/// ``y[o] = src[o]`` for ``o in [0, n)``.
inline void copy_f32(float* __restrict__ y, const float* __restrict__ src,
                     int64_t n) {
  int64_t o = 0;
#if defined(__AVX2__)
  for (; o + 8 <= n; o += 8) {
    _mm256_storeu_ps(y + o, _mm256_loadu_ps(src + o));
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  for (; o + 4 <= n; o += 4) {
    vst1q_f32(y + o, vld1q_f32(src + o));
  }
#endif
  for (; o < n; ++o) {
    y[o] = src[o];
  }
}

/// ``y[o] += src[o]``.
inline void add_f32(float* __restrict__ y, const float* __restrict__ src,
                    int64_t n) {
  int64_t o = 0;
#if defined(__AVX2__)
  for (; o + 8 <= n; o += 8) {
    const __m256 vy = _mm256_loadu_ps(y + o);
    const __m256 vs = _mm256_loadu_ps(src + o);
    _mm256_storeu_ps(y + o, _mm256_add_ps(vy, vs));
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  for (; o + 4 <= n; o += 4) {
    const float32x4_t vy = vld1q_f32(y + o);
    const float32x4_t vs = vld1q_f32(src + o);
    vst1q_f32(y + o, vaddq_f32(vy, vs));
  }
#endif
  for (; o < n; ++o) {
    y[o] += src[o];
  }
}

/// ``y[o] += xi * row[o]`` — one contiguous W row (inner D_out).
inline void axpy_row_f32(float* __restrict__ y, const float* __restrict__ row,
                         float xi, int64_t n) {
  int64_t o = 0;
#if defined(__AVX2__)
  const __m256 vx = _mm256_set1_ps(xi);
  for (; o + 8 <= n; o += 8) {
    const __m256 vw = _mm256_loadu_ps(row + o);
    __m256 vy = _mm256_loadu_ps(y + o);
#if defined(__FMA__)
    vy = _mm256_fmadd_ps(vx, vw, vy);
#else
    vy = _mm256_add_ps(vy, _mm256_mul_ps(vx, vw));
#endif
    _mm256_storeu_ps(y + o, vy);
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t vx = vdupq_n_f32(xi);
  for (; o + 4 <= n; o += 4) {
    const float32x4_t vw = vld1q_f32(row + o);
    float32x4_t vy = vld1q_f32(y + o);
    vy = vfmaq_f32(vy, vx, vw);
    vst1q_f32(y + o, vy);
  }
#endif
  for (; o < n; ++o) {
    y[o] += xi * row[o];
  }
}

/// Leaf GEMV: ``y = x @ W + b`` (overwrite) or ``y += x @ W + b`` (accumulate).
///
/// W is row-major ``(D_in, D_out)`` so each inner row is a contiguous
/// D_out-vector — the hot stream for L1 / SIMD.
inline void leaf_gemv_f32(float* __restrict__ y, const float* __restrict__ x,
                          const float* __restrict__ w, const float* __restrict__ b,
                          int64_t d_in, int64_t d_out, bool accumulate) {
  if (accumulate) {
    add_f32(y, b, d_out);
  } else {
    copy_f32(y, b, d_out);
  }
  for (int64_t i = 0; i < d_in; ++i) {
    axpy_row_f32(y, w + i * d_out, x[i], d_out);
  }
}

/// Branchless heap walk: ``s > 0 → right (2i+2), else left (2i+1)``.
/// Identical to PyTorch ``node = (node << 1) + 1 + (score > 0)``.
inline int64_t hard_walk_leaf(const float* __restrict__ x,
                              const float* __restrict__ wr,
                              const float* __restrict__ br, int64_t d_in,
                              int64_t depth, int64_t num_leaves) {
  int64_t node = 0;
  for (int64_t d = 0; d < depth; ++d) {
    const float score = br[node] + dot_f32(wr + node * d_in, x, d_in);
    // Branchless child select — shallower trees already cut mispredicts;
    // removing the if here keeps the decoder pipeline full on AVX/NEON.
    const int64_t go_right = static_cast<int64_t>(score > 0.0f);
    node = (node << 1) + 1 + go_right;
  }
  const int64_t leaf_base = num_leaves - 1;
  int64_t leaf = node - leaf_base;
  if (leaf < 0) {
    leaf = 0;
  } else if (leaf >= num_leaves) {
    leaf = num_leaves - 1;
  }
  return leaf;
}

enum class CmmReduce : int64_t { Sum = 0, Concat = 1 };

struct CmmLayout {
  int64_t N;
  int64_t D_in;
  int64_t D_leaf;  // leaf output width
  int64_t D_out;   // packed output width (D_leaf or K*D_leaf)
  int64_t K;
  int64_t depth;
  int64_t num_leaves;
  int64_t num_routers;
  CmmReduce reduce;
};

void cmm_hard_kernel(const CmmLayout& L, const float* __restrict__ x_ptr,
                     const float* __restrict__ wr_ptr,
                     const float* __restrict__ br_ptr,
                     const float* __restrict__ wl_ptr,
                     const float* __restrict__ bl_ptr,
                     float* __restrict__ y_ptr) {
  const int64_t tree_router_stride = L.num_routers * L.D_in;
  const int64_t tree_bias_stride = L.num_routers;
  const int64_t leaf_mat_stride = L.D_in * L.D_leaf;
  const int64_t tree_leaf_w_stride = L.num_leaves * leaf_mat_stride;
  const int64_t tree_leaf_b_stride = L.num_leaves * L.D_leaf;

  at::parallel_for(0, L.N, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; ++n) {
      const float* xn = x_ptr + n * L.D_in;
      float* yn = y_ptr + n * L.D_out;

      for (int64_t k = 0; k < L.K; ++k) {
        const float* wr_k = wr_ptr + k * tree_router_stride;
        const float* br_k = br_ptr + k * tree_bias_stride;
        const int64_t leaf = hard_walk_leaf(xn, wr_k, br_k, L.D_in, L.depth,
                                            L.num_leaves);

        const float* wl =
            wl_ptr + k * tree_leaf_w_stride + leaf * leaf_mat_stride;
        const float* bl = bl_ptr + k * tree_leaf_b_stride + leaf * L.D_leaf;

        if (L.reduce == CmmReduce::Concat) {
          leaf_gemv_f32(yn + k * L.D_leaf, xn, wl, bl, L.D_in, L.D_leaf,
                        /*accumulate=*/false);
        } else {
          leaf_gemv_f32(yn, xn, wl, bl, L.D_in, L.D_leaf,
                        /*accumulate=*/k > 0);
        }
      }
    }
  });
}

void check_cpu_f32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cpu(), name, " must be on CPU");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
}

}  // namespace

/// Multi-tree CMM hard forward (CPU, float32).
///
/// Parameters
/// ----------
/// x:         (N, D_in)
/// w_router:  (K, R, D_in) or legacy (R, D_in) ⇒ K=1
/// b_router:  (K, R)       or legacy (R,)
/// w_leaf:    (K, L, D_in, D_out_leaf) or legacy (L, D_in, D_out)
/// b_leaf:    (K, L, D_out_leaf)       or legacy (L, D_out)
/// depth:     tree depth d  (L = 2^d, R = 2^d - 1)
/// reduce:    0 = sum over trees (output D_out_leaf)
///            1 = concat tree slices (output K * D_out_leaf)
///
/// Returns y of shape (N, D_out).
torch::Tensor cmm_hard_forward_cpu(torch::Tensor x, torch::Tensor w_router,
                                   torch::Tensor b_router, torch::Tensor w_leaf,
                                   torch::Tensor b_leaf, int64_t depth,
                                   int64_t reduce) {
  depth = checked_depth(depth);
  TORCH_CHECK(reduce == 0 || reduce == 1,
              "cmm_hard_forward_cpu: reduce must be 0 (sum) or 1 (concat)");

  check_cpu_f32(x, "x");
  check_cpu_f32(w_router, "w_router");
  check_cpu_f32(b_router, "b_router");
  check_cpu_f32(w_leaf, "w_leaf");
  check_cpu_f32(b_leaf, "b_leaf");

  x = x.contiguous();
  w_router = w_router.contiguous();
  b_router = b_router.contiguous();
  w_leaf = w_leaf.contiguous();
  b_leaf = b_leaf.contiguous();

  TORCH_CHECK(x.dim() == 2, "cmm_hard_forward_cpu: x must be (N, D_in)");

  const int64_t N = x.size(0);
  const int64_t D_in = x.size(1);
  const int64_t num_leaves = 1LL << depth;
  const int64_t num_routers = num_leaves - 1;

  int64_t K = 1;
  int64_t D_leaf = 0;

  if (w_router.dim() == 2 && w_leaf.dim() == 3) {
    // Legacy single-tree: (R, D_in), (L, D_in, D_out).
    TORCH_CHECK(w_router.size(0) == num_routers && w_router.size(1) == D_in,
                "w_router shape mismatch with depth / D_in");
    TORCH_CHECK(b_router.dim() == 1 && b_router.size(0) == num_routers,
                "b_router must be (R,)");
    TORCH_CHECK(w_leaf.size(0) == num_leaves && w_leaf.size(1) == D_in,
                "w_leaf shape mismatch");
    D_leaf = w_leaf.size(2);
    TORCH_CHECK(b_leaf.dim() == 2 && b_leaf.size(0) == num_leaves &&
                    b_leaf.size(1) == D_leaf,
                "b_leaf must be (L, D_out)");
    K = 1;
  } else if (w_router.dim() == 3 && w_leaf.dim() == 4) {
    K = checked_num_trees(w_router.size(0));
    TORCH_CHECK(w_router.size(1) == num_routers && w_router.size(2) == D_in,
                "w_router must be (K, R, D_in)");
    TORCH_CHECK(b_router.dim() == 2 && b_router.size(0) == K &&
                    b_router.size(1) == num_routers,
                "b_router must be (K, R)");
    TORCH_CHECK(w_leaf.size(0) == K && w_leaf.size(1) == num_leaves &&
                    w_leaf.size(2) == D_in,
                "w_leaf must be (K, L, D_in, D_out_leaf)");
    D_leaf = w_leaf.size(3);
    TORCH_CHECK(b_leaf.dim() == 3 && b_leaf.size(0) == K &&
                    b_leaf.size(1) == num_leaves && b_leaf.size(2) == D_leaf,
                "b_leaf must be (K, L, D_out_leaf)");
  } else {
    TORCH_CHECK(false,
                "cmm_hard_forward_cpu: expected w_router (K,R,D_in) + "
                "w_leaf (K,L,D_in,D_out) or legacy 2-D/3-D single-tree shapes");
  }

  const CmmReduce red = (reduce == 1) ? CmmReduce::Concat : CmmReduce::Sum;
  const int64_t D_out = (red == CmmReduce::Concat) ? (K * D_leaf) : D_leaf;

  auto y = torch::empty({N, D_out}, x.options());
  CmmLayout layout{N,     D_in, D_leaf, D_out, K,
                   depth, num_leaves, num_routers, red};
  cmm_hard_kernel(layout, x.data_ptr<float>(), w_router.data_ptr<float>(),
                  b_router.data_ptr<float>(), w_leaf.data_ptr<float>(),
                  b_leaf.data_ptr<float>(), y.data_ptr<float>());
  return y;
}

/// Legacy single-tree hard forward — identical to ``cmm`` with K=1, reduce=sum.
torch::Tensor fff_hard_forward_cpu(torch::Tensor x, torch::Tensor w_router,
                                   torch::Tensor b_router, torch::Tensor w_leaf,
                                   torch::Tensor b_leaf, int64_t depth) {
  return cmm_hard_forward_cpu(std::move(x), std::move(w_router),
                              std::move(b_router), std::move(w_leaf),
                              std::move(b_leaf), depth, /*reduce=*/0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cmm_hard_forward_cpu", &cmm_hard_forward_cpu,
      "Multi-tree CMM hard-routing forward (CPU float32, K shallow trees)",
      py::arg("x"), py::arg("w_router"), py::arg("b_router"), py::arg("w_leaf"),
      py::arg("b_leaf"), py::arg("depth"), py::arg("reduce") = 0,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "fff_hard_forward_cpu", &fff_hard_forward_cpu,
      "FFF hard-routing forward (CPU, float32) — single tree, K=1 wrapper",
      py::arg("x"), py::arg("w_router"), py::arg("b_router"), py::arg("w_leaf"),
      py::arg("b_leaf"), py::arg("depth"),
      py::call_guard<py::gil_scoped_release>());
}
