// Fast Feedforward hard-routing CPU kernel.
//
// For each token, walk the binary decision tree with threshold
//   go_right = (w_router[i] · x + b_router[i]) > 0
// then evaluate a single leaf affine map.
// Parallelism: at::parallel_for over the flattened token axis.

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

inline int64_t checked_depth(int64_t depth) {
  if (depth < 1 || depth > 30) {
    throw std::invalid_argument(
        "fff_hard_forward_cpu: depth must be in [1, 30]");
  }
  return depth;
}

}  // namespace

/// Hard-routing FFF forward on CPU (float32).
///
/// x:          (N, D_in) contiguous float32  — flattened tokens
/// w_router:   (R, D_in)  R = 2^depth - 1
/// b_router:   (R,)
/// w_leaf:     (L, D_in, D_out)  L = 2^depth
/// b_leaf:     (L, D_out)
/// depth:      tree depth
///
/// Returns y:  (N, D_out)
torch::Tensor fff_hard_forward_cpu(
    torch::Tensor x,
    torch::Tensor w_router,
    torch::Tensor b_router,
    torch::Tensor w_leaf,
    torch::Tensor b_leaf,
    int64_t depth) {
  depth = checked_depth(depth);

  TORCH_CHECK(x.is_cpu(), "fff_hard_forward_cpu: x must be on CPU");
  TORCH_CHECK(w_router.is_cpu() && b_router.is_cpu() && w_leaf.is_cpu() &&
                  b_leaf.is_cpu(),
              "fff_hard_forward_cpu: all tensors must be on CPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat &&
                  w_router.scalar_type() == at::kFloat &&
                  b_router.scalar_type() == at::kFloat &&
                  w_leaf.scalar_type() == at::kFloat &&
                  b_leaf.scalar_type() == at::kFloat,
              "fff_hard_forward_cpu: only float32 is supported");

  // Contiguous memory — required for raw pointer walks.
  x = x.contiguous();
  w_router = w_router.contiguous();
  b_router = b_router.contiguous();
  w_leaf = w_leaf.contiguous();
  b_leaf = b_leaf.contiguous();

  TORCH_CHECK(x.dim() == 2, "fff_hard_forward_cpu: x must be (N, D_in)");
  TORCH_CHECK(w_router.dim() == 2, "w_router must be (R, D_in)");
  TORCH_CHECK(b_router.dim() == 1, "b_router must be (R,)");
  TORCH_CHECK(w_leaf.dim() == 3, "w_leaf must be (L, D_in, D_out)");
  TORCH_CHECK(b_leaf.dim() == 2, "b_leaf must be (L, D_out)");

  const int64_t N = x.size(0);
  const int64_t D_in = x.size(1);
  const int64_t num_leaves = 1LL << depth;
  const int64_t num_routers = num_leaves - 1;
  const int64_t D_out = w_leaf.size(2);

  TORCH_CHECK(w_router.size(0) == num_routers && w_router.size(1) == D_in,
              "w_router shape mismatch with depth / D_in");
  TORCH_CHECK(b_router.size(0) == num_routers, "b_router length mismatch");
  TORCH_CHECK(w_leaf.size(0) == num_leaves && w_leaf.size(1) == D_in,
              "w_leaf shape mismatch with depth / D_in");
  TORCH_CHECK(b_leaf.size(0) == num_leaves && b_leaf.size(1) == D_out,
              "b_leaf shape mismatch");

  auto y = torch::empty({N, D_out}, x.options());

  const float* __restrict__ x_ptr = x.data_ptr<float>();
  const float* __restrict__ wr_ptr = w_router.data_ptr<float>();
  const float* __restrict__ br_ptr = b_router.data_ptr<float>();
  const float* __restrict__ wl_ptr = w_leaf.data_ptr<float>();
  const float* __restrict__ bl_ptr = b_leaf.data_ptr<float>();
  float* __restrict__ y_ptr = y.data_ptr<float>();

  const int64_t leaf_base = num_leaves - 1;  // heap leaf index offset
  const int64_t leaf_stride = D_in * D_out;

  // Grain size 0 → ATen picks a sensible split for the thread pool.
  at::parallel_for(0, N, /*grain_size=*/0, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; ++n) {
      const float* xn = x_ptr + n * D_in;
      float* yn = y_ptr + n * D_out;

      // --- Tree walk: root = 0 ---
      int64_t node = 0;
      for (int64_t d = 0; d < depth; ++d) {
        const float* wr = wr_ptr + node * D_in;
        float score = br_ptr[node];
        // s = w · x + b  (naive gemv; depth is small, D_in dominates cost at leaf)
        for (int64_t i = 0; i < D_in; ++i) {
          score += wr[i] * xn[i];
        }
        // s > 0 → right (2i+2), else left (2i+1)
        if (score > 0.0f) {
          node = 2 * node + 2;
        } else {
          node = 2 * node + 1;
        }
      }

      const int64_t leaf = node - leaf_base;
      // Defensive clamp (should never trip for a complete binary tree).
      const int64_t leaf_id =
          (leaf < 0) ? 0 : (leaf >= num_leaves ? (num_leaves - 1) : leaf);

      const float* wl = wl_ptr + leaf_id * leaf_stride;
      const float* bl = bl_ptr + leaf_id * D_out;

      // y = x @ W_leaf[leaf] + b   with W shaped (D_in, D_out), row-major
      for (int64_t o = 0; o < D_out; ++o) {
        float acc = bl[o];
        for (int64_t i = 0; i < D_in; ++i) {
          acc += xn[i] * wl[i * D_out + o];
        }
        yn[o] = acc;
      }
    }
  });

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fff_hard_forward_cpu",
      &fff_hard_forward_cpu,
      "FFF hard-routing forward (CPU, float32, parallel over tokens)",
      py::arg("x"),
      py::arg("w_router"),
      py::arg("b_router"),
      py::arg("w_leaf"),
      py::arg("b_leaf"),
      py::arg("depth"));
}
