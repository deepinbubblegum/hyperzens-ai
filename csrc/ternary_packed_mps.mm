/* Metal (MPS) implementation of the packed-ternary grouped matmul.
 *
 * The shader source is embedded at build time by `csrc/build_ext.py` into
 * `csrc/metal_ternary_kernel.h` (canonical source: `csrc/ternary_add.metal`).
 *
 * The compute encoder is dispatched on torch's current MPS stream command
 * buffer, so ordering with surrounding torch ops is preserved and no host
 * round-trip is needed. The kernel reads only the routed leaf's packed rows
 * and never materializes per-branch buffers.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <torch/library.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include "ternary_packed.h"
#include "metal_ternary_kernel.h"

namespace ternary {
namespace {

id<MTLComputePipelineState> getPipeline(id<MTLDevice> dev) {
  static id<MTLComputePipelineState> ps = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
      opts.mathMode = MTLMathModeFast;
    } else {
      opts.fastMathEnabled = YES;
    }
    id<MTLLibrary> lib =
        [dev newLibraryWithSource:@(TERNARY_ADD_METAL_SOURCE)
                          options:opts error:&err];
    TORCH_CHECK(lib != nil, "Metal library compile failed: ",
                err.localizedDescription.UTF8String);
    id<MTLFunction> fn = [lib newFunctionWithName:@"ternary_mm"];
    TORCH_CHECK(fn != nil, "Metal function 'ternary_mm' not found");
    ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(ps != nil, "Metal pipeline state failed: ",
                err.localizedDescription.UTF8String);
  });
  return ps;
}

void setBuffer(id<MTLComputeCommandEncoder> enc, const at::Tensor& t,
               uint32_t index) {
  const size_t offset = t.storage_offset() * t.element_size();
  [enc setBuffer:(__bridge id<MTLBuffer>)t.data_ptr() offset:offset atIndex:index];
}

}  // namespace

bool mps_supported() {
  return MTLCreateSystemDefaultDevice() != nil;
}

at::Tensor ternary_mm_mps(const at::Tensor& x, const at::Tensor& packed_w,
                          const at::Tensor& leaf_idx) {
  TORCH_CHECK(x.is_mps(), "ternary_mm_mps expects MPS tensors");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(packed_w.is_contiguous(), "packed_w must be contiguous");
  TORCH_CHECK(leaf_idx.is_contiguous(), "leaf_idx must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(packed_w.scalar_type() == at::kByte, "packed_w must be uint8");
  TORCH_CHECK(leaf_idx.scalar_type() == at::kInt, "leaf_idx must be int32");
  TORCH_CHECK(x.dim() == 2, "x must be (B, d_in)");
  TORCH_CHECK(packed_w.dim() == 3, "packed_w must be (L, d_out, K)");
  TORCH_CHECK(x.size(1) % 4 == 0, "d_in must be a multiple of 4");
  TORCH_CHECK(packed_w.size(2) == x.size(1) / 4,
              "packed_w columns must equal ceil(d_in / 4)");

  const uint32_t B = static_cast<uint32_t>(x.size(0));
  const uint32_t d_in = static_cast<uint32_t>(x.size(1));
  const uint32_t d_out = static_cast<uint32_t>(packed_w.size(1));
  const uint32_t K = static_cast<uint32_t>(packed_w.size(2));
  TORCH_CHECK(d_out <= 1024, "d_out exceeds Metal threadgroup width limit");

  at::Tensor out = at::empty({B, d_out}, x.options());

  id<MTLDevice> dev = at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> ps = getPipeline(dev);

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  // The MPS stream defers committing its command buffer until a sync point, so
  // pending torch ops (e.g. `.to("mps")` copies of x / packed_w) may still be
  // uncommitted. Force the stream to commit/flush now so those ops are enqueued
  // on the command queue ahead of ours, then draw our own command buffer from
  // the same serial queue to keep ordering with subsequent torch ops.
  stream->synchronize(at::mps::SyncType::COMMIT);
  id<MTLCommandQueue> queue = stream->commandQueue();
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  cb.label = @"ternary_mm";

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ps];
  setBuffer(enc, x, 0);
  setBuffer(enc, packed_w, 1);
  setBuffer(enc, leaf_idx, 2);
  setBuffer(enc, out, 3);
  [enc setBytes:&d_in length:sizeof(d_in) atIndex:4];
  [enc setBytes:&d_out length:sizeof(d_out) atIndex:5];
  [enc setBytes:&K length:sizeof(K) atIndex:6];

  MTLSize threadsPerGroup = MTLSizeMake(d_out, 1, 1);
  MTLSize threadgroups = MTLSizeMake(B, 1, 1);
  [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
  [enc endEncoding];
  [cb commit];

  return out;
}

}  // namespace ternary

TORCH_LIBRARY_IMPL(ternary_packed, MPS, m) {
  m.impl("ternary_mm", &ternary::ternary_mm_mps);
}
