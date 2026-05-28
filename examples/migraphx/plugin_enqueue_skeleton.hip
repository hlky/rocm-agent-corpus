#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// MIGraphX plugin enqueue sketch.
//
// This file intentionally avoids depending on MIGraphX headers so it can be
// read as a compile-minded HIP boundary example. A real plugin would implement
// IPluginV2DynamicExt or the current MIGraphX plugin interface, then call
// launch_fused_bias_relu_plugin_kernel() from enqueue() using MIGraphX's stream.
//
// Agent checklist:
// - Confirm shape inference for every optimization profile.
// - Match supportsFormatCombination() to the kernel's dtype/layout contract.
// - Keep workspace explicit; do not allocate in enqueue().
// - Launch on the provided stream and avoid device synchronization.
// - Compare against MIGraphX native fusion/tactic before claiming a win.
// - Label kernel-only timings separately from engine enqueue timings.

struct FusedBiasReluLaunchParams {
  const float* input;
  const float* bias;
  float* output;
  std::int32_t elements;
  std::int32_t channels;
};

__device__ __forceinline__ float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

__global__ void fused_bias_relu_kernel(FusedBiasReluLaunchParams params) {
  const std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= params.elements) {
    return;
  }

  // Contract: dense linear tensor where the last dimension is channels.
  // A MIGraphX plugin would only advertise formats that satisfy this contract.
  const std::int32_t c = i % params.channels;
  params.output[i] = relu(params.input[i] + params.bias[c]);
}

hipError_t launch_fused_bias_relu_plugin_kernel(
    const float* input,
    const float* bias,
    float* output,
    std::int32_t elements,
    std::int32_t channels,
    void* workspace,
    std::size_t workspace_bytes,
    hipStream_t stream) {
  (void)workspace;
  (void)workspace_bytes;

  if (input == nullptr || bias == nullptr || output == nullptr ||
      elements < 0 || channels <= 0) {
    return hipErrorInvalidValue;
  }

  constexpr int kBlockSize = 256;
  const int grid = (elements + kBlockSize - 1) / kBlockSize;
  if (grid == 0) {
    return hipSuccess;
  }

  FusedBiasReluLaunchParams params{input, bias, output, elements, channels};
  fused_bias_relu_kernel<<<grid, kBlockSize, 0, stream>>>(params);
  return hipGetLastError();
}

// Pseudocode for the real MIGraphX enqueue boundary:
//
// bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut,
//                                int32_t nbInputs, int32_t nbOutputs) noexcept {
//   // Example contract for this skeleton:
//   // - input, bias, and output are FP32.
//   // - tensors use MIGraphX's dense linear format.
//   // - bias is one-dimensional with length == channels.
//   // A real plugin should reject every dtype/format pair the kernel has not
//   // been written and tested to handle.
//   return inOut[pos].type == DataType::kFLOAT &&
//          inOut[pos].format == TensorFormat::kLINEAR;
// }
//
// int32_t enqueue(..., void const* const* inputs, void* const* outputs,
//                 void* workspace, hipStream_t stream) noexcept {
//   const auto dims = inputDesc[0].dims;
//   const int32_t elements = volume(dims);
//   const int32_t channels = dims.d[dims.nbDims - 1];
//   return launch_fused_bias_relu_plugin_kernel(
//       static_cast<const float*>(inputs[0]),
//       static_cast<const float*>(inputs[1]),
//       static_cast<float*>(outputs[0]),
//       elements, channels, workspace, workspace_bytes_, stream);
// }
