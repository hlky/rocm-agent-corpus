#include <hip/hip_runtime.h>

// Template-only scaffold for top-p-nucleus-sampling.
// Summary: Extend top-k/logits work to top-p sampling, where sorted or partially sorted probabilities are accumulated until a probability threshold is reached.
// Baseline strategy: Apply softmax, sort probabilities with hipCUB/rocThrust or framework ops, scan cumulative probability, then sample from the selected nucleus.

namespace {

__global__ void top_p_nucleus_sampling_baseline_kernel() {
  // TODO: replace with the baseline implementation before measuring.
}

}  // namespace

extern "C" void launch_baseline(hipStream_t stream) {
  top_p_nucleus_sampling_baseline_kernel<<<1, 1, 0, stream>>>();
}
