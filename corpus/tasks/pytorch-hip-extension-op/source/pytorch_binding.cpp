#include <stdexcept>

// Compile-oriented placeholder for a PyTorch C++ binding.
// Replace TensorSketch with at::Tensor and use AT_DISPATCH_FLOATING_TYPES when
// building the real extension. The checks show the boundary that must be kept
// explicit in benchmark metadata.

struct TensorSketch {
  float* data;
  int sizes[2];
  int strides[2];
  bool is_cuda;
  bool is_float32;
};

extern "C" void launch_rowwise_sum_extension(const float* x,
                                              float* y,
                                              int rows,
                                              int cols,
                                              int stride_row,
                                              int stride_col,
                                              void* stream);

void check_rowwise_input(const TensorSketch& x) {
  if (!x.is_cuda) {
    throw std::invalid_argument("rowwise extension expects a GPU tensor");
  }
  if (!x.is_float32) {
    throw std::invalid_argument("rowwise extension expects float32");
  }
  if (x.sizes[0] <= 0 || x.sizes[1] <= 0) {
    throw std::invalid_argument("rowwise extension expects a non-empty 2D tensor");
  }
}
