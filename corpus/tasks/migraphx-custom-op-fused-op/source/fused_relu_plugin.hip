// MIGraphX plugin lifecycle sketch for a fused relu(alpha*x) op.
//
// This intentionally keeps MIGraphX declarations out of the compile path so the
// corpus can validate without a MIGraphX SDK install. Replace the placeholder
// structs with nvinfer1::IPluginV2DynamicExt or the current MIGraphX plugin
// interface when building the real plugin.

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstring>

extern "C" void launch_fused_relu(const float* x,
                                   float* y,
                                   int elements,
                                   float alpha,
                                   hipStream_t stream);

struct PluginTensorDescSketch {
  int dims[8];
  int nb_dims;
  int type_fp32;
  int format_linear;
};

class FusedReluPluginSketch {
 public:
  explicit FusedReluPluginSketch(float alpha) : alpha_(alpha) {}

  explicit FusedReluPluginSketch(const void* data, size_t length) {
    if (length == sizeof(float)) {
      std::memcpy(&alpha_, data, sizeof(float));
    }
  }

  size_t getSerializationSize() const {
    return sizeof(float);
  }

  void serialize(void* buffer) const {
    std::memcpy(buffer, &alpha_, sizeof(float));
  }

  bool supportsFormatCombination(const PluginTensorDescSketch& desc) const {
    return desc.type_fp32 != 0 && desc.format_linear != 0;
  }

  int getOutputDimensions(const PluginTensorDescSketch& input,
                          PluginTensorDescSketch* output) const {
    *output = input;
    return 0;
  }

  int enqueue(const PluginTensorDescSketch& input_desc,
              const float* input,
              float* output,
              hipStream_t stream) const {
    int elements = 1;
    for (int i = 0; i < input_desc.nb_dims; ++i) {
      elements *= input_desc.dims[i];
    }
    launch_fused_relu(input, output, elements, alpha_, stream);
    return 0;
  }

 private:
  float alpha_ = 1.0f;
};
