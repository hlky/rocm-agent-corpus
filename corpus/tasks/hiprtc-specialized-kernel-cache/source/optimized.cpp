#include <hip/hip_runtime_api.h>
#include <string>
#include <unordered_map>

struct SpecializationKey {
  int n;
  std::string gfx_target;
  std::string dtype;
  std::string options;
};

struct CachedKernel {
  hipModule_t module;
  hipFunction_t function;
};

// Optimized sketch: compile once with hipRTC/hipRTC cache/link path, then reuse by exact key.
class SpecializedKernelCache {
 public:
  hipError_t Launch(const SpecializationKey& key,
                    hipDeviceptr_t input,
                    hipDeviceptr_t output,
                    hipStream_t stream);

 private:
  std::unordered_map<std::string, CachedKernel> cache_;
};
