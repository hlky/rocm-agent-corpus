#include <hip/hip_runtime_api.h>
#include <string>

struct GenericKernelRequest {
  int n;
  std::string dtype;
};

// Baseline sketch: use one prebuilt generic kernel and pass shape at runtime.
hipError_t launch_generic_runtime_kernel(const GenericKernelRequest& request,
                                         hipDeviceptr_t input,
                                         hipDeviceptr_t output,
                                         hipStream_t stream);
