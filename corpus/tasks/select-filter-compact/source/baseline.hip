#include <hip/hip_runtime.h>

#include <cub/cub.hpp>

#include <cstddef>

namespace {

void* g_temp_storage = nullptr;
size_t g_temp_storage_bytes = 0;
int g_capacity = 0;

size_t required_temp_bytes(const int* input,
                           const unsigned char* flags,
                           int* output,
                           int* selected_count,
                           int n,
                           hipStream_t stream) {
  size_t bytes = 0;
  hipcub::DeviceSelect::Flagged(
      nullptr, bytes, input, flags, output, selected_count, n, stream);
  return bytes;
}

}  // namespace

extern "C" size_t select_filter_compact_temp_bytes(const int* input,
                                                   const unsigned char* flags,
                                                   int* output,
                                                   int* selected_count,
                                                   int n,
                                                   hipStream_t stream) {
  return required_temp_bytes(input, flags, output, selected_count, n, stream);
}

extern "C" void launch_baseline(const int* input,
                                const unsigned char* flags,
                                int* output,
                                int* selected_count,
                                int n,
                                hipStream_t stream) {
  if (g_capacity < n || g_temp_storage == nullptr) {
    if (g_temp_storage != nullptr) {
      hipFree(g_temp_storage);
      g_temp_storage = nullptr;
    }
    g_temp_storage_bytes =
        required_temp_bytes(input, flags, output, selected_count, n, stream);
    hipMalloc(&g_temp_storage, g_temp_storage_bytes);
    g_capacity = n;
  }

  hipcub::DeviceSelect::Flagged(g_temp_storage,
                             g_temp_storage_bytes,
                             input,
                             flags,
                             output,
                             selected_count,
                             n,
                             stream);
}
