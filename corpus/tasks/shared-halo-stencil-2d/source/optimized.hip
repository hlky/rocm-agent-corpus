#include <hip/hip_runtime.h>

namespace {
constexpr int kTileX = 16;
constexpr int kTileY = 16;
}  // namespace

__global__ void stencil5_shared_halo_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int width,
                                            int height) {
  __shared__ float tile[kTileY + 2][kTileX + 2];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int x = blockIdx.x * kTileX + tx;
  const int y = blockIdx.y * kTileY + ty;
  const int local_x = tx + 1;
  const int local_y = ty + 1;

  const bool inside = x < width && y < height;
  tile[local_y][local_x] = inside ? input[y * width + x] : 0.0f;

  if (tx == 0) {
    const int halo_x = x - 1;
    tile[local_y][0] = (halo_x >= 0 && y < height) ? input[y * width + halo_x] : 0.0f;
  }
  if (tx == kTileX - 1) {
    const int halo_x = x + 1;
    tile[local_y][kTileX + 1] = (halo_x < width && y < height) ? input[y * width + halo_x] : 0.0f;
  }
  if (ty == 0) {
    const int halo_y = y - 1;
    tile[0][local_x] = (x < width && halo_y >= 0) ? input[halo_y * width + x] : 0.0f;
  }
  if (ty == kTileY - 1) {
    const int halo_y = y + 1;
    tile[kTileY + 1][local_x] = (x < width && halo_y < height) ? input[halo_y * width + x] : 0.0f;
  }

  __syncthreads();

  if (!inside) {
    return;
  }

  const int idx = y * width + x;
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    output[idx] = tile[local_y][local_x];
    return;
  }

  output[idx] = 0.25f * tile[local_y][local_x] +
                0.1875f * (tile[local_y][local_x - 1] +
                           tile[local_y][local_x + 1] +
                           tile[local_y - 1][local_x] +
                           tile[local_y + 1][local_x]);
}

extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int width,
                                 int height,
                                 hipStream_t stream) {
  dim3 block(kTileX, kTileY);
  dim3 grid((width + kTileX - 1) / kTileX,
            (height + kTileY - 1) / kTileY);
  stencil5_shared_halo_kernel<<<grid, block, 0, stream>>>(input, output, width, height);
}
