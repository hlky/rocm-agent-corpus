#include <algorithm>
#include <vector>

void gemm_bias_relu_host_reference(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   const std::vector<float>& bias,
                                   std::vector<float>& d,
                                   int m,
                                   int n,
                                   int k) {
  d.assign(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        acc += a[row * k + kk] * b[kk * n + col];
      }
      d[row * n + col] = std::max(acc + bias[col], 0.0f);
    }
  }
}
