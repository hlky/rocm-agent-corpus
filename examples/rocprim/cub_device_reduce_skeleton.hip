#include <cub/cub.hpp>

// Reference sketch: prefer hipCUB for production reductions before writing a
// custom reduction kernel.
//
// Typical flow:
// 1. Call hipcub::DeviceReduce::Sum with temp storage = nullptr to get bytes.
// 2. Allocate temp storage.
// 3. Call hipcub::DeviceReduce::Sum again on the target stream.
// 4. Benchmark against any custom kernel using the same input and tolerance.

