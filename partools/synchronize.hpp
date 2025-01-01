#ifndef __PARTOOLS_SYNCHRONIZE_HPP__
#define __PARTOOLS_SYNCHRONIZE_HPP__

#include "macros.hpp"

namespace partools
{
  /// @brief Synchronizes the execution of the current device.
  static void synchronize()
  {
#ifdef PARTOOLS_USING_CUDA
    CHECK_CALL(cudaDeviceSynchronize());
#elif defined(PARTOOLS_USING_HIP)
    CHECK_CALL(hipDeviceSynchronize());
#endif
  }
} // namespace partools

#endif
