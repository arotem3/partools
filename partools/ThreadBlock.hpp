#ifndef __PARTOOLS_THREAD_BLOCK_HPP__
#define __PARTOOLS_THREAD_BLOCK_HPP__

#include "macros.hpp"

namespace partools
{
  /// @brief For use in kernel functions with 1D thread blocks.
  template <typename Function>
  PARTOOLS_HOST_DEVICE static inline void thread_for_1d(int n, Function &&fun)
  {
#ifdef PARTOOLS_DEVICE_CODE
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      fun(i);
#else
    for (int i = 0; i < n; ++i)
      fun(i);
#endif
  }

  /// @brief For use in kernel functions with 2D thread blocks.
  template <typename Function>
  PARTOOLS_HOST_DEVICE static inline void thread_for_2d(int nx, int ny, Function &&fun)
  {
#ifdef PARTOOLS_DEVICE_CODE
    for (int j = threadIdx.y; j < ny; j += blockDim.y)
    {
      for (int i = threadIdx.x; i < nx; i += blockDim.x)
      {
        fun(i, j);
      }
    }
#else
    for (int j = 0; j < ny; ++j)
    {
      for (int i = 0; i < nx; ++i)
      {
        fun(i, j);
      }
    }
#endif
  }

  /// @brief For use in kernel functions with 3D thread blocks.
  template <typename Function>
  PARTOOLS_HOST_DEVICE static inline void thread_for_3d(int nx, int ny, int nz, Function &&fun)
  {
#ifdef PARTOOLS_DEVICE_CODE
    for (int k = threadIdx.z; k < nz; k += blockDim.z)
    {
      for (int j = threadIdx.y; j < ny; j += blockDim.y)
      {
        for (int i = threadIdx.x; i < nx; i += blockDim.x)
        {
          fun(i, j, k);
        }
      }
    }
#else
    for (int k = 0; k < nz; ++k)
    {
      for (int j = 0; j < ny; ++j)
      {
        for (int i = 0; i < nx; ++i)
        {
          fun(i, j, k);
        }
      }
    }
#endif
  }

  PARTOOLS_HOST_DEVICE static inline void thread_sync()
  {
#ifdef PARTOOLS_DEVICE_CODE
    __syncthreads();
#endif
  }
} // namespace partools

#endif
