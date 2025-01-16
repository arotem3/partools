#ifndef __PARTOOLS_FORALL_HPP__
#define __PARTOOLS_FORALL_HPP__

#include "policy.hpp"
#include "macros.hpp"
#include "synchronize.hpp"

#include <stdexcept>
#include <utility>

namespace partools
{
#ifdef PARTOOLS_USING_GPU
  namespace gpu
  {
    template <typename Function>
    /**
     * @brief Executes a function in parallel on the GPU.
     * 
     * @param n The number of iterations.
     * @param fun The function to execute.
     */
    __global__ static void forall_kernel(int n, Function fun)
    {
      const int k = threadIdx.x + blockIdx.x * blockDim.x;

      if (k >= n)
        return;

      fun(k);
    }

    /**
     * @brief Executes a function in parallel on the GPU in batches.
     * 
     * @param n The number of iterations.
     * @param fun The function to execute.
     * @param batch_size The batch size.
     */
    template <typename Function>
    __global__ static void batched_forall_kernel(int n, Function fun, int batch_size)
    {
      const int k = threadIdx.x + blockIdx.x * blockDim.x;

      for (int i = 0; i < batch_size; ++i)
      {
        const int j = k * batch_size + i;

        if (j < n)
          fun(j);
      }
    }

    /// @brief Executes a function in parallel on the GPU.
    template <typename Function>
    static inline void forall(int n, Function &&fun, gpuPolicy policy)
    {
      if (n <= 0)
        return;

      const int block_size = policy.block_size;
      const int batch_size = policy.batch_size;

      if (policy.batch_size <= 1)
      {
        const int n_blocks = (n + block_size - 1) / block_size;
        PARTOOLS_LAUNCH_KERNEL(forall_kernel, n_blocks, block_size, n, fun);
      }
      else
      {
        int N = (n + batch_size - 1) / batch_size;
        const int n_blocks = (N + block_size - 1) / block_size;

        PARTOOLS_LAUNCH_KERNEL(batched_forall_kernel, n_blocks, block_size, n, fun, batch_size);
      }

      if (policy.barrier)
        synchronize();
    }

    /**
      * @brief Executes a function in parallel on the GPU, where each block handles one iteration.
      * 
      * @param n The number of iterations.
      * @param fun The function to execute.
      */
    template <typename Function>
    __global__ static void forall_block_kernel(int n, Function fun)
    {
      const int k = blockIdx.x;

      if (k >= n)
        return;

      fun(k);
    }

    /**
     * @brief Executes a function in parallel on the GPU in batches, where each block handles one batch.
     * 
     * @param n The number of iterations.
     * @param fun The function to execute.
     * @param batch_size The batch size.
     */
    template <typename Function>
    __global__ static void batched_forall_block_kernel(int n, Function fun, int batch_size)
    {
      const int k = blockIdx.x;

      for (int i = 0; i < batch_size; ++i)
      {
        const int j = k * batch_size + i;

        if (j < n)
          fun(j);
      }
    }

    /// @brief Executes n tasks in parallel on the GPU where each task is executed by bx threads in a block.
    template <typename Function>
    static inline void forall_1d(int bx, int n, Function &&fun, gpuPolicy policy)
    {
      if (n <= 0)
        return;

      const int batch_size = policy.batch_size;

      if (batch_size <= 1)
      {
        PARTOOLS_LAUNCH_KERNEL(forall_block_kernel, n, bx, n, fun);
      }
      else
      {
        int N = (n + batch_size - 1) / batch_size;
        PARTOOLS_LAUNCH_KERNEL(batched_forall_block_kernel, N, bx, n, fun, batch_size);
      }

      if (policy.barrier)
        synchronize();
    }

    template <typename Function>
    static inline void forall_2d(int bx, int by, int n, Function &&fun, gpuPolicy policy)
    {
      if (n <= 0)
        return;

      dim3 block_size(bx, by);

      const int batch_size = policy.batch_size;

      if (batch_size <= 1)
      {
        PARTOOLS_LAUNCH_KERNEL(forall_block_kernel, n, block_size, n, fun);
      }
      else
      {
        int N = (n + batch_size - 1) / batch_size;
        PARTOOLS_LAUNCH_KERNEL(batched_forall_block_kernel, N, block_size, n, fun, batch_size);
      }

      if (policy.barrier)
        synchronize();
    }

    template <typename Function>
    static inline void forall_3d(int bx, int by, int bz, int n, Function &&fun, gpuPolicy policy)
    {
      if (n <= 0)
        return;

      dim3 block_size(bx, by, bz);

      const int batch_size = policy.batch_size;

      if (batch_size <= 1)
      {
        PARTOOLS_LAUNCH_KERNEL(forall_block_kernel, n, block_size, n, fun);
      }
      else
      {
        int N = (n + batch_size - 1) / batch_size;
        PARTOOLS_LAUNCH_KERNEL(batched_forall_block_kernel, N, block_size, n, fun, batch_size);
      }

      if (policy.barrier)
        synchronize();
    }
  } // namespace partools_gpu
#endif

#ifdef PARTOOLS_USING_OPENMP
  namespace openmp
  {
    /// @brief Executes a function in parallel using OpenMP.
    template <typename Function>
    static inline void forall(int n, Function &&fun, OpenMPPolicy policy)
    {
      if (n <= 0)
        return;

      int num_threads = policy.num_threads;
      int batch_size = policy.batch_size;

      if (batch_size <= 0)
        batch_size = (n + num_threads - 1) / num_threads;

#pragma omp parallel for num_threads(num_threads) schedule(static, batch_size)
      for (int i = 0; i < n; ++i)
        fun(i);
    }
  } // namespace partools_openmp
#endif

  /// @brief For loop abstraction where all iterations are independent and may be executed in parallel.
  template <ExecutionPolicy policy, typename Function>
  static inline void forall(int n, Function &&fun)
  {
    if constexpr (policy == GPU)
    {
#ifdef PARTOOLS_USING_GPU
      gpu::forall(n, std::forward<Function>(fun), gpuPolicy());
#else
      throw std::runtime_error("GPU policy is not available.");
#endif
    }
    else if constexpr (policy == OpenMP)
    {
#ifdef PARTOOLS_USING_OPENMP
      openmp::forall(n, std::forward<Function>(fun), OpenMPPolicy());
#else
      throw std::runtime_error("OpenMP policy is not available.");
#endif
    }
    else
    {
      for (int i = 0; i < n; ++i)
        fun(i);
    }
  }

  /// @brief Execute n independent tasks in parallel where each task is executed by bx threads in a block.
  template <ExecutionPolicy policy, typename Function>
  static inline void forall_1d(int bx, int n, Function &&fun)
  {
    if constexpr (policy == GPU)
    {
#ifdef PARTOOLS_USING_GPU
      gpu::forall_1d(bx, n, std::forward<Function>(fun), gpuPolicy());
#else
      throw std::runtime_error("GPU policy is not available.");
#endif
    }
    else if constexpr (policy == OpenMP)
    {
#ifdef PARTOOLS_USING_OPENMP
      openmp::forall(n, std::forward<Function>(fun), OpenMPPolicy());
#else
      throw std::runtime_error("OpenMP policy is not available.");
#endif
    }
    else
    {
      for (int i = 0; i < n; ++i)
        fun(i);
    }
  }

  /// @brief Executes n independent tasks in parallel where each task is executed by a block of bx x by threads.
  template <ExecutionPolicy policy, typename Function>
  static inline void forall_2d(int bx, int by, int n, Function &&fun)
  {
    if constexpr (policy == GPU)
    {
#ifdef PARTOOLS_USING_GPU
      gpu::forall_2d(bx, by, n, std::forward<Function>(fun), gpuPolicy());
#else
      throw std::runtime_error("GPU policy is not available.");
#endif
    }
    else if constexpr (policy == OpenMP)
    {
#ifdef PARTOOLS_USING_OPENMP
      openmp::forall(n, std::forward<Function>(fun), OpenMPPolicy());
#else
      throw std::runtime_error("OpenMP policy is not available.");
#endif
    }
    else
    {
      for (int i = 0; i < n; ++i)
        fun(i);
    }
  }

  /// @brief Executes n independent tasks in parallel where each task is executed by a block of bx x by x bz threads.
  template <ExecutionPolicy policy, typename Function>
  static inline void forall_3d(int bx, int by, int bz, int n, Function &&fun)
  {
    if constexpr (policy == GPU)
    {
#ifdef PARTOOLS_USING_GPU
      gpu::forall_3d(bx, by, bz, n, std::forward<Function>(fun), gpuPolicy());
#else
      throw std::runtime_error("GPU policy is not available.");
#endif
    }
    else if constexpr (policy == OpenMP)
    {
#ifdef PARTOOLS_USING_OPENMP
      openmp::forall(n, std::forward<Function>(fun), OpenMPPolicy());
#else
      throw std::runtime_error("OpenMP policy is not available.");
#endif
    }
    else
    {
      for (int i = 0; i < n; ++i)
        fun(i);
    }
  }
} // namespace partools

#endif
