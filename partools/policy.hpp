#ifndef __PARTOOLS_POLICY_HPP__
#define __PARTOOLS_POLICY_HPP__

namespace partools
{
  enum ExecutionPolicy
  {
    Sequential,
    GPU,
    OpenMP,
  };

  struct gpuPolicy
  {
    int block_size; // Number of threads per block. default value is specified by PARTOOLS_FORALL_BLOCK_SIZE.
    int batch_size; // Number of tasks per thread. One by default.
    bool barrier;   // If true, gpu::forall will block until all tasks are completed. False by default.

    gpuPolicy()
    {
#ifdef PARTOOLS_USING_GPU
      block_size = PARTOOLS_FORALL_BLOCK_SIZE;
#endif
      batch_size = 1;
      barrier = false;
    }
  };

  struct OpenMPPolicy
  {
    int num_threads; // Number of threads. Default value is the number of threads available.
    int batch_size;  // Number of tasks per thread. Used with static scheduling. Automatic by default.

    OpenMPPolicy()
    {
#ifdef PARTOOLS_USING_OPENMP
      num_threads = omp_get_num_threads();
#endif
      batch_size = 0;
    }
  };
} // namespace partools

#endif
