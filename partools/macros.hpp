#ifndef __PARTOOLS_MACROS_HPP__
#define __PARTOOLS_MACROS_HPP__

#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#define PARTOOLS_USING_CUDA
#include <cuda_runtime.h>

#define CHECK_CALL(call) checkCudaError(call, __FILE__, __LINE__, __func__)
static inline void checkCudaError(cudaError_t code, const char *file, int line, const char *func)
{
  if (code != cudaSuccess)
  {
    using namespace std::string_literals;
    throw std::runtime_error("CUDA error: "s + file + ":" + std::to_string(line) + ": " + func + ": " + cudaGetErrorString(code));
  }
}

#define PARTOOLS_LAUNCH_KERNEL(kernel, N, B, ...) kernel<<<N, B>>>(__VA_ARGS__); CHECK_CALL(cudaGetLastError());

#endif

#ifdef __HIPCC__
#define PARTOOLS_USING_HIP
#include <hip/hip_runtime.h>

#define CHECK_CALL(call) checkHipError(call, __FILE__, __LINE__, __func__)
static inline void checkHipError(hipError_t code, const char *file, int line, const char *func)
{
  if (code != hipSuccess)
  {
    using namespace std::string_literals;
    throw std::runtime_error("HIP error: "s + file + ":" + std::to_string(line) + ": " + func + ": " + hipGetErrorString(code));
  }
}

#define PARTOOLS_LAUNCH_KERNEL(kernel, N, B, ...) hipLaunchKernelGGL(kernel, N, B, 0, 0, __VA_ARGS__); CHECK_CALL(hipGetLastError());
#endif

#if defined(PARTOOLS_USING_CUDA) || defined(PARTOOLS_USING_HIP)
#define PARTOOLS_USING_GPU
#endif

#ifdef PARTOOLS_USING_GPU

#ifndef PARTOOLS_FORALL_BLOCK_SIZE
#define PARTOOLS_FORALL_BLOCK_SIZE 1024
#endif

#define PARTOOLS_HOST_DEVICE __host__ __device__
#define PARTOOLS_DEVICE __device__

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define PARTOOLS_DEVICE_CODE
#endif

#else

#define PARTOOLS_HOST_DEVICE
#define PARTOOLS_DEVICE

#endif

#ifdef _OPENMP
#define PARTOOLS_USING_OPENMP
#include <omp.h>
#endif

#endif
