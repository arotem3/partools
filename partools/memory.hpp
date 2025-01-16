#ifndef __PARTOOLS_MEMORY_HPP__
#define __PARTOOLS_MEMORY_HPP__

#include "macros.hpp"

#include <algorithm>
#include <stdexcept>
#include <memory>

namespace partools
{
  enum MemorySpace
  {
    Host,  // Memory explicitly on the host
    Device // Memory on a device (e.g., GPU) if available, otherwise on the host (for compatibility)
  };

  /// @brief Allocates an array of a given size on the specified memory space.
  template <MemorySpace m, typename T>
  static inline T *allocate(size_t n)
  {
    if (n == 0)
      return nullptr;

#ifdef PARTOOLS_USING_GPU
    T *ptr = nullptr;
    const size_t size = n * sizeof(T);

    if constexpr (m == MemorySpace::Host)
    {
      ptr = new T[n]{};
    }
    else // Device
    {
#ifdef PARTOOLS_USING_CUDA
      CHECK_CALL(cudaMalloc(&ptr, size));
#elif defined(PARTOOLS_USING_HIP)
      CHECK_CALL(hipMalloc(&ptr, size));
#endif
    }
    return ptr;
#else
    return new T[n];
#endif
  }

  /// @brief Deallocates an array on the specified memory space. Returns nullptr.
  template <MemorySpace m, typename T>
  static inline T *deallocate(T *ptr)
  {
    if (ptr == nullptr)
      return nullptr;

#ifdef PARTOOLS_USING_GPU
    if constexpr (m == MemorySpace::Host)
    {
      delete[] ptr;
    }
    else // Device
    {
#ifdef PARTOOLS_USING_CUDA
      CHECK_CALL(cudaFree(ptr));
#elif defined(PARTOOLS_USING_HIP)
      CHECK_CALL(hipFree(ptr));
#endif
    }
#else
    delete[] ptr;
#endif
    return nullptr;
  }

  /// @brief Allocator for STL containers.
  template <MemorySpace m, typename T>
  class allocator
  {
  public:
    using value_type = T;
    using pointer = T *;
    using size_type = size_t;

    allocator() noexcept = default;
    allocator(const allocator &) noexcept = default;
    allocator &operator=(const allocator &) noexcept = default;
    allocator(allocator &&) noexcept = default;
    allocator &operator=(allocator &&) noexcept = default;
    ~allocator() noexcept = default;

    static inline pointer allocate(size_type n)
    {
      return partools::allocate<m, T>(n);
    }

    static inline void deallocate(pointer p, size_type n)
    {
      partools::deallocate<m, T>(p);
    }

    constexpr bool operator==(const allocator &)
    {
      return true;
    }

    constexpr bool operator!=(const allocator &)
    {
      return false;
    }
  };

  /// @brief Deleter for unique pointers.
  template <MemorySpace m, typename T>
  class deleter
  {
  public:
    deleter() = default;
    deleter(const deleter &) = default;
    deleter &operator=(const deleter &) = default;

    void operator()(T *ptr) const
    {
      partools::deallocate<m, T>(ptr);
    }
  };

  template <typename T>
  using unique_device_pointer = std::unique_ptr<T, partools::deleter<Device, T>>;

  /// @brief Copies n elements from src to dst. This is aware of the memory space of the pointers.
  template <typename T>
  static inline void copy_n(const T *src, const size_t n, T *dst)
  {
    if (n == 0)
      return;

    if (src == nullptr || dst == nullptr)
      throw std::invalid_argument("partools::copy_n: src and dst must be non-null");

    if (src == dst)
      return;

#ifdef PARTOOLS_USING_CUDA
    CHECK_CALL(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDefault));
#elif defined(PARTOOLS_USING_HIP)
    CHECK_CALL(hipMemcpy(dst, src, n * sizeof(T), hipMemcpyDefault));
#else
    std::copy_n(src, n, dst);
#endif
  }
} // namespace partools

#endif
