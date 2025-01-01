#ifndef __PARTOOLS_ARRAY_HPP__
#define __PARTOOLS_ARRAY_HPP__

#include <algorithm>

#include "macros.hpp"
#include "memory.hpp"

namespace partools
{
  /// @brief An array class that dynamically partools::allocates memory on the host and optionally on the device.
  template <typename T>
  class Array
  {
  public:
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;

    /// @brief Initialize an array with a given size.
    Array(size_t len = 0)
    {
      n = len;

      host_data = nullptr;

#ifdef PARTOOLS_USING_GPU
      device_data = nullptr;

      host_is_valid = false;
      device_is_valid = false;
#endif
    }

    ~Array()
    {
      if (host_data != nullptr)
      {
        partools::deallocate<MemorySpace::Host, T>(host_data);
      }
#ifdef PARTOOLS_USING_GPU
      if (device_data != nullptr)
      {
        partools::deallocate<MemorySpace::Device, T>(device_data);
      }
#endif
    }

    /// @brief Copy constructor.
    Array(const Array &other)
    {
      n = other.n;

#ifdef PARTOOLS_USING_GPU
      if (other.host_is_valid)
      {
        host_data = new T[n];
        partools::copy_n(other.host_data, n, host_data);
        host_is_valid = true;
      }
      if (other.device_is_valid)
      {
        device_data = partools::allocate<MemorySpace::Device, T>(n);
        partools::copy_n(other.device_data, n, device_data);
        device_is_valid = true;
      }
#else
      if (other.host_data)
      {
        host_data = new T[n];
        partools::copy_n(other.host_data, n, host_data);
      }
#endif
    }

    /// @brief Copy assignment operator.
    Array &operator=(const Array &other)
    {
      n = other.n;

#ifdef PARTOOLS_USING_GPU
      if (other.host_is_valid)
      {
        if (host_data == nullptr || n < other.n)
        {
          partools::deallocate<MemorySpace::Host, T>(host_data);
          host_data = partools::allocate<MemorySpace::Host, T>(n);
        }

        partools::copy_n(other.host_data, n, host_data);
        host_is_valid = true;
      }
      if (other.device_is_valid)
      {
        if (device_data == nullptr || n < other.n)
        {
          partools::deallocate<MemorySpace::Device, T>(device_data);
          device_data = partools::allocate<MemorySpace::Device, T>(n);
        }

        partools::copy_n(other.device_data, n, device_data);
        device_is_valid = true;
      }
#else
      if (other.host_data)
      {
        if (host_data == nullptr || n < other.n)
        {
          partools::deallocate<MemorySpace::Host, T>(host_data);
          host_data = partools::allocate<MemorySpace::Host, T>(n);
        }

        partools::copy_n(other.host_data, n, host_data);
      }
#endif

      return *this;
    }

    /// @brief Move constructor.
    Array(Array &&other)
    {
      n = std::exchange(other.n, 0);
      host_data = std::exchange(other.host_data, nullptr);

#ifdef PARTOOLS_USING_GPU
      device_data = std::exchange(other.device_data, nullptr);
      host_is_valid = std::exchange(other.host_is_valid, false);
      device_is_valid = std::exchange(other.device_is_valid, false);
#endif
    }

    /// @brief Move assignment operator.
    Array &operator=(Array &&other)
    {
      n = std::exchange(other.n, 0);
      host_data = std::exchange(other.host_data, nullptr);

#ifdef PARTOOLS_USING_GPU
      device_data = std::exchange(other.device_data, nullptr);
      host_is_valid = std::exchange(other.host_is_valid, false);
      device_is_valid = std::exchange(other.device_is_valid, false);
#endif
    }

    /// @brief Get the size of the array.
    inline constexpr size_t size() const
    {
      return n;
    }

    /// @brief Resize the array.
    inline void resize(size_t len)
    {
      if (len <= n)
      {
        n = len;
        return;
      }

      host_data = partools::deallocate<MemorySpace::Host, T>(host_data);

#ifdef PARTOOLS_USING_GPU
      device_data = partools::deallocate<MemorySpace::Device, T>(device_data);

      host_is_valid = false;
      device_is_valid = false;
#endif

      n = len;
    }

    /// @brief get a read-only pointer to the array data on the host and optionally copy the data from the device. Ensures consistency between the host and device data.
    inline const_pointer host_read(bool force_copy = false) const
    {
      if (n < 1)
        return nullptr;

#ifdef PARTOOLS_USING_GPU
      if (!host_is_valid || force_copy)
      {
        if (host_data == nullptr)
          host_data = partools::allocate<MemorySpace::Host, T>(n);

        if (device_is_valid)
          partools::copy_n(device_data, n, host_data);
      }
      host_is_valid = true;
#else
      if (not host_data)
        host_data = partools::allocate<MemorySpace::Host, T>(n);
#endif
      return host_data;
    }

    /// @brief get a write-aceess pointer to the array data on the host and optionally release the ownership of the data. Invalidates the device data.
    inline pointer host_write(bool release = false) const
    {
      if (n < 1)
        return nullptr;

      if (not host_data)
        host_data = partools::allocate<MemorySpace::Host, T>(n);

#ifdef PARTOOLS_USING_GPU
      if (release)
        host_is_valid = false;
      else
      {
        host_is_valid = true;
        device_is_valid = false;
      }
#endif

      if (release)
        return std::exchange(host_data, nullptr);
      else
        return host_data;
    }

    /// @brief get a read-write pointer to the array data on the host and optionally release the ownership of the data. Synchronizes the host and device data then invalidates the device data.
    inline pointer host_read_write(bool force_copy = false, bool release = false) const
    {
      host_read(force_copy);
      return host_write(release);
    }

    /// @brief release the ownership of the host data.
    inline pointer host_release() const
    {
      return host_write(true);
    }

    /// @brief get a read-only pointer to the array data on the device and optionally copy the data from the host. Ensures consistency between the host and device data.
    inline const_pointer device_read(bool force_copy = false) const
    {
#ifdef PARTOOLS_USING_GPU
      if (n < 1)
        return nullptr;

      if (!device_is_valid || force_copy)
      {
        if (not device_data)
          device_data = partools::allocate<MemorySpace::Device, T>(n);

        if (host_is_valid)
          partools::copy_n(host_data, n, device_data);
      }
      device_is_valid = true;

      return device_data;
#else
      return host_read(force_copy);
#endif
    }

    /// @brief get a write-aceess pointer to the array data on the device and optionally release the ownership of the data. Invalidates the host data.
    inline pointer device_write(bool release = false) const
    {
#ifdef PARTOOLS_USING_GPU
      if (n < 1)
        return nullptr;

      if (not device_data)
        device_data = partools::allocate<MemorySpace::Device, T>(n);

      if (release)
      {
        device_is_valid = false;
        return std::exchange(device_data, nullptr);
      }
      else
      {
        device_is_valid = true;
        host_is_valid = false;
        return device_data;
      }
#else
      return host_write(release);
#endif
    }

    /// @brief get a read-write pointer to the array data on the device and optionally release the ownership of the data. Synchronizes the host and device data then invalidates the host data.
    inline pointer device_read_write(bool force_copy = false, bool release = false) const
    {
      device_read(force_copy);
      return device_write(release);
    }

    /// @brief release the ownership of the device data.
    inline pointer device_release() const
    {
      return device_write(true);
    }

    /// @brief get a read-only pointer to the array data on the specified memory space and optionally copy the data from the other memory space.
    inline const_pointer read(bool force_copy = false, MemorySpace m = MemorySpace::Device) const
    {
      if (m == MemorySpace::Host)
        return host_read(force_copy);
      else // Device
        return device_read(force_copy);
    }

    /// @brief get a write-aceess pointer to the array data on the specified memory space and optionally release the ownership of the data.
    inline pointer write(bool release = false, MemorySpace m = MemorySpace::Device) const
    {
      if (m == MemorySpace::Host)
        return host_write(release);
      else // Device
        return device_write(release);
    }

    /// @brief get a read-write pointer to the array data on the specified memory space and optionally release the ownership of the data. Synchronizes the host and device data then invalidates the device data.
    inline pointer read_write(bool force_copy = false, bool release = false, MemorySpace m = MemorySpace::Device) const
    {
      read(force_copy, m);
      return write(release, m);
    }

    /// @brief release the ownership of the data on the specified memory space.
    inline pointer release(MemorySpace m = MemorySpace::Device) const
    {
      return write(true, m);
    }

  private:
    size_t n;

    mutable pointer host_data;

#ifdef PARTOOLS_USING_GPU
    mutable pointer device_data;

    mutable bool host_is_valid;
    mutable bool device_is_valid;
#endif
  };
} // namespace partools

#endif
