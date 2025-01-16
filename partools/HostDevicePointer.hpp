#ifndef __PARTOOLS_HOST_DEVICE_POINTER_HPP__
#define __PARTOOLS_HOST_DEVICE_POINTER_HPP__

#include <algorithm>

#include "macros.hpp"
#include "memory.hpp"

namespace partools
{
  /// @brief A class that manages a pointer to an array that can be accessed on both host and device. Behaves like a unique pointer.
  template <typename T>
  class HostDevicePointer
  {
  public:
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;

    /// @brief Initialize an array with a given size. Default size is 0.
    HostDevicePointer(size_t len = 0)
    {
      n = len;

      host_data = nullptr;

#ifdef PARTOOLS_USING_GPU
      device_data = nullptr;

      host_is_valid = false;
      device_is_valid = false;
#endif
    }

    ~HostDevicePointer()
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

    HostDevicePointer(const HostDevicePointer &other) = delete;
    HostDevicePointer &operator=(const HostDevicePointer &other) = delete;

    HostDevicePointer(HostDevicePointer &&other) = default;
    HostDevicePointer &operator=(HostDevicePointer &&other) = default;

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

    /// @brief get a read-only pointer to the array data on the specified memory space. If the data in the memory space is outdated or if force_copy==true, copy the data from the other memory space.
    template <MemorySpace m>
    inline const_pointer read(bool force_copy = false) const
    {
      if constexpr (m == Device)
        return device_read(force_copy);
      else
        return host_read(force_copy);
    }

    /// @brief get a read-only pointer to the array data on the specified memory space.If the data in the memory space is outdated or if force_copy==true, copy the data from the other memory space.
    inline const_pointer read(MemorySpace m, bool force_copy = false) const
    {
      if (m == Host)
        return read<Host>(force_copy);
      else // Device
        return read<Device>(force_copy);
    }

    /// @brief get a write-aceess pointer to the array data on the specified memory space and optionally release the ownership of the data. Does not synchronize data.
    template <MemorySpace m>
    inline pointer write(bool release = false)
    {
      if constexpr (m == MemorySpace::Device)
        return device_write(release);
      else
        return host_write(release);
    }

    /// @brief get a write-aceess pointer to the array data on the specified memory space and optionally release the ownership of the data. Does not synchronize data.
    inline pointer write(MemorySpace m, bool release = false)
    {
      if (m == MemorySpace::Host)
        return host_write(release);
      else // Device
        return device_write(release);
    }

    /// @brief get a read-write pointer to the array data on the specified memory space and optionally release the ownership of the data. Data between host and device is synchronized.
    template <MemorySpace m>
    inline pointer get(bool force_copy = false, bool release = false)
    {
      read<m>(force_copy);
      return write<m>(release);
    }

    /// @brief get a read-write pointer to the array data on the specified memory space and optionally release the ownership of the data. Data between host and device is synchronized.
    inline pointer get(MemorySpace m, bool force_copy = false, bool release = false)
    {
      read(m, force_copy);
      return write(m, release);
    }

    /// @brief get a read-only pointer to the array data on the specified memory space.
    template <MemorySpace m>
    inline const_pointer get(bool force_copy = false) const
    {
      return read<m>(force_copy);
    }

    /// @brief get a read-only pointer to the array data on the specified memory space.
    inline const_pointer get(MemorySpace m, bool force_copy = false) const
    {
      return read(m, force_copy);
    }

    /// @brief release the ownership of the data on the host.
    template <MemorySpace m>
    inline pointer release()
    {
      return write<m>(true);
    }

    /// @brief release the ownership of the data on the specified memory space.
    inline pointer release(MemorySpace m)
    {
      return write(m, true);
    }

    /// @brief Synchronize the data between host and device.
    /// If the host data is valid and the device data is not, it copies the data to the device.
    /// If the device data is valid and the host data is not, it copies the data to the host.
    inline void synchronize() const
    {
#ifdef PARTOOLS_USING_GPU
      if (host_is_valid && !device_is_valid)
        read<Device>(true);
      else if (!host_is_valid && device_is_valid)
        read<Host>(true);
#endif
    }

    /// @brief Deep copy of the data to a new HostDevicePointer object.
    inline HostDevicePointer<T> copy() const
    {
      HostDevicePointer<T> result(n);

      if (n < 1)
        return result;

#ifdef PARTOOLS_USING_GPU
      if (host_is_valid)
      {
        partools::copy_n(host_data, n, result.write<Host>());
        result.host_is_valid = true;
      }
      if (device_is_valid)
      {
        partools::copy_n(device_data, n, result.write<Device>());
        result.device_is_valid = true;
      }
#else
      if (host_data)
      {
        partools::copy_n(host_data, n, result.write<Host>());
      }
#endif

      return result;
    }

  private:
    size_t n;

    mutable pointer host_data;

#ifdef PARTOOLS_USING_GPU
    mutable pointer device_data;

    mutable bool host_is_valid;
    mutable bool device_is_valid;
#endif

    inline const_pointer host_read(bool force_copy = false) const
    {
      if (n < 1)
        return nullptr;

#ifdef PARTOOLS_USING_GPU
      if (!host_is_valid || force_copy)
      {
        if (not host_data)
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

    inline pointer host_write(bool release = false)
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

    inline pointer device_write(bool release = false)
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
  };
} // namespace partools

#endif
