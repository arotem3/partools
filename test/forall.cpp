#include "partools.hpp"
#include <iostream>
#include <assert.h>

using namespace partools;

template <ExecutionPolicy policy>
int forall_test()
{
  int n = 1<<20;
  Array<int> arr(n);

  MemorySpace m = (policy == GPU) ? MemorySpace::Device : MemorySpace::Host;

  int *x = arr.write(false, m);

  forall<policy>(n, [=] PARTOOLS_HOST_DEVICE (int i) mutable
  {
    x[i] = 2 * i;
  });

  const int *y = arr.host_read();

  int num_failed = 0;
  for (int i = 0; i < n; i++)
  {
    num_failed += (y[i] != 2 * i);
  }

  std::string policy_name = (policy == GPU) ? "GPU" : (policy == OpenMP) ? "OpenMP" : "Sequential";

  if (num_failed)
  {
    std::cout << "forall<" << policy_name << "> test failed" << std::endl;
  }
  else
  {
    std::cout << "forall<" << policy_name << "> test passed" << std::endl;
  }

  return num_failed;
}

template <ExecutionPolicy policy>
int forall_3d_test()
{
  int n = 1<<1;
  int bx = 8, by = 5, bz = 3;

  Array<int> arr(n * bx * by * bz);

  MemorySpace m = (policy == GPU) ? MemorySpace::Device : MemorySpace::Host;
  int *x = arr.write(false, m);

  forall_3d<policy>(bx, by, bz, n, [=] PARTOOLS_HOST_DEVICE (int l) mutable
  {
    thread_for_3d(bx, by, bz, [=](int i, int j, int k) mutable
    {
      int idx = i + bx * (j + by * (k + bz * l));
      int val = 2 * l + 3 * i + 5 * j + 7 * k;
      x[idx] = val;
    });
  });
  // synchronize();

  const int *y = arr.host_read();

  int num_failed = 0;

  for (int l = 0; l < n; l++)
  {
    for (int k = 0; k < bz; k++)
    {
      for (int j = 0; j < by; j++)
      {
        for (int i = 0; i < bx; i++)
        {
          int idx = i + bx * (j + by * (k + bz * l));
          int val = 2 * l + 3 * i + 5 * j + 7 * k;
          num_failed += (y[idx] != val);
        }
      }
    }
  }

  std::string policy_name = (policy == GPU) ? "GPU" : (policy == OpenMP) ? "OpenMP" : "Sequential";

  if (num_failed)
  {
    std::cout << "forall_3d<" << policy_name << "> test failed" << std::endl;
  }
  else
  {
    std::cout << "forall_3d<" << policy_name << "> test passed" << std::endl;
  }

  return num_failed;
}

int main()
{
  int failed = 0;

  failed += forall_test<Sequential>();

#ifdef PARTOOLS_USING_GPU
  failed += forall_test<GPU>();
#endif

#ifdef PARTOOLS_USING_OPENMP
  failed += forall_test<OpenMP>();
#endif

  failed += forall_3d_test<Sequential>();

#ifdef PARTOOLS_USING_GPU
  failed += forall_3d_test<GPU>();
#endif

#ifdef PARTOOLS_USING_OPENMP
  failed += forall_3d_test<OpenMP>();
#endif

  return failed;
}