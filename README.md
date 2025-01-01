# partools
basic tools for parallel execution in C++ using openmp and cuda.

# Installation
```bash
mkdir build && cd build
cmake -D USING_CUDA=ON -D USING_OMP=ON ..  # modify these options as needed
make -j
sudo make install
```

## Basic use example
```c++
#include "partools.hpp"

using namespace partools;

int main()
{
  int n = 1<<20;
  Array<double> x(n);

  double *d_x = x.write(); // device pointer

  forall<GPU>(n, [=] PARTOOLS_DEVICE (int i)
  {
    d_x[i] = 2.0 * i;
  });

  return 0;
}
```