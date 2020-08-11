#ifndef QUICKSORT_GPU_H
#define QUICKSORT_GPU_H

#include <vector>

template <typename T>
void quicksort_gpu_par(std::vector<T> &list);
template <typename T>
void quicksort_gpu_dyn(std::vector<T> &list);
template <typename T>
void quicksort_cpu_coordinated(std::vector<T> &list);
void initCuda();
#endif // QUICKSORT_GPU_H
