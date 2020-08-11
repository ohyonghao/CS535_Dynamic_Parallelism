#ifndef QUICKSORT_GPU_H
#define QUICKSORT_GPU_H

#include <vector>

unsigned long quicksort_gpu_par(std::vector<int> &list);
template <typename T>
unsigned long quicksort_gpu_dyn(std::vector<T> &list);
void initCuda();
#endif // QUICKSORT_GPU_H
