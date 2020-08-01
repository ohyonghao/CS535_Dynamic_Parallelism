#ifndef QUICKSORT_GPU_H
#define QUICKSORT_GPU_H

#include <vector>

void quicksort_gpu_par(std::vector<int> &list);
void quicksort_gpu_dyn(std::vector<int> &list);
void initCuda();
#endif // QUICKSORT_GPU_H
