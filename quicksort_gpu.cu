#include "quicksort_gpu.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

template <typename T>
__device__
void swap_d(T& lhs, T& rhs){
    T tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

__device__
size_t partition_gpu_par( int* d, int low, int high){
    auto pivot = d[high];

    int i = low - 1;

    for( int j = low; j <= high - 1; ++j){
        if( d[j] < pivot ){
            ++i;
            swap_d(d[j], d[i]);
        }
    }

    swap_d( d[i+1], d[high]);
    return i + 1;
}

__global__
void quicksort_gpu_par_worker( int* d, int low, int high ){
    if( high >= low ) return;

    auto p = partition_gpu_par( d, low, high );

    cudaStream_t s_l, s_h;
    cudaStreamCreateWithFlags(&s_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_h, cudaStreamNonBlocking);
    quicksort_gpu_par_worker<<<1, 1, 0, s_l>>>(d, low, p - 1);
    quicksort_gpu_par_worker<<<1, 1, 0, s_h>>>(d, p + 1, high );

    cudaStreamDestroy(s_l);
    cudaStreamDestroy(s_h);
}

__host__
void quicksort_gpu_par(std::vector<int> &list){
    // Detect settings
    // Copy to device
    int *d;
    cudaMalloc((void**)d, sizeof(int) * list.size() );
    quicksort_gpu_par_worker<<<1,1>>>(d, 0, list.size() - 1);
    cudaMemcpy((void**)d, list.data(), sizeof(int) * list.size(), cudaMemcpyHostToDevice );
    cudaFree(d);
    // Launch kernel
    // Copy to host
}


void quicksort_gpu_dyn(std::vector<int> &list){
    // Detect settings
    // Copy to device
    // Launch kernel
    // Copy to host
}

__host__
void initCuda(){cudaFree(0);}
