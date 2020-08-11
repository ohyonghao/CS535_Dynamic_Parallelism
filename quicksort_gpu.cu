#include "quicksort_gpu.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <chrono>
#include <algorithm>


template <typename T>
__device__
void swap_d(T& lhs, T& rhs){
    T tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

template <typename T>
__device__
void selection_sort_gpu(T* d, int low, int high){
    for( auto i = low; i <= high; ++i){
        auto min_val = d[i];
        auto min_idx = i;

        for( int j = i + 1; j <= high; ++j ){
            auto val_j = d[j];

            if( val_j < min_val ){
                min_idx = j;
                min_val = val_j;
            }
        }
        swap_d(d[i], d[min_idx]);
    }
}

template <typename T>
__device__
std::pair<int,int> partition_gpu( T* d, int low, int high){
    // Take pivot in center
    auto pivot = d[(high + low) >> 1];

    while( low < high ){
        // Move low index up while low value is less than pivot
        while( d[low] < pivot ){
            ++low;
        }
        // Move high index down while high value is greater than pivot
        while( d[high] > pivot ){
            --high;
        }

        // Swap points are valid, do the swap!???
        if( low <= high ){
            swap_d(d[low++],d[high--]);
        }

    }

    return std::make_pair(low, high);
}

constexpr int MAX_DEPTH = 24;
constexpr int MIN_SIZE = 1024; // 32 gets us about 10xcpu, 1024 gets about 3xcpu
template <typename T>
__global__
void quicksort_gpu_dyn_worker( T* d, int low, int high, int level ){
    if( high <= low ) return;
    if( level >= MAX_DEPTH || (high - low) < MIN_SIZE ){
        selection_sort_gpu(d, low, high);
        return;
    }

    auto lh = partition_gpu( d, low, high );

    cudaStream_t s_l, s_h;
    cudaStreamCreateWithFlags(&s_l, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_h, cudaStreamNonBlocking);
    quicksort_gpu_dyn_worker<<<1, 1, 0, s_l>>>(d, low, lh.second, level + 1);
    quicksort_gpu_dyn_worker<<<1, 1, 0, s_h>>>(d, lh.first, high, level + 1 );

    cudaStreamDestroy(s_l);
    cudaStreamDestroy(s_h);
}

template <typename T>
__host__
unsigned long quicksort_gpu_dyn(std::vector<T> &list){
    // Detect settings
    // Copy to device
    T *d{};
    auto err = cudaMalloc((void**)&d, sizeof(T) * list.size() );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Malloc" << err << std::endl;
        return 0;
    }
    err = cudaMemcpy((void**)d, list.data(), sizeof(T) * list.size(), cudaMemcpyHostToDevice );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Memcpy->Device" << err << std::endl;
        cudaFree(d);
        return 0;
    }

    err = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
    if( err != cudaSuccess ){
        std::cout << "CUDA ERROR set cudaLimitDevRuntimeSyncDepth" << std::endl;
        return 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    quicksort_gpu_dyn_worker<<<1,1>>>(d, 0, list.size() - 1, 0);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    err = cudaMemcpy(list.data(), (void**)d, sizeof(T) * list.size(), cudaMemcpyDeviceToHost );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Memcpy->Host" << err << std::endl;
        cudaFree(d);
        return 0;
    }
    cudaFree(d);

    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
}


unsigned long quicksort_gpu_par(std::vector<int> &list){
    // Detect settings
    // Copy to device
    int *d{};
    auto err = cudaMalloc((void**)&d, sizeof(int) * list.size() );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Malloc" << err << std::endl;
        return 0;
    }
    err = cudaMemcpy((void**)d, list.data(), sizeof(int) * list.size(), cudaMemcpyHostToDevice );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Memcpy->Device" << err << std::endl;
        cudaFree(d);
        return 0;
    }
    err = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 24);
    if( err != cudaSuccess ){
        std::cout << "CUDA ERROR set cudaLimitDevRuntimeSyncDepth" << std::endl;
        return 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    quicksort_gpu_dyn_worker<<<1,1>>>(d, 0, list.size() - 1, 0);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    err = cudaMemcpy(list.data(), (void**)d, sizeof(int) * list.size(), cudaMemcpyDeviceToHost );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Memcpy->Host" << err << std::endl;
        cudaFree(d);
        return 0;
    }
    cudaFree(d);

    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
}

__host__
void initCuda(){cudaFree(0);}


template unsigned long quicksort_gpu_dyn<int>(std::vector<int> &list);
template unsigned long quicksort_gpu_dyn<float>(std::vector<float> &list);
