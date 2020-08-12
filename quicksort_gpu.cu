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
    for( auto i = low; i <= high - 1; ++i){
        auto min_val = d[i];
        auto min_idx = i;

        for( int j = i + 1; j <= high; ++j ){
            auto val_j = d[j];

            if( val_j < min_val ){
                min_idx = j;
                min_val = val_j;
            }
        }
        if( i != min_idx )
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
void quicksort_gpu_dyn(std::vector<T> &list){
    // Detect settings
    // Copy to device
    T *d{};
    auto err = cudaMalloc((void**)&d, sizeof(T) * list.size() );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Malloc" << err << std::endl;
        return;
    }
    err = cudaMemcpy((void**)d, list.data(), sizeof(T) * list.size(), cudaMemcpyHostToDevice );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Memcpy->Device" << err << std::endl;
        cudaFree(d);
        return;
    }

    err = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
    if( err != cudaSuccess ){
        std::cout << "CUDA ERROR set cudaLimitDevRuntimeSyncDepth" << std::endl;
        return;
    }

    quicksort_gpu_dyn_worker<<<1,1>>>(d, 0, list.size() - 1, 0);
    cudaDeviceSynchronize();

    err = cudaMemcpy(list.data(), (void**)d, sizeof(T) * list.size(), cudaMemcpyDeviceToHost );
    if( err != cudaSuccess ) {
        std::cout << "CUDA ERROR Memcpy->Host" << err << std::endl;
        cudaFree(d);
        return;
    }
    cudaFree(d);
}

constexpr unsigned int TILE_WIDTH = 32;
template <typename T>
__global__
void ss_worker( T* data, size_t * stack, unsigned int size){
    auto idx = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if( idx >= size ) return;

    size_t low  = stack[idx * 2];
    size_t high = stack[idx * 2 + 1];
    selection_sort_gpu(data, low, high);
}

template <typename T>
__global__
void qs_worker( T* data, size_t * qs_stack, size_t * qs_result_stack, size_t * ss_stack, unsigned int *qs_stack_size, unsigned int *ss_stack_size, unsigned int size){
    // thread_id to access the stack
    auto idx = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // guard against too many threads
    if( idx >= size ) return;

    // get the bounds
    size_t low  = qs_stack[idx * 2];
    size_t high = qs_stack[idx * 2 + 1];

    // partition
    auto lh = partition_gpu( data, low, high);

    if( low < lh.second ){
        // if below MIN_SIZE send to selection sort
        if( lh.second - low < MIN_SIZE ){
            int idx = atomicAdd(ss_stack_size, 1);
            ss_stack[idx * 2] = low;
            ss_stack[idx * 2 + 1] = lh.second;
        }else{
            int idx = atomicAdd(qs_stack_size, 1);
            qs_result_stack[idx * 2] = low;
            qs_result_stack[idx * 2 + 1] = lh.second;
        }
    }

    if( lh.first < high ){
        // if below MIN_SIZE send to selection sort
        if( high - lh.first < MIN_SIZE ){
            int idx = atomicAdd(ss_stack_size, 1);
            ss_stack[idx * 2] = lh.first;
            ss_stack[idx * 2 + 1] = high;
        }else{
            int idx = atomicAdd(qs_stack_size, 1);
            qs_result_stack[idx * 2] = lh.first;
            qs_result_stack[idx * 2 + 1] = high;
        }
    }
}

template <typename T>
void quicksort_cpu_coordinated(std::vector<T> &list){
    if( list.empty() ) return;

    // data
    T *d{nullptr};

    // QuickSort Stack
    size_t * qs_stack{nullptr};
    unsigned int * qs_stack_size{nullptr};
    size_t * qs_result_stack{nullptr};

    // SelectionSort Stack
    size_t * ss_stack{nullptr};
    unsigned int * ss_stack_size{nullptr};
    size_t * ss_worker_stack{nullptr};

    // local sizes
    unsigned int h_qs = 1;
    unsigned int h_ss = 0;

    // prime the stack

    auto err = cudaMalloc(reinterpret_cast<void**>(&qs_stack_size), sizeof(unsigned int));
         err = cudaMalloc(reinterpret_cast<void**>(&ss_stack_size), sizeof(unsigned int));
         err = cudaMalloc(reinterpret_cast<void**>(&qs_stack),        sizeof(size_t) * 2);
         err = cudaMalloc(reinterpret_cast<void**>(&qs_result_stack), sizeof(size_t) * 4);
         err = cudaMalloc(reinterpret_cast<void**>(&ss_stack),        sizeof(size_t) * 4);
         err = cudaMalloc(reinterpret_cast<void**>(&d),               sizeof(T) * list.size() );

    if( err != cudaSuccess ){
        std::cout << "CUDA ERROR getting memory" << std::endl;
        return;
    }

    // Copy data over
    err = cudaMemcpy(reinterpret_cast<void**>(d), list.data(), sizeof(T)*list.size(), cudaMemcpyHostToDevice);

    // Initialize first round
    err = cudaMemcpy(reinterpret_cast<void**>(qs_stack_size), reinterpret_cast<void**>(&h_qs), sizeof(unsigned int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(reinterpret_cast<void**>(ss_stack_size), reinterpret_cast<void**>(&h_ss), sizeof(unsigned int), cudaMemcpyHostToDevice);

    std::vector<size_t> init{0, list.size()-1};
    err = cudaMemcpy(reinterpret_cast<void**>(qs_result_stack), init.data(), sizeof(size_t) * 2, cudaMemcpyHostToDevice);

    if( err != cudaSuccess ){
        std::cout << "CUDA ERROR copying memory" << std::endl;
        return;
    }

    // Check if there is any work to do
    while( h_qs || h_ss ){
        if( h_ss ){
            // loads the ss_stack into ss_worker_stack
            std::swap( ss_worker_stack, ss_stack);

            // launch kernel for selection sort
            ss_worker<<<(h_ss + TILE_WIDTH - 1 )/TILE_WIDTH, TILE_WIDTH >>>(d, ss_worker_stack, h_ss );

            // reset size of stack
            h_ss = 0;
            err = cudaMemcpy(reinterpret_cast<void**>(ss_stack_size), reinterpret_cast<void**>(&h_ss), sizeof(unsigned int), cudaMemcpyHostToDevice);
        }

        // free ss stack ( this was either unused last round, or we are finished)
        cudaFree(ss_stack);
        ss_stack = nullptr;

        if( h_qs ){
            // load the qs_result_stack into qs_stack
            std::swap(qs_stack, qs_result_stack);
            cudaFree(qs_result_stack);

            // make both stacks large enough to accomodate all partitions
            size_t size = h_qs;
            err = cudaMalloc(reinterpret_cast<void**>(&ss_stack), sizeof(size_t) * size * 2 );
            err = cudaMalloc(reinterpret_cast<void**>(&qs_result_stack), sizeof(size_t) * size * 2 );
            if( err != cudaSuccess ){
                std::cout << "CUDA ERROR getting memory in loop" << std::endl;
                return;
            }

            // reset size of stack
            h_qs = 0;
            err = cudaMemcpy(reinterpret_cast<void**>(qs_stack_size), reinterpret_cast<void**>(&h_qs), sizeof(unsigned int), cudaMemcpyHostToDevice);

            // launch kernel
            qs_worker<<<(size + TILE_WIDTH - 1)/TILE_WIDTH, TILE_WIDTH >>>(d, qs_stack, qs_result_stack, ss_stack, qs_stack_size, ss_stack_size, size);
        }

        // wait for work to be done on both QuickSort and SelectionSort
        cudaDeviceSynchronize();

        // Copy back how much work is to be done
        err = cudaMemcpy(reinterpret_cast<void**>(&h_qs), reinterpret_cast<void**>(qs_stack_size), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        err = cudaMemcpy(reinterpret_cast<void**>(&h_ss), reinterpret_cast<void**>(ss_stack_size), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

    // Copy back results
    err = cudaMemcpy(list.data(), (void**)d, sizeof(T) * list.size(), cudaMemcpyDeviceToHost );

    // Deallocate GPU memory
    cudaFree(qs_stack_size);
    cudaFree(qs_stack);
    cudaFree(qs_result_stack);
    cudaFree(ss_stack);
    cudaFree(ss_stack_size);
    cudaFree(ss_worker_stack);
}
__host__
void initCuda(){cudaFree(0);}


template void quicksort_gpu_dyn<int>(std::vector<int> &list);
template void quicksort_gpu_dyn<float>(std::vector<float> &list);
template void quicksort_cpu_coordinated(std::vector<int> &list);
template void quicksort_cpu_coordinated(std::vector<float> &list);
