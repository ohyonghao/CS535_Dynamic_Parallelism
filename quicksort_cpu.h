#ifndef QUICKSORT_CPU_H
#define QUICKSORT_CPU_H

#include <algorithm>
#include <execution>
#include <vector>

/******************************************************************************
 * Implementation comes from overview at https://www.geeksforgeeks.org/quick-sort/
 ******************************************************************************/

template <typename T>
void quicksort_cpu_seq(std::vector<T> &list);
template <typename T>
size_t partition_cpu_seq( std::vector<T>& list, int low, int high );
template <typename T>
void quicksort_cpu_seq_imp( std::vector<T> &list, int low, int high);

template <typename T>
void quicksort_cpu_seq(std::vector<T> &list)
{
    quicksort_cpu_seq_imp(list, 0, list.size() - 1);
}


template <typename T>
size_t partition_cpu_seq( std::vector<T>& list, int low, int high )
{
    auto pivot = list[high];

    int i = low - 1;

    for( int j = low; j <= high - 1; ++j ){
        if( list[j] < pivot ){
            ++i;
            std::swap( list[i], list[j]);
        }
    }

    std::swap( list[i+1], list[high]);
    return i + 1;
}

template <typename T>
void quicksort_cpu_seq_imp( std::vector<T> &list, int low, int high){
    if( low < high )
    {
        auto p_i = partition_cpu_seq( list, low, high );

        quicksort_cpu_seq_imp( list, low, p_i - 1);
        quicksort_cpu_seq_imp( list, p_i + 1, high);
    }
}

/******************************************************************************
 * Currently just uses the TBB sort parallel algorithm, can change to cpu parallel
 * quick sort later on.
*******************************************************************************/

template <typename T>
void quicksort_cpu_par(std::vector<T> &list)
{
    sort(std::execution::par, list.begin(), list.end());
}


#endif // QUICKSORT_CPU_H
