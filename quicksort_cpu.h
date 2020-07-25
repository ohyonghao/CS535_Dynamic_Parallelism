#ifndef QUICKSORT_CPU_H
#define QUICKSORT_CPU_H

#include <algorithm>
#include <execution>
#include <vector>

template <typename T>
void quicksort_cpu_seq(std::vector<T> &list)
{
    sort(list.begin(), list.end());
}

template <typename T>
void quicksort_cpu_par(std::vector<T> &list)
{
    sort(std::execution::par, list.begin(), list.end());
}


#endif // QUICKSORT_CPU_H
