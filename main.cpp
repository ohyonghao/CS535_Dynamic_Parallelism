#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <functional>
#include <random>

#include "quicksort_cpu.h"
#include "quicksort_gpu.h"

using namespace std;

// Generates a list of size length using a uniform distribution
template <typename T>
vector<T> generateList( size_t length ){
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_int_distribution<T> distribution(0, length);

    vector<T> v(length);
    generate(v.begin(),v.end(),[&]{ return distribution(generator);});

    return v;
}

template<>
vector<float> generateList<float>( size_t length ){
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_real_distribution<float> distribution(0, length);

    vector<float> v(length);
    generate(v.begin(),v.end(),[&]{ return distribution(generator);});

    return v;
}

// Convenience function for printing lists of items in a vector
template <typename T>
ostream& operator<<(ostream& out, const vector<T> &list){

    bool first = true;
    for( const auto &item: list ){
        if( !first ) out << ", ";
        else first = false;
        out << item;
    }
    return out;
}

// Templatized function to take a list ( pass by copy intended ), a function to run, and whether to produce
// benchmark output
template <typename T, typename F>
long run_benchmark( vector<T> list, F f, bool benchmark ){

    auto start = chrono::high_resolution_clock::now();
    f(list);
    auto stop = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    if( !benchmark ) cout << ", " << duration;

    if( !benchmark && !is_sorted(list.begin(),list.end()) ) cout << " !!Sort failed!! ";

    return duration;
}

// Runs the quicksort algorithm on our suite of implementations
vector<long> run_quicksort( size_t length, bool benchmark ){

    const auto data{generateList<int>( length )};

    vector<long> runtime;
    if( !benchmark ) cout << data << endl;
    //***********************************************************************************************
    //***********************************************************************************************
    if( !benchmark ) cout << "CPU Seq" << endl;
    runtime.push_back(run_benchmark(data, quicksort_cpu_seq<int>, benchmark));

    //***********************************************************************************************
    //***********************************************************************************************
    if(!benchmark) cout << "CPU Par" << endl;
    runtime.push_back(run_benchmark(data, quicksort_cpu_par<int>, benchmark));

    //***********************************************************************************************
    //***********************************************************************************************
    if(!benchmark) cout << "GPU Dynamic Par" << endl;
    runtime.push_back(run_benchmark(data, quicksort_gpu_dyn<int>, benchmark));

    //***********************************************************************************************
    //***********************************************************************************************
    if(!benchmark) cout << "GPU Non Dyn" << endl;
    runtime.push_back(run_benchmark(data, quicksort_cpu_coordinated<int>, benchmark));

    return runtime;
}

int main(int argc, char** argv){
    if( argc < 3 ){
        cout << "Usage: QuickSort <length> <iterations> <benchmark>" << endl;
        cout << "    length: the length of random list to generate" << endl;
        cout << "    iterations: number of iterations to print" << endl;
        cout << "    benchmark: prints benchmark information in csv format to standard out" << endl;
        return 0;
    }

    bool benchmark{argc >= 4};
    const size_t length = stoul(argv[1]);
    const size_t iterations = stoul(argv[2]);

    vector<vector<long>> runtimes;
    initCuda();
    if( !benchmark ) cout << "id, cpu_seq, cpu_par, gpu_dyn, gpu_non_dyn" << endl;
    for( size_t i = 0; i < iterations; ++i ){
        if( !benchmark ) cout << i << endl;
        runtimes.push_back(run_quicksort(length, benchmark));
    }

    // Print averages
    if( benchmark ){
        vector<double> averages(4, 0);
        // transform to averages
        for( auto i = 0ul; i < iterations; ++i ){
            for( auto j = 0ul; j < 4ul; ++j)
                averages[j] += runtimes[i][j]/4.0;
        }
        cout << averages << endl;
    }
    return 0;
}
