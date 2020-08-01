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
void run_benchmark( vector<T> list, F f, bool benchmark ){
    chrono::system_clock::time_point start;
    chrono::system_clock::time_point stop;

    start = chrono::high_resolution_clock::now();
    f(list);
    stop = chrono::high_resolution_clock::now();

    if( !benchmark ) cout << "Duration: ";
    else cout << ", ";
    cout << chrono::duration_cast<chrono::microseconds>(stop - start).count();
    if( !benchmark )
        cout << " microseconds" << endl
             << list << endl;

    if(!is_sorted(list.begin(),list.end())){
        if( !benchmark )
            cout << "!!Sort failed!!" << endl;
    }
}

// Runs the quicksort algorithm on our suite of implementations
void run_quicksort( size_t length, bool benchmark ){

    const auto data{generateList<int>( length )};

    if( !benchmark ) cout << data << endl;
    //***********************************************************************************************
    //***********************************************************************************************
    if( !benchmark ) cout << "CPU Seq" << endl;
    run_benchmark(data, quicksort_cpu_seq<int>, benchmark);

    //***********************************************************************************************
    //***********************************************************************************************
    if(!benchmark) cout << "CPU Par" << endl;
    run_benchmark(data, quicksort_cpu_par<int>, benchmark);

    //***********************************************************************************************
    //***********************************************************************************************
    if(!benchmark) cout << "GPU Par" << endl;
    run_benchmark(data, quicksort_gpu_par, benchmark);

    //***********************************************************************************************
    //***********************************************************************************************
    if(!benchmark) cout << "GPU Dynamic Par" << endl;
    run_benchmark(data, quicksort_gpu_dyn, benchmark);

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
    size_t length = stoul(argv[1]);
    size_t iterations = stoul(argv[2]);

    initCuda();
    if( benchmark ) cout << "id, cpu_seq, cpu_par, gpu_par, gpu_dyn" << endl;
    for( size_t i = 0; i < iterations; ++i ){
        if( !benchmark ) cout << "Iteration " << i << ":" << endl;
        else cout << i;
        run_quicksort(length, benchmark);
        cout << endl;
    }

    return 0;
}
