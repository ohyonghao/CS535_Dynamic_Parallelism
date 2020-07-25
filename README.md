# CS535_Dynamic_Parallelism

# Building
This uses `qmake` from the Qt Framework to generate and build. This is setup to send `cu` files to the nvidia
compiler and everything else to `g++`, and the linker links them together.

## Qmake then Make
```
qmake
make
```

# Running

This software is setup for benchmarking and debugging purposes. Usage information can be found by running
the program with no arguments. Arguments are currently in a sequential order.

# Example

```
./QuickSort 1000 10 1
```

Runs each of the quick sort implementations on 10 different vectors of 1000 elements that are randomly generated
and may include repeats on the set {0, 1000 - 1}. The same vector is passed to each implementation, benchmarks
made, then a new vector is generated. The 1 at the end is simply an argument that tells it to produce benchmark
output rather than more verbose output.
