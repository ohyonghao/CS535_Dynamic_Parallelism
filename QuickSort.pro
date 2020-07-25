TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

QMAKE_CXXFLAGS += -std=c++17 -pthread -O2
LIBS += -pthread

CUDA_OBJECTS_DIR = OBJECTS_DIR/../cuda

#QMAKE_CXXFLAGS += -xcuda
## CUDA_SOURCES - the source (generally .cu) files for nvcc. No spaces in path names
CUDA_SOURCES += \
    quicksort_gpu.cu

# CUDA settings
SYSTEM_NAME = x86_64                   # Depending on your system either 'Win32', 'x64', or 'Win64'
## SYSTEM_TYPE - compiling for 32 or 64 bit architecture
SYSTEM_TYPE = 64

## CUDA_COMPUTE_ARCH - This will enable nvcc to compiler appropriate architecture specific code for different compute versions.
## Multiple architectures can be requested by using a space to seperate. example:

CUDA_COMPUTE_ARCH = 35 75

## CUDA_DEFINES - The seperate defines needed for the cuda device and host methods
CUDA_DEFINES +=

## CUDA_DIR - the directory of cuda such that CUDA\<version-number\ contains the bin, lib, src and include folders
CUDA_DIR= /usr/local/cuda-11.0

## CUDA_LIBS - the libraries to link
CUDA_LIBS= -lcuda -lcudart
CUDA_LIBS_DIR=$$CUDA_DIR/lib64

## CUDA_INC - all includes needed by the cuda files (such as CUDA\<version-number\include)
CUDA_INC+= $$CUDA_DIR/include

## NVCC_OPTIONS - any further options for the compiler
NVCC_OPTIONS += -O2 #--use_fast_math --ptxas-options=-v

## Windows Options
win32:{
CUDA_DIR= /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0"
CUDA_LIBS_DIR=$$CUDA_DIR/lib/x64;
  win32-msvc2019:contains(QMAKE_TARGET.arch, x86_64):{
       #Can also set SYSTEM_TYPE here
       CONFIG(debug, debug|release) {
            #Debug settings
            message("Using x64 Debug arch config MSVC2019 for build")
            #read as: --compiler-options options,... + ISO-standard C++ exception handling
            # + speed over size, + create debug symbols, + code generation multi-threaded debug
            NVCC_OPTIONS += -Xcompiler /EHsc,/O2,/Zi,/MTd -g
        }
        else {
            #Release settings
            message("Using x64 Release arch config MSVC2019 for build")
            #read as: --compiler-options options,... + ISO-standard C++ exception handling
            # + speed over size, + code generation multi-threaded
            NVCC_OPTIONS += -Xcompiler /EHsc,/O2,/MT
        }
    }
}


## correctly formats CUDA_COMPUTE_ARCH to CUDA_ARCH with code gen flags
## resulting format example: -gencode arch=compute_20,code=sm_20
for(_a, CUDA_COMPUTE_ARCH):{
    formatted_arch =$$join(_a,'',' -gencode arch=compute_',',code=sm_$$_a')
    CUDA_ARCH += $$formatted_arch
}

## correctly formats CUDA_DEFINES for nvcc
for(_defines, CUDA_DEFINES):{
    formatted_defines += -D$$_defines
}
CUDA_DEFINES = $$formatted_defines

#nvcc config
CONFIG(debug, debug|release) {
        #Debug settings
        CUDA_OBJECTS_DIR = cudaobj/$$SYSTEM_NAME/Debug
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda_d.commands = $$CUDA_DIR/bin/nvcc -g -G -lineinfo --std=c++17 -D_DEBUG $$CUDA_DEFINES $$NVCC_OPTIONS -I $$CUDA_INC -L$$CUDA_LIBS_DIR $$CUDA_LIBS --machine $$SYSTEM_TYPE $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
        # Release settings
        CUDA_OBJECTS_DIR = cudaobj/$$SYSTEM_NAME/Release
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.commands = $$CUDA_DIR/bin/nvcc --std=c++17  $$CUDA_DEFINES $$NVCC_OPTIONS -I $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
}

LIBS += -L$$CUDA_LIBS_DIR $$CUDA_LIBS

INCLUDEPATH += $$CUDA_DIR/targets/x86_64-linux/include


HEADERS += \
    quicksort_cpu.h

