TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

QMAKE_CXXFLAGS += -std=c++17 -pthread -O2
LIBS += -pthread -ltbb

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

CUDA_COMPUTE_ARCH = 75

## CUDA_DEFINES - The seperate defines needed for the cuda device and host methods
CUDA_DEFINES +=

## CUDA_DIR - the directory of cuda such that CUDA\<version-number\ contains the bin, lib, src and include folders
CUDA_DIR= /usr/local/cuda-11.0

## CUDA_LIBS - the libraries to link
CUDA_LIBS= -lcudart -lcudadevrt
CUDA_LIBS_DIR=$$CUDA_DIR/lib64

## CUDA_INC - all includes needed by the cuda files (such as CUDA\<version-number\include)
CUDA_INC+= $$CUDA_DIR/include

## NVCC_OPTIONS - any further options for the compiler
NVCC_OPTIONS += -O2 #--use_fast_math --ptxas-options=-v

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
        # Compile only
        CUDA_OBJECTS_DIR = cudaobj/$$SYSTEM_NAME/Debug
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.variable_out = CUDA_OBJ
        cuda.variable_out += OBJECTS
        cuda.commands = $$CUDA_DIR/bin/nvcc --machine $$SYSTEM_TYPE -g -G $$CUDA_ARCH -dc $$CUDA_DEFINES $$NVCC_OPTIONS -I $$CUDA_INC -o ${QMAKE_FILE_OUT} -c ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        cuda.clean = $$CUDA_OBJECTS_DIR/*.o
        QMAKE_EXTRA_COMPILERS += cuda


        # nvcc Link
        cuda_link.input = CUDA_OBJ
        cuda_link.output = ${QMAKE_FILE_BASE}_cuda_all.o
        cuda_link.commands = $$CUDA_DIR/bin/nvcc \
            --machine $$SYSTEM_TYPE -g -G  \
            $$CUDA_ARCH  -L$$CUDA_LIBS_DIR $$CUDA_LIBS \
            -dlink -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda_link.dependency_type = TYPE_C
        cuda_link.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCC_OPTIONS ${QMAKE_FILE_NAME}
        QMAKE_EXTRA_COMPILERS += cuda_link
}
else {
        # Release settings
        CUDA_OBJECTS_DIR = cudaobj/$$SYSTEM_NAME/Release
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.variable_out = CUDA_OBJ
        cuda.variable_out += OBJECTS
        cuda.commands = $$CUDA_DIR/bin/nvcc --machine $$SYSTEM_TYPE $$CUDA_ARCH -dc $$CUDA_DEFINES $$NVCC_OPTIONS -I $$CUDA_INC -o ${QMAKE_FILE_OUT} -c ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda

        # nvcc Link
        cuda_link.input = CUDA_OBJ
        cuda_link.output = ${QMAKE_FILE_BASE}_cuda_all.o
        cuda_link.commands = $$CUDA_DIR/bin/nvcc \
            --machine $$SYSTEM_TYPE  \
            $$CUDA_ARCH  -L$$CUDA_LIBS_DIR $$CUDA_LIBS \
            -dlink -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda_link.dependency_type = TYPE_C
        cuda_link.depend_command = $$CUDA_DIR/bin/nvcc -M $$CUDA_INC $$NVCC_OPTIONS ${QMAKE_FILE_NAME}
        QMAKE_EXTRA_COMPILERS += cuda_link
}


LIBS += -L$$CUDA_LIBS_DIR $$CUDA_LIBS

INCLUDEPATH += $$CUDA_DIR/targets/x86_64-linux/include


HEADERS += \
    quicksort_cpu.h \
    quicksort_gpu.h

