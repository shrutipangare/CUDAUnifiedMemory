# CUDA Unified Memory Performance Analysis

A comprehensive performance comparison between CPU vector addition, GPU vector addition with traditional memory management, and GPU vector addition using CUDA Unified Memory. This project provides detailed analysis of memory transfer bottlenecks, kernel execution patterns, and the trade-offs between programming simplicity and performance optimization.

**Key Finding**: Memory bandwidth, not compute throughput, dominates performance for vector addition operations, making Unified Memory an attractive option with minimal performance penalty.

## Project Structure
├── Batchfiles PartB/          # Batch scripts for automated execution
├── Results PartB/             # Performance results and analysis
├── CPUVecAdd                  # CPU vector addition executable
├── CPUVecAdd.cpp             # CPU implementation source code
├── GPUVecAdd                 # GPU vector addition (traditional) executable
├── GPUVecAdd.cu              # GPU implementation with cudaMalloc/cudaMemcpy
├── VecAddUM                  # GPU vector addition with Unified Memory executable
├── VecAddUM.cu               # GPU implementation with cudaMallocManaged
├── vecadd.cu                 # Additional vector addition implementation
├── vecaddKernel.h            # CUDA kernel header file
├── Makefile                  # Build configuration
└── README.md                 # Project documentation

## Features

- **CPU Vector Addition**: Baseline implementation for performance comparison
- **Traditional GPU Memory Management**: Using `cudaMalloc()` and `cudaMemcpy()`
- **CUDA Unified Memory**: Simplified memory management with `cudaMallocManaged()`
- **Performance Profiling**: Comprehensive timing and analysis across different configurations
- **Scalability Testing**: Variable array sizes and thread configurations

## The tasks are divided as follows:

- Vector Addition on CPU:
Implement and profile a C++ program that adds two large arrays. CPUVecAdd
- Vector Addition on GPU without Unified Memory:
Implement the addition using traditional cudaMalloc() and cudaMemcpy() for memory management. GPUVecAdd
- Vector Addition on GPU with Unified Memory:
Utilize cudaMallocManaged() to handle memory which simplifies data handling between CPU and GPU. GPUVecAddUM

## Prerequisites

- **NVIDIA GPU** with CUDA Compute Capability 3.0 or higher
- **CUDA Toolkit** version 10.0 or later
- **GCC/G++** compiler compatible with your CUDA version
- **Make** build system

### Installation

1. **Install CUDA Toolkit**
   ```bash
   # Download from NVIDIA Developer website
   # https://developer.nvidia.com/cuda-downloads
2. **Verify installation
   nvcc --version
    nvidia-smi
git clone https://github.com/yourusername/cuda-unified-memory.git
cd cuda-unified-memory

**CPU Vector Addition
  ./CPUVecAdd [array_size]

**GPU Vector Addition (Traditional):
  ./GPUVecAdd [array_size] [threads_per_block] [blocks]

**GPU Vector Addition (Unified Memory):
  ./VecAddUM [array_size] [threads_per_block] [blocks]

Automated Batch Execution: 
### CPU performance analysis
bash Batchfiles\ PartB/batchscript_CPUVecAdd.sh

### GPU traditional memory management
bash Batchfiles\ PartB/batchscript_GPUVecAdd.sh

### GPU with Unified Memory
bash Batchfiles\ PartB/batchscript_GPUVecAddUM.sh

### CPU Performance:

- Execution time scales linearly with array size
- For K=1 to K=100 million elements: 0.003176s to 0.325852s
- Serves as baseline for GPU performance comparison
- Consistent and predictable performance scaling

### GPU Traditional Memory Management:

- Scenario 1 (1 Block, 1 Thread): Poor performance, slower than CPU for all array sizes
- Scenario 2 (1 Block, 256 Threads): Moderate improvement, still limited by single block
- Scenario 3 (Multiple Blocks, 256 Threads): Best performance with optimal parallelization
- Clear scaling advantage as array size increases
- Memory transfer overhead becomes negligible for larger datasets

### GPU Unified Memory:

- Simplified Programming: Eliminates explicit cudaMemcpy calls
- Performance Impact: Generally slower than traditional GPU approach
- Migration Overhead: Automatic data movement between CPU/GPU adds latency
- Consistency: More predictable performance across different thread configurations
- Still significantly faster than CPU for larger datasets



