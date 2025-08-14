# CUDA Unified Memory Performance Analysis

A comprehensive performance comparison between CPU vector addition, GPU vector addition with traditional memory management, and GPU vector addition using CUDA Unified Memory. This project provides detailed analysis of memory transfer bottlenecks, kernel execution patterns, and the trade-offs between programming simplicity and performance optimization.

**Key Finding**: Memory bandwidth, not compute throughput, dominates performance for vector addition operations, making Unified Memory an attractive option with minimal performance penalty.

## Project Structure

```
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
```

## Features

- **CPU Vector Addition**: Baseline implementation for performance comparison
- **Traditional GPU Memory Management**: Using `cudaMalloc()` and `cudaMemcpy()`
- **CUDA Unified Memory**: Simplified memory management with `cudaMallocManaged()`
- **Performance Profiling**: Comprehensive timing and analysis across different configurations
- **Scalability Testing**: Variable array sizes and thread configurations

## Task Overview

The tasks are divided as follows:

- **Vector Addition on CPU**: Implement and profile a C++ program that adds two large arrays (`CPUVecAdd`)
- **Vector Addition on GPU without Unified Memory**: Implement the addition using traditional `cudaMalloc()` and `cudaMemcpy()` for memory management (`GPUVecAdd`)
- **Vector Addition on GPU with Unified Memory**: Utilize `cudaMallocManaged()` to handle memory which simplifies data handling between CPU and GPU (`GPUVecAddUM`)

## Prerequisites

- **NVIDIA GPU** with CUDA Compute Capability 3.0 or higher
- **CUDA Toolkit** version 10.0 or later
- **GCC/G++** compiler compatible with your CUDA version
- **Make** build system

## Installation

1. **Install CUDA Toolkit**
   ```bash
   # Download from NVIDIA Developer website
   # https://developer.nvidia.com/cuda-downloads
   ```

2. **Verify installation**
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **Clone and build the project**
   ```bash
   git clone https://github.com/yourusername/cuda-unified-memory.git
   cd cuda-unified-memory
   make
   ```

## Usage

### Manual Execution

**CPU Vector Addition**
```bash
./CPUVecAdd [array_size]
```

**GPU Vector Addition (Traditional)**
```bash
./GPUVecAdd [array_size] [threads_per_block] [blocks]
```

**GPU Vector Addition (Unified Memory)**
```bash
./VecAddUM [array_size] [threads_per_block] [blocks]
```

### Automated Batch Execution

```bash
# CPU performance analysis
bash "Batchfiles PartB/batchscript_CPUVecAdd.sh"

# GPU traditional memory management
bash "Batchfiles PartB/batchscript_GPUVecAdd.sh"

# GPU with Unified Memory
bash "Batchfiles PartB/batchscript_GPUVecAddUM.sh"
```

## Performance Results

### CPU Performance

- Execution time scales linearly with array size
- For K=1 to K=100 million elements: 0.003176s to 0.325852s
- Serves as baseline for GPU performance comparison
- Consistent and predictable performance scaling

### GPU Traditional Memory Management

- **Scenario 1** (1 Block, 1 Thread): Poor performance, slower than CPU for all array sizes
- **Scenario 2** (1 Block, 256 Threads): Moderate improvement, still limited by single block
- **Scenario 3** (Multiple Blocks, 256 Threads): Best performance with optimal parallelization
- Clear scaling advantage as array size increases
- Memory transfer overhead becomes negligible for larger datasets

### GPU Unified Memory

- **Simplified Programming**: Eliminates explicit `cudaMemcpy` calls
- **Performance Impact**: Generally slower than traditional GPU approach
- **Migration Overhead**: Automatic data movement between CPU/GPU adds latency
- **Consistency**: More predictable performance across different thread configurations
- **Still significantly faster than CPU for larger datasets**

## Key Insights

1. **Memory Bandwidth Dominance**: For simple operations like vector addition, memory bandwidth is the primary performance bottleneck, not computational throughput.

2. **Unified Memory Trade-off**: While Unified Memory simplifies programming and reduces code complexity, it introduces some performance overhead due to automatic memory migration.

3. **Scalability**: All GPU implementations show better scaling characteristics compared to CPU as array sizes increase.

4. **Thread Configuration Impact**: Proper thread and block configuration is crucial for optimal GPU performance, particularly in traditional memory management scenarios.

## Example Usage

### Running CPU Vector Addition
```bash
# Add two arrays of 1 million elements each
./CPUVecAdd 1000000

# Output example:
# Array size: 1000000
# CPU Execution Time: 0.003176 seconds
```

### Running GPU Vector Addition (Traditional)
```bash
# Add two arrays with optimal configuration
./GPUVecAdd 1000000 256 3906

# Parameters:
# - array_size: 1000000 elements
# - threads_per_block: 256
# - blocks: 3906 (calculated as ceil(array_size/threads_per_block))
```

### Running GPU Vector Addition (Unified Memory)
```bash
# Same operation using unified memory
./VecAddUM 1000000 256 3906

# Simplified memory management with automatic data migration
```

## Building the Project

### Using Make
```bash
make all          # Build all executables
make CPUVecAdd    # Build CPU version only
make GPUVecAdd    # Build GPU traditional version only
make VecAddUM     # Build GPU unified memory version only
make clean        # Clean build artifacts
```

### Manual Compilation
```bash
# CPU Version
g++ -o CPUVecAdd CPUVecAdd.cpp

# GPU Traditional Version
nvcc -o GPUVecAdd GPUVecAdd.cu

# GPU Unified Memory Version
nvcc -o VecAddUM VecAddUM.cu
```

## Troubleshooting

### Common Issues

1. **CUDA Not Found**
   ```bash
   # Add CUDA to PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **GPU Not Detected**
   ```bash
   # Check GPU availability
   nvidia-smi
   # Check CUDA installation
   nvcc --version
   ```

3. **Compilation Errors**
   - Ensure CUDA Toolkit version compatibility with your GPU
   - Check GCC version compatibility with CUDA
   - Verify all source files are present

## Performance Optimization Tips

1. **Optimal Thread Configuration**: Use multiples of 32 threads per block (warp size)
2. **Memory Coalescing**: Ensure memory access patterns are optimized
3. **Block Size**: Experiment with different block sizes (128, 256, 512, 1024)
4. **Grid Size**: Calculate grid size as `ceil(N/block_size)` for optimal coverage

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA CUDA Documentation and Samples
- CUDA Programming Guide
- Performance optimization techniques from GPU computing community



