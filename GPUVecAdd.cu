#include <iostream>
#include <cmath>

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(int argc, char** argv) {
    int millions = 1;
    int N = 1 << 20;
    int grid_size = 1;
    int block_size = 1;

    if (argc == 4) {
        sscanf(argv[1], "%d", &grid_size);
        sscanf(argv[2], "%d", &block_size);
		sscanf(argv[3], "%d", &millions);
    }

    std::cout << "K: " << millions << ", Grid size: " << grid_size << ", Block size: " << block_size << std::endl;
    
	N = millions * N;
    size_t size = N * sizeof(float);
    float *device_x, *device_y;
    
	// Allocate input vectors host_x and host_y in host memory
    float* host_x = (float*)malloc(size);
    float* host_y = (float*)malloc(size);

    // Initialize host_x and host_y arrays on the host
    for (int i = 0; i < N; i++) {
        host_x[i] = 1.0f;
        host_y[i] = 2.0f;
    }

    // Allocate vectors in device memory
    cudaMalloc(&device_x, size);
    cudaMalloc(&device_y, size);
    
	// Copy vectors from host memory to device global memory
    cudaMemcpy(device_x, host_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(grid_size);
    dim3 dimBlock(block_size);

    // Invoke kernel
    add<<<dimGrid, dimBlock>>>(N, device_x, device_y);
    cudaMemcpy(host_y, device_y, size, cudaMemcpyDeviceToHost);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, std::fabs(host_y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;
	// Free memory
    cudaFree(device_x); 
    cudaFree(device_y);
    free(host_x); 
    free(host_y);
    return 0;
}