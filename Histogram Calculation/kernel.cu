#include <stdio.h>

#define N 1024
#define NUM_BINS 256

__global__ void histogram(float* input, int* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        int bin = static_cast<int>(input[tid] * NUM_BINS);
        atomicAdd(&output[bin], 1);
    }
}

int main() {
    float* h_input;
    int* h_output;
    float* d_input;
    int* d_output;
    int size = N * sizeof(float);

    // Allocate host memory
    h_input = (float*)malloc(size);
    h_output = (int*)calloc(NUM_BINS, sizeof(int));

    // Initialize input array
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, NUM_BINS * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the histogram kernel on the GPU
    histogram << <gridSize, blockSize >> > (d_input, d_output, N);

    // Copy the result from device to host
    cudaMemcpy(h_output, d_output, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the histogram
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d: %d\n", i, h_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
