#include <stdio.h>

#define N 512
#define FILTER_SIZE 3

__global__ void imageBlur(float* input, float* output, float* filter) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int i = -FILTER_SIZE / 2; i <= FILTER_SIZE / 2; i++) {
            for (int j = -FILTER_SIZE / 2; j <= FILTER_SIZE / 2; j++) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < N && c >= 0 && c < N) {
                    sum += input[r * N + c] * filter[(i + FILTER_SIZE / 2) * FILTER_SIZE + (j + FILTER_SIZE / 2)];
                }
            }
        }
        output[row * N + col] = sum;
    }
}

int main() {
    float* h_input, * h_output, * h_filter;
    float* d_input, * d_output, * d_filter;
    int size = N * N * sizeof(float);
    int filterSize = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    // Allocate host memory
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    h_filter = (float*)malloc(filterSize);

    // Initialize input image and filter
    for (int i = 0; i < N * N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
    }
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
        h_filter[i] = 1.0f / (FILTER_SIZE * FILTER_SIZE);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_filter, filterSize);

    // Copy input data and filter from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch the imageBlur kernel on the GPU
    imageBlur << <gridSize, blockSize >> > (d_input, d_output, d_filter);

    // Copy the result from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the result (part of it)
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f\t", h_output[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    // Free host memory
    free(h_input);
    free(h_output);
    free(h_filter);

    return 0;
}
