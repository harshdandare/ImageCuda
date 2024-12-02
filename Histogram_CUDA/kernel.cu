#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Histogram.h"

__global__ void Histogram_CUDA(unsigned char* Image, int* Histogram);

void Histogram_Calculation_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram) {
	unsigned char* Dev_Image = NULL;
	int* Dev_Histogram = NULL;

	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Histogram, 256 * sizeof(int));

	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram, Histogram, 256 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Histogram_CUDA << <Grid_Image, 1 >> > (Dev_Image, Dev_Histogram);

	//copy memory back to CPU from GPU
	cudaMemcpy(Histogram, Dev_Histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	//free up the memory of GPU
	cudaFree(Dev_Histogram);
	cudaFree(Dev_Image);
}

__global__ void Histogram_CUDA(unsigned char* Image, int* Histogram) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = x + y * gridDim.x;

	atomicAdd(&Histogram[Image[Image_Idx]], 1);
}
