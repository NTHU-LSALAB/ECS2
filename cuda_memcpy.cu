#include "stdio.h"
#include <chrono>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>
#define MYCLOCK CLOCK_MONOTONIC

#define ASSERT(x)                                                       \
    do                                                                  \
        {                                                               \
            if (!(x))                                                   \
                {                                                       \
                    fprintf(stdout, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
                    /*exit(EXIT_FAILURE);*/                                 \
                }                                                       \
        } while (0)


#define ASSERTDRV(stmt)				\
    do                                          \
        {                                       \
            CUresult result = (stmt);           \
            ASSERT(CUDA_SUCCESS == result);     \
        } while (0)

#define ASSERTRT(stmt)				\
    do                                          \
        {                                       \
            cudaError_t result = (stmt);           \
            ASSERT(cudaSuccess == result);     \
        } while (0)


int main(int argc, char *argv[])
{
	int dev_id = atoi(argv[1]);
	printf("%d\n", dev_id);
    const unsigned int N = 1048576;
    size_t bytes = 1024 * N;
    printf("%ld\n", bytes);
    cudaSetDevice(dev_id);
    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));
    //int *h_a = (int*)malloc(bytes);
    void *h_a;
    ASSERTDRV(cuMemAllocHost(&h_a, bytes));
    printf("h_a %x\n",h_a);
    printf("&h_a %x\n",&h_a);
    //printf("n %d", cuMemAllocHost(&h_a, bytes));
    cudaDeviceSynchronize();
    //int *d_A;
    //cudaMalloc((int**)&d_A, bytes);
    CUdeviceptr d_A;
    cuMemAlloc(&d_A, bytes);
    printf("d_A %x\n",d_A);
    printf("&d_A %x\n",&d_A);
//    memset(h_a, 0, bytes);
    cudaDeviceSynchronize();
    unsigned int flag = 1;
    cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A);

    cudaFree(0);
//    struct timespec beg, end;
	for(int i=0;i<1000;i++){
//	    clock_gettime(MYCLOCK, &beg);
	    auto t0 = std::chrono::steady_clock::now();
	    //cudaMemcpy(d_A, h_a, bytes, cudaMemcpyHostToDevice);
	    cuMemcpyHtoD(d_A, h_a, bytes);
	    //cuMemcpyDtoH(h_a, d_A, bytes);
	    cudaDeviceSynchronize();
//	    clock_gettime(MYCLOCK, &end);
	    auto t1 = std::chrono::steady_clock::now();
	    printf("%f elapsed.\n", std::chrono::duration<double>(t1-t0).count());
	    //double t = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
	    //printf("Time %lf\n", t);
	   // h_a[0] = i;
	///cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
	}
    return 0;
}
