/*
 * Copyright(c) 2017 H3 Platform, Inc. All rights reserved.
 * Author: Yang Yuanzhi <yyz@h3platform.com>
 *
 * All information contained herein is proprietary and confidential to
 * H3 Platform, Inc. Any use, reproduction, or disclosure without the
 * written permission of H3 Platform, Inc. is prohibited.
 */
/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef	AIO
#include <libaio.h>
#endif
#include <unistd.h>
#include <stdlib.h>
#include <getopt.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pthread.h>

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

#define OUT cout
//#define OUT TESTSTACK

//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC

// manually tuned...
static int num_write_iters = 10000;
static int num_read_iters  = 100;

static pthread_barrier_t barrier;

#define	MAX_GPUS	8

static CUcontext context[MAX_GPUS];
static CUdeviceptr d_A[MAX_GPUS];
static gdr_mh_t mh[MAX_GPUS];
static void *bar_ptr[MAX_GPUS];
static uint32_t *buf_ptr[MAX_GPUS];
static void *host_buf[MAX_GPUS];

static size_t copy_size = 0;
static size_t copy_offset = 0;

static uint32_t *init_buf;

static int compare = 1;

static void gpu_memory_init(gdr_t g, int dev_id, size_t size)
{
    OUT << "selecting device " << dev_id << endl;
    ASSERTRT(cudaSetDevice(dev_id));
    ASSERTDRV(cuCtxGetCurrent(&context[dev_id]));

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    ASSERTDRV(cuMemAlloc(&d_A[dev_id], size));
    OUT << "device ptr: " << hex << d_A[dev_id] << dec << endl;

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A[dev_id]));

	ASSERTDRV(cuMemsetD32(d_A[dev_id], 0xdeadbeef, size / sizeof(unsigned int)));

        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
        ASSERT_EQ(gdr_pin_buffer(g, d_A[dev_id], size, 0, 0, &mh[dev_id]), 0);
        ASSERT_NEQ(mh[dev_id], 0U);

        bar_ptr[dev_id]  = NULL;
        ASSERT_EQ(gdr_map(g, mh[dev_id], &bar_ptr[dev_id], size), 0);
        OUT << "bar_ptr: " << bar_ptr[dev_id] << endl;

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh[dev_id], &info), 0);
        OUT << "info.va: " << hex << info.va << dec << endl;
        OUT << "info.mapped_size: " << info.mapped_size << endl;
        OUT << "info.page_size: " << info.page_size << endl;

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        int off = d_A[dev_id] - info.va;
        OUT << "page offset: " << off << endl;

        buf_ptr[dev_id] = (uint32_t *)((char *)bar_ptr[dev_id] + off);
        OUT << "user-space pointer:" << buf_ptr[dev_id] << endl;

	ASSERTDRV(cuMemAllocHost(&host_buf[dev_id], copy_size));
}

static void show_elapsed_time(struct timespec& beg, struct timespec& end, const char *prefix)
{
	double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
	OUT << prefix << " Time elapsed: " << dt_ms << "milliseconds" << endl;
}

static void *nvme_to_gpu(void *arg)
{
	int dev_id = (int)(long)arg;
	char path[] = "/mnt/0/test.bin";
	sprintf(path, "/mnt/%d/test.bin", dev_id);
	int fd = open(path, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

        for (int iter=0; iter<num_write_iters; ++iter)
	{
		ssize_t ret = pread(fd, buf_ptr[dev_id] + copy_offset/4, copy_size, 0);
		if (ret != copy_size) {
			perror("read failed");
			printf("%zd bytes read\n", ret);
			//exit(1);
			break;
		}
	}

	close(fd);

	if (compare)
		compare_buf(init_buf, buf_ptr[dev_id] + copy_offset/4, copy_size);

#if 0
	if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
		clock_gettime(MYCLOCK, &end);
		show_elapsed_time("NVMe-to-GPU");
	}
#else
	pthread_barrier_wait(&barrier);
#endif

	pthread_exit(NULL);
}

//static void nvme_to_gpu_via_host(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
static void *nvme_to_gpu_via_host(void *arg)
{
	int dev_id = (int)(long)arg;
	char path[] = "/mnt/0/test.bin";
	sprintf(path, "/mnt/%d/test.bin", dev_id);
	int fd = open(path, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

	ASSERTDRV(cuCtxSetCurrent(context[dev_id]));

        for (int iter=0; iter<num_write_iters; ++iter)
	{
		ssize_t ret = pread(fd, host_buf[dev_id], copy_size, 0);
		if (ret != copy_size) {
			perror("read failed");
			printf("%zd bytes read\n", ret);
			//exit(1);
			break;
		}
		//gdr_copy_to_bar(buf_ptr + copy_offset/4, host_buf, copy_size);
		ASSERTDRV(cuMemcpyHtoD(d_A[dev_id]+copy_offset, host_buf[dev_id], copy_size));
	}

	close(fd);

	if (compare)
		compare_buf(init_buf, buf_ptr[dev_id] + copy_offset/4, copy_size);

#if 0
	if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
		clock_gettime(MYCLOCK, &end);
		show_elapsed_time("NVMe-to-host-to-GPU");
	}
#else
	pthread_barrier_wait(&barrier);
#endif

	pthread_exit(NULL);
}

static void *gpu_to_nvme(void *arg)
{
	int dev_id = (int)(long)arg;
	char path[] = "/mnt/0/result.bin";
	sprintf(path, "/mnt/%d/result.bin", dev_id);
	int fd = open(path, O_CREAT | O_WRONLY | O_DIRECT, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

#ifdef	AIO
	io_context_t ctx;
	memset(&ctx, 0, sizeof(ctx));
	int ret = io_setup(num_read_iters, &ctx);
	if (ret < 0) {
		perror("io_setup failed");
		exit(1);
	}

	struct iocb **cbs = new struct iocb *[num_read_iters];
	struct iocb *cb = new struct iocb[num_read_iters];
#endif
        for (int iter=0; iter<num_read_iters; ++iter)
	{
#ifdef	AIO
		io_prep_pwrite(&cb[iter], fd, buf_ptr, copy_size, iter * copy_size);
		cbs[iter] = &cb[iter];
#else
		ssize_t ret = pwrite(fd, buf_ptr[dev_id] + copy_offset/4, copy_size, 0);
		if (ret != copy_size) {
			perror("write failed");
			printf("%zd bytes written\n", ret);
			exit(1);
		}
#endif
	}

#ifdef	AIO
	ret = io_submit(ctx, num_read_iters, cbs);
	if (ret < 0) {
		perror("io_submit failed");
		exit(1);
	}

	struct io_event *events = new struct io_event[num_read_iters];
	ret = io_getevents(ctx, num_read_iters, num_read_iters, events, NULL);
	if (ret < 0) {
		perror("io_getevents failed");
		exit(1);
	}

	for (int i=0; i<num_read_iters; i++) {
		if (events[i].res != copy_size)
			printf("i=%d res=%lu\n", i, events[i].res);
	}

	io_destroy(ctx);
#endif
	close(fd);

#if 0
	if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
		clock_gettime(MYCLOCK, &end);
		show_elapsed_time("GPU-to-NVMe");
	}
#else
	pthread_barrier_wait(&barrier);
#endif

	pthread_exit(NULL);
}

//static void gpu_to_nvme_via_host(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
static void *gpu_to_nvme_via_host(void *arg)
{
	int dev_id = (int)(long)arg;
	char path[] = "/mnt/0/result.bin";
	sprintf(path, "/mnt/%d/result.bin", dev_id);
	int fd = open(path, O_CREAT | O_WRONLY | O_DIRECT, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

	ASSERTDRV(cuCtxSetCurrent(context[dev_id]));

#ifdef	AIO
	io_context_t ctx;
	memset(&ctx, 0, sizeof(ctx));
	int ret = io_setup(num_read_iters, &ctx);
	if (ret < 0) {
		perror("io_setup failed");
		exit(1);
	}

	struct iocb **cbs = new struct iocb *[num_read_iters];
	struct iocb *cb = new struct iocb[num_read_iters];
#endif
        for (int iter=0; iter<num_read_iters; ++iter)
	{
#ifdef	AIO
		io_prep_pwrite(&cb[iter], fd, buf_ptr, copy_size, iter * copy_size);
		cbs[iter] = &cb[iter];
#else
		//gdr_copy_from_bar(host_buf, buf_ptr + copy_offset/4, copy_size);
		ASSERTDRV(cuMemcpyDtoH(host_buf[dev_id], d_A[dev_id]+copy_offset, copy_size));
		ssize_t ret = pwrite(fd, host_buf[dev_id], copy_size, 0);
		if (ret != copy_size) {
			perror("write failed");
			printf("%zd bytes written\n", ret);
			exit(1);
		}
#endif
	}

#ifdef	AIO
	ret = io_submit(ctx, num_read_iters, cbs);
	if (ret < 0) {
		perror("io_submit failed");
		exit(1);
	}

	struct io_event *events = new struct io_event[num_read_iters];
	ret = io_getevents(ctx, num_read_iters, num_read_iters, events, NULL);
	if (ret < 0) {
		perror("io_getevents failed");
		exit(1);
	}

	for (int i=0; i<num_read_iters; i++) {
		if (events[i].res != copy_size)
			printf("i=%d res=%lu\n", i, events[i].res);
	}

	io_destroy(ctx);
#endif

	close(fd);

#if 0
	if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
		clock_gettime(MYCLOCK, &end);
		show_elapsed_time("GPU-to-host-to-NVMe");
	}
#else
	pthread_barrier_wait(&barrier);
#endif

	pthread_exit(NULL);
}

static void gpu_memory_destroy(gdr_t g, int dev_id, size_t size)
{
	ASSERTDRV(cuCtxSetCurrent(context[dev_id]));

        OUT << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh[dev_id], bar_ptr[dev_id], size), 0);

        OUT << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh[dev_id]), 0);
	ASSERTDRV(cuMemFreeHost(host_buf[dev_id]));
	ASSERTDRV(cuMemFree(d_A[dev_id]));
}

int main(int argc, char *argv[])
{
    size_t _size = 128*1024;
    int dev_id = 0;
    int numGPUs = 1;

    while(1) {        
        int c;
        c = getopt(argc, argv, "s:k:m:g:d:o:c:w:r:hn");
        if (c == -1)
            break;

        switch (c) {
        case 's':
            _size = strtol(optarg, NULL, 0);
            break;
	case 'k':
            _size = strtol(optarg, NULL, 0);
	    _size <<= 10;
	    break;
	case 'm':
            _size = strtol(optarg, NULL, 0);
	    _size <<= 20;
	    break;
	case 'g':
            _size = strtol(optarg, NULL, 0);
	    _size <<= 30;
	    break;
        case 'c':
            copy_size = strtol(optarg, NULL, 0);
            break;
        case 'o':
            copy_offset = strtol(optarg, NULL, 0);
            break;
        case 'd':
            numGPUs = strtol(optarg, NULL, 0);
	    if (numGPUs > MAX_GPUS)
		numGPUs = MAX_GPUS;
            break;
	case 'n':
		compare = 0;
		break;
        case 'h':
            printf("syntax: %s -s <buf size> -c <copy size> -o <copy offset> -d <gpu count> -h\n", argv[0]);
            exit(EXIT_FAILURE);
            break;
	case 'w':
		num_write_iters = strtol(optarg, NULL, 0);
		break;
	case 'r':
		num_read_iters = strtol(optarg, NULL, 0);
		break;
        default:
            printf("ERROR: invalid option\n");
            exit(EXIT_FAILURE);
        }
    }
    
	if (numGPUs == 0)
		numGPUs = 1;

    if (!copy_size)
        copy_size = _size;

    if (copy_offset % sizeof(uint32_t) != 0) {
        printf("ERROR: offset must be multiple of 4 bytes\n");
        exit(EXIT_FAILURE);
    }

    if (copy_offset + copy_size > _size) {
        printf("ERROR: offset + copy size run past the end of the buffer\n");
        exit(EXIT_FAILURE);
    }

    size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

	if (!num_write_iters)
		num_write_iters = 1000;

	if (!num_read_iters)
		num_read_iters = 100;

    int n_devices = 0;
    ASSERTRT(cudaGetDeviceCount(&n_devices));

    cudaDeviceProp prop;
    for (int n=0; n<n_devices; ++n) {
        ASSERTRT(cudaGetDeviceProperties(&prop,n));
        OUT << "GPU id:" << n << " name:" << prop.name 
            << " PCI domain: " << prop.pciDomainID 
            << " bus: " << prop.pciBusID 
            << " device: " << prop.pciDeviceID << endl;
    }

    OUT << "testing size: " << _size << endl;
    OUT << "rounded size: " << size << endl;

    init_buf = (uint32_t *)malloc(size);
    ASSERT_NEQ(init_buf, (void*)0);
    //init_hbuf_walking_bit(init_buf, size);
    memset(init_buf, 0, size);

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

	ASSERT_EQ(pthread_barrier_init(&barrier, NULL, numGPUs+1), 0);
	for (dev_id = 0; dev_id < numGPUs; dev_id++) {
		gpu_memory_init(g, dev_id, size);
	}

    BEGIN_CHECK {
        // copy to BAR benchmark
        cout << "File read test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_write_iters << endl;
	struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);
	for (dev_id = 0; dev_id < numGPUs; dev_id++) {
		pthread_t thread;
		ASSERT_EQ(pthread_create(&thread, NULL, nvme_to_gpu, (void *)(long)dev_id), 0);
	}

#if 0
	if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
		clock_gettime(MYCLOCK, &end);
		show_elapsed_time("NVMe-to-GPU");
	}
#else
	pthread_barrier_wait(&barrier);
	clock_gettime(MYCLOCK, &end);
	show_elapsed_time(beg, end, "NVMe-to-GPU");
#endif

        clock_gettime(MYCLOCK, &beg);
	for (dev_id = 0; dev_id < numGPUs; dev_id++) {
		pthread_t thread;
		ASSERT_EQ(pthread_create(&thread, NULL, nvme_to_gpu_via_host, (void *)(long)dev_id), 0);
	}

#if 0
	if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
		clock_gettime(MYCLOCK, &end);
		show_elapsed_time("NVMe-to-host-to-GPU");
	}
#else
	pthread_barrier_wait(&barrier);
	clock_gettime(MYCLOCK, &end);
	show_elapsed_time(beg, end, "NVMe-to-host-to-GPU");
#endif
	
        // copy from BAR benchmark
        cout << "File write test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_read_iters << endl;
        clock_gettime(MYCLOCK, &beg);
	for (dev_id = 0; dev_id < numGPUs; dev_id++) {
		pthread_t thread;
		ASSERT_EQ(pthread_create(&thread, NULL, gpu_to_nvme, (void *)(long)dev_id), 0);
	}

#if 0
	if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
		clock_gettime(MYCLOCK, &end);
		show_elapsed_time("GPU-to-NVMe");
	}
#else
	pthread_barrier_wait(&barrier);
	clock_gettime(MYCLOCK, &end);
	show_elapsed_time(beg, end, "GPU-to-NVMe");
#endif
	
        clock_gettime(MYCLOCK, &beg);
	for (dev_id = 0; dev_id < numGPUs; dev_id++) {
		pthread_t thread;
		ASSERT_EQ(pthread_create(&thread, NULL, gpu_to_nvme_via_host, (void *)(long)dev_id), 0);
	}

#if 0
	if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
		clock_gettime(MYCLOCK, &end);
		show_elapsed_time("GPU-to-host-to-NVMe");
	}
#else
	pthread_barrier_wait(&barrier);
	clock_gettime(MYCLOCK, &end);
	show_elapsed_time(beg, end, "GPU-to-host-to-NVMe");
#endif
	
    } END_CHECK;

	for (dev_id = 0; dev_id < numGPUs; dev_id++) {
		gpu_memory_destroy(g, dev_id, size);
	}

    OUT << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

  	ASSERT_EQ(pthread_barrier_destroy(&barrier), 0);

	//pthread_exit(NULL);
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
