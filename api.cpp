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
#include <pwd.h>

// RON
// libgibraltar
#include <gibraltar.h>
#include <gib_cpu_funcs.h>
#include <cstdlib>
#include <sys/time.h>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include "api.h"
#include <vector>
// ron

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

#define OUT cout
//#define OUT TESTSTACK

//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC

double etime() {
  /* Return time since epoch (in seconds) */
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + 1.e-6*t.tv_usec;
}

#define time_iters(var, cmd, iters) {					\
    var = -1*etime();							\
    for (int iterations = 0; iterations < iters; iterations++) cmd;	\
    var = (var + etime()) / iters; }

const int NVME_SECTOR_SIZE = 512;


struct gpu_context_t {
  CUdevice dev;
  CUmodule module;
  CUcontext pCtx;
  CUfunction checksum;
  CUfunction recover_sparse;
  CUfunction recover;
  CUdeviceptr buffers;
};

typedef struct gpu_context_t * gpu_context;
void* child(void* data);


// manually tuned...
static int num_write_iters = 1;
static int num_read_iters  = 1;
// For gib, will later be replaced
size_t size = 1024*1024 * 4;
int gib_block_size = 1024*1024;
int buf_size = 1024*1024/4;
size_t small_size = 128*1024; 
int M = 4;
int K = 2;
static int which_disk = 3;
const int ORIGINAL = 0;
const int OUT_FILE = 1;
const int ENCODE = 2;
char file_name[3][30] = {"/mnt/4/test.bin","/mnt/4/result.bin","/mnt/4/result_encode.bin"};
int dev_id = 0;

struct time_consumption {
    // Init
    double init;
    // Calculation
    double encode;
    double decode;
    // Communication
    // H host, D device, S SSD
    double HtoD;
    double DtoH;
    double StoD;
    double DtoS;
    double HtoS;
    double StoH;
};

void print_title()
{
    cout << "M = " << M << "\tK = " << K << "\tSize = " << small_size << "\tEncoded Size=" << size << endl;
    cout << fixed << setprecision(6);
    cout  << "\t" << "Init" << "\t" << "SSDComm" << 
          "\t" << "Calculation" << "\t" << "HDComm" << "\t" << "Total"
          "\t" << "Throughput" << "\t" << "GibThroughput" <<endl;
}

void print_all(struct time_consumption tc)
{
  cout << "Init " << tc.init << endl;
  cout << "Encode " << tc.encode << endl;
  cout << "Decode " << tc.decode << endl;
  cout << "HtoD " << tc.HtoD << endl;
  cout << "DtoH " << tc.DtoH << endl;
  cout << "StoD " << tc.StoD << endl;
  cout << "DtoS " << tc.DtoS << endl;
  cout << "HtoS " << tc.HtoS << endl;
  cout << "StoH " << tc.StoH << endl;
}

void print_statistic(const char* title, struct time_consumption tc)
{
    double calculation = tc.encode + tc.decode;
    double hdcomm = tc.HtoD + tc.DtoH;
    double ssdcomm = tc.StoD + tc.DtoS + tc.HtoS + tc.StoH;
    double total = tc.init + calculation + hdcomm + ssdcomm;
    double Bps = size / total * 1e3;
    double woMBps = Bps / 1024.0 / 1024.0;
    double gibBps = size / (hdcomm + calculation) * 1e3;
    double gibWoMBps = gibBps / 1024.0 / 1024.0;
    cout << title << "\t" << tc.init << "\t" << ssdcomm << "\t"
         << calculation << "\t" << hdcomm << "\t" << total << "\t" 
         << woMBps << "\t" << gibWoMBps << endl;
}

struct time_consumption CPU_encode(gib_context gc, void *original, size_t copy_offset);
struct time_consumption GPU_encode(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset);
struct time_consumption GPU_gdrcopy_encode(gib_context gc,
    void *original, CUdeviceptr d_A, uint32_t *bar_ptr, size_t copy_offset);

struct time_consumption CPU_decode(gib_context gc, size_t copy_offset);
struct time_consumption GPU_decode(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset);
struct time_consumption GPU_gdrcopy_decode(gib_context gc, void *original,
    CUdeviceptr d_A, uint32_t *buf_ptr, size_t copy_offset);


static void read_file(const char* dump_file_path, uint32_t *buf_ptr, size_t read_size)
{
    int fd = open(dump_file_path, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}
    ssize_t ret = pread(fd, buf_ptr, read_size, 0);
    if (ret != read_size) {
        perror("read failed");
        printf("%zd bytes read\n", ret);
        exit(1);
    }
    close(fd);
}

static double nvme_to_cpu(const char* dump_file_path, uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
	int fd = open(dump_file_path, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

        for (int iter=0; iter<num_write_iters; ++iter)
	{
		ssize_t ret = pread(fd, buf_ptr + copy_offset/4, copy_size, 0);
		if (ret != copy_size) {
			perror("read failed");
			printf("%zd bytes read\n", ret);
			//exit(1);
			break;
		}
	}

        clock_gettime(MYCLOCK, &end);
	close(fd);

        double woMBps;
        {
            double byte_count = (double) copy_size * num_write_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            cout << "NVMe-to-CPU throughput: " << woMBps << "MB/s" << endl;
            return dt_ms;
        }
    
}

static double nvme_to_cpu(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
  return nvme_to_cpu(file_name[ORIGINAL], buf_ptr, copy_size, copy_offset);
}

static double nvme_to_gpu(const char* dump_file_path, uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
	int fd = open(dump_file_path, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror("open failed");
		//exit(1);
    return -1;
	}

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

        for (int iter=0; iter<num_write_iters; ++iter)
	{
		ssize_t ret = pread(fd, buf_ptr + copy_offset/4, copy_size, 0);
		if (ret != copy_size) {
			perror("read failed");
			printf("%zd bytes read\n", ret);
			//exit(1);
			break;
		}
	}

        clock_gettime(MYCLOCK, &end);
	close(fd);

        double woMBps;
        {
            double byte_count = (double) copy_size * num_write_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            cout << "NVMe-to-GPU throughput: " << woMBps << "MB/s" << endl;
            return dt_ms;
        }
}

static double nvme_to_gpu(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
  return nvme_to_gpu(file_name[ORIGINAL], buf_ptr, copy_size, copy_offset);
}

//static void nvme_to_gpu_via_host(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
static double nvme_to_gpu_via_host(const char* dump_file_path, CUdeviceptr d_A, size_t copy_size, size_t copy_offset)
{
	void *host_buf;
	ASSERTDRV(cuMemAllocHost(&host_buf, copy_size));
	int fd = open(dump_file_path, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

        for (int iter=0; iter<num_write_iters; ++iter)
	{
		ssize_t ret = pread(fd, host_buf, copy_size, 0);
		if (ret != copy_size) {
			perror("read failed");
			printf("%zd bytes read\n", ret);
			//exit(1);
			break;
		}
		//gdr_copy_to_bar(buf_ptr + copy_offset/4, host_buf, copy_size);
		ASSERTDRV(cuMemcpyHtoD(d_A+copy_offset, host_buf, copy_size));
	}

        clock_gettime(MYCLOCK, &end);
	close(fd);
	ASSERTDRV(cuMemFreeHost(host_buf));

        double woMBps;
        {
            double byte_count = (double) copy_size * num_write_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            cout << "NVMe-to-host-to-GPU throughput: " << woMBps << "MB/s" << endl;
            return dt_ms;
        }
}

static double nvme_to_gpu_via_host(CUdeviceptr d_A, size_t copy_size, size_t copy_offset)
{
  return nvme_to_gpu_via_host(file_name[ORIGINAL], d_A, copy_size, copy_offset);
}

static double cpu_to_nvme(const char* dump_file_path, uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
	int fd = open(dump_file_path, O_CREAT | O_WRONLY | O_DIRECT, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

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
		ssize_t ret = pwrite(fd, buf_ptr + copy_offset/4, copy_size, 0);
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
        clock_gettime(MYCLOCK, &end);
	close(fd);

        double roMBps;
        {
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            cout << "CPU-to-NVMe throughput: " << roMBps << "MB/s" << endl;
            return dt_ms;
        }
}

static double cpu_to_nvme(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
  return cpu_to_nvme(file_name[OUT_FILE], buf_ptr, copy_size, copy_offset);
}

static double gpu_to_nvme(const char* dump_file_path, uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
	int fd = open(dump_file_path, O_CREAT | O_WRONLY | O_DIRECT, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

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
		ssize_t ret = pwrite(fd, buf_ptr + copy_offset/4, copy_size, 0);
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
        clock_gettime(MYCLOCK, &end);
	close(fd);

        double roMBps;
        {
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            cout << "GPU-to-NVMe throughput: " << roMBps << "MB/s" << endl;
            return dt_ms;
        }
}

static double gpu_to_nvme(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
  return gpu_to_nvme(file_name[OUT_FILE], buf_ptr, copy_size, copy_offset);
}

//static void gpu_to_nvme_via_host(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
static double gpu_to_nvme_via_host(const char* dump_file_path, CUdeviceptr d_A, size_t copy_size, size_t copy_offset)
{
	void *host_buf;
	ASSERTDRV(cuMemAllocHost(&host_buf, copy_size));
	int fd = open(dump_file_path, O_CREAT | O_WRONLY | O_DIRECT, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

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
		ASSERTDRV(cuMemcpyDtoH(host_buf, d_A+copy_offset, copy_size));
		ssize_t ret = pwrite(fd, host_buf, copy_size, 0);
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

        clock_gettime(MYCLOCK, &end);
	close(fd);
	ASSERTDRV(cuMemFreeHost(host_buf));

        double roMBps;
        {
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            cout << "GPU-to-host-to-NVMe throughput: " << roMBps << "MB/s" << endl;
            return dt_ms;
        }
}

static double gpu_to_nvme_via_host(CUdeviceptr d_A, size_t copy_size, size_t copy_offset)
{
  gpu_to_nvme_via_host(file_name[OUT_FILE], d_A, copy_size, copy_offset);
}

/**********************************/
// GPU
/**********************************/

void dump_file_cpu(const char* dump_file_path, int *A, size_t dump_size)
{
    int fd = open(dump_file_path, O_CREAT | O_WRONLY, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}
    ssize_t ret = pwrite(fd, A, dump_size, 0);
    if (ret != dump_size) {
        perror("write failed");
        printf("%zd bytes written\n", ret);
        exit(1);
    }
}

int test_config(gib_context gc, int *fail_config, int *buf, bool use_gpu) {
  for (int i = 0; i < gc->n+gc->m; i++)
    printf("%s", (fail_config[i])?"X":".");
  printf("\n");
  /* There are n entries in fail_config, with a 1 for each buffer that should
     be destroyed, recovered, and tested.
  */
  
  int good_buffers[256]; /* n of these are needed. */
  int bad_buffers[256]; /* Up to m of these can be used */  
  int ngood = 0;
  int nbad = 0;
  for (int i = 0; i < gc->n+gc->m; i++) {
    if (fail_config[i] == 0)
      good_buffers[ngood++] = i;
    else if (i < gc->n) {
      bad_buffers[nbad++] = i;
      /* destroy the buffer contents */
      memset(buf + i*buf_size, 0, buf_size);
    }
  }
  if (ngood < gc->n) {
    printf("There are not enough good buffers.\n");
    exit(1);
  }
  
  /* Reshuffle to prevent extraneous memory copies */
  for (int i = 0; i < ngood; i++) {
    if (good_buffers[i] != i && good_buffers[i] < gc->n) {
      int j = i+1;
      while(good_buffers[j] < gc->n) 
	j++;
      int tmp = good_buffers[j];
      memmove(good_buffers+i+1, good_buffers+i, 
	      sizeof(int)*(j-i));
      good_buffers[i] = tmp;
    }
  }
  /* Sanity check */
  for (int i = 0; i < gc->n; i++) {
    if (good_buffers[i] != i && good_buffers[i] < gc->n) {
      printf("Didn't work...\n");
      exit(1);
    }
  }
  
  for (int i = 0; i < gc->n; i++) {
    if (good_buffers[i] != i) {
      memcpy(buf + buf_size*i, buf + buf_size*good_buffers[i], 
           buf_size*sizeof(int));
    }
  }
  
  int buf_ids[256];
  memcpy(buf_ids, good_buffers, gc->n*sizeof(int));
  memcpy(buf_ids+gc->n, bad_buffers, nbad*sizeof(int));
  if(use_gpu){
    gib_recover(buf, buf_size*sizeof(int), buf_ids, nbad, gc);
  }
  else{
    gib_cpu_recover(buf, buf_size*sizeof(int), buf_ids, nbad, gc);
  }
  

  void *tmp_buf = malloc(sizeof(int)*buf_size);
  for (int i = 0; i < gc->n; i++) {
    if (buf_ids[i] != i) {
      int j;
      for (j = i+1; buf_ids[j] != i; j++)
	;
      memcpy(buf + buf_size*i, buf + buf_size*j, buf_size*sizeof(int));
      buf_ids[i] = i;
    }
  }
  free(tmp_buf);

  return 0;
}

int test_config(gib_context gc, int *fail_config, int *buf){
  return test_config(gc, fail_config, buf, true);
}

void set_fail_config(int n, int m, int *fail_config)
{
    int random_pick[n+m] = {0};
    int probe;
    srand(time(NULL));
    for (int i = 0; i < m; i++)
    {
        do {
	        probe = rand() % (n+m);
	    } while (random_pick[probe] == 1);
        random_pick[probe] = 1;
    }
    for (int i = 0; i < n+m; i++){
        if(random_pick[i] == 1){
            fail_config[i] = 1;
        }
        else{
            fail_config[i] = 0;
        }
    }    
}

void gib_wrapper(int n, int m, int *data)
{
    int *buf;
    int size_sc; /* scratch */
    int *backup_buf = (int *)malloc((n+m)*buf_size*sizeof(int));
    gib_context gc;
    int rc = gib_init(n, m, &gc);
    gib_alloc((void **)(&buf), buf_size*sizeof(int), &size_sc, gc);
    if (rc) {
	    printf("Error:  %i\n", rc);
	    exit(EXIT_FAILURE);
    }
    memcpy(buf, data, n*buf_size*sizeof(int));

    memcpy(backup_buf, buf, n*buf_size*sizeof(int));
    gib_generate(buf, buf_size*sizeof(int), gc);
    if (memcmp(buf, backup_buf, n*buf_size*sizeof(int))) {
	    printf("Generation failed.\n");
	    exit(1);
    }

    int *fail_config = (int *)malloc(sizeof(int)*(n+m));
    for (int i = 0; i < n+m; i++)
	    fail_config[i] = 0;

    set_fail_config(n, m, fail_config);
    test_config(gc, fail_config, buf);
    /*
    if (memcmp(buf, backup_buf, n*buf_size*sizeof(int))) {
        printf("Recovery failed.\n");
        exit(1);
    }
    */
/* 
    int fd = open("/mnt/4/recover_data.bin", O_CREAT | O_WRONLY | O_DIRECT, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}
    ssize_t ret = pwrite(fd, buf,  n*buf_size*sizeof(int), 0);
    if (ret !=  n*buf_size*sizeof(int)) {
        perror("write failed");
        printf("%zd bytes written\n", ret);
        exit(1);
    }
    */
}

/**********************************/
// GPU NVMe
/**********************************/

void dump_file(const char* dump_file_path, uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
{
    int fd = open(dump_file_path, O_CREAT | O_WRONLY, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}
    ssize_t ret = pwrite(fd, buf_ptr + copy_offset/4, copy_size, 0);
    if (ret != copy_size) {
        perror("write failed");
        printf("%zd bytes written\n", ret);
        exit(1);
    }
    close(fd);
}

void dump_file(const char* dump_file_path, CUdeviceptr d_A, size_t dump_size)
{
    int *A = (int *)malloc(dump_size);
    cuMemcpyDtoH(A, d_A, dump_size);
    int fd = open(dump_file_path, O_CREAT | O_WRONLY, 0777);
	if (fd < 0) {
		perror("open failed");
		exit(1);
	}
    ssize_t ret = pwrite(fd, A, dump_size, 0);
    if (ret != dump_size) {
        perror("write failed");
        printf("%zd bytes written\n", ret);
        exit(1);
    }
    close(fd);
}

int test_config_nvme(gib_context gc, int *fail_config, int *buf, CUdeviceptr d_buf) {
  for (int i = 0; i < gc->n+gc->m; i++)
    printf("%s", (fail_config[i])?"X":".");
  printf("\n");
  /* There are n entries in fail_config, with a 1 for each buffer that should
     be destroyed, recovered, and tested.
  */
  
  int good_buffers[256]; /* n of these are needed. */
  int bad_buffers[256]; /* Up to m of these can be used */  
  int ngood = 0;
  int nbad = 0;
  for (int i = 0; i < gc->n+gc->m; i++) {
    if (fail_config[i] == 0)
      good_buffers[ngood++] = i;
    else if (i < gc->n) {
      bad_buffers[nbad++] = i;
      /* destroy the buffer contents */
      cuMemsetD8(d_buf + i*buf_size*4, 0, buf_size);
    }
  }
  if (ngood < gc->n) {
    printf("There are not enough good buffers.\n");
    exit(1);
  }
  
  /* Reshuffle to prevent extraneous memory copies */
  for (int i = 0; i < ngood; i++) {
    if (good_buffers[i] != i && good_buffers[i] < gc->n) {
      int j = i+1;
      while(good_buffers[j] < gc->n) 
	j++;
      int tmp = good_buffers[j];
      memmove(good_buffers+i+1, good_buffers+i, 
	      sizeof(int)*(j-i));
      good_buffers[i] = tmp;
    }
  }
  /* Sanity check */
  for (int i = 0; i < gc->n; i++) {
    if (good_buffers[i] != i && good_buffers[i] < gc->n) {
      printf("Didn't work...\n");
      exit(1);
    }
  }
  
  for (int i = 0; i < gc->n; i++) {
    if (good_buffers[i] != i) {
      cuMemcpyDtoD(d_buf + buf_size*i*4, d_buf + buf_size*good_buffers[i]*4, buf_size*sizeof(int));
    }
  }
  
  int buf_ids[256];
  memcpy(buf_ids, good_buffers, gc->n*sizeof(int));
  memcpy(buf_ids+gc->n, bad_buffers, nbad*sizeof(int));
  cudaDeviceSynchronize();
  gib_recover_nvme(buf, buf_size*sizeof(int), buf_ids, nbad, gc);
  
  for (int i = 0; i < gc->n; i++) {
    if (buf_ids[i] != i) {
      int j;
      for (j = i+1; buf_ids[j] != i; j++)
	;
      cuMemcpyDtoD(d_buf + buf_size*i*4, d_buf + buf_size*j*4, buf_size*sizeof(int));

      buf_ids[i] = i;
    }
  }
  return 0;
}

bool cuda_cmp(void *A, CUdeviceptr d_B, size_t cmp_size)
{
    int *B = (int *)malloc(cmp_size);
    cuMemcpyDtoH(B, d_B, cmp_size);
    int return_value = memcmp(A, B, cmp_size);
    printf("Return value: %d\n",return_value);
    free(B);
    return return_value;
}

bool cuda_cmp(CUdeviceptr d_A, CUdeviceptr d_B, size_t cmp_size)
{
    int *A = (int *)malloc(cmp_size);
    cuMemcpyDtoH(A, d_A, cmp_size);
    int *B = (int *)malloc(cmp_size);
    cuMemcpyDtoH(B, d_B, cmp_size);
    int return_value = memcmp(A, B, cmp_size);
    printf("Return value: %d\n",return_value);
    free(A);
    free(B);
    return return_value;
}

void gib_wrapper_nvme(int n, int m, int *data, CUdeviceptr d_A)
{
    int *buf = (int *)malloc(size);
    cuMemcpyDtoH(buf, d_A, size);
    int size_sc; /* scratch */

    CUdeviceptr d_backup_buf;
    ASSERTDRV(cuMemAlloc(&d_backup_buf, small_size));

    gib_context gc;
    int rc = gib_init_nvme (d_A, n, m, &gc, gib_block_size, dev_id);
    if (rc) {
	    printf("Error:  %i\n", rc);
	    exit(EXIT_FAILURE);
    }

    cuMemcpyDtoD(d_backup_buf, d_A, small_size);

    gib_generate_nvme(buf, gib_block_size, gc);
    
    if(cuda_cmp(d_A, d_backup_buf, small_size)){
        printf("Generation failed.\n");
	    exit(1);
    }

    int *fail_config = (int *)malloc(sizeof(int)*(n+m));
    for (int i = 0; i < n+m; i++)
	    fail_config[i] = 0;

    set_fail_config(n, m, fail_config);
    test_config_nvme(gc, fail_config, buf, d_A);
    /* 
    if(cuda_cmp(d_A, d_backup_buf, small_size)){
        printf("Recovery failed.\n");
	    exit(1);
    }
    */
    cuMemFree(d_backup_buf);
    free(buf);
}

size_t roundSize(size_t size, int m)
{
	int mSectorSize = m * NVME_SECTOR_SIZE;
	int howManySector = size / mSectorSize;
	int needPlusOne = ((size % mSectorSize) != 0);
	return mSectorSize * (howManySector + needPlusOne);
}

size_t divRoundUp(size_t size, int m)
{
	int left = size % m;
	return size / m + (left != 0);
}

GE_config GE_init(int dev_id, int buffer_size, char* path){
struct GE_config config;  
BEGIN_CHECK{
	int n_devices = 0;
	ASSERTRT(cudaGetDeviceCount(&n_devices));

	cudaDeviceProp prop;
	for (int n = 0; n < n_devices; ++n) {
		ASSERTRT(cudaGetDeviceProperties(&prop,n));
		OUT << "GPU id:" << n << " name:" << prop.name
			<< " PCI domain: " << prop.pciDomainID
			<< " bus: " << prop.pciBusID
			<< " device: " << prop.pciDeviceID << endl;
	}
	OUT << "selecting device " << dev_id << endl;
	ASSERTRT(cudaSetDevice(dev_id));

	void* dummy;
	ASSERTRT(cudaMalloc(&dummy, 0));

	OUT << "testing size: " << buffer_size << endl;

	ASSERTRT(cudaSetDevice(dev_id));
	CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, buffer_size));
    cudaDeviceSynchronize();
	
	unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));
    ASSERTDRV(cuMemsetD32(d_A, 0xdeadbeef, buffer_size / sizeof(unsigned int)));

	uint32_t *init_buf = NULL;
    init_buf = (uint32_t *)malloc(buffer_size);
    ASSERT_NEQ(init_buf, (void*)0);
	memset(init_buf, 0, buffer_size);
	
	gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

	gdr_mh_t mh;
	BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, buffer_size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, 0U);

	void *bar_ptr  = NULL;
    ASSERT_EQ(gdr_map(g, mh, &bar_ptr, buffer_size), 0);

	gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
	OUT << "info.va: " << hex << info.va << dec << endl;
	OUT << "info.mapped_size: " << info.mapped_size << endl;
	OUT << "info.page_size: " << info.page_size << endl;

	int off = d_A - info.va;
    OUT << "page offset: " << off << endl;

    uint32_t *buf_ptr = (uint32_t *)((char *)bar_ptr + off);
    OUT << "user-space pointer:" << buf_ptr << endl;	
	
    config.dev_id = dev_id;
    config.buffer_size = buffer_size;
    config.d_buffer = d_A;
    config.buf_ptr = buf_ptr;
    config.path = path;
    
 } END_CHECK;
	return config;
}

GE_file GE_read_metadata(GE_config config, char* filename){
  FILE *pFile;
  char configFileName[20] = ".metadata";
  char* configFullPath = new char[strlen(config.path)+strlen(filename)+strlen(configFileName)+1];
  strcpy(configFullPath, config.path);
  strcat(configFullPath, filename);
  strcat(configFullPath, configFileName);
  pFile = fopen(configFullPath,"r" );
  struct GE_file file = {};
  if( NULL == pFile ){
    printf( "open failure" );
    return file;
  }

  int m, k, size, block_size;
  char name[50];
  int char_read = 0;
  char_read = fscanf(pFile, "%d\n%d\n%d\n%d\n%s\n", &m, &k, &size, &block_size, name);
  
  file.filename = new char[strlen(name)];
  file.m = m;
  file.k = k;
  file.size = size;
  file.block_size = block_size;
  strcpy(file.filename, name);
  file.dir_path = (char **)malloc((m+k)*sizeof(char*));
  for(int i = 0; i<m+k;i++){
    char dir_path[50] = {0};
	  char_read = fscanf(pFile, "%s", dir_path);
    file.dir_path[i] = (char *)malloc(strlen(dir_path)+1);
    strcpy(file.dir_path[i], dir_path);
  }
  return file;
}

int GE_write_metadata(GE_config config, GE_file file){
	// write file metadata to config.path
	FILE *pFile;
	char configFileName[50] = ".metadata";
	char* configFullPath = new char[strlen(config.path)+strlen(file.filename)+strlen(configFileName)];
	strcpy(configFullPath, config.path);
	strcat(configFullPath, file.filename);
	strcat(configFullPath, configFileName);
	printf("%s\n", configFullPath);
	pFile = fopen(configFullPath,"w" );
	if( NULL == pFile ){
		printf( "open failure\n" );
		return 1;
	}else{
		printf("Metadata write success\n");
		fprintf(pFile, "%d\n%d\n%lu\n%lu\n%s\n", file.m, file.k, file.size, file.block_size, file.filename);
		for(int i = 0; i< file.m+ file.k ; i++){
			fprintf(pFile, "%s\n", file.dir_path[i]);
		}
	}
	fclose(pFile);
	return 0;
}

int GE_create(GE_config config, int m, int k, char* filename, char** dir_path){
	struct GE_file file = {m, k, 0, 0, filename, dir_path};
	return GE_write_metadata(config, file);
}

GE_file GE_open(GE_config config, char* filename){
	// use the filename to get info
	//struct GE_file file = {m, k, filename, dir_path, size, block_size};
	struct GE_file file;
	file = GE_read_metadata(config, filename);
  gib_context gc;
  int rc = gib_init_nvme (config.d_buffer, file.m, file.k, &gc, config.buffer_size, config.dev_id);
  if (rc) {
    printf("Error:  %i\n", rc);
    exit(EXIT_FAILURE);
  }
  file.gc = gc;

  int totalDisks = file.m+file.k;
  pthread_t t[totalDisks];
  for(int i = 0 ;i<totalDisks;i++){
    pthread_create(&t[i], NULL, child, NULL);
  }
	return file;
}

struct GE_thread {
  int tid;
  GE_config* config;
  GE_file* file;
  int failed;
};

enum TaskType {READ, WRITE};

struct GE_task {
  int tid;
  GE_config* config;
  GE_file* file;
  TaskType task_type;
};

std::vector<GE_task> task_queue;
volatile int result;
int *failed;
pthread_mutex_t qmutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t rmutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_var = PTHREAD_COND_INITIALIZER;

/*
void *GE_read_internal(void *arg){
  GE_thread* ge_thread = (GE_thread *)arg;
  int tid = ge_thread->tid;
  printf("thread %d\n", tid);
  GE_config* config = ge_thread->config;
  GE_file* file = ge_thread->file;
  char* blockFullPath = new char[strlen(file->dir_path[tid])+strlen(file->filename)+1];
  strcpy(blockFullPath, file->dir_path[tid]);
  strcat(blockFullPath, file->filename);
  int retVal = nvme_to_gpu(blockFullPath, config->buf_ptr, file->block_size, tid * file->block_size);
  ge_thread->failed = (retVal == -1) ? 1 : 0;
}
*/

int GE_read_internal(int tid, GE_config* config, GE_file* file){
  char* blockFullPath = new char[strlen(file->dir_path[tid])+strlen(file->filename)+1];
  strcpy(blockFullPath, file->dir_path[tid]);
  strcat(blockFullPath, file->filename);
  int retVal = nvme_to_gpu(blockFullPath, config->buf_ptr, file->block_size, tid * file->block_size);
  return (retVal == -1) ? 1 : 0;
}

int GE_write_internal(int tid, GE_config* config, GE_file* file){
  char* blockFullPath = new char[strlen(file->dir_path[tid])+strlen(file->filename)+1];
  strcpy(blockFullPath, file->dir_path[tid]);
  strcat(blockFullPath, file->filename);
  int retVal = gpu_to_nvme(blockFullPath, config->buf_ptr, file->block_size, tid * file->block_size);
  return (retVal == -1) ? 1 : 0;
}

void* child(void* data) {
  int getJob = 0;
  GE_task task = {};
  while(1){
    pthread_mutex_lock( &qmutex );
    while(task_queue.size() == 0){
       pthread_cond_wait (&cond_var, &qmutex);
    } 
    if(task_queue.size() > 0){
      task = task_queue.back();
      task_queue.pop_back();
      getJob = 1;
    }
    pthread_mutex_unlock( &qmutex );
    if(getJob){
      int op_failed = 0;
      if(task.task_type == READ){
        // read fron disk
        op_failed = GE_read_internal(task.tid, task.config, task.file);
      }
      else if (task.task_type == WRITE){
        op_failed = GE_write_internal(task.tid, task.config, task.file);
      }

      // write to result
      pthread_mutex_lock( &rmutex );
      result += 1;
      failed[task.tid] = op_failed;
      pthread_mutex_unlock( &rmutex );
    }
  }
}

int GE_pread(GE_config config, GE_file* file, void* buffer){
  // for each path, read block_size data
	// combine blocks
	// trim combined data to size
  int failDisks = 0;
  int totalDisks = file->m + file->k;
  int *fail_config = (int *)malloc(sizeof(int)*(file->m + file->k));

  result = 0;
  failed = (int *)malloc(sizeof(int)*(file->m + file->k));

  pthread_mutex_lock( &qmutex );
  for(int i = 0;i < totalDisks;++i) {
    failed[i] = 0;
    // fill in task
    GE_task task = {i, &config, file, READ};
    task_queue.push_back(task);
  } 
  pthread_cond_broadcast(&cond_var);
  pthread_mutex_unlock( &qmutex );
  while(result < totalDisks);
  cudaDeviceSynchronize();
  // check if all data success
  for(int i = 0 ;i<totalDisks;i++){
    failDisks += failed[i];
    fail_config[i] = failed[i];
  }
  //delete failed;
  /*
  struct GE_thread thread_item[totalDisks];
  pthread_t t[totalDisks];
  for(int i = 0 ;i<totalDisks;i++){
    thread_item[i].tid = i;
    thread_item[i].config = &config;
    thread_item[i].file = file;
  }

  for(int i = 0 ;i<totalDisks;i++){
    pthread_create(&t[i], NULL, GE_read_internal, (void*) &thread_item[i]);
  }
  for(int i = 0 ;i<totalDisks;i++){
    pthread_join(t[i], NULL);
    fail_config[i] = thread_item[i].failed;
    failDisks += (thread_item[i].failed > 0);
  }
  */

  if(failDisks == 0) {
    cuMemcpyDtoH(buffer, config.d_buffer, file->size);
    return true;
  }
  
  // any data disks fail
  int buf_ids[file->m + file->k];
  int repair_idx = 0;
  int parity_idx = 0;
  for(int i = 0; i < file->m; i++){
    if(fail_config[i] == 0){
      buf_ids[i] = i;
    }
    else if(fail_config[i] == 1){
      while(fail_config[file->m + parity_idx] == 1) parity_idx++;
      buf_ids[file->m + repair_idx] = i;
      repair_idx++;
      buf_ids[i] = file->m + parity_idx;
      parity_idx++;
    }
  }

  for (int i = 0; i < file->gc->n; i++) {
    if (buf_ids[i] != i) {
      int j;
      for (j = i+1; buf_ids[j] != i; j++);
      cuMemcpyDtoD(config.d_buffer + file->block_size*i, config.d_buffer + file->block_size*j, file->block_size);
      cudaDeviceSynchronize();
    }
  }

  int *not_in_use;
  gib_recover_nvme(not_in_use, file->block_size, buf_ids, failDisks, file->gc);
  cudaDeviceSynchronize();

  for (int i = 0; i < file->gc->n; i++) {
    if (buf_ids[i] != i) {
      int j;
      for (j = i+1; buf_ids[j] != i; j++);
      cuMemcpyDtoD(config.d_buffer + file->block_size*i, config.d_buffer + file->block_size*j, file->block_size);
      buf_ids[i] = i;
    }
  }

  cuMemcpyDtoH(buffer, config.d_buffer, file->size);
  return true;
}

int GE_read(GE_config config, GE_file* file, void* buffer){
  // for each path, read block_size data
	// combine blocks
	// trim combined data to size
  int failDisks = 0;
  int *fail_config = (int *)malloc(sizeof(int)*(file->m + file->k));
  for(int i = 0; i < file->m + file->k; i++){
    if(i >= file->m+ file->k) return false;
    char* blockFullPath = new char[strlen(file->dir_path[i])+strlen(file->filename)+1];
    strcpy(blockFullPath, file->dir_path[i]);
    strcat(blockFullPath, file->filename);

    int return_val = nvme_to_gpu(blockFullPath, config.buf_ptr, file->block_size, i * file->block_size);

    // if fail, failDisks += 1
    if(return_val == -1){
      failDisks ++;
      fail_config[i] = 1;
    }
    else{
      fail_config[i] = 0;
    }
  }
  //cout << "failed Disks: " << failDisks << endl;
  if(failDisks == 0) {
    cuMemcpyDtoH(buffer, config.d_buffer, file->size);
    return true;
  }

  // any data disks fail
  int buf_ids[file->m + file->k];
  int repair_idx = 0;
  int parity_idx = 0;
  for(int i = 0; i < file->m; i++){
    if(fail_config[i] == 0){
      buf_ids[i] = i;
    }
    else if(fail_config[i] == 1){
      while(fail_config[file->m + parity_idx] == 1) parity_idx++;
      buf_ids[file->m + repair_idx] = i;
      repair_idx++;
      buf_ids[i] = file->m + parity_idx;
      parity_idx++;
    }
  }

  for (int i = 0; i < file->gc->n; i++) {
    if (buf_ids[i] != i) {
      int j;
      for (j = i+1; buf_ids[j] != i; j++);
      CUdeviceptr from = config.d_buffer+file->block_size*j;
      CUdeviceptr dest = config.d_buffer+file->block_size*i;
      cuMemcpyDtoD(dest, from, file->block_size);
      cudaDeviceSynchronize();
    }
  }

  int *not_in_use;
  gib_recover_nvme(not_in_use, file->block_size, buf_ids, failDisks, file->gc);
  cudaDeviceSynchronize();
  for (int i = 0; i < file->gc->n; i++) {
    if (buf_ids[i] != i) {
      int j;
      for (j = i+1; buf_ids[j] != i; j++);
      CUdeviceptr from = config.d_buffer+file->block_size*j;
      CUdeviceptr dest = config.d_buffer+file->block_size*i;
      cuMemcpyDtoD(dest, from, file->block_size);
      buf_ids[i] = i;
    }
  }

  cuMemcpyDtoH(buffer, config.d_buffer, file->size);
  return true;
}
/*
void *GE_write_internal(void *arg){
  GE_thread* ge_thread = (GE_thread *)arg;
  int tid = ge_thread->tid;
  printf("thread %d\n", tid);
  GE_config* config = ge_thread->config;
  GE_file* file = ge_thread->file;
  char* blockFullPath = new char[strlen(file->dir_path[tid])+strlen(file->filename)+1];
  strcpy(blockFullPath, file->dir_path[tid]);
  strcat(blockFullPath, file->filename);
  gpu_to_nvme(blockFullPath, config->buf_ptr, file->block_size, tid * file->block_size);
}*/

int GE_pwrite(GE_config config, GE_file* file, void* buffer, size_t size){
	size_t roundedSize = roundSize(size, file->m);
	size_t block_size = roundedSize / file->m;
	file->size = size;
	file->block_size = block_size;
  GE_write_metadata(config, *file);

  cuMemcpyHtoD(config.d_buffer, buffer, file->size);
  cudaDeviceSynchronize();

  int *not_in_use;
  gib_generate_nvme(not_in_use, file->block_size, file->gc);
  cudaDeviceSynchronize();

  int totalDisks = file->m + file->k;
  /*
  pthread_t t[totalDisks];
  struct GE_thread thread_item[totalDisks];
  for(int i = 0 ;i<totalDisks;i++){
    thread_item[i].tid = i;
    thread_item[i].config = &config;
    thread_item[i].file = file;
  }*/
  result = 0;
  failed = (int *)malloc(sizeof(int)*(file->m + file->k));
  pthread_mutex_lock( &qmutex );
  for(int i = 0;i < totalDisks;++i) {
    failed[i] = 0;
    // fill in task
    GE_task task = {i, &config, file, WRITE};
    task_queue.push_back(task);
  }  
  pthread_cond_broadcast(&cond_var);
  pthread_mutex_unlock( &qmutex );
  while(result < totalDisks);
  cudaDeviceSynchronize();
  
  /*
  for(int i = 0 ;i<totalDisks;i++){
    pthread_create(&t[i], NULL, GE_write_internal, (void*) &thread_item[i]);
  }
  for(int i = 0 ;i<totalDisks;i++){
    pthread_join(t[i], NULL);
  }*/
	return true;
}

int GE_write(GE_config config, GE_file* file, void* buffer, size_t size){
	size_t roundedSize = roundSize(size, file->m);
	size_t block_size = roundedSize / file->m;
  printf("size %d block size %d \n",size, block_size);
	file->size = roundedSize;
	file->block_size = block_size;
  GE_write_metadata(config, *file);

  cuMemcpyHtoD(config.d_buffer, buffer, file->size);
	cudaDeviceSynchronize();

  int *not_in_use;
  gib_generate_nvme(not_in_use, file->block_size, file->gc);
  cudaDeviceSynchronize();

  for(int i = 0; i < file->m + file->k ; i++){
		char* blockFullPath = new char[strlen(file->dir_path[i])+strlen(file->filename)+1];
		strcpy(blockFullPath, file->dir_path[i]);
		strcat(blockFullPath, file->filename);
		gpu_to_nvme(blockFullPath, config.buf_ptr, file->block_size, i * file->block_size);
		// if fail, failDisks += 1
	}
	return true;
}

int GE_close(GE_file file){
}

/*
for(int i = 0; i<ge_file.size*3/2/4;i++)
    {
      printf("%x ", buffer[i]);
      if((i+1)%(ge_file.block_size/4) == 0){
        printf("\n========\n");
      }
    }
*/
/*
main(int argc, char *argv[])
{
	size_t _size = 128 * 1024;
	int dev_id = 0;
	int m = 4;
	int k = 2;
  char configPath[50] = "/home/jerrychou/GE/";
  char targetFilename[50] = "api_file";
  char userFilename[50] = "user_file";
  char* dir_path[50] = { "/mnt/1/","/mnt/2/","/mnt/7/","/mnt/4/","/mnt/5/","/mnt/6/"};

  // Actions
  int createFile = 0;
  int readFile = 0;
  int writeFile = 0;

	while (1) {
		int c;
		c = getopt(argc, argv, "s:k:m:g:d:o:c:C:R:W:r:p:f:M:K:hn");
		if (c == -1)
			break;
		switch (c) {
		case 's':
			_size = strtol(optarg, NULL, 0);
			break;
		case 'd':
			dev_id = strtol(optarg, NULL, 0);
			break;
		case 'm':
			m = strtol(optarg, NULL, 0);
			break;
		case 'k':
			k = strtol(optarg, NULL, 0);
			break;
    case 'f':
      strcpy(userFilename, optarg);
      break;
    case 'c':
      strcpy(configPath, optarg);
      break;


    case 'C':
      createFile = 1;
      strcpy(targetFilename, optarg);      
      break;
    case 'W':
      writeFile = 1;
      strcpy(targetFilename, optarg);
      break;
    case 'R':
      readFile = 1;
      strcpy(targetFilename, optarg);
      break;
		}
	}
	
	struct GE_config config = GE_init(dev_id, _size, configPath);
	
  if(createFile){
    int create = -1;
  
    create = GE_create(config, m, k, targetFilename, dir_path);
    if (create != 0) {
      printf("Create error.\n");
      exit(0);
    }
    return 0;
  }

  if(writeFile){
    FILE* uFile = fopen(userFilename, "r");
    if( NULL == uFile ){
      printf( "open failure" );
      exit(0);
    }
    struct stat st;
    stat(userFilename, &st);
    size_t uFileSize = st.st_size;
    int* buffer = new int[uFileSize/sizeof(int)];
    read(fileno(uFile), (void *)buffer, uFileSize);

    GE_file ge_file = GE_open(config, targetFilename);
    GE_write(config, &ge_file, (void*)buffer);
    return 0;
  }

	if(readFile){
    GE_file ge_file = GE_open(config, targetFilename);
    int* buffer;
    GE_read(config, &ge_file, buffer);
    buffer = new int[ge_file.size/4];
    memset (buffer, 0, ge_file.size);
    cuMemcpyDtoH(buffer, config.d_buffer, ge_file.size);
    
    for(int i = 0; i<ge_file.size/4;i++)
    {
      printf("%x ", buffer[i]);
    }

    return 0;
  }

  return 0;
}
*/