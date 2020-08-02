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

// RON
// libgibraltar
#include <gibraltar.h>
#include <gib_cpu_funcs.h>
#include <cstdlib>
#include <sys/time.h>
#include <cstring>
#include <cstdio>
#include <iomanip>
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

// manually tuned...
static int num_write_iters = 10000;
static int num_read_iters  = 100;
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

    double time[10];
};

void print_title()
{
    cout << "M = " << M << "\tK = " << K << "\tSize = " << small_size << "\tEncoded Size=" << size << endl;
    cout << fixed << setprecision(6);
    cout  << "\t" << "Init" << "\t" << "SSDComm" << 
          "\t" << "Calculation" << "\t" << "HDComm" << "\t" << "Total"
          "\t" << "Throughput" << "\t" << "GibThroughput" <<endl;
}

void print_title2()
{
    cout << "M = " << M << "\tK = " << K << "\tSize = " << small_size << "\tEncoded Size=" << size << endl;
    cout << fixed << setprecision(6);
    cout  << "\t" << "Init" << "\t" << "H2D" << "\t" << "D2H" <<
          "\t" << "S2D" << "\t" << "D2S" << "\t" << "H2S" << "\t" << "S2H" <<
          "\t" << "Encode" << "\t" << "Decode" <<
          "\t" << "Total" << "\t" << "Throughput" << "\t" << "GibThroughput" << endl;
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

void print_statistic2(const char* title, struct time_consumption tc)
{
    double calculation = tc.encode + tc.decode;
    double hdcomm = tc.HtoD + tc.DtoH;
    double ssdcomm = tc.StoD + tc.DtoS + tc.HtoS + tc.StoH;
    double total = tc.init + calculation + hdcomm + ssdcomm;
    double Bps = size / total * 1e3;
    double woMBps = Bps / 1024.0 / 1024.0;
    double gibBps = size / (hdcomm + calculation) * 1e3;
    double gibWoMBps = gibBps / 1024.0 / 1024.0;
    cout << title << "\t" << tc.init << "\t" << tc.HtoD << "\t" << tc.DtoH << "\t" 
         << tc.StoD << "\t" << tc.DtoS << "\t" << tc.HtoS << "\t" << tc.StoH << "\t" 
         << tc.encode << "\t" << tc.decode << "\t" 
         << total << "\t" << woMBps << "\t" << gibWoMBps << "\t" << endl;
}

int RUN_ITER = 5;
void print_statistic3(const char* title, struct time_consumption tc)
{
    cout << title << "\t";
    double avg = 0;
    for(int i=0;i<RUN_ITER;i++){
        cout << tc.time[i] << "\t";
        avg += tc.time[i];
    }
    cout << avg/RUN_ITER << endl;
}

struct time_consumption CPU_encode(gib_context gc, void *original, size_t copy_offset);
struct time_consumption GPU_encode(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset);
struct time_consumption GPU_gdrcopy_encode(gib_context gc,
    void *original, CUdeviceptr d_A, uint32_t *bar_ptr, size_t copy_offset);

struct time_consumption CPU_decode(gib_context gc, size_t copy_offset);
struct time_consumption GPU_decode(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset);
struct time_consumption GPU_gdrcopy_decode(gib_context gc, void *original,
    CUdeviceptr d_A, uint32_t *buf_ptr, size_t copy_offset);

struct time_consumption Get_Cce(gib_context gc, void *original, size_t copy_offset);
struct time_consumption Get_Ccd(gib_context gc, size_t copy_offset);
struct time_consumption Get_Dhs(gib_context gc, void *original, size_t copy_offset);
struct time_consumption Get_Cge(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset);
struct time_consumption Get_Cgd(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset);
struct time_consumption Get_Dhg(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset);
struct time_consumption Get_Dgh(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset);
struct time_consumption Get_Dgs(gib_context gc,
    void *original, CUdeviceptr d_A, uint32_t *buf_ptr, size_t copy_offset);


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
		cout << iter <<endl;
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
		exit(1);
	}

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

        for (int iter=0; iter<num_write_iters; ++iter)
	{
		cout << iter <<endl;
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
      for (j = i+1; buf_ids[j] != i; j++);
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

main(int argc, char *argv[])
{
    size_t _size = 128*1024;
    // For size encoding
    size_t copy_size = 0;
    size_t copy_offset = 0;
    int compare = 1;

    while(1) {        
        int c;
        c = getopt(argc, argv, "s:k:m:g:d:o:c:w:r:p:M:K:hn");
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
            dev_id = strtol(optarg, NULL, 0);
            break;
	case 'n':
		compare = 0;
		break;
        case 'h':
            printf("syntax: %s -s <buf size> -c <copy size> -o <copy offset> -d <gpu dev id> -h\n", argv[0]);
            exit(EXIT_FAILURE);
            break;
	case 'w':
		num_write_iters = strtol(optarg, NULL, 0);
		break;
	case 'r':
		num_read_iters = strtol(optarg, NULL, 0);
		break;
    case 'p':
        which_disk = strtol(optarg, NULL, 0);
        for(int i=0;i<3;i++){
            file_name[i][5] = '0' + which_disk;
        }    
        break;
    case 'M':
        M = strtol(optarg, NULL, 0);
        break;
    case 'K':
        K = strtol(optarg, NULL, 0);
        break;
        default:
            printf("ERROR: invalid option\n");
            exit(EXIT_FAILURE);
        }
    }

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

    size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    size_t BinKB = 1024;
    cout << "original " << size/BinKB << endl;  
    // remainder in KB
    int remainder = ((size / BinKB) %M > 0);
    size_t each_block = (size / BinKB) / M + remainder;
    size = each_block * M * BinKB;
    cout << "added " << size/BinKB << endl;
    // we have the padded original file size now.

    // n = m + k;
    small_size = size;
    size = size / M * (M+K);
    gib_block_size = small_size/M;
    buf_size = gib_block_size/4;


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
    OUT << "selecting device " << dev_id << endl;
    ASSERTRT(cudaSetDevice(dev_id));

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    OUT << "testing size: " << _size << endl;
    OUT << "rounded size: " << size << endl;

    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    double d_A_init_time = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    OUT << "device ptr: " << hex << d_A << dec << endl;

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

	ASSERTDRV(cuMemsetD32(d_A, 0xdeadbeef, size / sizeof(unsigned int)));

    uint32_t *init_buf = NULL;
    init_buf = (uint32_t *)malloc(size);
    ASSERT_NEQ(init_buf, (void*)0);
    //init_hbuf_walking_bit(init_buf, size);
    memset(init_buf, 0, size);

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
        BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, 0U);

        void *bar_ptr  = NULL;
        ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);
        OUT << "bar_ptr: " << bar_ptr << endl;

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        OUT << "info.va: " << hex << info.va << dec << endl;
        OUT << "info.mapped_size: " << info.mapped_size << endl;
        OUT << "info.page_size: " << info.page_size << endl;

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        int off = d_A - info.va;
        OUT << "page offset: " << off << endl;

        uint32_t *buf_ptr = (uint32_t *)((char *)bar_ptr + off);
        OUT << "user-space pointer:" << buf_ptr << endl;

        // copy to BAR benchmark
        cout << "File read test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_write_iters << endl;
	// RON
    void *cpu_memory;
	ASSERTDRV(cuMemAllocHost(&cpu_memory, copy_size));
    nvme_to_cpu((uint32_t*)cpu_memory, copy_size, copy_offset);


  nvme_to_gpu(buf_ptr, copy_size, copy_offset);
	if (compare)
		compare_buf(init_buf, buf_ptr + copy_offset/4, copy_size);
	
	nvme_to_gpu_via_host(d_A, copy_size, copy_offset);
	if (compare)
		compare_buf(init_buf, buf_ptr + copy_offset/4, copy_size);


    // Libgib
    //gib_wrapper(M, K, (int*)cpu_memory);
    //gib_wrapper_nvme(M, K, (int*)cpu_memory, d_A);
    // Libgib


        // copy from BAR benchmark
        cout << "File write test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_read_iters << endl;
	cpu_to_nvme((uint32_t*)cpu_memory, copy_size, copy_offset);
  gpu_to_nvme(buf_ptr, copy_size, copy_offset);
	gpu_to_nvme_via_host(d_A, copy_size, copy_offset);

    // EXPERIMENT
    cudaDeviceSynchronize();
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    gib_context gc;
    int rc = gib_init_nvme (d_A, M, K, &gc, gib_block_size, dev_id);
    if (rc) {
	    printf("Error:  %i\n", rc);
	    exit(EXIT_FAILURE);
    }
    clock_gettime(MYCLOCK, &end);
    double init_time = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    cout << "init time " << init_time << endl;
    init_time += d_A_init_time;
    cout << "init time " << d_A_init_time << endl;

    cout << "===== Cc =====" << endl;
    struct time_consumption tc_Cc = Get_Cce(gc, cpu_memory, copy_offset);
    cout << "===== Dhs =====" << endl;
    struct time_consumption tc_Dhs = Get_Dhs(gc, cpu_memory, copy_offset);
    cout << "===== Cg =====" << endl;
    struct time_consumption tc_Cg = Get_Cge(gc, cpu_memory, d_A, copy_offset);
    cout << "===== Dhg =====" << endl;
    struct time_consumption tc_Dhg = Get_Dhg(gc, cpu_memory, d_A, copy_offset);
    cout << "===== Dgh =====" << endl;
    struct time_consumption tc_Dgh = Get_Dgh(gc, cpu_memory, d_A, copy_offset);
    cout << "===== Dgs =====" << endl;
    struct time_consumption tc_Dgs = Get_Dgs(gc, cpu_memory, d_A, buf_ptr, copy_offset);
    
    bool include_init_time = true;
    if(include_init_time)
    {
      //tc_ce.init = init_time;
      //tc_ge.init = init_time;
      //tc_gge.init = init_time;
      //tc_cd.init = init_time;
      //tc_gd.init = init_time;
      //tc_ggd.init = init_time;
    }

    //print_title2();
    print_statistic3("tc_Cc", tc_Cc);
    print_statistic3("tc_Cg", tc_Cg);
    print_statistic3("tc_Dhs", tc_Dhs);
    print_statistic3("tc_Dgs", tc_Dgs);
    print_statistic3("tc_Dhg", tc_Dhg);
    print_statistic3("tc_Dgh", tc_Dgh);

    // EXPERIMENT

        OUT << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);

        OUT << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    OUT << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFree(d_A));
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */

struct time_consumption Get_Cce(gib_context gc, void *original, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    //int *buf = (int*)malloc(size);
    //memcpy(buf, original, small_size);
    void *buf;
    ASSERTDRV(cuMemAllocHost(&buf, size));

    // Step 1.
    for(int i=0;i<RUN_ITER;i++){
        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);
        gib_generate_nc(buf, gib_block_size, gib_block_size, gc);
        cudaDeviceSynchronize();
        clock_gettime(MYCLOCK, &end);
        tc.time[i] = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    }
    cuMemFreeHost(buf);
    return tc;
}

struct time_consumption Get_Ccd(gib_context gc, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    void *buf;
    ASSERTDRV(cuMemAllocHost(&buf, size));
    // Step 1.
    for(int j=0;j<RUN_ITER;j++){
        struct timespec beg, end;

        int *fail_config = (int *)malloc(sizeof(int)*(M+K));
        for (int i = 0; i < M+K; i++)
            fail_config[i] = 0;

        set_fail_config(M, K, fail_config);

        clock_gettime(MYCLOCK, &beg);
        // Step 2.
        //test_config(gc, fail_config, (int*)buf, d_A, false);
        test_config(gc, fail_config, (int *)buf, false);
        clock_gettime(MYCLOCK, &end);
        tc.time[j] = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    }
    return tc;
}

struct time_consumption Get_Dhs(gib_context gc, void *original, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    //int *buf = (int*)malloc(size);
    //memcpy(buf, original, small_size);
    void *buf;
      ASSERTDRV(cuMemAllocHost(&buf, size));

    // Step 2.
    for(int i=0;i<RUN_ITER;i++){
        tc.time[i] = cpu_to_nvme(file_name[ENCODE], (uint32_t *)buf,size, copy_offset);
    }

    cuMemFreeHost(buf);
    return tc;
}

struct time_consumption Get_Cge(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    //int *buf = (int*)malloc(size);
    //memcpy(buf, original, small_size);
    void *buf;
      ASSERTDRV(cuMemAllocHost(&buf, size));

    int size_sc; /* scratch */

    // Step 1.
    struct timespec beg, end;

    // Step 2.
    for(int i=0;i<RUN_ITER;i++){
        clock_gettime(MYCLOCK, &beg);
        gib_generate_nvme(buf, gib_block_size, gc);
        cudaDeviceSynchronize();
        clock_gettime(MYCLOCK, &end);
        tc.time[i] = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    }

    cuMemFreeHost(buf);
    return tc;
}

struct time_consumption Get_Cgd(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    void *buf;
      ASSERTDRV(cuMemAllocHost(&buf, size));

    struct timespec beg, end;

    // Step 3.
    for(int j=0;j<RUN_ITER;j++){
        int *fail_config = (int *)malloc(sizeof(int)*(M+K));
        for (int i = 0; i < M+K; i++)
            fail_config[i] = 0;

        set_fail_config(M, K, fail_config);
        clock_gettime(MYCLOCK, &beg);
        test_config_nvme(gc, fail_config, (int*)buf, d_A);
        cudaDeviceSynchronize();
        clock_gettime(MYCLOCK, &end);
        tc.time[j] = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    }

    cuMemFreeHost(buf);

    return tc;
}

struct time_consumption Get_Dhg(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    void *buf;
      ASSERTDRV(cuMemAllocHost(&buf, size));

    struct timespec beg, end;
    // Step 2.
    for(int i=0;i<RUN_ITER;i++){
        clock_gettime(MYCLOCK, &beg);
        cuMemcpyHtoD(d_A, buf, size);
        cudaDeviceSynchronize();
        clock_gettime(MYCLOCK, &end);
        tc.time[i] = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    }

    cuMemFreeHost(buf);

    return tc;
}

struct time_consumption Get_Dgh(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    void *buf;
      ASSERTDRV(cuMemAllocHost(&buf, size));

    // Step 2.
    struct timespec beg, end;
    for(int i=0;i<RUN_ITER;i++){
        clock_gettime(MYCLOCK, &beg);
        cuMemcpyDtoH(buf, d_A, small_size);
        cudaDeviceSynchronize();
        clock_gettime(MYCLOCK, &end);
        tc.time[i] = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    }

    cuMemFreeHost(buf);

    return tc;
}

struct time_consumption Get_Dgs(gib_context gc,
    void *original, CUdeviceptr d_A, uint32_t *buf_ptr, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    //int *buf = (int*)malloc(size);
    //memcpy(buf, original, small_size);
    void *buf;
      ASSERTDRV(cuMemAllocHost(&buf, size));

    int size_sc; /* scratch */

     // Step 3.
    for(int i=0;i<RUN_ITER;i++){
        tc.time[i] = gpu_to_nvme(file_name[ENCODE], buf_ptr, size, copy_offset);
    }

    cuMemFreeHost(buf);
    return tc;
}


struct time_consumption CPU_encode(gib_context gc, void *original, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    //int *buf = (int*)malloc(size);
    //memcpy(buf, original, small_size);
    void *buf;
	  ASSERTDRV(cuMemAllocHost(&buf, size));

    // Step 1.
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    gib_generate_nc (buf, gib_block_size, gib_block_size, gc);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.encode = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    // Step 2.
    tc.HtoD = cpu_to_nvme(file_name[ENCODE], (uint32_t *)buf,size, copy_offset);

    cuMemFreeHost(buf);
    return tc;
}
struct time_consumption GPU_encode(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    //int *buf = (int*)malloc(size);
    //memcpy(buf, original, small_size);
    void *buf;
	  ASSERTDRV(cuMemAllocHost(&buf, size));

    int size_sc; /* scratch */

    // Step 1.
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyHtoD(d_A, buf, small_size);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.HtoD = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    // Step 2.
    clock_gettime(MYCLOCK, &beg);
    gib_generate_nvme(buf, gib_block_size, gc);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.encode = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    /*
    if(cuda_cmp(original, d_A, small_size)){
        printf("Generation failed.\n");
	    exit(1);
    }
    */
    
    // Step 3.
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyDtoH(buf, d_A, size);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.DtoH = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    
    // Step 4.
    tc.HtoS = cpu_to_nvme(file_name[ENCODE], (uint32_t *)buf,size, copy_offset);
    //dump_file_cpu(file_name[ENCODE], buf, size);

    cuMemFreeHost(buf);
    return tc;
}
struct time_consumption GPU_gdrcopy_encode(gib_context gc,
    void *original, CUdeviceptr d_A, uint32_t *buf_ptr, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    //int *buf = (int*)malloc(size);
    //memcpy(buf, original, small_size);
    void *buf;
	  ASSERTDRV(cuMemAllocHost(&buf, size));

    int size_sc; /* scratch */

    // Step 1.
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyHtoD(d_A, buf, small_size);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.HtoD = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    // Step 2.
    clock_gettime(MYCLOCK, &beg);
    gib_generate_nvme(buf, gib_block_size, gc);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.encode = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    /*
    if(cuda_cmp(original, d_A, small_size)){
        printf("Generation failed.\n");
	    exit(1);
    }
    */

     // Step 3.
    cudaDeviceSynchronize();
    tc.DtoS = gpu_to_nvme(file_name[ENCODE], buf_ptr, size, copy_offset);
    

    cuMemFreeHost(buf);
    return tc;
}

struct time_consumption CPU_decode(gib_context gc, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    void *buf;
	  ASSERTDRV(cuMemAllocHost(&buf, size));
    // Step 1.
    struct timespec beg, end;
    tc.StoH = nvme_to_cpu(file_name[ENCODE], (uint32_t*) buf, size, copy_offset);

    int *fail_config = (int *)malloc(sizeof(int)*(M+K));
    for (int i = 0; i < M+K; i++)
	    fail_config[i] = 0;

    set_fail_config(M, K, fail_config);

    clock_gettime(MYCLOCK, &beg);
    // Step 2.
    //test_config(gc, fail_config, (int*)buf, d_A, false);
    test_config(gc, fail_config, (int *)buf, false);
    clock_gettime(MYCLOCK, &end);
    tc.decode = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    dump_file_cpu(file_name[OUT_FILE], (int*)buf, small_size);

    cuMemFreeHost(buf);

    return tc;
}
struct time_consumption GPU_decode(gib_context gc, void *original, CUdeviceptr d_A, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    void *buf;
	  ASSERTDRV(cuMemAllocHost(&buf, size));
    // Step 1.
    struct timespec beg, end;
    //read_file(file_name[ENCODE],(uint32_t*) buf, size);
    tc.StoH = nvme_to_cpu(file_name[ENCODE], (uint32_t*) buf, size, copy_offset);
    cudaDeviceSynchronize();

    // Step 2.
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyHtoD(d_A, buf, size);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.HtoD = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    // Step 3.
    int *fail_config = (int *)malloc(sizeof(int)*(M+K));
    for (int i = 0; i < M+K; i++)
	    fail_config[i] = 0;

    set_fail_config(M, K, fail_config);
    clock_gettime(MYCLOCK, &beg);
    test_config_nvme(gc, fail_config, (int*)buf, d_A);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.decode = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    // Step 4.
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyDtoH(buf, d_A, small_size);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.DtoH = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    /*
    if(memcmp(buf, original, small_size)){
        printf("Recovery failed.\n");
	    exit(1);
    }
    */
    
    dump_file_cpu(file_name[OUT_FILE], (int*)buf, small_size);

    cuMemFreeHost(buf);

    return tc;
}
struct time_consumption GPU_gdrcopy_decode(gib_context gc, void *original,
    CUdeviceptr d_A, uint32_t *buf_ptr, size_t copy_offset)
{
    struct time_consumption tc = { 0 };

    void *buf;
	  ASSERTDRV(cuMemAllocHost(&buf, size));
    // Step 1.
    struct timespec beg, end;
    tc.StoD = nvme_to_gpu(file_name[ENCODE], buf_ptr, size, copy_offset);
    cudaDeviceSynchronize();
    //read_file(file_name[ENCODE],(uint32_t*) buf_ptr, size);

    // Step 2.
    int *fail_config = (int *)malloc(sizeof(int)*(M+K));
    for (int i = 0; i < M+K; i++)
	    fail_config[i] = 0;

    set_fail_config(M, K, fail_config);
    clock_gettime(MYCLOCK, &beg);
    test_config_nvme(gc, fail_config, (int*)buf, d_A);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.decode = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    /*
    if(cuda_cmp(original, d_A, small_size)){
        printf("Recovery failed.\n");
	    exit(1);
    }
    */

    // Step 3.
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyDtoH(buf, d_A, small_size);
    cudaDeviceSynchronize();
    clock_gettime(MYCLOCK, &end);
    tc.DtoH = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    
    dump_file_cpu(file_name[OUT_FILE], (int*)buf, small_size);

    cuMemFreeHost(buf);

    return tc;
}
