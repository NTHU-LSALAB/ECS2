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
#ifdef  AIO
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
// ron

#include <mpi.h>

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

#define OUT cout
//#define OUT TESTSTACK

//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC

// manually tuned...
//static int num_write_iters = 10000;
//static int num_read_iters  = 100;
static int num_write_iters = 1;
static int num_read_iters  = 1;
const int ORIGINAL = 0;
const int OUT_FILE = 1;
const int ENCODE = 2;
char file_name[3][30] = {"/mnt/2/test.bin","/mnt/2/result.bin","/mnt/2/result_encode.bin"};

int world_rank;
int dev_id = 0;

static double nvme_to_cpu(void *cpu_ptr, size_t copy_size, size_t copy_offset, int which, int print)
{
    int fd = open(file_name[which], O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open failed");
        exit(1);
    }

    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);

    for (int iter = 0; iter<num_write_iters; ++iter)
    {
        ssize_t ret = pread(fd, cpu_ptr, copy_size, 0);
        if(print == 1)
        cout << "copy size:" << ret << endl;
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
    
        double byte_count = (double)copy_size * num_write_iters;
        double dt_ms = (end.tv_nsec - beg.tv_nsec) / 1000000.0 + (end.tv_sec - beg.tv_sec)*1000.0;
        double Bps = byte_count / dt_ms * 1e3;
        woMBps = Bps / 1024.0 / 1024.0;
        if(print == 1)
        cout << "NVMe-to-CPU throughput: " << woMBps << "MB/s" << endl;
    
    return dt_ms;
}

static double nvme_to_gpu(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset, int which, int print)
{
    int fd = open(file_name[which], O_RDONLY | O_DIRECT);
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
        
            double byte_count = (double) copy_size * num_write_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            if(print == 1)
            cout << "NVMe-to-GPU throughput: " << woMBps << "MB/s" << endl;
        
    return dt_ms;
}

//static void nvme_to_gpu_via_host(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
static double nvme_to_gpu_via_host(CUdeviceptr d_A, size_t copy_size, size_t copy_offset, int which, int print)
{
    void *host_buf;
    ASSERTDRV(cuMemAllocHost(&host_buf, copy_size));
    int fd = open(file_name[which], O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open failed");
        exit(1);
    }

        struct timespec beg, end, beg1, end1;
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
        //ASSERTDRV(cuMemcpyHtoD(d_A+copy_offset, host_buf, copy_size));
    }
        clock_gettime(MYCLOCK, &end);
        clock_gettime(MYCLOCK, &beg1);
    for (int iter=0; iter<num_write_iters; ++iter)
    {
        /*ssize_t ret = pread(fd, host_buf, copy_size, 0);
        if (ret != copy_size) {
            perror("read failed");
            printf("%zd bytes read\n", ret);
            //exit(1);
            break;
        }*/
        //gdr_copy_to_bar(buf_ptr + copy_offset/4, host_buf, copy_size);
        ASSERTDRV(cuMemcpyHtoD(d_A+copy_offset, host_buf, copy_size));
    }
        clock_gettime(MYCLOCK, &end1);
        
    close(fd);
    //ASSERTDRV(cuMemFreeHost(host_buf));

        double woMBps;
        
            double byte_count = (double) copy_size * num_write_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double dt_ms1 = (end1.tv_nsec-beg1.tv_nsec)/1000000.0 + (end1.tv_sec-beg1.tv_sec)*1000.0;
            double Bps = byte_count / (dt_ms+dt_ms1) * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            if(print == 1)
            cout << "NVMe-to-host-to-GPU throughput: " << woMBps << "MB/s" << endl;
            cout << "To host:   " << dt_ms << " To GPU: " << dt_ms1 << endl;
        
    return dt_ms+dt_ms1;
}

static double encode(int n, int m, void *data, CUdeviceptr d_A, int gib_block_size, int print){
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);

    if(print == 1)
    cout << "original GPU allocation size: " << (n + m) * gib_block_size << endl;
    gib_context gc;
    int rc = gib_init_nvme(d_A, n, m, &gc, gib_block_size, dev_id);
    if (rc) {
        printf("Error:  %i\n", rc);
        exit(EXIT_FAILURE);
    }
    int gib_encode_size = gib_block_size;
    /*
    rc = gib_alloc(&data, gib_encode_size, &gib_encode_size, gc);
    if (rc) {
        printf("Error:  %i\n", rc);
        exit(EXIT_FAILURE);
    }*/
    rc = gib_generate_nvme(data, gib_encode_size, gc);
    if (rc) {
        printf("Error:  %i\n", rc);
        exit(EXIT_FAILURE);
    }

    clock_gettime(MYCLOCK, &end);

    double woMBps;
    
        double byte_count = (double)gib_encode_size * n * 1;
        double dt_ms = (end.tv_nsec - beg.tv_nsec) / 1000000.0 + (end.tv_sec - beg.tv_sec)*1000.0;
        double Bps = byte_count / dt_ms * 1e3;
        woMBps = Bps / 1024.0 / 1024.0;
        if(print == 1)
        cout << "GPU encode throughput: " << woMBps << "MB/s" << endl;

    return dt_ms;
}

static double decode(int n, int m, void *data, CUdeviceptr d_A, int gib_block_size, int print){
    struct timespec beg1, beg, end;
    
    clock_gettime(MYCLOCK, &beg1);
    clock_gettime(MYCLOCK, &beg);
    // pick some good ones
    char failed[256];
    for (int i = 0; i < n + m; i++)
        failed[i] = 0;
    for (int i = 0; i < ((m < n) ? m : n); i++) {
        int probe;
        do {
            probe = rand() % n;
        } while (failed[probe] == 1);
        failed[probe] = 1;

        /* Destroy the buffer */
        //memset((char *)data + size * probe, 0, size);
    }
    clock_gettime(MYCLOCK, &end);
    double dt_ms = (end.tv_nsec - beg.tv_nsec) / 1000000.0 + (end.tv_sec - beg.tv_sec)*1000.0;
    //cout << "Random: " << dt_ms;

    clock_gettime(MYCLOCK, &beg);
    int buf_ids[256];
    int index = 0;
    int f_index = n;
    for (int i = 0; i < n; i++) {
        while (failed[index]) {
            buf_ids[f_index++] = index;
            index++;
        }
        buf_ids[i] = index;
        index++;
    }
    while (f_index != n + m) {
        buf_ids[f_index] = f_index;
        f_index++;
    }
    clock_gettime(MYCLOCK, &end);
    dt_ms = (end.tv_nsec - beg.tv_nsec) / 1000000.0 + (end.tv_sec - beg.tv_sec)*1000.0;
    //cout << " Preprocess: " << dt_ms;
    clock_gettime(MYCLOCK, &beg);

    // recover
    int gib_decode_size = gib_block_size;
    
    // dirty d_A here !!!!!
    //void *dense_data;
    //gib_alloc((void **)&dense_data, gib_decode_size, &gib_decode_size, gc);
    CUdeviceptr d_dense_data;
    ASSERTDRV(cuMemAlloc(&d_dense_data, (m+n)*gib_decode_size));
    for (int i = 0; i < m + n; i++) {
        //memcpy((unsigned char *)dense_data + i * gib_decode_size,
            //(unsigned char *)data + buf_ids[i] * gib_decode_size, gib_decode_size);
        cuMemcpyDtoD (d_dense_data + i * gib_decode_size, d_A + buf_ids[i] * gib_decode_size, gib_decode_size);
    }

    int nfailed = (m < n) ? m : n;
    //memset((unsigned char *)dense_data + n * gib_decode_size, 0, gib_decode_size*nfailed);
    cuMemsetD8(d_dense_data + n * gib_decode_size, 0, gib_decode_size*nfailed);
    clock_gettime(MYCLOCK, &end);
    dt_ms = (end.tv_nsec - beg.tv_nsec) / 1000000.0 + (end.tv_sec - beg.tv_sec)*1000.0;
    //cout << " Memcpy: " << dt_ms;

    clock_gettime(MYCLOCK, &beg);

    gib_context gc;
    int rc = gib_init_nvme(d_dense_data, n, m, &gc, gib_decode_size, dev_id);
    if (rc) {
        printf("Error:  %i\n", rc);
        exit(EXIT_FAILURE);
    }

    //gib_recover_nvme(dense_data, gib_decode_size, buf_ids, nfailed, gc);
    gib_recover_nvme(data, gib_decode_size, buf_ids, nfailed, gc);

    clock_gettime(MYCLOCK, &end);
    dt_ms = (end.tv_nsec - beg.tv_nsec) / 1000000.0 + (end.tv_sec - beg.tv_sec)*1000.0;
    //cout << " Recover " << dt_ms << endl;

    double woMBps;
    
        double byte_count = (double)gib_decode_size * n * 1;
        //double 
        dt_ms = (end.tv_nsec - beg1.tv_nsec) / 1000000.0 + (end.tv_sec - beg1.tv_sec)*1000.0;
        double Bps = byte_count / dt_ms * 1e3;
        woMBps = Bps / 1024.0 / 1024.0;
        if(print == 1)
        cout << "GPU decode throughput: " << woMBps << "MB/s" << endl;
    

    // copy back
    //cuMemcpyDtoD (d_A, d_dense_data, (m+n) * gib_decode_size);
    for (int i = 0; i < m + n; i++) {
        //memcpy((unsigned char *)dense_data + i * gib_decode_size,
            //(unsigned char *)data + buf_ids[i] * gib_decode_size, gib_decode_size);
        cuMemcpyDtoD (d_A + buf_ids[i] * gib_decode_size, d_dense_data + i * gib_decode_size, gib_decode_size);
    }
    return dt_ms;
}

static double cpu_to_nvme(void *cpu_ptr, size_t copy_size, size_t copy_offset, int which, int print)
{
    int fd = open(file_name[which], O_CREAT | O_WRONLY | O_DIRECT, 0777);
    if (fd < 0) {
        perror("open failed");
        exit(1);
    }

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

        for (int iter=0; iter<num_read_iters; ++iter)
    {
        //gdr_copy_from_bar(host_buf, buf_ptr + copy_offset/4, copy_size);
        ssize_t ret = pwrite(fd, cpu_ptr, copy_size, 0);
        if (ret != copy_size) {
            perror("write failed");
            printf("%zd bytes written\n", ret);
            exit(1);
        }
    }

        clock_gettime(MYCLOCK, &end);
    close(fd);
    //ASSERTDRV(cuMemFreeHost(host_buf));

        double roMBps;
        
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            if(print == 1)
            cout << "GPU-to-host-to-NVMe throughput: " << roMBps << "MB/s" << endl;
        
    return dt_ms;
}


static double gpu_to_nvme(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset, int which, int print)
{
    int fd = open(file_name[which], O_CREAT | O_WRONLY | O_DIRECT, 0777);
    if (fd < 0) {
        perror("open failed");
        exit(1);
    }

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

#ifdef  AIO
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
#ifdef  AIO
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

#ifdef  AIO
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
        
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            if(print == 1)
            cout << "GPU-to-NVMe throughput: " << roMBps << "MB/s" << endl;
        
    return dt_ms;
}

//static void gpu_to_nvme_via_host(uint32_t *buf_ptr, size_t copy_size, size_t copy_offset)
static double gpu_to_nvme_via_host(CUdeviceptr d_A, size_t copy_size, size_t copy_offset, int which, int print)
{
    void *host_buf;
    ASSERTDRV(cuMemAllocHost(&host_buf, copy_size));
    int fd = open(file_name[which], O_CREAT | O_WRONLY | O_DIRECT, 0777);
    if (fd < 0) {
        perror("open failed");
        exit(1);
    }

        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);

#ifdef  AIO
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
#ifdef  AIO
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

#ifdef  AIO
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
    //ASSERTDRV(cuMemFreeHost(host_buf));

        double roMBps;
        
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            if(print == 1)
            cout << "GPU-to-host-to-NVMe throughput: " << roMBps << "MB/s" << endl;
        
    return dt_ms;
}

void CPU_solution_encode(int n, int m, int size, int copy_size, int copy_offset, int gib_block_size, CUdeviceptr d_A){
    void *cpu_ptr;
    ASSERTDRV(cuMemAllocHost(&cpu_ptr, size));
    
    cout << "cpu_malloc size: " << size << endl;
    if (cpu_ptr == NULL) {
        cout << "cpu malloc error." << endl;
    }
    nvme_to_cpu(cpu_ptr, copy_size, copy_offset, ORIGINAL, 0);

    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    gib_context gc;
    int rc = gib_init_nvme(d_A, n, m, &gc, gib_block_size, dev_id);
    if (rc) {
        printf("Error:  %i\n", rc);
        exit(EXIT_FAILURE);
    }
    
    gib_generate_nc (cpu_ptr, gib_block_size, gib_block_size, gc);
    clock_gettime(MYCLOCK, &end);
    double cal_time = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    double HtoSSD = cpu_to_nvme(cpu_ptr, size, copy_offset, ENCODE, 0);
    double total = cal_time + HtoSSD;
    double transfer_SSD = HtoSSD;
    double transfer_HD = 0;
    if(world_rank == 0)
    cout << "CPU Encode:," << total << "," << cal_time << "," << transfer_HD << "," << transfer_SSD << endl;

    //cpu_to_nvme(cpu_ptr, copy_size, copy_offset);
}

void GPU_solution_encode(int n, int m, int size, int copy_size, int copy_offset, int gib_block_size, CUdeviceptr d_A){
    //double transmit_time = nvme_to_gpu_via_host(d_A, copy_size, copy_offset, 0);  
    void *cpu_ptr;
    ASSERTDRV(cuMemAllocHost(&cpu_ptr, size));
    nvme_to_cpu(cpu_ptr, copy_size, copy_offset, ORIGINAL, 0);

    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyHtoD(d_A+copy_offset, cpu_ptr, copy_size);
    clock_gettime(MYCLOCK, &end);
    double HtoD = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    void *data;
    // ENCODE
    double cal_time = encode(n, m, data, d_A, gib_block_size, 0);

    void *host_buf;
    ASSERTDRV(cuMemAllocHost(&host_buf, size));
    
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyDtoH(host_buf, d_A+copy_offset, size);
    clock_gettime(MYCLOCK, &end);
    double DtoH = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    double HtoSSD = cpu_to_nvme(host_buf, size, copy_offset, ENCODE, 0);
    double total = HtoD + cal_time + DtoH + HtoSSD;
    double transfer_SSD = HtoSSD;
    double transfer_HD = HtoD + DtoH;
    if(world_rank == 0)
    cout << "GPU Encode:," << total << "," << cal_time << "," << transfer_HD << "," << transfer_SSD << endl;
}

void GPU_gdrcopy_solution_encode(uint32_t *buf_ptr, int n, int m, int size, int copy_size, int copy_offset, int gib_block_size, CUdeviceptr d_A){
    //double transmit_time = nvme_to_gpu(buf_ptr, copy_size, copy_offset, 0);
    void *cpu_ptr;
    ASSERTDRV(cuMemAllocHost(&cpu_ptr, size));
    nvme_to_cpu(cpu_ptr, copy_size, copy_offset, ORIGINAL, 0);

    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyHtoD(d_A+copy_offset, cpu_ptr, copy_size);
    clock_gettime(MYCLOCK, &end);
    double HtoD = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    void *data;
    // ENCODE
    double cal_time = encode(n, m, data, d_A, gib_block_size, 0);
    double DtoSSD = gpu_to_nvme(buf_ptr, size, copy_offset, ENCODE, 0);
    /*
    void *host_buf;
    ASSERTDRV(cuMemAllocHost(&host_buf, size));
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyDtoH(host_buf, d_A+copy_offset, size);
    clock_gettime(MYCLOCK, &end);
    double transmit_time2 = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;*/
    double total = HtoD + cal_time + DtoSSD;
    double transfer_SSD = DtoSSD;
    double transfer_HD = HtoD;
    if(world_rank == 0)
    cout << "GPU+Encode:," << total << "," << cal_time << "," << transfer_HD << "," << transfer_SSD << endl;
}

void CPU_solution_decode(int n, int m, int size, int copy_size, int copy_offset, int gib_block_size, CUdeviceptr d_A, void *cpu_ptr){
    struct timespec beg, end;
    double SSDtoH = nvme_to_cpu(cpu_ptr, size, copy_offset, ENCODE, 0);
        // pick some good ones
    char failed[256];
    for (int i = 0; i < n + m; i++)
        failed[i] = 0;
    for (int i = 0; i < ((m < n) ? m : n); i++) {
        int probe;
        do {
            probe = rand() % n;
        } while (failed[probe] == 1);
        failed[probe] = 1;

        /* Destroy the buffer */
        //memset((char *)data + size * probe, 0, size);
    }

    clock_gettime(MYCLOCK, &beg);
    int buf_ids[256];
    int index = 0;
    int f_index = n;
    for (int i = 0; i < n; i++) {
        while (failed[index]) {
            buf_ids[f_index++] = index;
            index++;
        }
        buf_ids[i] = index;
        index++;
    }
    while (f_index != n + m) {
        buf_ids[f_index] = f_index;
        f_index++;
    }

    // recover
    int gib_decode_size = gib_block_size;
    
    // dirty d_A here !!!!!
    void *dense_data;
    gib_context gc1;
    int rc = gib_init_nvme(d_A, n, m, &gc1, gib_decode_size, dev_id);
    if (rc) {
        printf("Error:  %i\n", rc);
        exit(EXIT_FAILURE);
    }
    gib_alloc((void **)&dense_data, gib_decode_size, &gib_decode_size, gc1);
    for (int i = 0; i < m + n; i++) {
        memcpy((unsigned char *)dense_data + i * gib_decode_size,
            (unsigned char *)cpu_ptr + buf_ids[i] * gib_decode_size, gib_decode_size);
        
    }

    int nfailed = (m < n) ? m : n;
    memset((unsigned char *)dense_data + n * gib_decode_size, 0, gib_decode_size*nfailed);

    //gib_recover_nc (dense_data, gib_block_size, gib_block_size, buf_ids, nfailed, gc1);
    gib_cpu_recover(dense_data, gib_block_size, buf_ids, nfailed, gc1);
    clock_gettime(MYCLOCK, &end);
    double cal_time = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;

    compare_buf((uint32_t *)dense_data, (uint32_t *)cpu_ptr + copy_offset/4, copy_size/4);
    double total = SSDtoH + cal_time;
    double transfer_SSD = SSDtoH;
    double transfer_HD = 0;
    if(world_rank == 0)
    cout << "CPU Decode:," << total << "," << cal_time << "," << transfer_HD << "," << transfer_SSD << endl;
}

void GPU_solution_decode(int n, int m, int size, int copy_size, int copy_offset, int gib_block_size, CUdeviceptr d_A){
    void *data;
    void *cpu_ptr;
    ASSERTDRV(cuMemAllocHost(&cpu_ptr, size));
    struct timespec beg, end;
    double SSDtoH = nvme_to_cpu(cpu_ptr, size, copy_offset, ENCODE, 0);
    clock_gettime(MYCLOCK, &beg);
    cuMemcpyHtoD(d_A+copy_offset, cpu_ptr, size);
    clock_gettime(MYCLOCK, &end);
    double HtoD = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    
    double cal_time = decode(n, m, data, d_A, gib_block_size, 0);

    clock_gettime(MYCLOCK, &beg);
    cuMemcpyDtoH(cpu_ptr, d_A+copy_offset, size);
    clock_gettime(MYCLOCK, &end);
    double DtoH = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    double total = SSDtoH + HtoD + cal_time + DtoH;
    double transfer_SSD = SSDtoH;
    double transfer_HD = HtoD + DtoH;
    if(world_rank == 0)
    cout << "GPU Decode:," << total << "," << cal_time << "," << transfer_HD << "," << transfer_SSD << endl;
}

void GPU_gdrcopy_solution_decode(uint32_t *buf_ptr, int n, int m, int size, int copy_size, int copy_offset, int gib_block_size, CUdeviceptr d_A){
    void *data;
    void *cpu_ptr;
    ASSERTDRV(cuMemAllocHost(&cpu_ptr, size));
    struct timespec beg, end;
    double SSDtoD = nvme_to_gpu(buf_ptr, size, copy_offset, ENCODE, 0);

    double cal_time = decode(n, m, data, d_A, gib_block_size, 0);

    clock_gettime(MYCLOCK, &beg);
    cuMemcpyDtoH(cpu_ptr, d_A+copy_offset, size);
    clock_gettime(MYCLOCK, &end);
    double DtoH = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    double total = SSDtoD + cal_time + DtoH;
    double transfer_SSD = SSDtoD;
    double transfer_HD = DtoH;
    if(world_rank == 0)
    cout << "GPU+Decode:," << total << "," << cal_time << "," << transfer_HD << "," << transfer_SSD << endl;
}

main(int argc, char *argv[])
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    size_t _size = 128*1024;
    size_t copy_size = 0;
    size_t copy_offset = 0;
    int compare = 1;

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
            dev_id = strtol(optarg, NULL, 0);
            dev_id += world_rank;
            file_name[0][5] = '0'+dev_id;
            file_name[1][5] = '0'+dev_id;
            file_name[2][5] = '0'+dev_id;
            cout << file_name[0] << file_name[1] << file_name[2] << endl;
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
        default:
            printf("ERROR: invalid option\n");
            exit(EXIT_FAILURE);
        }
    }

    int n = 4;
    int m = 2;
    int small_size = _size;
    _size = _size * (n+m)/n;
    int gib_block_size = small_size/n;

    cout << "small_size: " << small_size << " _size: " << _size; 
    if (!copy_size)
        //copy_size = _size;
        copy_size = small_size;

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
    OUT << "selecting device " << dev_id << endl;
    ASSERTRT(cudaSetDevice(dev_id));

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    OUT << "testing size: " << _size << endl;
    OUT << "rounded size: " << size << endl;

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
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
    // Ron
    void *cpu_ptr;
    ASSERTDRV(cuMemAllocHost(&cpu_ptr, size));
    
    cout << "cpu_malloc size: " << size << endl;
    if (cpu_ptr == NULL) {
        cout << "cpu malloc error." << endl;
    }
    nvme_to_cpu(cpu_ptr, copy_size, copy_offset, ORIGINAL, 1);
    if (compare)
        compare_buf(init_buf, buf_ptr + copy_offset/4, copy_size);
    
    //int gib_block_size = 1024 * 1024;

    gib_context gc1;
    int rc = gib_init_nvme(d_A, n, m, &gc1, gib_block_size, dev_id);
    if (rc) {
        printf("Error:  %i\n", rc);
        exit(EXIT_FAILURE);
    }
    
    gib_generate_nc (cpu_ptr, gib_block_size, gib_block_size, gc1);
    
    // ron

    nvme_to_gpu(buf_ptr, copy_size, copy_offset, ORIGINAL, 1);
    if (compare)
        compare_buf(init_buf, buf_ptr + copy_offset/4, copy_size);

    //nvme_to_gpu_via_host(buf_ptr, copy_size);
    nvme_to_gpu_via_host(d_A, copy_size, copy_offset, ORIGINAL, 1);
    if (compare)
        compare_buf(init_buf, buf_ptr + copy_offset/4, copy_size);

    // check cpu
    /*
    for(int i=0;i<copy_size/4;i++){
        if(*((uint32_t *)cpu_ptr+i) != *(buf_ptr + copy_offset/4 + i))
        cout << *((uint32_t *)cpu_ptr+i) << " " << *(buf_ptr + copy_offset/4 + i) << endl;
    }*/


    void *data;
    // ENCODE
    encode(n, m, data, d_A, gib_block_size, 1);
    void *host_buf;
    ASSERTDRV(cuMemAllocHost(&host_buf, size));
    cuMemcpyDtoH(host_buf, d_A+copy_offset, size);
    //compare_buf((uint32_t *)host_buf, buf_ptr + copy_offset/4, copy_size);
    /*
    for(int i=0;i<copy_size/4;i++){
        if(*((uint32_t *)host_buf+i) != *(buf_ptr + copy_offset/4 + i))
        cout << *((uint32_t *)host_buf+i) << " " << *(buf_ptr + copy_offset/4 + i) << endl;
    }
    for(int i=0;i<copy_size/4;i++){
        if(*((uint32_t *)host_buf+i) != *((uint32_t *)cpu_ptr + i))
        cout << *((uint32_t *)host_buf+i) << " " << *((uint32_t *)cpu_ptr + i) << endl;
    }*/
    // ENCODE

    // DECODE
    decode(n, m, data, d_A, gib_block_size, 1);
    void *host_buf2;
    ASSERTDRV(cuMemAllocHost(&host_buf2, copy_size));
    cuMemcpyDtoH(host_buf2, d_A+copy_offset, copy_size);
    // NEED CHANGE
    //compare_buf((uint32_t *)host_buf2, (uint32_t *)cpu_ptr + copy_offset/4, copy_size/4);
    
    cout << "DtoH done."<< endl;

    // DECODE
    // pick some good ones
    char failed[256];
    for (int i = 0; i < n + m; i++)
        failed[i] = 0;
    for (int i = 0; i < ((m < n) ? m : n); i++) {
        int probe;
        do {
            probe = rand() % n;
        } while (failed[probe] == 1);
        failed[probe] = 1;

        /* Destroy the buffer */
        //memset((char *)data + size * probe, 0, size);
    }

    cout << "Random fail done."<< endl;

    int buf_ids[256];
    int index = 0;
    int f_index = n;
    for (int i = 0; i < n; i++) {
        while (failed[index]) {
            buf_ids[f_index++] = index;
            index++;
        }
        buf_ids[i] = index;
        index++;
    }
    while (f_index != n + m) {
        buf_ids[f_index] = f_index;
        f_index++;
    }

    cout << "Preprocess done."<< endl;

    // recover
    int gib_decode_size = gib_block_size;
    
    // dirty d_A here !!!!!
    void *dense_data;
    gib_alloc((void **)&dense_data, gib_decode_size, &gib_decode_size, gc1);
    for (int i = 0; i < m + n; i++) {
        memcpy((unsigned char *)dense_data + i * gib_decode_size,
            (unsigned char *)cpu_ptr + buf_ids[i] * gib_decode_size, gib_decode_size);
        
    }
    cout << "Alloc done."<< endl;


    int nfailed = (m < n) ? m : n;
    memset((unsigned char *)dense_data + n * gib_decode_size, 0, gib_decode_size*nfailed);
    cout << "memset done."<< endl;

    //gib_recover_nc (dense_data, gib_block_size, gib_block_size, buf_ids, nfailed, gc1);
    gib_cpu_recover(dense_data, gib_block_size, buf_ids, nfailed, gc1);
    cout << "Recover done."<< endl;

    compare_buf((uint32_t *)dense_data, (uint32_t *)cpu_ptr + copy_offset/4, copy_size/4);
    cout << "Comp done."<< endl;


        // copy from BAR benchmark
        cout << "File write test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_read_iters << endl;
    gpu_to_nvme(buf_ptr, copy_size, copy_offset, OUT_FILE, 1);
    //gpu_to_nvme_via_host(buf_ptr, copy_size);
    gpu_to_nvme_via_host(d_A, copy_size, copy_offset, OUT_FILE, 1);



    // EXPERIMENT START
    cout << "EXPERIMENT STARTS" << endl << endl;

    // clean up d_A
    ASSERTDRV(cuMemsetD32(d_A, 0xdeadbeef, size / sizeof(unsigned int)));
    MPI_Barrier(MPI_COMM_WORLD);
    CPU_solution_encode(n, m, size, copy_size, copy_offset, gib_block_size, d_A);
    MPI_Barrier(MPI_COMM_WORLD);
    GPU_gdrcopy_solution_encode(buf_ptr, n, m, size, copy_size, copy_offset, gib_block_size, d_A);
    MPI_Barrier(MPI_COMM_WORLD);
    GPU_solution_encode(n, m, size, copy_size, copy_offset, gib_block_size, d_A);
    MPI_Barrier(MPI_COMM_WORLD);
    GPU_solution_encode(n, m, size, copy_size, copy_offset, gib_block_size, d_A);
    MPI_Barrier(MPI_COMM_WORLD);
    GPU_gdrcopy_solution_encode(buf_ptr, n, m, size, copy_size, copy_offset, gib_block_size, d_A);
    
    if(world_rank == 0)
    cout << endl;

    MPI_Barrier(MPI_COMM_WORLD);
    CPU_solution_decode(n, m, size, copy_size, copy_offset, gib_block_size, d_A, cpu_ptr);
    MPI_Barrier(MPI_COMM_WORLD);
    GPU_gdrcopy_solution_decode(buf_ptr, n, m, size, copy_size, copy_offset, gib_block_size, d_A);
    MPI_Barrier(MPI_COMM_WORLD);
    GPU_solution_decode(n, m, size, copy_size, copy_offset, gib_block_size, d_A);    
    MPI_Barrier(MPI_COMM_WORLD);
    GPU_solution_decode(n, m, size, copy_size, copy_offset, gib_block_size, d_A);
    MPI_Barrier(MPI_COMM_WORLD);
    GPU_gdrcopy_solution_decode(buf_ptr, n, m, size, copy_size, copy_offset, gib_block_size, d_A);
    

    MPI_Barrier(MPI_COMM_WORLD);


        OUT << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);

        OUT << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    OUT << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFree(d_A));
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
