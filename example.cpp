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
#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
// ron

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

#define OUT cout
//#define OUT TESTSTACK

//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC

main(int argc, char *argv[])
{
	size_t _size = 2147483647;
	int dev_id = 0;
	int m = 4;
	int k = 2;
  char configPath[50];
  // Set default config at HOME
  strcpy(configPath, getenv("HOME"));

  // File name to be stored in ECS2
  char ecs2Filename[50] = "ecs2_file";
  // File name to be read from/write to in base file system
  char baseFilename[50] = "src_file";

  char* dir_path[50];
  string line;
  ifstream infile ("directoryfile");
  if (infile.is_open())
  {
          int i = 0;
          while(getline(infile, line))
          {
                  dir_path[i] = (char*)malloc(100);
                  strcpy(dir_path[i],line.c_str());
                  cout << dir_path[i] << endl;
          }
          infile.close();
  }
  
  // Actions
  int createFile = 0;
  int readFile = 0;
  int writeFile = 0;
  int pwriteFile = 0;
  int preadFile = 0;

	while (1) {
		int c;
		c = getopt(argc, argv, "s:k:m:g:d:o:c:C:R:r:W:w:r:p:P:f:M:K:hn");
		if (c == -1)
			break;
		switch (c) {
		case 'S':
			_size = strtol(optarg, NULL, 0);
			break;
    case 'K':
			_size = strtol(optarg, NULL, 0);
      _size <<= 10;
			break;
    case 'M':
			_size = strtol(optarg, NULL, 0);
      _size <<= 20;
			break;
    case 'G':
			_size = strtol(optarg, NULL, 0);
      _size <<= 30;
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
      strcpy(baseFilename, optarg);
      break;
    case 'c':
      strcpy(configPath, optarg);
      break;


    case 'C':
      createFile = 1;
      strcpy(ecs2Filename, optarg);      
      break;
    case 'W':
      pwriteFile = 1;
      strcpy(ecs2Filename, optarg);
      break;
    case 'w':
      writeFile = 1;
      strcpy(ecs2Filename, optarg);
      break;
    case 'R':
      preadFile = 1;
      strcpy(ecs2Filename, optarg);
      break;
    case 'r':
      readFile = 1;
      strcpy(ecs2Filename, optarg);
      break;
		}
	}
	
	struct GE_config config = GE_init(dev_id, _size, configPath);
	
  if(createFile){
    int create = -1;
  
    create = GE_create(config, m, k, ecs2Filename, dir_path);
    if (create != 0) {
      printf("Create error.\n");
      exit(0);
    }
    return 0;
  }

  else if(writeFile){
    printf("Writing %s into ECS2...\n", baseFilename);
    FILE* uFile = fopen(baseFilename, "r");
    if( NULL == uFile ){
      printf( "open failure" );
      exit(0);
    }
    struct stat st;
    stat(baseFilename, &st);
    size_t uFileSize = st.st_size;
    int* buffer = new int[uFileSize/sizeof(int)];
    int read_from_file = read(fileno(uFile), (void *)buffer, uFileSize);

    GE_file ge_file = GE_open(config, ecs2Filename);
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    GE_write(config, &ge_file, (void*)buffer, uFileSize);
    clock_gettime(MYCLOCK, &end);
    double time = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    printf("Time elapsed: %lf\n",time);
    return 0;
  }

	else if(readFile){
    GE_file ge_file = GE_open(config, ecs2Filename);
    //int* buffer = new int[ge_file.size/4];
    void* buffer;
    ASSERTDRV(cuMemAllocHost(&buffer, ge_file.size));

    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    GE_read(config, &ge_file, buffer);
    clock_gettime(MYCLOCK, &end);
    double time = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    printf("Time elapsed: %lf\n",time);
    
    cuMemcpyDtoH(buffer, config.d_buffer, ge_file.size);
    FILE* uFile = fopen(baseFilename, "w");
    write(fileno(uFile), (void *)buffer, ge_file.size);
    fclose(uFile);
    return 0;
  }

  else if(pwriteFile){
    printf("Parallelly writing %s into ECS2...\n", baseFilename);
    FILE* uFile = fopen(baseFilename, "r");
    if( NULL == uFile ){
      printf( "open failure" );
      exit(0);
    }
    struct stat st;
    stat(baseFilename, &st);
    size_t uFileSize = st.st_size;
    //int* buffer = new int[uFileSize/sizeof(int)];
    void* buffer;
    ASSERTDRV(cuMemAllocHost(&buffer, uFileSize));
    int read_from_file = read(fileno(uFile), (void *)buffer, uFileSize);

    GE_file ge_file = GE_open(config, ecs2Filename);
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    GE_pwrite(config, &ge_file, (void*)buffer, uFileSize);
    clock_gettime(MYCLOCK, &end);
    double time = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    printf("Time elapsed: %lf\n",time);
    return 0;
  }

  else if(preadFile){
    GE_file ge_file = GE_open(config, ecs2Filename);
    //int* buffer = new int[ge_file.size/4];
    void* buffer;
    ASSERTDRV(cuMemAllocHost(&buffer, ge_file.size));
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    GE_pread(config, &ge_file, buffer);
    clock_gettime(MYCLOCK, &end);
    double time = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    printf("Time elapsed: %lf\n",time);
    
    cuMemcpyDtoH(buffer, config.d_buffer, ge_file.size);
    FILE* uFile = fopen(baseFilename, "w");
    write(fileno(uFile), (void *)buffer, ge_file.size);
    fclose(uFile); 
    return 0;
  }

  return 0;
}