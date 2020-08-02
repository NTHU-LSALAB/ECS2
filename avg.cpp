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
#include <fstream>

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
#include <iomanip>


using namespace std;

#include "gdrapi.h"
#include "common.hpp"

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

void convert(struct time_consumption* tc, string line)
{
    char cstr[1000];
    strcpy(cstr, line.c_str());
    char *token = strtok(cstr, "\t");
    token = strtok(NULL, "\t");
    tc->init += atof(token);
    //cout << tc->init << "\t";

    token = strtok(NULL, "\t");
    tc->HtoD += atof(token);
    //cout << tc->HtoD << "\t";

    token = strtok(NULL, "\t");
    tc->DtoH += atof(token);
    //cout << tc->DtoH << "\t";

    token = strtok(NULL, "\t");
    tc->StoD += atof(token);
    //cout << tc->StoD << "\t";

    token = strtok(NULL, "\t");
    tc->DtoS += atof(token);
    //cout << tc->DtoS << "\t";

    token = strtok(NULL, "\t");
    tc->HtoS += atof(token);
    //cout << tc->HtoS << "\t";

    token = strtok(NULL, "\t");
    tc->StoH += atof(token);
    //cout << tc->StoH << "\t";

    token = strtok(NULL, "\t");
    tc->encode += atof(token);
    //cout << tc->encode << "\t";

    token = strtok(NULL, "\t");
    tc->decode += atof(token);
    //cout << tc->decode << "\t";

    double calculation = tc->encode + tc->decode;
    double hdcomm = tc->HtoD + tc->DtoH;
    double ssdcomm = tc->StoD + tc->DtoS + tc->HtoS + tc->StoH;
    double total = tc->init + calculation + hdcomm + ssdcomm;

    //cout << total << "\t";
    //cout << endl;
}

void print_title()
{
    cout << fixed << setprecision(6);
    cout  << "\t" << "Init" << "\t" << "H2D" << "\t" << "D2H" <<
          "\t" << "S2D" << "\t" << "D2S" << "\t" << "H2S" << "\t" << "S2H" <<
          "\t" << "Encode" << "\t" << "Decode" << "\t" << "Total" << endl;
    //cout  << "Init" << "\t" << "SSDComm" << 
    //      "\t" << "Calculation" << "\t" << "HDComm" << "\t" << "Total" <<endl;
}

void print_title2()
{
    cout << fixed << setprecision(6);
    cout  << "\t" << "Init" << "\t" << "SSDComm" << 
          "\t" << "Calculation" << "\t" << "HDComm" << "\t" << "Total" <<endl;
}

void print_avg(struct time_consumption* tc, int cnt)
{
    cout << tc->init/cnt << "\t";
    cout << tc->HtoD/cnt << "\t";
    cout << tc->DtoH/cnt << "\t";
    cout << tc->StoD/cnt << "\t";
    cout << tc->DtoS/cnt << "\t";
    cout << tc->HtoS/cnt << "\t";
    cout << tc->StoH/cnt << "\t";
    cout << tc->encode/cnt << "\t";
    cout << tc->decode/cnt << "\t";
    double calculation = tc->encode + tc->decode;
    double hdcomm = tc->HtoD + tc->DtoH;
    double ssdcomm = tc->StoD + tc->DtoS + tc->HtoS + tc->StoH;
    double total = tc->init + calculation + hdcomm + ssdcomm;

    cout << total/cnt << "\t";
    cout << endl;
}

void print_avg2(struct time_consumption* tc, int cnt)
{
    double calculation = tc->encode + tc->decode;
    double hdcomm = tc->HtoD + tc->DtoH;
    double ssdcomm = tc->StoD + tc->DtoS + tc->HtoS + tc->StoH;
    double total = tc->init + calculation + hdcomm + ssdcomm;
    cout << tc->init/cnt << "\t" << ssdcomm/cnt << "\t"
         << calculation/cnt << "\t" << hdcomm/cnt << "\t" << total/cnt << endl;
}

main(int argc, char *argv[])
{
  struct time_consumption gpue = { 0 };
  struct time_consumption gpud = { 0 };
  struct time_consumption gpupe = { 0 };
  struct time_consumption gpupd = { 0 };

  string line;
  string line2;
  string line3;
  string line4;
	ifstream myfile;
  myfile.open("latency.txt");
  if(myfile.is_open())
  {
    int cnt = 0;
    while(getline(myfile,line) && getline(myfile,line2)
         && getline(myfile,line3) && getline(myfile,line4))
    {
        convert(&gpue, line);
        convert(&gpud, line2); 
        convert(&gpupe, line3); 
        convert(&gpupd, line4);
        cnt++;
    }
    print_title2();
    print_avg2(&gpue, cnt);
    print_avg2(&gpud, cnt);
    print_avg2(&gpupe, cnt);
    print_avg2(&gpupd, cnt);

    myfile.close();
  }

  return 0;
}