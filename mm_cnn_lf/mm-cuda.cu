
char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* MM                                  */
/*        Copyright (C) 2013- by NAIST */
/*         Primary writer: Y.Nakashima */
/*                nakashim@is.naist.jp */

/* ★★★ TX2で実行の際には, root になる必要あり ★★★ */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <cuda_runtime.h>

typedef unsigned long long Ull;
typedef unsigned int Uint;
typedef unsigned char Uchar;

void mm();
void reset_nanosec();
void show_nanosec();
void reset_time();
void show_time();

#define M 480
#define THREADX 8
#define THREADY 8
#define BLOCKX (M/THREADX)
#define BLOCKY (M/THREADY)

__global__ void mm(float *a, float *b, float *c)
{
  int row, col, n;
  float sum = 0.0f;
  row = blockIdx.x*blockDim.x + threadIdx.x;
  col = blockIdx.y*blockDim.y + threadIdx.y;

  for (n=0; n<M; n++) {
    sum += a[row*M+n] * b[n*M+col];
  }
  c[row*M+col] = sum;
  __syncthreads();
}

int count2;

main(int argc, char **argv)
{
  float *hA, *hB, *hC0, *hC1, *dA, *dB, *dC;
  int row, col, n;

  hA               = (float*)malloc(sizeof(float) * M * M);
  hB               = (float*)malloc(sizeof(float) * M * M);
  hC0              = (float*)malloc(sizeof(float) * M * M);
  hC1              = (float*)malloc(sizeof(float) * M * M);
  if (cudaSuccess != cudaMalloc((void**)&dA, sizeof(float) * M * M)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dB, sizeof(float) * M * M)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dC, sizeof(float) * M * M)) { printf("can't cudaMalloc\n"); exit(1); }

  /* initialize a and b */
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      hA[row*M+col] = row%120+1;
      hB[row*M+col] = col%120+1;
    }
  }

  /* CPU */
  printf("CPU ");
  reset_nanosec();
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      float sum = 0.0f;
      for (n=0; n<M; n++)
        sum += hA[row*M+n] * hB[n*M+col];
      hC0[row*M+col] = sum;
    }
  }
  show_nanosec();

  /* GPU */
  printf("GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(dA, hA, sizeof(float)*M*M, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dB, hB, sizeof(float)*M*M, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  dim3 Thread  = dim3(THREADX, THREADY, 1);
  dim3 Block   = dim3(BLOCKX, BLOCKY, 1);
  mm<<<Block,Thread>>>(dA, dB, dC); /* search triangle in {frontier,next} */
  if (cudaSuccess != cudaMemcpy(hC1, dC, sizeof(float)*M*M, cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }
  show_nanosec();

#if 1
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      if (hC0[row*M+col] != hC1[row*M+col]) {
        count2++;
        printf("C0[%d][%d]=%f C1[%d][%d]=%f\n", row, col, (double)hC0[row*M+col],
                                                row, col, (double)hC1[row*M+col]);
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
#endif

  return (0);
}

Ull     nanosec_sav, nanosec;
double  tmssave, tms;
long    ticksave, ticks;
struct  rusage rusage;

void reset_nanosec()
{
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec_sav = 1000000000*ts.tv_sec + ts.tv_nsec;
}

void show_nanosec()
{
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec = 1000000000*ts.tv_sec + ts.tv_nsec;
  printf("nanosec: ARM:%llu\n", nanosec - nanosec_sav);
  nanosec_sav = nanosec;
}

void reset_time(void)
{
  struct tms    utms;

  times(&utms);
  ticksave = utms.tms_utime;
}

void show_time(void)
{
  struct tms    utms;

  times(&utms);
  ticks = utms.tms_utime;
  printf("====TOTAL-CPUS-TIME(w/o IO) %g sec===\n", (double)(ticks-ticksave)/sysconf(_SC_CLK_TCK));
  ticksave = ticks;
}
