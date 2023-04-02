
char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* CNN                                 */
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

void cnn();
void reset_nanosec();
void show_nanosec();
void reset_time();
void show_time();

#define IC    18
#define OC    16
#define M     242
#define K     3
#define THREADX 8
#define THREADY 8
#define BLOCKX ((M-2)/THREADX)
#define BLOCKY ((M-2)/THREADY)

__global__ void cnn(float *in, float *ker, float *out)
{
  int ic, row, col, oc, y, x, kidx;
  float *ip0, *kp, sum;
  row = blockIdx.x*blockDim.x + threadIdx.x + 1;
  col = blockIdx.y*blockDim.y + threadIdx.y + 1;
  for (oc=0; oc<OC; oc++) { /* set output channel */
    sum = 0.0f;
    for (ic=0; ic<IC; ic++) { /* set input channel */
      ip0 = &in[ic*M*M]; /* top of input */
      kp = &ker[(oc*IC+ic)*K*K];
      kidx = 0;
      for (y=-((K-1)/2); y<=(K-1)/2; y++) { /* kernel loop */
	for (x=-((K-1)/2); x<=(K-1)/2; x++) {
	  sum += ip0[(row+y)*M+col+x] * kp[kidx];
	  kidx++;
	}
      }
    }
    out[oc*M*M+row*M+col] = sum; /* top of output */
  }
  __syncthreads();
}

int count2;

main(int argc, char **argv)
{
  float *hin, *hker, *hout0, *hout1, *din, *dker, *dout;
  int ic, row, col, oc, y, x, kidx;
  float *ip0, *kp, sum;

  hin              = (float*)malloc(sizeof(float) * IC*M*M);
  hker             = (float*)malloc(sizeof(float) * IC*OC*K*K);
  hout0            = (float*)malloc(sizeof(float) * OC*M*M);
  hout1            = (float*)malloc(sizeof(float) * OC*M*M);
  if (cudaSuccess != cudaMalloc((void**)&din,  sizeof(float) *IC*M*M))    { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dker, sizeof(float) *IC*OC*K*K)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dout, sizeof(float) *OC*M*M))    { printf("can't cudaMalloc\n"); exit(1); }

  /* initialize a and b */
  for (ic=0; ic<IC; ic++) {
    for (row=0; row<M; row++) {
      for (col=0; col<M; col++) {
        hin[ic*M*M+row*M+col] = ic<<12|(((M/2-abs(row-M/2))*(M/2-abs(col-M/2)))&0xfff);
      }
    }
  }
  for (oc=0; oc<OC; oc++) {
    for (ic=0; ic<IC; ic++) {
      for (y=0; y<K; y++) {
        for (x=0; x<K; x++) {
          hker[ic*OC*K*K+oc*K*K+y*K+x] = (oc-ic)*((2-abs(y-K/2))*(2-abs(x-K/2)))/OC;
        }
      }
    }
  }

  /* CPU */
  printf("CPU ");
  reset_nanosec();
  for (row=1; row<M-1; row++) { /* image loop */
    for (col=1; col<M-1; col++) {
      for (oc=0; oc<OC; oc++) { /* set output channel */
	sum = 0.0f;
	for (ic=0; ic<IC; ic++) { /* set input channel */
	  ip0 = &hin[ic*M*M]; /* top of input */
	  kp = &hker[(oc*IC+ic)*K*K];
          kidx = 0;
          for (y=-((K-1)/2); y<=(K-1)/2; y++) { /* kernel loop */
            for (x=-((K-1)/2); x<=(K-1)/2; x++) {
	      sum += ip0[(row+y)*M+col+x] * kp[kidx];
              kidx++;
            }
          }
        }
	hout0[oc*M*M+row*M+col] = sum; /* top of output */
      }
    }
  }
  show_nanosec();

  /* GPU */
  printf("GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(din,   hin,  sizeof(float)*IC*M*M,    cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dker,  hker, sizeof(float)*IC*OC*K*K, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  dim3 Thread  = dim3(THREADX, THREADY, 1);
  dim3 Block   = dim3(BLOCKX, BLOCKY, 1);
  cnn<<<Block,Thread>>>(din, dker, dout); /* search triangle in {frontier,next} */
  if (cudaSuccess != cudaMemcpy(hout1, dout, sizeof(float)*OC*M*M,    cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }
  show_nanosec();

#if 1
  for (oc=0; oc<OC; oc++) {
    for (row=1; row<M-1; row++) {
      for (col=1; col<M-1; col++) {
        if (hout0[oc*M*M+row*M+col] != hout1[oc*M*M+row*M+col]) {
          count2++;
          printf("o0[%d]=%f o1[%d]=%f\n",
                 oc*M*M+row*M+col, (double)hout0[oc*M*M+row*M+col],
                 oc*M*M+row*M+col, (double)hout1[oc*M*M+row*M+col]);
        }
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
