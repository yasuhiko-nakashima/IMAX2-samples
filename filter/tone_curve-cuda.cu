
char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* tone_curve                          */
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

void tone_curve();
void reset_nanosec();
void show_nanosec();
void reset_time();
void show_time();

#define WD      320
#define HT      240
#define THREADX 16
#define THREADY 16
#define BLOCKX (WD/THREADX)
#define BLOCKY (HT/THREADY)

__global__ void tone_curve(Uint *out, Uint *in, Uchar *t)
{
  int row, col;
  row = blockIdx.y*blockDim.y + threadIdx.y;
  col = blockIdx.x*blockDim.x + threadIdx.x;
  Uint pix = in[row*WD+col];
  out[row*WD+col] = ((t)[pix>>24])<<24 | (t[256+((pix>>16)&255)])<<16 | (t[512+((pix>>8)&255)])<<8;
  __syncthreads();
}

int count2;

main(int argc, char **argv)
{
  Uint *hout0, *hout1, *hin, *dout, *din;
  Uchar *ht, *dt;
  int row, col, i;

  hout0            = (Uint*)malloc(sizeof(Uint) * WD * HT);
  hout1            = (Uint*)malloc(sizeof(Uint) * WD * HT);
  hin              = (Uint*)malloc(sizeof(Uint) * WD * HT);
  ht               = (Uchar*)malloc(sizeof(Uchar) * 256 * 3);
  if (cudaSuccess != cudaMalloc((void**)&dout, sizeof(Uint) * WD * HT)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&din,  sizeof(Uint) * WD * HT)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dt,   sizeof(Uint) * 256 * 3)) { printf("can't cudaMalloc\n"); exit(1); }

  /* initialize a and b */
  for (row=0; row<HT; row++) {
    for (col=0; col<WD; col++) {
      hin[row*WD+col] = (row<<24)|(col<<8);
    }
  }
  for(i=0; i<256; i++) {
    ht[i+  0] = 0xff-i;
    ht[i+256] = 0xff-i;
    ht[i+512] = 0xff-i;
  }

  /* CPU */
  printf("CPU ");
  reset_nanosec();
  for (row=0; row<HT; row++) {
    for (col=0; col<WD; col++) {
      Uint pix = hin[row*WD+col];
      hout0[row*WD+col] = ((ht)[pix>>24])<<24 | (ht[256+((pix>>16)&255)])<<16 | (ht[512+((pix>>8)&255)])<<8;
    }
  }
  show_nanosec();

  /* GPU */
  printf("GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(din, hin, sizeof(Uint)*WD*HT, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dt, ht, sizeof(Uint)*256*3, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  dim3 Thread  = dim3(THREADX, THREADY, 1);
  dim3 Block   = dim3(BLOCKX, BLOCKY, 1);
  tone_curve<<<Block,Thread>>>(dout, din, dt); /* search triangle in {frontier,next} */
  if (cudaSuccess != cudaMemcpy(hout1, dout, sizeof(Uint)*WD*HT, cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }
  show_nanosec();

#if 1
  for (row=0; row<HT; row++) {
    for (col=0; col<WD; col++) {
      if (hout0[row*WD+col] != hout1[row*WD+col]) {
        count2++;
        printf("[%d][%d] out0=%x out1=%x\n", row, col, hout0[row*WD+col], hout1[row*WD+col]);
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
