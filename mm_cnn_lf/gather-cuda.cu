
char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* GATHER                              */
/*        Copyright (C) 2013- by NAIST */
/*         Primary writer: Y.Nakashima */
/*                nakashim@is.naist.jp */

/* ★★★ TX2で実行の際には, root になる必要あり ★★★ */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <cuda_runtime.h>

typedef unsigned long long Ull;
typedef unsigned int Uint;
typedef unsigned char Uchar;

void gather();
void reset_nanosec();
void show_nanosec();
void reset_time();
void show_time();

#define IM    7500
#define OM    1600
#define R     75
#define PAD   32
#define THREADX 8
#define THREADY 8
#define BLOCKX ((OM-PAD*2)/THREADX)
#define BLOCKY ((OM-PAD*2)/THREADY)
#define MAXDELTA  4  /* -3,-2,-1,0,1,2,3 */
#define WBASE    (MAXDELTA*MAXDELTA*2)
#define ofs     14
#define delta  ((R/2/(14+1)-1) ? (R/2/(14+1)-1) : 1)
int total_weight;  /* 7 x+3   +60   (+3,+3) */
                   /* 6 x+2   +52           */
                   /* 5 x+1   +44           */
                   /* 4 x 0   +36*-center   */
                   /* 3 x-1   +28           */
                   /* 2 x-2   +20  MAXDELTA */
                   /* 1 x-3   +12 /         */
                   /* 0 x x x x x x x x     */
                   /*           0->         */
                   /*   0 1 2 3 4 5 6 7     */
Uchar *rgb; /*[IM*IM*3];*/

__global__ void gather(int *in, int *wt, int *out)
{
  int c, i, j;
  int row, col, rx, ry;
  int w, cvalR, cvalG, cvalB;
  row = blockIdx.x*blockDim.x + threadIdx.x + PAD;
  col = blockIdx.y*blockDim.y + threadIdx.y + PAD;
  ry = (R+ofs)*IM;
  rx = (R+ofs);
       c = ((row>>4)*R + (((~row&15)*ofs)>>4))*IM
	  + (col>>4)*R + (((~col&15)*ofs)>>4);
      cvalR=0;
      cvalG=0;
      cvalB=0;
      for (i=-1; i<=1; i++) {
	for (j=-1; j<=1; j++) {
	  Uint pix = in[c+ry*i+rx*j];
	  w = wt[WBASE+i*MAXDELTA*2+j];
	  cvalR += ((pix>>24)&255)*w;
	  cvalG += ((pix>>16)&255)*w;
	  cvalB += ((pix>> 8)&255)*w;
	}
      }
      out[row*OM+col] = ((cvalR>>8)<<24) | ((cvalG>>8)<<16) | ((cvalB>>8)<<8);
  __syncthreads();
}

int count2;

main(int argc, char **argv)
{
  int fd, *hin, *hwt, *hout0, *hout1, *din, *dwt, *dout;
  int c, i, j;
  int row, col, rx, ry;
  int w, cvalR, cvalG, cvalB;

  hin              = (int*)malloc(sizeof(int) * IM*IM);
  hwt              = (int*)malloc(sizeof(int) * MAXDELTA*MAXDELTA*2*2);
  hout0            = (int*)malloc(sizeof(int) * OM*OM);
  hout1            = (int*)malloc(sizeof(int) * OM*OM);
  if (cudaSuccess != cudaMalloc((void**)&din,  sizeof(int) *IM*IM)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dwt,  sizeof(int) *MAXDELTA*MAXDELTA*2*2)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dout, sizeof(int) *OM*OM)) { printf("can't cudaMalloc\n"); exit(1); }

  /* initialize a and b */
  if ((fd = open("472.ini", O_RDONLY)) < 0) {
    printf("can't open 472.ini\n");
    exit(1);
  }
  read(fd, hin,  IM*IM*sizeof(int));
  printf("reading init_file 1stWORD=%x\n", hin[0]);
  close(fd);

  total_weight=0;
  for (i=-delta; i<=delta; i++) {
    for (j=-delta; j<=delta; j++) {
      hwt[WBASE+i*MAXDELTA*2+j] = delta*delta*4/(abs(i)+abs(j)+1);
      total_weight += (hwt[WBASE+i*MAXDELTA*2+j] = delta*delta*4/(abs(i)+abs(j)+1));
    }
  }
  for (i=-delta; i<=delta; i++) {
    for (j=-delta; j<=delta; j++) {
      hwt[WBASE+i*MAXDELTA*2+j] = hwt[WBASE+i*MAXDELTA*2+j]*256/total_weight;
    }
  }

  /* CPU */
  printf("CPU ");
  reset_nanosec();
  ry = (R+ofs)*IM;
  rx = (R+ofs); /* ofs: from 8 to 14 */
  for (row=PAD; row<OM-PAD; row++) {
    for (col=PAD; col<OM-PAD; col++) {
      c = ((row>>4)*R + (((~row&15)*ofs)>>4))*IM
	+ (col>>4)*R + (((~col&15)*ofs)>>4);
      cvalR=0;
      cvalG=0;
      cvalB=0;
      for (i=-1; i<=1; i++) {
	for (j=-1; j<=1; j++) {
	  Uint pix = hin[c+ry*i+rx*j];
	  w = hwt[WBASE+i*MAXDELTA*2+j];
	  cvalR += ((pix>>24)&255)*w;
	  cvalG += ((pix>>16)&255)*w;
	  cvalB += ((pix>> 8)&255)*w;
	}
      }
      hout0[row*OM+col] = ((cvalR>>8)<<24) | ((cvalG>>8)<<16) | ((cvalB>>8)<<8);
    }
  }

  show_nanosec();

  /* GPU */
  printf("GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(din,  hin,   sizeof(int)*IM*IM, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dwt,  hwt,   sizeof(int)*MAXDELTA*MAXDELTA*2*2, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  dim3 Thread  = dim3(THREADX, THREADY, 1);
  dim3 Block   = dim3(BLOCKX, BLOCKY, 1);
  gather<<<Block,Thread>>>(din, dwt, dout); /* search triangle in {frontier,next} */
  if (cudaSuccess != cudaMemcpy(hout1, dout, sizeof(int)*OM*OM, cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }
  show_nanosec();

#if 1
  for (row=1; row<OM-1; row++) {
    for (col=1; col<OM-1; col++) {
      if (hout0[row*OM+col] != hout1[row*OM+col]) {
	count2++;
	printf("o0[%d]=%x o1[%d]=%x\n",
	       row*OM+col, hout0[row*OM+col],
	       row*OM+col, hout1[row*OM+col]);
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
