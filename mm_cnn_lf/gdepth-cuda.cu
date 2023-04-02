
char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* GDEPTH                              */
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

void gdepth();
void reset_nanosec();
void show_nanosec();
void reset_time();
void show_time();

#define MINOFFSET 8
#define MAXOFFSET 14
#define IM    7500
#define OM    1600
#define R     75
#define PAD   32
#define TH    137
#define THREADX 8
#define THREADY 8
#define BLOCKX ((OM-PAD*2)/THREADX)
#define BLOCKY ((OM-PAD*2)/THREADY)

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))

__global__ void gdepth(int ofs, int *in, int *out, int *sad)
{
  int c, s, i, j, p0, p1;
  int row, col, rx, ry, y, x;
  row = blockIdx.x*blockDim.x + threadIdx.x + PAD;
  col = blockIdx.y*blockDim.y + threadIdx.y + PAD;
  ry = (R+ofs)*IM;
  rx = (R+ofs); /* ofs: from 8 to 14 */
  c =((row>>4)*R + (((~row&15)*ofs)>>4))*IM
    + (col>>4)*R + (((~col&15)*ofs)>>4);
  s = 0;
  for (y=-1; y<=1; y++) {
    for (x=-1; x<=1; x++) {
      if (x == 0) continue;
      for (i=-1; i<=1; i++) {
	for (j=-1; j<=1; j++) {
	  if (j == 0) continue;
	  p0 = in[c     +(i*IM)     +j]; /* center */
	  p1 = in[c+ry*y+(i*IM)+rx*x+j]; /* comparand */
	  s += dif(p0, p1);
	  if (s > 0xffff) s = 0xffff;
	}
      }
    }
  }
  if (sad[row*OM+col]>TH && s<sad[row*OM+col]) {
    sad[row*OM+col] = s;
    out[row*OM+col] = ofs;
  }
  __syncthreads();
}

int count2;

main(int argc, char **argv)
{
  int fd, *hin, *hout0, *hout1, *hsad0, *hsad1, *din, *dout, *dsad;
  int ofs, c, s, i, j, p0, p1;
  int row, col, rx, ry, y, x;

  hin              = (int*)malloc(sizeof(int) * IM*IM);
  hout0            = (int*)malloc(sizeof(int) * OM*OM);
  hout1            = (int*)malloc(sizeof(int) * OM*OM);
  hsad0            = (int*)malloc(sizeof(int) * OM*OM);
  hsad1            = (int*)malloc(sizeof(int) * OM*OM);
  if (cudaSuccess != cudaMalloc((void**)&din,  sizeof(int) *IM*IM)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dout, sizeof(int) *OM*OM)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dsad, sizeof(int) *OM*OM)) { printf("can't cudaMalloc\n"); exit(1); }

  /* initialize a and b */
  if ((fd = open("472.ini", O_RDONLY)) < 0) {
    printf("can't open 472.ini\n");
    exit(1);
  }
  read(fd, hin,  IM*IM*sizeof(int));
  printf("reading init_file 1stWORD=%x\n", hin[0]);
  read(fd, hout0, OM*OM*sizeof(int));
  read(fd, hout1, OM*OM*sizeof(int));
  read(fd, hsad0, OM*OM*sizeof(int));
  read(fd, hsad1, OM*OM*sizeof(int));
  close(fd);

  /* CPU */
  printf("CPU ");
  reset_nanosec();
  for (ofs=MINOFFSET+1; ofs<=MAXOFFSET-1; ofs++) {
    ry = (R+ofs)*IM;
    rx = (R+ofs); /* ofs: from 8 to 14 */
    for (row=PAD; row<OM-PAD; row++) {
      for (col=PAD; col<OM-PAD; col++) {
	c =((row>>4)*R + (((~row&15)*ofs)>>4))*IM
	  + (col>>4)*R + (((~col&15)*ofs)>>4);
	s = 0;
	for (y=-1; y<=1; y++) {
	  for (x=-1; x<=1; x++) {
	    if (x == 0) continue;
	    for (i=-1; i<=1; i++) {
	      for (j=-1; j<=1; j++) {
		if (j == 0) continue;
		p0 = hin[c     +(i*IM)     +j]; /* center */
		p1 = hin[c+ry*y+(i*IM)+rx*x+j]; /* comparand */
		s += dif(p0, p1);
		if (s > 0xffff) s = 0xffff;
	      }
	    }
	  }
	}
	if (hsad0[row*OM+col]>TH && s<hsad0[row*OM+col]) {
	  hsad0[row*OM+col] = s;
	  hout0[row*OM+col] = ofs;
	}
      }
    }
  }

  show_nanosec();

  /* GPU */
  printf("GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(din,  hin,   sizeof(int)*IM*IM, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dout, hout1, sizeof(int)*OM*OM, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dsad, hsad1, sizeof(int)*OM*OM, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  dim3 Thread  = dim3(THREADX, THREADY, 1);
  dim3 Block   = dim3(BLOCKX, BLOCKY, 1);
  for (ofs=MINOFFSET+1; ofs<=MAXOFFSET-1; ofs++)
    gdepth<<<Block,Thread>>>(ofs, din, dout, dsad); /* search triangle in {frontier,next} */
  if (cudaSuccess != cudaMemcpy(hout1, dout, sizeof(int)*OM*OM, cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(hsad1, dsad, sizeof(int)*OM*OM, cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }
  show_nanosec();

#if 1
  for (row=1; row<OM-1; row++) {
    for (col=1; col<OM-1; col++) {
      if (hsad0[row*OM+col] != hsad1[row*OM+col]) {
	count2++;
	printf("s0[%d]=%x s1[%d]=%x\n",
	       row*OM+col, hsad0[row*OM+col],
	       row*OM+col, hsad1[row*OM+col]);
      }
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
