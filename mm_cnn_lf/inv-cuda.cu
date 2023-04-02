
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

void inv();
void reset_nanosec();
void show_nanosec();
void reset_time();
void show_time();

#define M 256
#define ERRTH  (5.0E-4)
#define THREADY 64
#define BLOCKY (M/THREADY)

__global__ void inv(float *A, Uint *p, float *b, float *x, float *inv)
{
  int row, i, j, k;
  float pmax;
  i = blockIdx.y*blockDim.y + threadIdx.y;

  /* LU分解 */
  /*for (i=0; i<M; i++)*/
  p[i] = i;
  for (row=0; row<M; row++) { /* 列方向 */
    if (i==0) {
      pmax = 0.0;
      k = -1;
      for (j=row; j<M; j++) { /* 行方向に探索 */
	if (pmax < fabsf(A[p[j]*M+row])) {
	  pmax = fabsf(A[p[j]*M+row]);
	  k = j;
	}
      }
      if (k == -1) {
	/*fprintf(stderr, "can't solve\n");*/
	return;
      }
      j = p[k]; p[k] = p[row]; p[row] = j;
      A[p[row]*M+row] = 1.0/A[p[row]*M+row];
      for (j=row+1; j<M; j++) /* 行方向 */
	A[p[j]*M+row] *= A[p[row]*M+row];
    }
    /* FPGA実機でj-loopの最終(len=1)が動かないので,ついでにARMのほうが速そうなlenをARMで実行 2019/3/1 Nakashima */
  /*for (j=row+1; j<M; j++) {*/
      if (i>=row+1 && i<M) {
	for (k=0; k<M-(row+1); k++) { /* 最内列方向 */
	  A[p[i]*M+row+1+k] -= A[p[i]*M+row]*A[p[row]*M+row+1+k];
	}
      }
  /*}*/
  }

  /* 逆行列前半 */
/*for (i=0; i<M; i++) {*/
    for (j=0; j<M; j++) /* 行方向 */
      b[i*M+j] = (i==j)?1.0:0.0;
/*}*/
/*for (i=0; i<M; i++) {*/
    for (j=i+1; j<M; j++) { /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでj=i+1から開始 */
      /********************************************/
      for (k=i; k<j; k++) { /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでk=iから開始 */
	b[i*M+j] -= A[p[j]*M+k]*b[i*M+k];
      }
      /********************************************/
    } /* j-loop */
/*}*/

  /* 逆行列後半 */
/*for (i=0; i<M; i++) {*/
    for (j=M-1; j>=0; j--) { /* 行方向 */
      if (j<M-1) {
	/********************************************/
	for (k=M-1; k>j; k--) { /* 最内列方向 */
	  b[i*M+j] -= A[p[j]*M+k]*x[i*M+k];
	}
        /********************************************/
      } /* if (j<M-1) */
      inv[j*M+p[i]] = x[i*M+j] = A[p[j]*M+j]*b[i*M+j];
    } /* j-loop */
/*}*/
    __syncthreads();
}

int count2;

main(int argc, char **argv)
{
  float *hA0, *hA; Uint *hp; float *hinv, *hb, *hx, *hC;
  float       *dA; Uint *dp; float *dinv, *db, *dx, *dC;
  int row, col, k;
  float pmax;

  hA0              = (float*)malloc(sizeof(float) * M * M);
  hA               = (float*)malloc(sizeof(float) * M * M);
  hp               = (Uint*) malloc(sizeof(Uint) * M);
  hinv             = (float*)malloc(sizeof(float) * M * M);
  hb               = (float*)malloc(sizeof(float) * M * M);
  hx               = (float*)malloc(sizeof(float) * M * M);
  hC               = (float*)malloc(sizeof(float) * M * M);
  if (cudaSuccess != cudaMalloc((void**)&dA,   sizeof(float) * M * M)) { printf("can't cudaMalloc1\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dp,   sizeof(Uint) * M))      { printf("can't cudaMalloc2\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dinv, sizeof(float) * M * M)) { printf("can't cudaMalloc3\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&db,   sizeof(float) * M * M)) { printf("can't cudaMalloc4\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dx,   sizeof(float) * M * M)) { printf("can't cudaMalloc5\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dC,   sizeof(float) * M * M)) { printf("can't cudaMalloc6\n"); exit(1); }

  /* initialize a and b */
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++)
      hA[row*M+col] = hA0[row*M+col] = (float)(row%120+col);
  }
  hA[0] = hA0[0] = 1;
  for (col=1;col<M;col++)
    hA[col*M+col] = hA0[col*M+col] = 3;

  /* CPU */
  printf("CPU ");
  reset_nanosec();
  /* LU分解 */
  for (row=0; row<M; row++)
    hp[row] = row;
  for (row=0; row<M; row++) {
    pmax = 0.0;
    k = -1;
    for (col=row; col<M; col++) {
      if (pmax < fabsf(hA[hp[col]*M+row])) {
	pmax = fabsf(hA[hp[col]*M+row]);
	k = col;
      }
    }
    if (k == -1) {
      fprintf(stderr, "can't solve\n");
      exit(1);
    }
    col = hp[k]; hp[k] = hp[row]; hp[row] = col;
    hA[hp[row]*M+row] = 1.0/hA[hp[row]*M+row];
    for (col=row+1; col<M; col++) {
      hA[hp[col]*M+row] *= hA[hp[row]*M+row];
      for (k=row+1; k<M; k++)
	hA[hp[col]*M+k] -= hA[hp[col]*M+row]*hA[hp[row]*M+k];
    }
  }

  /* 逆行列前半 */
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++)
      hb[row*M+col] = (row==col)?1.0:0.0;
    /*for (col=1; col<M; col++) { *//*通常の連立一時方程式の場合*/
    for (col=row+1; col<M; col++) { /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでj=i+1から開始 */
      /*for (k=0; k<col; k++) *//*通常の連立一時方程式の場合*/
      for (k=row; k<col; k++) /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでk=iから開始 */
	hb[row*M+col] -= hA[hp[col]*M+k]*hb[row*M+k];
    }
  }

  /* 逆行列後半 */
  for (row=0; row<M; row++) {
    for (col=M-1; col>=0; col--) {
      for (k=M-1; k>col; k--)
	hb[row*M+col] -= hA[hp[col]*M+k]*hx[row*M+k];
      hinv[col*M+hp[row]] = hx[row*M+col] = hb[row*M+col]*hA[hp[col]*M+col];
    }
  }
  show_nanosec();

#if 0
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++)
      printf(" %f", hinv[row*M+col]);
    printf("\n");
  }
#endif

#if 1
  /* 検算 */
  count2 = 0;
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      for (k=0; k<M; k++) {
        if (k==0) hC[row*M+col]  = hA0[row*M+k] * hinv[k*M+col];
        else      hC[row*M+col] += hA0[row*M+k] * hinv[k*M+col];
      }
      if (row == col && fabsf(hC[row*M+col]-1.0)>ERRTH) {
	count2++;
	printf("A*A'!=E C[%d][%d]=%f\n", row, col, hC[row*M+col]);
      }
      else if (row != col && (fabsf(hC[row*M+col])>ERRTH)) {
	count2++;
	printf("A*A'!=E C[%d][%d]=%f\n", row, col, hC[row*M+col]);
      }
    }
  }
  if (count2)
    printf("A*A'!=E (ERRTH=%f) Num of diffs: %d\n", ERRTH, count2);
  else
    printf("A*A'==E (ERRTH=%f) Confirmed\n", ERRTH);
#endif

  /* GPU */
  printf("GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(dA, hA0,    sizeof(float)*M*M, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  dim3 Thread  = dim3(1, THREADY, 1);
  dim3 Block   = dim3(1, BLOCKY, 1);
  inv<<<Block,Thread>>>(dA, dp, db, dx, dinv); /* search triangle in {frontier,next} */
  if (cudaSuccess != cudaMemcpy(hinv, dinv, sizeof(float)*M*M, cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }
  show_nanosec();

#if 0
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++)
      printf(" %f", hinv[row*M+col]);
    printf("\n");
  }
#endif

#if 1
  /* 検算 */
  count2 = 0;
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      for (k=0; k<M; k++) {
        if (k==0) hC[row*M+col]  = hA0[row*M+k] * hinv[k*M+col];
        else      hC[row*M+col] += hA0[row*M+k] * hinv[k*M+col];
      }
      if (row == col && fabsf(hC[row*M+col]-1.0)>ERRTH) {
	count2++;
	printf("A*A'!=E C[%d][%d]=%f\n", row, col, hC[row*M+col]);
      }
      else if (row != col && (fabsf(hC[row*M+col])>ERRTH)) {
	count2++;
	printf("A*A'!=E C[%d][%d]=%f\n", row, col, hC[row*M+col]);
      }
    }
  }
  if (count2)
    printf("A*A'!=E (ERRTH=%f) Num of diffs: %d\n", ERRTH, count2);
  else
    printf("A*A'==E (ERRTH=%f) Confirmed\n", ERRTH);
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
