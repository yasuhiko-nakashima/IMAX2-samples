
char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* string_search                       */
/*        Copyright (C) 2013- by NAIST */
/*         Primary writer: Y.Nakashima */
/*                nakashim@is.naist.jp */

/* ★★★ TX2で実行の際には, root になる必要あり ★★★ */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/stat.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <cuda_runtime.h>

typedef unsigned long long Ull;
typedef unsigned int Uint;
typedef unsigned char Uchar;

void search();
void reset_nanosec();
void show_nanosec();
void reset_time();
void show_time();

#define NCHIP   2
#define OMAP    9
#define BUFLEN  256
#define SSTLEN  8
#define SSTNUM  (OMAP*64)
#define SSTBUF  (SSTNUM*SSTLEN)
#define MAXLEN  (32768-SSTLEN)
#define TXTBUF  (MAXLEN*NCHIP)

#define GPUBL 64
#define GPUTH 1
#define THNUM (GPUBL*GPUTH)

static size_t table[UCHAR_MAX + 1]; /* for ARM */
char    *htarget;  /*fixed -> TBUF*/
int     htlen;     /*file.target-string total_len */
int     hclen;     /*file.target-string current_len */
int     hsnum;     /*file.search-string数*/
char    getbuf[BUFLEN];        /*for fgets()*/
char    hsstr[SSTNUM][SSTLEN]; /*file.search*/
int     hslen[SSTNUM];         /*file.search*/
char    *hout0;    /*arm  results*/
char    *hout1;    /*imax results*/

char    *dtarget;  /*fixed -> TBUF*/
int     *dclen;    /*file.target-string current_len */
int     *dsnum;    /*file.search-string数*/
char    *dsstr;    /*file.search*/
int     *dslen;    /*file.search*/
char    *dout;

void init_search(int i)/* for ARM */
{
  char *str = (char*)hsstr[i];
  int  len  = hslen[i];
  int  j;

  for (j = 0; j <= UCHAR_MAX; j++)
    table[j] = len;
  for (j = 0; j < len; j++)
    table[(Uchar)str[j]] = len - j - 1;
  for (j = 0; j < hclen; j++)
    *(hout0+hclen*i+j) = 0;
}

void strsearch(int i)
{
  char *str = (char*)hsstr[i];
  int  len  = hslen[i];
  register size_t shift;
  register size_t pos = len - 1;
  
  /*printf("%s len=%d clen=%d\n", str, len, clen);*/

  while (pos < hclen) {
    while (pos < hclen && (shift = table[(unsigned char)htarget[pos]]) > 0)
      pos += shift;
    if (!shift) {
      if (!strncmp(str, &htarget[pos-len+1], len))
	hout0[i*hclen+(pos-len+1)] = 0xff;
      pos++;
    }
  }
}

void search_CPU()
{
  int i;

  for (i=0; i<hsnum; i++) {
    init_search(i);
    strsearch(i);
  }
}

__global__ void search_GPU(char *out, char *target, int *clen, int *snum, char *sstr, int *slen)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x; /* 0-63 */
  int i, j, k;
  int i0 = (tid==0)?0:(tid)**snum/THNUM;
  int i1 =          (tid+1)**snum/THNUM;

  for (i=i0; i<i1; i++) {
    for (j=0; j<*clen; j++) {
      for (k=0; k<slen[i]; k++) {
	if (sstr[i*SSTLEN+k] != target[j+k])
	  break;
      }
      if (k == slen[i]) {
	out[i**clen+j] = 0xff;
	/*printf(" %s %s [%d][%d]=1\n", sstr[i], target+j, i, j);*/
      }
      else {
	out[i**clen+j] = 0;
	/*printf(" %s %s [%d][%d]=0\n", sstr[i], target+j, i, j);*/
      }
    }
  }
  __syncthreads();
}

char* membase;
int   count2;

main(int argc, char **argv)
{
  char   *fr0=NULL, *fr1=NULL;
  FILE   *fp;
  int    fdi, fdo[SSTNUM];
  struct stat sb;
  int    i, j;

  if (argc != 3) {
    printf("usage: %s <file(search_string)> <file(target)>\n", *argv);
    printf("       output_file: target_file.xxxx\n");
    exit(1);
  }

  for (argc--, argv++; argc; argc--, argv++) {
    if      (!fr0) strcpy(fr0 = (char*)malloc(strlen(*argv) + 1), *argv);
    else if (!fr1) strcpy(fr1 = (char*)malloc(strlen(*argv) + 1), *argv);
  }

  if ((fp = fopen(fr0,  "r")) == NULL) {
    printf("can't open %s (search_string)\n", fr0);
    exit(1);
  }

  while (hsnum < SSTNUM && fgets(getbuf, BUFLEN, fp)) {
    int len = strlen(getbuf);
    if (getbuf[len-1] == '\n') {
      getbuf[len-1] = 0;
      len--;
    }
    if (len) {
      strncpy(hsstr[hsnum], getbuf, SSTLEN);
      hslen[hsnum] = len<SSTLEN?len:SSTLEN;
      /*printf("sstr[%d]=%s\n", hsnum, hsstr[hsnum]);*/
      hsnum++;
    }
  }
  printf("total search_strings=%d\n", hsnum);

  if ((fdi = open(fr1,  O_RDONLY)) < 0) {
    printf("can't open %s (search_target)\n", fr1);
    exit(1);
  }
  if (fstat(fdi, &sb) < 0) {
    printf("can't get fstat of %s (errno=%d)\n", fr1, errno);
    exit(1);
  }

  htlen = sb.st_size; /* total length of target */
  printf("total_size of target is %d bytes\n", htlen);

  Uint memsize   = TXTBUF+TXTBUF*SSTNUM+TXTBUF*SSTNUM;
  membase = (char*)malloc(memsize);
  printf("membase: %16.16lx\n", membase);
  htarget = (char*)membase;
  hout0   = (char*)htarget + TXTBUF;
  hout1   = (char*)hout0   + TXTBUF*SSTNUM;
  printf("target:  %16.16lx\n", htarget);
  printf("out0:    %16.16lx\n", hout0);
  printf("out1:    %16.16lx\n", hout1);

  while ((hclen = htlen<TXTBUF ? htlen : TXTBUF)>0) {
    if (read(fdi, htarget, hclen) != hclen) {
      printf("can't read target (errno=%d)\n", errno);
      exit(1);
    }
    hclen = (hclen + (SSTLEN-1))&~(SSTLEN-1);

    if (cudaSuccess != cudaMalloc((void**)&dtarget, hclen))        { printf("can't cudaMalloc\n"); exit(1); }
    if (cudaSuccess != cudaMalloc((void**)&dclen,   sizeof(int)))  { printf("can't cudaMalloc\n"); exit(1); }
    if (cudaSuccess != cudaMalloc((void**)&dsnum,   sizeof(int)))  { printf("can't cudaMalloc\n"); exit(1); }
    if (cudaSuccess != cudaMalloc((void**)&dsstr,   hsnum*SSTLEN)) { printf("can't cudaMalloc\n"); exit(1); }
    if (cudaSuccess != cudaMalloc((void**)&dslen,   hsnum*4))      { printf("can't cudaMalloc\n"); exit(1); }
    if (cudaSuccess != cudaMalloc((void**)&dout,    hclen*hsnum))  { printf("can't cudaMalloc\n"); exit(1); }

    /* CPU */
    printf("CPU ");
    reset_nanosec();
    search_CPU();
    show_nanosec();

    /* GPU */
    printf("GPU ");
    reset_nanosec();
    if (cudaSuccess != cudaMemcpy(dtarget, htarget, hclen,        cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy1\n"); exit(1); }
    if (cudaSuccess != cudaMemcpy(dclen,   &hclen,  sizeof(int),  cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy2\n"); exit(1); }
    if (cudaSuccess != cudaMemcpy(dsnum,   &hsnum,  sizeof(int),  cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy3\n"); exit(1); }
    if (cudaSuccess != cudaMemcpy(dsstr,   hsstr,   hsnum*SSTLEN, cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy4\n"); exit(1); }
    if (cudaSuccess != cudaMemcpy(dslen,   hslen,   hsnum*4,      cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy5\n"); exit(1); }
    search_GPU<<<GPUBL,GPUTH>>>(dout, dtarget, dclen, dsnum, dsstr, dslen); /* search triangle in {frontier,next} */
    if (cudaSuccess != cudaMemcpy(hout1,   dout,    hclen*hsnum,  cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy6\n"); exit(1); }
    show_nanosec();

#if 1
    /* display the result */
    int hltop[SSTNUM];
    int highlight[SSTNUM];
    int hlold=0, hlnew;
    for (j=0; j<hsnum; j++)
      highlight[j] = 0;
    for (i=0; i<hclen; i++) {
      for (j=0; j<hsnum; j++) {
	if (*(hout1+hclen*j+i)) {
	  highlight[j] = 1;
	  hltop[j] = i;
	}
	if (highlight[j] && hltop[j]+hslen[j] == i)
	  highlight[j] = 0;
      }
      hlnew = 0;
      for (j=0; j<hsnum; j++) {
	if (highlight[j]) {
	  hlnew = 1;
	  break;
	}
      }
      if (!hlold && hlnew)
	printf("\033[7m");
      else if (hlold && !hlnew)
	printf("\033[0m");
      printf("%c", htarget[i]);
      hlold = hlnew;
    }    
    printf("\n");
#endif

#if 1
    /* compare the result */
    for (i=0; i<hsnum; i++) {
      for (j=0; j<hclen; j++) {
	if (*(hout0+hclen*i+j) != *(hout1+hclen*i+j)) {
	  count2++;
	  printf("o0[%d][%d]=%x o1[%d][%d]=%x\n",
		 i, j, *(hout0+hclen*i+j), i, j, *(hout1+hclen*i+j));
	}
      }
    }
    if (count2)
      printf("Num of diffs: %d\n", count2);
    else
      printf("Results are equal\n");
    show_nanosec();
#endif

    htlen -= hclen;
  }

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
