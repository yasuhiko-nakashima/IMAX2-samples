
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/* Gather data from light-field-camera and display image */
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

#undef PRECISE_SCALE

#ifndef ARMSIML
#define _POSIX_THREADS
#endif

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct {Ull u[2];} Dll;
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/Xdbe.h>
#endif

#if defined(EMAX5)
#include "../../src/conv-c2b/emax5.h"
#include "../../src/conv-c2b/emax5lib.c"
#endif

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif

void *gather_kernel();
#define abs(a) (((a)<0)?-(a):(a))

/****************/
/*** IN/OUT   ***/
/****************/
float *A, *B, *C, *D;

#define XSIZE 64
#define YSIZE 16

void *kernel_top();
void *kernel_cgra();

/****************/
/*** ZYNQ     ***/
/****************/

Uchar* membase;

sysinit(memsize, alignment) Uint memsize, alignment;
{
#if defined(ARMZYNQ) && defined(EMAX5)
  if (emax5_open() == NULL)
    exit(1);
  membase = emax_info.hpp_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif __linux__ == 1
  posix_memalign(&membase, alignment, memsize);
#else
  membase = (void*)malloc(memsize+alignment);
  if ((int)membase & (alignment-1))
    membase = (void*)(((int)membase & ~(alignment-1))+alignment);
#endif

#if !defined(ARMZYNQ) && defined(EMAX5)
  emax_info.hpp_phys = membase;
  emax_info.hpp_mmap = emax_info.hpp_phys;
  emax_info.acp_phys = ACP_BASE2_PHYS; /* defined in emax5lib.h >= ALOCLIMIT */
  emax_info.acp_mmap = emax_info.acp_phys;
#endif
#if defined(EMAX5)
  acp_conf = emax_info.acp_mmap; /* 8KB * 256sets */
  acp_lmmi = emax_info.acp_mmap + 0x200000;
  acp_regv = emax_info.acp_mmap + 0x304000;
#endif

#if !defined(ARMZYNQ) && defined(EMAX6)
  emax_info.dma_phys = DMA_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.dma_mmap = emax_info.dma_phys;
  emax_info.reg_phys = REG_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.reg_mmap = emax_info.reg_phys;
  emax_info.lmm_phys = LMM_BASE2_PHYS;
  emax_info.lmm_mmap = emax_info.lmm_phys;
  emax_info.ddr_phys = membase;
  emax_info.ddr_mmap = emax_info.ddr_phys;
#endif
#if (defined(ARMSIML) || defined(ARMZYNQ)) && defined(EMAX6)
  emax6.dma_ctrl  = emax_info.dma_mmap;
  emax6.reg_ctrl  = emax_info.reg_mmap;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // ’¡ú’¡ú’¡ú RESET
#if defined(ARMZYNQ)
  usleep(1);
#endif
  switch (((struct reg_ctrl*)emax6.reg_ctrl)->i[0].stat>>8 & 0xf) {
  case  3:EMAX_DEPTH = 64;break;
  case  2:EMAX_DEPTH = 32;break;
  case  1:EMAX_DEPTH = 16;break;
  default:EMAX_DEPTH =  8;break;
  }
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].adtr = emax_info.ddr_mmap - emax_info.lmm_phys;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].dmrp = 0LL;
#endif
}

int main(int argc, char *argv[])
{
  sysinit((sizeof(float)*XSIZE*YSIZE*4), 32);

  printf("membase: %08.8x\n", (Uint)membase);
  A = (float*)membase;
  B = A + XSIZE*YSIZE;
  C = B + XSIZE*YSIZE;
  D = C + XSIZE*YSIZE;
  printf("A addr:0x%08.8x\n",(Uint)A );
  printf("B addr:0x%08.8x\n",(Uint)B );
  printf("C addr:0x%08.8x\n",(Uint)C );
  printf("D addr:0x%08.8x\n",(Uint)D );

  Uint x, y;
  for (y=0; y<YSIZE; y++) {
    for (x=0; x<XSIZE; x++) {
      A[y*XSIZE+x] = (float)(y*XSIZE+x);
      B[y*XSIZE+x] = 0.1;
      C[y*XSIZE+x] = (float)(y*XSIZE+x);
      D[y*XSIZE+x] = 0;
    }
  }

  printf("array A:\n");
  print_array(A);
  printf("array B:\n");
  print_array(B);
  printf("array C:\n");
  print_array(C);

  kernel_cgra();

  printf("array D:\n");
  print_array(D);

  return 0;
}

void *kernel_cgra() /* for EMAX func */
{
  int y;

  for (y=0; y<YSIZE; y++) {
    Ull r0[4], r1[4], r2[4], r3[4];
    Ull *base_a = (Ull*)((Ull)A+y*sizeof(float)*XSIZE);
    Ull *base_b = (Ull*)((Ull)B+y*sizeof(float)*XSIZE);
    Ull *base_c = (Ull*)((Ull)C+y*sizeof(float)*XSIZE);
    Ull *base_d = (Ull*)((Ull)D+y*sizeof(float)*XSIZE);
    Ull *base_d0 = base_d;
    Ull *base_d1 = base_d0+XSIZE/8;
    Ull *base_d2 = base_d1+XSIZE/8;
    Ull *base_d3 = base_d2+XSIZE/8;
    Ull loop   = XSIZE/8/4;

    struct {Ull u[4];} *top_a0  = base_a;
    struct {Ull u[4];} *top_a1  = base_a;
    struct {Ull u[4];} *top_a2  = base_a;
    struct {Ull u[4];} *top_a3  = base_a;
    struct {Ull u[4];} *top_b0  = base_b;
    struct {Ull u[4];} *top_b1  = base_b;
    struct {Ull u[4];} *top_b2  = base_b;
    struct {Ull u[4];} *top_b3  = base_b;
    struct {Ull u[4];} *top_c0  = base_c;
    struct {Ull u[4];} *top_c1  = base_c;
    struct {Ull u[4];} *top_c2  = base_c;
    struct {Ull u[4];} *top_c3  = base_c;
    struct {Ull u[4];} *top_d0  = base_d0;
    struct {Ull u[4];} *top_d1  = base_d1;
    struct {Ull u[4];} *top_d2  = base_d2;
    struct {Ull u[4];} *top_d3  = base_d3;

//EMAX5A begin fma mapdist=0
    while(loop--) {
      mo4(OP_LDRQ, 3, r0, (Ull)(top_a0++), 0LL, MSK_D0, (Ull)base_a, XSIZE, 0, 0, NULL, XSIZE);
      mo4(OP_LDRQ, 3, r1, (Ull)(top_b0++), 0LL, MSK_D0, (Ull)base_b, XSIZE, 0, 0, NULL, XSIZE);
      mo4(OP_LDRQ, 3, r2, (Ull)(top_c0++), 0LL, MSK_D0, (Ull)base_c, XSIZE, 0, 0, NULL, XSIZE);
      ex4(OP_FMA, r3, r0, EXP_H3210, r1, EXP_H3210, r2, EXP_H3210, OP_NOP, 0LL,  OP_NOP, 0LL);
      mo4(OP_STRQ, 3, r3, (Ull)(top_d0++), 0LL, MSK_D0, (Ull)base_d0, XSIZE/4, 0, 1, NULL, XSIZE/4);
      mo4(OP_LDRQ, 3, r0, (Ull)(top_a1++), 0LL, MSK_D0, (Ull)base_a, XSIZE, 0, 0, NULL, XSIZE);
      mo4(OP_LDRQ, 3, r1, (Ull)(top_b1++), 0LL, MSK_D0, (Ull)base_b, XSIZE, 0, 0, NULL, XSIZE);
      mo4(OP_LDRQ, 3, r2, (Ull)(top_c1++), 0LL, MSK_D0, (Ull)base_c, XSIZE, 0, 0, NULL, XSIZE);
      ex4(OP_FMA, r3, r0, EXP_H3210, r1, EXP_H3210, r2, EXP_H3210, OP_NOP, 0LL,  OP_NOP, 0LL);
      mo4(OP_STRQ, 3, r3, (Ull)(top_d1++), 0LL, MSK_D0, (Ull)base_d1, XSIZE/4, 0, 1, NULL, XSIZE/4);
      mo4(OP_LDRQ, 3, r0, (Ull)(top_a2++), 0LL, MSK_D0, (Ull)base_a, XSIZE, 0, 0, NULL, XSIZE);
      mo4(OP_LDRQ, 3, r1, (Ull)(top_b2++), 0LL, MSK_D0, (Ull)base_b, XSIZE, 0, 0, NULL, XSIZE);
      mo4(OP_LDRQ, 3, r2, (Ull)(top_c2++), 0LL, MSK_D0, (Ull)base_c, XSIZE, 0, 0, NULL, XSIZE);
      ex4(OP_FMA, r3, r0, EXP_H3210, r1, EXP_H3210, r2, EXP_H3210, OP_NOP, 0LL,  OP_NOP, 0LL);
      mo4(OP_STRQ, 3, r3, (Ull)(top_d2++), 0LL, MSK_D0, (Ull)base_d2, XSIZE/4, 0, 1, NULL, XSIZE/4);
      mo4(OP_LDRQ, 3, r0, (Ull)(top_a3++), 0LL, MSK_D0, (Ull)base_a, XSIZE, 0, 0, NULL, XSIZE);
      mo4(OP_LDRQ, 3, r1, (Ull)(top_b3++), 0LL, MSK_D0, (Ull)base_b, XSIZE, 0, 0, NULL, XSIZE);
      mo4(OP_LDRQ, 3, r2, (Ull)(top_c3++), 0LL, MSK_D0, (Ull)base_c, XSIZE, 0, 0, NULL, XSIZE);
      ex4(OP_FMA, r3, r0, EXP_H3210, r1, EXP_H3210, r2, EXP_H3210, OP_NOP, 0LL,  OP_NOP, 0LL);
      mo4(OP_STRQ, 3, r3, (Ull)(top_d3++), 0LL, MSK_D0, (Ull)base_d3, XSIZE/4, 0, 1, NULL, XSIZE/4);
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm
}

print_array(float *array)
{
  int x, y;
  for (y=0; y<YSIZE; y++) {
    for (x=0; x<XSIZE; x++) {
      //printf("%08.2f/%08.8x ", (float)array[y*XSIZE+x].f, (Uint)array[y*XSIZE+x].i); /* format: float | Uint */
      //printf("%08.8x ", (Uint)array[y*XSIZE+x].i); /* format: Uint */
      //printf("%010u ", (Uint)array[y*XSIZE+x].i); /* format: Uint */
      printf("%5.1f ", array[y*XSIZE+x]); /* format: float */
    }
    printf("\n");
  }
}
