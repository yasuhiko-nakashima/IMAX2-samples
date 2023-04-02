
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

int WD=320, HT=240, BITMAP=320*240, SCRWD=5, SCRHT=5, VECWD=240, VECHT=240, VECSTEP=4;

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif
#if !defined(ARMSIML)
#include "./xdisp.c"
#endif

void *gather_kernel();

#define MAXTHNUM 2048
#ifdef PTHREAD
#define THNUM 8
#ifndef ARMSIML
pthread_t th[MAXTHNUM];
#endif
#else
#define THNUM 1
#endif

struct param_kernel {
  int th;
  int v; /* valid */
  int from;
  int to;
} param_kernel[MAXTHNUM];

/****************/
/*** IN/OUT   ***/
/****************/
Uint image_WD, image_HT, image_GRAD;
Uint image_size;
Uint    *ACCI; /* accelerator input */
Uint    *ACCO; /* accelerator output */

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
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // ★★★ RESET
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

/****************/
/*** MAIN     ***/
/****************/
/* 4D-ppm: small-windowサイズは概ね75x75            */
/*         同一点は8x8のsmall-windowの80%領域に存在 */
/*         つまりsmall-windowの有効範囲は64x64      */
/*         7200x5400画像の場合,small-window数は横96,縦72 */
/*         960x720画像とする場合,7.5点で1点を計算   */
/*         960x720画像を10点計算する毎に1-smallwindowずれる */

#define WINSIZE   75
#define MINOFFSET 8
#define MAXOFFSET 14
#define MAXDELTA  4  /* -3,-2,-1,0,1,2,3 */
#define WBASE    (MAXDELTA*MAXDELTA*2)
#ifdef PRECISE_SCALE
  int shift_x = 40;  /* initial shift to x-center of WIN */
  int shift_y = 64;  /* initial shift to y-center of WIN */
#else
  int shift_x = 30;  /* initial shift to x-center of WIN */
  int shift_y = 56;  /* initial shift to y-center of WIN */
#endif
  int offset  = 14;  /* initial offset between WIN */
  int delta;         /* degree of stencil across WIN */
  int smallwin_offset_x;
  int smallwin_offset_y;
  int weight[MAXDELTA*MAXDELTA*2*2];
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
main(argc, argv)
     int argc;
     char **argv;
{
  int i, j, k, fc;

  image_WD = 8;
  image_HT = 1;
  image_size = image_WD*image_HT;
  sysinit((sizeof(int)*image_size)*2, 32);

  printf("membase: %08.8x\n", (Uint)membase);
  ACCI = (Uchar*)((Uchar*)membase);
  ACCO = (Uint*) ((Uchar*)ACCI + (sizeof(int)*image_size));
  printf("ACCI: %08.8x\n", (Uint)ACCI);
  printf("ACCO: %08.8x\n", (Uint)ACCO);

#if !defined(ARMSIML)
  x11_open(0);
#endif

  int x, y;

  gather_x1(ACCI, ACCO);
//EMAX5A drain_dirty_lmm

printf("EMAX6 result:");
  for (x=0; x<image_WD; x++)
    printf(" %08.8x", *(ACCO+x));
  printf("\n");

#ifdef EMAX6
  Uint *lmmp = ((Uchar*)ACCO-(Uchar*)emax_info.ddr_mmap+(Uchar*)emax_info.lmm_mmap);

printf("LMM direct read:");
  for (x=0; x<image_WD; x++)
    printf(" %08.8x", *(lmmp+x));
  printf("\n");

printf("LMM direct write:...\n");
  for (x=0; x<image_WD; x++)
    *(lmmp+x) = x+0x99990000;

printf("LMM direct read:");
  for (x=0; x<image_WD; x++)
    printf(" %08.8x", *(lmmp+x));
  printf("\n");
#endif
}

gather_x1(Uint *yin, Uint *yout)
{
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
  Uint *yout0 = yout;
  Ull  loop = image_WD;
  Ull  AR[16][4];                     /* output of EX     in each unit */
  Ull  BR[16][4][4];                  /* output registers in each unit */
  Ull  r0=0x0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  c0, c1, c2, c3, ex0, ex1;
#if 1
//EMAX5A begin x1 mapdist=0
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
    exe(OP_ADD, &r0,  r0, EXP_H3210,  1LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#0 */
    mop(OP_STWR,  3, &r0, (Ull)(yout++), 0LL, MSK_D0, (Ull)yout0, 320, 0, 1, (Ull)NULL, 320);     /* stage#0 */
  }
//EMAX5A end
#endif

  return(0);
}
