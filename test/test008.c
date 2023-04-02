
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
#define abs(a) (((a)<0)?-(a):(a))

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
  FILE *fp;
  int i, j, k, fc;
  char dummy[16];
#ifndef ARMSIML
  fd_set rfds;
  struct timeval tv;
  char cmd[1024];
#endif

  image_WD = 18;
  image_HT = 8;
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

  for (i=0; i<image_HT; i++) {
    for (j=0; j<image_WD; j++) {
      ACCI[i*image_WD+j] = i*65536 + j;
    }
  }

  while (1) {
    int x, y, dx, dy;
    int cvalR, cvalG, cvalB;
                                          /*                    8  9 10 11 12 13 14  */
#ifdef ARMSIML
    _getpa();
#endif
    gather_kernel(); /* search triangle in {frontier,next} */
#ifdef ARMSIML
    _getpa();
    _copyX(0, ACCI);
    _copyX(1, ACCO);
    _copyX(2, ACCI);
    _updateX();
#endif
#if !defined(ARMSIML)
    BGR_to_X(0, ACCI);
    BGR_to_X(1, ACCO);
    BGR_to_X(2, ACCI);
    x11_update();
#endif
    break;
  }

#if !defined(ARMSIML)
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (x11_checkevent());
#endif

  exit(0);
}

void *gather_kernel()
{
  int x, y;
  Uint *LMM;
  for (y=1; y<image_HT-1; y++)
    gather_x1(ACCI+y*image_WD, ACCO+y*image_WD);
//EMAX5A drain_dirty_lmm
//emax5_drain_dirty_lmm();
  for (x=0; x<image_WD; x++)
    printf(" %08.8x", *(ACCO+         (1)*image_WD+x));
  printf("\n");
  for (x=0; x<image_WD; x++)
    printf(" %08.8x", *(ACCO+(image_HT-2)*image_WD+x));
  printf("\n");

#if !defined(ARMZYNQ) && defined(EMAX6)
  LMM = (Uint)ACCI-emax_info.ddr_mmap+emax_info.lmm_mmap;
  printf("R/W test start\n");
  printf("writing LMM\n");
  for (x=0; x<image_WD*image_HT; x++) {
    printf(" %08.8x", x);
    if (x%image_WD == image_WD-1)
      printf("\n");
    *(LMM+x) = x;
  }
  LMM = (Uint)ACCO-emax_info.ddr_mmap+emax_info.lmm_mmap;
  printf("reading LMM\n");
  for (x=0; x<image_WD*image_HT; x++) {
    printf(" %08.8x", *(LMM+x));
    if (x%image_WD == image_WD-1)
      printf("\n");
  }
#endif
}

gather_x1(Uint *yin, Uint *yout)
{
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
  Ull  yin_mm = yin-image_WD-2;
  Ull  yin_m  = yin-image_WD;
  Ull  yin_mp = yin-image_WD+2;
  Ull  yin_zm = yin-2;
  Ull  yin_z  = yin;
  Ull  yin_zp = yin+2;
  Ull  yin_pm = yin+image_WD-2;
  Ull  yin_p  = yin+image_WD;
  Ull  yin_pp = yin+image_WD+2;
  Ull  yin_q  = yin+image_WD*2;
  Ull  *yout_z1 = yout+2;
  Ull  *yout_z2 = yout+2;
  Ull  *yout_w2 = yout-image_WD+2;
  Ull  loop = image_WD/2-2;
  Ull  x = 0;
  Ull  AR[16][4];                     /* output of EX     in each unit */
  Ull  BR[16][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  c0, c1, c2, c3, ex0, ex1;
  /* 以下の例では，lmpとlmdが干渉し,lmdだけがlmmiに残る */
  /* しかし，lmpがないと事前loadが必要なため，EXに先立ち,lmd追い出しが実行され, pipeline動作しない */
#if 0
//EMAX5A begin x1 mapdist=1
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
    exe(OP_ADD,  &x,             x,  EXP_H3210,         8LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,  OP_NOP, 0LL); /* stage#0 */
    mop(OP_LDR,  3, &BR[1][0][1],  (Ull)yin_mm, x, MSK_D0, (Ull)yin_m, 320, 0, 0, (Ull)NULL, 320);        /* stage#1 */
    mop(OP_LDR,  3, &BR[1][0][0],  (Ull)yin_mp, x, MSK_D0, (Ull)yin_m, 320, 0, 0, (Ull)NULL, 320);        /* stage#1 */
    mop(OP_LDR,  3, &BR[2][0][1],  (Ull)yin_zm, x, MSK_D0, (Ull)yin_z, 320, 0, 0, (Ull)NULL, 320);        /* stage#2 */
    mop(OP_LDR,  3, &BR[2][0][0],  (Ull)yin_zp, x, MSK_D0, (Ull)yin_z, 320, 0, 0, (Ull)NULL, 320);        /* stage#2 */
    mop(OP_LDR,  3, &BR[3][0][1],  (Ull)yin_pm, x, MSK_D0, (Ull)yin_p, 320, 0, 0, (Ull)yin_q, 320);       /* stage#3 */
    mop(OP_LDR,  3, &BR[3][0][0],  (Ull)yin_pp, x, MSK_D0, (Ull)yin_p, 320, 0, 0, (Ull)yin_q, 320);       /* stage#3 */
    exe(OP_ADD3, &r10, BR[1][0][0], EXP_H3210,  BR[2][0][0], EXP_H3210,  BR[3][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
    exe(OP_ADD3, &r11, BR[1][0][1], EXP_H3210,  BR[2][0][1], EXP_H3210,  BR[3][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
    exe(OP_ADD, &r12,  r10, EXP_H3210,  r11, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#5 */
    mop(OP_STR,  3, &r12, (Ull)(yout_z1++), 0LL, MSK_D0, (Ull)yout_z2, 316, 0, 0, (Ull)yout_w2, 316);     /* stage#5 */
  }
//EMAX5A end
#endif
#if 0
//EMAX5A begin x1 mapdist=1
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
    exe(OP_ADD,  &x,             x,  EXP_H3210,         8LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,  OP_NOP, 0LL); /* stage#0 */
    mop(OP_LDR,  3, &BR[1][0][1],  (Ull)yin_mm, x, MSK_D0, (Ull)yin_m, 320, 0, 0, (Ull)NULL, 320);        /* stage#1 */
    mop(OP_LDR,  3, &BR[1][0][0],  (Ull)yin_mp, x, MSK_D0, (Ull)yin_m, 320, 0, 0, (Ull)NULL, 320);        /* stage#1 */
    mop(OP_LDR,  3, &BR[2][0][1],  (Ull)yin_zm, x, MSK_D0, (Ull)yin_z, 320, 0, 0, (Ull)NULL, 320);        /* stage#2 */
    mop(OP_LDR,  3, &BR[2][0][0],  (Ull)yin_zp, x, MSK_D0, (Ull)yin_z, 320, 0, 0, (Ull)NULL, 320);        /* stage#2 */
    mop(OP_LDR,  3, &BR[3][0][1],  (Ull)yin_pm, x, MSK_D0, (Ull)yin_p, 320, 0, 0, (Ull)yin_q, 320);       /* stage#3 */
    mop(OP_LDR,  3, &BR[3][0][0],  (Ull)yin_pp, x, MSK_D0, (Ull)yin_p, 320, 0, 0, (Ull)yin_q, 320);       /* stage#3 */
    exe(OP_ADD3, &AR[5][0], BR[1][0][0], EXP_H3210,  BR[2][0][0], EXP_H3210,  BR[3][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_ADD3, &AR[5][1], BR[1][0][1], EXP_H3210,  BR[2][0][1], EXP_H3210,  BR[3][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_ADD, &AR[6][0],  AR[5][0], EXP_H3210,  AR[5][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#6 */
    mop(OP_STR,  3, &AR[6][0], (Ull)(yout_z1++), 0LL, MSK_D0, (Ull)yout_z2, 316, 0, 0, (Ull)yout_w2, 316);     /* stage#6 */
  }
//EMAX5A end
#endif
#if 1
//EMAX5A begin x1 mapdist=1
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
    exe(OP_ADD,  &x,             x,  EXP_H3210,         8LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,  OP_NOP, 0LL); /* stage#0 */
    mop(OP_LDR,  3, &BR[1][0][1],  (Ull)yin_mm, x, MSK_D0, (Ull)yin_m, 18, 0, 0, (Ull)NULL, 18);        /* stage#1 */
    mop(OP_LDR,  3, &BR[1][0][0],  (Ull)yin_mp, x, MSK_D0, (Ull)yin_m, 18, 0, 0, (Ull)NULL, 18);        /* stage#1 */
    mop(OP_LDR,  3, &BR[2][0][1],  (Ull)yin_zm, x, MSK_D0, (Ull)yin_z, 18, 0, 0, (Ull)NULL, 18);        /* stage#2 */
    mop(OP_LDR,  3, &BR[2][0][0],  (Ull)yin_zp, x, MSK_D0, (Ull)yin_z, 18, 0, 0, (Ull)NULL, 18);        /* stage#2 */
    mop(OP_LDR,  3, &BR[3][0][1],  (Ull)yin_pm, x, MSK_D0, (Ull)yin_p, 18, 0, 0, (Ull)yin_q, 18);       /* stage#3 */
    mop(OP_LDR,  3, &BR[3][0][0],  (Ull)yin_pp, x, MSK_D0, (Ull)yin_p, 18, 0, 0, (Ull)yin_q, 18);       /* stage#3 */
    exe(OP_ADD3, &r10, BR[1][0][0], EXP_H3210,  BR[2][0][0], EXP_H3210,  BR[3][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
    exe(OP_ADD3, &r11, BR[1][0][1], EXP_H3210,  BR[2][0][1], EXP_H3210,  BR[3][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
    exe(OP_ADD, &AR[6][0],  r10, EXP_H3210,  r11, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#6 */
    mop(OP_STR,  3, &AR[6][0], (Ull)(yout_z1++), 0LL, MSK_D0, (Ull)yout_z2, 14, 0, 0, (Ull)yout_w2, 14);     /* stage#6 */
    /*printf("%08.8x %08.8x %08.8x %08.8x %08.8x %08.8x -> %08.8x\n",(Uint) BR[3][0][1], (Uint)BR[3][0][0], (Uint)BR[2][0][1], (Uint)BR[2][0][0], (Uint)BR[1][0][1], (Uint)BR[1][0][0], (Uint)AR[6][0]);*/
  }
//EMAX5A end
#endif
//emax5_start((Ull*)emax5_conf_x1, (Ull*)emax5_lmmi_x1, (Ull*)emax5_regv_x1);

  return(0);
}
