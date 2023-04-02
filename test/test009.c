
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

#define WINSIZE   16
#define MINOFFSET 8
#define MAXOFFSET 14
#define MAXDELTA  4  /* -3,-2,-1,0,1,2,3 */
#define WBASE    (MAXDELTA*MAXDELTA*2)
#ifdef PRECISE_SCALE
  int shift_x =  0;  /* initial shift to x-center of WIN */
  int shift_y = 64;  /* initial shift to y-center of WIN */
#else
  int shift_x =  0;  /* initial shift to x-center of WIN */
  int shift_y = 56;  /* initial shift to y-center of WIN */
#endif
  int offset  =  0;  /* initial offset between WIN */
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

  image_WD = 320;
  image_HT = 240;
  image_size = image_WD*image_HT;
  sysinit((sizeof(int)*image_size)*2, 32);

  printf("membase: %08.8x\n", (Uint)membase);
  ACCI = (Uchar*)((Uchar*)membase);
  ACCO = (Uint*) ((Uchar*)ACCI + (sizeof(int)*image_size));
  printf("ACCI: %08.8x\n", ACCI);
  printf("ACCO: %08.8x\n", ACCO);

#if !defined(ARMSIML)
  x11_open(0);
#endif

  for (i=0; i<image_HT; i++) {
    for (j=0; j<image_WD; j++) {
      ACCI[i*image_WD+j] = (i<<24)| + (j<<8);
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
  int y;

  smallwin_offset_x = (WINSIZE+offset);
  smallwin_offset_y = (WINSIZE+offset)*image_WD;

  for (y=WINSIZE; y<image_HT-WINSIZE; y++)
    gather_x1(y*image_WD, y*image_WD);
//EMAX5A drain_dirty_lmm
//emax5_drain_dirty_lmm();
}

gather_x1(Uint yin, Uint yout)
{
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
  Ull  loop = (image_WD-WINSIZE*2)/2;
  Ull  x = WINSIZE;
  Uint *ym_xm   = ACCI         -smallwin_offset_y-smallwin_offset_x;
  Uint *ym_xz   = ACCI         -smallwin_offset_y                  ;
  Uint *ym_xp   = ACCI         -smallwin_offset_y+smallwin_offset_x;
  Uint *yz_xm   = ACCI                           -smallwin_offset_x;
  Uint *yz_xz   = ACCI                                             ;
  Uint *yz_xp   = ACCI                           +smallwin_offset_x;
  Uint *yp_xm   = ACCI         +smallwin_offset_y-smallwin_offset_x;
  Uint *yp_xz   = ACCI         +smallwin_offset_y                  ;
  Uint *yp_xp   = ACCI         +smallwin_offset_y+smallwin_offset_x;
  Uint *acci_ym = ACCI+yin     -smallwin_offset_y;
  Uint *acci_yz = ACCI+yin;
  Uint *acci_yp = ACCI+yin     +smallwin_offset_y;
  Ull  *acco_base = (Ull*)(ACCO+yout+x);
  Ull  *acco      = (Ull*)(ACCO+yout+x);
  Ull  AR[16][4];                     /* output of EX     in each unit */
  Ull  BR[16][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  c0, c1, c2, c3, ex0, ex1;
//EMAX5A begin x1 mapdist=2
  while (loop--) {                                                /* mapped to WHILE() on BR[15][0][0] stage#0 */
    exe(OP_ADD,  &x,             x,  EXP_H3210,         2LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,  OP_NOP, 0LL); /* stage#0 */
    exe(OP_SUB, &r1,          -1LL,  EXP_H3210,           x, EXP_H3210, 0LL, EXP_H3210, OP_AND,  15LL, OP_NOP, 0LL); /* stage#1 */
    exe(OP_NOP, &r2,             x,  EXP_H3210,         0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,  OP_SRL, 4LL); /* stage#1 */
    exe(OP_MLUH, &r3,           r1,  EXP_H3210, (Ull)offset, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,  OP_SRL, 4LL); /* stage#2 */
    exe(OP_MLUH, &r4,           r2,  EXP_H3210,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,  OP_NOP, 0LL); /* stage#2 */
    exe(OP_ADD, &r5,  (Ull)shift_x,  EXP_H3210,    (Ull)yin, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,  OP_NOP, 0LL); /* stage#2 */
    exe(OP_ADD3, &r0,           r3,  EXP_H3210,          r4, EXP_H3210,  r5, EXP_H3210, OP_OR,   0LL,  OP_SLL, 2LL); /* stage#3 */
    mop(OP_LDR,  1, &BR[4][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym, 320, 0, 0, (Ull)NULL, 320);        /* stage#4 */
    mop(OP_LDR,  1, &BR[4][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym, 320, 0, 0, (Ull)NULL, 320);        /* stage#4 */
    mop(OP_LDR,  1, &BR[4][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym, 320, 0, 0, (Ull)NULL, 320);        /* stage#4 */
    exe(OP_MLUH, &r10,     BR[4][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_MLUH, &r11,     BR[4][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_MLUH, &r12,     BR[4][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_MLUH, &r13,     BR[4][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
    exe(OP_MLUH, &r14,     BR[4][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
    exe(OP_MLUH, &r15,     BR[4][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
    exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#6 */
    mop(OP_LDR,  1, &BR[6][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz, 320, 0, 0, (Ull)NULL, 320);         /* stage#6 */
    mop(OP_LDR,  1, &BR[6][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz, 320, 0, 0, (Ull)NULL, 320);         /* stage#6 */
    mop(OP_LDR,  1, &BR[6][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz, 320, 0, 0, (Ull)NULL, 320);         /* stage#6 */
    exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#7 */
    exe(OP_MLUH, &r10,     BR[6][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
    exe(OP_MLUH, &r11,     BR[6][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
    exe(OP_MLUH, &r12,     BR[6][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
    exe(OP_MLUH, &r13,     BR[6][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
    exe(OP_MLUH, &r14,     BR[6][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
    exe(OP_MLUH, &r15,     BR[6][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
    exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#8 */
    mop(OP_LDR,  1, &BR[8][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp, 320, 0, 0, (Ull)NULL, 320);        /* stage#8 */
    mop(OP_LDR,  1, &BR[8][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp, 320, 0, 0, (Ull)NULL, 320);        /* stage#8 */
    mop(OP_LDR,  1, &BR[8][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp, 320, 0, 0, (Ull)NULL, 320);        /* stage#8 */
    exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#9 */
    exe(OP_MLUH, &r10,     BR[8][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
    exe(OP_MLUH, &r11,     BR[8][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
    exe(OP_MLUH, &r12,     BR[8][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
    exe(OP_MLUH, &r13,     BR[8][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
    exe(OP_MLUH, &r14,     BR[8][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
    exe(OP_MLUH, &r15,     BR[8][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
    exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#10 */
    exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#11 */
#if 0
printf("r21=%08.8x_%08.8x ", (Uint)(r21>>32), (Uint)r21);
printf("r20=%08.8x_%08.8x\n", (Uint)(r20>>32), (Uint)r20);
printf("r23=%08.8x_%08.8x ", (Uint)(r23>>32), (Uint)r23);
printf("r22=%08.8x_%08.8x\n", (Uint)(r22>>32), (Uint)r22);
printf("r25=%08.8x_%08.8x ", (Uint)(r25>>32), (Uint)r25);
printf("r24=%08.8x_%08.8x\n", (Uint)(r24>>32), (Uint)r24);
#endif
    exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#12 */
    exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#12 */
#if 0
printf("r31=%08.8x_%08.8x ", (Uint)(r31>>32), (Uint)r31);
printf("r30=%08.8x_%08.8x\n", (Uint)(r30>>32), (Uint)r30);
#endif
    exe(OP_MH2BW, &r1,  r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#13 */
    mop(OP_STR,      3, &r1, (Ull)(acco++), 0LL, MSK_D0, (Ull)acco_base, 288, 0, 0, (Ull)NULL, 288);       /* stage#13 */
  }
//EMAX5A end
//emax5_start((Ull*)emax5_conf_x1, (Ull*)emax5_lmmi_x1, (Ull*)emax5_regv_x1);
  return(0);
}
