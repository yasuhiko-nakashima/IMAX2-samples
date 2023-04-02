
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm64/sample/mm_cnn_lf/RCS/mm.c,v 1.4 2018/02/04 10:28:53 nakashim Exp nakashim $";

/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

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
#include <time.h>
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

int WD, HT, BITMAP, SCRWD, SCRHT, VECWD, VECHT, VECSTEP;
int enable_x11 = 1;

Uchar* membase;

#define TRACE_SPIKE
#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif
#if !defined(ARMSIML)
#include "./xdisp.c"
#endif

sysinit(memsize, alignment) Uint memsize, alignment;
{
#if defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif __linux__ == 1
  posix_memalign(&membase, alignment, memsize);
#else
  membase = (void*)malloc(memsize+alignment);
  if ((Ull)membase & (Ull)(alignment-1))
    membase = (void*)(((Ull)membase & ~(Ull)(alignment-1))+alignment);
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

/* LMM:16KB, RMM:64KB: M/NCHIP=124 M/NCHIP/RMGRP=31 */
/* A A   B B B B B B   C C C C C C */
/* A A   B B B B B B   C C C C C C */
/* A A                 C C C C C C */
/* A A                 C C C C C C */
/* L=2, M1=4, M2=6     L<M1,M2     */

#define L  224LL
#define M1 224LL
#define M2 224LL
Uint *A32_0; /*[M1][L];*/
Uint *A32_1; /*[M1][L];*/
Uint *B32_0; /*[L][M2];*/
Uint *B32_1; /*[M2][L];*/
Uint *C32_0; /*[M1][M2];*/
Uint *C32_1; /*[M1][M2];*/
Uint *C32_2; /*[M1][M2];*/
Uint *C32_3; /*[M1][M2];*/
Uint *C32_4; /*[M1][M2];*/
char *A08_0; /*[M1][L];*/
char *B08_1; /*[M2][L];*/
char *C08_0; /*[M1][M2];*/
int row, col, n;
int top, blk;
int w, h;
int count0, count1, count2;

#define CSIMWD 320
#define CSIMHT 240
#define CSIMBM (CSIMWD*CSIMHT)
Uint Z[CSIMBM];

#define ERRTH  (5.0E-3)
#define udiff(a,b) (((a)-(b)>=0.0?(a)-(b):(b)-(a))/((a)==0.0?1:(a)))
#define setmax(max, new) { if (max < (new)) max = (new); }

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))

#ifdef ARMSIML
void FP_to_X(int id, float *from)
{
  int i, j;
  Uint *to;

  to = Z; /* 320*240 buffer for CSM */
  for (i=0; i<HT; i++,to+=(CSIMWD-WD)) {
    for (j=0; j<WD; j++,to++,from++) {
      Uint color;
      if      (fabsf(*from) < 0.000)
        color = 0x00000000;
      else if (fabsf(*from) < 0.063)
        color = 0x80000000;
      else if (fabsf(*from) < 0.125)
        color = 0xff000000;
      else if (fabsf(*from) < 0.188)
        color = 0x00008000;
      else if (fabsf(*from) < 0.250)
        color = 0x0000ff00;
      else if (fabsf(*from) < 0.313)
        color = 0x80008000;
      else if (fabsf(*from) < 0.375)
        color = 0xff00ff00;
      else if (fabsf(*from) < 0.437)
        color = 0x00800000;
      else if (fabsf(*from) < 0.500)
        color = 0x00ff0000;
      else if (fabsf(*from) < 0.563)
        color = 0x00808000;
      else if (fabsf(*from) < 0.625)
        color = 0x00ffff00;
      else if (fabsf(*from) < 0.688)
        color = 0x80800000;
      else if (fabsf(*from) < 0.750)
        color = 0xc0c00000;
      else if (fabsf(*from) < 0.813)
        color = 0xffff0000;
      else if (fabsf(*from) < 0.875)
        color = 0x80808000;
      else if (fabsf(*from) < 0.938)
        color = 0xc0c0c000;
      else
        color = 0xffffff00;
      *to = color;
    }
  }
  _copyX(id, Z);
  _updateX();
}
#endif

main()
{
  sysinit((Uint)(M1*L *sizeof(Uint)
		+M1*L *sizeof(Uint)
                +L*M2 *sizeof(Uint)
                +M2*L *sizeof(Uint)
                +M1*M2*sizeof(Uint)
                +M1*M2*sizeof(Uint)
                +M1*M2*sizeof(Uint)
                +M1*M2*sizeof(Uint)
                +M1*M2*sizeof(Uint)
                +M1*L *sizeof(char)
                +M2*L *sizeof(char)
                +M1*M2*sizeof(char)),32);
  printf("membase: %08.8x_%08.8x\n", (Uint)((Ull)membase>>32), (Uint)membase);
  A32_0 = (Uint*)membase;
  A32_1 = (Uint*)((Uchar*)A32_0 + M1*L *sizeof(Uint));
  B32_0 = (Uint*)((Uchar*)A32_1 + M1*L *sizeof(Uint));
  B32_1 = (Uint*)((Uchar*)B32_0 + L*M2 *sizeof(Uint));
  C32_0 = (Uint*)((Uchar*)B32_1 + M2*L *sizeof(Uint));
  C32_1 = (Uint*)((Uchar*)C32_0 + M1*M2*sizeof(Uint));
  C32_2 = (Uint*)((Uchar*)C32_1 + M1*M2*sizeof(Uint));
  C32_3 = (Uint*)((Uchar*)C32_2 + M1*M2*sizeof(Uint));
  C32_4 = (Uint*)((Uchar*)C32_3 + M1*M2*sizeof(Uint));
  A08_0 = (char*)((Uchar*)C32_4 + M1*M2*sizeof(Uint));
  B08_1 = (char*)((Uchar*)A08_0 + M1*L *sizeof(char));
  C08_0 = (char*)((Uchar*)B08_1 + M2*L *sizeof(char));
  printf("A32_0: %08.8x\n", (Uint)A32_0);
  printf("A32_1: %08.8x\n", (Uint)A32_1);
  printf("B32_0: %08.8x\n", (Uint)B32_0);
  printf("B32_1: %08.8x\n", (Uint)B32_1);
  printf("C32_0: %08.8x\n", (Uint)C32_0);
  printf("C32_1: %08.8x\n", (Uint)C32_1);
  printf("C32_2: %08.8x\n", (Uint)C32_2);
  printf("C32_3: %08.8x\n", (Uint)C32_3);
  printf("C32_4: %08.8x\n", (Uint)C32_4);
  printf("A08_0: %08.8x\n", (Uint)A08_0);
  printf("B08_1: %08.8x\n", (Uint)B08_1);
  printf("C08_0: %08.8x\n", (Uint)C08_0);

  WD      = M1;
  HT      = M2;
  BITMAP  = WD*HT;
  SCRWD   = 4;
  SCRHT   = 2;
  VECWD   = 1;
  VECHT   = 1;
  VECSTEP = 4;
#if !defined(ARMSIML)
  x11_open(1); /*sh_video->disp_w, sh_video->disp_h, # rows of output_screen*/
#endif

  for (row=0; row<M1; row++) {
    for (col=0; col<L; col++) {
      *(float*)&A32_0[row*L+col] = (float)(row-col)/(float)(L*8);
      convf32tou7(&A08_0[row*L+col], *(float*)&A32_0[row*L+col]);
    }
  }

  for (row=0; row<L; row++) {
    for (col=0; col<M2; col++) {
      *(float*)&B32_0[row*M2+col] = (float)(row-col)/(float)(L*8);
      *(float*)&B32_1[col*L+row]  = (float)(row-col)/(float)(L*8);
      convf32tou7(&B08_1[col*L+row], *(float*)&B32_1[col*L+row]);
    }
  }

  FP_to_X(0, A32_0);
  FP_to_X(1, B32_0);
  FP_to_X(2, B32_1);

  orig();  FP_to_X(3, C32_0);
  imax();  FP_to_X(4, C32_1);
  smax0(); FP_to_X(5, C32_2);
  smax1(); FP_to_X(6, C32_3);
  smax2();
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      convu8tof32(&C32_4[row*M2+col], *(u8bit*)&C08_0[row*M2+col]);
    //printf("%d.%d smax1:%7.4f smax2:%02.2x->%7.4f %08.8x\n", row, col, *(float*)&C32_3[row*M2+col], *(Uchar*)&C08_0[row*M2+col], *(float*)&C32_4[row*M2+col], *(Uint*)&C32_4[row*M2+col]);
    }
  }
  FP_to_X(7, C32_4);

#if !defined(ARMSIML)
  /* X:ideal result Y:f32->u8->f32 */
  for (row=0; row<M1; row++) {
    for (col=0; col<L; col++) {
      convf32tou7(&A08_0[row*L+col], *(float*)&A32_0[row*L+col]);
      convu8tof32(&A32_1[row*L+col], *(u8bit*)&A08_0[row*L+col]); /* for graph */
    //printf("%7.4f -> %02.2x -> %7.4f\n", *(float*)&A32_0[row*L+col], *(Uchar*)&A08_0[row*L+col], *(float*)&A32_1[row*L+col]);
    }
  }
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0);
  XFillRectangle(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0, 0, WD*VECWD, HT*VECHT);
  for (row=0; row<M1; row++) {
    for (col=0; col<L; col++) {
      XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xff00ff); /* purple */
      XDrawArc(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD*VECWD/2+(int)(*(float*)&A32_0[row*L+col]*WD/8), HT*VECHT/2-(int)(*(float*)&A32_1[row*L+col]*HT/8), 4, 4, 0, 360*64);
    }
  }
  /* X:ideal result Y:f32->u8->f32 */
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0x00ff00); /* green */
      XDrawArc(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD*VECWD/2+(int)(*(float*)&C32_3[row*M2+col]*WD/8), HT*VECHT/2-(int)(*(float*)&C32_4[row*M2+col]*HT/8), 4, 4, 0, 360*64);
    }
  }
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xff0000);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0, HT*VECHT, WD*VECWD, 0);
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xffffff);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0, HT*VECHT/2, WD*VECWD, HT*VECHT/2);
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xffffff);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD*VECWD/2, 0, WD*VECWD/2, HT*VECHT);
  XdbeSwapBuffers(xvectorinfo.dpy, &swapInfo, 1);
  XSync(xvectorinfo.dpy, 0);
#endif

#if 1
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      float origC  = *(float*)&C32_0[row*M2+col];
      float imaxC  = *(float*)&C32_1[row*M2+col];
      float smax0C = *(float*)&C32_2[row*M2+col];
      float smax1C = *(float*)&C32_3[row*M2+col];
      float smax2C = *(float*)&C32_4[row*M2+col];
      if (udiff(origC,imaxC )>ERRTH) printf("orig[%d][%d]=%7.4f imaxC [%d][%d]=%7.4f\n", row, col, origC, row, col, imaxC);
      if (udiff(origC,smax0C)>ERRTH) printf("orig[%d][%d]=%7.4f smax0C[%d][%d]=%7.4f\n", row, col, origC, row, col, smax0C);
      if (udiff(origC,smax1C)>ERRTH) printf("orig[%d][%d]=%7.4f smax1C[%d][%d]=%7.4f\n", row, col, origC, row, col, smax1C);
    //if (udiff(origC,smax2C)>ERRTH) printf("orig[%d][%d]=%7.4f smax2C[%d][%d]=%7.4f\n", row, col, origC, row, col, smax2C);
    }
  }
#endif

#if !defined(ARMSIML)
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (!x11_checkevent());
#endif
}

orig() {
  printf("<<<ORIG>>>\n");
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      for (n=0; n<L; n++) {
        if (n==0) *(float*)&C32_0[row*M2+col]  = *(float*)&A32_0[row*L+n] * *(float*)&B32_0[n*M2+col];
        else      *(float*)&C32_0[row*M2+col] += *(float*)&A32_0[row*L+n] * *(float*)&B32_0[n*M2+col];
      }
    }
  }
}

imax() {
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cofs, rofs, oofs, k;

#undef  NCHIP
#undef  RMGRP
#undef  W
#undef  H
#define NCHIP 4
#define RMGRP 7
#define W     4LL
#define H     56
  /*                           →最内loop      →最内loop         */
  /*  ┌──L ──┐┐      ┌MT─M2──┐  ┌MT─M2──┐┐      */
  /*  │a a a a a ││RMGRP │bbbb     H│  │oooo      ││RMGRP */
  /*  │          │┤CHIP  │bbbb      ┤  │          │┤CHIP  */
  /*  M1   A(in)  ││      L bbbb  B0 H│  M1   out    ││      */
  /*  │          │┤      │bbbb      ┤  │          │┤      */
  /*  │H   H   H ││      │bbbb     H│  │ 60*4並列 ││      */
  /*  └─┴─┴─┘┘      └─────┘  └─────┘┘      */
  printf("<<<IMAX>>>\n");
  /* M1/NCHIP/RMGRP * L/H * NCHIP * RMGRP * M2/W/2 * W*2 * H = M1*M2*L */
  for (top=0; top<M1/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    for (blk=0; blk<L; blk+=H) { /* 3重ループ展開の外側対象 */
      typedef struct {Uint i[8]} Ui8;
      Uint *a[NCHIP], *a0[H][NCHIP];
      Ui8  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui8  *c[NCHIP], *c0[NCHIP], *c1[NCHIP], *c2[NCHIP], *c3[NCHIP];
      for (k=0; k<H; k++) {
        b[k] = B32_0+(blk+k)*M2; b0[k] = b[k]; b1[k] = (Uint*)b[k]+2; b2[k] = (Uint*)b[k]+4;  b3[k] = (Uint*)b[k]+6;
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        a[CHIP] = A32_0+(CHIP*M1/NCHIP+top)*L;
        for (k=0; k<H; k++)
          a0[k][CHIP] = a[CHIP]+blk+k;
        c[CHIP] = C32_1+(CHIP*M1/NCHIP+top)*M2;
        c0[CHIP]= (Uint*)c[CHIP]+0; c1[CHIP]= (Uint*)c[CHIP]+2; c2[CHIP]= (Uint*)c[CHIP]+4; c3[CHIP]= (Uint*)c[CHIP]+6;
      }

#define sgemm00_core1(r, rm1, rp1) \
            mop(OP_LDR,  3, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], M2, 0, 0, (Ull)NULL, M2);\
            mop(OP_LDR,  3, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], M2, 0, 0, (Ull)NULL, M2);\
            mop(OP_LDR,  3, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], M2, 0, 0, (Ull)NULL, M2);\
            mop(OP_LDR,  3, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], M2, 0, 0, (Ull)NULL, M2);\
            mop(OP_LDWR, 1, &BR[r][2][1],  (Ull)a0[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a[CHIP], L*RMGRP, 0, 0, (Ull)NULL, L*RMGRP);\
            exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H1010, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H1010, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H1010, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H1010, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_final(r, rp1) \
            mop(OP_LDR,  3, &BR[rp1][0][1],  (Ull)c0[CHIP], (Ull)oofs, MSK_W0, (Ull)c[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
            mop(OP_LDR,  3, &BR[rp1][1][1],  (Ull)c1[CHIP], (Ull)oofs, MSK_W0, (Ull)c[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
            mop(OP_LDR,  3, &BR[rp1][2][1],  (Ull)c2[CHIP], (Ull)oofs, MSK_W0, (Ull)c[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
            mop(OP_LDR,  3, &BR[rp1][3][1],  (Ull)c3[CHIP], (Ull)oofs, MSK_W0, (Ull)c[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
            exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
            mop(OP_STR,  3, &AR[rp1][0],     (Ull)oofs, (Ull)c0[CHIP], MSK_D0, (Ull)c[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
            mop(OP_STR,  3, &AR[rp1][1],     (Ull)oofs, (Ull)c1[CHIP], MSK_D0, (Ull)c[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
            mop(OP_STR,  3, &AR[rp1][2],     (Ull)oofs, (Ull)c2[CHIP], MSK_D0, (Ull)c[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
            mop(OP_STR,  3, &AR[rp1][3],     (Ull)oofs, (Ull)c3[CHIP], MSK_D0, (Ull)c[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP)

//EMAX5A begin imax mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-L*4)<<32|((0-M2*4)&0xffffffff); LOOP1--; INIT1=0) {        /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=M2/W/2,cofs=(0-W*8)<<32|((0-W*8)&0xffffffff); LOOP0--; INIT0=0) {      /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*8)<<32|(W*8), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
            exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?(L*4)<<32|(M2*4):0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);      /* stage#1 */

            mop(OP_LDR,  3, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], M2, 0, 0, (Ull)NULL, M2);                     /* stage#1 */
            mop(OP_LDR,  3, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], M2, 0, 0, (Ull)NULL, M2);                     /* stage#1 */
            mop(OP_LDR,  3, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], M2, 0, 0, (Ull)NULL, M2);                     /* stage#1 */
            mop(OP_LDR,  3, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], M2, 0, 0, (Ull)NULL, M2);                     /* stage#1 2KB */
            mop(OP_LDWR, 1, &BR[1][2][1],  (Ull)a0[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a[CHIP], L*RMGRP, 0, 0, (Ull)NULL, L*RMGRP); /* stage#1 16KB */
            exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#2 */
            exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#2 */
            exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#2 */
            exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#2 */

            sgemm00_core1( 2,  1,  3);
            sgemm00_core1( 3,  2,  4);
            sgemm00_core1( 4,  3,  5);
            sgemm00_core1( 5,  4,  6);
            sgemm00_core1( 6,  5,  7);
            sgemm00_core1( 7,  6,  8);
            sgemm00_core1( 8,  7,  9);
            sgemm00_core1( 9,  8, 10);
            sgemm00_core1(10,  9, 11);
            sgemm00_core1(11, 10, 12);
            sgemm00_core1(12, 11, 13);
            sgemm00_core1(13, 12, 14);
            sgemm00_core1(14, 13, 15);
            sgemm00_core1(15, 14, 16);
            sgemm00_core1(16, 15, 17);
            sgemm00_core1(17, 16, 18);
            sgemm00_core1(18, 17, 19);
            sgemm00_core1(19, 18, 20);
            sgemm00_core1(20, 19, 21);
            sgemm00_core1(21, 20, 22);
            sgemm00_core1(22, 21, 23);
            sgemm00_core1(23, 22, 24);
            sgemm00_core1(24, 23, 25);
            sgemm00_core1(25, 24, 26);
            sgemm00_core1(26, 25, 27);
            sgemm00_core1(27, 26, 28);
            sgemm00_core1(28, 27, 29);
            sgemm00_core1(29, 28, 30);
            sgemm00_core1(30, 29, 31);
            sgemm00_core1(31, 30, 32);
            sgemm00_core1(32, 31, 33);
            sgemm00_core1(33, 32, 34);
            sgemm00_core1(34, 33, 35);
            sgemm00_core1(35, 34, 36);
            sgemm00_core1(36, 35, 37);
            sgemm00_core1(37, 36, 38);
            sgemm00_core1(38, 37, 39);
            sgemm00_core1(39, 38, 40);
            sgemm00_core1(40, 39, 41);
            sgemm00_core1(41, 40, 42);
            sgemm00_core1(42, 41, 43);
            sgemm00_core1(43, 42, 44);
            sgemm00_core1(44, 43, 45);
            sgemm00_core1(45, 44, 46);
            sgemm00_core1(46, 45, 47);
            sgemm00_core1(47, 46, 48);
            sgemm00_core1(48, 47, 49);
            sgemm00_core1(49, 48, 50);
            sgemm00_core1(50, 49, 51);
            sgemm00_core1(51, 50, 52);
            sgemm00_core1(52, 51, 53);
            sgemm00_core1(53, 52, 54);
            sgemm00_core1(54, 53, 55);
            sgemm00_core1(55, 54, 56);
            sgemm00_core1(56, 55, 57); /* H=56 */
            /****final*****/
            sgemm00_final(57,     58);
          }
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
}

smax0() {
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][6][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cofs, rofs, bofs, oofs, k, b00, b01, b02, b03;

#undef  NCHIP
#undef  RMGRP
#undef  W
#undef  H
#define NCHIP 4
#define RMGRP 14
#define W     4LL
#define H     56
  //   IMAXは，AとBを縦に整列し,FMAチェインのΣを形成(Aを最大限再利用).結果は最終段に累積
  //   SMAXは，AとBを横に整列し,マルチスレッディングとチャージポンプを対応付け,1stageにてΣ->1out
  //
  /*     →最内loop            →最内loop          Σ最内loop        */
  /*        A                     B1               →MT*RMGRP        */
  /*  ┌──L ──┐┐      ┌──L ──┐      ┌MT─M2──┐┐     */
  /*  │a a a a a  H│      MTb b b b b │RMGRP │oooo       H│     */
  /*  │          │┤CHIP  │b b b b b │↓    │          │┤CHIP */
  /*  M1   A(in)   H│      M2b b b b b │      M1   out     H│     */
  /*  │          │┤      │b b b b b │      │          │┤     */
  /*  │8W  8W  8W H│      │8W  8W  8W│      │           H│     */
  /*  └─┴─┴─┘┘      └─┴─┴─┘      └─────┘┘     */
  printf("<<<SMAX0>>>\n");
  /* M1/NCHIP/H * M2/W/RMGRP * NCHIP * RMGRP * L/8 * W*8*H = M1*M2*L */
  for (top=0; top<M1/NCHIP; top+=H) { /* will be parallelized by multi-chip (M/#chip) */
    for (blk=0; blk<M2; blk+=W*RMGRP) { /* 3重ループ展開の外側対象 */
      Uint *a[H][NCHIP];
      Uint *b, *b0, *b1, *b2, *b3;
      Uint *c[H][NCHIP], *c0[H][NCHIP], *c1[H][NCHIP], *c2[H][NCHIP], *c3[H][NCHIP];
      b = B32_1+blk*L; b0 = b+L*0; b1 = b+L*1; b2 = b+L*2;  b3 = b+L*3;
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        for (k=0; k<H; k++) {
          a[k][CHIP] = A32_0+(CHIP*M1/NCHIP+top+k)*L;
          c[k][CHIP] = C32_2+(CHIP*M1/NCHIP+top+k)*M2+blk;
          c0[k][CHIP]= c[k][CHIP]+0; c1[k][CHIP]= c[k][CHIP]+1; c2[k][CHIP]= c[k][CHIP]+2; c3[k][CHIP]= c[k][CHIP]+3;
        }
      }

#define spike00_core1(r, s) \
  mo4(OP_LDRQ,  1,  BR[r][2],      (Ull)b0,           (Ull)bofs,   MSK_W1,    (Ull)b,           L*W*RMGRP, 0,      0,   (Ull)NULL,   L*W*RMGRP); /* stage#2 */\
  mo4(OP_LDRQ,  1,  BR[r][3],      (Ull)b1,           (Ull)bofs,   MSK_W1,    (Ull)b,           L*W*RMGRP, 0,      0,   (Ull)NULL,   L*W*RMGRP); /* stage#2 */\
  mo4(OP_LDRQ,  1,  BR[r][4],      (Ull)b2,           (Ull)bofs,   MSK_W1,    (Ull)b,           L*W*RMGRP, 0,      0,   (Ull)NULL,   L*W*RMGRP); /* stage#2 */\
  mo4(OP_LDRQ,  1,  BR[r][5],      (Ull)b3,           (Ull)bofs,   MSK_W1,    (Ull)b,           L*W*RMGRP, 0,      0,   (Ull)NULL,   L*W*RMGRP); /* stage#2 */\
  mo4(OP_LDRQ,  1,  BR[r][1],      (Ull)a[s][CHIP],   (Ull)cofs,   MSK_W1,    (Ull)a[s][CHIP],  L,         0,      0,   (Ull)NULL,   L);         /* stage#2 IMXlenが大きいのでLMM*2使用 */\
  exe(OP_NOP,      &AR[r][0], 0LL, EXP_H3210,         0LL,         EXP_H3210, 0LL,              EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#2 (dummy to set target location) */\
  mop(OP_LDWR,  1, &b00,           (Ull)c0[s][CHIP],  (Ull)oofs,   MSK_W0,    (Ull)c[s][CHIP],  W*RMGRP,   0,      1,   (Ull)NULL,   W*RMGRP);   /* stage#2 foldはunit[0]に要指定 */\
  mop(OP_LDWR,  1, &b01,           (Ull)c1[s][CHIP],  (Ull)oofs,   MSK_W0,    (Ull)c[s][CHIP],  W*RMGRP,   0,      1,   (Ull)NULL,   W*RMGRP);   /* stage#2 foldはunit[0]に要指定 */\
  mop(OP_LDWR,  1, &b02,           (Ull)c2[s][CHIP],  (Ull)oofs,   MSK_W0,    (Ull)c[s][CHIP],  W*RMGRP,   0,      1,   (Ull)NULL,   W*RMGRP);   /* stage#2 foldはunit[0]に要指定 */\
  mop(OP_LDWR,  1, &b03,           (Ull)c3[s][CHIP],  (Ull)oofs,   MSK_W0,    (Ull)c[s][CHIP],  W*RMGRP,   0,      1,   (Ull)NULL,   W*RMGRP);   /* stage#2 foldはunit[0]に要指定 */\
  *(float*)&b00 += *((float*)&BR[r][1][0]+0) * *((float*)&BR[r][2][0]+0)\
                +  *((float*)&BR[r][1][0]+1) * *((float*)&BR[r][2][0]+1)\
                +  *((float*)&BR[r][1][1]+0) * *((float*)&BR[r][2][1]+0)\
                +  *((float*)&BR[r][1][1]+1) * *((float*)&BR[r][2][1]+1)\
                +  *((float*)&BR[r][1][2]+0) * *((float*)&BR[r][2][2]+0)\
                +  *((float*)&BR[r][1][2]+1) * *((float*)&BR[r][2][2]+1)\
                +  *((float*)&BR[r][1][3]+0) * *((float*)&BR[r][2][3]+0)\
                +  *((float*)&BR[r][1][3]+1) * *((float*)&BR[r][2][3]+1);\
  *(float*)&b01 += *((float*)&BR[r][1][0]+0) * *((float*)&BR[r][3][0]+0)\
                +  *((float*)&BR[r][1][0]+1) * *((float*)&BR[r][3][0]+1)\
                +  *((float*)&BR[r][1][1]+0) * *((float*)&BR[r][3][1]+0)\
                +  *((float*)&BR[r][1][1]+1) * *((float*)&BR[r][3][1]+1)\
                +  *((float*)&BR[r][1][2]+0) * *((float*)&BR[r][3][2]+0)\
                +  *((float*)&BR[r][1][2]+1) * *((float*)&BR[r][3][2]+1)\
                +  *((float*)&BR[r][1][3]+0) * *((float*)&BR[r][3][3]+0)\
                +  *((float*)&BR[r][1][3]+1) * *((float*)&BR[r][3][3]+1);\
  *(float*)&b02 += *((float*)&BR[r][1][0]+0) * *((float*)&BR[r][4][0]+0)\
                +  *((float*)&BR[r][1][0]+1) * *((float*)&BR[r][4][0]+1)\
                +  *((float*)&BR[r][1][1]+0) * *((float*)&BR[r][4][1]+0)\
                +  *((float*)&BR[r][1][1]+1) * *((float*)&BR[r][4][1]+1)\
                +  *((float*)&BR[r][1][2]+0) * *((float*)&BR[r][4][2]+0)\
                +  *((float*)&BR[r][1][2]+1) * *((float*)&BR[r][4][2]+1)\
                +  *((float*)&BR[r][1][3]+0) * *((float*)&BR[r][4][3]+0)\
                +  *((float*)&BR[r][1][3]+1) * *((float*)&BR[r][4][3]+1);\
  *(float*)&b03 += *((float*)&BR[r][1][0]+0) * *((float*)&BR[r][5][0]+0)\
                +  *((float*)&BR[r][1][0]+1) * *((float*)&BR[r][5][0]+1)\
                +  *((float*)&BR[r][1][1]+0) * *((float*)&BR[r][5][1]+0)\
                +  *((float*)&BR[r][1][1]+1) * *((float*)&BR[r][5][1]+1)\
                +  *((float*)&BR[r][1][2]+0) * *((float*)&BR[r][5][2]+0)\
                +  *((float*)&BR[r][1][2]+1) * *((float*)&BR[r][5][2]+1)\
                +  *((float*)&BR[r][1][3]+0) * *((float*)&BR[r][5][3]+0)\
                +  *((float*)&BR[r][1][3]+1) * *((float*)&BR[r][5][3]+1);\
  mop(OP_STWR,  1, &b00,           (Ull)oofs,  (Ull)c0[s][CHIP],   MSK_D0,    (Ull)c[s][CHIP],    W*RMGRP, 0,      1,   (Ull)NULL,   W*RMGRP); /* stage#2 */\
  mop(OP_STWR,  1, &b01,           (Ull)oofs,  (Ull)c1[s][CHIP],   MSK_D0,    (Ull)c[s][CHIP],    W*RMGRP, 0,      1,   (Ull)NULL,   W*RMGRP); /* stage#2 */\
  mop(OP_STWR,  1, &b02,           (Ull)oofs,  (Ull)c2[s][CHIP],   MSK_D0,    (Ull)c[s][CHIP],    W*RMGRP, 0,      1,   (Ull)NULL,   W*RMGRP); /* stage#2 */\
  mop(OP_STWR,  1, &b03,           (Ull)oofs,  (Ull)c3[s][CHIP],   MSK_D0,    (Ull)c[s][CHIP],    W*RMGRP, 0,      1,   (Ull)NULL,   W*RMGRP)  /* stage#2 */

/*//EMAX5A begin smax0 mapdist=0*/
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-L*4*W)<<32|((0-16LL)&0xffffffff); LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=L/8,cofs=(0-32LL)<<32|((0)&0xffffffff); LOOP0--; INIT0=0) {            /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (32LL)<<32|(0), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?(L*4*W)<<32|(16LL):0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &bofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);        /* stage#1 */
            exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);        /* stage#1 */

            spike00_core1( 2,  0);
            spike00_core1( 3,  1);
            spike00_core1( 4,  2);
            spike00_core1( 5,  3);
            spike00_core1( 6,  4);
            spike00_core1( 7,  5);
            spike00_core1( 8,  6);
            spike00_core1( 9,  7);
            spike00_core1(10,  8);
            spike00_core1(11,  9);
            spike00_core1(12, 10);
            spike00_core1(13, 11);
            spike00_core1(14, 12);
            spike00_core1(15, 13);
            spike00_core1(16, 14);
            spike00_core1(17, 15);
            spike00_core1(18, 16);
            spike00_core1(19, 17);
            spike00_core1(20, 18);
            spike00_core1(21, 19);
            spike00_core1(22, 20);
            spike00_core1(23, 21);
            spike00_core1(24, 22);
            spike00_core1(25, 23);
            spike00_core1(26, 24);
            spike00_core1(27, 25);
            spike00_core1(28, 26);
            spike00_core1(29, 27);
            spike00_core1(30, 28);
            spike00_core1(31, 29);
            spike00_core1(32, 30);
            spike00_core1(33, 31);
            spike00_core1(34, 32);
            spike00_core1(35, 33);
            spike00_core1(36, 34);
            spike00_core1(37, 35);
            spike00_core1(38, 36);
            spike00_core1(39, 37);
            spike00_core1(40, 38);
            spike00_core1(41, 39);
            spike00_core1(42, 40);
            spike00_core1(43, 41);
            spike00_core1(44, 42);
            spike00_core1(45, 43);
            spike00_core1(46, 44);
            spike00_core1(47, 45);
            spike00_core1(48, 46);
            spike00_core1(49, 47);
            spike00_core1(50, 48);
            spike00_core1(51, 49);
            spike00_core1(52, 50);
            spike00_core1(53, 51);
            spike00_core1(54, 52);
            spike00_core1(55, 53);
            spike00_core1(56, 54);
            spike00_core1(57, 55); /* H=56 */
          }
        }
      }
/*//EMAX5A end*/
    }
  }
/*//EMAX5A drain_dirty_lmm*/
}

smax1() {
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cofs, rofs, bofs, oofs, k, b00;

#undef  NCHIP
#undef  RMGRP
#undef  W
#undef  H
#define NCHIP 4
#define RMGRP 14
#define W     4LL
#define H     56
  //   IMAXは，AとBを縦に整列し,FMAチェインのΣを形成(Aを最大限再利用).結果は最終段に累積
  //   SMAXは，AとBを横に整列し,マルチスレッディングとチャージポンプを対応付け,1stageにてΣ->1out
  //
  /*     →最内loop            →最内loop          Σ最内loop        */
  /*        A                     B1               →MT*RMGRP        */
  /*  ┌──L ──┐┐      ┌──L ──┐      ┌MT─M2──┐┐     */
  /*  │a a a a a  H│      MTb b b b b │RMGRP │oooo       H│     */
  /*  │          │┤CHIP  │b b b b b │↓    │          │┤CHIP */
  /*  M1   A(in)   H│      M2b b b b b │      M1   out     H│     */
  /*  │          │┤      │b b b b b │      │          │┤     */
  /*  │8W  8W  8W H│      │8W  8W  8W│      │           H│     */
  /*  └─┴─┴─┘┘      └─┴─┴─┘      └─────┘┘     */
  printf("<<<SMAX1>>>\n");
  /* M1/NCHIP/H * M2/RMGRP * NCHIP * RMGRP * L/8 * 8*H = M1*M2*L */
  for (top=0; top<M1/NCHIP; top+=H) { /* will be parallelized by multi-chip (M/#chip) */
    for (blk=0; blk<M2; blk+=RMGRP) { /* 3重ループ展開の外側対象 */
      Uint *a[H][NCHIP];
      Uint *b, *b0;
      Uint *c[H][NCHIP], *c0[H][NCHIP];
      b = B32_1+blk*L; b0 = b+L*0;
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        for (k=0; k<H; k++) {
          a[k][CHIP] = A32_0+(CHIP*M1/NCHIP+top+k)*L;
          c[k][CHIP] = C32_3+(CHIP*M1/NCHIP+top+k)*M2+blk;
          c0[k][CHIP]= c[k][CHIP]+0;
        }
      }

#define spike01_core1(r, s) \
  mo4(OP_LDRQ,  1,  BR[r][2], (Ull)b0,                  (Ull)bofs,        MSK_W1,    (Ull)b,          L*RMGRP,   0,      0,   (Ull)NULL,   L*RMGRP); /* stage#2 */\
  mo4(OP_LDRQ,  1,  BR[r][1], (Ull)a[s][CHIP],          (Ull)cofs,        MSK_W1,    (Ull)a[s][CHIP], L,         0,      0,   (Ull)NULL,   L);       /* stage#2 IMXlenが大きいのでLMM*2使用 */\
  exe(OP_NOP,      &AR[r][0], 0LL,           EXP_H3210, 0LL,              EXP_H3210, 0LL,             EXP_H3210, OP_NOP, 0LL, OP_NOP,      0LL);     /* stage#2 (dummy to set target location) */\
  mop(OP_LDWR,  1, &b00,      (Ull)c0[s][CHIP],         (Ull)oofs,        MSK_W0,    (Ull)c[s][CHIP], RMGRP,     0,      1,   (Ull)NULL,   RMGRP);   /* stage#2 foldはunit[0]に要指定 */\
/*ex4(OP_SMA,      &b00,      INIT0?b00:b00, EXP_H3210, BR[r][1],         EXP_H3210, BR[r][2],        EXP_H3210, OP_NOP, 0LL, OP_NOP,      0LL);   *//* stage#2 */\
  *(float*)&b00 += *((float*)&BR[r][1][0]+0) * *((float*)&BR[r][2][0]+0)\
                +  *((float*)&BR[r][1][0]+1) * *((float*)&BR[r][2][0]+1)\
                +  *((float*)&BR[r][1][1]+0) * *((float*)&BR[r][2][1]+0)\
                +  *((float*)&BR[r][1][1]+1) * *((float*)&BR[r][2][1]+1)\
                +  *((float*)&BR[r][1][2]+0) * *((float*)&BR[r][2][2]+0)\
                +  *((float*)&BR[r][1][2]+1) * *((float*)&BR[r][2][2]+1)\
                +  *((float*)&BR[r][1][3]+0) * *((float*)&BR[r][2][3]+0)\
                +  *((float*)&BR[r][1][3]+1) * *((float*)&BR[r][2][3]+1);\
  mop(OP_STWR,  1, &b00,      (Ull)oofs,                (Ull)c0[s][CHIP], MSK_D0,    (Ull)c[s][CHIP], RMGRP,     0,      1,   (Ull)NULL,   RMGRP)    /* stage#2 */

/*//EMAX5A begin smax1 mapdist=0*/
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-L*4)<<32|((0-4LL)&0xffffffff); LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=L/8,cofs=(0-32LL)<<32|((0)&0xffffffff); LOOP0--; INIT0=0) {         /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (32LL)<<32|(0), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?(L*4)<<32|(4LL):0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &bofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);     /* stage#1 */
            exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);     /* stage#1 */

            spike01_core1( 2,  0);
            spike01_core1( 3,  1);
            spike01_core1( 4,  2);
            spike01_core1( 5,  3);
            spike01_core1( 6,  4);
            spike01_core1( 7,  5);
            spike01_core1( 8,  6);
            spike01_core1( 9,  7);
            spike01_core1(10,  8);
            spike01_core1(11,  9);
            spike01_core1(12, 10);
            spike01_core1(13, 11);
            spike01_core1(14, 12);
            spike01_core1(15, 13);
            spike01_core1(16, 14);
            spike01_core1(17, 15);
            spike01_core1(18, 16);
            spike01_core1(19, 17);
            spike01_core1(20, 18);
            spike01_core1(21, 19);
            spike01_core1(22, 20);
            spike01_core1(23, 21);
            spike01_core1(24, 22);
            spike01_core1(25, 23);
            spike01_core1(26, 24);
            spike01_core1(27, 25);
            spike01_core1(28, 26);
            spike01_core1(29, 27);
            spike01_core1(30, 28);
            spike01_core1(31, 29);
            spike01_core1(32, 30);
            spike01_core1(33, 31);
            spike01_core1(34, 32);
            spike01_core1(35, 33);
            spike01_core1(36, 34);
            spike01_core1(37, 35);
            spike01_core1(38, 36);
            spike01_core1(39, 37);
            spike01_core1(40, 38);
            spike01_core1(41, 39);
            spike01_core1(42, 40);
            spike01_core1(43, 41);
            spike01_core1(44, 42);
            spike01_core1(45, 43);
            spike01_core1(46, 44);
            spike01_core1(47, 45);
            spike01_core1(48, 46);
            spike01_core1(49, 47);
            spike01_core1(50, 48);
            spike01_core1(51, 49);
            spike01_core1(52, 50);
            spike01_core1(53, 51);
            spike01_core1(54, 52);
            spike01_core1(55, 53);
            spike01_core1(56, 54);
            spike01_core1(57, 55); /* H=56 */
          }
        }
      }
/*//EMAX5A end*/
    }
  }
/*//EMAX5A drain_dirty_lmm*/
}

smax2() {
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cofs, rofs, bofs, oofs, k, b00;

#undef  NCHIP
#undef  RMGRP
#undef  W
#undef  H
/*#define NCHIP 4*/
/*#define RMGRP 15*/
/*#define W     4LL*/
/*#define H     60*/
#define NCHIP 1
#define RMGRP 4
#define W     4LL
#define H     1
  //   IMAXは，AとBを縦に整列し,FMAチェインのΣを形成(Aを最大限再利用).結果は最終段に累積
  //   SMAXは，AとBを横に整列し,マルチスレッディングとチャージポンプを対応付け,1stageにてΣ->1out
  //
  /*     →最内loop            →最内loop          Σ最内loop        */
  /*        A                     B1               →MT*RMGRP        */
  /*  ┌──L ──┐┐      ┌──L ──┐      ┌MT─M2──┐┐     */
  /*  │a a a a a  H│      MTb b b b b │RMGRP │oooo       H│     */
  /*  │          │┤CHIP  │b b b b b │↓    │          │┤CHIP */
  /*  M1   A(in)   H│      M2b b b b b │      M1   out     H│     */
  /*  │          │┤      │b b b b b │      │          │┤     */
  /*  │8W  8W  8W H│      │8W  8W  8W│      │           H│     */
  /*  └─┴─┴─┘┘      └─┴─┴─┘      └─────┘┘     */
  //                                                          HOSTが8bit固定長圧縮しデータ量削減
  //                                                          EAGは通常通り使用,ハードが8bitをSpikeに変換(周期は当面64サイクル)
  //              SpikeArray            LDQ      ST            │
  //   ┌→■  32*8bit->32Spike      ■ □□ ■ ■■ ■     □□□□
  //   │  │  ξξξξξξξξAND   │└─┘│└─┘│        │
  //   │  ■                        ■ ■■ ■ □□ ■        │
  //   │  │  SEL     SEL           │└─┘│└─┘│        │
  //   │  ■                        ■ ■■ ■ □□ ■        │
  //   │  │┌→    ADD->8bit       │└─┘│└─┘│        │
  //   │  ■└───□              ■  □  ■  □  ■     □□□□ 8bit圧縮済(256bit幅なので32データ)
  //   │  ┌────┴──ST→───┬───┬───┬─←──┤
  //   │  WD────────────WD─AD─WD─AD─WD        │             ┌────────────┐
  //   │  A0■■■■■■  ■■■■■■B0  ■■■■■■        │ ┌──┐  ┌┤ IMAX2 IMAX2 IMAX2 IMAX2│
  //   │  A1■■■in■■  ■■■w ■■B1  ■■out ■■        │ │ARM ├─┤├────────────┤
  //   │  A2■■■■■■  ■■■■■■B2  ■■■■■■        │ └──┘  └┤ SMAX1 SMAX1 SMAX1 SMAX1│
  //   │  A3■■■■■■  ■■■■■■B3  ■■■■■■        │             └────────────┘
  //   │  RD──────RD──────RD──────RD        │
  //   │  ├──────┼──────┼──────┼──→─┤->DDR
  //   └─■            ■            ■            ■ LDQ-in │
  //       □            □            □            □        │
  //       ■            ■            ■            ■ LDQ-w  │
  //       □            □            □            ■ LD-out │

  //             t=0         t=1         t=2         t=3    |    t=4         t=5         t=6         t=7    |    t=0         t=1         t=2         t=3    |    t=4         t=5         t=6         t=7    |
  //     LMM      C           A           B                 |                                               |                                               |                                               |
  // st5 BR0 □ □ ■ □ □ □ ■ □ □ □ ■ □ □ □ ■ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ ■ □ □ □ ■ □ □ □ ■ □ □ □ ■ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|
  // st6 BR1 □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|
  // st7 BR2 □ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ □ □ □ □ □ □ ■ ■ ■ ■ ■ ■ ■ ■|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|
  // st8 BR3 □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|
  // st5 BR0 □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ ■ □ □ □ ■ □ □ □ ■ □ □ □ ■ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ ■ □ □ □ ■ □ □ □ ■ □ □ □ ■ □|
  // st6 BR1 □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■|
  // st7 BR2 □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■|
  // st8 BR3 □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □|
  //             t=0         t=1         t=2         t=3    |    t=4         t=5         t=6         t=7         t=0         t=1         t=2         t=3    |    t=4         t=5         t=6         t=7
  // st1 select         ┏━━━━━━━━━━━━━━━┓  | C  A0 B0       A1 B1       A2 B2       A3 B3       A0 B0       A1 B1       A2 B2       A3 B3  | C  A0 B0
  //                    ┃      50MHz->■8b*2*8         ┃  | ■ ■*■    □ ■*■    □ ■*■    □ ■*■    □ ■*■    □ ■*■    □ ■*■    □ ■*■  | ■ ■*■
  // st2 ex1            ┃ 16x 800MHz->ξ1bitAND*8(mul) ┃  | ↓ AND8        AND8        AND8        AND8        AND8        AND8        AND8        AND8   |    AND8        AND8
  //                    ┃ 16x 800MHz->ξ               ┃  | □ □ □    ■ ■ □AB0 □ ■ □AB1 □ ■ □AB2 □ ■ □AB3 □ ■ □AB0 □ ■ □AB1 □ ■ □AB2 □ ■ □AB3 ■ ■ □AB0
  // st3 ex2            ┃ 16x 800MHz->ξ1bitSEL*8(add) ┃  |             ↓ ADD8        ADD8        ADD8        ADD8        ADD8        ADD8        ADD8   |    ADD8        ADD8        ADD8
  //                    ┃      50MHz->■8b ┌┐        ┃  | □ □       □ □       ■+■8b*2   □ ■8b  ┌─┐■8b  ┌─┐■8b  ┌─┐■8b  ┌─┐■8b  ┌─┐■8b  ┌─┐■8b  ┌ ■+■8b*2   □ ■8b  ┌
  // st4 ex3            ┃             ACC  ↓│        ┃  |                            ACC         ACC   │  │ACC   │  │ACC   │  │ACC   │  │ACC   │  │ACC   │  │ACC   │  │ACC         ACC   │
  //     ──           ┃      50MHz->■8b ─┘        ┃  |    □          □          □          ■ST─┘  └■ST─┘  └■ST─┘  └■ST─┘  └■ST─┘  └■ST─┘  └■ST─┘  └■ST        ■ST─┘
  // st5 LMM            ┃                              ┃  |                                                    ↓Σ        ↓Σ        ↓Σ        ↓Σ   |    ↓Σ        ↓Σ        ↓Σ        ↓Σ
  //                    ┗━━━━━━━┯━━━━━━━┛  |    □          □          □          □     |    ■０        ■１        ■２        ■３   |    ■４        ■５        ■６        ■７   |
  //                                    ↓
  // 【単線コンピューティング】 32bitFloat→8bitUnary-Stochastic 疎行列圧縮 LMMに圧縮表現
  //             LMMにはUnary記憶          RREG以降にはレジスタを置かないことで省電力化
  //   (Unary圧縮)-DDR4-AXI-> LMM(W)-READ-(Spike変換:Dup)-(ランダムシフタ)-(乗算:AND)-(加算:RND-SEL)-SelTree-(encoder)->LMM(W)
  //                in:32read ( 0.0 +1.0)
  //                 w:32read (-1.0 +1.0)
  //               out:1write ( 0.0 +1.0) 1.0/(1.0+exp(-x))
  //
  //  convf32tou8 e=126     0.992 -> s1111111  1..1 1111111111111111111111111111111111111111111111111111111111111110
  //              e=126     0.984 -> s1111110  1..1 1111111111111111111111111111111111111111111111111111111111111100
  //              e=126     0.969 -> s1111100  1..1 1111111111111111111111111111111111111111111111111111111111110000
  //              e=126     0.938 -> s1111000  1..1 1111111111111111111111111111111111111111111111111111111100000000
  //              e=126     0.875 -> s1110000  1..1 1111111111111111111111111111111111111111111111110000000000000000
  //              e=126     0.750 -> s1100000  1..1 1111111111111111111111111111111100000000000000000000000000000000
  //  convf32tou8 e=126 f=0 0.500 -> s1000000  1..1 0000000000000000000000000000000000000000000000000000000000000000
  //  convf32tou8 e=125 f=0 0.250 -> s0100000  0..0 0000000000000000000000000000000011111111111111111111111111111111
  //  convf32tou8 e=124 f=0 0.125 -> s0010000  0..0 0000000000000000000000000000000000000000000000001111111111111111
  //  convf32tou8 e=123 f=0 0.062 -> s0001000  0..0 0000000000000000000000000000000000000000000000000000000011111111
  //  convf32tou8 e=122 f=0 0.031 -> s0000100  0..0 0000000000000000000000000000000000000000000000000000000000001111
  //  convf32tou8 e=121 f=0 0.016 -> s0000010  0..0 0000000000000000000000000000000000000000000000000000000000000011
  //  convf32tou8 e=120 f=0 0.008 -> s0000001  0..0 0000000000000000000000000000000000000000000000000000000000000001
  //                        0.000 -> s0000000  0..0 0000000000000000000000000000000000000000000000000000000000000000
  //  0.00     in:0.016 w:0.16   in:0.20  w:0.25   in:0.33  w:0.50   in:0.50  w:-0.66  in:0.75  w:-0.80  in:0.83  w:0.9843 in:0.75  w:-0.80  in:0.83  w:0.9843 1.00
  //  00000000 00111111 00000101 00000100 00000011 00000010 00000001 01000001 11000010 01000011 11000100 01000101 01111111 01000011 11000100 01000101 01111111 01000000
  //            counter  counter  counter  counter  counter  counter counter  counter  counter  counter  counter  counter  counter  counter  counter  counter
  //  00000000 00000000 01000001 00100001 00010001 01001001 01010101 10101010 11011011 11101110 11110111 11111011 11111111 11111111 11101110 11110111 11111011 11111111 11111111
  //            +-----+   +----+    +---+     +--+      +-+       ++  +-----+   +----+    +---+     +--+      +-+       ++    +---+     +--+      +-+       ++
  //                  +--AND---+        +--AND---+        +--AND---+        +--AND---+        +--AND---+        +--AND---+        +--AND---+        +--AND---+
  //                     | |               | |               | |               | |               | |               | |               | |               | |
  //                    +| |-             +| |-             +| |-             +| |-             +| |-             +| |-             +| |-             +| |-
  //                     o x               o x               o x               x o               x o               o x               x o               o x
  //                     | |              +| |-             +| |-             +| |-             +| |-             +| |-             +| |-             +| |-
  //            +----->--+-------------->--+-------------->--+-------------->--+-------------->--+-------------->--+-----+ pos scaled adder(rot-selector) + pop-counter + 
  //                       |                 |                 |                 |                 |                 |                                                  | up/dn counter(8bit)
  //            +------->--+-------------->--+-------------->--+-------------->--+-------------->--+-------------->--+---+ neg scaled adder(rot-selector) + pop-counter +

  printf("<<<SMAX2>>>\n");
  /* M1/NCHIP/H * M2/RMGRP * NCHIP * RMGRP * L/8 * 8*H = M1*M2*L */
  for (top=0; top<M1/NCHIP; top+=H) { /* will be parallelized by multi-chip (M/#chip) */
    for (blk=0; blk<M2; blk+=RMGRP) { /* 3重ループ展開の外側対象 */
      char *a[H][NCHIP];
      char *b, *b0;
      char *c[H][NCHIP], *c0[H][NCHIP];
      b = B08_1+blk*L; b0 = b+L*0;
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        for (k=0; k<H; k++) {
          a[k][CHIP] = A08_0+(CHIP*M1/NCHIP+top+k)*L;
          c[k][CHIP] = C08_0+(CHIP*M1/NCHIP+top+k)*M2+blk;
          c0[k][CHIP]= c[k][CHIP]+0;
        }
      }

#if !defined(ARMSIML)
      x11_softu64_update();
#endif

#define spike02_core1(r, s) \
  mo4(OP_LDRQ,  1,  BR[r][2], (Ull)b0,                  (Ull)bofs,        MSK_W1,    (Ull)b,          L*RMGRP/4, 0,      0,    (Ull)NULL,   L*RMGRP/4); /* stage#2 */\
  mo4(OP_LDRQ,  1,  BR[r][1], (Ull)a[s][CHIP],          (Ull)cofs,        MSK_W1,    (Ull)a[s][CHIP], L/4,       0,      0,    (Ull)NULL,   L/4);       /* stage#2 IMXlenが大きいのでLMM*2使用 */\
  exe(OP_NOP,      &AR[r][0], 0LL,           EXP_H3210, 0LL,              EXP_H3210, 0LL,             EXP_H3210, OP_NOP, 0LL,  OP_NOP,      0LL);       /* stage#2 (dummy to set target location) */\
  mop(OP_LDBR,  1, &b00,      (Ull)c0[s][CHIP],         (Ull)oofs,        MSK_W0,    (Ull)c[s][CHIP], RMGRP/4,   0,      1,    (Ull)NULL,   RMGRP/4);   /* stage#2 foldはunit[0]に要指定 */\
  ex4(OP_SFMA,     &b00,      INIT0?b00:b00, EXP_H3210, BR[r][1],         EXP_H3210, BR[r][2],        EXP_H3210, OP_NOP, 2LL,  OP_NOP,      0LL);       /* stage#2 */\
  mop(OP_STBR,  1, &b00,      (Ull)oofs,                (Ull)c0[s][CHIP], MSK_D0,    (Ull)c[s][CHIP], RMGRP/4,   0,      1,    (Ull)NULL,   RMGRP/4)    /* stage#2 */

//                 rofs                cofs                LOOP1               LOOP0
// stage0  □■□□BR[0][3]    □■□□BR[0][2]    □□□□BR[0][1]    □□□□BR[0][0]
//
//                                                         oofs                bofs
// stage1  □□□□BR[1][3]    □□□□BR[1][2]    □■□□BR[1][1]    □■□□BR[1][0]
//
//         SFMA                SFMA [B]            SFMA [A]            SFMA [C]
// stage2  □□□□BR[2][3]    ■■■■BR[2][2]    ■■■■BR[2][1]    ■□■□BR[2][0]
//             cofs bofs                                               0LL null b00 oofs

//EMAX5A begin smax2 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-L)<<32|((0-1LL)&0xffffffff); LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=L/32,cofs=(0-32LL)<<32|((0)&0xffffffff); LOOP0--; INIT0=0) {         /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (32LL)<<32|(0), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?(L)<<32|(1LL):0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);  /* stage#0 */
            exe(OP_ADD,    &bofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);    /* stage#1 */
            exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);    /* stage#1 */

            spike02_core1( 2,  0);
#if 0
            spike02_core1( 3,  1);
            spike02_core1( 4,  2);
            spike02_core1( 5,  3);
            spike02_core1( 6,  4);
            spike02_core1( 7,  5);
            spike02_core1( 8,  6);
            spike02_core1( 9,  7);
            spike02_core1(10,  8);
            spike02_core1(11,  9);
            spike02_core1(12, 10);
            spike02_core1(13, 11);
            spike02_core1(14, 12);
            spike02_core1(15, 13);
            spike02_core1(16, 14);
            spike02_core1(17, 15);
            spike02_core1(18, 16);
            spike02_core1(19, 17);
            spike02_core1(20, 18);
            spike02_core1(21, 19);
            spike02_core1(22, 20);
            spike02_core1(23, 21);
            spike02_core1(24, 22);
            spike02_core1(25, 23);
            spike02_core1(26, 24);
            spike02_core1(27, 25);
            spike02_core1(28, 26);
            spike02_core1(29, 27);
            spike02_core1(30, 28);
            spike02_core1(31, 29);
            spike02_core1(32, 30);
            spike02_core1(33, 31);
            spike02_core1(34, 32);
            spike02_core1(35, 33);
            spike02_core1(36, 34);
            spike02_core1(37, 35);
            spike02_core1(38, 36);
            spike02_core1(39, 37);
            spike02_core1(40, 38);
            spike02_core1(41, 39);
            spike02_core1(42, 40);
            spike02_core1(43, 41);
            spike02_core1(44, 42);
            spike02_core1(45, 43);
            spike02_core1(46, 44);
            spike02_core1(47, 45);
            spike02_core1(48, 46);
            spike02_core1(49, 47);
            spike02_core1(50, 48);
            spike02_core1(51, 49);
            spike02_core1(52, 50);
            spike02_core1(53, 51);
            spike02_core1(54, 52);
            spike02_core1(55, 53);
            spike02_core1(56, 54);
            spike02_core1(57, 55); /* H=56 */
#endif
          }
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
}
