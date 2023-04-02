
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
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // ¡ú¡ú¡ú RESET
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
/* M1=4 L=2, M2=6      L<M1,M2     */

#if 1
#define M1 512LL
#define M2 512LL
#define L  512LL
#define LP 32LL
#else
#define M1 64LL
#define M2 64LL
#define L  64LL
#define LP 8LL
#endif

typedef struct {
  float d; /* low */
  Uint  x; /* high */
} packed;

float  *A32_0; /*[M1][L]         */
float  *A32_1; /*[M1][L]   xdisp */
packed *A32_P; /*[M1][L]   pack  */
float  *B32_0; /*[M2][L] T       */
float  *B32_1; /*[M2][L] T xdisp */
packed *B32_P; /*[M2][L] T pack  */
float  *C32_0; /*[M1][M2]  orig  */
float  *C32_1; /*[M1][M2]  imax  */
int row, col, n;
int top, blk;
int w, h;
int count0, count1, count2;

#define CSIMWD 320
#define CSIMHT 240
#define CSIMBM (CSIMWD*CSIMHT)
Uint Z[CSIMBM];

#define ERRTH  (5.0E-6)
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
                +M1*L *sizeof(packed)
                +M2*L *sizeof(Uint)
                +M2*L *sizeof(Uint)
                +M2*L *sizeof(packed)
                +M1*M2*sizeof(Uint)
                +M1*M2*sizeof(Uint)),32);
  printf("membase: %08.8x_%08.8x\n", (Uint)((Ull)membase>>32), (Uint)membase);
  A32_0 = (Uint*)membase;
  A32_1 = (Uint*)((Uchar*)A32_0 + M1*L *sizeof(Uint));
  A32_P = (Uint*)((Uchar*)A32_1 + M1*L *sizeof(Uint));
  B32_0 = (Uint*)((Uchar*)A32_P + M1*L *sizeof(packed));
  B32_1 = (Uint*)((Uchar*)B32_0 + M2*L *sizeof(Uint));
  B32_P = (Uint*)((Uchar*)B32_1 + M2*L *sizeof(Uint));
  C32_0 = (Uint*)((Uchar*)B32_P + M2*L *sizeof(packed));
  C32_1 = (Uint*)((Uchar*)C32_0 + M1*M2*sizeof(Uint));
  printf("A32_0: %08.8x\n", (Uint)A32_0);
  printf("A32_1: %08.8x\n", (Uint)A32_1);
  printf("A32_P: %08.8x\n", (Uint)A32_P);
  printf("B32_0: %08.8x\n", (Uint)B32_0);
  printf("B32_1: %08.8x\n", (Uint)B32_1);
  printf("B32_P: %08.8x\n", (Uint)B32_P);
  printf("C32_0: %08.8x\n", (Uint)C32_0);
  printf("C32_1: %08.8x\n", (Uint)C32_1);

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

  /**************************************************/
  /* A                                              */
  /**************************************************/
  for (row=0; row<M1; row++) {
    for (col=0; col<L; col++) {
      if (abs(row-col) < LP/2)
        A32_0[row*L+col] = (float)(row-col)/(L/(128.0/(float)LP));
      else
        A32_0[row*L+col] = 0.0;
    }
  }

  //pack(A32_P, A32_0);
  Uint a32pcolmax = 0;
  for (row=0; row<M1; row++) {
    Uint a32pcol = 0;
    for (col=0; col<L; col++) {
      if (A32_0[row*L+col] == 0)
        continue;
      if (a32pcol >= LP) {
        a32pcol++;
        continue;
      }
      A32_1[row*LP+a32pcol]   = A32_0[row*L+col];
      A32_P[row*LP+a32pcol].d = A32_0[row*L+col];
      A32_P[row*LP+a32pcol].x = col;
      a32pcol++;
    }
    if (a32pcolmax < a32pcol)
      a32pcolmax = a32pcol;
    for (; a32pcol<LP; a32pcol++) {
      A32_1[row*LP+a32pcol]   = 0;
      A32_P[row*LP+a32pcol].d = 0;
      A32_P[row*LP+a32pcol].x = -1;
    }
  }
  printf("a32pcolmax=%d\n", a32pcolmax);
  if (a32pcolmax >= LP)
    printf("a32pcolmax overflow\n");

  /**************************************************/
  /* B                                              */
  /**************************************************/
  for (row=0; row<M2; row++) {
    for (col=0; col<L; col++) {
      if (abs(row-col) < LP/2)
        B32_0[row*L+col] = (float)(row-col)/(L/(128.0/(float)LP));
      else
        B32_0[row*L+col] = 0.0;
    }
  }

  //pack(B32_P, B32_0);
  Uint b32pcolmax = 0;
  for (row=0; row<M2; row++) {
    Uint b32pcol = 0;
    for (col=0; col<L; col++) {
      if (B32_0[row*L+col] == 0)
        continue;
      if (b32pcol >= LP) {
        b32pcol++;
        continue;
      }
      B32_1[row*LP+b32pcol]   = B32_0[row*L+col];
      B32_P[row*LP+b32pcol].d = B32_0[row*L+col];
      B32_P[row*LP+b32pcol].x = col;
      b32pcol++;
    }
    if (b32pcolmax < b32pcol)
      b32pcolmax = b32pcol;
    for (; b32pcol<LP; b32pcol++) {
      B32_1[row*LP+b32pcol]   = 0;
      B32_P[row*LP+b32pcol].d = 0;
      B32_P[row*LP+b32pcol].x = -1;
    }
  }
  printf("b32pcolmax=%d\n", b32pcolmax);
  if (b32pcolmax >= LP)
    printf("b32pcolmax overflow\n");

  FP_to_X(0, A32_0);
  FP_to_X(1, A32_1);
  FP_to_X(2, B32_0);
  FP_to_X(3, B32_1);

  if (M1<=8) {
    for (col=0; col<LP; col++) {
      for (row=0; row<M1; row++)
	printf(" %08.8x_%08.8x", (Uint)A32_P[row*LP+col].x, *(Uint*)&A32_P[row*LP+col].d);
      printf("\n");
    }
  }
  if (M2<=8) {
    for (col=0; col<LP; col++) {
      for (row=0; row<M2; row++)
	printf(" %08.8x_%08.8x", (Uint)B32_P[row*LP+col].x, *(Uint*)&B32_P[row*LP+col].d);
      printf("\n");
    }
  }

  orig();  FP_to_X(6, C32_0);
  imax();  FP_to_X(7, C32_1);

#if 1
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      float origC  = *(float*)&C32_0[row*M2+col];
      float imaxC  = *(float*)&C32_1[row*M2+col];
//    if (udiff(origC, imaxC)>ERRTH) {
      if (origC != imaxC) {
        count2++;
        printf("orig[%d][%d]=%7.4f imaxC [%d][%d]=%7.4f\n", row, col, origC, row, col, imaxC);
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
#endif

#if !defined(ARMSIML)
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (!x11_checkevent());
#endif
}

orig() {
  printf("<<<ORIG>>>\n");
  reset_nanosec();
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      for (n=0; n<L; n++) {
        if (n==0) *(float*)&C32_0[row*M2+col]  = *(float*)&A32_0[row*L+n] * *(float*)&B32_0[col*L+n];
        else      *(float*)&C32_0[row*M2+col] += *(float*)&A32_0[row*L+n] * *(float*)&B32_0[col*L+n];
        count0++;
      }
    }
  }
  get_nanosec(0);
  printf("orig.FMA = %d\n", count0);
  show_nanosec();
}

init_array(Ull *a, int len) {
  int i;
  for (i=0; i<len; i++)
    a[i] = 0;
}

imax() {
#if 0
  FC¤ò¸µ¤Ë¹Í¤¨¤ë¤È,A½ÄÄ¹(M1*L ),BÅ¾ÃÖÁ°¤Ï²£Ä¹(L *M2),C¤Ï¾®¥µ¥¤¥º¤ÇÌ©¹ÔÎó(data¤Î¤ßM1*M2)¤ÎÁÛÄê
            °µ½Ì¸å,A½ÄÄ¹(M1*LP),BÅ¾ÃÖÁ°¤Ï²£Ä¹(LP*M2),C¤Ï¾®¥µ¥¤¥º¤ÇÌ©¹ÔÎó(data¤Î¤ßM1*M2)¤ÎÁÛÄê
            col#2(32KB)         col#1(16KB)      col#0(16KB)
            ¢¢¢¢¢¢¢¢¢¢          ¢¢¢¢¢¢¢¢¢¢
             b¢¢¢¢¢¢¢¢           a¢¢¢¢¢¢¢¢
stage1¨£¨¡¨¤¨¢      ¨¢    ¨£¨¡¨¤¨¢      ¨¢
      ¨¢  ¨ª¨ª    bofs    ¨¢  ¨ª¨ª    bofs      ¨£¨¡¨¨¨¨¨¤
      ¨¢  b0¢¢xBxA  ¢¢    ¨¢  a0¢¢xBxA  ¢¢      ¨¢  ¢¢¢¢¢¢ if (yx==yx) FMA(lower)
      ¨¢¡¡  ¨¢¨£¨¨¨¡¨¡¨¡¨¡¨¢¨¡¨¡¨¢¨¨¨¨¨¡¨¡¨¡¨¤  ¨¢  ¨¢¨¢¨¢ else    pass S1(lower)
      ¨¢   _¨ª¨ª¨ª_ ¨¢    ¨¢   _¨ª¨ª¨ª_ ¨¢  ¨¢  ¨¢ _¨¢¨¢¨¢_
stage2¨¢   ¡À cmp¡¿ ¨¢    ¨¢   ¡À cmp¡¿ ¨¢  ¨¢  ¨¢ ¡ÀCFMA¡¿ ¢¢¸½¾õ     base++ sb =(bus[i][j].ea0brv?0:2)|(dec[i][j].dmop0.¡üupdt?1:0); 0:ea0br, 1:ea0dr(ea0br+loop), 2:eabbrs, 3:ea0dr(eabbrs+loop)
      ¨¢ add b0+8¢£ ¨¢    ¨¢ add a0+8¢£ ¨¢  ¨¢  ¨¢ ¡¡ ¨¢ ¡¡            ea0b=(!(¡üsb&1)||!one_shot)? (((sb&2)?eab:ea0b)|strqofs) : ea04dr_prev; //sb=1½é²óreg. °Ê¸åea04dr
      ¨¢    ¨¢   ¡¡ ¨¢    ¨¢    ¨¢      ¨¢  ¨¢  ¨¢ ¡¡ ¨¢ ¡¡            ea0o=(!(¡üsb&1)|| one_shot)? ( (so&1)?eao:ea0o         ) : 0LL;         //sb=1½é²ó0.   °Ê¸åofs
      ¨¢¨£¨¡¢¢__¨£¨¡¢£    ¨¢¨£¨¡¢¢__¨£¨¡¢£  ¨¢  ¨¢ ___¢¢___ ¢£¿·µ¬ÄÉ²Ã INIT0?(b0[h]=b):ic1?++b0[h]:b0[h] ... sb=Æ±¾å(¸¡½Ð»þ¤Ëconv-c2c¤¬updt=1)
stage3¨¢¨¢ ¡À_add¡¿       ¨¢¨¢ ¡À_add¡¿     ¨¢  ¨¢ ¡ÀCFMA¡¿     stage1 ea0b=(!(¡üsb&1)||!one_shot||(reg_ctrl.i[cid].conf[i][j].cdw0.¢£ea0init&&(exring[cid].unit[i].unit1_forstat&1)))
      ¨¢¢£    ¢¢b0+8+ofs  ¨¢¢£    ¢¢a0+8+ofs¨¢  ¨¢    ¢¢                   ? (((sb&2)?eab:ea0b)|strqofs) : ¢£ea04woofs_prev; //sb=1½é²óreg. °Ê¸åea04woofs_prev
      ¨¢woofs_¨¢___       ¨¢woofs_¨¢___     ¨¢  ¨¢ ___¨¢___            ea0o=(!(¡üsb&1)|| one_shot|| reg_ctrl.i[cid].conf[i][j].cdw0.¢£ea0init)? ((so&1)?eao:ea0o) : 0LL;
stage4¨¢¨¢  |_msk_|       ¨¢¨¢  |_msk_|     ¨¢  ¨¢ ¡ÀCFMA¡¿     stage2 ea02dr    = (OP_CMPADD_LE && xA!=0xffffffff && xB<=xA) ? a0+8:
      ¨¦¢£    ¢¢          ¨¦¢£    ¢¢        ¨¢  ¨§¨¡¨¡¢¢                           (OP_CMPADD_GE && xB!=0xffffffff && xB>=xA) ? a0+8: a0;
 ea14woofs¨£¨¡¨ª¨¡¨¤ ea04woofs¨£¨¡¨ª¨¡¨¤    ¨¢  ¨¢¨£¨¡¨ª¨¡¨¤    stage3 ea03dr    = ea02dr + ofs;
          ¨¢ yxB[]¨¢          ¨¢ yxA[]¨¢    ¨¢  ¨¢¨¢   C  ¨¢           ea03woofs = ea02dr;
          ¨¦¨¡¨¨¨¡¨¥          ¨¦¨¡¨¨¨¡¨¥    ¨¢  ¨¢¨¦¨¡¨¡¨¨¨¥    stage4 ea04woofs = ea03woofs;
          ¡¡¡¡¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨§¨¡¨¡¨¡¨¡¨¢¨¡¨¢¨¨¨¤¡¡¨¢     
              ¢¢xB¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¢¢xA¨¡¨¡¨¡¨¥  ¨¢xBxB¢¢¢¢  idx¤Èdata¤òÊ¬¤±¤ë¤ÈEAG*4¤ÎÆ±»þ¶îÆ°¤¬É¬Í×¤Ê¤Î¤Ç,idx¤Èdata¤ò°ìÂÎ²½(16+16+32bit)
                                                ¨¦¢¢¢¢¢¢¢¢  ¾å¤Ø¤ÏBR¥Ð¥Ã¥Õ¥¡¥ê¥ó¥°¤Ï»È¤ï¤ºÄ¾ÀÜÌá¤¹.FPU¤Ø¤ÏÄÌ¾ïÄÌ¤êBR¥Ð¥Ã¥Õ¥¡¥ê¥ó¥°
#endif
#undef  NCHIP
#undef  RMGRP
#undef  W
#undef  H
#ifdef EMAXSC
#define NCHIP EMAX_NCHIP
#define RMGRP M1
#else
#define NCHIP 1
#define RMGRP M1/4
#endif
#define W     4LL
#define H     16
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cofs, rofs, bofs, oofs, k, c00, c01;

  printf("<<<IMAX>>>\n");
  reset_nanosec();
#if 0
  /* M1/NCHIP/H * M2/RMGRP * NCHIP * RMGRP * L/8 * 8*H = M1*M2*L */
  /* col3  col2  col1  col0               */
  /*       LD B        LD C               */
  /*       LD A        EXST  fold¤Ïcol0   */
  /*        B+A         C  Æ°ºî¥¿¥¤¥ß¥ó¥° */
  /*         B    A     C  LMM¤ÎÇÛÃÖ      */
  /*       RMGRP LP*2  2W                 */
  for (blk=0; blk<M2; blk+=RMGRP) { /* 3½Å¥ë¡¼¥×Å¸³«¤Î³°Â¦ÂÐ¾Ý */
    for (top=0; top<M1/NCHIP; top+=H) { /* will be parallelized by multi-chip (M/#chip) */
      packed *a[H][NCHIP], *a0[H][NCHIP];
      packed *b, *b0[H];
      float  *c[H][NCHIP], *c0[H][NCHIP];
      b = B32_P+blk*LP;
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        for (k=0; k<H; k++) {
          a[k][CHIP]  = A32_P+(CHIP*M1/NCHIP+top+k)*LP;
          c[k][CHIP]  = C32_1+(CHIP*M1/NCHIP+top+k)*M2+blk;
          c0[k][CHIP] = c[k][CHIP]+0;
        }
      }

#define sparse_core1(r, h) \
  mex(OP_CMPA_LE,   &b0[h], INIT0?b:b0[h], INIT0?0:8, OP_CMPA_GE, &a0[h][CHIP], INIT0?a[h][CHIP]:a0[h][CHIP], INIT0?0:8, 0LL, BR[r][2][1], BR[r][2][0]);/*|yx|B|A| prefix¤È¤·¤Æ»ÈÍÑ.col2[1/0]ÇÛÃÖ³ÎÄê.LMM[2/1]ÇÛÃÖ³ÎÄê.bit63-32¤òÈæ³Ó*/\
  mop(OP_LDR,   3,  &BR[r][2][1], b0[h],                        bofs,        MSK_W1,      b,          2*LP*RMGRP,  0, 0,      NULL,      2*LP*RMGRP);/*LMM[2]³ÎÄê   LD¼Â¹Ô¤Ïcol2*/\
  mop(OP_LDR,   3,  &BR[r][2][0], a0[h][CHIP],                  bofs,        MSK_W0,      a[h][CHIP], 2*LP,        0, 0,      NULL,      2*LP);      /*LMM[1]´Ö¼Ú¤ê LD¼Â¹Ô¤Ïcol2*/\
  exe(OP_NOP,       &AR[r][0],    0LL,                          EXP_H3210,   0,           EXP_H3210,  0,           EXP_H3210, OP_NOP, 0, OP_NOP, 0);\
  mop(OP_LDWR,  1,  &c00,         c0[h][CHIP],                  oofs,        MSK_W0,      c[h][CHIP], RMGRP,       0, 1,      NULL,      RMGRP);\
  exe(OP_CFMA,      &c00,         INIT0?c00:c00,                EXP_H3210,   BR[r][2][1], EXP_H3210,  BR[r][2][0], EXP_H3210, OP_NOP, 0, OP_NOP, 0);\
  mop(OP_STWR,  1,  &c00,         oofs,                         c0[h][CHIP], MSK_D0,      c[h][CHIP], RMGRP,       0, 1,      NULL,      RMGRP)

//EMAX5A begin imax mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-LP*8)<<32|((0-4LL)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=LP,cofs=(0LL)<<32|((0LL)&0xffffffff); LOOP0--; INIT0=0) {         /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &rofs, rofs,            EXP_H3210, INIT0?(LP*8)<<32|(4LL):0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &bofs, rofs,            EXP_H3210, 0LL,                      EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
            exe(OP_ADD,    &oofs, rofs,            EXP_H3210, 0LL,                      EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

            sparse_core1(  2,  0);
            sparse_core1(  3,  1); /* H=2 */
            sparse_core1(  4,  2);
            sparse_core1(  5,  3); /* H=4 */
            sparse_core1(  6,  4);
            sparse_core1(  7,  5);
            sparse_core1(  8,  6);
            sparse_core1(  9,  7); /* H=8 */
            sparse_core1( 10,  8);
            sparse_core1( 11,  9);
            sparse_core1( 12, 10);
            sparse_core1( 13, 11);
            sparse_core1( 14, 12);
            sparse_core1( 15, 13);
            sparse_core1( 16, 14);
            sparse_core1( 17, 15); /* H=16 */
          }
        }
      }
//EMAX5A end
      count1+=H*RMGRP*LP;
    }
  }
//EMAX5A drain_dirty_lmm
#else
  /* M1/NCHIP/H * M2/RMGRP * NCHIP * RMGRP * L/8 * 8*H = M1*M2*L */
  /* col3  col2  col1  col0               */
  /* LD B  LD B  LD C  LD C               */
  /* LD A  LD A  EXST  EXST  fold¤Ïcol0   */
  /*  B+A   B+A   C     C  Æ°ºî¥¿¥¤¥ß¥ó¥° */
  /*         B    A     C  LMM¤ÎÇÛÃÖ      */
  /*       RMGRP LP*2  2W                 */
  for (blk=0; blk<M2; blk+=RMGRP) { /* 3½Å¥ë¡¼¥×Å¸³«¤Î³°Â¦ÂÐ¾Ý */
    for (top=0; top<M1/NCHIP; top+=H) { /* will be parallelized by multi-chip (M/#chip) */
      packed *a[H][NCHIP], *a0[H][2][NCHIP]; /* a¤Ï¶¦ÄÌ,a0¤ÏÆ±°ì¹Ô¤ò»²¾È(°ÜÆ°¥Ñ¥¿¡¼¥ó¤¬°ã¤¦¤Î¤Ç2¤ÄÉ¬Í×) */
      packed *b[2],        *b0[H][2];        /* b¤Ï¶¦ÄÌ,b0¤Î2¹Ô¤ò1unit¤Ë¥Þ¥ë¥Á¥¹¥ì¥Ã¥É²½¤¹¤ì¤Ðc¤¬Ï¢Â³ */
      float  *c[H][NCHIP], *c0[H][2][NCHIP]; /* c¤Ï¶¦ÄÌ,c0¤ÏÏ¢Â³ */
      b[0] = B32_P+(blk+0)*LP;
      b[1] = B32_P+(blk+1)*LP;
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        for (k=0; k<H; k++) {
          a[k][CHIP]  = A32_P+(CHIP*M1/NCHIP+top+k)*LP;
          c[k][CHIP]  = C32_1+(CHIP*M1/NCHIP+top+k)*M2+blk;
          c0[k][0][CHIP] = c[k][CHIP]+0;
          c0[k][1][CHIP] = c[k][CHIP]+1;
        }
      }

#define sparse_core1(r, h) \
  mex(OP_CMPA_LE, &b0[h][0], INIT0?b[0]:b0[h][0], INIT0?0:8, OP_CMPA_GE, &a0[h][0][CHIP], INIT0?a[h][CHIP]:a0[h][0][CHIP], INIT0?0:8, 0LL, BR[r][2][1], BR[r][2][0]);/*|yx|B|A| prefix¤È¤·¤Æ»ÈÍÑ.col2[1/0]ÇÛÃÖ³ÎÄê.LMM[2/1]ÇÛÃÖ³ÎÄê.bit63-32¤òÈæ³Ó*/\
  mop(OP_LDR,   3,  &BR[r][2][1],    b0[h][0],                        bofs,           MSK_W1,      b[0],          2*LP*RMGRP,  0, 0,      NULL,      2*LP*RMGRP);/*LMM[2]³ÎÄê   LD¼Â¹Ô¤Ïcol2*/\
  mop(OP_LDR,   3,  &BR[r][2][0],    a0[h][0][CHIP],                  bofs,           MSK_W0,      a[h][CHIP],    2*LP,        0, 0,      NULL,      2*LP);      /*LMM[1]´Ö¼Ú¤ê LD¼Â¹Ô¤Ïcol2*/\
  mex(OP_CMPA_LE, &b0[h][1], INIT0?b[1]:b0[h][1], INIT0?0:8, OP_CMPA_GE, &a0[h][1][CHIP], INIT0?a[h][CHIP]:a0[h][1][CHIP], INIT0?0:8, 0LL, BR[r][3][1], BR[r][3][0]);/*|yx|B|A| prefix¤È¤·¤Æ»ÈÍÑ.col2[1/0]ÇÛÃÖ³ÎÄê.LMM[2/1]ÇÛÃÖ³ÎÄê.bit63-32¤òÈæ³Ó*/\
  mop(OP_LDR,   3,  &BR[r][3][1],    b0[h][1],                        bofs,           MSK_W1,      b[0],          2*LP*RMGRP,  0, 0,      NULL,      2*LP*RMGRP);/*LMM[2]³ÎÄê   LD¼Â¹Ô¤Ïcol2*/\
  mop(OP_LDR,   3,  &BR[r][3][0],    a0[h][1][CHIP],                  bofs,           MSK_W0,      a[h][CHIP],    2*LP,        0, 0,      NULL,      2*LP);      /*LMM[1]´Ö¼Ú¤ê LD¼Â¹Ô¤Ïcol2*/\
  exe(OP_NOP,       &AR[r][0],       0LL,                             EXP_H3210,      0,           EXP_H3210,     0,           EXP_H3210, OP_NOP, 0, OP_NOP, 0);\
  mop(OP_LDWR,  1,  &c00,            c0[h][0][CHIP],                  oofs,           MSK_W0,      c[h][CHIP],    RMGRP,       0, 1,      NULL,      RMGRP);\
  exe(OP_CFMA,      &c00,            INIT0?c00:c00,                   EXP_H3210,      BR[r][2][1], EXP_H3210,     BR[r][2][0], EXP_H3210, OP_NOP, 0, OP_NOP, 0);\
  mop(OP_STWR,  1,  &c00,            oofs,                            c0[h][0][CHIP], MSK_D0,      c[h][CHIP],    RMGRP,       0, 1,      NULL,      RMGRP);\
  mop(OP_LDWR,  1,  &c01,            c0[h][1][CHIP],                  oofs,           MSK_W0,      c[h][CHIP],    RMGRP,       0, 1,      NULL,      RMGRP);\
  exe(OP_CFMA,      &c01,            INIT0?c01:c01,                   EXP_H3210,      BR[r][3][1], EXP_H3210,     BR[r][3][0], EXP_H3210, OP_NOP, 0, OP_NOP, 0);\
  mop(OP_STWR,  1,  &c01,            oofs,                            c0[h][1][CHIP], MSK_D0,      c[h][CHIP],    RMGRP,       0, 1,      NULL,      RMGRP)

//EMAX5A begin imax mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP/2,rofs=(0-8*LP*2)<<32|((0-8LL)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=LP,cofs=(0LL)<<32|((0LL)&0xffffffff); LOOP0--; INIT0=0) {         /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &rofs, rofs,            EXP_H3210, INIT0?(8*LP*2)<<32|(8LL):0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &bofs, rofs,            EXP_H3210, 0LL,                        EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
            exe(OP_ADD,    &oofs, rofs,            EXP_H3210, 0LL,                        EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

            sparse_core1(  2,  0);
            sparse_core1(  3,  1); /* H=2 */
            sparse_core1(  4,  2);
            sparse_core1(  5,  3); /* H=4 */
            sparse_core1(  6,  4);
            sparse_core1(  7,  5);
            sparse_core1(  8,  6);
            sparse_core1(  9,  7); /* H=8 */
            sparse_core1( 10,  8);
            sparse_core1( 11,  9);
            sparse_core1( 12, 10);
            sparse_core1( 13, 11);
            sparse_core1( 14, 12);
            sparse_core1( 15, 13);
            sparse_core1( 16, 14);
            sparse_core1( 17, 15); /* H=16 */
          }
        }
      }
//EMAX5A end
      count1+=H*RMGRP*LP;
    }
  }
//EMAX5A drain_dirty_lmm
#endif
  get_nanosec(0);
  printf("imax.FMA = %d\n", count1);
  show_nanosec();
}
