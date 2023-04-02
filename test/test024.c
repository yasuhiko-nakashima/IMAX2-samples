
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

#define M1 4096LL
#define M2 4096LL
//#define M1 64LL
//#define M2 64LL
#define LP 8LL

typedef struct {
  float d; /* low */
  Uint  x; /* high */
} packed;

float  *A32_0; /*[M1][L]         */
packed *A32_P; /*[M1][L]   pack  */
float  *B32_0; /*[M2][L] T       */
packed *B32_P; /*[M2][L] T pack  */
Ull    *Bas1P;
int row, col, n;
int top, blk;
int w, h;
int count0, count1, count2;

main()
{
  sysinit((Uint)(M1*M2 *sizeof(Uint)
                +M1*M2 *sizeof(packed)
                +M1*M2 *sizeof(Uint)
                +M1*M2 *sizeof(packed)
                +1     *sizeof(Ull)),32);
  printf("membase: %08.8x_%08.8x\n", (Uint)((Ull)membase>>32), (Uint)membase);
  A32_0 = (Uint*)membase;
  A32_P = (Uint*)((Uchar*)A32_0 + M1*M2 *sizeof(Uint));
  B32_0 = (Uint*)((Uchar*)A32_P + M1*M2 *sizeof(packed));
  B32_P = (Uint*)((Uchar*)B32_0 + M1*M2 *sizeof(Uint));
  Bas1P = (Ull*) ((Uchar*)B32_P + M1*M2 *sizeof(packed));
  printf("A32_0: %08.8x\n", (Uint)A32_0);
  printf("A32_P: %08.8x\n", (Uint)A32_P);
  printf("B32_0: %08.8x\n", (Uint)B32_0);
  printf("B32_P: %08.8x\n", (Uint)B32_P);
  printf("Bas1P: %08.8x\n", (Uint)Bas1P);

  WD      = M1;
  HT      = M2;
  BITMAP  = WD*HT;
  SCRWD   = 4;
  SCRHT   = 2;
  VECWD   = 1;
  VECHT   = 1;
  VECSTEP = 4;

  /**************************************************/
  /* A                                              */
  /**************************************************/
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      if (abs(row-col) < LP/2)
        A32_0[row*M2+col] = (float)(row-col)/(M2/(128.0/(float)LP));
      else
        A32_0[row*M2+col] = 0.0;
      //printf(" %08.8x", *(Uint*)&A32_0[row*M2+col]);
    }
    //printf("\n");
  }
  //printf("\n");

  /**************************************************/
  /* B                                              */
  /**************************************************/
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      if (abs(row-col) < LP/2)
        B32_0[row*M2+col] = (float)(row-col)/(M2/(128.0/(float)LP));
      else
        B32_0[row*M2+col] = 0.0;
      //printf(" %08.8x", *(Uint*)&B32_0[row*M2+col]);
    }
    //printf("\n");
  }

  orig();
  imax();

#if 1
  count2 = 0;
  for (n=0; n<count0; n++) {
    packed origA = A32_P[n];
    packed imaxB = B32_P[n];
    if (origA.d != imaxB.d) {
      count2++;
      printf("origA[%d]=%08.8x.%08.8x imaxB[%d]=%08.8x.%08.8x\n", n, origA.x, *(Uint*)&origA.d, n, imaxB.x, *(Uint*)&imaxB.d);
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
#endif
}

orig() {
  printf("<<<ORIG>>>\n");
  reset_nanosec();

  //pack(A32_P, A32_0);
  count0 = 0;
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      if (A32_0[row*M2+col] == 0)
        continue;
      if (count0 >= M1*M2)
        continue;
      A32_P[count0].d = A32_0[row*M2+col];
      A32_P[count0].x = row*M2+col;
      count0++;
    }
  }

  get_nanosec(0);
  printf("orig.PACK = %d\n", count0);
  show_nanosec();
#if 0
  for (row=0; row<M1; row++) {
    for (col=0; col<LP; col++)
      printf(" %08.8x_%08.8x", (Uint)A32_P[row*LP+col].x, *(Uint*)&A32_P[row*LP+col].d);
    printf("\n");
  }
#endif
}

imax() {
#undef  NCHIP
#undef  RMGRP
#undef  W
#undef  H
#define NCHIP 1
#define RMGRP 2
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
  Uint *ibase0, *itop0 = NULL, *itop1;
  Ull  *obase0, *otop0 = NULL, *otop1;
  Ull  rofs, cofs, oofs, std;
  int  i;

  printf("<<<IMAX>>>\n");
  reset_nanosec();
  
  *Bas1P = B32_P-1; /* end of B32_0 */

  for (i=0; i<M1; i+=RMGRP) {
    r1     = (Ull)(i*M2-1)<<32;
    ibase0 = B32_0+i*M2;
    itop0  = B32_0+i*M2;
    itop1  = itop0+RMGRP*M2;
    obase0 = *Bas1P;   /* end of B32_0 */
    otop1  = otop0;
    otop0  = *Bas1P+8; /* top of B32_P */

#if 0
//no-prefetch/post-drain
//EMAX5A begin imax mapdist=0
/*3*/for (CHIP=0; CHIP<NCHIP; CHIP++) {
  /*2*/for (INIT1=1,LOOP1=RMGRP,rofs=0; LOOP1--; INIT1=0) {
    /*1*/for (INIT0=1,LOOP0=M2,cofs=0; LOOP0--; INIT0=0) {
           mop(OP_LDWR, 1,   &r0,       ibase0++,          0,             MSK_D0,    itop0, M2*RMGRP,    0,       0,    NULL,   M2*RMGRP);
	   exe(OP_ADD,       &r1,       r1,     EXP_H3210, 0x100000000LL, EXP_H3210, 0,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   exe(OP_NOP,       &std,      r1,     EXP_H3210, 0,             EXP_H3210, 0,     EXP_H3210,   OP_OR,   r0,   OP_NOP, 0LL);
	   exe(OP_CMP_EQ,    &cc0,      r0,     EXP_H1010, 0x00000000LL,  EXP_H1010, 0,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   exe(OP_CMP_EQ,    &cc1,      r0,     EXP_H1010, 0x80000000LL,  EXP_H1010, 0,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   exe(OP_NOP,       &cc2,      cc0,    EXP_H3210, 0,             EXP_H3210, 0,     EXP_H1010,   OP_OR,   cc1,  OP_NOP, 0LL);
	   exe(OP_CMOV,      &oofs,     cc2,    EXP_H3210, 0,             EXP_H3210, 8,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   exe(OP_ADD,       &obase0,   obase0, EXP_H3210, oofs,          EXP_H3210, 0,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   mop(OP_STR,  3,   &obase0,   Bas1P,             0,             MSK_D0,    Bas1P, 2,           0,       0,    NULL,   2);
	   exe(OP_NOP,       &AR[5][0], 0,      EXP_H3210, 0,             EXP_H3210, 0,     EXP_H1010,   OP_NOP,  0,    OP_NOP, 0LL);
	   cex(OP_CEXE,      &ex0,      0, 0, 0, cc2, 0x0001);
	   mop(OP_STR,  ex0, &std,      obase0,            0,             MSK_D0,    otop0, LP*2*RMGRP,  0,       0,    NULL,   LP*2*RMGRP);
         }
       }
     }
//EMAX5A end
//EMAX5A drain_dirty_lmm
  }
#else
//with-prefetch/post-drain
//EMAX5A begin imax mapdist=0
/*3*/for (CHIP=0; CHIP<NCHIP; CHIP++) {
  /*2*/for (INIT1=1,LOOP1=RMGRP,rofs=0; LOOP1--; INIT1=0) {
    /*1*/for (INIT0=1,LOOP0=M2,cofs=0; LOOP0--; INIT0=0) {
           mop(OP_LDWR, 1,   &r0,       ibase0++,          0,             MSK_D0,    itop0, M2*RMGRP,    0,       0,    itop1,  M2*RMGRP);
	   exe(OP_ADD,       &r1,       r1,     EXP_H3210, 0x100000000LL, EXP_H3210, 0,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   exe(OP_NOP,       &std,      r1,     EXP_H3210, 0,             EXP_H3210, 0,     EXP_H3210,   OP_OR,   r0,   OP_NOP, 0LL);
	   exe(OP_CMP_EQ,    &cc0,      r0,     EXP_H1010, 0x00000000LL,  EXP_H1010, 0,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   exe(OP_CMP_EQ,    &cc1,      r0,     EXP_H1010, 0x80000000LL,  EXP_H1010, 0,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   exe(OP_NOP,       &cc2,      cc0,    EXP_H3210, 0,             EXP_H3210, 0,     EXP_H1010,   OP_OR,   cc1,  OP_NOP, 0LL);
	   exe(OP_CMOV,      &oofs,     cc2,    EXP_H3210, 0,             EXP_H3210, 8,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   exe(OP_ADD,       &obase0,   obase0, EXP_H3210, oofs,          EXP_H3210, 0,     EXP_H3210,   OP_NOP,  0,    OP_NOP, 0LL);
	   mop(OP_STR,  3,   &obase0,   Bas1P,             0,             MSK_D0,    Bas1P, 2,           0,       0,    NULL,   2);
	   exe(OP_NOP,       &AR[5][0], 0,   EXP_H3210,    0,             EXP_H3210, 0,     EXP_H1010,   OP_NOP,  0,    OP_NOP, 0LL);
	   cex(OP_CEXE,      &ex0,      0, 0, 0, cc2, 0x0001);
	   mop(OP_STR,  ex0, &std,      obase0,            0,             MSK_D0,    otop0, LP*2*RMGRP,  0,       0,    otop1,  LP*2*RMGRP);
         }
       }
     }
//EMAX5A end
//EMAX5A drain_dirty_lmm
  }
#endif

  count1 = (packed*)*Bas1P-(packed*)B32_P+1;

  get_nanosec(0);
  printf("Bas1P=%08.8x_%08.8x Packed=%d\n", (Uint)(*Bas1P>>32), (Uint)*Bas1P, count1);
  show_nanosec();
#if 0
  for (row=0; row<M1; row++) {
    for (col=0; col<LP; col++)
      printf(" %08.8x_%08.8x", (Uint)B32_P[row*LP+col].x, *(Uint*)&B32_P[row*LP+col].d);
    printf("\n");
  }
#endif
}
