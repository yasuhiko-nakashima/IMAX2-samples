
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

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif

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
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // б·б·б· RESET
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
#define M 128
#define RMGRP 1
#define NCHIP 4
#define W 4
#define H 1
Uint *A;  /*[M][M];*/
Uint *B;  /*[M][M];*/
Uint *C0; /*[M][M];*/
Uint *C1; /*[M][M];*/
int row, col, n;
int top, blk;
int w, h;
int count0, count1, count2;

main()
{
  sysinit(M*M*sizeof(Uint)
         +M*M*sizeof(Uint)
         +M*M*sizeof(Uint)
         +M*M*sizeof(Uint),32);
  printf("membase: %08.8x\n", (Uint)membase);
  A  = (Uint*)membase;
  B  = (Uint*)((Uchar*)A  + M*M*sizeof(Uint));
  C0 = (Uint*)((Uchar*)B  + M*M*sizeof(Uint));
  C1 = (Uint*)((Uchar*)C0 + M*M*sizeof(Uint));
  printf("A : %08.8x\n", A);
  printf("B : %08.8x\n", B);
  printf("C0: %08.8x\n", C0);
  printf("C1: %08.8x\n", C1);

  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      C0[row*M+col] = row*256+col+1;
      C1[row*M+col] = row*256+col+1;
    }
  }

  reset_nanosec();
  orig();
  get_nanosec(NANOS_ARM);
  show_nanosec();

  reset_nanosec();
  imax();
  get_nanosec(NANOS_ARM);
  show_nanosec();

  printf("Num of MULT: orig=%d imax=%d\n", count0, count1);

  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      if (C0[row*M+col] != C1[row*M+col]) {
        count2++;
        printf("C0[%d][%d]=%d C1[%d][%d]=%d\n", row, col, C0[row*M+col],
                                                row, col, C1[row*M+col]);
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
}

orig() {
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  cofs, rofs, oofs;
  printf("<<<ORIG>>>\n");
  for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
    for (INIT1=1,LOOP1=RMGRP,rofs=0-M; LOOP1--; INIT1=0) {                             /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
      rofs+=M;
      for (INIT0=1,LOOP0=M/W,cofs=(0-W); LOOP0--; INIT0=0) {      /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
	cofs+=W;
	oofs=rofs+cofs;
        C0[(CHIP*M/NCHIP)*M+oofs+0] += 1;
        C0[(CHIP*M/NCHIP)*M+oofs+1] += 1;
        C0[(CHIP*M/NCHIP)*M+oofs+2] += 1;
        C0[(CHIP*M/NCHIP)*M+oofs+3] += 1;
        count0++;
        /*printf("[%d %d %d]", row, col, n);*/
      }
      /*printf("\n");*/
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
  Ull  cofs, rofs, oofs;
  /*  игибибибибибидconvolutionд╬╛ь╣ч                                                  */
  /*  ивигибибибибикидBдм╩г┐Їд╚╣═диды                                                  */
  /*  ививигибибибибикидигибибибибибидид        игибибибибибидид                       */
  /*  ивививb         ививa a a a a ививRMGRP   ивo o o o o ививRMGRP                  */
  /*  ивививb         ийив          ивий/CHIP   ив          ивий/CHIP                  */
  /*  ивививb   B0   bивив A(weight)ивив        ив   out    ивив mmд╬╛ь╣чд╧╣╘д╟╩м│ф    */
  /*  ижививb        lийив          ивий        ив          ивий cnnд╬╛ь╣чд╧outд╟╩м│ф  */
  /*    иживb        kививblk       ивив        ивblk       ивив                       */
  /*      ижибибибибибиеижибикибикибиеие        ижибикибикибиеие                       */
  printf("<<<IMAX>>>\n");
  typedef struct {Uint i[4]} Ui4;
  Ui4  *c0[NCHIP];
  Ui4  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];

  for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
    c0[CHIP] = C1+(CHIP*M/NCHIP)*M;
    c00[CHIP]= (Uint*)c0[CHIP]+0, c01[CHIP]= (Uint*)c0[CHIP]+1, c02[CHIP]= (Uint*)c0[CHIP]+2, c03[CHIP]= (Uint*)c0[CHIP]+3;
  }
//EMAX5A begin x1 mapdist=0
  for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
    for (INIT1=1,LOOP1=RMGRP,rofs=0-M*4; LOOP1--; INIT1=0) {                             /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
      for (INIT0=1,LOOP0=M/W,cofs=0-W*4; LOOP0--; INIT0=0) {      /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
	exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, W*4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);/* stage#0 */
	exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?M*4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#0 */
	exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);         /* stage#1 */

	mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);  /* stage#62 */
	mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);  /* stage#62 */
	mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);  /* stage#62 */
	mop(OP_LDWR,   1, &BR[2][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);  /* stage#62 */
	exe(OP_ADD, &AR[2][0], 1LL, EXP_H3210,  BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#62 */
	exe(OP_ADD, &AR[2][1], 1LL, EXP_H3210,  BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#62 */
	exe(OP_ADD, &AR[2][2], 1LL, EXP_H3210,  BR[2][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#62 */
	exe(OP_ADD, &AR[2][3], 1LL, EXP_H3210,  BR[2][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#62 */
	mop(OP_STWR,   1, &AR[2][0], (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);      /* stage#62 */
	mop(OP_STWR,   1, &AR[2][1], (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);      /* stage#62 */
	mop(OP_STWR,   1, &AR[2][2], (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);      /* stage#62 */
	mop(OP_STWR,   1, &AR[2][3], (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], M*RMGRP, 0, 1, (Ull)NULL, M*RMGRP);      /* stage#62 */
      }
    }
  }
//EMAX5A end
//EMAX5A drain_dirty_lmm
}
