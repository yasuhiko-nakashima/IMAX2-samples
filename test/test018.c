
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
#define M 16
#define RMGRP 1
#define NCHIP 1
#define W 2
#define H 1
float *A;  /*[M][M];*/
float *B;  /*[M][M];*/
float *C;  /*[M][M];*/
float *D;  /*[M][M];*/
float *E;  /*[M][M];*/
int row, col, n;
int top, blk;
int w, h;
int count0, count1, count2;

main()
{
  sysinit(M*sizeof(Uint)
         +M*sizeof(Uint)
         +M*sizeof(Uint)
         +M*sizeof(Uint)
         +M*sizeof(Uint),32);
  printf("membase: %08.8x\n", (Uint)membase);
  A = (Uint*)membase;
  B = (Uint*)((Uchar*)A  + M*sizeof(Uint));
  C = (Uint*)((Uchar*)B  + M*sizeof(Uint));
  D = (Uint*)((Uchar*)C  + M*sizeof(Uint));
  E = (Uint*)((Uchar*)D  + M*sizeof(Uint));
  printf("A : %08.8x\n", A);
  printf("B : %08.8x\n", B);
  printf("C : %08.8x\n", C);
  printf("D : %08.8x\n", D);
  printf("E : %08.8x\n", E);

  for (row=0; row<M; row++) {
    A[row] = row;
    B[row] = row;
    C[row] = row;
    D[row] = row;
    E[row] = row;
  }

  imax();
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
  Ull a00, b00, b01, c00, c01, d00, d01, e00, e01;

  printf("=====================\n");
  for (row=0; row<M; row++)
    printf(" %f %f\n", D[row], E[row]);

//EMAX5A begin inv_x0 mapdist=0
  for (INIT0=1,LOOP0=M,cofs=0-4; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
    exe(OP_ADD, &cofs, cofs, EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);       /* stage#0 */
    mop(OP_LDWR,  1, &a00,         A,       cofs, MSK_W0, A, M, 0, 0, NULL, M);                                       /* stage#1.0 inv_x1の同位置topと偶然一致するとrdy=1となりkick_dmaが出ない */
    mop(OP_LDWR,  1, &BR[1][2][1], B,       cofs, MSK_W0, B, M, 0, 1, NULL, M);                                       /* stage#1.2 *//*OK read-modify-write + exe-loop*/
    mop(OP_LDWR,  1, &BR[1][3][1], C,       cofs, MSK_W0, C, M, 0, 1, NULL, M);                                       /* stage#1.3              LD     */
    mop(OP_LDWR,  1, &d00,         D,       0,    MSK_W0, D, M, 0, 1, NULL, M);                                         /* stage#2.0              |      */
    mop(OP_LDWR,  1, &d01,         E,       8,    MSK_W0, E, M, 0, 1, NULL, M);                                         /* stage#2.1  +->xxx      |      */
    exe(OP_FMA,      &d00,         d00,     EXP_H3210, a00, EXP_H3210, BR[1][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2.0  |   ■■■  | 2.0  */
    exe(OP_FMA,      &d01,         d01,     EXP_H3210, a00, EXP_H3210, BR[1][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2.1  +- xxx      |      */
    mop(OP_STWR,  1, &d00,         0,       D,    MSK_D0, D, M,   0, 1, NULL, M);                                         /* stage#2.0  |    + ST   v      */
    mop(OP_STWR,  1, &d01,         8,       E,    MSK_D0, E, M,   0, 1, NULL, M);                                         /* stage#2.1  +--------- xxx     */
  }
//EMAX5A end
//EMAX5A drain_dirty_lmm

  printf("=====================\n");
  for (row=0; row<M; row++)
    printf(" %f %f %f\n", A[row], D[row], E[row]);

//EMAX5A begin inv_x1 mapdist=0
  for (CHIP=0; CHIP<NCHIP; CHIP++) {
    for (INIT0=1,LOOP0=M,cofs=0-4; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
      exe(OP_ADD, &cofs, cofs, EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
      exe(OP_CMP_LT,   &cc0,         0,           EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#0              LD     */
      mop(OP_LDWR,  1, &BR[1][2][1], B,         cofs, MSK_W0, B,        M, 0, 0, NULL, M);  /* A[p[i]*M+k]                       stage#1              |      */
      mop(OP_LDWR,  1, &BR[1][0][1], A,         cofs, MSK_W0, A,        M, 0, 1, NULL, M);  /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#1  +->         |      */
      exe(OP_FMA,      &AR[1][0],    BR[1][0][1], EXP_H3210, BR[1][0][1], EXP_H3210, BR[1][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#1  |   ■■■  | 1.0  */
      cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#1  |  AR[1]    |      */
      mop(OP_STWR,ex0, &AR[1][0],    cofs,  A,    MSK_D0, A, M, 0, 1, NULL, M);                                    /* stage#1  |    + ST   v      */
    }
  }
//EMAX5A end
//EMAX5A drain_dirty_lmm

  printf("=====================\n");
  for (row=0; row<M; row++)
    printf(" %f %f %f\n", A[row], D[row], E[row]);

//EMAX5A begin inv_x2 mapdist=0
  for (CHIP=0; CHIP<NCHIP; CHIP++) {
    for (INIT0=1,LOOP0=M,cofs=0-4; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
      exe(OP_ADD, &cofs, cofs, EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
      mop(OP_LDWR,  1, &a00,         A,          cofs, MSK_W0, A,        M, 0, 0, NULL, M);  /* A[p[j]*M+k]                  *//* stage#1.0 inv_x1の同位置topと偶然一致するとrdy=1となりkick_dmaが出ない */
      mop(OP_LDWR,  1, &BR[1][2][1], B,   cofs, MSK_W0, B, M, 0, 1, NULL, M);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#1.2 *//*OK read-modify-write + exe-loop*/
      mop(OP_LDWR,  1, &BR[1][3][1], C,   cofs, MSK_W0, C, M, 0, 1, NULL, M);  /* b[(i+CHIP*W*H+h*W+1)*M+k]    *//* stage#1.3              LD     */
      mop(OP_LDWR,  1, &b00,        D,   0,    MSK_W0, D, 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#2.0              |      */
      mop(OP_LDWR,  1, &b01,        E,   0,    MSK_W0, E, 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+1)*M+j]    *//* stage#2.1  +->xxx      |      */
      exe(OP_FMS,      &b00,        b00,         EXP_H3210, a00, EXP_H3210, BR[1][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#2.0  |   ■■■  | 2.0  */
      exe(OP_FMS,      &b01,        b01,         EXP_H3210, a00, EXP_H3210, BR[1][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#2.1  +- xxx      |      */
      mop(OP_STWR,  1, &b00,        0,      D, MSK_D0, D, 1,   0, 1, NULL, 1);                                      /* stage#2.0  |    + ST   v      */
      mop(OP_STWR,  1, &b01,        0,      E, MSK_D0, E, 1,   0, 1, NULL, 1);                                      /* stage#2.1  +--------- xxx     */
    }
  }
//EMAX5A end
//EMAX5A drain_dirty_lmm

  printf("=====================\n");
  for (row=0; row<M; row++)
    printf(" %f %f %f\n", A[row], D[row], E[row]);

//EMAX5A begin inv_x3 mapdist=0
  for (CHIP=0; CHIP<NCHIP; CHIP++) {
    for (INIT0=1,LOOP0=M,cofs=0-4; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
      exe(OP_ADD, &cofs, cofs, EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
      mop(OP_LDWR,  1, &a00,         A,          cofs, MSK_W0, A,        M, 0, 0, NULL, M);  /* A[p[j]*M+k]                  *//* stage#1.0 inv_x1の同位置topと偶然一致するとrdy=1となりkick_dmaが出ない */
      mop(OP_LDWR,  1, &BR[1][2][1], B,   cofs, MSK_W0, B, M, 0, 1, NULL, M);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#1.2 *//*OK read-modify-write + exe-loop*/
      mop(OP_LDWR,  1, &BR[1][3][1], C,   cofs, MSK_W0, C, M, 0, 1, NULL, M);  /* b[(i+CHIP*W*H+h*W+1)*M+k]    *//* stage#1.3              LD     */
      mop(OP_LDWR,  1, &b00,        D,   0,    MSK_W0, D, 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#2.0              |      */
      mop(OP_LDWR,  1, &b01,        E,   0,    MSK_W0, E, 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+1)*M+j]    *//* stage#2.1  +->xxx      |      */
      exe(OP_FMS,      &b00,        b00,         EXP_H3210, a00, EXP_H3210, BR[1][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#2.0  |   ■■■  | 2.0  */
      exe(OP_FMS,      &b01,        b01,         EXP_H3210, a00, EXP_H3210, BR[1][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#2.1  +- xxx      |      */
      mop(OP_STWR,  1, &b00,        0,      D, MSK_D0, D, 1,   0, 1, NULL, 1);                                      /* stage#2.0  |    + ST   v      */
      mop(OP_STWR,  1, &b01,        0,      E, MSK_D0, E, 1,   0, 1, NULL, 1);                                      /* stage#2.1  +--------- xxx     */
    }
  }
//EMAX5A end
//EMAX5A drain_dirty_lmm

  printf("=====================\n");
  for (row=0; row<M; row++)
    printf(" %f %f %f\n", A[row], D[row], E[row]);
}
