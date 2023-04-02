
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
#endif

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif

Uchar* membase;

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
  if ((int)membase & (alignment-1))
    membase = (void*)(((int)membase & ~(alignment-1))+alignment);
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
/*#define M 4096*/
#define M 4
#define RMGRP 2
/*#define NCHIP 4*/
#define NCHIP 1
/*#define W 1*/
#define H 1
volatile float *A0;  /*[M][M];*/
volatile float *A;   /*[M][M];*/
volatile Uint  *p;   /*[M];*/
volatile float *inv0;/*[M][M];*/
volatile float *inv1;/*[M][M];*/
volatile float *b;   /*[M][M];*/
volatile float *x;   /*[M][M];*/
volatile float *C;   /*[M][M];*/
int top, blk, h;
int count0, count1, count2;

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define ERRTH  (5.0E-4)
#define abs(a) ((a)>0?(a):-(a))

main()
{
  int i, j, k;

  sysinit(M*M*sizeof(float)
         +(M+RMGRP)*M*sizeof(float)
         +(M+M)*sizeof(Uint) /*奇数では×*/
         +M*M*sizeof(float)
         +M*M*sizeof(float)
         +M*M*sizeof(float)
         +M*M*sizeof(float)
         +M*M*sizeof(float),32);
  printf("membase: %08.8x\n", (Uint)membase);
  A0  = (float*)membase;
  A   = (float*)((Uchar*)A0  + M*M*sizeof(float));
  p   = (Uint*) ((Uchar*)A   +(M+RMGRP)*M*sizeof(float));
  inv0= (float*)((Uchar*)p   +(M+M)*sizeof(Uint));
  inv1= (float*)((Uchar*)inv0+ M*M*sizeof(float));
  b   = (float*)((Uchar*)inv1+ M*M*sizeof(float));
  x   = (float*)((Uchar*)b   + M*M*sizeof(float));
  C   = (float*)((Uchar*)x   + M*M*sizeof(float));
  printf("A0  : %08.8x\n", A0);
  printf("A   : %08.8x\n", A);
  printf("p   : %08.8x\n", p);
  printf("inv0: %08.8x\n", inv0);
  printf("inv1: %08.8x\n", inv1);
  printf("b   : %08.8x\n", b);
  printf("x   : %08.8x\n", x);
  printf("C   : %08.8x\n", C);

  srand(100);
  /*  入力行列を作成  */
  for (i=0; i<M; i++) {
    for (j=0; j<M; j++)
      A[i*M+j] = A0[i*M+j] = (float)(i%M+j);
  }
  A[0] = A0[0] = 1;
  for (j=1;j<M;j++)
    A[j*M+j] = A0[j*M+j] = 3;

  imax();

  /* 検算 */
  for (i=0; i<M; i++) {
    for (j=0; j<M; j++) {
      for (k=0; k<M; k++) {
        if (k==0) C[i*M+j]  = A0[i*M+k] * inv1[k*M+j];
        else      C[i*M+j] += A0[i*M+k] * inv1[k*M+j];
      }
      if (i == j && fabsf(C[i*M+j]-1.0)>ERRTH) {
	count2++;
	printf("A*A'!=E C[%d][%d]=%f\n", i, j, C[i*M+j]);
      }
      else if (i != j && (fabsf(C[i*M+j])>ERRTH)) {
	count2++;
	printf("A*A'!=E C[%d][%d]=%f\n", i, j, C[i*M+j]);
      }
    }
  }
  if (count2)
    printf("A*A'!=E (ERRTH=%f) Num of diffs: %d\n", ERRTH, count2);
  else
    printf("A*A'==E (ERRTH=%f) Confirmed\n", ERRTH);
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
  int  i, j, k;
  Ull  cofs, rofs, oofs;
  float pmax, tmp;
  printf("<<<IMAX>>>\n");

#if 0
  printf("==========================\n");
  {Uint ii, jj;  for (ii=0; ii<M; ii++) { for (jj=0; jj<M; jj++) printf(" %08.8x", *(Uint*)&A[ii*M+jj]); printf("\n"); }}
#endif

  /* LU分解 */
  for (i=0; i<M+1; i++)
    p[i] = i;
  for (i=0; i<M; i++) { /* 列方向 */
    pmax = 0.0;
    k = -1;
    for (j=i; j<M; j++) { /* 行方向に探索 */
      if (pmax < fabsf(A[j*M+i])) {
	pmax = fabsf(A[j*M+i]);
	k = j;
      }
    }
    if (k == -1) {
      fprintf(stderr, "can't solve\n");
      exit(1);
    }
    j = p[k]; p[k] = p[i]; p[i] = j;
    for (j=0; j<M; j++) { /* real pivotting */            /*★*/
      tmp = A[k*M+j]; A[k*M+j] = A[i*M+j]; A[i*M+j] = tmp;/*★*/
    }                                                     /*★*/
    A[i*M+i] = 1.0/A[i*M+i];                              /*★*/
    for (j=i+1; j<M; j++) /* 行方向 */
      A[j*M+i] *= A[i*M+i];

    Uint *top  = &A[i*M+i];
    Uint *topw = (Ull)top;
    Uint  len  = M-i;
    Uint  len2 = len+(RMGRP-1)*M;
    Uint  grp;
    for (j=i+1; j<M; j+=NCHIP*H*RMGRP) { /* 行方向 */
      /********************************************/
      Uint  l00[NCHIP],  l01[NCHIP],  l02[NCHIP],  l03[NCHIP],  l04[NCHIP],  l05[NCHIP],  l06[NCHIP],  l07[NCHIP]; /* j<M-(h*NCHIP+CHIP) */
      Uint *d00[NCHIP], *d01[NCHIP], *d02[NCHIP], *d03[NCHIP], *d04[NCHIP], *d05[NCHIP], *d06[NCHIP], *d07[NCHIP]; /* A[p[j+h*NCHIP+CHIP]*M+k] */
      Uint *d00w[NCHIP],*d01w[NCHIP],*d02w[NCHIP],*d03w[NCHIP],*d04w[NCHIP],*d05w[NCHIP],*d06w[NCHIP],*d07w[NCHIP];/* A[p[j+h*NCHIP+CHIP]*M+k] */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	l00[CHIP]=(j+ 0*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 0*NCHIP*RMGRP+CHIP*RMGRP):M;l01[CHIP]=(j+ 1*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 1*NCHIP*RMGRP+CHIP*RMGRP):M;
	d00[CHIP] = &A[l00[CHIP]*M+i];   d01[CHIP] = &A[l01[CHIP]*M+i];   d02[CHIP] = &A[l02[CHIP]*M+i];   d03[CHIP] = &A[l03[CHIP]*M+i];
	d00w[CHIP]= (Ull)d00[CHIP];      d01w[CHIP]= (Ull)d01[CHIP];      d02w[CHIP]= (Ull)d02[CHIP];      d03w[CHIP]= (Ull)d03[CHIP];
      }
//EMAX5A begin inv_x1 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) {
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=0-M*4; LOOP1--; INIT1=0) {                             /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=M-(i+1),cofs=0; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD, &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD, &rofs, rofs, EXP_H3210, INIT0?M*4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                     /* stage#0 */
            exe(OP_ADD, &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);           /* stage#1 */
        /*for (k=i+1; k<M; k++) {*/ /* 最内列方向 */
          /*for (h=0; h<H; h++) {*/ /* vertical (parallel) execution */
          /*  if (j+h*NCHIP+CHIP<M) A[p[j+h*NCHIP+CHIP]*M+k] -= A[p[j+h*NCHIP+CHIP]*M+i]*A[p[i]*M+k];*/ /* 後続の逆行列と異なり,accumurateではなく要素毎の単独減算の繰返し */
	                                                                                                /* const:A[p[j]][0] * LMM A[p[  0]][*] */
                                                                                                        /*        ↓                           */
	    /*   v A[p[j]*M+i]         */                                                               /*   LMM A[p[j>0]][*] accumulate (column方向にj,j+1,..,479のため依存無) */
	    /***************************/
	    /* + - - - - - - - - - - - */ /* A[p[i]] 先頭行       */ /* 先頭行はi更新まで再利用可能 */
	    /* | * > > > > > > > > > > */ /* A[p[j]] 次行から引く */ /* 1行をLMMに写像 */
	    /* | v + - - - - - - - - - */ 
	    /* | v | * > > > > > > > > */ /* M/60を収容してi更新までj+=60を繰り返す *//* 行番号比較とcstによる端数制御 */
	    /* | v | v + - - - - - - - */ /* + CHIP#0 h=0 grp=0 */
	    /* | v | v - + - - - - - - */ /* + CHIP#0 h=0 grp=1 */
	    /* | v | v - - + - - - - - */ /* + CHIP#1 h=0 grp=0 */
	    /* | v | v - - - + - - - - */ /* + CHIP#1 h=0 grp=1 */
	    /* | v | v - - - - + - - - */ /* + CHIP#0 h=1 grp=0 */
	    /* | v | v - - - - - + - - */ /* + CHIP#0 h=1 grp=1 */
	    /* | v | v - - - - - - + - */ /* + CHIP#1 h=1 grp=0 */
	    /* | v | v - - - - - - - + */ /* + CHIP#1 h=1 grp=1 */
	    /***************************/ /* 最大60行まで写像可能 */
	    /* FOLDING時は,少なくとも第0列がFOLDINGであることが必要(conv-c2c仕様) */
	    /* CEXEにも関わらずSTWRの無意味なLMM入れ換えが発生するため,A[M][*](枠外領域)を使用 */                                           /*                 OK exe-loop */
	    exe(OP_CMP_LT,   &cc0,         l00[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#1              LD     */
	    mop(OP_LDWR,  1, &BR[2][2][1], top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k]                       stage#2              |      */
	    mop(OP_LDWR,  1, &BR[2][0][1], d00[CHIP],   oofs, MSK_W0, d00w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#2  +->         |      */
	    mop(OP_LDWR,  1, &BR[2][1][1], d00[CHIP],   rofs, MSK_W0, d00w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#2  +->         |      */
	    exe(OP_FMS,      &AR[2][0],    BR[2][0][1], EXP_H3210, BR[2][1][1], EXP_H3210, BR[2][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0); /* stage#2  |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#2  |  AR[1]    |      */
	    mop(OP_STWR,ex0, &AR[2][0],    oofs,   d00[CHIP], MSK_D0, d00w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#2  |    + ST   v      */
          }
	}
      }
//EMAX5A end
      /********************************************/
    } /* j-loop */
//EMAX5A drain_dirty_lmm
  }

#if 0
  printf("==========================\n");
  {Uint ii, jj;  for (ii=0; ii<M; ii++) { for (jj=0; jj<M; jj++) printf(" %08.8x", *(Uint*)&A[ii*M+jj]); printf("\n"); }}
#endif

  /* 前進消去 */
  for (i=0; i<M; i++) { /* 列方向 */
    for (j=0; j<M; j++) /* 行方向 */
      b[i*M+j] = (i==j)?1.0:0.0;
  }
  for (i=0; i<M; i+=NCHIP*H) { /* 列方向 */
    Uint  l000[NCHIP],  l010[NCHIP],  l020[NCHIP],  l030[NCHIP],  l040[NCHIP],  l050[NCHIP],  l060[NCHIP],  l070[NCHIP];  /* (i+CHIP*W*H+h*W+w)        */
    Uint *t000[NCHIP], *t010[NCHIP], *t020[NCHIP], *t030[NCHIP], *t040[NCHIP], *t050[NCHIP], *t060[NCHIP], *t070[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+k] */
    Uint *t000w[NCHIP],*t010w[NCHIP],*t020w[NCHIP],*t030w[NCHIP],*t040w[NCHIP],*t050w[NCHIP],*t060w[NCHIP],*t070w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+k] */
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
      l000[CHIP] = (i+CHIP*H+ 0+0)*M;	l010[CHIP] = (i+CHIP*H+ 1+0)*M;	l020[CHIP] = (i+CHIP*H+ 2+0)*M;	l030[CHIP] = (i+CHIP*H+ 3+0)*M;
      t000[CHIP] = &b[l000[CHIP]+i];	t010[CHIP] = &b[l010[CHIP]+i];	t020[CHIP] = &b[l020[CHIP]+i];	t030[CHIP] = &b[l030[CHIP]+i];
      t000w[CHIP]= (Ull)t000[CHIP];	t010w[CHIP]= (Ull)t010[CHIP];	t020w[CHIP]= (Ull)t020[CHIP];	t030w[CHIP]= (Ull)t030[CHIP];
    }
    /*for (j=1; j<M; j++) { *//*通常の連立一時方程式の場合*/
    for (j=i+1; j<M; j++) { /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでj=i+1から開始 */
      Uint *top  = &A[j*M+i];                                     /* A[p[j]*M+k] */
      Uint *topw = (Ull)top;
      /*Uint  len = (j+1)/2;*/
      Uint  len  = j-i;/* bが単位行列の場合,k<iではb[]==0なのでk=iから開始 */
      /********************************************/
      Uint  jc = j-i;
      Ull   Ajk; /* k=0...j-1 */
      Ull   b000, b001;
      Uint *d000[NCHIP], *d010[NCHIP], *d020[NCHIP], *d030[NCHIP], *d040[NCHIP], *d050[NCHIP], *d060[NCHIP], *d070[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+j] */
      Uint *d000w[NCHIP],*d010w[NCHIP],*d020w[NCHIP],*d030w[NCHIP],*d040w[NCHIP],*d050w[NCHIP],*d060w[NCHIP],*d070w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+j] */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	d000[CHIP] = &b[l000[CHIP]+j];	d010[CHIP] = &b[l010[CHIP]+j];	d020[CHIP] = &b[l020[CHIP]+j];	d030[CHIP] = &b[l030[CHIP]+j];
	d000w[CHIP]= (Ull)d000[CHIP];	d010w[CHIP]= (Ull)d010[CHIP];	d020w[CHIP]= (Ull)d020[CHIP];	d030w[CHIP]= (Ull)d030[CHIP];
      }
//EMAX5A begin inv_x2 mapdist=0
/*2*/ for (CHIP=0; CHIP<NCHIP; CHIP++) {
  /*1*/ for (INIT0=1,LOOP0=jc,cofs=0-4; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
          exe(OP_ADD, &cofs, cofs, EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
      /*for (k=0; k<j; k++) {*/ /* 最内列方向 */
	/*for (h=0; h<H; h++) {*/ /* vertical (parallel) execution */
	  /*for (w=0; w<W; w++) {*/   /* horizontal (parallel) execution */
	    /*b[(i+CHIP*W*H+h*W+w)*M+j] -= A[p[j]*M+k]*b[(i+CHIP*W*H+h*W+w)*M+k];*/ /* b[*]を縦に配置する場合,Aも縦配置. jが列方向に対応するが列方向の移動でkも長くなる */
	                                                                            /* 可変長kをH方向に展開写像するのは難しい.kはread-modify-writeの回転数に写像するしかない */
       	                                                                            /* b[*]とA[j][*]が同一LMMに入る前提 最大32KB/4/2=各4K要素,b[*]をいかに動かさないか */
	                                                                            /* 回転数jを一斉適用するには,iをWxH方向に展開するのが自然 */
	  /*           ↓★A[p[j]][*]をbroadcast可能 各A[p[j]][*]はp[j]が不連続なので1K要素まで収容.つまり二重ループ展開は無理 */
	  /* +-----------------------------------+     +------+ +------+ +------+ */
	  /* b[ 3][j]-=A[p[j]][0:j-1] b[ 3][0:j-1]     b[ 2][*] b[ 1][*] b[ 0][*] *//* この場合はb[][*]は1K要素までだが,b[0-3]は連続領域なのでFPDDMAは4K要素 */
	  /*                                                                      *//* LMM#3と#2にAを格納(1K要素)SPx1clk読み出し, LMM#1と#0にbを格納(4K要素)DPx2clk(ix4)読み出し */
	  /* b[ 7][j]-=A[p[j]][0:j-1] b[ 7][0:j-1]     b[ 6][*] b[ 5][*] b[ 4][*] *//* kループ後,i入れ換えにはb[0-59][*]全体の入れ替えが必要 */
	  /* b[59][j]-=A[p[j]][0:j-1] b[59][0:j-1]     b[58][*] b[57][*] b[56][*] *//* kループ後,j入れ換えにはA[p[j]][*]のbroadcastと，b[j]切替えのPIOで済む */
	  /* FOLDING時は,少なくとも第0列がFOLDINGであることが必要(conv-c2c仕様) */
	  mop(OP_LDWR,  1, &Ajk,         top,        cofs, MSK_W0, topw,        len, 0, 0, NULL, len);  /* A[p[j]*M+k]                  *//* stage#1.0 inv_x1の同位置topと偶然一致するとrdy=1となりkick_dmaが出ない */
	  mop(OP_LDWR,  1, &BR[1][3][1], t000[CHIP], cofs, MSK_W0, t000w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#1.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d000[CHIP], 0,    MSK_W0, d000w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#2.0  |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[1][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#2.0  +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d000[CHIP], MSK_D0, d000w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#2.0  +--------- xxx     */
	}
      }
//EMAX5A end
//EMAX5A drain_dirty_lmm
      /********************************************/
    } /* j-loop */
  }

#if 0
  printf("==========================\n");
  {Uint ii, jj;  for (ii=0; ii<M; ii++) { for (jj=0; jj<M; jj++) printf(" %08.8x", *(Uint*)&b[ii*M+jj]); printf("\n"); }}
#endif

  /* 後退代入 */
  for (i=0; i<M; i+=NCHIP*H) { /* 列方向 */
    for (j=M-1; j>=0; j--) { /* 行方向 */
      if (j<M-1) {
	Uint *top  = &A[j*M+j+1];                                  /* A[p[j]*M+k] */
	Uint *topw = (Ull)top;
	Uint  len  = M-j-1;
	/********************************************/
	Uint  jc = M-j-1;
	Ull   Ajk; /* k=j+1...M-1 */
	Ull   b000, b001;
	Uint  l000[NCHIP],  l010[NCHIP],  l020[NCHIP],  l030[NCHIP],  l040[NCHIP],  l050[NCHIP],  l060[NCHIP],  l070[NCHIP];  /* (i+CHIP*W*H+h*W+w)        */
	Uint *t000[NCHIP], *t010[NCHIP], *t020[NCHIP], *t030[NCHIP], *t040[NCHIP], *t050[NCHIP], *t060[NCHIP], *t070[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+k] */
	Uint *t000w[NCHIP],*t010w[NCHIP],*t020w[NCHIP],*t030w[NCHIP],*t040w[NCHIP],*t050w[NCHIP],*t060w[NCHIP],*t070w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+k] */
	Uint *d000[NCHIP], *d010[NCHIP], *d020[NCHIP], *d030[NCHIP], *d040[NCHIP], *d050[NCHIP], *d060[NCHIP], *d070[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+j] */
	Uint *d000w[NCHIP],*d010w[NCHIP],*d020w[NCHIP],*d030w[NCHIP],*d040w[NCHIP],*d050w[NCHIP],*d060w[NCHIP],*d070w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+j] */
	for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	  l000[CHIP] = (i+CHIP*H+ 0+0)*M+j+1;	l010[CHIP] = (i+CHIP*H+ 1+0)*M+j+1;	      l020[CHIP] = (i+CHIP*H+ 2+0)*M+j+1;	l030[CHIP] = (i+CHIP*H+ 3+0)*M+j+1;	    
	  t000[CHIP] = &x[l000[CHIP]];	t010[CHIP] = &x[l010[CHIP]];  t020[CHIP] = &x[l020[CHIP]];  t030[CHIP] = &x[l030[CHIP]];		    
	  t000w[CHIP]= (Ull)t000[CHIP]; t010w[CHIP]= (Ull)t010[CHIP]; t020w[CHIP]= (Ull)t020[CHIP]; t030w[CHIP]= (Ull)t030[CHIP];
	  d000[CHIP] = &b[l000[CHIP]-1];d010[CHIP] = &b[l010[CHIP]-1];d020[CHIP] = &b[l020[CHIP]-1];d030[CHIP] = &b[l030[CHIP]-1];		    
	  d000w[CHIP]= (Ull)d000[CHIP]; d010w[CHIP]= (Ull)d010[CHIP]; d020w[CHIP]= (Ull)d020[CHIP]; d030w[CHIP]= (Ull)d030[CHIP];
	}
//EMAX5A begin inv_x3 mapdist=0
  /*2*/ for (CHIP=0; CHIP<NCHIP; CHIP++) {
    /*1*/ for (INIT0=1,LOOP0=jc,cofs=jc*4; LOOP0--; INIT0=0) { /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD, &cofs, cofs, EXP_H3210, -4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
        /*for (k=M-1; k>j; k--) {*/ /* 最内列方向 */
          /*for (h=0; h<H; h++) {*/ /* vertical (parallel) execution */
	    /*for (w=0; w<W; w++) {*/   /* horizontal (parallel) execution */
	      /*b[(i+CHIP*W*H+h*W+w)*M+j] -= A[p[j]*M+k]*x[(i+CHIP*W*H+h*W+w)*M+k];*/
	                                        /* x[*]とA[j][*]が同一LMMに入る前提 最大32KB/4/2=各4K要素,x[*]をいかに動かさないか */
	                                        /* 回転数jを一斉適用するには,iをWxH方向に展開するのが自然 */
	    /*           ↓★A[p[j]][*]をbroadcast可能 各A[p[j]][*]はp[j]が不連続なので1K要素まで収容.つまり二重ループ展開は無理 */
	    /* +-----------------------------------+     +------+ +------+ +------+ */
	    /* b[ 3][j]-=A[p[j]][M-1:j+1] x[ 3][M-1:j+1] b[ 2][*] b[ 1][*] b[ 0][*] *//* この場合はb[][*]は1K要素までだが,x[0-3]は連続領域なのでFPDDMAは4K要素 */
	    /*                                                                      *//* LMM#3と#2にAを格納(1K要素)SPx1clk読み出し, LMM#1と#0にbを格納(4K要素)DPx2clk(ix4)読み出し */
	    /* b[ 7][j]-=A[p[j]][M-1:j+1] x[ 7][M-1:j+1] b[ 6][*] b[ 5][*] b[ 4][*] *//* kループ後,i入れ換えにはb[0-59][*]全体の入れ替えが必要 */
	    /* b[59][j]-=A[p[j]][M-1:j+1] x[59][M-1:j+1] b[58][*] b[57][*] b[56][*] *//* kループ後,j入れ換えにはA[p[j]][*]のbroadcastと，b[j]切替えのPIOで済む */
	    mop(OP_LDWR,  1, &Ajk,         top,          cofs, MSK_W0, topw,        len, 0, 0, NULL, len);  /* A[p[j]*M+k]                  *//* stage#1.0 inv_x1の同位置topと偶然一致するとrdy=1となりkick_dmaが出ない */
	    mop(OP_LDWR,  1, &BR[1][3][1], t000[CHIP],   cofs, MSK_W0, t000w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#1.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d000[CHIP],   0,    MSK_W0, d000w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#2.0  |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[1][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#2.0  +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d000[CHIP], MSK_D0, d000w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#2.0  +--------- xxx     */
	  }
        }
//EMAX5A end
//EMAX5A drain_dirty_lmm
        /********************************************/
      } /* if (j<M-1) */
      for (CHIP=0; CHIP<NCHIP; CHIP++) {
	for (h=0; h<H; h++) { /* vertical (parallel) execution */
	  inv1[j*M+p[i+CHIP*H+h]] = x[(i+CHIP*H+h)*M+j] = A[j*M+j]*b[(i+CHIP*H+h)*M+j]; /* PIOにてLMMのx[i*M+j]を直接更新 */
                                                                                           /* iはそのままで,jを切替え */
	}
      }
    } /* j-loop */
  }

#if 0
  printf("==========================\n");
  {Uint ii, jj;  for (ii=0; ii<M; ii++) { for (jj=0; jj<M; jj++) printf(" %08.8x", *(Uint*)&b[ii*M+jj]); printf("\n"); }}
#endif
}
