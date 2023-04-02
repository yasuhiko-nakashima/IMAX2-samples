
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

int WD=320, HT=240, BITMAP=320*240, SCRWD=5, SCRHT=5, VECWD=240, VECHT=240, VECSTEP=4;

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif
#if !defined(ARMSIML)
#include "./xdisp.c"
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
/*#define M 4096*/
#define M 256
#define RMGRP 8
/*#define NCHIP 4*/
#define NCHIP 1
/*#define W 1*/
#define H 16
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

#define CSIMWD 320
#define CSIMHT 240
#define CSIMBM (CSIMWD*CSIMHT)
Uint Z[CSIMBM];

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define ERRTH  (5.0E-3)
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

#if !defined(ARMSIML)
  x11_open(0);
#endif

#if 0
  reset_nanosec();
  orig();
  get_nanosec(0);
  show_nanosec();

  for (i=0; i<M; i++) {
    for (j=0; j<M; j++)
      A[i*M+j] = A0[i*M+j];
  }
#endif

  reset_nanosec();
  imax();
  get_nanosec(0);
  show_nanosec();

#ifdef ARMSIML
  copy_Z(0, A0);   _copyX(0, Z);
  copy_Z(1, A0);   _copyX(1, Z);
  copy_Z(4, A0);   _copyX(4, Z);
  copy_Z(5, A0);   _copyX(5, Z);
  copy_Z(8, A0);   _copyX(8, Z);
  copy_Z(9, A0);   _copyX(9, Z);
  copy_Z(0, inv1); _copyX(2, Z);
  copy_Z(1, inv1); _copyX(3, Z);
  copy_Z(4, inv1); _copyX(6, Z);
  copy_Z(5, inv1); _copyX(7, Z);
  copy_Z(8, inv1); _copyX(10,Z);
  copy_Z(9, inv1); _copyX(11,Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, A0);   BGR_to_X(0, Z);
  copy_Z(1, A0);   BGR_to_X(1, Z);
  copy_Z(4, A0);   BGR_to_X(5, Z);
  copy_Z(5, A0);   BGR_to_X(6, Z);
  copy_Z(8, A0);   BGR_to_X(10,Z);
  copy_Z(9, A0);   BGR_to_X(11,Z);
  copy_Z(0, inv1); BGR_to_X(2, Z);
  copy_Z(1, inv1); BGR_to_X(3, Z);
  copy_Z(4, inv1); BGR_to_X(7, Z);
  copy_Z(5, inv1); BGR_to_X(8, Z);
  copy_Z(8, inv1); BGR_to_X(12,Z);
  copy_Z(9, inv1); BGR_to_X(13,Z);
  x11_update();
#endif

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

  show_nanosec();

#if !defined(ARMSIML)
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (!x11_checkevent());
#endif
}

copy_Z(id, from)
     int id; /* 0 .. 11 */
     unsigned int *from;
{
  int i, j;
  volatile unsigned int *to = Z;
  unsigned int *offs;

  switch (id) {
  case 0:  offs = from;               break;
  case 1:  offs = from + WD;          break;
  case 2:  offs = from + WD*2;        break;
  case 3:  offs = from + WD*3;        break;
  case 4:  offs = from + M*HT;        break;
  case 5:  offs = from + M*HT+WD;     break;
  case 6:  offs = from + M*HT+WD*2;   break;
  case 7:  offs = from + M*HT+WD*3;   break;
  case 8:  offs = from + M*HT*2;      break;
  case 9:  offs = from + M*HT*2+WD;   break;
  case 10: offs = from + M*HT*2+WD*2; break;
  case 11: offs = from + M*HT*2+WD*3; break;
  case 12: offs = from + M*HT*3;      break;
  case 13: offs = from + M*HT*3+WD;   break;
  case 14: offs = from + M*HT*3+WD*2; break;
  case 15: offs = from + M*HT*3+WD*3; break;
  }
  for (i=0; i<HT; i++, offs+=M) {
    if (offs<from+M*M) {
      for (j=0; j<WD; j++) {
	if (j+(id%4)*WD<M) *to++ = (*(offs+j))>>0;
	else               *to++ = 0;
      }
    }
    else {
      for (j=0; j<WD; j++)
	*to++ = 0;
    }
  }
}

orig()
{
  int i, j, k;
  float pmax;

  printf("<<<ORIG>>>\n");

  /* LU分解 */
  for (i=0; i<M+1; i++)
    p[i] = i;
  for (i=0; i<M; i++) {
    pmax = 0.0;
    k = -1;
    for (j=i; j<M; j++) {
      if (pmax < fabsf(A[p[j]*M+i])) {
	pmax = fabsf(A[p[j]*M+i]);
	k = j;
      }
    }
    if (k == -1) {
      fprintf(stderr, "can't solve\n");
      exit(1);
    }
    j = p[k]; p[k] = p[i]; p[i] = j;
    A[p[i]*M+i] = 1.0/A[p[i]*M+i];
    for (j=i+1; j<M; j++) {
      A[p[j]*M+i] *= A[p[i]*M+i];
      for (k=i+1; k<M; k++)
	A[p[j]*M+k] -= A[p[j]*M+i]*A[p[i]*M+k];
    }
  }

  /* 逆行列求める */
  for (i=0; i<M; i++) {
    for (j=0; j<M; j++)
      b[p[j]] = (i==j)?1.0:0.0;
    /*for (j=1; j<M; j++) { *//*通常の連立一時方程式の場合*/
    for (j=i+1; j<M; j++) { /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでj=i+1から開始 */
      /*for (k=0; k<j; k++) *//*通常の連立一時方程式の場合*/
      for (k=i; k<j; k++) /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでk=iから開始 */
	b[p[j]] -= A[p[j]*M+k]*b[p[k]];
    }
    for (j=M-1; j>=0; j--) {
      for (k=M-1; k>j; k--)
	b[p[j]] -= A[p[j]*M+k]*x[k];
      inv0[j*M+p[i]] = x[j] = b[p[j]]*A[p[j]*M+j];
    }  
  }
}

#if 0
imax() {
  int i, j, k;
  float pmax;
  printf("<<<IMAX>>>\n");

#define INFO
  /* LU分解 */
  for (i=0; i<M+1; i++)
    p[i] = i;
  for (i=0; i<M; i++) { /* 列方向 */
    pmax = 0.0;
    k = -1;
    for (j=i; j<M; j++) { /* 行方向に探索 */
      if (pmax < fabsf(A[p[j]*M+i])) {
	pmax = fabsf(A[p[j]*M+i]);
	k = j;
      }
    }
    if (k == -1) {
      fprintf(stderr, "can't solve\n");
      exit(1);
    }
    j = p[k]; p[k] = p[i]; p[i] = j;
    printf("pivot p[%d]=%d\n", i, p[i]);
    A[p[i]*M+i] = 1.0/A[p[i]*M+i];
    for (j=i+1; j<M; j++) /* 行方向 */
      A[p[j]*M+i] *= A[p[i]*M+i];
    /********************************************/
    for (j=i+1; j<M; j++) { /* 行方向 */
      for (k=i+1; k<M; k++) { /* 最内列方向 */
#ifdef INFO
	printf(" A[%d][%d]-=A[%d][%d]*A[%d][%d]", p[j], k, p[j], i, p[i], k);
#endif
	A[p[j]*M+k] -= A[p[j]*M+i]*A[p[i]*M+k]; /* 3 */
      }
#ifdef INFO
      printf("\n");
#endif
    }
    /********************************************/
  }

  /* 逆行列求める */
  for (i=0; i<M; i++) { /* 列方向 */
    for (j=0; j<M; j++) /* 行方向 */
      b[p[j]] = (i==j)?1.0:0.0;
    /********************************************/
    for (j=1; j<M; j++) { /* 行方向 */
      for (k=0; k<j; k++) { /* 最内列方向 */
#ifdef INFO
	printf(" b[%d]-=A[%d][%d]*b[%d]", p[j], p[j], k, p[k]);
#endif
	b[p[j]] -= A[p[j]*M+k]*b[p[k]]; /* 3 */
      }
#ifdef INFO
      printf("\n");
#endif
    }
    /********************************************/
    x[M-1] = A[p[M-1]*M+M-1]*b[p[M-1]];
    /********************************************/
    for (j=M-2; j>=0; j--) { /* 行方向 */
      for (k=M-1; k>j; k--) { /* 最内列方向 */
#ifdef INFO
	printf(" x%d=A%d.%d*(b%d-=A%d.%d*x%d]", j, j, j, p[j], p[j], k, k);
#endif
	x[j] = A[p[j]*M+j]*(b[p[j]] -= A[p[j]*M+k]*x[k]); /* 3 */
      }
#ifdef INFO
      printf("\n");
#endif
    }  
    /********************************************/
    for (j=0; j<M; j++) /* 行方向 */
      inv1[j*M+p[i]] = x[j];
  }
}
#endif

#if 0
imax() {
  Ull CHIP;
  int i, j, k, h;
  float pmax, tmp;
  printf("<<<IMAX>>>\n");

  /* LU分解 */
  for (i=0; i<M+1; i++)
    p[i] = i;
  for (i=0; i<M; i++) { /* 列方向 */
    pmax = 0.0;
    k = -1;
    for (j=i; j<M; j++) { /* 行方向に探索 */
      if (pmax < fabsf(A[j*M+i])) {/*★*/
	pmax = fabsf(A[j*M+i]);/*★*/
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
      A[j*M+i] *= A[i*M+i];                               /*★*/
    for (j=i+1; j<M; j+=NCHIP*H) { /* 行方向 */
      /********************************************/
      for (CHIP=0; CHIP<NCHIP; CHIP++) {
	for (k=0; k<M-(i+1); k++) { /* 最内列方向 */
	  for (h=0; h<H; h++) { /* vertical (parallel) execution */
	    if (j+h*NCHIP+CHIP<M) A[(j+h*NCHIP+CHIP)*M+i+1+k] -= A[(j+h*NCHIP+CHIP)*M+i]*A[i*M+i+1+k]; /* 後続の逆行列と異なり,accumurateではなく要素毎の単独減算の繰返し */
	                            /*★*/                         /*★*/                  /*★*/      /* const:A[p[j]][0] * LMM A[p[  0]][*] */
                                                                                                       /*        ↓                           */
	    /*   v A[p[j]*M+i]         */                                                              /*   LMM A[p[j>0]][*] accumulate (column方向にj,j+1,..,479のため依存無) */
	    /***************************/
	    /* + - - - - - - - - - - - */ /* A[p[i]] 先頭行       */ /* 先頭行はi更新まで再利用可能 */
	    /* | * > > > > > > > > > > */ /* A[p[j]] 次行から引く */ /* 1行をLMMに写像 */
	    /* | v + - - - - - - - - - */ 
	    /* | v | * > > > > > > > > */ /* M/60を収容してi更新までj+=60を繰り返す */
	    /* | v | v + - - - - - - - */ /* 行番号比較とcstによる端数制御 */
	    /* | v | v - + - - - - - - */ /* + CHIP#1 h=0 */
	    /* | v | v - - + - - - - - */ /* + CHIP#0 h=1 */
	    /* | v | v - - - + - - - - */ /* + CHIP#1 h=1 */
	    /* | v | v - - - - + - - - */ /* + CHIP#0 h=2 */
	    /* | v | v - - - - - + - - */ /* + CHIP#1 h=2 */
	    /* | v | v - - - - - - + - */ /* + CHIP#0 h=3 */
	    /* | v | v - - - - - - - + */ /* + CHIP#1 h=3 */
	    /***************************/ /* 最大60行まで写像可能 */
	  }
	}
      }
      /********************************************/
    }
  }

  /* 逆行列求める */
  for (i=0; i<M; i++) { /* 列方向 */
    for (j=0; j<M; j++) /* 行方向 */
      b[i*M+j] = (i==j)?1.0:0.0;
  }
  for (i=0; i<M; i+=NCHIP*H) { /* 列方向 */
    for (j=1; j<M; j++) { /* 行方向 */
      /********************************************/
      for (CHIP=0; CHIP<NCHIP; CHIP++) {
	for (k=0; k<j; k++) { /* 最内列方向 */
	  for (h=0; h<H; h++) { /* vertical (parallel) execution */
	    b[(i+CHIP*H+h)*M+j] -= A[j*M+k]*b[(i+CHIP*H+h)*M+k]; /* b[*]を縦に配置する場合,Aも縦配置. jが列方向に対応するが列方向の移動でkも長くなる */
	                             /*★*/                      /* 可変長kをH方向に展開写像するのは難しい.kはread-modify-writeの回転数に写像するしかない */
       	                                                         /* b[*]とA[j][*]が同一LMMに入る前提 最大32KB/4/2=各4K要素,b[*]をいかに動かさないか */
	                                                         /* 回転数jを一斉適用するには,iをWxH方向に展開するのが自然 */
	      /*           ↓★A[p[j]][*]をbroadcast可能 各A[p[j]][*]はp[j]が不連続なので1K要素まで収容.つまり二重ループ展開は無理 */
	      /* +-----------------------------------+     +------+ +------+ +------+ */
	      /* b[ 3][j]-=A[p[j]][0:j-1] b[ 3][0:j-1]     b[ 2][*] b[ 1][*] b[ 0][*] *//* この場合はb[][*]は1K要素までだが,b[0-3]は連続領域なのでFPDDMAは4K要素 */
	      /*                                                                      *//* LMM#3と#2にAを格納(1K要素)SPx1clk読み出し, LMM#1と#0にbを格納(4K要素)DPx2clk(ix4)読み出し */
	      /* b[ 7][j]-=A[p[j]][0:j-1] b[ 7][0:j-1]     b[ 6][*] b[ 5][*] b[ 4][*] *//* kループ後,i入れ換えにはb[0-59][*]全体の入れ替えが必要 */
	      /* b[59][j]-=A[p[j]][0:j-1] b[59][0:j-1]     b[58][*] b[57][*] b[56][*] *//* kループ後,j入れ換えにはA[p[j]][*]のbroadcastと，b[j]切替えのPIOで済む */
	  }
	}
      }
      /********************************************/
    }
  }
  for (i=0; i<M; i+=NCHIP*H) { /* 列方向 */
    for (j=M-1; j>=0; j--) { /* 行方向 */
      /********************************************/
      for (CHIP=0; CHIP<NCHIP; CHIP++) {
	for (k=M-1; k>j; k--) { /* 最内列方向 */
	  for (h=0; h<H; h++) { /* vertical (parallel) execution */
	    b[(i+CHIP*H+h)*M+j] -= A[j*M+k]*x[(i+CHIP*H+h)*M+k];
	                             /*★*/       /* x[*]とA[j][*]が同一LMMに入る前提 最大32KB/4/2=各4K要素,x[*]をいかに動かさないか */
	                                          /* 回転数jを一斉適用するには,iをWxH方向に展開するのが自然 */
	    /*           ↓★A[p[j]][*]をbroadcast可能 各A[p[j]][*]はp[j]が不連続なので1K要素まで収容.つまり二重ループ展開は無理 */
	    /* +-----------------------------------+     +------+ +------+ +------+ */
	    /* b[ 3][j]-=A[p[j]][M-1:j+1] x[ 3][M-1:j+1] b[ 2][*] b[ 1][*] b[ 0][*] *//* この場合はb[][*]は1K要素までだが,x[0-3]は連続領域なのでFPDDMAは4K要素 */
	    /*                                                                      *//* LMM#3と#2にAを格納(1K要素)SPx1clk読み出し, LMM#1と#0にbを格納(4K要素)DPx2clk(ix4)読み出し */
	    /* b[ 7][j]-=A[p[j]][M-1:j+1] x[ 7][M-1:j+1] b[ 6][*] b[ 5][*] b[ 4][*] *//* kループ後,i入れ換えにはb[0-59][*]全体の入れ替えが必要 */
	    /* b[59][j]-=A[p[j]][M-1:j+1] x[59][M-1:j+1] b[58][*] b[57][*] b[56][*] *//* kループ後,j入れ換えにはA[p[j]][*]のbroadcastと，b[j]切替えのPIOで済む */
	  }
	}
      }
      /********************************************/
      for (CHIP=0; CHIP<NCHIP; CHIP++) {
	for (h=0; h<H; h++) { /* vertical (parallel) execution */
	  inv1[j*M+p[i+CHIP*H+h]] = x[(i+CHIP*H+h)*M+j] = A[j*M+j]*b[(i+CHIP*H+h)*M+j]; /* PIOにてLMMのx[i*M+j]を直接更新 */
                                                            /*★*/                      /* iはそのままで,jを切替え */
	}
      }
    }  
  }
}
#endif

#if 0
 pivot p[0]=4 i=0 j=1-4
 p[j] k   p[j] c    c  k |p[j] k   p[j] c    c  k |p[j] k   p[j] c    c  k |p[j] k   p[j] c    c  k
 A[1][1]-=A[1][0]*A[4][1] A[1][2]-=A[1][0]*A[4][2] A[1][3]-=A[1][0]*A[4][3] A[1][4]-=A[1][0]*A[4][4]
 A[2][1]-=A[2][0]*A[4][1] A[2][2]-=A[2][0]*A[4][2] A[2][3]-=A[2][0]*A[4][3] A[2][4]-=A[2][0]*A[4][4]
 A[3][1]-=A[3][0]*A[4][1] A[3][2]-=A[3][0]*A[4][2] A[3][3]-=A[3][0]*A[4][3] A[3][4]-=A[3][0]*A[4][4]
 A[0][1]-=A[0][0]*A[4][1] A[0][2]-=A[0][0]*A[4][2] A[0][3]-=A[0][0]*A[4][3] A[0][4]-=A[0][0]*A[4][4]
 pivot p[1]=1 i=1 j=2-4
 p[j] k   p[j] c    c  k |p[j] k   p[j] c    c  k |p[j] k   p[j] c    c  k
 A[2][2]-=A[2][1]*A[1][2] A[2][3]-=A[2][1]*A[1][3] A[2][4]-=A[2][1]*A[1][4]
 A[3][2]-=A[3][1]*A[1][2] A[3][3]-=A[3][1]*A[1][3] A[3][4]-=A[3][1]*A[1][4]
 A[0][2]-=A[0][1]*A[1][2] A[0][3]-=A[0][1]*A[1][3] A[0][4]-=A[0][1]*A[1][4]
 pivot p[2]=0 i=2 j=3-4
 p[j] k   p[j] c    c  k |p[j] k   p[j] c    c  k
 A[3][3]-=A[3][2]*A[0][3] A[3][4]-=A[3][2]*A[0][4]
 A[2][3]-=A[2][2]*A[0][3] A[2][4]-=A[2][2]*A[0][4]
 pivot p[3]=3 i=3 j=4-4
 p[j] k   p[j] c    c  k
 A[2][4]-=A[2][3]*A[3][4]
---------------------------------------------------------------------------------------------------
 b[0][1]-=A[1][0]*b[0][0]
 b[0][2]-=A[0][0]*b[0][0] b[0][2]-=A[0][1]*b[0][1]
 b[0][3]-=A[3][0]*b[0][0] b[0][3]-=A[3][1]*b[0][1] b[0][3]-=A[3][2]*b[0][2]
 b[0][4]-=A[2][0]*b[0][0] b[0][4]-=A[2][1]*b[0][1] b[0][4]-=A[2][2]*b[0][2] b[0][4]-=A[2][3]*b[0][3]
 b[1][1]-=A[1][0]*b[1][0]
 b[1][2]-=A[0][0]*b[1][0] b[1][2]-=A[0][1]*b[1][1]
 b[1][3]-=A[3][0]*b[1][0] b[1][3]-=A[3][1]*b[1][1] b[1][3]-=A[3][2]*b[1][2]
 b[1][4]-=A[2][0]*b[1][0] b[1][4]-=A[2][1]*b[1][1] b[1][4]-=A[2][2]*b[1][2] b[1][4]-=A[2][3]*b[1][3]
 b[2][1]-=A[1][0]*b[2][0]
 b[2][2]-=A[0][0]*b[2][0] b[2][2]-=A[0][1]*b[2][1]
 b[2][3]-=A[3][0]*b[2][0] b[2][3]-=A[3][1]*b[2][1] b[2][3]-=A[3][2]*b[2][2]
 b[2][4]-=A[2][0]*b[2][0] b[2][4]-=A[2][1]*b[2][1] b[2][4]-=A[2][2]*b[2][2] b[2][4]-=A[2][3]*b[2][3]
 b[3][1]-=A[1][0]*b[3][0]
 b[3][2]-=A[0][0]*b[3][0] b[3][2]-=A[0][1]*b[3][1]
 b[3][3]-=A[3][0]*b[3][0] b[3][3]-=A[3][1]*b[3][1] b[3][3]-=A[3][2]*b[3][2]
 b[3][4]-=A[2][0]*b[3][0] b[3][4]-=A[2][1]*b[3][1] b[3][4]-=A[2][2]*b[3][2] b[3][4]-=A[2][3]*b[3][3]
 b[4][1]-=A[1][0]*b[4][0]
 b[4][2]-=A[0][0]*b[4][0] b[4][2]-=A[0][1]*b[4][1]
 b[4][3]-=A[3][0]*b[4][0] b[4][3]-=A[3][1]*b[4][1] b[4][3]-=A[3][2]*b[4][2]
 b[4][4]-=A[2][0]*b[4][0] b[4][4]-=A[2][1]*b[4][1] b[4][4]-=A[2][2]*b[4][2] b[4][4]-=A[2][3]*b[4][3]
---------------------------------------------------------------------------------------------------
 x0.3=A3.3*(b0.3-=A3.4*x0.4]
 x0.2=A0.2*(b0.2-=A0.4*x0.4] x0.2=A0.2*(b0.2-=A0.3*x0.3]
 x0.1=A1.1*(b0.1-=A1.4*x0.4] x0.1=A1.1*(b0.1-=A1.3*x0.3] x0.1=A1.1*(b0.1-=A1.2*x0.2]
 x0.0=A4.0*(b0.0-=A4.4*x0.4] x0.0=A4.0*(b0.0-=A4.3*x0.3] x0.0=A4.0*(b0.0-=A4.2*x0.2] x0.0=A4.0*(b0.0-=A4.1*x0.1]
 x1.3=A3.3*(b1.3-=A3.4*x1.4]
 x1.2=A0.2*(b1.2-=A0.4*x1.4] x1.2=A0.2*(b1.2-=A0.3*x1.3]
 x1.1=A1.1*(b1.1-=A1.4*x1.4] x1.1=A1.1*(b1.1-=A1.3*x1.3] x1.1=A1.1*(b1.1-=A1.2*x1.2]
 x1.0=A4.0*(b1.0-=A4.4*x1.4] x1.0=A4.0*(b1.0-=A4.3*x1.3] x1.0=A4.0*(b1.0-=A4.2*x1.2] x1.0=A4.0*(b1.0-=A4.1*x1.1]
 x2.3=A3.3*(b2.3-=A3.4*x2.4]
 x2.2=A0.2*(b2.2-=A0.4*x2.4] x2.2=A0.2*(b2.2-=A0.3*x2.3]
 x2.1=A1.1*(b2.1-=A1.4*x2.4] x2.1=A1.1*(b2.1-=A1.3*x2.3] x2.1=A1.1*(b2.1-=A1.2*x2.2]
 x2.0=A4.0*(b2.0-=A4.4*x2.4] x2.0=A4.0*(b2.0-=A4.3*x2.3] x2.0=A4.0*(b2.0-=A4.2*x2.2] x2.0=A4.0*(b2.0-=A4.1*x2.1]
 x3.3=A3.3*(b3.3-=A3.4*x3.4]
 x3.2=A0.2*(b3.2-=A0.4*x3.4] x3.2=A0.2*(b3.2-=A0.3*x3.3]
 x3.1=A1.1*(b3.1-=A1.4*x3.4] x3.1=A1.1*(b3.1-=A1.3*x3.3] x3.1=A1.1*(b3.1-=A1.2*x3.2]
 x3.0=A4.0*(b3.0-=A4.4*x3.4] x3.0=A4.0*(b3.0-=A4.3*x3.3] x3.0=A4.0*(b3.0-=A4.2*x3.2] x3.0=A4.0*(b3.0-=A4.1*x3.1]
 x4.3=A3.3*(b4.3-=A3.4*x4.4]
 x4.2=A0.2*(b4.2-=A0.4*x4.4] x4.2=A0.2*(b4.2-=A0.3*x4.3]
 x4.1=A1.1*(b4.1-=A1.4*x4.4] x4.1=A1.1*(b4.1-=A1.3*x4.3] x4.1=A1.1*(b4.1-=A1.2*x4.2]
 x4.0=A4.0*(b4.0-=A4.4*x4.4] x4.0=A4.0*(b4.0-=A4.3*x4.3] x4.0=A4.0*(b4.0-=A4.2*x4.2] x4.0=A4.0*(b4.0-=A4.1*x4.1]
#endif

#if 1
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
    /* FPGA実機でj-loopの最終(len=1)が動かないので,ついでにARMのほうが速そうなlenをARMで実行 2019/3/1 Nakashima */
    if (len < 16) { /* len<1でも正常なので性能最大化で決めてよい */
      for (j=i+1; j<M; j+=NCHIP*H*RMGRP) { /* 行方向 */
	for (CHIP=0; CHIP<NCHIP; CHIP++) {
	  for (h=0; h<H; h++) { /* vertical (parallel) execution */
	    for (grp=0; grp<RMGRP; grp++) {
	      for (k=0; k<M-(i+1); k++) { /* 最内列方向 */
		if (j+h*NCHIP*RMGRP+CHIP*RMGRP+grp<M) A[(j+h*NCHIP*RMGRP+CHIP*RMGRP+grp)*M+i+1+k] -= A[(j+h*NCHIP*RMGRP+CHIP*RMGRP+grp)*M+i]*A[i*M+i+1+k];
	      }
	    }
	  }
	}
      }
    }
    else {
    for (j=i+1; j<M; j+=NCHIP*H*RMGRP) { /* 行方向 */
      /********************************************/
      Uint  l00[NCHIP],  l01[NCHIP],  l02[NCHIP],  l03[NCHIP],  l04[NCHIP],  l05[NCHIP],  l06[NCHIP],  l07[NCHIP]; /* j<M-(h*NCHIP+CHIP) */
      Uint  l08[NCHIP],  l09[NCHIP],  l10[NCHIP],  l11[NCHIP],  l12[NCHIP],  l13[NCHIP],  l14[NCHIP],  l15[NCHIP]; /* j<M-(h*NCHIP+CHIP) */
      Uint *d00[NCHIP], *d01[NCHIP], *d02[NCHIP], *d03[NCHIP], *d04[NCHIP], *d05[NCHIP], *d06[NCHIP], *d07[NCHIP]; /* A[p[j+h*NCHIP+CHIP]*M+k] */
      Uint *d08[NCHIP], *d09[NCHIP], *d10[NCHIP], *d11[NCHIP], *d12[NCHIP], *d13[NCHIP], *d14[NCHIP], *d15[NCHIP]; /* A[p[j+h*NCHIP+CHIP]*M+k] */
      Uint *d00w[NCHIP],*d01w[NCHIP],*d02w[NCHIP],*d03w[NCHIP],*d04w[NCHIP],*d05w[NCHIP],*d06w[NCHIP],*d07w[NCHIP];/* A[p[j+h*NCHIP+CHIP]*M+k] */
      Uint *d08w[NCHIP],*d09w[NCHIP],*d10w[NCHIP],*d11w[NCHIP],*d12w[NCHIP],*d13w[NCHIP],*d14w[NCHIP],*d15w[NCHIP];/* A[p[j+h*NCHIP+CHIP]*M+k] */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	l00[CHIP]=(j+ 0*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 0*NCHIP*RMGRP+CHIP*RMGRP):M;l01[CHIP]=(j+ 1*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 1*NCHIP*RMGRP+CHIP*RMGRP):M;
        l02[CHIP]=(j+ 2*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 2*NCHIP*RMGRP+CHIP*RMGRP):M;l03[CHIP]=(j+ 3*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 3*NCHIP*RMGRP+CHIP*RMGRP):M;
	l04[CHIP]=(j+ 4*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 4*NCHIP*RMGRP+CHIP*RMGRP):M;l05[CHIP]=(j+ 5*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 5*NCHIP*RMGRP+CHIP*RMGRP):M;
        l06[CHIP]=(j+ 6*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 6*NCHIP*RMGRP+CHIP*RMGRP):M;l07[CHIP]=(j+ 7*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 7*NCHIP*RMGRP+CHIP*RMGRP):M;
	l08[CHIP]=(j+ 8*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 8*NCHIP*RMGRP+CHIP*RMGRP):M;l09[CHIP]=(j+ 9*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+ 9*NCHIP*RMGRP+CHIP*RMGRP):M;
        l10[CHIP]=(j+10*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+10*NCHIP*RMGRP+CHIP*RMGRP):M;l11[CHIP]=(j+11*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+11*NCHIP*RMGRP+CHIP*RMGRP):M;
	l12[CHIP]=(j+12*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+12*NCHIP*RMGRP+CHIP*RMGRP):M;l13[CHIP]=(j+13*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+13*NCHIP*RMGRP+CHIP*RMGRP):M;
        l14[CHIP]=(j+14*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+14*NCHIP*RMGRP+CHIP*RMGRP):M;l15[CHIP]=(j+15*NCHIP*RMGRP+CHIP*RMGRP<M)?(j+15*NCHIP*RMGRP+CHIP*RMGRP):M;
	d00[CHIP] = &A[l00[CHIP]*M+i];   d01[CHIP] = &A[l01[CHIP]*M+i];	  d02[CHIP] = &A[l02[CHIP]*M+i];   d03[CHIP] = &A[l03[CHIP]*M+i];
	d04[CHIP] = &A[l04[CHIP]*M+i];   d05[CHIP] = &A[l05[CHIP]*M+i];	  d06[CHIP] = &A[l06[CHIP]*M+i];   d07[CHIP] = &A[l07[CHIP]*M+i];
	d08[CHIP] = &A[l08[CHIP]*M+i];   d09[CHIP] = &A[l09[CHIP]*M+i];	  d10[CHIP] = &A[l10[CHIP]*M+i];   d11[CHIP] = &A[l11[CHIP]*M+i];
	d12[CHIP] = &A[l12[CHIP]*M+i];   d13[CHIP] = &A[l13[CHIP]*M+i];	  d14[CHIP] = &A[l14[CHIP]*M+i];   d15[CHIP] = &A[l15[CHIP]*M+i];
	d00w[CHIP]= (Ull)d00[CHIP];      d01w[CHIP]= (Ull)d01[CHIP];      d02w[CHIP]= (Ull)d02[CHIP];      d03w[CHIP]= (Ull)d03[CHIP];
	d04w[CHIP]= (Ull)d04[CHIP];      d05w[CHIP]= (Ull)d05[CHIP];      d06w[CHIP]= (Ull)d06[CHIP];      d07w[CHIP]= (Ull)d07[CHIP];
	d08w[CHIP]= (Ull)d08[CHIP];      d09w[CHIP]= (Ull)d09[CHIP];      d10w[CHIP]= (Ull)d10[CHIP];      d11w[CHIP]= (Ull)d11[CHIP];
	d12w[CHIP]= (Ull)d12[CHIP];      d13w[CHIP]= (Ull)d13[CHIP];      d14w[CHIP]= (Ull)d14[CHIP];      d15w[CHIP]= (Ull)d15[CHIP];
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
#if (H>1)                                                                                                                                   /*          *--------- BR[2]   */
	    exe(OP_CMP_LT,   &cc1,         l01[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2              LD     */
	    mop(OP_LDWR,  1, &BR[3][2][1], top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k]                       stage#3              |      */
	    mop(OP_LDWR,  1, &BR[3][0][1], d01[CHIP],   oofs, MSK_W0, d01w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#3  +->         |      */
	    mop(OP_LDWR,  1, &BR[3][1][1], d01[CHIP],   rofs, MSK_W0, d01w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#3  +->         |      */
	    exe(OP_FMS,      &AR[3][0],    BR[3][0][1], EXP_H3210, BR[3][1][1], EXP_H3210, BR[3][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0); /* stage#3  |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc1, 0xaaaa);                                                                                 /* stage#3  |  AR[2]    |      */
	    mop(OP_STWR,ex0, &AR[3][0],    oofs,   d01[CHIP], MSK_D0, d01w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#3  |    + ST   v      */
#if (H>2)                                                                                                                                   /*          *--------- BR[3]   */
	    exe(OP_CMP_LT,   &cc0,         l02[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3              LD     */
	    mop(OP_LDWR,  1, &BR[4][2][1], top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#4              |      */
	    mop(OP_LDWR,  1, &BR[4][0][1], d02[CHIP],   oofs, MSK_W0, d02w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#4  +->         |      */
	    mop(OP_LDWR,  1, &BR[4][1][1], d02[CHIP],   rofs, MSK_W0, d02w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#4  +->         |      */
	    exe(OP_FMS,      &AR[4][0],    BR[4][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, BR[4][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0); /* stage#4  |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#4  |  AR[3]    |      */
	    mop(OP_STWR,ex0, &AR[4][0],    oofs,   d02[CHIP], MSK_D0, d02w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#4  |    + ST   v      */
#if (H>3)                                                                                                                                   /*          *--------- BR[4]   */
	    exe(OP_CMP_LT,   &cc1,         l03[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4              LD     */
	    mop(OP_LDWR,  1, &BR[5][2][1], top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#5              |      */
	    mop(OP_LDWR,  1, &BR[5][0][1], d03[CHIP],   oofs, MSK_W0, d03w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#5  +->         |      */
	    mop(OP_LDWR,  1, &BR[5][1][1], d03[CHIP],   rofs, MSK_W0, d03w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#5  +->         |      */
	    exe(OP_FMS,      &AR[5][0],    BR[5][0][1], EXP_H3210, BR[5][1][1], EXP_H3210, BR[5][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0); /* stage#5  |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc1, 0xaaaa);                                                                                 /* stage#5  |  AR[4]    |      */
	    mop(OP_STWR,ex0, &AR[5][0],    oofs,   d03[CHIP], MSK_D0, d03w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#5  |    + ST   v      */
#if (H>4)                                                                                                                                   /*          *--------- BR[5]   */
	    exe(OP_CMP_LT,   &cc0,         l04[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5              LD     */
	    mop(OP_LDWR,  1, &BR[6][2][1], top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k]                       stage#6              |      */
	    mop(OP_LDWR,  1, &BR[6][0][1], d04[CHIP],   oofs, MSK_W0, d04w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#6  +->         |      */
	    mop(OP_LDWR,  1, &BR[6][1][1], d04[CHIP],   rofs, MSK_W0, d04w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#6  +->         |      */
	    exe(OP_FMS,      &AR[6][0],    BR[6][0][1], EXP_H3210, BR[6][1][1], EXP_H3210, BR[6][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0); /* stage#6  |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#6  |  AR[5]    |      */
	    mop(OP_STWR,ex0, &AR[6][0],    oofs,   d04[CHIP], MSK_D0, d04w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#6  |    + ST   v      */
#if (H>5)                                                                                                                                   /*          *--------- BR[6]   */
	    exe(OP_CMP_LT,   &cc1,         l05[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6              LD     */
	    mop(OP_LDWR,  1, &BR[7][2][1], top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k]                       stage#7              |      */
	    mop(OP_LDWR,  1, &BR[7][0][1], d05[CHIP],   oofs, MSK_W0, d05w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#7  +->         |      */
	    mop(OP_LDWR,  1, &BR[7][1][1], d05[CHIP],   rofs, MSK_W0, d05w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#7  +->         |      */
	    exe(OP_FMS,      &AR[7][0],    BR[7][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, BR[7][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0); /* stage#7  |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc1, 0xaaaa);                                                                                 /* stage#7  |  AR[6]    |      */
	    mop(OP_STWR,ex0, &AR[7][0],    oofs,   d05[CHIP], MSK_D0, d05w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#7  |    + ST   v      */
#if (H>6)                                                                                                                                   /*          *--------- BR[7]   */
	    exe(OP_CMP_LT,   &cc0,         l06[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7              LD     */
	    mop(OP_LDWR,  1, &BR[8][2][1], top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#8              |      */
	    mop(OP_LDWR,  1, &BR[8][0][1], d06[CHIP],   oofs, MSK_W0, d06w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#8  +->         |      */
	    mop(OP_LDWR,  1, &BR[8][1][1], d06[CHIP],   rofs, MSK_W0, d06w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#8  +->         |      */
	    exe(OP_FMS,      &AR[8][0],    BR[8][0][1], EXP_H3210, BR[8][1][1], EXP_H3210, BR[8][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0); /* stage#8  |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#8  |  AR[7]    |      */
	    mop(OP_STWR,ex0, &AR[8][0],    oofs,   d06[CHIP], MSK_D0, d06w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#8  |    + ST   v      */
#if (H>7)                                                                                                                                   /*          *--------- BR[8]   */
	    exe(OP_CMP_LT,   &cc1,         l07[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8              LD     */
	    mop(OP_LDWR,  1, &BR[9][2][1], top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#9              |      */
	    mop(OP_LDWR,  1, &BR[9][0][1], d07[CHIP],   oofs, MSK_W0, d07w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#9  +->         |      */
	    mop(OP_LDWR,  1, &BR[9][1][1], d07[CHIP],   rofs, MSK_W0, d07w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#9  +->         |      */
	    exe(OP_FMS,      &AR[9][0],    BR[9][0][1], EXP_H3210, BR[9][1][1], EXP_H3210, BR[9][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0); /* stage#9  |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc1, 0xaaaa);                                                                                 /* stage#9  |  AR[8]    |      */
	    mop(OP_STWR,ex0, &AR[9][0],    oofs,   d07[CHIP], MSK_D0, d07w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#9  |    + ST   v      */
#if (H>8)                                                                                                                                   /*          *--------- BR[9]   */
	    exe(OP_CMP_LT,   &cc0,         l08[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9              LD     */
	    mop(OP_LDWR,  1, &BR[10][2][1],top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k]                       stage#10             |      */
	    mop(OP_LDWR,  1, &BR[10][0][1],d08[CHIP],   oofs, MSK_W0, d08w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#10 +->         |      */
	    mop(OP_LDWR,  1, &BR[10][1][1],d08[CHIP],   rofs, MSK_W0, d08w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#10 +->         |      */
	    exe(OP_FMS,      &AR[10][0],   BR[10][0][1],EXP_H3210, BR[10][1][1], EXP_H3210, BR[10][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* stage#10 |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#10 |  AR[9]    |      */
	    mop(OP_STWR,ex0, &AR[10][0],   oofs,   d08[CHIP], MSK_D0, d08w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#10 |    + ST   v      */
#if (H>9)                                                                                                                                   /*          *--------- BR[10]  */
	    exe(OP_CMP_LT,   &cc1,         l09[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10             LD     */
	    mop(OP_LDWR,  1, &BR[11][2][1],top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k]                       stage#11             |      */
	    mop(OP_LDWR,  1, &BR[11][0][1],d09[CHIP],   oofs, MSK_W0, d09w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#11 +->         |      */
	    mop(OP_LDWR,  1, &BR[11][1][1],d09[CHIP],   rofs, MSK_W0, d09w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#11 +->         |      */
	    exe(OP_FMS,      &AR[11][0],   BR[11][0][1],EXP_H3210, BR[11][1][1], EXP_H3210, BR[11][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* stage#11 |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc1, 0xaaaa);                                                                                 /* stage#11 |  AR[10]   |      */
	    mop(OP_STWR,ex0, &AR[11][0],   oofs,   d09[CHIP], MSK_D0, d09w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#11 |    + ST   v      */
#if (H>10)                                                                                                                                  /*          *--------- BR[11]  */
	    exe(OP_CMP_LT,   &cc0,         l10[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11             LD     */
	    mop(OP_LDWR,  1, &BR[12][2][1],top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#12             |      */
	    mop(OP_LDWR,  1, &BR[12][0][1],d10[CHIP],   oofs, MSK_W0, d10w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#12 +->         |      */
	    mop(OP_LDWR,  1, &BR[12][1][1],d10[CHIP],   rofs, MSK_W0, d10w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#12 +->         |      */
	    exe(OP_FMS,      &AR[12][0],   BR[12][0][1],EXP_H3210, BR[12][1][1], EXP_H3210, BR[12][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* stage#12 |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#12 |  AR[11]   |      */
	    mop(OP_STWR,ex0, &AR[12][0],   oofs,   d10[CHIP], MSK_D0, d10w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#12 |    + ST   v      */
#if (H>11)                                                                                                                                  /*          *--------- BR[12]  */
	    exe(OP_CMP_LT,   &cc1,         l11[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12             LD     */
	    mop(OP_LDWR,  1, &BR[13][2][1],top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#13             |      */
	    mop(OP_LDWR,  1, &BR[13][0][1],d11[CHIP],   oofs, MSK_W0, d11w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#13 +->         |      */
	    mop(OP_LDWR,  1, &BR[13][1][1],d11[CHIP],   rofs, MSK_W0, d11w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#13 +->         |      */
	    exe(OP_FMS,      &AR[13][0],   BR[13][0][1],EXP_H3210, BR[13][1][1], EXP_H3210, BR[13][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* stage#13 |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc1, 0xaaaa);                                                                                 /* stage#13 |  AR[12]   |      */
	    mop(OP_STWR,ex0, &AR[13][0],   oofs,   d11[CHIP], MSK_D0, d11w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#13 |    + ST   v      */
#if (H>12)                                                                                                                                  /*          *--------- BR[13]  */
	    exe(OP_CMP_LT,   &cc0,         l12[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13             LD     */
	    mop(OP_LDWR,  1, &BR[14][2][1],top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#14             |      */
	    mop(OP_LDWR,  1, &BR[14][0][1],d12[CHIP],   oofs, MSK_W0, d12w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#14 +->         |      */
	    mop(OP_LDWR,  1, &BR[14][1][1],d12[CHIP],   rofs, MSK_W0, d12w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#14 +->         |      */
	    exe(OP_FMS,      &AR[14][0],   BR[14][0][1],EXP_H3210, BR[14][1][1], EXP_H3210, BR[14][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* stage#14 |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#14 |  AR[13]   |      */
	    mop(OP_STWR,ex0, &AR[14][0],   oofs,   d12[CHIP], MSK_D0, d12w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#14 |    + ST   v      */
#if (H>13)                                                                                                                                  /*          *--------- BR[14]  */
	    exe(OP_CMP_LT,   &cc1,         l13[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14             LD     */
	    mop(OP_LDWR,  1, &BR[15][2][1],top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#15             |      */
	    mop(OP_LDWR,  1, &BR[15][0][1],d13[CHIP],   oofs, MSK_W0, d13w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#15 +->         |      */
	    mop(OP_LDWR,  1, &BR[15][1][1],d13[CHIP],   rofs, MSK_W0, d13w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#15 +->         |      */
	    exe(OP_FMS,      &AR[15][0],   BR[15][0][1],EXP_H3210, BR[15][1][1], EXP_H3210, BR[15][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* stage#15 |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc1, 0xaaaa);                                                                                 /* stage#15 |  AR[14]   |      */
	    mop(OP_STWR,ex0, &AR[15][0],   oofs,   d13[CHIP], MSK_D0, d13w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#15 |    + ST   v      */
#if (H>14)                                                                                                                                  /*          *--------- BR[15]  */
	    exe(OP_CMP_LT,   &cc0,         l14[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15             LD     */
	    mop(OP_LDWR,  1, &BR[16][2][1],top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#16             |      */
	    mop(OP_LDWR,  1, &BR[16][0][1],d14[CHIP],   oofs, MSK_W0, d14w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#16 +->         |      */
	    mop(OP_LDWR,  1, &BR[16][1][1],d14[CHIP],   rofs, MSK_W0, d14w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#16 +->         |      */
	    exe(OP_FMS,      &AR[16][0],   BR[16][0][1],EXP_H3210, BR[16][1][1], EXP_H3210, BR[16][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* stage#16 |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc0, 0xaaaa);                                                                                 /* stage#16 |  AR[15]   |      */
	    mop(OP_STWR,ex0, &AR[16][0],   oofs,   d14[CHIP], MSK_D0, d14w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#16 |    + ST   v      */
#if (H>15)                                                                                                                                  /*          *--------- BR[16]  */
	    exe(OP_CMP_LT,   &cc1,         l15[CHIP],   EXP_H3210, M,         EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16             LD     */
	    mop(OP_LDWR,  1, &BR[17][2][1],top,         cofs, MSK_W0, topw,       len, 0, 0, NULL, len);  /* A[p[i]*M+k] stage#1               stage#17             |      */
	    mop(OP_LDWR,  1, &BR[17][0][1],d15[CHIP],   oofs, MSK_W0, d15w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#17 +->         |      */
	    mop(OP_LDWR,  1, &BR[17][1][1],d15[CHIP],   rofs, MSK_W0, d15w[CHIP], len2,0, 1, NULL, len2); /* A[p[j+h*NCHIP+CHIP]*M+k]          stage#17 +->         |      */
	    exe(OP_FMS,      &AR[17][0],   BR[17][0][1],EXP_H3210, BR[17][1][1], EXP_H3210, BR[17][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* stage#17 |   ■■■  | 1.0  */
	    cex(OP_CEXE,     &ex0,   0, 0, 0, cc1, 0xaaaa);                                                                                 /* stage#17 |  AR[16]   |      */
	    mop(OP_STWR,ex0, &AR[17][0],   oofs,   d15[CHIP], MSK_D0, d15w[CHIP], len2, 0, 1, NULL, len2);                                  /* stage#17 |    + ST   v      */
                                                                                                                                            /*          *--------- BR[17]  */
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
          }
	}
      }
//EMAX5A end
      /********************************************/
    } /* j-loop */
//EMAX5A drain_dirty_lmm
    } /* else */
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
    Uint  l080[NCHIP],  l090[NCHIP],  l100[NCHIP],  l110[NCHIP],  l120[NCHIP],  l130[NCHIP],  l140[NCHIP],  l150[NCHIP];  /* (i+CHIP*W*H+h*W+w)        */
    Uint *t000[NCHIP], *t010[NCHIP], *t020[NCHIP], *t030[NCHIP], *t040[NCHIP], *t050[NCHIP], *t060[NCHIP], *t070[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+k] */
    Uint *t080[NCHIP], *t090[NCHIP], *t100[NCHIP], *t110[NCHIP], *t120[NCHIP], *t130[NCHIP], *t140[NCHIP], *t150[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+k] */
    Uint *t000w[NCHIP],*t010w[NCHIP],*t020w[NCHIP],*t030w[NCHIP],*t040w[NCHIP],*t050w[NCHIP],*t060w[NCHIP],*t070w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+k] */
    Uint *t080w[NCHIP],*t090w[NCHIP],*t100w[NCHIP],*t110w[NCHIP],*t120w[NCHIP],*t130w[NCHIP],*t140w[NCHIP],*t150w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+k] */
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
      l000[CHIP] = (i+CHIP*H+ 0+0)*M;	l010[CHIP] = (i+CHIP*H+ 1+0)*M;	l020[CHIP] = (i+CHIP*H+ 2+0)*M;	l030[CHIP] = (i+CHIP*H+ 3+0)*M;
      l040[CHIP] = (i+CHIP*H+ 4+0)*M;	l050[CHIP] = (i+CHIP*H+ 5+0)*M;	l060[CHIP] = (i+CHIP*H+ 6+0)*M;	l070[CHIP] = (i+CHIP*H+ 7+0)*M;
      l080[CHIP] = (i+CHIP*H+ 8+0)*M;	l090[CHIP] = (i+CHIP*H+ 9+0)*M;	l100[CHIP] = (i+CHIP*H+10+0)*M;	l110[CHIP] = (i+CHIP*H+11+0)*M;
      l120[CHIP] = (i+CHIP*H+12+0)*M;	l130[CHIP] = (i+CHIP*H+13+0)*M;	l140[CHIP] = (i+CHIP*H+14+0)*M;	l150[CHIP] = (i+CHIP*H+15+0)*M;
      t000[CHIP] = &b[l000[CHIP]+i];	t010[CHIP] = &b[l010[CHIP]+i];	t020[CHIP] = &b[l020[CHIP]+i];	t030[CHIP] = &b[l030[CHIP]+i];
      t040[CHIP] = &b[l040[CHIP]+i];	t050[CHIP] = &b[l050[CHIP]+i];	t060[CHIP] = &b[l060[CHIP]+i];	t070[CHIP] = &b[l070[CHIP]+i];
      t080[CHIP] = &b[l080[CHIP]+i];	t090[CHIP] = &b[l090[CHIP]+i];	t100[CHIP] = &b[l100[CHIP]+i];	t110[CHIP] = &b[l110[CHIP]+i];
      t120[CHIP] = &b[l120[CHIP]+i];	t130[CHIP] = &b[l130[CHIP]+i];	t140[CHIP] = &b[l140[CHIP]+i];	t150[CHIP] = &b[l150[CHIP]+i];
      t000w[CHIP]= (Ull)t000[CHIP];	t010w[CHIP]= (Ull)t010[CHIP];	t020w[CHIP]= (Ull)t020[CHIP];	t030w[CHIP]= (Ull)t030[CHIP];
      t040w[CHIP]= (Ull)t040[CHIP];	t050w[CHIP]= (Ull)t050[CHIP];	t060w[CHIP]= (Ull)t060[CHIP];	t070w[CHIP]= (Ull)t070[CHIP];
      t080w[CHIP]= (Ull)t080[CHIP];	t090w[CHIP]= (Ull)t090[CHIP];	t100w[CHIP]= (Ull)t100[CHIP];	t110w[CHIP]= (Ull)t110[CHIP];
      t120w[CHIP]= (Ull)t120[CHIP];	t130w[CHIP]= (Ull)t130[CHIP];	t140w[CHIP]= (Ull)t140[CHIP];	t150w[CHIP]= (Ull)t150[CHIP];	
    }
    /*for (j=1; j<M; j++) { *//*通常の連立一時方程式の場合*/
    for (j=i+1; j<M; j++) { /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでj=i+1から開始 */
      Uint *top  = &A[j*M+i];                                     /* A[p[j]*M+k] */
      Uint *topw = (Ull)top;
      /*Uint  len = (j+1)/2;*/
      Uint  len = j-i;/* bが単位行列の場合,k<iではb[]==0なのでk=iから開始 */
      /********************************************/
      if (len < 16) { /* len<1でも正常なので性能最大化で決めてよい */
	for (CHIP=0; CHIP<NCHIP; CHIP++) {
	  for (h=0; h<H; h++) { /* vertical (parallel) execution */
	  /*for (k=0; k<j; k++) { *//*通常の連立一時方程式の場合*/
	    for (k=i; k<j; k++) { /* 逆行列(b[]=E)の場合,k<iではb[]==0なのでk=iから開始 */
	      b[(i+CHIP*H+h)*M+j] -= A[j*M+k]*b[(i+CHIP*H+h)*M+k];
	    }
	  }
	}
      }
      else {
      Uint  jc = j-i;
      Ull   Ajk; /* k=0...j-1 */
      Ull   b000, b001;
      Uint *d000[NCHIP], *d010[NCHIP], *d020[NCHIP], *d030[NCHIP], *d040[NCHIP], *d050[NCHIP], *d060[NCHIP], *d070[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+j] */
      Uint *d080[NCHIP], *d090[NCHIP], *d100[NCHIP], *d110[NCHIP], *d120[NCHIP], *d130[NCHIP], *d140[NCHIP], *d150[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+j] */
      Uint *d000w[NCHIP],*d010w[NCHIP],*d020w[NCHIP],*d030w[NCHIP],*d040w[NCHIP],*d050w[NCHIP],*d060w[NCHIP],*d070w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+j] */
      Uint *d080w[NCHIP],*d090w[NCHIP],*d100w[NCHIP],*d110w[NCHIP],*d120w[NCHIP],*d130w[NCHIP],*d140w[NCHIP],*d150w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+j] */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	d000[CHIP] = &b[l000[CHIP]+j];	d010[CHIP] = &b[l010[CHIP]+j];	d020[CHIP] = &b[l020[CHIP]+j];	d030[CHIP] = &b[l030[CHIP]+j];
	d040[CHIP] = &b[l040[CHIP]+j];	d050[CHIP] = &b[l050[CHIP]+j];	d060[CHIP] = &b[l060[CHIP]+j];	d070[CHIP] = &b[l070[CHIP]+j];
	d080[CHIP] = &b[l080[CHIP]+j];	d090[CHIP] = &b[l090[CHIP]+j];	d100[CHIP] = &b[l100[CHIP]+j];	d110[CHIP] = &b[l110[CHIP]+j];
	d120[CHIP] = &b[l120[CHIP]+j];	d130[CHIP] = &b[l130[CHIP]+j];	d140[CHIP] = &b[l140[CHIP]+j];	d150[CHIP] = &b[l150[CHIP]+j];
	d000w[CHIP]= (Ull)d000[CHIP];	d010w[CHIP]= (Ull)d010[CHIP];	d020w[CHIP]= (Ull)d020[CHIP];	d030w[CHIP]= (Ull)d030[CHIP];
	d040w[CHIP]= (Ull)d040[CHIP];	d050w[CHIP]= (Ull)d050[CHIP];	d060w[CHIP]= (Ull)d060[CHIP];	d070w[CHIP]= (Ull)d070[CHIP];
	d080w[CHIP]= (Ull)d080[CHIP];	d090w[CHIP]= (Ull)d090[CHIP];	d100w[CHIP]= (Ull)d100[CHIP];	d110w[CHIP]= (Ull)d110[CHIP];
	d120w[CHIP]= (Ull)d120[CHIP];	d130w[CHIP]= (Ull)d130[CHIP];	d140w[CHIP]= (Ull)d140[CHIP];	d150w[CHIP]= (Ull)d150[CHIP];
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
#if (H>1)
	  mop(OP_LDWR,  1, &BR[2][3][1], t010[CHIP], cofs, MSK_W0, t010w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#2.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d010[CHIP], 0,    MSK_W0, d010w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#3.0  |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[2][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#3.0  +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d010[CHIP], MSK_D0, d010w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#3.0  +--------- xxx     */
#if (H>2)
	  mop(OP_LDWR,  1, &BR[3][3][1], t020[CHIP], cofs, MSK_W0, t020w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#3.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d020[CHIP], 0,    MSK_W0, d020w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#4.0  |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[3][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#4.0  +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d020[CHIP], MSK_D0, d020w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#4.0  +--------- xxx     */
#if (H>3)
	  mop(OP_LDWR,  1, &BR[4][3][1], t030[CHIP], cofs, MSK_W0, t030w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#4.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d030[CHIP], 0,    MSK_W0, d030w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#5.0  |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[4][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#5.0  +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d030[CHIP], MSK_D0, d030w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#5.0  +--------- xxx     */
#if (H>4)
	  mop(OP_LDWR,  1, &BR[5][3][1], t040[CHIP], cofs, MSK_W0, t040w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#5.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d040[CHIP], 0,    MSK_W0, d040w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#6.0  |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[5][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#6.0  +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d040[CHIP], MSK_D0, d040w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#6.0  +--------- xxx     */
#if (H>5)
	  mop(OP_LDWR,  1, &BR[6][3][1], t050[CHIP], cofs, MSK_W0, t050w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#6.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d050[CHIP], 0,    MSK_W0, d050w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#7.0  |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[6][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#7.0  +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d050[CHIP], MSK_D0, d050w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#7.0  +--------- xxx     */
#if (H>6)
	  mop(OP_LDWR,  1, &BR[7][3][1], t060[CHIP], cofs, MSK_W0, t060w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#7.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d060[CHIP], 0,    MSK_W0, d060w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#8.0  |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[7][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#8.0  +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d060[CHIP], MSK_D0, d060w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#8.0  +--------- xxx     */
#if (H>7)
	  mop(OP_LDWR,  1, &BR[8][3][1], t070[CHIP], cofs, MSK_W0, t070w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#8.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d070[CHIP], 0,    MSK_W0, d070w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#9.0  |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[8][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#9.0  +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d070[CHIP], MSK_D0, d070w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#9.0  +--------- xxx     */
#if (H>8)
	  mop(OP_LDWR,  1, &BR[9][3][1], t080[CHIP], cofs, MSK_W0, t080w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#9.3  +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d080[CHIP], 0,    MSK_W0, d080w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#10.0 |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[9][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#10.0 +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d080[CHIP], MSK_D0, d080w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#10.0 +--------- xxx     */
#if (H>9)
	  mop(OP_LDWR,  1, &BR[10][3][1],t090[CHIP], cofs, MSK_W0, t090w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#10.3 +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d090[CHIP], 0,    MSK_W0, d090w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#11.0 |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[10][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#11.0 +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d090[CHIP], MSK_D0, d090w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#11.0 +--------- xxx     */
#if (H>10)
	  mop(OP_LDWR,  1, &BR[11][3][1],t100[CHIP], cofs, MSK_W0, t100w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#11.3 +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d100[CHIP], 0,    MSK_W0, d100w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#12.0 |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[11][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#12.0 +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d100[CHIP], MSK_D0, d100w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#12.0 +--------- xxx     */
#if (H>11)
	  mop(OP_LDWR,  1, &BR[12][3][1],t110[CHIP], cofs, MSK_W0, t110w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#12.3 +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d110[CHIP], 0,    MSK_W0, d110w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#13.0 |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[12][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#13.0 +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d110[CHIP], MSK_D0, d110w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#13.0 +--------- xxx     */
#if (H>12)
	  mop(OP_LDWR,  1, &BR[13][3][1],t120[CHIP], cofs, MSK_W0, t120w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#13.3 +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d120[CHIP], 0,    MSK_W0, d120w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#14.0 |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[13][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#14.0 +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d120[CHIP], MSK_D0, d120w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#14.0 +--------- xxx     */
#if (H>13)
	  mop(OP_LDWR,  1, &BR[14][3][1],t130[CHIP], cofs, MSK_W0, t130w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#14.3 +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d130[CHIP], 0,    MSK_W0, d130w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#15.0 |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[14][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#15.0 +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d130[CHIP], MSK_D0, d130w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#15.0 +--------- xxx     */
#if (H>14)
	  mop(OP_LDWR,  1, &BR[15][3][1],t140[CHIP], cofs, MSK_W0, t140w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#15.3 +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d140[CHIP], 0,    MSK_W0, d140w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#16.0 |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[15][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#16.0 +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d140[CHIP], MSK_D0, d140w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#16.0 +--------- xxx     */
#if (H>15)
	  mop(OP_LDWR,  1, &BR[16][3][1],t150[CHIP], cofs, MSK_W0, t150w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#16.3 +->xxx      LD     */
	  mop(OP_LDWR,  1, &b000,        d150[CHIP], 0,    MSK_W0, d150w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#17.0 |   ■■■  |      */
	  exe(OP_FMS,      &b000,        b000,       EXP_H3210, Ajk,    EXP_H3210, BR[16][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);     /* stage#17.0 +- xxx+ST   v      */
	  mop(OP_STWR,  1, &b000,        0,    d150[CHIP], MSK_D0, d150w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#17.0 +--------- xxx     */
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
	}
      }
//EMAX5A end
//EMAX5A drain_dirty_lmm
      } /* else */
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
        if (len < 16) { /* len<1でも正常なので性能最大化で決めてよい */
	  for (CHIP=0; CHIP<NCHIP; CHIP++) {
	    for (h=0; h<H; h++) { /* vertical (parallel) execution */
	      for (k=M-1; k>j; k--) { /* 最内列方向 */
		b[(i+CHIP*H+h)*M+j] -= A[j*M+k]*x[(i+CHIP*H+h)*M+k];
	      }
	    }
	  }
	}
        else {
	Uint  jc = M-j-1;
	Ull   Ajk; /* k=j+1...M-1 */
	Ull   b000, b001;
	Uint  l000[NCHIP],  l010[NCHIP],  l020[NCHIP],  l030[NCHIP],  l040[NCHIP],  l050[NCHIP],  l060[NCHIP],  l070[NCHIP];  /* (i+CHIP*W*H+h*W+w)        */
	Uint  l080[NCHIP],  l090[NCHIP],  l100[NCHIP],  l110[NCHIP],  l120[NCHIP],  l130[NCHIP],  l140[NCHIP],  l150[NCHIP];  /* (i+CHIP*W*H+h*W+w)        */
	Uint *t000[NCHIP], *t010[NCHIP], *t020[NCHIP], *t030[NCHIP], *t040[NCHIP], *t050[NCHIP], *t060[NCHIP], *t070[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+k] */
	Uint *t080[NCHIP], *t090[NCHIP], *t100[NCHIP], *t110[NCHIP], *t120[NCHIP], *t130[NCHIP], *t140[NCHIP], *t150[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+k] */
	Uint *t000w[NCHIP],*t010w[NCHIP],*t020w[NCHIP],*t030w[NCHIP],*t040w[NCHIP],*t050w[NCHIP],*t060w[NCHIP],*t070w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+k] */
	Uint *t080w[NCHIP],*t090w[NCHIP],*t100w[NCHIP],*t110w[NCHIP],*t120w[NCHIP],*t130w[NCHIP],*t140w[NCHIP],*t150w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+k] */
	Uint *d000[NCHIP], *d010[NCHIP], *d020[NCHIP], *d030[NCHIP], *d040[NCHIP], *d050[NCHIP], *d060[NCHIP], *d070[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+j] */
	Uint *d080[NCHIP], *d090[NCHIP], *d100[NCHIP], *d110[NCHIP], *d120[NCHIP], *d130[NCHIP], *d140[NCHIP], *d150[NCHIP];  /* b[(i+CHIP*W*H+h*W+w)*M+j] */
	Uint *d000w[NCHIP],*d010w[NCHIP],*d020w[NCHIP],*d030w[NCHIP],*d040w[NCHIP],*d050w[NCHIP],*d060w[NCHIP],*d070w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+j] */
	Uint *d080w[NCHIP],*d090w[NCHIP],*d100w[NCHIP],*d110w[NCHIP],*d120w[NCHIP],*d130w[NCHIP],*d140w[NCHIP],*d150w[NCHIP]; /* b[(i+CHIP*W*H+h*W+w)*M+j] */
	for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	  l000[CHIP] = (i+CHIP*H+ 0+0)*M+j+1;	l010[CHIP] = (i+CHIP*H+ 1+0)*M+j+1;	      l020[CHIP] = (i+CHIP*H+ 2+0)*M+j+1;	l030[CHIP] = (i+CHIP*H+ 3+0)*M+j+1;	    
	  l040[CHIP] = (i+CHIP*H+ 4+0)*M+j+1;	l050[CHIP] = (i+CHIP*H+ 5+0)*M+j+1;	      l060[CHIP] = (i+CHIP*H+ 6+0)*M+j+1;	l070[CHIP] = (i+CHIP*H+ 7+0)*M+j+1;	    
	  l080[CHIP] = (i+CHIP*H+ 8+0)*M+j+1;	l090[CHIP] = (i+CHIP*H+ 9+0)*M+j+1;	      l100[CHIP] = (i+CHIP*H+10+0)*M+j+1;	l110[CHIP] = (i+CHIP*H+11+0)*M+j+1;	    
	  l120[CHIP] = (i+CHIP*H+12+0)*M+j+1;	l130[CHIP] = (i+CHIP*H+13+0)*M+j+1;	      l140[CHIP] = (i+CHIP*H+14+0)*M+j+1;	l150[CHIP] = (i+CHIP*H+15+0)*M+j+1;	    
	  t000[CHIP] = &x[l000[CHIP]];	 t010[CHIP] = &x[l010[CHIP]];	t020[CHIP] = &x[l020[CHIP]];   t030[CHIP] = &x[l030[CHIP]];		    
	  t040[CHIP] = &x[l040[CHIP]];	 t050[CHIP] = &x[l050[CHIP]];	t060[CHIP] = &x[l060[CHIP]];   t070[CHIP] = &x[l070[CHIP]];		    
	  t080[CHIP] = &x[l080[CHIP]];	 t090[CHIP] = &x[l090[CHIP]];	t100[CHIP] = &x[l100[CHIP]];   t110[CHIP] = &x[l110[CHIP]];		    
	  t120[CHIP] = &x[l120[CHIP]];	 t130[CHIP] = &x[l130[CHIP]];	t140[CHIP] = &x[l140[CHIP]];   t150[CHIP] = &x[l150[CHIP]];		    
	  t000w[CHIP]= (Ull)t000[CHIP];  t010w[CHIP]= (Ull)t010[CHIP];  t020w[CHIP]= (Ull)t020[CHIP];  t030w[CHIP]= (Ull)t030[CHIP];
	  t040w[CHIP]= (Ull)t040[CHIP];  t050w[CHIP]= (Ull)t050[CHIP];  t060w[CHIP]= (Ull)t060[CHIP];  t070w[CHIP]= (Ull)t070[CHIP];
	  t080w[CHIP]= (Ull)t080[CHIP];  t090w[CHIP]= (Ull)t090[CHIP];  t100w[CHIP]= (Ull)t100[CHIP];  t110w[CHIP]= (Ull)t110[CHIP];
	  t120w[CHIP]= (Ull)t120[CHIP];  t130w[CHIP]= (Ull)t130[CHIP];  t140w[CHIP]= (Ull)t140[CHIP];  t150w[CHIP]= (Ull)t150[CHIP];
	  d000[CHIP] = &b[l000[CHIP]-1]; d010[CHIP] = &b[l010[CHIP]-1];	d020[CHIP] = &b[l020[CHIP]-1]; d030[CHIP] = &b[l030[CHIP]-1];		    
	  d040[CHIP] = &b[l040[CHIP]-1]; d050[CHIP] = &b[l050[CHIP]-1];	d060[CHIP] = &b[l060[CHIP]-1]; d070[CHIP] = &b[l070[CHIP]-1];		    
	  d080[CHIP] = &b[l080[CHIP]-1]; d090[CHIP] = &b[l090[CHIP]-1];	d100[CHIP] = &b[l100[CHIP]-1]; d110[CHIP] = &b[l110[CHIP]-1];		    
	  d120[CHIP] = &b[l120[CHIP]-1]; d130[CHIP] = &b[l130[CHIP]-1];	d140[CHIP] = &b[l140[CHIP]-1]; d150[CHIP] = &b[l150[CHIP]-1];		    
	  d000w[CHIP]= (Ull)d000[CHIP];  d010w[CHIP]= (Ull)d010[CHIP];  d020w[CHIP]= (Ull)d020[CHIP];  d030w[CHIP]= (Ull)d030[CHIP];
	  d040w[CHIP]= (Ull)d040[CHIP];  d050w[CHIP]= (Ull)d050[CHIP];  d060w[CHIP]= (Ull)d060[CHIP];  d070w[CHIP]= (Ull)d070[CHIP];
	  d080w[CHIP]= (Ull)d080[CHIP];  d090w[CHIP]= (Ull)d090[CHIP];  d100w[CHIP]= (Ull)d100[CHIP];  d110w[CHIP]= (Ull)d110[CHIP];
	  d120w[CHIP]= (Ull)d120[CHIP];  d130w[CHIP]= (Ull)d130[CHIP];  d140w[CHIP]= (Ull)d140[CHIP];  d150w[CHIP]= (Ull)d150[CHIP];
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
#if (H>1)
	    mop(OP_LDWR,  1, &BR[2][3][1], t010[CHIP],   cofs, MSK_W0, t010w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#2.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d010[CHIP],   0,    MSK_W0, d010w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#3.0  |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[2][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#3.0  +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d010[CHIP], MSK_D0, d010w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#3.0  +--------- xxx     */
#if (H>2)
	    mop(OP_LDWR,  1, &BR[3][3][1], t020[CHIP],   cofs, MSK_W0, t020w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#3.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d020[CHIP],   0,    MSK_W0, d020w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#4.0  |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[3][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#4.0  +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d020[CHIP], MSK_D0, d020w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#4.0  +--------- xxx     */
#if (H>3)
	    mop(OP_LDWR,  1, &BR[4][3][1], t030[CHIP],   cofs, MSK_W0, t030w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#4.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d030[CHIP],   0,    MSK_W0, d030w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#5.0  |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[4][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#5.0  +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d030[CHIP], MSK_D0, d030w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#5.0  +--------- xxx     */
#if (H>4)
	    mop(OP_LDWR,  1, &BR[5][3][1], t040[CHIP],   cofs, MSK_W0, t040w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#5.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d040[CHIP],   0,    MSK_W0, d040w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#6.0  |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[5][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#6.0  +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d040[CHIP], MSK_D0, d040w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#6.0  +--------- xxx     */
#if (H>5)
	    mop(OP_LDWR,  1, &BR[6][3][1], t050[CHIP],   cofs, MSK_W0, t050w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#6.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d050[CHIP],   0,    MSK_W0, d050w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#7.0  |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[6][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#7.0  +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d050[CHIP], MSK_D0, d050w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#7.0  +--------- xxx     */
#if (H>6)
	    mop(OP_LDWR,  1, &BR[7][3][1], t060[CHIP],   cofs, MSK_W0, t060w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#7.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d060[CHIP],   0,    MSK_W0, d060w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#8.0  |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[7][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#8.0  +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d060[CHIP], MSK_D0, d060w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#8.0  +--------- xxx     */
#if (H>7)
	    mop(OP_LDWR,  1, &BR[8][3][1], t070[CHIP],   cofs, MSK_W0, t070w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#8.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d070[CHIP],   0,    MSK_W0, d070w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#9.0  |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[8][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#9.0  +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d070[CHIP], MSK_D0, d070w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#9.0  +--------- xxx     */
#if (H>8)
	    mop(OP_LDWR,  1, &BR[9][3][1], t080[CHIP],   cofs, MSK_W0, t080w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#9.3  +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d080[CHIP],   0,    MSK_W0, d080w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#10.0 |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[9][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#10.0 +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d080[CHIP], MSK_D0, d080w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#10.0 +--------- xxx     */
#if (H>9)
	    mop(OP_LDWR,  1, &BR[10][3][1],t090[CHIP],   cofs, MSK_W0, t090w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#10.3 +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d090[CHIP],   0,    MSK_W0, d090w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#11.0 |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[10][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#11.0 +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d090[CHIP], MSK_D0, d090w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#11.0 +--------- xxx     */
#if (H>10)
	    mop(OP_LDWR,  1, &BR[11][3][1],t100[CHIP],   cofs, MSK_W0, t100w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#11.3 +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d100[CHIP],   0,    MSK_W0, d100w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#12.0 |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[11][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#12.0 +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d100[CHIP], MSK_D0, d100w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#12.0 +--------- xxx     */
#if (H>11)
	    mop(OP_LDWR,  1, &BR[12][3][1],t110[CHIP],   cofs, MSK_W0, t110w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#12.3 +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d110[CHIP],   0,    MSK_W0, d110w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#13.0 |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[12][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#13.0 +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d110[CHIP], MSK_D0, d110w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#13.0 +--------- xxx     */
#if (H>12)
	    mop(OP_LDWR,  1, &BR[13][3][1],t120[CHIP],   cofs, MSK_W0, t120w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#13.3 +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d120[CHIP],   0,    MSK_W0, d120w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#14.0 |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[13][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#14.0 +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d120[CHIP], MSK_D0, d120w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#14.0 +--------- xxx     */
#if (H>13)
	    mop(OP_LDWR,  1, &BR[14][3][1],t130[CHIP],   cofs, MSK_W0, t130w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#14.3 +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d130[CHIP],   0,    MSK_W0, d130w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#15.0 |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[14][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#15.0 +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d130[CHIP], MSK_D0, d130w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#15.0 +--------- xxx     */
#if (H>14)
	    mop(OP_LDWR,  1, &BR[15][3][1],t140[CHIP],   cofs, MSK_W0, t140w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#15.3 +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d140[CHIP],   0,    MSK_W0, d140w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#16.0 |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[15][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#16.0 +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d140[CHIP], MSK_D0, d140w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#16.0 +--------- xxx     */
#if (H>15)
	    mop(OP_LDWR,  1, &BR[16][3][1],t150[CHIP],   cofs, MSK_W0, t150w[CHIP], len, 0, 1, NULL, len);  /* b[(i+CHIP*W*H+h*W+0)*M+k]    *//* stage#16.3 +->xxx      LD     *//*read-modify-write + exe-loop*/
	    mop(OP_LDWR,  1, &b000,        d150[CHIP],   0,    MSK_W0, d150w[CHIP], 1,   0, 1, NULL, 1);    /* b[(i+CHIP*W*H+h*W+0)*M+j]    *//* stage#17.0 |   ■■■  |      */
	    exe(OP_FMS,      &b000,        b000,         EXP_H3210, Ajk, EXP_H3210, BR[16][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);        /* stage#17.0 +- xxx+ST   v      */
	    mop(OP_STWR,  1, &b000,        0,      d150[CHIP], MSK_D0, d150w[CHIP], 1,   0, 1, NULL, 1);                                      /* stage#17.0 +--------- xxx     */
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
	  }
        }
//EMAX5A end
//EMAX5A drain_dirty_lmm
        } /* else */
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
#endif
