
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
/* A A   B B B B B B   C C C C C C */
/* A A   B B B B B B   C C C C C C */
/* A A                 C C C C C C */
/* A A                 C C C C C C */
/* L=2, M1=4, M2=6     L<M1,M2     */

#define L  480LL
#define M1 480LL
#define M2 480LL
#define RMGRP 30
/*#define NCHIP 4*/
#define NCHIP 1
#define W  4LL
#define H  12
Uint *A;  /*[M1][L];*/
Uint *B;  /*[L][M2];*/
Uint *C0; /*[M1][M2];*/
Uint *C1; /*[M1][M2];*/
int row, col, n;
int top, blk;
int w, h;
int count0, count1, count2;

#define CSIMWD 320
#define CSIMHT 240
#define CSIMBM (CSIMWD*CSIMHT)
Uint Z[CSIMBM];

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))

main()
{
  sysinit((Uint)(M1*L*sizeof(Uint)
                +L*M2*sizeof(Uint)
                +M1*M2*sizeof(Uint)
                +M1*M2*sizeof(Uint)),32);
  printf("membase: %08.8x\n", (Uint)membase);
  A  = (Uint*)membase;
  B  = (Uint*)((Uchar*)A  + M1*L*sizeof(Uint));
  C0 = (Uint*)((Uchar*)B  + L*M2*sizeof(Uint));
  C1 = (Uint*)((Uchar*)C0 + M1*M2*sizeof(Uint));
  printf("A : %08.8x\n", A);
  printf("B : %08.8x\n", B);
  printf("C0: %08.8x\n", C0);
  printf("C1: %08.8x\n", C1);

  for (row=0; row<M1; row++) {
    for (col=0; col<L; col++)
      *(float*)&A[row*L+col] = row%120+1;
  }
  for (row=0; row<L; row++) {
    for (col=0; col<M2; col++)
      *(float*)&B[row*M2+col] = col%120+1;
  }

#if !defined(ARMSIML)
  x11_open(0);
#endif

  reset_nanosec();
  orig();
  get_nanosec(0);
  show_nanosec();

  reset_nanosec();
  imax();
  get_nanosec(0);
  show_nanosec();

#ifdef ARMSIML
  copy_Z(0, C1); _copyX(0, Z);
  copy_Z(1, C1); _copyX(1, Z);
  copy_Z(4, C1); _copyX(4, Z);
  copy_Z(5, C1); _copyX(5, Z);
  copy_Z(8, C1); _copyX(8, Z);
  copy_Z(9, C1); _copyX(9, Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, C1); BGR_to_X(0, Z);
  copy_Z(1, C1); BGR_to_X(1, Z);
  copy_Z(4, C1); BGR_to_X(5, Z);
  copy_Z(5, C1); BGR_to_X(6, Z);
  copy_Z(8, C1); BGR_to_X(10,Z);
  copy_Z(9, C1); BGR_to_X(11,Z);
  x11_update();
#endif

  printf("Num of MULT: orig=%d imax=%d\n", count0, count1);

  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      if (C0[row*M2+col] != C1[row*M2+col]) {
        count2++;
        printf("C0[%d][%d]=%f C1[%d][%d]=%f\n", row, col, (double)*(float*)&C0[row*M2+col],
                                                row, col, (double)*(float*)&C1[row*M2+col]);
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");

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
  case 4:  offs = from + M2*HT;        break;
  case 5:  offs = from + M2*HT+WD;     break;
  case 6:  offs = from + M2*HT+WD*2;   break;
  case 7:  offs = from + M2*HT+WD*3;   break;
  case 8:  offs = from + M2*HT*2;      break;
  case 9:  offs = from + M2*HT*2+WD;   break;
  case 10: offs = from + M2*HT*2+WD*2; break;
  case 11: offs = from + M2*HT*2+WD*3; break;
  case 12: offs = from + M2*HT*3;      break;
  case 13: offs = from + M2*HT*3+WD;   break;
  case 14: offs = from + M2*HT*3+WD*2; break;
  case 15: offs = from + M2*HT*3+WD*3; break;
  }
  for (i=0; i<HT; i++, offs+=M2) {
    if (offs<from+M1*M2) {
      for (j=0; j<WD; j++) {
	if (j+(id%4)*WD<M2) *to++ = (*(offs+j))>>0;
	else                *to++ = 0;
      }
    }
    else {
      for (j=0; j<WD; j++)
	*to++ = 0;
    }
  }
}

orig() {
  printf("<<<ORIG>>>\n");
  for (row=0; row<M1; row++) {
    for (col=0; col<M2; col++) {
      for (n=0; n<L; n++) {
        if (n==0) *(float*)&C0[row*M2+col]  = *(float*)&A[row*L+n] * *(float*)&B[n*M2+col];
        else      *(float*)&C0[row*M2+col] += *(float*)&A[row*L+n] * *(float*)&B[n*M2+col];
        count0++;
        /*printf("[%d %d %d]", row, col, n);*/
      }
      /*printf("\n");*/
    }
  }
}

#if 0
imax() {
  Ull CHIP;
  Ull rofs;
  printf("<<<IMAX>>>\n");
  for (top=0; top<M1/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    for (blk=0; blk<L; blk+=H) { /* 3重ループ目 (Cが確定するまでのDMA入れ換えはR/Wを伴うためオーバヘッドになる. Bのbroadcast回数を増やす方が結果的に高速) */
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
          /*【3重ループ制御方法】                                                                                                                                                              */
          /*    loop0-reg 4           4           4           4           4           4           4           4           4           4           4           4           4                    */
          /*                 3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  3  2  1  0  - (stop1=1)       */
          /*    loop1-reg 4                                               4                                               4                                               4                    */
          /*                          3           2           1           0           3           2           1           0           3           2           1           0                    */
          /*    loop2-reg 3                                                                                                                                               3                    */
          /*                                                              2                                               1                                               0                    */
          /*                                                                                                                                                   【★Ａ★】 ↑arbrk=1(停止)      */
          /*                ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex ex --------- */
          /*        loop0    0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3  - (stop1=1)       */
          /*        loop1    0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3  0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3  0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3                    */
          /*        loop2    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2                    */

          /*【★Ａ★】部分の拡大              0         1         2         3      |  0         1         2         3      |  0         1         2         3      |  0                        */
          /*                 unit1 clk  ___/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____                  */
          /*                   stage1d  ----< loop0=2 X loop1=1 X loop2=1 X ======= X loop0=1 X loop1=1 X loop2=1 X ======= X loop0=4★-----------------------------X loop0=3                  */
          /*                                                                                                                  init0=1★                            |                           */
          /*                   stage2d  --------------< 0nzero  X nop     X nop     X ======= X 0zero★ X 1zero★ X 2zero★ X ======= >--------------------------------------                  */
          /*                                                                                                ↑        ↑                                           |                           */
          /*                                                                                           前cycle:zeroの場合decr                                      |                           */
          /*                   stage3d  ------------------------< loop0=1 X loop1=1 X loop2=1 X ======= X loop0=0 X loop1=0 X lop2=0★X ======= X loop0=3 >------------------                  */
          /*                                                                                                            ↑stage1dに戻る                            |                           */
          /*                                                                                                            │この時0ならstage1dにinit0=1を通知.stage1dに初期値をBRからセット      */
          /*                                                                                                            │init0=1は下方(BR)に伝搬                  |                           */
          /*                   stage4d  ----------------------------------< loop0=1 X loop1=1 X loop2=1 X ======= X loop0=0 X lop1=0  X lop2=0★X ======= X loop0=3 >--------       init0=1が  */
          /*                                                                                                                                                init0=1|              │次段unitへ */
          /*                                                                                                                                                       |stop1=1       ↓           */
          /*                                  0         1         2         3      |  0         1         2         3      |  0         1         2         3      |  0        stage1dに初期値 */
          /*                 unit2 clk  ___/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____/~~~~\____                  */
          /*                                                                                                          ★OP_WHILEが存在  かつ zeroの場合にarbrk=1セット★                       */
          /*                【3重ループのパターン】                                                       ★col=0から順に調べ nonzeroの場合はarbrk=0に戻す★                                   */
          /*                                     col2=0  col1=0  col0=0                                                                                                                        */
          /*                                       ↓      ↓      ↓     DMA以外のレジスタ値設定を自動化                                                                                      */
          /*                                     arbrk=1 init1=1 init0=1  受信したら初期値再セット                                                                                             */
          /*                                                   0       1                                                                                                                       */
          /*                                                   1       0                                                                                                                       */
          /*                                                   1       1                                                                                                                       */
          /*                                           1       X       X  IMAX終了                                                                                                             */
          /*                                                                                                                                                                                   */
          /*                【2重ループのパターン】      col1=0  col0=0                                                                                                                        */
          /*                                               ↓      ↓     DMA以外のレジスタ値設定を自動化                                                                                      */
          /*                                             arbrk=1 init0=1  受信したら初期値再セット                                                                                             */
          /*                                                   0       1  A先頭はA[0][0]からA[1][0]に変更（480x4B加算）                                                                        */
          /*                                                              B先頭は元に戻す(ofs=-Wx4に戻す:実際にはselfloopを一度解除しBRから入力するだけ)                                       */
          /*                                                              RANGEは60行x480列x4B=115200を加算(lenは無変更)                                                                       */
          /*                                                                exe(OP_ADD, &ofs, ★INIT0, ofs, EXP_H3210, W*4, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              */
          /*                                                                                            ↑Cの記述はそのまま.IMAXではBRに初期値が残っているので利用                             */
          /*                                                                mop(OP_LDWR,   1, &BR[1][0][1],  ★(Ull)b000, (Ull)ofs, MSK_D0, ★(Ull)b00, M/2, 0, 0, (Ull)NULL, M/2);            */
          /*                                                                                            ↑Cの記述はそのまま.2重ループの範囲では増分指定不要                                    */
          /*                                                   1       X  IMAX終了                                                                                                             */

          /*【3重ループ参照パターン】                                                                                                                                                          */
          /* a0000 a0001 a0002 a0003 a0004 ... a0059 | a0060 a0061 ... *//* LMM[ 0] b0000 b0001 b0002 b0003 b0004 ... b0059 b0099 ... *//* partial c0000 c0001 c0002 c0003 c0004 ... c0099 ... */
          /* a0100 a0101 a0102 a0103 a0104 ... a0159 | a0160 a0161 ... *//* LMM[ 1] b0100 b0101 b0102 b0103 b0104 ... b0159 b0199 ... *//* partial c0100 c0101 c0102 c0103 c0104 ... c0199 ... */
          /* a0200 a0201 a0202 a0203 a0204 ... a0259 | a0260 a0261 ... *//* LMM[ 2] b0200 b0201 b0202 b0203 b0204 ... b0259 b0299 ... *//* partial c0200 c0201 c0202 c0203 c0204 ... c0299 ... */
          /* a0300 a0301 a0302 a0303 a0304 ... a0359 | a0360 a0361 ... *//* LMM[ 3] b0300 b0301 b0302 b0303 b0304 ... b0359 b0399 ... *//* partial c0300 c0301 c0302 c0303 c0304 ... c0399 ... */
          /*                                                           *//* LMM[59] b5900 b5901 b5902 b5903 b5904 ... b5959 b5999 ... *//*                                                     */
          /*                                                           *//* --------------------------------------------------------- *//*                                                     */
          /* a9900 a9901 a9902 a9903 a9904 ... a9959 | a9960 a9961 ... *//* LMM[99] b9900 b9901 b9902 b9903 b9904 ... b9959 b9999 ... *//* partial c9900 c9901 c9902 c9903 c9904 ... c9999 ... */

          /*【3重ループ実行手順】                                                                                                                                                              */
          /* ================================== 3重ループ開始 ================================================================================================================================ */
          /*   LMM00: A[0:7][0:479] (必要なのはa000000,001000,...007000) 480x8x4B=16KB | B[00][0:479] 480x4B=2KB                                                                               */
          /*                                   a000060,001060,...007060                |                                                                                                       */
          /*                                   a000420,001420,...007420                |                                                                                                       */
          /*   LMM01: A[0:7][0:479] (必要なのはa000001,001001,...007001)               | B[01][0:479] 480x4B=2KB                                                                               */
          /*                                   a000061,001061,...007061                |                                                                                                       */
          /*                                   a000421,001421,...007421                |                                                                                                       */
          /*   LMM59: A[0:7][0:479] (必要なのはa000059,001059,...007059) 8要素         | B[59][0:479] 480x4B=2KB                                                                               */
          /*                                   a000119,001119,...007119  8要素         |                                                                                                       */
          /*                                   a000479,001479,...007479  8要素x8blk分  |                                                                                                       */
          /*   LMM60: ---------------------------------------------------------------------------------------- C[0:7][0:479] 480x8x4B=16KB  Cの途中結果をR/Wで入れ換えるのは損                 */
          /*                                                                                                   multi-chipに対するR/Wがシリアライズされるし,LMMのREADは遅いので回数を減らすべき */
          /* ★RANGE設定,A[0:7][0:479]を一度に供給                                                                                           ★RANGE設定,C[0:7][*]初期化書き込み               */
          /* ================================== 3重ループ先頭開始 BLK=0======================================================================================================================= */
          /*                                                                        ★BのREGV+RANGEを設定,次のB[  0: 59]を1回のburstで供給(MC-broadcast)                                       */
          /* ---------------------------------- 2重ループIMAX開始 --3重loopのblk-iteration-- 1回目 (blk=0)                            *//*                                                     */
          /* A[0][  0: 59]->LMMを再利用                                *//*   row= 0: B[  0][*]:480*4B=2KB /LMM(1/2) b00=B+(blk+ 0)*M *//*  row=0: C[0][*]:480*4B=2KB / LMM┐                  */
          /* A[1][  0: 59]->LMMを再利用                                *//*   row= 1: B[  1][*]:480*4B=2KB /LMM(1/2) b01=B+(blk+ 1)*M *//*  row=1: C[1][*]:480*4B=2KB / LMM│合計16KBは実際は  */
          /* A[7][  0: 59]->LMMを再利用                                *//*   row=59: B[ 59][*]:480*4B=2KB /LMM(1/2) b59=B+(blk+59)*M *//*  row=7: C[7][*]:480*4B=2KB / LMM┘1LMMに収容可能    */
          /* ---------------------------------- IMAX動作一旦終了 ----------------------------------------------------------------------------------------------------------------------------- */
          /*                                                                        ★BのREGV+RANGEを設定,次のB[ 60:119]を1回のburstで供給(MC-broadcast)                                       */
          /* ---------------------------------- 2重ループIMAX開始 --3重loopのblk-iteration-- 2回目 (blk=1)   B[0]とB[60]の距離は480*60*4B(128KB),LMM共存無理                                   */
          /* A[0][ 60:119]->LMMを再利用                                *//*   row= 0: B[ 60][*]:480*4B=2KB /LMM(1/2) b00+=(H-1)*M*4B  *//*  row=0: C[0][*]:480*4B=2KB / LMM (update)           */
          /* A[1][ 60:119]->LMMを再利用                                *//*   row= 1: B[ 61][*]:480*4B=2KB /LMM(1/2) b01+=(H-1)*M*4B  *//*  row=1: C[1][*]:480*4B=2KB / LMM (update)           */
          /* A[7][ 60:119]->LMMを再利用                                *//*   row=59: B[119][*]:480*4B=2KB /LMM(1/2) b59+=(H-1)*M*4B  *//*  row=7: C[7][*]:480*4B=2KB / LMM (update)           */
          /* ---------------------------------- IMAX動作一旦終了 ----------------------------------------------------------------------------------------------------------------------------- */
          /*                                                                        ★BのREGV+RANGEを設定,次のB[420:479]を1回のburstで供給(MC-broadcast)                                       */
          /* ---------------------------------- 2重ループIMAX開始 --3重loopのblk-iteration-- 8回目 (blk=7)                            *//*                                                     */
          /* A[0][420:479]->LMMを再利用                                *//*   row= 0: B[420][*]:480*4B=2KB /LMM(1/2) b00+=(H-1)*M*4B  *//*  row=0: C[0][*]:480*4B=2KB / LMM (update)           */
          /* A[1][420:479]->LMMを再利用                                *//*   row= 1: B[421][*]:480*4B=2KB /LMM(1/2) b01+=(H-1)*M*4B  *//*  row=1: C[1][*]:480*4B=2KB / LMM (update)           */
          /* A[7][420:479]->LMMを再利用                                *//*   row=59: B[479][*]:480*4B=2KB /LMM(1/2) b59+=(H-1)*M*4B  *//*  row=7: C[7][*]:480*4B=2KB / LMM (update)           */
          /* ---------------------------------- IMAX動作一旦終了 ----------------------------------------------------------------------------------------------------------------------------- */
          /* ================================== 3重ループ全体終了 ============================================================================================================================ */
          /* ★A[8:15][0:479]を一度に供給                                                                                                    ★RANGE設定,C[0:7][*]READ+C[8:15][*]WRITE         */
          /* ================================== 3重ループ先頭開始 BLK=0======================================================================================================================= */
          /*                                                                        ★BのREGV+RANGEを設定,次のB[  0: 59]を1回のburstで供給(MC-broadcast)                                       */
          /* ---------------------------------- 2重ループIMAX開始 --3重loopのblk-iteration-- 1回目 (blk=0)                            *//*                                                     */
          /* A[ 8][  0: 59]->LMMを再利用                               *//*   row= 0: B[  0][*]:480*4B=2KB /LMM(1/2) b00=B+(blk+ 0)*M *//*  row=0: C[ 8][*]:480*4B=2KB / LMM┐                 */
          /* A[ 9][  0: 59]->LMMを再利用                               *//*   row= 1: B[  1][*]:480*4B=2KB /LMM(1/2) b01=B+(blk+ 1)*M *//*  row=1: C[ 9][*]:480*4B=2KB / LMM│合計16KBは実際は */
          /* A[15][  0: 59]->LMMを再利用                               *//*   row=59: B[ 59][*]:480*4B=2KB /LMM(1/2) b59=B+(blk+59)*M *//*  row=7: C[15][*]:480*4B=2KB / LMM┘1LMMに収容可能   */

    /*1*/ for (col=0; col<M2; col+=W) { /* one-horizontal-line is calculated by EMAX-while(loop--) */
                                        /* C0xの部分和を生成（1行分）1chip分の総量はMword*M/#chip  */
                                        /*                          M=504の場合は64Kword(256KB)    */
                                        /*      さらにchip内でも行を分割すればcsimLMM(128KB)に入る */
            for (w=0; w<W; w++) {   /* horizontal (parallel) execution */
              for (h=0; h<H; h++) { /* vertical (pipelined) execution */
                if (blk == 0 && h == 0)
                  *(float*)&C1[(CHIP*M1/NCHIP+top+rofs)*M2+col+w]  = *(float*)&A[(CHIP*M1/NCHIP+top+rofs)*L+blk+h]**(float*)&B[(blk+h)*M2+col+w];
                else
                  *(float*)&C1[(CHIP*M1/NCHIP+top+rofs)*M2+col+w] += *(float*)&A[(CHIP*M1/NCHIP+top+rofs)*L+blk+h]**(float*)&B[(blk+h)*M2+col+w];
                count1++;
                /*printf("[%d %d %d %d %d %d %d]", CHIP, top, rofs, blk, col, w, h);*/
              }
            }
            /*printf("\n");*/
          }
        }
      }
    }
  }
}

#else

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
  /*  ┌─────┐convolutionの場合                                                  */
  /*  │┌────┴┐Bが複数と考える                                                  */
  /*  ││┌────┴┐┌─────┐┐        ┌─────┐┐                       */
  /*  │││b         ││a a a a a ││RMGRP   │o o o o o ││RMGRP                  */
  /*  │││b         ┤│          │┤/CHIP   │          │┤/CHIP                  */
  /*  │││b   B0   b││ A(weight)││        │   out    ││ mmの場合は行で分割    */
  /*  └││b        l┤│          │┤        │          │┤ cnnの場合はoutで分割  */
  /*    └│b        k││blk       ││        │blk       ││                       */
  /*      └─────┘└─┴─┴─┘┘        └─┴─┴─┘┘                       */
  printf("<<<IMAX>>>\n");
  for (top=0; top<M1/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    for (blk=0; blk<L; blk+=H) { /* 3重ループ展開の外側対象 */
      typedef struct {Uint i[8]} Ui8;
      Uint *a0[NCHIP];
      Uint *a[H][NCHIP];
      Ui8  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui8  *c0[NCHIP];
      Ui8  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
      for (k=0; k<H; k++) {
	b[k] = B+(blk+k)*M2; b0[k] = b[k]; b1[k] = (Uint*)b[k]+2; b2[k] = (Uint*)b[k]+4;  b3[k] = (Uint*)b[k]+6; 
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	a0[CHIP] = A+(CHIP*M1/NCHIP+top)*L;
	for (k=0; k<H; k++)
	  a[k][CHIP] = a0[CHIP]+blk+k;
	c0[CHIP] = C1+(CHIP*M1/NCHIP+top)*M2;
	c00[CHIP]= (Uint*)c0[CHIP]+0; c01[CHIP]= (Uint*)c0[CHIP]+2; c02[CHIP]= (Uint*)c0[CHIP]+4; c03[CHIP]= (Uint*)c0[CHIP]+6;
      }

#define sgemm00_core1(r, rm1, rp1) \
	    mop(OP_LDR,  3, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], M2, 0, 0, (Ull)NULL, M2);\
	    mop(OP_LDR,  3, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], M2, 0, 0, (Ull)NULL, M2);\
	    mop(OP_LDR,  3, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], M2, 0, 0, (Ull)NULL, M2);\
	    mop(OP_LDR,  3, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], M2, 0, 0, (Ull)NULL, M2);\
	    mop(OP_LDWR, 1, &BR[r][2][1],  (Ull)a[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], L*RMGRP, 0, 0, (Ull)NULL, L*RMGRP);\
	    exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H1010, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H1010, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H1010, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H1010, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_final(r, rp1) \
	    mop(OP_LDR,  3, &BR[rp1][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
	    mop(OP_LDR,  3, &BR[rp1][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
	    mop(OP_LDR,  3, &BR[rp1][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
	    mop(OP_LDR,  3, &BR[rp1][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
	    exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_STR,  3, &AR[rp1][0],     (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
	    mop(OP_STR,  3, &AR[rp1][1],     (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
	    mop(OP_STR,  3, &AR[rp1][2],     (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP);\
	    mop(OP_STR,  3, &AR[rp1][3],     (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], M2*RMGRP, 0, 1, (Ull)NULL, M2*RMGRP)

//EMAX5A begin mm mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-L*4)<<32|((0-M2*4)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=M2/W/2,cofs=(0-W*8)<<32|((0-W*8)&0xffffffff); LOOP0--; INIT0=0) {      /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*8)<<32|(W*8), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
            exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?(L*4)<<32|(M2*4):0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#0 */
            exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);            /* stage#1 */

            mop(OP_LDR,  3, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], M2, 0, 0, (Ull)NULL, M2);             /* stage#1 */
            mop(OP_LDR,  3, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], M2, 0, 0, (Ull)NULL, M2);             /* stage#1 */
            mop(OP_LDR,  3, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], M2, 0, 0, (Ull)NULL, M2);             /* stage#1 */
            mop(OP_LDR,  3, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], M2, 0, 0, (Ull)NULL, M2);             /* stage#1 2KB */
            mop(OP_LDWR, 1, &BR[1][2][1],  (Ull)a[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], L*RMGRP, 0, 0, (Ull)NULL, L*RMGRP); /* stage#1 16KB */
            exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
            exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
            exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
            exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */

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
	    sgemm00_core1(12, 11, 13); /* H=12 */
	    /****final*****/
	    sgemm00_final(13,     14);
          }
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
}
#endif
