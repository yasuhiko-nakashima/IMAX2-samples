#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#endif

#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"

Uchar *membase;
typedef struct {
  Uint idx;
  Uint val;}
Data;

Data *In;
Data *Out;
Data *pseudoLMM;

sysinit(Uint memsize, Uint alignment)
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

unsigned NumberOfBitsNeeded(unsigned PowerOfTwo)
{
  unsigned i;

  if (PowerOfTwo < 2) {
    fprintf(stderr, ">>> Error: argument %d to NumberOfBitsNeeded is too small.\n", PowerOfTwo);
    exit(1);
  }
  
  for (i=0; ; i++) {
    if (PowerOfTwo&(1<<i))
      return (i);
  }
}

int main(int argc, char *argv[]) {
  unsigned MAXSIZE;
  unsigned i,j;

  if (argc<2) {
    printf("Usage: msort <size>\n");
    exit(-1);
  }
  MAXSIZE=atoi(argv[1]);
  
  sysinit((MAXSIZE *sizeof(Data)
	  +MAXSIZE *sizeof(Data)
	  +MAXSIZE *sizeof(Data)*NumberOfBitsNeeded(MAXSIZE)*2), 32);
  printf("membase: %08.8x_%08.8x\n", (Uint)((Ull)membase>>32), (Uint)membase);
  In        = (Data*)membase;
  Out       = (Data*)((Uchar*)In  + MAXSIZE*sizeof(Data));
  pseudoLMM = (Data*)((Uchar*)Out + MAXSIZE*sizeof(Data)); /* for IMAX2 */

  printf("In:   %08.8x\n",     (Uint)In);
  printf("Out:  %08.8x\n",     (Uint)Out);
  printf("pseudoLMM:%08.8x-%08.8x\n", (Uint)pseudoLMM, (Uint)pseudoLMM+MAXSIZE*sizeof(Data)*NumberOfBitsNeeded(MAXSIZE)*2-1);

  /* ----- Initialize ----- */ 
  for (i=0; i<MAXSIZE; i++) {
    In[i].val = (i^0x0055)&0xffff;
    In[i].idx = i;
  }

  printf("In: ");
  for (i=0;i<MAXSIZE;i++)
    printf(" %d.%d", In[i].val, In[i].idx);
  printf("\n");

//sort_quick(In, Out, 0, MAXSIZE-1);
  sort_merge(In, Out, 0, MAXSIZE-1);
    
  printf("Out:");
  for (i=0;i<MAXSIZE;i++)
    printf(" %d.%d", Out[i].val, Out[i].idx);
  printf("\n");
  
  show_nanosec();

  exit(0);
}

sort_quick(Data *In, Data *Out, int lo, int hi)
{
  int i = lo, j = hi, ref;
  Data tp;

  ref = In[(i+j)/2].val;           /* 中間位置の面積を基準値とする */
  while (i <= j) {
    while ((i<hi) && In[i].val<ref)/* 基準値以上の最初の県 */
      i++;
    while ((j>lo) && In[j].val>ref)/* 基準値以下の最初の県 */
      j--; 
    if (i<=j) {                              /* 交換するか */
      tp = In[i]; In[i] = In[j]; In[j] = tp; /* ポインタの交換 */
      i++;
      j--;
    }
  }
  if (lo < j) sort_quick(In, Out, lo, j);    /* 上半分を整列 */
  if (i < hi) sort_quick(In, Out, i, hi);    /* 下半分を整列 */
  for (i=lo; i<=hi; i++)
    Out[i] = In[i];
}

sort_merge(Data *In, Data *Out, int lo, int hi)
{
  Uint NumSamples, NumSamples2, NumBits, BlockEnd, BlockSize;
  Ull  i, j, k, n, t;

  NumSamples  = hi-lo+1;
  NumSamples2 = NumSamples*2;
  NumBits = NumberOfBitsNeeded(NumSamples);
#if 0
  if ((n = ((hi-lo+1))/2) >= 2) {
    sort_merge(In, Out, lo, lo+(n-1)); /* 上半分を整列 */
    sort_merge(In, Out, hi-(n-1), hi); /* 下半分を整列 */
  }
  for (i=lo,j=hi-(n-1),t=lo; t<=hi;) {
    if (i<=lo+n-1 && j<=hi) {
      if (In[i].val < In[j].val)
	Out[t++] = In[i++];
      else 
	Out[t++] = In[j++];
    }
    else if (i<=lo+n-1)
      Out[t++] = In[i++];
    else if (j<=hi)
      Out[t++] = In[j++];
  }
  for (i=lo; i<=hi; i++)
    In[i] = Out[i];
#elif !defined(EMAX6)
  printf("<<<ORIG>>>\n");
  reset_nanosec();
  BlockEnd = 1;
  for (BlockSize=2; BlockSize<=NumSamples; BlockSize<<=1) {
    for (i=0; i<NumSamples; i+=BlockSize) {
      for (j=i,k=i+BlockEnd,t=i; t<i+BlockSize; t++) {
	int cc0 = j<i+BlockEnd;
	int cc1 = k<i+BlockSize;
	int cc2 = In[j].val < In[k].val;
	if ((( cc2 && cc1) || !cc1) && cc0) /* 7,5,1 0x00a2 */
	  Out[t] = In[j++];
	if (((!cc2 && cc0) || !cc0) && cc1) /* 6,3,2 0x004c */
	  Out[t] = In[k++];
      }
    }
    BlockEnd = BlockSize;
    for (i=0; i<NumSamples; i++)
      In[i] = Out[i];
  }
#else
#undef  NCHIP
#undef  RMGRP
#undef  W
#undef  H
#define NCHIP 1
#define RMGRP 2
//#define W     4LL
//#define H     16
#define H     4096
  Ull  CHIP;
  Ull  LOOP1, LOOP0, L8;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];           /* output of EX     in each unit */
  Ull  BR[64][4][4];        /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1, ex2, ex3;
  Ull  Buf[32];
  Ull  base, J[16], K[16];  /* log2(NumSamples=65536)=16まで対応可 */
  Ull  Pipeline, Lmmrotate; /* log2(NumSamples=65536)=16回繰り返すと,最終段のLMMに,最初のOutが格納される */

  printf("<<<IMAX>>> NumSamples=%d (LMM should be >= %dB)\n", NumSamples, NumSamples2*8);
  reset_nanosec();
  for (i=0; i<NumBits; i++) {
    J[i]=0;
    K[i]=0;
  }
#define sort_core0(r, rp1, x, MASK_M, In, BE8) \
        exe(OP_NOP,       &AR[r][1],    0LL,            EXP_H3210,  0LL,         EXP_H3210,  0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#1 dmy              */\
        exe(OP_ADD,       &i,           L8,             EXP_H3210,  0LL,         EXP_H3210,  0LL,          EXP_H3210, OP_AND, MASK_M, OP_NOP, 0LL); /* stage#1 i     = L8&M     */\
        exe(OP_NOP,       &AR[rp1][3],  0LL,            EXP_H3210,  0LL,         EXP_H3210,  0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#2 dmy              */\
        exe(OP_ADD,       &base,        In,             EXP_H3210,  i,           EXP_H3210,  0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL)  /* stage#2 baseJ = &In[i], baseK = &In[i+BE] */

#define sort_core1(r, rp1, x, In, Ilen, BE8, Buf, Out, Olen) \
        mex(OP_CMPA_LE, &J[x], INIT0?0LL:J[x], INIT0?0LL:8LL, OP_CMPA_GE, &K[x], INIT0?BE8:K[x], INIT0?0LL:8LL, BE8, BR[r][3][1], BR[r][3][0]);/*|yx|B|prefixとして使用.col2[1]配置確定.LMM[2]配置確定.bit63-32を比較*//* stage#3 */\
        mop(OP_LDR,  3,   &BR[r][3][1], J[x],           base,       MSK_D0,      In,         Ilen,         0, 0,      NULL,   Ilen);/*LMM[2]確定   LD実行はcol2*//* stage#3 */\
        mop(OP_LDR,  3,   &BR[r][3][0], K[x],           base,       MSK_D0,      In,         Ilen,         0, 0,      NULL,   Ilen);/*LMM[1]間借り LD実行はcol2*//* stage#3 */\
      /*exe(OP_NOP,       &AR[rp1][1],  0LL,            EXP_H3210,  0LL,         EXP_H3210,  0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL);*/            /* stage#4 */\
        exe(OP_CMP_LT,    &cc0,         J[x],           EXP_H1010,  BE8,         EXP_H1010,  0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* J[x]<BE8 */     /* stage#4 */\
        exe(OP_CMP_LT,    &cc1,         K[x],           EXP_H1010,  BE8*2,       EXP_H1010,  0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* K[x]<BE8 */     /* stage#4 */\
        exe(OP_CMP_LT,    &cc2,         BR[r][3][1],    EXP_H3232,  BR[r][3][0], EXP_H3232,  0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);/* *J<*K    */     /* stage#4 */\
        cex(OP_CEXE,      &ex0,         0, cc2, cc1, cc0, 0x00a2);                                 /* if (( cc2 && cc1 && cc0) || (!cc1 &&  cc0)) 7,5,1 0x00a2 *//* stage#5 */\
        mop(OP_STR,  ex0, &BR[r][3][1], Buf,            L8,         MSK_W0,      Out,        Olen,         0, 0,      NULL,   Olen);/*LMM[1]間借り LD実行はcol2*//* stage#5 */\
        cex(OP_CEXE,      &ex1,         0, cc2, cc1, cc0, 0x004c);                                 /* if ((!cc2 && cc1 && cc0) || ( cc1 && !cc0)) 6,3,2 0x004c *//* stage#5 */\
        mop(OP_STR,  ex1, &BR[r][3][0], Buf,            L8,         MSK_W0,      Out,        Olen,         0, 0,      NULL,   Olen) /*LMM[1]間借り LD実行はcol2*//* stage#5 */

  for (Pipeline=0; Pipeline<NumBits; Pipeline++) {
    /* 0: buf[0]=[(0+4-0)%4]:0 buf[1]=[(1+4-0)%4]:1 buf[2]=[(2+4-0)%4]:2 buf[3]=[(3+4-0)%4]:3 */
    /* 1: buf[0]=[(0+4-1)%4]:3 buf[1]=[(1+4-1)%4]:0 buf[2]=[(2+4-1)%4]:1 buf[3]=[(3+4-1)%4]:2 */
    /* 2: buf[0]=[(0+4-2)%4]:2 buf[1]=[(1+4-2)%4]:3 buf[2]=[(2+4-2)%4]:0 buf[3]=[(3+4-2)%4]:1 */
    for (Lmmrotate=0; Lmmrotate<NumBits*2; Lmmrotate++)
      Buf[Lmmrotate] = &pseudoLMM[NumSamples*((Lmmrotate+NumBits*2-Pipeline)%(NumBits*2))];
#if 0
    if (Pipeline>0) {
      printf("Buf%d", (Uint)Pipeline*2-1);
      for (i=0;i<NumSamples;i++)
	printf(" %d.%d", ((Data*)Buf[Pipeline*2-1]+i)->val, ((Data*)Buf[Pipeline*2-1]+i)->idx);
      printf("\n");
    }
#endif
//EMAX5A begin pipeline mapdist=0
/*3*/for (CHIP=0; CHIP<NCHIP; CHIP++) {
 /*1*/for (INIT0=1,LOOP0=NumSamples,L8=0LL<<32|(0-8LL)&0xffffffff; LOOP0--; INIT0=0) { /* NumSamples<=4096 */
        exe(OP_ADD, &L8, L8, EXP_H3210, 0LL<<32|8LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#0 */
#if (H==2)
        sort_core0( 1,  2, 0, 0xfffffffffffffff0LL, In,      8LL);                                   /* stage#1-2   */
        sort_core1( 3,  4, 0, In,       NumSamples2,         8LL,     Out,    Out,     NumSamples2); /* stage#3-5   */
	//if (Pipeline==0) printf("i=%d ja=%08.8x J=%d:%d.%d ka=%08.8x K=%d:%d.%d Out=%d\n", (Uint)i, (Uint)(J[0]+base), (Uint)J[0], (Uint)(BR[3][2][1]>>32), (Uint)BR[3][2][1], (Uint)(K[0]+base), (Uint)K[0], (Uint)(BR[3][2][0]>>32), (Uint)BR[3][2][0], Out[L8/8].val);
#endif
#if (H==4)
        sort_core0( 1,  2, 0, 0xfffffffffffffff0LL, In,      8LL);                                   /* stage#1-2   */
        sort_core1( 3,  4, 0, In,       NumSamples2,         8LL,     Buf[0], Buf[1],  0LL);         /* stage#3-5   */
	//if (Pipeline==0) printf("i=%d ja=%08.8x J=%d:%d.%d ka=%08.8x K=%d:%d.%d Out=%d\n", (Uint)i, (Uint)(J[0]+base), (Uint)J[0], (Uint)(BR[3][2][1]>>32), (Uint)BR[3][2][1], (Uint)(K[0]+base), (Uint)K[0], (Uint)(BR[3][2][0]>>32), (Uint)BR[3][2][0], ((Data*)(Buf[0]+L8))->val);
        sort_core0( 3,  4, 1, 0xffffffffffffffe0LL, Buf[1],  16LL);                                  /* stage#3-4   */
        sort_core1( 5,  6, 1, Buf[1],   0LL,                 16LL,    Out,    Out,     NumSamples2); /* stage#5-7   */
	//if (Pipeline==1) printf("i=%d ja=%08.8x J=%d:%d.%d ka=%08.8x K=%d:%d.%d Out=%d\n", (Uint)i, (Uint)(J[1]+base), (Uint)J[1], (Uint)(BR[6][2][1]>>32), (Uint)BR[6][2][1], (Uint)(K[1]+base), (Uint)K[1], (Uint)(BR[6][2][0]>>32), (Uint)BR[6][2][0], Out[L8/8].val);
#endif
#if (H==8)
        sort_core0( 1,  2, 0, 0xfffffffffffffff0LL, In,      8LL);                                   /* stage#1-2   */
        sort_core1( 3,  4, 0, In,       NumSamples2,         8LL,     Buf[0], Buf[1],  0LL);         /* stage#3-5   */
        sort_core0( 3,  4, 1, 0xffe0LL, 0x0003LL, Buf[1],    16LL);                                  /* stage#3-4   */
        sort_core1( 5,  6, 1, Buf[1],   0LL,                 16LL,    Buf[2], Buf[3],  0LL);         /* stage#5-7   */
        sort_core0( 5,  6, 2, 0xffc0LL, 0x0007LL, Buf[3],    32LL);                                  /* stage#5-6   */
        sort_core1( 7,  8, 2, Buf[3],   0LL,                 32LL,    Out,    Out,     NumSamples2); /* stage#7-9   */
#endif
#if (H==16)
        sort_core0( 1,  2, 0, 0xfffffffffffffff0LL, In,      8LL);                                   /* stage#1-2   */
        sort_core1( 3,  4, 0, In,       NumSamples2,         8LL,     Buf[0], Buf[1],  0LL);         /* stage#3-5   */
	//if (Pipeline==0) printf("i=%d ja=%08.8x J=%d:%d.%d ka=%08.8x K=%d:%d.%d Out=%d\n", (Uint)i, (Uint)(J[0]+base), (Uint)J[0], (Uint)(BR[3][2][1]>>32), (Uint)BR[3][2][1], (Uint)(K[0]+base), (Uint)K[0], (Uint)(BR[3][2][0]>>32), (Uint)BR[3][2][0], ((Data*)(Buf[0]+L8))->val);
        sort_core0( 3,  4, 1, 0xffffffffffffffe0LL, Buf[1],  16LL);                                  /* stage#3-4   */
        sort_core1( 5,  6, 1, Buf[1],   0LL,                 16LL,    Buf[2], Buf[3],  0LL);         /* stage#5-7   */
	//if (Pipeline==1) printf("i=%d ja=%08.8x J=%d:%d.%d ka=%08.8x K=%d:%d.%d Out=%d\n", (Uint)i, (Uint)(J[1]+base), (Uint)J[1], (Uint)(BR[5][2][1]>>32), (Uint)BR[5][2][1], (Uint)(K[1]+base), (Uint)K[1], (Uint)(BR[5][6][0]>>32), (Uint)BR[5][2][0], ((Data*)(Buf[2]+L8))->val);
        sort_core0( 5,  6, 2, 0xffffffffffffffc0LL, Buf[3],  32LL);                                  /* stage#5-6   */
        sort_core1( 7,  8, 2, Buf[3],   0LL,                 32LL,    Buf[4], Buf[5],  0LL);         /* stage#7-9   */
	//if (Pipeline==2) printf("i=%d ja=%08.8x J=%d:%d.%d ka=%08.8x K=%d:%d.%d Out=%d\n", (Uint)i, (Uint)(J[2]+base), (Uint)J[2], (Uint)(BR[7][2][1]>>32), (Uint)BR[7][2][1], (Uint)(K[2]+base), (Uint)K[2], (Uint)(BR[7][2][0]>>32), (Uint)BR[7][2][0], ((Data*)(Buf[4]+L8))->val);
        sort_core0( 7,  8, 3, 0xffffffffffffff80LL, Buf[5],  64LL);                                  /* stage#7-8   */
        sort_core1( 9, 10, 3, Buf[5],   0LL,                 64LL,    Out,    Out,     NumSamples2); /* stage#9-11  */
	//if (Pipeline==3) printf("i=%d ja=%08.8x J=%d:%d.%d ka=%08.8x K=%d:%d.%d Out=%d\n", (Uint)i, (Uint)(J[3]+base), (Uint)J[3], (Uint)(BR[9][2][1]>>32), (Uint)BR[9][2][1], (Uint)(K[3]+base), (Uint)K[3], (Uint)(BR[9][2][0]>>32), (Uint)BR[9][2][0], Out[L8/8].val);
#endif
#if (H==256)
        sort_core0( 1,  2, 0, 0xfffffffffffffff0LL, In,      8LL);                                   /* stage#1-2   */
        sort_core1( 3,  4, 0, In,       NumSamples2,         8LL,     Buf[0],  Buf[1], 0LL);         /* stage#3-5   */
        sort_core0( 3,  4, 1, 0xffffffffffffffe0LL, Buf[1],  16LL);                                  /* stage#3-4   */
        sort_core1( 5,  6, 1, Buf[1],   0LL,                 16LL,    Buf[2],  Buf[3], 0LL);         /* stage#5-7   */
        sort_core0( 5,  6, 2, 0xffffffffffffffc0LL, Buf[3],  32LL);                                  /* stage#5-6   */
        sort_core1( 7,  8, 2, Buf[3],   0LL,                 32LL,    Buf[4],  Buf[5], 0LL);         /* stage#7-9   */
        sort_core0( 7,  8, 3, 0xffffffffffffff80LL, Buf[5],  64LL);                                  /* stage#7-8   */
        sort_core1( 9, 10, 3, Buf[5],   0LL,                 64LL,    Buf[6],  Buf[7], 0LL);         /* stage#9-11  */
        sort_core0( 9, 10, 4, 0xffffffffffffff00LL, Buf[7],  128LL);                                 /* stage#9-10  */
        sort_core1(11, 12, 4, Buf[7],   0LL,                 128LL,   Buf[8],  Buf[9], 0LL);         /* stage#11-13 */
        sort_core0(11, 12, 5, 0xfffffffffffffe00LL, Buf[9],  256LL);                                 /* stage#11-12 */
        sort_core1(13, 14, 5, Buf[9],   0LL,                 256LL,   Buf[10], Buf[11],0LL);         /* stage#13-15 */
        sort_core0(13, 14, 6, 0xfffffffffffffc00LL, Buf[11], 512LL);                                 /* stage#13-14 */
        sort_core1(15, 16, 6, Buf[11],  0LL,                 512LL,   Buf[12], Buf[13],0LL);         /* stage#15-17 */
        sort_core0(15, 16, 7, 0xfffffffffffff800LL, Buf[13], 1024LL);                                /* stage#15-16 */
        sort_core1(17, 18, 7, Buf[13],  0LL,                 1024LL,  Out,     Out,    NumSamples2); /* stage#17-19 */
#endif
#if (H==4096)
        sort_core0( 1,  2, 0, 0xfffffffffffffff0LL, In,      8LL);                                   /* stage#1-2   */
        sort_core1( 3,  4, 0, In,       NumSamples2,         8LL,     Buf[0],  Buf[1], 0LL);         /* stage#3-5   */
        sort_core0( 3,  4, 1, 0xffffffffffffffe0LL, Buf[1],  16LL);                                  /* stage#3-4   */
        sort_core1( 5,  6, 1, Buf[1],   0LL,                 16LL,    Buf[2],  Buf[3], 0LL);         /* stage#5-7   */
        sort_core0( 5,  6, 2, 0xffffffffffffffc0LL, Buf[3],  32LL);                                  /* stage#5-6   */
        sort_core1( 7,  8, 2, Buf[3],   0LL,                 32LL,    Buf[4],  Buf[5], 0LL);         /* stage#7-9   */
        sort_core0( 7,  8, 3, 0xffffffffffffff80LL, Buf[5],  64LL);                                  /* stage#7-8   */
        sort_core1( 9, 10, 3, Buf[5],   0LL,                 64LL,    Buf[6],  Buf[7], 0LL);         /* stage#9-11  */
        sort_core0( 9, 10, 4, 0xffffffffffffff00LL, Buf[7],  128LL);                                 /* stage#9-10  */
        sort_core1(11, 12, 4, Buf[7],   0LL,                 128LL,   Buf[8],  Buf[9], 0LL);         /* stage#11-13 */
        sort_core0(11, 12, 5, 0xfffffffffffffe00LL, Buf[9],  256LL);                                 /* stage#11-12 */
        sort_core1(13, 14, 5, Buf[9],   0LL,                 256LL,   Buf[10], Buf[11],0LL);         /* stage#13-15 */
        sort_core0(13, 14, 6, 0xfffffffffffffc00LL, Buf[11], 512LL);                                 /* stage#13-14 */
        sort_core1(15, 16, 6, Buf[11],  0LL,                 512LL,   Buf[12], Buf[13],0LL);         /* stage#15-17 */
        sort_core0(15, 16, 7, 0xfffffffffffff800LL, Buf[13], 1024LL);                                /* stage#15-16 */
        sort_core1(17, 18, 7, Buf[13],  0LL,                 1024LL,  Buf[14], Buf[15],0LL);         /* stage#17-19 */
        sort_core0(17, 18, 8, 0xfffffffffffff000LL, Buf[15], 2048LL);                                /* stage#17-18 */
        sort_core1(19, 20, 8, Buf[15],  0LL,                 2048LL,  Buf[16], Buf[17],0LL);         /* stage#19-21 */
        sort_core0(19, 20, 9, 0xffffffffffffe000LL, Buf[17], 4096LL);                                /* stage#19-20 */
        sort_core1(21, 22, 9, Buf[17],  0LL,                 4096LL,  Buf[18], Buf[19],0LL);         /* stage#21-23 */
        sort_core0(21, 22,10, 0xffffffffffffc000LL, Buf[19], 8192LL);                                /* stage#21-22 */
        sort_core1(23, 24,10, Buf[19],  0LL,                 8192LL,  Buf[20], Buf[21],0LL);         /* stage#23-25 */
        sort_core0(23, 24,11, 0xffffffffffff8000LL, Buf[21], 16384LL);                               /* stage#23-24 */
        sort_core1(25, 26,11, Buf[21],  0LL,                 16384LL, Out,     Out,    NumSamples2); /* stage#25-27 */
#endif
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm
#endif
  get_nanosec(0);
}
