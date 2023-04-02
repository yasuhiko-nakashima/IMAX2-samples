/*============================================================================

    fourierf.c  -  Don Cross <dcross@intersrv.com>

    http://www.intersrv.com/~dcross/fft.html

    Contains definitions for doing Fourier transforms
    and inverse Fourier transforms.

    This module performs operations on arrays of 'float'.

    Revision history:

1998 September 19 [Don Cross]
    Updated coding standards.
    Improved efficiency of trig calculations.

============================================================================*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "fourier.h"
#include "ddcmath.h"

#define NO_EMAX6LIB_BODY
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"

void fft_float(unsigned  NumSamples,
	       int       InverseTransform,
	       float    *RealIn,
	       float    *ImagIn,
	       float    *RealOut,
	       float    *ImagOut)
{
  unsigned NumBits;    /* Number of bits needed to store indices */
  Ull i, j, k, n, idx;
  unsigned BlockSize, BlockEnd;

  float angle_numerator = 2.0 * DDC_PI;
  float tr, ti;     /* temp real, temp imaginary */

  if (!IsPowerOfTwo(NumSamples)) {
    fprintf(stderr, "Error in fft():  NumSamples=%u is not power of two\n", NumSamples);
    exit(1);
  }

  if (InverseTransform)
    angle_numerator = -angle_numerator;

  NumBits = NumberOfBitsNeeded(NumSamples);

  /*
  **   Do simultaneous data copy and bit-reversal ordering into outputs...
  */

  for (i=0; i<NumSamples; i++) {
    j = ReverseBits(i, NumBits);
    RealOut[j] = RealIn[i];
    ImagOut[j] = ImagIn[i];
  }
  /* RealIn should be input for pipelined IMAX2 */
  for (i=0; i<NumSamples; i++) {
    RealIn[i] = RealOut[i];
    ImagIn[i] = ImagOut[i];
  }

  /* Initialize art/ait */

  BlockEnd = 1;
  for (BlockSize=2; BlockSize<=NumSamples; BlockSize<<=1) {
    float delta_angle = angle_numerator / (float)BlockSize;
    float sm2 = sinf( -2 * delta_angle );
    float sm1 = sinf( -delta_angle );
    float cm2 = cosf( -2 * delta_angle );
    float cm1 = cosf( -delta_angle );
    float w = 2 * cm1;
    float ar0, ar1, ar2, ai0, ai1, ai2;
    for (i=0; i<NumSamples; i+=BlockSize) {
      ar2 = cm2;
      ai2 = sm2;
      ar1 = cm1;
      ai1 = sm1;
      for (j=i,n=0; n<BlockEnd; j++,n++) {
	ar0      = w*ar1 - ar2;
	ai0      = w*ai1 - ai2;
	ar2      = ar1;
	ai2      = ai1;
	ar1      = ar0;
	ai1      = ai0;
	k   = j + BlockEnd;
	idx = n + BlockEnd;
	art[idx] = ar0;
	ait[idx] = ai0;
      }
    }
    BlockEnd = BlockSize;
  }

  /*
  **   Do the FFT itself...
  */

#if !defined(EMAX6)
  printf("<<<ORIG>>>\n");
  reset_nanosec();
  BlockEnd = 1;
  for (BlockSize=2; BlockSize<=NumSamples; BlockSize<<=1) {
    for (i=0; i<NumSamples; i+=BlockSize) {
      for (j=i,n=0; n<BlockEnd; j++,n++) {
	k   = j + BlockEnd;
	idx = n + BlockEnd;
	tr = art[idx]*RealOut[k] - ait[idx]*ImagOut[k];
	ti = art[idx]*ImagOut[k] + ait[idx]*RealOut[k];
	RealOut[k] = RealOut[j] - tr;
	ImagOut[k] = ImagOut[j] - ti;
	RealOut[j] += tr;
	ImagOut[j] += ti;
	//printf("j=%d %7.1f %7.1f k=%d %7.1f %7.1f\n", (Uint)j, RealOut[j], ImagOut[j], (Uint)k, RealOut[k], ImagOut[k]);
      }
    }
    BlockEnd = BlockSize;
  }
#elif 0
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
  Ull  ar, ai, rok, iok, roj, ioj;
  Ull  tr0, ti0, tr1, ti1, roj1, ioj1;

  printf("<<<IMAX>>> NumSamples=%d (LMM should be >= %dB)\n", NumSamples, NumSamples*4*2);
  reset_nanosec();

#define fft_core0(r) \
          exe(OP_ADD,     &j,         i,        EXP_H3210,  n,       EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_SLL, 2LL); /* stage#1 */\
          exe(OP_ADD3,    &k,         i,        EXP_H3210,  n,       EXP_H3210, BlockEnd,     EXP_H3210, OP_NOP, 0LL, OP_SLL, 2LL); /* stage#1 */\
          exe(OP_ADD3,    &idx,       0LL,      EXP_H3210,  n,       EXP_H3210, BlockEnd,     EXP_H3210, OP_NOP, 0LL, OP_SLL, 2LL); /* stage#1 */\
	  exe(OP_NOP,     &AR[r][0],  0LL,      EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 [2][0]     */\
	  mop(OP_LDWR, 3, &rok,       RealOut,  k,          MSK_D0,  RealOut,   NumSamples,   0, 1,  NULL,  NumSamples);            /* stage#2 RealOut[k] */\
	  mop(OP_LDWR, 3, &roj,       j,        RealOut,    MSK_D0,  RealOut,   NumSamples,   0, 1,  NULL,  NumSamples);            /* stage#2 RealOut[j] */\
	  exe(OP_NOP,     &AR[r][2],  0LL,      EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 [2][2]     */\
	  mop(OP_LDWR, 3, &iok,       ImagOut,  k,          MSK_D0,  ImagOut,   NumSamples,   0, 1,  NULL,  NumSamples);            /* stage#2 ImagOut[k] */\
	  mop(OP_LDWR, 3, &ioj,       j,        ImagOut,    MSK_D0,  ImagOut,   NumSamples,   0, 1,  NULL,  NumSamples)             /* stage#2 ImagOut[j] */
#define fft_core1(r) \
	  exe(OP_NOP,     &AR[r][0],  0LL,      EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 [3][0]     */\
	  mop(OP_LDWR, 3, &ar,        art,      idx,        MSK_D0,  art,       NumSamples,   0, 0,  NULL,  NumSamples);            /* stage#3 art[idx]   */\
	  exe(OP_NOP,     &AR[r][2],  0LL,      EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 [3][2]     */\
	  mop(OP_LDWR, 3, &ai,        ait,      idx,        MSK_D0,  ait,       NumSamples,   0, 0,  NULL,  NumSamples);            /* stage#3 ait[idx]   */\
          \
	  exe(OP_FML,     &tr0,       ar,       EXP_H3210,  rok,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */\
	  exe(OP_FML,     &ti0,       ar,       EXP_H3210,  iok,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */\
	  exe(OP_FMS,     &tr1,       tr0,      EXP_H3210,  ai,      EXP_H3210, iok,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */\
	  exe(OP_FMA,     &ti1,       ti0,      EXP_H3210,  ai,      EXP_H3210, rok,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)  /* stage#5 */
#define fft_final(r) \
	  exe(OP_FMS,     &AR[r][0],  roj,      EXP_H3210,  tr1,     EXP_H3210, 0x3f800000LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */\
	  mop(OP_STWR, 3, &AR[r][0],  RealOut,  k,          MSK_D0,  RealOut,   NumSamples,   0, 0,  NULL,  NumSamples);            /* stage#6 */\
	  exe(OP_FAD,     &AR[r][1],  roj,      EXP_H3210,  tr1,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */\
	  mop(OP_STWR, 3, &AR[r][1],  RealOut,  j,          MSK_D0,  RealOut,   NumSamples,   0, 0,  NULL,  NumSamples);            /* stage#6 */\
	  exe(OP_FMS,     &AR[r][2],  ioj,      EXP_H3210,  ti1,     EXP_H3210, 0x3f800000LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */\
	  mop(OP_STWR, 3, &AR[r][2],  ImagOut,  k,          MSK_D0,  ImagOut,   NumSamples,   0, 0,  NULL,  NumSamples);            /* stage#6 */\
	  exe(OP_FAD,     &AR[r][3],  ioj,      EXP_H3210,  ti1,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */\
	  mop(OP_STWR, 3, &AR[r][3],  ImagOut,  j,          MSK_D0,  ImagOut,   NumSamples,   0, 0,  NULL,  NumSamples)             /* stage#6 */

  BlockEnd = 1;
  for (BlockSize=2; BlockSize<=NumSamples; BlockSize<<=1) {
//with-prefetch/post-drain
//EMAX5A begin imax mapdist=0
/*3*/for (CHIP=0; CHIP<NCHIP; CHIP++) {
  /*2*/for (INIT1=1,LOOP1=NumSamples/BlockSize,i=0LL<<32|(0-BlockSize)&0xffffffff; LOOP1--; INIT1=0) {
    /*1*/for (INIT0=1,LOOP0=BlockEnd,n=0LL<<32|(0-1LL)&0xffffffff; LOOP0--; INIT0=0) {
          exe(OP_ADD,     &i,     i,         EXP_H3210,   INIT0?BlockSize:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);          /* stage#0 */
          exe(OP_ADD,     &n,     INIT0?n:n, EXP_H3210,   0LL<<32|1LL,       EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#0 */
	  fft_core0(2); /* stage#2   */
	  fft_core1(3); /* stage#3-5 */
	  fft_final(6); /* stage#6   */
        }
      }
    }
//EMAX5A end
//EMAX5A drain_dirty_lmm
    BlockEnd = BlockSize;
  }
#else
#undef  NCHIP
#undef  RMGRP
#undef  W
#undef  H
#define NCHIP 1
#define RMGRP 2
#define W     4LL
//#define H     8
#define H     4096
  Ull  CHIP;
  Ull  LOOP1, LOOP0, L;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  J[16], K[16], IDX[16]; /* log2(NumSamples=65536)=16まで対応可 */
  Ull  BufReal[16], BufImag[16];
  Ull  ar, ai, rok, iok, roj, ioj, tr0, ti0, tr1, ti1;
  Ull  Pipeline, Lmmrotate; /* log2(NumSamples=65536)=16回繰り返すと,最終段のLMMに,最初のRealOut/ImagOutが格納される */

  printf("<<<IMAX>>> NumSamples=%d (LMM should be >= %dB)\n", NumSamples, NumSamples*4*2);
  reset_nanosec();

#define fft_core0(r, x, MASK_M, MASK_N, BS) \
        exe(OP_ADD,     &i,         L,          EXP_H3210,  L,       EXP_H3210, 0LL,          EXP_H3210, OP_AND, MASK_M, OP_NOP, 0LL); /* stage#1 i  =(L*2)&M   */\
        exe(OP_ADD,     &n,         L,          EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_AND, MASK_N, OP_NOP, 0LL); /* stage#1 n  =L    &N   */\
        exe(OP_ADD,     &J[x],      i,          EXP_H3210,  n,       EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_SLL, 2LL); /* stage#2 j  =i+n       */\
        exe(OP_ADD3,    &K[x],      i,          EXP_H3210,  n,       EXP_H3210, BS,           EXP_H3210, OP_NOP, 0LL,    OP_SLL, 2LL); /* stage#2 k  =i+n+BS(2) */\
        exe(OP_ADD3,    &IDX[x],    0LL,        EXP_H3210,  n,       EXP_H3210, BS,           EXP_H3210, OP_NOP, 0LL,    OP_SLL, 2LL)  /* stage#2 idx=  n+BS(2) */
#define fft_core1(r, x) \
	exe(OP_NOP,     &AR[r][0],  0LL,        EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#3 [3][0]        */\
	mop(OP_LDWR, 3, &rok,       RealIn,     K[x],       MSK_D0,  RealIn,    NumSamples,   0, 1,  NULL,  NumSamples);               /* stage#3 RealIn[k]     */\
	mop(OP_LDWR, 3, &roj,       J[x],       RealIn,     MSK_D0,  RealIn,    NumSamples,   0, 1,  NULL,  NumSamples);               /* stage#3 RealIn[j]     */\
	exe(OP_NOP,     &AR[r][2],  0LL,        EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#3 [3][2]        */\
	mop(OP_LDWR, 3, &iok,       ImagIn,     K[x],       MSK_D0,  ImagIn,    NumSamples,   0, 1,  NULL,  NumSamples);               /* stage#3 ImagIn[k]     */\
	mop(OP_LDWR, 3, &ioj,       J[x],       ImagIn,     MSK_D0,  ImagIn,    NumSamples,   0, 1,  NULL,  NumSamples)                /* stage#3 ImagIn[j]     */
#define fft_core2(r, x, y, MASK_M, MASK_N, BS) \
	exe(OP_NOP,     &AR[r][0],  0LL,        EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#4 [4][0]        */\
        exe(OP_ADD,     &i,         L,          EXP_H3210,  L,       EXP_H3210, 0LL,          EXP_H3210, OP_AND, MASK_M, OP_NOP, 0LL); /* stage#4 i  =(L*2)&M   */\
	mop(OP_LDWR, 3, &ar,        art,        IDX[x],     MSK_D0,  art,       NumSamples,   0, 0,  NULL,  NumSamples);               /* stage#4 art[idx]      */\
	exe(OP_NOP,     &AR[r][2],  0LL,        EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#4 [4][2]        */\
        exe(OP_ADD,     &n,         L,          EXP_H3210,  0LL,     EXP_H3210, 0LL,          EXP_H3210, OP_AND, MASK_N, OP_NOP, 0LL); /* stage#4 n  = L   &N   */\
	mop(OP_LDWR, 3, &ai,        ait,        IDX[x],     MSK_D0,  ait,       NumSamples,   0, 0,  NULL,  NumSamples);               /* stage#4 ait[idx]      */\
        \
        exe(OP_ADD,     &J[y],      i,          EXP_H3210,  n,       EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_SLL, 2LL); /* stage#5 j  =i+n       */\
        exe(OP_ADD3,    &K[y],      i,          EXP_H3210,  n,       EXP_H3210, BS,           EXP_H3210, OP_NOP, 0LL,    OP_SLL, 2LL); /* stage#5 k  =i+n+BS(2) */\
	exe(OP_FML,     &tr0,       ar,         EXP_H3210,  rok,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#5 */\
	exe(OP_FML,     &ti0,       ar,         EXP_H3210,  iok,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#5 */\
        \
        exe(OP_ADD3,    &IDX[y],    0LL,        EXP_H3210,  n,       EXP_H3210, BS,           EXP_H3210, OP_NOP, 0LL,    OP_SLL, 2LL); /* stage#6 idx=  n+BS(2) */\
	exe(OP_FMS,     &tr1,       tr0,        EXP_H3210,  ai,      EXP_H3210, iok,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#6 */\
	exe(OP_FMA,     &ti1,       ti0,        EXP_H3210,  ai,      EXP_H3210, rok,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL)  /* stage#6 */
#define fft_core3(r, x, y) \
	exe(OP_FMS,     &AR[r][0],  roj,        EXP_H3210,  tr1,     EXP_H3210, 0x3f800000LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#7 */\
	mop(OP_STWR, 3, &AR[r][0],  BufReal[x], K[x],       MSK_D0,  BufReal[x],0LL,          0, 0,  NULL,  0LL);                      /* stage#7 */\
	mop(OP_LDWR, 3, &rok,       K[y],       BufReal[y], MSK_D0,  BufReal[x],0LL,          0, 0,  NULL,  0LL);                      /* stage#7 BufReal[k] */\
	exe(OP_FAD,     &AR[r][1],  roj,        EXP_H3210,  tr1,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#7 */\
	mop(OP_STWR, 3, &AR[r][1],  BufReal[x], J[x],       MSK_D0,  BufReal[x],0LL,          0, 0,  NULL,  0LL);                      /* stage#7 */\
	mop(OP_LDWR, 3, &roj,       J[y],       BufReal[y], MSK_D0,  BufReal[x],0LL,          0, 0,  NULL,  0LL);                      /* stage#7 BufReal[j] */\
	exe(OP_FMS,     &AR[r][2],  ioj,        EXP_H3210,  ti1,     EXP_H3210, 0x3f800000LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#7 */\
	mop(OP_STWR, 3, &AR[r][2],  BufImag[x], K[x],       MSK_D0,  BufImag[x],0LL,          0, 0,  NULL,  0LL);                      /* stage#7 */\
	mop(OP_LDWR, 3, &iok,       K[y],       BufImag[y], MSK_D0,  BufImag[x],0LL,          0, 0,  NULL,  0LL);                      /* stage#7 BufImag[k] */\
	exe(OP_FAD,     &AR[r][3],  ioj,        EXP_H3210,  ti1,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#7 */\
	mop(OP_STWR, 3, &AR[r][3],  BufImag[x], J[x],       MSK_D0,  BufImag[x],0LL,          0, 0,  NULL,  0LL);                      /* stage#7 */\
	mop(OP_LDWR, 3, &ioj,       J[y],       BufImag[y], MSK_D0,  BufImag[x],0LL,          0, 0,  NULL,  0LL)                       /* stage#7 BufImag[j] */
#define fft_final(r, x) \
	exe(OP_FMS,     &AR[r][0],  roj,        EXP_H3210,  tr1,     EXP_H3210, 0x3f800000LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#7 */\
	mop(OP_STWR, 3, &AR[r][0],  RealOut,    K[x],       MSK_D0,  RealOut,   NumSamples,   0, 0,  NULL,  NumSamples);               /* stage#7 */\
	exe(OP_FAD,     &AR[r][1],  roj,        EXP_H3210,  tr1,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#7 */\
	mop(OP_STWR, 3, &AR[r][1],  RealOut,    J[x],       MSK_D0,  RealOut,   NumSamples,   0, 0,  NULL,  NumSamples);               /* stage#7 */\
	exe(OP_FMS,     &AR[r][2],  ioj,        EXP_H3210,  ti1,     EXP_H3210, 0x3f800000LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#7 */\
	mop(OP_STWR, 3, &AR[r][2],  ImagOut,    K[x],       MSK_D0,  ImagOut,   NumSamples,   0, 0,  NULL,  NumSamples);               /* stage#7 */\
	exe(OP_FAD,     &AR[r][3],  ioj,        EXP_H3210,  ti1,     EXP_H3210, 0LL,          EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#7 */\
	mop(OP_STWR, 3, &AR[r][3],  ImagOut,    J[x],       MSK_D0,  ImagOut,   NumSamples,   0, 0,  NULL,  NumSamples)                /* stage#7 */

  for (Pipeline=0; Pipeline<NumBits; Pipeline++) {
    /* 0: buf[0]=[(0+4-0)%4]:0 buf[1]=[(1+4-0)%4]:1 buf[2]=[(2+4-0)%4]:2 buf[3]=[(3+4-0)%4]:3 */
    /* 1: buf[0]=[(0+4-1)%4]:3 buf[1]=[(1+4-1)%4]:0 buf[2]=[(2+4-1)%4]:1 buf[3]=[(3+4-1)%4]:2 */
    /* 2: buf[0]=[(0+4-2)%4]:2 buf[1]=[(1+4-2)%4]:3 buf[2]=[(2+4-2)%4]:0 buf[3]=[(3+4-2)%4]:1 */
    for (Lmmrotate=0; Lmmrotate<=NumBits; Lmmrotate++) {
      BufReal[Lmmrotate] = &pseudoLMM[NumSamples*(((Lmmrotate+NumBits+1-Pipeline)%(NumBits+1))*2  )];
      BufImag[Lmmrotate] = &pseudoLMM[NumSamples*(((Lmmrotate+NumBits+1-Pipeline)%(NumBits+1))*2+1)];
    }
//EMAX5A begin pipeline mapdist=0
/*3*/for (CHIP=0; CHIP<NCHIP; CHIP++) {
 /*1*/for (INIT0=1,LOOP0=NumSamples/2,L=0LL<<32|(0-1LL)&0xffffffff; LOOP0--; INIT0=0) { /* NumSamples<=4096 */
        exe(OP_ADD,     &L,         L,          EXP_H3210,  0LL<<32|1LL, EXP_H3210, 0LL,      EXP_H3210, OP_NOP, 0LL,    OP_NOP, 0LL); /* stage#0 */
#if (H==2)
        fft_core0( 1, 0,    0xfffeLL, 0x0000LL, 1LL); /* stage#1-2   */
        fft_core1( 3, 0);                             /* stage#3     */
        fft_core2( 4, 0, 1, 0xfffcLL, 0x0001LL, 2LL); /* stage#4-6   */
        fft_final( 7, 0);                             /* stage#7     */
	//if (Pipeline==0) printf("j=%d %7.1f %7.1f k=%d %7.1f %7.1f\n", (Uint)J[0]/4, RealOut[J[0]/4], ImagOut[J[0]/4], (Uint)K[0]/4, RealOut[K[0]/4], ImagOut[K[0]/4]);
#endif
#if (H==4)
        fft_core0( 1, 0,    0xfffeLL, 0x0000LL, 1LL); /* stage#1-2   */
        fft_core1( 3, 0);                             /* stage#3     */
        fft_core2( 4, 0, 1, 0xfffcLL, 0x0001LL, 2LL); /* stage#4-6   */
        fft_core3( 7, 0, 1);                          /* stage#7     */
	//if (Pipeline==0) printf("j=%d %7.1f %7.1f k=%d %7.1f %7.1f\n", (Uint)J[0]/4, *(float*)(Uint)(BufReal[0]+J[0]), *(float*)(Uint)(BufImag[0]+J[0]), (Uint)K[0]/4, *(float*)(Uint)(BufReal[0]+K[0]), *(float*)(Uint)(BufImag[0]+K[0]));
        fft_core2( 8, 1, 2, 0xfff8LL, 0x0003LL, 4LL); /* stage#8-10  */
        fft_final(11, 1);                             /* stage#11    */
	//if (Pipeline==1) printf("j=%d %7.1f %7.1f k=%d %7.1f %7.1f\n", (Uint)J[1]/4, RealOut[J[1]/4], ImagOut[J[1]/4], (Uint)K[1]/4, RealOut[K[1]/4], ImagOut[K[1]/4]);
#endif
#if (H==8)
        fft_core0( 1, 0,    0xfffeLL, 0x0000LL, 1LL); /* stage#1-2   */
        fft_core1( 3, 0);                             /* stage#3     */
        fft_core2( 4, 0, 1, 0xfffcLL, 0x0001LL, 2LL); /* stage#4-6   */
        fft_core3( 7, 0, 1);                          /* stage#7     */
	//if (Pipeline==0) printf("j=%d %7.1f %7.1f k=%d %7.1f %7.1f\n", (Uint)J[0]/4, *(float*)(Uint)(BufReal[0]+J[0]), *(float*)(Uint)(BufImag[0]+J[0]), (Uint)K[0]/4, *(float*)(Uint)(BufReal[0]+K[0]), *(float*)(Uint)(BufImag[0]+K[0]));
        fft_core2( 8, 1, 2, 0xfff8LL, 0x0003LL, 4LL); /* stage#8-10  */
        fft_core3(11, 1, 2);                          /* stage#11    */
	//if (Pipeline==1) printf("j=%d %7.1f %7.1f k=%d %7.1f %7.1f\n", (Uint)J[1]/4, *(float*)(Uint)(BufReal[1]+J[1]), *(float*)(Uint)(BufImag[1]+J[1]), (Uint)K[1]/4, *(float*)(Uint)(BufReal[1]+K[1]), *(float*)(Uint)(BufImag[1]+K[1]));
        fft_core2(12, 2, 3, 0xfff0LL, 0x0007LL, 8LL); /* stage#12-14 */
        fft_final(15, 2);                             /* stage#15    */
	//if (Pipeline==2) printf("j=%d %7.1f %7.1f k=%d %7.1f %7.1f\n", (Uint)J[2]/4, RealOut[J[2]/4], ImagOut[J[2]/4], (Uint)K[2]/4, RealOut[K[2]/4], ImagOut[K[2]/4]);
#endif
#if (H==4096)
        fft_core0( 1,  0,     0xfffeLL, 0x0000LL,    1LL); /* stage#1-2   */
        fft_core1( 3,  0);                                 /* stage#3     */
        fft_core2( 4,  0,  1, 0xfffcLL, 0x0001LL,    2LL); /* stage#4-6   */
        fft_core3( 7,  0,  1);                             /* stage#7     */
        fft_core2( 8,  1,  2, 0xfff8LL, 0x0003LL,    4LL); /* stage#8-10  */
        fft_core3(11,  1,  2);                             /* stage#11    */
        fft_core2(12,  2,  3, 0xfff0LL, 0x0007LL,    8LL); /* stage#12-14 */
        fft_core3(15,  2,  3);                             /* stage#15    */
        fft_core2(16,  3,  4, 0xffe0LL, 0x000fLL,   16LL); /* stage#16-18 */
        fft_core3(19,  3,  4);                             /* stage#19    */
        fft_core2(20,  4,  5, 0xffc0LL, 0x001fLL,   32LL); /* stage#20-22 */
        fft_core3(23,  4,  5);                             /* stage#23    */
        fft_core2(24,  5,  6, 0xff80LL, 0x003fLL,   64LL); /* stage#24-26 */
        fft_core3(27,  5,  6);                             /* stage#27    */
        fft_core2(28,  6,  7, 0xff00LL, 0x007fLL,  128LL); /* stage#28-30 */
        fft_core3(31,  6,  7);                             /* stage#31    */
        fft_core2(32,  7,  8, 0xfe00LL, 0x00ffLL,  256LL); /* stage#32-34 */
        fft_core3(35,  7,  8);                             /* stage#35    */
        fft_core2(36,  8,  9, 0xfc00LL, 0x01ffLL,  512LL); /* stage#36-38 */
        fft_core3(39,  8,  9);                             /* stage#39    */
        fft_core2(40,  9, 10, 0xf800LL, 0x03ffLL, 1024LL); /* stage#40-42 */
        fft_core3(43,  9, 10);                             /* stage#43    */
        fft_core2(44, 10, 11, 0xf000LL, 0x07ffLL, 2048LL); /* stage#44-46 */
        fft_core3(47, 10, 11);                             /* stage#47    */
        fft_core2(48, 11, 12, 0xe000LL, 0x0fffLL, 4096LL); /* stage#48-50 */
        fft_final(51, 11);                                 /* stage#51    */
#endif
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm
#endif
  get_nanosec(0);

  /*
  **   Need to normalize if inverse transform...
  */

  if (InverseTransform) {
    float denom = (float)NumSamples;
    
    for (i=0; i<NumSamples; i++) {
      RealOut[i] /= denom;
      ImagOut[i] /= denom;
    }
  }
}

/*--- end of file fourierf.c ---*/
