/*********************************************************************
* Filename:   sha256.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the SHA-256 hashing algorithm.
              SHA-256 is one of the three algorithms in the SHA2
              specification. The others, SHA-384 and SHA-512, are not
              offered in this implementation.
              Algorithm specification can be found here:
               * http://csrc.nist.gov/publications/fips/fips180-2/fips180-2withchangenotice.pdf
              This implementation uses little endian byte order.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#endif
#include "sha256.h"
#define NO_EMAX6LIB_BODY
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"

/****************************** MACROS ******************************/
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
static const WORD k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

void sha256_init_imax_k()
{
  int i;
  for (i=0; i<64; i++)
    imax_k[i] =  k[i];
}

/*********************** FUNCTION DEFINITIONS ***********************/
void sha256_transform(WORD thnum, SHA256_CTX *ctx, WORD *mbuf, WORD *state, WORD *sregs, BYTE *hash)
{
#define DIV 16
#if !defined(EMAX6)
  WORD i, j, th, thm;
  WORD a, b, c, d, e, f, g, h, t1, t2;
  printf("<<CPU>>mbuf=%08.8x mbuflen=%08.8x\n", (Uint)mbuf, (Uint)ctx->mbuflen);
  for (i=0; i<ctx->mbuflen; i+=BLKSIZE) { /* 1データ流内の並列実行は不可能. 多数データ流のパイプライン実行のみ */
    for (th=0; th<thnum; th++) {
      sregs[th*8+0] = state[th*8+0];
      sregs[th*8+1] = state[th*8+1];
      sregs[th*8+2] = state[th*8+2];
      sregs[th*8+3] = state[th*8+3];
      sregs[th*8+4] = state[th*8+4];
      sregs[th*8+5] = state[th*8+5];
      sregs[th*8+6] = state[th*8+6];
      sregs[th*8+7] = state[th*8+7];
    }
    for (j=0; j<BLKSIZE; j+=BLKSIZE/DIV) {
//printf("--%08.8x %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x\n", sregs[0*8+0], sregs[0*8+1], sregs[0*8+2], sregs[0*8+3], sregs[0*8+4], sregs[0*8+5], sregs[0*8+6], sregs[0*8+7]);
      for (th=0; th<thnum; th++) {
        a = sregs[th*8+0];
        b = sregs[th*8+1];
        c = sregs[th*8+2];
        d = sregs[th*8+3];
        e = sregs[th*8+4];
        f = sregs[th*8+5];
        g = sregs[th*8+6];
        h = sregs[th*8+7];
#if   (DIV==64)
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 0]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 0]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
#elif (DIV==16)
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 0]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 0]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 1]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 1]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 2]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 2]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 3]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 3]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
#elif (DIV==8)
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 0]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 0]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 1]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 1]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 2]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 2]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 3]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 3]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 4]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 4]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 5]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 5]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 6]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 6]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 7]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 7]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
#elif (DIV==4)
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 0]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 0]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 1]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 1]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 2]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 2]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 3]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 3]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 4]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 4]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 5]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 5]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 6]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 6]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 7]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 7]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 8]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 8]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+ 9]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 9]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+10]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+10]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+11]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+11]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+12]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+12]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+13]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+13]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+14]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+14]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
        t1 = h+EP1(e)+CH(e,f,g)+k[j+15]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+15]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;
#endif
        sregs[th*8+0] = a;
        sregs[th*8+1] = b;
        sregs[th*8+2] = c;
        sregs[th*8+3] = d;
        sregs[th*8+4] = e;
        sregs[th*8+5] = f;
        sregs[th*8+6] = g;
        sregs[th*8+7] = h;
      }
//printf("  %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x\n", sregs[0*8+0], sregs[0*8+1], sregs[0*8+2], sregs[0*8+3], sregs[0*8+4], sregs[0*8+5], sregs[0*8+6], sregs[0*8+7]);
    }
    for (th=0; th<thnum; th++) {
      state[th*8+0] += sregs[th*8+0];
      state[th*8+1] += sregs[th*8+1];
      state[th*8+2] += sregs[th*8+2];
      state[th*8+3] += sregs[th*8+3];
      state[th*8+4] += sregs[th*8+4];
      state[th*8+5] += sregs[th*8+5];
      state[th*8+6] += sregs[th*8+6];
      state[th*8+7] += sregs[th*8+7];
    }
  }
#else
#undef  NCHIP
#undef  RMGRP
#undef  W
#define NCHIP 1
#define RMGRP 2
#define W     4LL
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  i, j, th, thm;
  Ull  a, b, c, d, d0, e, f, g, h, t1, t2;
  Ull  ep0maj, ep1ch, x, y, hd, gc, fb, ea;
  Ull  md, mbase, mtop, mptop, mlen=thnum==1?ctx->mbuflen:MAX_THNUM*BLKSIZE;
  Ull  kd, kbase, ktop=imax_k;
  Ull  sregs0 = sregs+0;
  Ull  sregs2 = sregs+2;
  Ull  sregs4 = sregs+4;
  Ull  sregs6 = sregs+6;
  printf("<<IMAX2>>mbuf=%08.8x mlen=%08.8x\n", (Uint)mbuf, (Uint)mlen);
  for (i=0; i<ctx->mbuflen; i+=BLKSIZE) { /* 1データ流内の並列実行は不可能. 多数データ流のパイプライン実行のみ */
    mtop  = &mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE];
    for (th=0; th<thnum; th++) {
      *(Ull*)&sregs[th*8+0] = (Ull)state[th*8+4]<<32|state[th*8+0];
      *(Ull*)&sregs[th*8+2] = (Ull)state[th*8+5]<<32|state[th*8+1];
      *(Ull*)&sregs[th*8+4] = (Ull)state[th*8+6]<<32|state[th*8+2];
      *(Ull*)&sregs[th*8+6] = (Ull)state[th*8+7]<<32|state[th*8+3];
    }
    /* mbuf    col0 col1 col2 ... hashは64Wごとに累算.IMAX2には16Wしか入らないので4分割実行        */
    /*         msg0 msg1 msg2 ... 4B*64W*64TH=16KBを一度にLMM.msg連続ならstride=64Wで64τ分        */
    /*                            PLOADを使う.中間hashは毎回取り出し,先頭に再投入,4回ごとに掃き出す*/
    /*------------------------------------------------------------------                           */
    /*                               j=0    j=16    j=32    j=48                                   */
    /* row0    abcd 1234 ABCD ...    k[0]   k[16]   k[32]   k[48]                                  */
    /* row1    efgh 5678 EFGH ...    k[1]   k[17]   k[33]   k[49]                                  */
    /* row2    ijkl 9012 IJKL ...    k[2]   k[18]   k[34]   k[50]                                  */
    /*  :                                                                                          */
    /* row15   wxyz 9999 WXYZ ...    k[15]  k[31]   k[47]   k[63]                                  */
    /*--------(64wordを4分割)-------------------------------------------                           */
    /* t1 = h+EP1(e)+CH(e,f,g)+k[ 0]+mbuf[i+ 0];                    */
    /*       k[r]                                                   *//* OP_LD    */
    /*        +                                                     */
    /*       mbuf[i+r]                                              *//* OP_LD    */
    /*        +                                                     */
    /*       h                                                      */
    /*        +                                                     */
    /*       EP1: (ROTRIGHT(e,6) ^ ROTRIGHT(e,11) ^ ROTRIGHT(e,25)) *//* OP_ROTS1 */
    /*        +                                                     */
    /*       CH:  (e & f) ^ (~e & g)                                *//* OP_CH    */

    /* t2 = EP0(a)+MAJ(a,b,c);                                      */
    /*       EP0: (ROTRIGHT(a,2) ^ ROTRIGHT(a,13) ^ ROTRIGHT(a,22)) *//* OP_ROTS2 */
    /*        +                                                     */
    /*       MAJ: (a & b) ^ (a & c) ^ (b & c)                       *//* OP_MAJ   */

    /* e = d+t1; a = t1+t2;                                         */

    /*      col#3        |      col#2          |       col#1          |       col#0          */
    /*                   |    H         L      |                      |    H         L       */
    /*                   |    e        efg     |                      |    a        abc      */
    /*      d0=d         | OP_ROTS=OP_CH OP_LD |                      | OP_ROTS=OP_MAJ OP_LD */
    /*                   |    ep1      ch    k |                      |    ep0      maj    m */
    /*      hd=gc        | OP_ADD3(h+ep1+ch)   | OP_ADD(k+m)          | OP_ADD(ep0+maj)      */
    /*                   |    t1.x             |    t1.y              |    t2                */
    /* a=OP_ADD3(t2+x+y) | e=OP_ADD3(d0+x+y)   |    fb=ea             |    gc=fb             */

    /* a,b,c,d,e,f,g,h をLMM経由でctx.stateに一旦ストアし，継続実行 */
    /* state[8]を各データ流にアサインして二次元配列化.CGRAとしてパイプライン処理 */
    /* t1 = h+EP1(e)+CH(e,f,g)+k[j+ 0]+mbuf[i/BLKSIZE*MAX_THNUM*BLKSIZE+th*BLKSIZE+j+ 0]; t2 = EP0(a)+MAJ(a,b,c); h = g; g = f; f = e; e = d+t1; d = c; c = b; b = a; a = t1+t2;*/
#define sha256_core1(r, ofs) \
  exe(OP_NOP,     &AR[r][0], 0,      EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_NOP,  0,                    OP_NOP, 0);\
  mop(OP_LDWR, 3, &md,       mbase,  ofs,       MSK_D0, mtop,      mlen,        0, 0,      mptop,   mlen);                           \
  exe(OP_MAJ,     &ep0maj,   a,      EXP_H1010, fb,     EXP_H1010, gc,          EXP_H1010, OP_ROTS, (2LL<<48)|(13LL<<40)|(22LL<<32), OP_NOP, 0);\
  exe(OP_NOP,     &AR[r][2], 0,      EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_NOP,  0,                    OP_NOP, 0);\
  mop(OP_LDWR, 3, &kd,       kbase,  ofs,       MSK_D0, ktop,      64,          0, 0,      NULL,    64);                             \
  exe(OP_CH,      &ep1ch,    e,      EXP_H3232, fb,     EXP_H3232, gc,          EXP_H3232, OP_ROTS, (6LL<<48)|(11LL<<40)|(25LL<<32), OP_NOP, 0);\
  exe(OP_NOP,     &d0,       hd,     EXP_H1010, 0,      EXP_H3210, 0,           EXP_H3210, OP_AND,  0xffffffff00000000LL, OP_NOP, 0);\
  exe(OP_ADD,     &t2,       ep0maj, EXP_H3232, ep0maj, EXP_H1010, 0,           EXP_H3210, OP_AND,  0x00000000ffffffffLL, OP_NOP, 0);\
  exe(OP_ADD,     &y,        kd,     EXP_H3210, md,     EXP_H3210, 0,           EXP_H3210, OP_AND,  0x00000000ffffffffLL, OP_NOP, 0);\
  exe(OP_ADD3,    &x,        hd,     EXP_H3232, ep1ch,  EXP_H3232, ep1ch,       EXP_H1010, OP_AND,  0x00000000ffffffffLL, OP_NOP, 0);\
  exe(OP_NOP,     &hd,       gc,     EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_OR,   0,                    OP_NOP, 0);\
  exe(OP_NOP,     &gc,       fb,     EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_OR,   0,                    OP_NOP, 0);\
  exe(OP_NOP,     &fb,       e,      EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_OR,   a,                    OP_NOP, 0);\
  exe(OP_ADD3,    &a,        t2,     EXP_H3210, x,      EXP_H1010, y,           EXP_H1010, OP_AND,  0x00000000ffffffffLL, OP_NOP, 0);\
  exe(OP_ADD3,    &e,        d0,     EXP_H3210, x,      EXP_H1010, y,           EXP_H1010, OP_AND,  0xffffffff00000000LL, OP_NOP, 0)

#define sha256_final(r) \
  exe(OP_NOP,    &ea,        e,      EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_OR,   a,                    OP_NOP, 0);\
  mop(OP_STR, 3, &ea,        sregs0, th,        MSK_W0, sregs,     MAX_THNUM*8, 0, 0,      NULL,    MAX_THNUM*8);                    \
  exe(OP_NOP,    &AR[r][1],  0,      EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_NOP,  0,                    OP_NOP, 0);\
  mop(OP_STR, 3, &fb,        sregs2, th,        MSK_W0, sregs,     MAX_THNUM*8, 0, 0,      NULL,    MAX_THNUM*8);                    \
  exe(OP_NOP,    &AR[r][2],  0,      EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_NOP,  0,                    OP_NOP, 0);\
  mop(OP_STR, 3, &gc,        sregs4, th,        MSK_W0, sregs,     MAX_THNUM*8, 0, 0,      NULL,    MAX_THNUM*8);                    \
  exe(OP_NOP,    &AR[r][3],  0,      EXP_H3210, 0,      EXP_H3210, 0,           EXP_H3210, OP_NOP,  0,                    OP_NOP, 0);\
  mop(OP_STR, 3, &hd,        sregs6, th,        MSK_W0, sregs,     MAX_THNUM*8, 0, 0,      NULL,    MAX_THNUM*8)

    for (INIT1=1,LOOP1=DIV,j=0; LOOP1--; INIT1=0,j+=BLKSIZE/DIV*4) {
      mptop = (LOOP1==0)? mtop+MAX_THNUM*BLKSIZE*4:mtop; /* 最終回のみPLOAD */
//printf("--%08.8x %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x\n", sregs[0*8+0], sregs[0*8+2], sregs[0*8+4], sregs[0*8+6], sregs[0*8+1], sregs[0*8+3], sregs[0*8+5], sregs[0*8+7]);
//with-prefetch
//EMAX5A begin imax mapdist=0
 /*3*/for (CHIP=0; CHIP<NCHIP; CHIP++) {
   /*1*/for (INIT0=1,LOOP0=thnum,th=(0-BLKSIZE*4)<<32|((0-32LL)&0xffffffff); LOOP0--; INIT0=0) {
          exe(OP_ADD,    &th,     INIT0?th:th, EXP_H3210, (BLKSIZE*4)<<32|32LL,  EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0); /* stage#0 */
          exe(OP_NOP,    &thm,    th,          EXP_H3232, 0,                     EXP_H3210, 0, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0); /* stage#1 */
          exe(OP_ADD3,   &mbase,  mtop,        EXP_H3210, thm,                   EXP_H3210, j, EXP_H3210, OP_NOP, 0,                    OP_NOP, 0); /* stage#2 */
          mop(OP_LDR, 3, &a,      sregs0, th,  MSK_W0,    sregs, MAX_THNUM*8, 0, 1, NULL,  MAX_THNUM*8); /* stage#2 */
          mop(OP_LDR, 3, &e,      sregs0, th,  MSK_W0,    sregs, MAX_THNUM*8, 0, 1, NULL,  MAX_THNUM*8); /* stage#2 */
          exe(OP_ADD3,   &kbase,  imax_k,      EXP_H3210, 0,                     EXP_H3210, j, EXP_H3210, OP_NOP, 0,                    OP_NOP, 0); /* stage#2 */
          mop(OP_LDR, 3, &fb,     sregs2, th,  MSK_W0,    sregs, MAX_THNUM*8, 0, 1, NULL,  MAX_THNUM*8); /* stage#2 */
          mop(OP_LDR, 3, &gc,     sregs4, th,  MSK_W0,    sregs, MAX_THNUM*8, 0, 1, NULL,  MAX_THNUM*8); /* stage#2 */
          mop(OP_LDR, 3, &hd,     sregs6, th,  MSK_W0,    sregs, MAX_THNUM*8, 0, 1, NULL,  MAX_THNUM*8); /* stage#2 */
#if   (DIV==64)
          sha256_core1( 3,  0);
          sha256_final( 6);
#elif (DIV==16)
          sha256_core1( 3,  0);
          sha256_core1( 6,  4);
          sha256_core1( 9,  8);
          sha256_core1(12, 12);
          sha256_final(15);
#elif (DIV==8)
          sha256_core1( 3,  0);
          sha256_core1( 6,  4);
          sha256_core1( 9,  8);
          sha256_core1(12, 12);
          sha256_core1(15, 16);
          sha256_core1(18, 20);
          sha256_core1(21, 24);
          sha256_core1(24, 28);
          sha256_final(27);
#elif (DIV==4)
          sha256_core1( 3,  0);
          sha256_core1( 6,  4);
          sha256_core1( 9,  8);
          sha256_core1(12, 12);
          sha256_core1(15, 16);
          sha256_core1(18, 20);
          sha256_core1(21, 24);
          sha256_core1(24, 28);
          sha256_core1(27, 32);
          sha256_core1(30, 36);
          sha256_core1(33, 40);
          sha256_core1(36, 44);
          sha256_core1(39, 48);
          sha256_core1(42, 52);
          sha256_core1(45, 56);
          sha256_core1(48, 60);
          sha256_final(51);
#endif
        }
      }
//printf("  %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x %08.8x\n", sregs[0*8+0], sregs[0*8+2], sregs[0*8+4], sregs[0*8+6], sregs[0*8+1], sregs[0*8+3], sregs[0*8+5], sregs[0*8+7]);
//EMAX5A end
    }
//EMAX5A drain_dirty_lmm
    for (th=0; th<thnum; th++) {
      state[th*8+0] += sregs[th*8+0];
      state[th*8+1] += sregs[th*8+2];
      state[th*8+2] += sregs[th*8+4];
      state[th*8+3] += sregs[th*8+6];
      state[th*8+4] += sregs[th*8+1];
      state[th*8+5] += sregs[th*8+3];
      state[th*8+6] += sregs[th*8+5];
      state[th*8+7] += sregs[th*8+7];
    }
  }
#endif

  // Since this implementation uses little endian byte ordering and SHA uses big endian,
  // reverse all the bytes when copying the final state to the output hash.
  for (th=0; th<thnum; th++) {
    for (i = 0; i < 4; ++i) {
      hash[th*SHA256_BLOCK_SIZE+i+ 0] = (state[th*8+0]>>(24-i*8))&0xff;
      hash[th*SHA256_BLOCK_SIZE+i+ 4] = (state[th*8+1]>>(24-i*8))&0xff;
      hash[th*SHA256_BLOCK_SIZE+i+ 8] = (state[th*8+2]>>(24-i*8))&0xff;
      hash[th*SHA256_BLOCK_SIZE+i+12] = (state[th*8+3]>>(24-i*8))&0xff;
      hash[th*SHA256_BLOCK_SIZE+i+16] = (state[th*8+4]>>(24-i*8))&0xff;
      hash[th*SHA256_BLOCK_SIZE+i+20] = (state[th*8+5]>>(24-i*8))&0xff;
      hash[th*SHA256_BLOCK_SIZE+i+24] = (state[th*8+6]>>(24-i*8))&0xff;
      hash[th*SHA256_BLOCK_SIZE+i+28] = (state[th*8+7]>>(24-i*8))&0xff;
    }
  }
}

void sha256_init(SHA256_CTX *ctx, WORD *mbuf, WORD *state, const BYTE *str, WORD repeat)
{
  WORD i, j, x, slen, tlen, m;
  Ull  bitlen;
  BYTE s0, s1, s2, s3;

  slen = strlen(str);
  tlen = slen*repeat;
  bitlen = tlen*8;

  ctx->bitlen  = bitlen;
  state[0] = 0x6a09e667;
  state[1] = 0xbb67ae85;
  state[2] = 0x3c6ef372;
  state[3] = 0xa54ff53a;
  state[4] = 0x510e527f;
  state[5] = 0x9b05688c;
  state[6] = 0x1f83d9ab;
  state[7] = 0x5be0cd19;

  m = 0;
#define m2i(m) (m)
  ctx->mbuflen = 0;
  for (i=0,j=0; i<tlen; i++,j++) {
    if (j == slen) j = 0;
    switch (i % 4) {
    case 0:  s3 = str[j];                       break;
    case 1:  s2 = str[j];                       break;
    case 2:  s1 = str[j];                       break;
    default: s0 = str[j];                       mbuf[m2i(m)] = (s3 << 24) | (s2 << 16) | (s1 << 8) | s0;   m++; ctx->mbuflen++;      break;
    }
    if (i % 64 == 63) { /* 64byte(16word)ごとに入力データを追加し,64word化 */
      for (x=0;   x<48; x++,m++,ctx->mbuflen++) mbuf[m2i(m)] = SIG1(mbuf[m2i(m-2)]) + mbuf[m2i(m-7)] + SIG0(mbuf[m2i(m-15)]) + mbuf[m2i(m-16)];
      m += (MAX_THNUM-1)*BLKSIZE;
    }
  }
  switch (i % 4) {
  case 3:  /* ***- */                           mbuf[m2i(m)] = (s3 << 24) | (s2 << 16) | (s1 << 8) | 0x80; m++; ctx->mbuflen++; i += 1; break;
  case 2:  /* **-- */                           mbuf[m2i(m)] = (s3 << 24) | (s2 << 16) |           0x8000; m++; ctx->mbuflen++; i += 2; break;
  case 1:  /* *--- */                           mbuf[m2i(m)] = (s3 << 24) |                      0x800000; m++; ctx->mbuflen++; i += 3; break;
  default: /* ---- */                           mbuf[m2i(m)] =                                 0x80000000; m++; ctx->mbuflen++; i += 4; break;
  }

  if (i%64 < 56) {
    /*      | 00 01 02    62 63 |                                 | */
    /* mbuf | CC CC CC .. CC CC | SIG.............................| */
    /* mbuf | CC 80 00 .. BL BL | SIG.............................| */
    for (; i%64 != 56; i+=4,m++,ctx->mbuflen++) mbuf[m2i(m)] = 0;
  }
  else {
    /*      | 00 01 02    62 63 |                                 | */
    /* mbuf | CC CC CC .. CC CC | SIG.............................| */
    /* mbuf | CC CC CC .. 80 00 | SIG.............................| */
    /* mbuf | 00 00 00 .. BL BL | SIG.............................| */
    for (; i%64 !=  0; i+=4,m++,ctx->mbuflen++) mbuf[m2i(m)] = 0;
    for (x=0;     x<48; x++,m++,ctx->mbuflen++) mbuf[m2i(m)] = SIG1(mbuf[m2i(m-2)]) + mbuf[m2i(m-7)] + SIG0(mbuf[m2i(m-15)]) + mbuf[m2i(m-16)];
    m += (MAX_THNUM-1)*BLKSIZE;
    for (; i%64 != 56; i+=4,m++,ctx->mbuflen++) mbuf[m2i(m)] = 0;
  }
                                                mbuf[m2i(m)] = (WORD)(bitlen>>32);  m++; ctx->mbuflen++;
                                                mbuf[m2i(m)] = (WORD)(bitlen);      m++; ctx->mbuflen++;
  for (x=0;       x<48; x++,m++,ctx->mbuflen++) mbuf[m2i(m)] = SIG1(mbuf[m2i(m-2)]) + mbuf[m2i(m-7)] + SIG0(mbuf[m2i(m-15)]) + mbuf[m2i(m-16)];

  if (m2i(m) >= MAX_MBUF) {    printf("m=%d >= MAX_MBUF(%d)\n", m, MAX_MBUF);    exit(1);  }
}
