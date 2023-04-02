
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
#include "tensor.h"
#include "cnnet.h"

#ifdef CBLAS_GEMM
#include "cblas.h"
#endif

#include "./emax6.h"
#include "./emax6lib.c"

void *memcpy();
int soft32(Uint, float, float, float, float *);
int hard32(Uint, float, float, float, float *, Uint);
int soft64(Uint, float, float, float, float *);
int hard64(Uint, float, float, float, float *);

extern struct c c[2][CNN_DEPTH_MAX];
extern struct f f[2][FC_DEPTH_MAX];
extern int      CNN_DEPTH;/* default 1   */
extern int      FC_DEPTH; /* default 1   */

Uchar   *membase;
int     memsize;
int     memalign;

Uint    *i_inp; /* for CNN on ZYNQ_PL */
Uint    *i_ker; /* for CNN on ZYNQ_PL */
Uint    *i_out; /* for CNN on ZYNQ_PL */
int     i_inp_max_size;
int     i_ker_max_size;
int     i_out_max_size;

Uint    *i_m0A; /* for sgemm00 on ZYNQ_PL */
Uint    *i_m0B; /* for sgemm00 on ZYNQ_PL */
Uint    *i_m0C; /* for sgemm00 on ZYNQ_PL */
int     i_m0A_max_size;
int     i_m0B_max_size;
int     i_m0C_max_size;

#define ERRTH  (5.0E-2)
#define udiff(a,b) (((a)-(b)>=0.0?(a)-(b):(b)-(a))/((a)==0.0?1:(a)))
#define setmax(max, new) { if (max < (new)) max = (new); }

void init_xmax(int batch_size, struct c *c, struct f *f)
{
  int l;

  for (l=0; l<CNN_DEPTH; l++) {
    setmax(i_inp_max_size, batch_size * c[l].ichan * (c[l].isize+c[l].ksize-1) * (c[l].isize+c[l].ksize-1));
    setmax(i_ker_max_size, c[l].ichan * ((c[l].ochan+3)&~3) * c[l].ksize * c[l].ksize);
    setmax(i_out_max_size, batch_size * ((c[l].ochan+3)&~3) * c[l].osize * c[l].osize);
  }
  /* sgemm00(!transA&&!transB)はforwardのfcのみ確保でOK */
  for (l=0; l<FC_DEPTH; l++) {
    setmax(i_m0A_max_size, batch_size * ((l==0)?c[CNN_DEPTH-1].ochan * c[CNN_DEPTH-1].osize * c[CNN_DEPTH-1].osize:f[FC_DEPTH_MAX-FC_DEPTH+l-1].osize));
    setmax(i_m0B_max_size,              ((l==0)?c[CNN_DEPTH-1].ochan * c[CNN_DEPTH-1].osize * c[CNN_DEPTH-1].osize:f[FC_DEPTH_MAX-FC_DEPTH+l-1].osize) * f[FC_DEPTH_MAX-FC_DEPTH+l].osize);
    setmax(i_m0C_max_size, batch_size *                                                                                                                 (f[FC_DEPTH_MAX-FC_DEPTH+l].osize+3)&~3);
  }
  setmax(memsize, (i_inp_max_size+i_ker_max_size+i_out_max_size)*sizeof(int));
  setmax(memsize, (i_m0A_max_size+i_m0B_max_size+i_m0C_max_size)*sizeof(int));
  memalign = 32;

#if defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
  /*{int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}*/
#else
  membase = (void*)malloc(memsize+memalign);
  if ((Ull)membase & (Ull)(memalign-1))
    membase = (void*)(((Ull)membase & ~(Ull)(memalign-1))+memalign);
#endif

  printf("membase: %08.8x\n", (Uint)membase);
  i_inp = (Uint*)membase;
  i_ker = (Uint*)i_inp + i_inp_max_size;
  i_out = (Uint*)i_ker + i_ker_max_size;
  printf("i_inp : %08.8x-%08.8x\n", (Uint)i_inp, (Uint)i_inp+i_inp_max_size*sizeof(int)-1);
  printf("i_ker : %08.8x-%08.8x\n", (Uint)i_ker, (Uint)i_ker+i_ker_max_size*sizeof(int)-1);
  printf("i_out : %08.8x-%08.8x\n", (Uint)i_out, (Uint)i_out+i_out_max_size*sizeof(int)-1);
  i_m0A = (Uint*)membase;
  i_m0B = (Uint*)i_m0A + i_m0A_max_size;
  i_m0C = (Uint*)i_m0B + i_m0B_max_size;
  printf("i_m0A : %08.8x-%08.8x\n", (Uint)i_m0A, (Uint)i_m0A+i_m0A_max_size*sizeof(int)-1);
  printf("i_m0B : %08.8x-%08.8x\n", (Uint)i_m0B, (Uint)i_m0B+i_m0B_max_size*sizeof(int)-1);
  printf("i_m0C : %08.8x-%08.8x\n", (Uint)i_m0C, (Uint)i_m0C+i_m0C_max_size*sizeof(int)-1);

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
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // RESET
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

void imemcpy(Uint *dst, Uint *src, int words)
{
  union {
    Uint i[4];
    Ull  l[2];
    Dll  d;
  } buf;

  Uint loop, i;
  if (words >= 1 && ((Ull)dst & sizeof(Uint))) { /* 4B-access odd */
    *dst++ = *src++;
    words--;
  }
  if (words >= 2 && ((Ull)dst & sizeof(Ull))) { /* 8B-access odd */
    if ((Ull)src & sizeof(Uint)) {
      buf.i[0] = *src++;
      buf.i[1] = *src++;
      *(Ull*)dst = buf.l[0];
    }
    else {
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull)/sizeof(Uint);
    }
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }

  if (loop = words/(sizeof(Dll)/sizeof(Uint))) {
    if ((Ull)src & sizeof(Uint)) {
      for(i=0; i<loop; i++) {
	buf.i[0] = *src++;
	buf.i[1] = *src++;
	buf.i[2] = *src++;
	buf.i[3] = *src++;
	*(Dll*)dst = buf.d;
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    else if ((Ull)src & sizeof(Ull)) {
      for(i=0; i<loop; i++) {
	buf.l[0] = *(Ull*)src;src += sizeof(Ull)/sizeof(Uint);
	buf.l[1] = *(Ull*)src;src += sizeof(Ull)/sizeof(Uint);
	*(Dll*)dst = buf.d;
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    else {
      for(i=0; i<loop; i++) {
	*(Dll*)dst = *(Dll*)src;
	src += sizeof(Dll)/sizeof(Uint);
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    words -= loop*(sizeof(Dll)/sizeof(Uint));
  }

  if (words >= 2) { /* 8B-access */
    if ((Ull)src & sizeof(Uint)) {
      buf.i[0] = *src++;
      buf.i[1] = *src++;
      *(Ull*)dst = buf.l[0];
    }
    else {
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull)/sizeof(Uint);
    }
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }
  if (words >= 1) { /* 4B-access */
    *dst++ = *src++;
    words--;
  }
}

void xmax_bzero(Uint *dst, int words)
{
  /* +----+-m-----+ */
  /* |3x3 |       | */
  /* |    |    src| */
  /* +----+       | */
  /* |       +----+ */
  /* |       |    | */
  /* |       | 3x3| */
  /* +-------+----+ */
  Uint loop, i;
  if (words >= 1 && ((Ull)dst & sizeof(Uint))) { /* 4B-access odd */
    *dst++ = 0;
    words--;
  }
  if (words >= 2 && ((Ull)dst & sizeof(Ull))) { /* 8B-access odd */
    *(Ull*)dst = 0;
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }

  if (loop = words/(sizeof(Dll)/sizeof(Uint))) {
    for(i=0; i<loop; i++) {
#if __AARCH64EL__ == 1
      *((Dll*)dst) = 0;
#else
      ((Dll*)dst)->u[0] = 0;
      ((Dll*)dst)->u[1] = 0;
#endif
      dst += sizeof(Dll)/sizeof(Uint);
    }
    words -= loop*(sizeof(Dll)/sizeof(Uint));
  }

  if (words >= 2) { /* 8B-access */
    *(Ull*)dst = 0;
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }
  if (words >= 1) { /* 4B-access */
    *dst++ = 0;
    words--;
  }
}

void xmax_cpyin(int order, Uint *dst, int *imo, Uint *src, int batch, int ic, int im, int m, int k)
{
  /* order 0: dst[batch][ic][im*im]  <- src[batch][ic][im*im] */
  /* order 1: dst[ic][im][batch][im] <- src[batch][ic][im*im] */
  /* order 2: dst[im][m]             <- src[im][m]            */

  switch (order) {
  case 0:
    /* num=batch+ichan                            */
    /* imiの周辺に0を追加しimoにコピー            */
    /* k=3,(IM==M)             k=2,(IM==M)        */
    /* +-------+imo-------+    +-----+--imo----+  */
    /* | 0 0 0 |       dst|    | 0 0 |      dst|  */
    /* |  +----+im=m---+  |    |  +--+--im=m---+  */
    /* | 0|3x3 |       |  |    | 0|  |         |  */
    /* | 0|    |    src|  |    +--+--+      src|  */
    /* +--+----+       |  |    |  |            |  */
    /* |  |       +----+--+    |  |            |  */
    /* |  |       |    |0 |    |  |            |  */
    /* |  |       | 3x3|0 |    |  |            |  */
    /* |  +-------+----+  |    +--+------------+  */
    /* |          | 0 0 0 |                       */
    /* +----------+-------+                       */

    /* imiとimoは同じサイズでコピー                                 */
    /* k=3,(IM-k)/1+1==M       k=2,(IM-k)/1+1==M    k=1,(IM==M)     */
    /* +-------+im--------+    +-----+--im-----+                    */
    /* | x x x |       dst|    | x x |      dst|                    */
    /* |  +----+-m-----+  |    |  +--+---m-----+    +--+--im=m---+  */
    /* | x|3x3 |       |  |    | x|  |         |    |  |         |  */
    /* | x|    |    src|  |    +--+--+      src|    +--+      src|  */
    /* +--+----+       |  |    |  |            |    |            |  */
    /* |  |       +----+--+    |  |            |    |            |  */
    /* |  |       |    |x |    |  |            |    |         +--+  */
    /* |  |       | 3x3|x |    |  |            |    |         |  |  */
    /* |  +-------+----+  |    +--+------------+    +---------+--+  */
    /* |          | x x x |                                         */
    /* +----------+-------+                                         */
    /* EMAX for large IM/M                                   *//*         burst_exe 6*6    ||         burst_exe 6*6    */
    /*     +-----+  +----+-+----+---------+    +-----------+ *//* 7*8... | 7*8... | 7*8... || 7*8... | 7*8... | 7*8... */
    /* unit|2    |  |7*7 | |7*7 |*IC  *100|    |2          | *//*-- -- --                  ||-- -- --                  *//* LMM=7*8*4B */
    /*  |  |*    |  |ch0 | |ch1 |         | -> |*          | *//*         -- -- --         ||         -- -- --         *//*    =244B   */
    /*  V  |2    |  +----+ +----+         |    |2          | *//*                  -- -- --||                  -- -- --*/
    /*     |*ich |  |loop=RMGRP(6)*M(6)   |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*6*4B*4och */
    /*     +-och-+  +---------------------+    +6*6*och----+ *//* img0     img0     img0   || img1     img1     img1   *//*    =576B        */
    /*        32 ... lmf+lmx毎回DMA            |    32/4   | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    if (im == m && 1<k) {
      int n, i, w = im+k-1;
      for (n=0; n<batch*ic; n++,dst+=w*w,src+=im*im) {
	for (i=0; i<k/2; i++)
	  xmax_bzero(dst+i*w, w);
	for (i=k/2; i<=im+k/2-1; i++) {
	  xmax_bzero (dst+i*w,               (k/2) );
	  imemcpy(dst+i*w+(k/2),   src+(i-k/2)*im, im);
	  if (k-1-(k/2)) xmax_bzero (dst+i*w+(k/2)+im, k-1-(k/2));
	}
	for (i=im+k/2; i<w; i++)
	  xmax_bzero(dst+i*w, w);
      }
      *imo = w;
    }
    else {
      imemcpy(dst, src, batch*ic*im*im);
      *imo = im;
    }
    break;
  case 1:
    /* EMAX for small IM/M                                   */
    /*     +-----+  +---------------------+    +-----------+ *//*         burst_exe 6*100  ||         burst_exe 6*100  *//* 100画像を1枚(7*700pix)に(7*100を7行) */
    /* unit|     |  |+----PAD----+        |    |           | *//* 7*8*100| 7*8*100| 7*8*100|| 7*8*100| 7*8*100| 7*8*100*//* または7*7連続アドレスを100セット     */
    /*  |  |2    |  ||7*7 | |7*7 |*100 *IC| -> |2          | *//*-- -- --                    -- -- --                  *//* LMM=7*8*4B*100 LMMstg2-7にload       */
    /*  |  |*    |  ||im0 | |im1 |        |    |*          | *//* top=0   -- -- --            top=1   -- -- --         *//*    =22400B(RMGRP=7で2回再利用)<32KB  */
    /*  V  |2    |  |+----+ +----+        |    |2          | *//*                  -- -- --                    -- -- --*/
    /*     |*ich |  |loop=M(6)*BATCH(100) |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*4B*100*4och */
    /*     +-och-+  +---------------------+    +6*100*och--+ *//* img0-99  img0-99  img0-99|| img0-99  img0-99  img0-99*//*    =9600B         */
    /*        32 ... lmf+lmx毎回DMA            |      32/4 | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    if (im == m && 1<k) {
      int n1, n0, i, w = im+k-1;
      for (n1=0; n1<batch; n1++) {           /* src-data順 */
	for (n0=0; n0<ic; n0++,src+=im*im) { /* src-data順 */
	  int ofs  = (n0*w*batch+n1)*w;      /* 複数imgの1行が連続,ch毎に連続 */
	  int dist =  batch*w;               /* 複数imgの1行が連続,時アドレスは次行 */
	  for (i=0; i<k/2; i++)
	    xmax_bzero(dst+ofs+i*dist, w);
	  for (i=k/2; i<=im+k/2-1; i++) {
	    xmax_bzero (dst+ofs+i*dist,               (k/2) );
	    imemcpy(dst+ofs+i*dist+(k/2),   src+(i-k/2)*im, im);
	    if (k-1-(k/2)) xmax_bzero (dst+ofs+i*dist+(k/2)+im, k-1-(k/2));
	  }
	  for (i=im+k/2; i<w; i++)
	    xmax_bzero(dst+ofs+i*dist, w);
	}
      }
      *imo = w;
    }
    else {
      int n1, n0, i;
      for (n1=0; n1<batch; n1++) {           /* src-data順 */
	for (n0=0; n0<ic; n0++,src+=im*im) { /* src-data順 */
	  int ofs  = (n0*im*batch+n1)*im;
	  int dist =  batch*im;
	  for (i=0; i<im; i++)
	    imemcpy(dst+ofs+i*dist, src+i*im, im);
	}
      }
      *imo = im;
    }
    break;
  case 2:
    imemcpy(dst, src, im*m);
    *imo = im;
    break;
  }
}

void xmax_cpyout(int order, Uint *dst, int batch, int oc, Uint *src, int m, int n, int oc4)
{
  /* order 0: dst[batch][oc][m*n] <- src[batch][oc4][m*n]  */
  /* order 1: dst[batch][oc][m*n] <- src[oc4][m][batch][n] */
  /* order 2: dst[m][n]           <- src[m][oc4=(n+3)&~3]  */

  /* +-dst--------------+    +-imo--------------+ */
  /* | OC | OC | OC |   | <- | OC4   | OC4   |  | */
  /* +------------------+    +------------------+ */
  int k, k2, k1, k0;

  switch (order) {
  case 0:
    for (k=0; k<batch; k++,dst+=oc*m*n,src+=oc4*m*n)
      imemcpy(dst, src, oc*m*n);
    break;
  case 1:
    for (k2=0; k2<batch; k2++) {
      for (k1=0; k1<oc; k1++) {
	for (k0=0; k0<m; k0++,dst+=n)
	  imemcpy(dst, src+((k1*m+k0)*batch+k2)*n, n);
      }
    }
    break;
  case 2:
    if (n == oc4)
      imemcpy(dst, src, m*n);
    else {
      for (k=0; k<m; k++,dst+=n,src+=oc4)
	imemcpy(dst, src, n);
    }
    break;
  }
}

void xmax_conv_forward(float4D *in, float2D *kernel, float4D *out, int ksize)
{
  /* float4D.nstrides    .. batch_size        */
  /* float4D.nchannel    .. ichan/ochan       */
  /* float4D.kstrides    .. isize/osize       */
  /* float4D.stride_size .. isize/osize       */
  /* float4D.data                             */
  /* float2D.nstrides    .. ochan             */
  /* float2D.stride_size .. ichan*ksize*ksize */
  /* float2D.data                             */
  /* in[batch_size, ichan, isize*isize] * weight[ochan, ichan, ksize*ksize] */
  /*  -> out[batch_size, ochan, osize*osize ] */
  /* IM == Mの場合, in->dataの周辺にPAD追加   */
  /* float *i_inp; PAD+in->data をcopy     */
  /* float *i_ker; ker->data    をcopy     */
  /* float *i_out; out->data    へcopy     */

  /* PAD+IM*2                                 */
  /*      <-----Nich*28*28----->  <next img>  */
  /* IM A +--------++--------+ .. +--------+  */
  /*  A | | +-24-+ || +----+ | .. | +----+ |  */
  /*  | | | | ch0| || | ch1| | .. | | ch0| |  */
  /*  | | | 24   | || |    | | .. | |    | |  */
  /*  V | | +----+ || +----+ | .. | +----+ |  */
  /*    V +--------++--------+ .. +--------+  */
  /*      <PAD+IM*2>                          */
  /*        <-IM->                            */

  int   BATCH  = in->nstrides;  // 100
  int   RMGRP;
  int   IC     = in->nchannel;  // IMAP*Xin
  int   IM     = in->kstrides;  // 28
  int   OC     = out->nchannel; // W*Xout
  int   M      = out->kstrides; // 24
  int   K      = ksize;         // 5,4,3,2,1
  int   Klen   = OC*IC*K*K;
  int   OC4    = (OC+3)&~3;
  Uint  *in0   = in->data;      // IC*IM*IM
  Uint  *ker   = kernel->data;  // OC*IC*K*K
  Uint  *out0  = out->data;     // OC*M*M
  Uint  *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp,  kidx, *op;
  int   pad;
  int   count, top, iset, oc, w, ic, y, x;
  Ull   IM4, M4, IM4M4, IMlen, Mlen, Force;
  Ull   CHIP, img, rofs, cofs, iofs, oofs;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;

  if (IM == M)
    pad = 0;   /* PAD無し.in周囲0.0を仮定 */
  else if ((IM - K)/1 + 1 == M)
    pad = K/2; /* PAD有り.in周囲特別扱い不要 */
  else {
    printf("xmax_conv_forward error: IM=%d K=%d M=%d\n", IM, K, M);
    printf("IM == M || (IM-K)/1+1 == M\n");
    exit(-1);
  }

  /* i_inp, i_ker, i_outは確保済だが性能評価には使わない */
  /*printf("<<<XMAX(C)>>>\n");*/
  /*printf("xmax IM=%d M=%d K=%d %d*%d*%d\n", IM, M, K, OC, BATCH*M*M, IC*K*K);*/
  /*printf("<<<XMAX(REAL)>>>\n");*/

  switch (K) { /* 5, 3, 2 */
  case 5:
    RMGRP = M; /* RMGRP = 24 {28,1,5,24,9,2} */
               /* RMGRP = 28 {32,3,5,28,11,2}*/
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ★★★ PBL1-1 ★★★ */
/* NCHIP  4 ★★★ PBL1-1 ★★★ */
#define IMAP  1
#define W     4
#define NCHIP 1
#undef XMAX_VALIDATE
//#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
    for (img=0; img<BATCH; img++) {
      for (top=0; top<M; top+=RMGRP) {
        for (iset=0; iset<IC; iset+=IMAP) { /* accumulate multiple sets of IC */
          for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */                                            /* ICHをなるべく温存し,外側LOOPで全OCHを先に片付ける */
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */ /* ICHをなるべく温存し,複数CHIPで全OCHを先に片付ける */
        /*2*/ for (rofs=0; rofs<RMGRP&&(top+rofs)<M; rofs++) { /* image loop (row) */                          /* 1.全ICH複数行   */
          /*1*/ for (cofs=0; cofs<M; cofs++) { /* image loop (cofs) */                                         /* 2.全ICH水平方向 */
                  iofs = rofs*IM+cofs;
                  oofs = rofs*M+cofs;
                  for (w=0; w<W&&(oc+w)<OC/NCHIP; w++) { /* set output channel */                              /* ICHをなるべく温存し,４列使用で全OCHを先に片付ける */
                    op = &out0[(img*OC+CHIP*OC/NCHIP+oc+w)*M*M+top*M+oofs]; /* top of output */
                    for (ic=0; ic<IMAP&&(iset+ic)<IC; ic++) { /* set offset of input channel */
                      ip0  = &in0[(img*IC+iset+ic)*IM*IM+pad*IM+pad]; /* top of input */
                      kp   = &ker[((CHIP*OC/NCHIP+oc+w)*IC+iset+ic)*K*K];
                      kidx = 0;
                      for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
                        for (x=-(K/2); x<K-(K/2); x++) {
                          float in = (0 <= top+rofs+y+pad && top+rofs+y+pad < IM
                                   && 0 <=     cofs+x+pad &&     cofs+x+pad < IM)
                            ? *(float*)&ip0[top*IM+iofs+y*IM+x] : 0.0;
                          if (iset == 0 && ic == 0 && kidx == 0)
                            *(float*)op  = in * *(float*)&kp[kidx];
                          else
                            *(float*)op += in * *(float*)&kp[kidx];
                          kidx++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
#endif
    /*{28,1,5,24,9,2}/{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6, 32,2}*/
    /* AAAAAAAAAAAAA / AAAAAAAAAAAAAA                                                    */
    xmax_cpyin(0, i_inp, &IM, in0, BATCH, IC, IM, M, K); /* この時点で0.0のPADを追加できる */
    xmax_cpyin(0, i_ker, &K,  ker, IC,    OC,  K, K, 1); /* 出力のみ回収すればよい */
    xmax_bzero(i_out, BATCH*OC4*M*M);
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlen = IM*(RMGRP+4);
    Mlen  = M*RMGRP;
    Force = 1;

    if (Klen > 65536/4/2 || IMlen > 65536/4/2 || Mlen > 65536/4/4)
      printf("   CNN5x5  Klen=%dB IMlen=%dB Mlen*4=%dB\n", (Uint)Klen*4, (Uint)IMlen*4, (Uint)Mlen*4*4);

    for (img=0; img<BATCH; img++) {
      for (top=0; top<M; top+=RMGRP) {
        for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
          Uint *ip0  = &i_inp[(img*IC+iset+0)*IM*IM]; /* top of input#0 */
          Uint *it00 = ip0+top*IM, *ip00[25];
	  ip00[ 0] = ip0+(top+0)*IM+0; ip00[ 1] = ip0+(top+0)*IM+1; ip00[ 2] = ip0+(top+0)*IM+2; ip00[ 3] = ip0+(top+0)*IM+3; ip00[ 4] = ip0+(top+0)*IM+4;
	  ip00[ 5] = ip0+(top+1)*IM+0; ip00[ 6] = ip0+(top+1)*IM+1; ip00[ 7] = ip0+(top+1)*IM+2; ip00[ 8] = ip0+(top+1)*IM+3; ip00[ 9] = ip0+(top+1)*IM+4;
	  ip00[10] = ip0+(top+2)*IM+0; ip00[11] = ip0+(top+2)*IM+1; ip00[12] = ip0+(top+2)*IM+2; ip00[13] = ip0+(top+2)*IM+3; ip00[14] = ip0+(top+2)*IM+4;
	  ip00[15] = ip0+(top+3)*IM+0; ip00[16] = ip0+(top+3)*IM+1; ip00[17] = ip0+(top+3)*IM+2; ip00[18] = ip0+(top+3)*IM+3; ip00[19] = ip0+(top+3)*IM+4;
	  ip00[20] = ip0+(top+4)*IM+0; ip00[21] = ip0+(top+4)*IM+1; ip00[22] = ip0+(top+4)*IM+2; ip00[23] = ip0+(top+4)*IM+3; ip00[24] = ip0+(top+4)*IM+4;

          for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
            Uint *kp00[NCHIP],*kp01[NCHIP],*kp02[NCHIP],*kp03[NCHIP];
            Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
            Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
              Uint choc  = CHIP*OC4/NCHIP+oc;
              kp00[CHIP]= (choc+0<OC) ? i_ker+((choc+0)*IC+iset+0)*K*K : i_ker;
	      kp01[CHIP]= (choc+1<OC) ? i_ker+((choc+1)*IC+iset+0)*K*K : i_ker;
	      kp02[CHIP]= (choc+2<OC) ? i_ker+((choc+2)*IC+iset+0)*K*K : i_ker;
	      kp03[CHIP]= (choc+3<OC) ? i_ker+((choc+3)*IC+iset+0)*K*K : i_ker;
              op0[CHIP] = i_out+(img*OC4+choc+0)*M*M+top*M; op1[CHIP] = i_out+(img*OC4+choc+1)*M*M+top*M; op2[CHIP] = i_out+(img*OC4+choc+2)*M*M+top*M; op3[CHIP] = i_out+(img*OC4+choc+3)*M*M+top*M;
              ot0[CHIP] = i_out+(img*OC4+choc+0)*M*M+top*M; ot1[CHIP] = i_out+(img*OC4+choc+1)*M*M+top*M; ot2[CHIP] = i_out+(img*OC4+choc+2)*M*M+top*M; ot3[CHIP] = i_out+(img*OC4+choc+3)*M*M+top*M;
            }

#define cnn5x5_core1(b, o, bp1, n) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp00[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp01[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp02[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp03[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip00[n], iofs, MSK_W1, (Ull)it00, IMlen, 0, 0, (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn5x5_final(b, bp1) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen)

//EMAX5A begin cnn5x5 mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
        /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                      /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,    &rofs, rofs,            EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &iofs, rofs,            EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,    &oofs, rofs,            EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                  /****in0*****/
                  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp00[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp01[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp02[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp03[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 10KB */
                  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip00[0],   iofs, MSK_W1, (Ull)it00, IMlen, 0, 0, (Ull)NULL, IMlen);    /* stage#2 10KB */
                  exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		  cnn5x5_core1( 3, 4LL, 4, 1);
		  cnn5x5_core1( 4, 8LL, 5, 2);
		  cnn5x5_core1( 5,12LL, 6, 3);
		  cnn5x5_core1( 6,16LL, 7, 4);
		  cnn5x5_core1( 7,20LL, 8, 5);
		  cnn5x5_core1( 8,24LL, 9, 6);
		  cnn5x5_core1( 9,28LL,10, 7);
		  cnn5x5_core1(10,32LL,11, 8);
		  cnn5x5_core1(11,36LL,12, 9);
		  cnn5x5_core1(12,40LL,13,10);
		  cnn5x5_core1(13,44LL,14,11);
		  cnn5x5_core1(14,48LL,15,12);
		  cnn5x5_core1(15,52LL,16,13);
		  cnn5x5_core1(16,56LL,17,14);
		  cnn5x5_core1(17,60LL,18,15);
		  cnn5x5_core1(18,64LL,19,16);
		  cnn5x5_core1(19,68LL,20,17);
		  cnn5x5_core1(20,72LL,21,18);
		  cnn5x5_core1(21,76LL,22,19);
		  cnn5x5_core1(22,80LL,23,20);
		  cnn5x5_core1(23,84LL,24,21);
		  cnn5x5_core1(24,88LL,25,22);
		  cnn5x5_core1(25,92LL,26,23);
		  cnn5x5_core1(26,96LL,27,24);
                  /****final*****/
		  cnn5x5_final(27,     28);
                }
              }
            }
//EMAX5A end
            if (Force) Force = 0;
          }
        }
      }
    }
//EMAX5A drain_dirty_lmm
#ifdef XMAX_VALIDATE
    count = 0;
    for (img=0; img<BATCH; img++) {
      for (oc=0; oc<OC; oc++) {
	for (rofs=0; rofs<M; rofs++) {
	  for (cofs=0; cofs<M; cofs++) {
	    float host = *(float*)&out0[((img*OC+oc)*M+rofs)*M+cofs];
	    float xmax = *(float*)&i_out[((img*OC4+oc)*M+rofs)*M+cofs];
	    if (udiff(host,xmax)>ERRTH) {
	      count++;
	      printf("K=5:img%02.2d.oc%02.2d.%02.2d.%02.2d out0=%7.5e(%8.8x) i_out=%7.5e(%08.8x)\n", (Uint)img, oc, (Uint)rofs, (Uint)cofs, host, *(Uint*)&host, xmax, *(Uint*)&xmax);
	    }
	  }
	}
      }
    }
    if (count)
      printf("Num of diffs: %d\n", count);
#endif
    xmax_cpyout(0, out0, BATCH, OC, i_out, M, M, OC4);
    break;
  case 3:
    RMGRP = M; /* RMGRP = 14 /*{14,11,3,14,16,2}*/
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ★★★ PBL1-1 ★★★ */
/* NCHIP  4 ★★★ PBL1-1 ★★★ */
#define IMAP  1
#define W     4
#define NCHIP 1
#undef XMAX_VALIDATE
//#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
    for (img=0; img<BATCH; img++) {
      for (top=0; top<M; top+=RMGRP) {
        for (iset=0; iset<IC; iset+=IMAP) { /* accumulate multiple sets of IC */
          for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */                                            /* ICHをなるべく温存し,外側LOOPで全OCHを先に片付ける */
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */ /* ICHをなるべく温存し,複数CHIPで全OCHを先に片付ける */
        /*2*/ for (rofs=0; rofs<RMGRP&&(top+rofs)<M; rofs++) { /* image loop (row) */                          /* 1.全ICH複数行   */
          /*1*/ for (cofs=0; cofs<M; cofs++) { /* image loop (cofs) */                                         /* 2.全ICH水平方向 */
                  iofs = rofs*IM+cofs;
                  oofs = rofs*M+cofs;
                  for (w=0; w<W&&(oc+w)<OC/NCHIP; w++) { /* set output channel */                              /* ICHをなるべく温存し,４列使用で全OCHを先に片付ける */
                    op = &out0[(img*OC+CHIP*OC/NCHIP+oc+w)*M*M+top*M+oofs]; /* top of output */
                    for (ic=0; ic<IMAP&&(iset+ic)<IC; ic++) { /* set offset of input channel */
                      ip0  = &in0[(img*IC+iset+ic)*IM*IM+pad*IM+pad]; /* top of input */
                      kp   = &ker[((CHIP*OC/NCHIP+oc+w)*IC+iset+ic)*K*K];
                      kidx = 0;
                      for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
                        for (x=-(K/2); x<K-(K/2); x++) {
                          float in = (0 <= top+rofs+y+pad && top+rofs+y+pad < IM
                                   && 0 <=     cofs+x+pad &&     cofs+x+pad < IM)
                            ? *(float*)&ip0[top*IM+iofs+y*IM+x] : 0.0;
                          if (iset == 0 && ic == 0 && kidx == 0)
                            *(float*)op  = in * *(float*)&kp[kidx];
                          else
                            *(float*)op += in * *(float*)&kp[kidx];
                          kidx++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
#endif
    /*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6, 32,2}*/
    /*                  AAAAAAAAAAAAAAA                                  */
    xmax_cpyin(0, i_inp, &IM, in0, BATCH, IC, IM, M, K); /* この時点で0.0のPADを追加できる */
    xmax_cpyin(0, i_ker, &K,  ker, IC,    OC,  K, K, 1); /* 出力のみ回収すればよい */
    xmax_bzero(i_out, BATCH*OC4*M*M);
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlen = IM*(RMGRP+2);
    Mlen  = M*RMGRP;
    Force = 1;

    if (Klen > 65536/4/2 || IMlen > 65536/4/2 || Mlen > 65536/4/4)
      printf("   CNN3x3  Klen=%dB IMlen=%dB Mlen*4=%dB\n", (Uint)Klen*4, (Uint)IMlen*4, (Uint)Mlen*4*4);

    for (img=0; img<BATCH; img++) {
      for (top=0; top<M; top+=RMGRP) {
        for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
          Uint *ip0  = &i_inp[(img*IC+iset+0)*IM*IM]; /* top of input#0 */
          Uint *it00 = ip0+top*IM, *ip00[9];
          ip00[0] = ip0+(top+0)*IM+0; ip00[1] = ip0+(top+0)*IM+1; ip00[2] = ip0+(top+0)*IM+2;
	  ip00[3] = ip0+(top+1)*IM+0; ip00[4] = ip0+(top+1)*IM+1; ip00[5] = ip0+(top+1)*IM+2;
	  ip00[6] = ip0+(top+2)*IM+0; ip00[7] = ip0+(top+2)*IM+1; ip00[8] = ip0+(top+2)*IM+2;

          for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
            Uint *kp00[NCHIP],*kp01[NCHIP],*kp02[NCHIP],*kp03[NCHIP];
            Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
            Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
              Uint choc  = CHIP*OC4/NCHIP+oc;
              kp00[CHIP] = (choc+0<OC) ? i_ker+((choc+0)*IC+iset+0)*K*K : i_ker;
	      kp01[CHIP] = (choc+1<OC) ? i_ker+((choc+1)*IC+iset+0)*K*K : i_ker;
	      kp02[CHIP] = (choc+2<OC) ? i_ker+((choc+2)*IC+iset+0)*K*K : i_ker;
	      kp03[CHIP] = (choc+3<OC) ? i_ker+((choc+3)*IC+iset+0)*K*K : i_ker;
              op0[CHIP] = i_out+(img*OC4+choc+0)*M*M+top*M; op1[CHIP] = i_out+(img*OC4+choc+1)*M*M+top*M; op2[CHIP] = i_out+(img*OC4+choc+2)*M*M+top*M; op3[CHIP] = i_out+(img*OC4+choc+3)*M*M+top*M;
              ot0[CHIP] = i_out+(img*OC4+choc+0)*M*M+top*M; ot1[CHIP] = i_out+(img*OC4+choc+1)*M*M+top*M; ot2[CHIP] = i_out+(img*OC4+choc+2)*M*M+top*M; ot3[CHIP] = i_out+(img*OC4+choc+3)*M*M+top*M;
            }

#define cnn3x3_core1(b, o, bp1, n) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp00[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp01[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp02[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp03[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip00[n], iofs, MSK_W1, (Ull)it00, IMlen, 0, 0, (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn3x3_final(b, bp1) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen)

//EMAX5A begin cnn3x3 mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
        /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                      /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,    &rofs, rofs,            EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &iofs, rofs,            EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,    &oofs, rofs,            EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                  /****in0*****/
                  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp00[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp01[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp02[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp03[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 10KB */
                  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip00[0],   iofs, MSK_W1, (Ull)it00, IMlen, 0, 0, (Ull)NULL, IMlen);    /* stage#2 10KB */
                  exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		  cnn3x3_core1( 3, 4LL, 4, 1);
		  cnn3x3_core1( 4, 8LL, 5, 2);
		  cnn3x3_core1( 5,12LL, 6, 3);
		  cnn3x3_core1( 6,16LL, 7, 4);
		  cnn3x3_core1( 7,20LL, 8, 5);
		  cnn3x3_core1( 8,24LL, 9, 6);
		  cnn3x3_core1( 9,28LL,10, 7);
		  cnn3x3_core1(10,32LL,11, 8);
                  /****final*****/
		  cnn3x3_final(11,     12);
                }
              }
            }
//EMAX5A end
            if (Force) Force = 0;
          }
        }
      }
    }
//EMAX5A drain_dirty_lmm
#ifdef XMAX_VALIDATE
    count = 0;
    for (img=0; img<BATCH; img++) {
      for (oc=0; oc<OC; oc++) {
	for (rofs=0; rofs<M; rofs++) {
	  for (cofs=0; cofs<M; cofs++) {
	    float host = *(float*)&out0[((img*OC+oc)*M+rofs)*M+cofs];
	    float xmax = *(float*)&i_out[((img*OC4+oc)*M+rofs)*M+cofs];
	    if (udiff(host,xmax)>ERRTH) {
	      count++;
	      printf("K=3:img%02.2d.oc%02.2d.%02.2d.%02.2d out0=%7.5e i_out=%7.5e\n", (Uint)img, oc, (Uint)rofs, (Uint)cofs, host, xmax);
	    }
	  }
	}
      }
    }
    if (count)
      printf("Num of diffs: %d\n", count);
#endif
    xmax_cpyout(0, out0, BATCH, OC, i_out, M, M, OC4);
    break;
  case 2:
    RMGRP = 1; /* RMGRP = 7 /*{7,16,2,7,32,1}*/
               /* RMGRP = 6 /*{7,32,2,6,32,2}*/
    /* CIFAR IMAGE                                           */
    /* +--------------------+------------                    */
    /* |W*W(R) W*W(G) W*W(B)|..batch_size                    */
    /* +--------------------+------------                    */
    /*                                                       */
    /* GPU ORIGINAL                                          */
    /*     +-2*2*ich-+  +-100*6*6-----+        +-100*6*6---+ */
    /* och0|         |  |2            |    och0|           | */
    /* och1|         |  |* 7*7をunpack| -> och1|           | */
    /* och2|         |  |2            |    och2|           | */
    /*     +---------+  |*ich         |        +-----------+ */
    /*                  +-------------+                      */
    /* EMAX ORIGINAL                                         *//*         burst_exe 6*6    ||         burst_exe 6*6    */
    /*     +-----+  +----+-+----+---------+    +-----------+ *//* 7*8... | 7*8... | 7*8... || 7*8... | 7*8... | 7*8... */
    /* unit|2    |  |7*7 | |7*7 |*IC  *100|    |2          | *//*-- -- --                  ||-- -- --                  *//* LMM=7*8*4B */
    /*  |  |*    |  |ch0 | |ch1 |         | -> |*          | *//*         -- -- --         ||         -- -- --         *//*    =244B   */
    /*  V  |2    |  +----+ +----+         |    |2          | *//*                  -- -- --||                  -- -- --*/
    /*     |*ich |  |loop=RMGRP(6)*M(6)   |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*6*4B*4och */
    /*     +-och-+  +---------------------+    +6*6*och----+ *//* img0     img0     img0   || img1     img1     img1   *//*    =576B        */
    /*        32 ... lmf+lmx毎回DMA            |    32/4   | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    /* EMAX for small IM/M                                    */
    /*     +-----+  +---------------------+    +-----------+ *//*         burst_exe 6*100                              */
    /* unit|     |  |+----PAD----+        |    |           | *//* 7*2... | 7*2... | 7*2... || 7*2... | 7*2... | 7*2..  */
    /*  |  |2    |  ||7*7 | |7*7 |*IC *100| -> |2          | *//* -                          -                         *//* LMM=7*8*4B*32ch*100 */
    /*  |  |*    |  ||ch0 | |ch1 |        |    |*          | *//*          -                          -                *//*    =716800B         */
    /*  V  |2    |  |+----+ +----+        |    |2          | *//*                   -                          -       */
    /*     |*ich |  |loop=M(6)*BATCH(100) |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*6*4B*100*4och */
    /*     +-och-+  +---------------------+    +6*100*och--+ *//* img0     img0     img0   || img1     img1     img1   *//*    =57600B          */
    /*        32 ... lmf+lmx毎回DMA            |      32/4 | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    /* EMAX for small IM/M                                   */
    /*     +-----+  +---------------------+    +-----------+ *//*         burst_exe 6*100  ||         burst_exe 6*100  *//* 100画像を1枚(7*700pix)に(7*100を7行) */
    /* unit|     |  |+----PAD----+        |    |           | *//* 7*8*100| 7*8*100| 7*8*100|| 7*8*100| 7*8*100| 7*8*100*//* または7*7連続アドレスを100セット     */
    /*  |  |2    |  ||7*7 | |7*7 |*100 *IC| -> |2          | *//*-- -- --                    -- -- --                  *//* LMM=7*8*4B*100 LMMstg2-7にload       */
    /*  |  |*    |  ||im0 | |im1 |        |    |*          | *//* top=0   -- -- --            top=1   -- -- --         *//*    =22400B(RMGRP=7で2回再利用)<32KB  */
    /*  V  |2    |  |+----+ +----+        |    |2          | *//*                  -- -- --                    -- -- --*/
    /*     |*ich |  |loop=M(6)*BATCH(100) |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*4B*100*4och */
    /*     +-och-+  +---------------------+    +6*100*och--+ *//* img0-99  img0-99  img0-99|| img0-99  img0-99  img0-99*//*    =9600B         */
    /*        32 ... lmf+lmx毎回DMA            |      32/4 | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ★★★ PBL1-1 ★★★ */
/* NCHIP  4 ★★★ PBL1-1 ★★★ */
#define IMAP  1
#define W     4
#define NCHIP 1
#undef XMAX_VALIDATE
//#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
    for (top=0; top<M; top+=RMGRP) {
      for (iset=0; iset<IC; iset+=IMAP) { /* accumulate multiple sets of IC */
/**/    for (rofs=0; rofs<RMGRP&&(top+rofs)<M; rofs++) { /* image loop (row) */                                /* 1.全ICH複数行   */
          for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */                                            /* ICHをなるべく温存し,外側LOOPで全OCHを先に片付ける */
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */ /* ICHをなるべく温存し,複数CHIPで全OCHを先に片付ける */
/**/    /*2*/ for (img=0; img<BATCH; img++) {
          /*1*/ for (cofs=0; cofs<M; cofs++) { /* image loop (cofs) */                                         /* 2.全ICH水平方向 */
                  iofs = rofs*IM+cofs;
                  oofs = rofs*M+cofs;
                  for (w=0; w<W&&(oc+w)<OC/NCHIP; w++) { /* set output channel */                              /* ICHをなるべく温存し,４列使用で全OCHを先に片付ける */
                    op = &out0[(img*OC+CHIP*OC/NCHIP+oc+w)*M*M+top*M+oofs]; /* top of output */
                    for (ic=0; ic<IMAP&&(iset+ic)<IC; ic++) { /* set offset of input channel */
                      ip0  = &in0[(img*IC+iset+ic)*IM*IM+pad*IM+pad]; /* top of input */
                      kp   = &ker[((CHIP*OC/NCHIP+oc+w)*IC+iset+ic)*K*K];
                      kidx = 0;
                      for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
                        for (x=-(K/2); x<K-(K/2); x++) {
                          float in = (0 <= top+rofs+y+pad && top+rofs+y+pad < IM
                                   && 0 <=     cofs+x+pad &&     cofs+x+pad < IM)
                            ? *(float*)&ip0[top*IM+iofs+y*IM+x] : 0.0;
                          if (iset == 0 && ic == 0 && kidx == 0)
                            *(float*)op  = in * *(float*)&kp[kidx];
                          else
                            *(float*)op += in * *(float*)&kp[kidx];
                          kidx++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
#endif
    /*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6,32,2}*/
    /*                                    AAAAAAAAAAAAA   AAAAAAAAAAAAA */
    xmax_cpyin(1, i_inp, &IM, in0, BATCH, IC, IM, M, K); /* この時点で0.0のPADを追加できる */
    xmax_cpyin(0, i_ker, &K,  ker, IC,    OC,  K, K, 1); /* 出力のみ回収すればよい */
    xmax_bzero(i_out, BATCH*OC4*M*M);
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlen = IM*BATCH*(RMGRP+1);
    Mlen  = M*BATCH*RMGRP;
    Force = 1;

    if (Klen > 65536/4/2 || IMlen > 65536/4/2 || Mlen > 65536/4/4)
      printf("   CNN2x2  Klen=%dB IMlen=%dB Mlen*4=%dB\n", (Uint)Klen*4, (Uint)IMlen*4, (Uint)Mlen*4*4);

    for (top=0; top<M; top+=RMGRP) {
      for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
	Uint *ip0  = &i_inp[iset*IM*BATCH*IM]; /* top of input#0 */
	Uint *it00 = ip0+top*IM*BATCH, *ip00[4];
	ip00[0] = ip0+(top+0)*IM*BATCH+0; ip00[1] = ip0+(top+0)*IM*BATCH+1;
	ip00[2] = ip0+(top+1)*IM*BATCH+0; ip00[3] = ip0+(top+1)*IM*BATCH+1;

        for (rofs=0; rofs<RMGRP&&(top+rofs)<M; rofs++) { /* image loop (row) */

	  for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
	    Uint *kp00[NCHIP],*kp01[NCHIP],*kp02[NCHIP],*kp03[NCHIP];
	    Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
	    Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
              Uint choc  = CHIP*OC4/NCHIP+oc;
              kp00[CHIP] = (choc+0<OC) ? i_ker+((choc+0)*IC+iset+0)*K*K : i_ker;
	      kp01[CHIP] = (choc+1<OC) ? i_ker+((choc+1)*IC+iset+0)*K*K : i_ker;
	      kp02[CHIP] = (choc+2<OC) ? i_ker+((choc+2)*IC+iset+0)*K*K : i_ker;
	      kp03[CHIP] = (choc+3<OC) ? i_ker+((choc+3)*IC+iset+0)*K*K : i_ker;
              op0[CHIP] = i_out+((choc+0)*M+top)*M*BATCH; op1[CHIP] = i_out+((choc+1)*M+top)*M*BATCH; op2[CHIP] = i_out+((choc+2)*M+top)*M*BATCH; op3[CHIP] = i_out+((choc+3)*M+top)*M*BATCH;
              ot0[CHIP] = i_out+((choc+0)*M+top)*M*BATCH; ot1[CHIP] = i_out+((choc+1)*M+top)*M*BATCH; ot2[CHIP] = i_out+((choc+2)*M+top)*M*BATCH; ot3[CHIP] = i_out+((choc+3)*M+top)*M*BATCH;
            }

#define cnn2x2_core1(b, o, bp1, n) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp00[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp01[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp02[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp03[CHIP], o, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip00[n], iofs, MSK_W1, (Ull)it00, IMlen, 0, 0, (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn2x2_final(b, bp1) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, 1, (Ull)NULL, Mlen)

//EMAX5A begin cnn2x2 mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
	/*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                       /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,    &img,  img,             EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &iofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,    &oofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                  /****in0*****/
                  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp00[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp01[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp02[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp03[CHIP], 0LL, MSK_D0, (Ull)i_ker, Klen, 0, Force, (Ull)NULL, Klen); /* stage#2 10KB */
                  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip00[0],   iofs, MSK_W1, (Ull)it00, IMlen, 0, 0, (Ull)NULL, IMlen);    /* stage#2 10KB */
                  exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		  cnn2x2_core1( 3, 4LL, 4, 1);
		  cnn2x2_core1( 4, 8LL, 5, 2);
		  cnn2x2_core1( 5,12LL, 6, 3);
                  /****final*****/
		  cnn2x2_final( 6,      7);
                }
              }
            }
//EMAX5A end
            if (Force) Force = 0;
          }
        }
      }
    }
//EMAX5A drain_dirty_lmm
#ifdef XMAX_VALIDATE
    count = 0;
    for (img=0; img<BATCH; img++) {
      for (oc=0; oc<OC; oc++) {
	for (rofs=0; rofs<M; rofs++) {
	  for (cofs=0; cofs<M; cofs++) {
	    float host = *(float*)&out0[((img*OC+oc)*M+rofs)*M+cofs];
	    float xmax = *(float*)&i_out[((oc*M+rofs)*BATCH+img)*M+cofs];
	    if (udiff(host,xmax)>ERRTH) {
	      count++;
	      printf("K=2:img%02.2d.oc%02.2d.%02.2d.%02.2d out0=%7.5e i_out=%7.5e\n", (Uint)img, oc, (Uint)rofs, (Uint)cofs, host, xmax);
	    }
	  }
	}
      }
    }
    if (count)
      printf("Num of diffs: %d\n", count);
#endif
    xmax_cpyout(1, out0, BATCH, OC, i_out, M, M, OC4);
    break;
  }
}

void xmax_sgemm00(int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  /*  ┌─────┐convolutionの場合                                                  */
  /*  │┌────┴┐Bが複数と考える                                                  */
  /*  ││┌────┴┐┌─────┐┐        ┌─────┐┐                       */
  /*  │││b         ││a a a a a ││RMGRP   │o o o o o ││RMGRP                  */
  /*  │││b         ┤│          │┤/CHIP   │          │┤/CHIP                  */
  /*  │││b   B0   b││ A(weight)││        │   out    ││ mmの場合は行で分割    */
  /*  └││b        l┤│          │┤        │          │┤ cnnの場合はoutで分割  */
  /*    └│b        k││blk       ││        │blk       ││                       */
  /*      └─────┘└─┴─┴─┘┘        └─┴─┴─┘┘                       */

  int  RMGRP, Alen, Blen, Clen;
  int  row, col, k;
  int  count, top, blk;
  Ull  KA4, N, n4, KA4n4;
  Ull  CHIP, rofs, cofs, oofs;
  Ull  cofslimit1, cofslimit2, cofslimit3;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1, ex2, ex3;

#undef  IMAP
#undef  W
#undef  H
#undef  NCHIP
#define IMAP  1
#define W     4LL
#define H     48
/* NCHIP  4 ★★★ nakashima ★★★ */
#define NCHIP 1
#undef XMAX_VALIDATE
//#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
#if defined(CBLAS_GEMM)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, ka, 1.0f, A, ka, B, n, 0.0f, C, n);
#else
  for (row=0; row<m; row++) {
    for (col=0; col<n; col++) {
      for (k=0; k<ka; k++) {
	if (k==0) C[row*n+col]  = A[row*ka+k] * B[k*n+col];
	else      C[row*n+col] += A[row*ka+k] * B[k*n+col];
      }
    }
  }
#endif
#endif
  N = (n+3)&~3;
  xmax_cpyin(2, i_m0A, &m, A, 1, 1, m, ka, 1);
  xmax_cpyin(2, i_m0B, &n, B, 1, 1, n, ka, 1);
  xmax_bzero(i_m0C, m*n); /* m*N */
  /*  m=100/NCHIP(4)を割り切れる値として,RMGRP=5              */
  /* xsim/xsim-zynq.emax6+dma -x -t -i1 -C4 -F1の場合 RMGRP=5 */
  /*  ka=288,288*RMGRP*4=5KB(<32KB)となりLMMに入る            */
  /* xsim/xsim-zynq.emax6+dma -x -t -i0 -C1 -F1の場合 RMGRP=5 */
  /*  ich=9, ka=1296,1296*RMGRP(5)*4=26KB(<32KB)となりrsimはLMMに入る     */
  /*  ich=17,ka=2448,2448*RMGRP(5)*4=49KB(>32KB)となりssimはLMMに入らない */
  /*  NCHIP=1なら 100を割れる数でよいのでRMGRP=2                          */
  /*  ich=17,ka=2448,2448*RMGRP(2)*4=20KB(<32KB)となりssimもLMMに入る     */
  RMGRP = 8192/ka>=5?5:2;/* CIFAR10:6KB,MNIST:20KB */
  Alen  = ka*RMGRP;      /* 288*5*4B  = 5760B    */
  Blen  = n;             /* 10/2      = 5        */
  Clen  = n*RMGRP;       /* 10*5*4B   = 200B     */
  KA4   = ka*4;          /* 288*4B               */
  n4    = n*4;           /* 10*4B                */
  KA4n4 = KA4<<32|n4;

  if (Blen > 65536/4/2 || Alen > 65536/4/2 || Clen > 65536/4)
    printf("   GEMM00  Blen=%dB Alen=%dB Clen=%dB\n", (Uint)Blen*4, (Uint)Alen*4, (Uint)Clen*4);

  for (top=0; top<m/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    for (blk=0; blk<ka; blk+=H) { /* 3重ループ展開の外側対象 */
      typedef struct {Uint i[4]} Ui4;
      Uint *a0[NCHIP];
      Uint *a[H][NCHIP];
      Ui4  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui4  *c0[NCHIP];
      Ui4  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
      for (k=0; k<H; k++) {
	b[k] = i_m0B+(blk+k)*n; b0[k] = b[k]; b1[k] = (Uint*)b[k]+1; b2[k] = (Uint*)b[k]+2;  b3[k] = (Uint*)b[k]+3; 
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	a0[CHIP] = i_m0A+(CHIP*m/NCHIP+top)*ka;
	for (k=0; k<H; k++)
	  a[k][CHIP] = a0[CHIP]+blk+k;
	c0[CHIP] = i_m0C+(CHIP*m/NCHIP+top)*n;
	c00[CHIP]= (Uint*)c0[CHIP]+0; c01[CHIP]= (Uint*)c0[CHIP]+1; c02[CHIP]= (Uint*)c0[CHIP]+2; c03[CHIP]= (Uint*)c0[CHIP]+3;
      }
      cofslimit1 = n4- 4; /* cofs32 < 36 x */
      cofslimit2 = n4- 8; /* cofs32 < 32 x */
      cofslimit3 = n4-12; /* cofs32 < 28 x */

#define sgemm00_core1(r, rm1, rp1) \
	    mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][2][1],  (Ull)a[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);\
	    exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_final(r, rp1) \
	    exe(OP_CMP_LT,   &cc1, cofs, EXP_H3210, cofslimit1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_CMP_LT,   &cc2, cofs, EXP_H3210, cofslimit2, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_CMP_LT,   &cc3, cofs, EXP_H3210, cofslimit3, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_LDWR,   1, &BR[rp1][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, 1, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, 1, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, 1, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, 1, (Ull)NULL, Clen);\
	    exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_STWR,   1, &AR[rp1][0],     (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, 1, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex1,   0, 0, 0, cc1, 0xaaaa);\
	    mop(OP_STWR, ex1, &AR[rp1][1],     (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, 1, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex2,   0, 0, 0, cc2, 0xaaaa);\
	    mop(OP_STWR, ex2, &AR[rp1][2],     (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, 1, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex3,   0, 0, 0, cc3, 0xaaaa);\
	    mop(OP_STWR, ex3, &AR[rp1][3],     (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, 1, (Ull)NULL, Clen)

//EMAX5A begin sgemm00 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-KA4)<<32|((0-n4)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=N/W,cofs=(0-W*4)<<32|((0-W*4)&0xffffffff); LOOP0--; INIT0=0) {  /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*4)<<32|(W*4), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
	    exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?KA4n4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* stage#0 */
	    exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);           /* stage#1 */

	    mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 2KB */
	    mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)a[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);/* stage#1 16KB */
	    exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */

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
	    sgemm00_core1(12, 11, 13);
	    sgemm00_core1(13, 12, 14);
	    sgemm00_core1(14, 13, 15);
	    sgemm00_core1(15, 14, 16);
	    sgemm00_core1(16, 15, 17);
	    sgemm00_core1(17, 16, 18);
	    sgemm00_core1(18, 17, 19);
	    sgemm00_core1(19, 18, 20);
	    sgemm00_core1(20, 19, 21);
	    sgemm00_core1(21, 20, 22);
	    sgemm00_core1(22, 21, 23);
	    sgemm00_core1(23, 22, 24);
	    sgemm00_core1(24, 23, 25);
	    sgemm00_core1(25, 24, 26);
	    sgemm00_core1(26, 25, 27);
	    sgemm00_core1(27, 26, 28);
	    sgemm00_core1(28, 27, 29);
	    sgemm00_core1(29, 28, 30);
	    sgemm00_core1(30, 29, 31);
	    sgemm00_core1(31, 30, 32);
	    sgemm00_core1(32, 31, 33);
	    sgemm00_core1(33, 32, 34);
	    sgemm00_core1(34, 33, 35);
	    sgemm00_core1(35, 34, 36);
	    sgemm00_core1(36, 35, 37);
	    sgemm00_core1(37, 36, 38);
	    sgemm00_core1(38, 37, 39);
	    sgemm00_core1(39, 38, 40);
	    sgemm00_core1(40, 39, 41);
	    sgemm00_core1(41, 40, 42);
	    sgemm00_core1(42, 41, 43);
	    sgemm00_core1(43, 42, 44);
	    sgemm00_core1(44, 43, 45);
	    sgemm00_core1(45, 44, 46);
	    sgemm00_core1(46, 45, 47);
	    sgemm00_core1(47, 46, 48);
	    sgemm00_core1(48, 47, 49); /* 288/6 H=48 */
	    /****final*****/
	    sgemm00_final(49,     51);
          }
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
#ifdef XMAX_VALIDATE
  count = 0;
  for (row=0; row<m; row++) {
    for (col=0; col<n; col++) {
      if (udiff(C[row*n+col],*(float*)&i_m0C[row*n+col])>ERRTH) {
	count++;
	printf("[%d][%d]: C=%7.5e i_m0C=%7.5e\n", row, col, C[row*n+col], *(float*)&i_m0C[row*n+col]);
      }
    }
  }
  if (count)
    printf("Num of diffs: %d\n", count);
#endif
  xmax_cpyout(2, C, 1, 1, i_m0C, m, n, n); /* i_m0C is contiguous w/ CEX+ST */
}

void xmax_sgemm10(int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  int row, col, k;

#undef XMAX_VALIDATE
#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
#if defined(CBLAS_GEMM)
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, ka, 1.0f, A, m, B, n, 0.0f, C, n);
#else
  for (k=0; k<ka; k++) {
    for (row=0; row<m; row++) {
      for (col=0; col<n; col++) {
	if (k==0) C[row*n+col]  = A[k*m+row] * B[k*n+col];
	else      C[row*n+col] += A[k*m+row] * B[k*n+col];
      }
    }
  }
#endif
#endif

  /* ★★★ PBL1-2 ★★★ */

#ifdef XMAX_VALIDATE
#endif
}

void xmax_sgemm01(int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  int row, col, k;

#undef XMAX_VALIDATE
#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
#if defined(CBLAS_GEMM)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, ka, 1.0f, A, ka, B, ka, 0.0f, C, n);
#else
  for (row=0; row<m; row++) {
    for (col=0; col<n; col++) {
      for (k=0; k<ka; k++) {
	if (k==0) C[row*n+col]  = A[row*ka+k] * B[col*ka+k];
	else      C[row*n+col] += A[row*ka+k] * B[col*ka+k];
      }
    }
  }
#endif
#endif

  /* ★★★ PBL1-3 ★★★ */

#ifdef XMAX_VALIDATE
#endif
}

void xmax_conv_backward(float4D *out, float2D *kernel, float2D *g_kernel, float4D *in, int ksize)
{
  int   kstride = 1;
  int   BATCH  = in->nstrides;  //100
  int   IC     = in->nchannel;  //3
  int   IM     = in->kstrides;  //28
  int   IMX;
  int   OC     = out->nchannel; //8
  int   M      = out->kstrides; //24
  int   K      = ksize;         // 5,4,3,2,1
  Uint  *in0   = in->data;      // IC*IM*IM
  Uint  *ker   = kernel->data;  // OC*IC*K*K
  Uint  *g_ker = g_kernel->data;// OC*IC*K*K
  Uint  *out0  = out->data;     // OC*M*M
  Uint  *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *op0, *kp, kidx, *kp0;
  int   pad;
  int   count, top, iset, oset, oc, w, ic, y, x;
  int   y0, x0, ch, xy;
  Ull   IMX4, IM4, M4, IMX4M4, M4IM4, IMXlen, IMlen, Mlen;
  Ull   CHIP, img, rofs, cofs, iofs, oofs, b00, c00;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;

  /*  unpack_patch2col(tmp_col, in, ksize, kstride, M, M); */
  /*  for (oc=0;oc<OC;oc++) {                              */
  /*    for (img=0;img<BATCH;img++)                        */
  /*      memcpy(&(tmp_dst->data[(oc*BATCH+img)*M*M]), &(out->data[(oc+img*OC)*M*M]), M*M*sizeof(float));*/
  /*  }                                                    */
  /*  multiply_float2D(g_kernel, tmp_dst, 0, tmp_col, 1); // 8x25 dot 25x57600 --> 8x57600 */
  /*  multiply_float2D(tmp_col, kernel, 1, tmp_dst, 0);    */
  /*  pack_col2patch(in, tmp_col, ksize, kstride, M, M);   */

  if (IM == M)
    pad = 0;   /* PAD無し.in周囲0.0を仮定 */
  else if ((IM - K)/1 + 1 == M)
    pad = K/2; /* PAD有り.in周囲特別扱い不要 */
  else {
    printf("xmax_conv_backward error: IM=%d K=%d M=%d\n", IM, K, M);
    printf("IM == M || (IM-K)/1+1 == M\n");
    exit(-1);
  }

  /*================================================================================================*/
  /*=== back_g_ker =================================================================================*/
  /*================================================================================================*/

#undef  IMAP
#undef  OMAP
#undef  W
#undef  NCHIP
#define IMAP  4
#define OMAP  8
#define W     1
#define NCHIP 2

#undef XMAX_VALIDATE
//#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
  /* g_kernel <- out, in */
  for (img=0; img<BATCH; img++) {
    for (ic=0; ic<IC; ic++) { /* set offset of input channel */
      for (oc=0; oc<OC; oc++) { /* set output channel */
	op0 = &out0[(img*OC+oc)*M*M]; /* top of output image */
	kp = &g_ker[(oc*IC+ic)*K*K];
	kidx = 0;
	for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
	  for (x=-(K/2); x<K-(K/2); x++) {
	    ip0 = &in0[(img*IC+ic)*IM*IM+(y+pad)*IM+(x+pad)]; /* top of input */
	    kp0 = kp+kidx;
	    for (rofs=0; rofs<M; rofs++) {
	      for (cofs=0; cofs<M; cofs++) { /* image loop (cofs) */
		float out = *(float*)&op0[rofs*M+cofs];
		float in  = (0 <= rofs+y+pad && rofs+y+pad < IM
			  && 0 <= cofs+x+pad && cofs+x+pad < IM)
		  ? *(float*)&ip0[rofs*IM+cofs] : 0.0;
		if (img==0 && rofs==0 && cofs==0) *(float*)kp0  = out * in;
		else                              *(float*)kp0 += out * in;
              }
            }
	    kidx++;
          }
        }
      }
    }
  }
#endif
//efine PBL1_4_VERSION0
#define PBL1_4_VERSION1
#ifdef  PBL1_4_VERSION0
  xmax_cpyin(0, i_out, &M,  out0, BATCH, OC,  M, M, 1); //imemcpy(i_out, out0, BATCH*OC*M*M);   M=M;
  xmax_cpyin(0, i_inp, &IMX, in0, BATCH, IC, IM, M, K); //imemcpy(i_inp, in0,  BATCH*IC*IM*IM); IMX=IM;
  xmax_bzero(i_ker, OC*IC*K*K); /* g_kernel */
#if 0
  for (img=0; img<BATCH; img++) {
    for (ic=0; ic<IC; ic++) { /* set offset of input channel */
      for (oc=0; oc<OC; oc++) { /* set output channel */
	op0 = &i_out[(img*OC+oc)*M*M]; /* top of output image */
	kp = &i_ker[(oc*IC+ic)*K*K];
	kidx = 0;
	for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
	  for (x=-(K/2); x<K-(K/2); x++) {
	    ip0 = &i_inp[(img*IC+ic)*IMX*IMX+(y+K/2)*IMX+(x+K/2)]; /* top of input */
	    kp0 = kp+kidx;
	    for (rofs=0; rofs<M; rofs++) {
	      for (cofs=0; cofs<M; cofs++) { /* image loop (cofs) */
		float in  = *(float*)&ip0[rofs*IMX+cofs];
		float out = *(float*)&op0[rofs*M+cofs];
		*(float*)kp0 += out * in;
              }
            }
	    kidx++;
          }
        }
      }
    }
  }
#else
  IMX4   = IMX*4;
  M4     = M*4;
  IMX4M4 = IMX4<<32|M4;
  IMXlen = IMX*IMX;
  Mlen   = M*M;
#if 0
printf("   BACK00  IMXlen=%dB Mlen=%dB\n", (Uint)IMXlen*4, (Uint)Mlen*4);
#endif
  for (img=0; img<BATCH; img++) {
    for (ic=0; ic<IC; ic++) { /* set offset of input channel */
      for (oc=0; oc<OC; oc++) { /* set output channel */
	op0 = &i_out[(img*OC+oc)*M*M]; /* top of output image */
	kp = &i_ker[(oc*IC+ic)*K*K];
	kidx = 0;
	for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
	  for (x=-(K/2); x<K-(K/2); x++) {
	    ip0 = &i_inp[(img*IC+ic)*IMX*IMX+(y+K/2)*IMX+(x+K/2)]; /* top of input */
	    kp0 = kp+kidx;
	    Uint *it00 = ip0;
	    Uint *ot00 = op0;
//EMAX5A begin back_g_ker mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
	/*2*/ for (INIT1=1,LOOP1=M,rofs=(0-IMX4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                            /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                          /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,      &rofs, rofs,            EXP_H3210, INIT0?IMX4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,      &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,    EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,      &iofs, rofs,            EXP_H3210, cofs,           EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,      &oofs, rofs,            EXP_H3210, cofs,           EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */
                  /****in0*****/
                  mop(OP_LDWR,  1, &BR[2][0][1],           (Ull)ip0,  iofs, MSK_W1,   (Ull)it00, IMXlen, 0, 0,   NULL,     IMXlen);           /* stage#2 */
                  mop(OP_LDWR,  1, &BR[2][2][1],           (Ull)op0,  oofs, MSK_W0,   (Ull)ot00, Mlen,   0, 0,   NULL,     Mlen);             /* stage#2 */
                  exe(OP_FML,      &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1],    EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#3 */
                  /****final*****/
		  exe(OP_NOP,      &AR[4][0], 0LL,         EXP_H3210, 0LL,            EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#4 (dummy to set target location) */
		  mop(OP_LDWR,  1, &b00,                   (Ull)kp0,  0LL,  MSK_W0,   (Ull)kp0,  1LL,    0, 1,   NULL,     1LL);              /* stage#4 */
                  exe(OP_FAD,      &b00,      b00,         EXP_H3210, AR[3][0],       EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#4 */
                  mop(OP_STWR,  1, &b00,                   (Ull)kp0,  0LL,  MSK_D0,   (Ull)kp0,  1LL,    0, 1,   NULL,     1LL);              /* stage#4 */
                }
              }
            }
//EMAX5A end
	    kidx++;
          }
        }
      }
    }
  }
//EMAX5A drain_dirty_lmm
#endif
#ifdef XMAX_VALIDATE
  count = 0;
  for (ic=0; ic<IC; ic++) {
    for (oc=0; oc<OC; oc++) {
      for (y=0; y<K; y++) {
        for (x=0; x<K; x++) {
          float host = *(float*)&g_ker[(oc*IC+ic)*K*K+y*K+x];
          float xmax = *(float*)&i_ker[(oc*IC+ic)*K*K+y*K+x];
          if (udiff(host,xmax)>ERRTH) {
            count++;
            printf("[%d][%d][%d][%d]: g_ker=%7.5e(%8.8x) i_ker=%7.5e(%8.8x)\n", oc, ic, y, x, host, *(Uint*)&host, xmax, *(Uint*)&xmax);
          }
        }
      }
    }
  }
  if (count)
    printf("Num of diffs 1-4: %d\n", count);
#endif
  xmax_cpyout(2, g_ker, 1, 1, i_ker, IC*K*K, OC, OC); /* g_kernel */
#endif
#ifdef  PBL1_4_VERSION1
  /***********************************/
  /* ★★★ PBL1-4 (g_kernel) ★★★ */
  /***********************************/
  xmax_cpyin(1, i_out, &M,  out0, BATCH, OC,  M, M, 1); //dst[OC][M][BATCH][M]     <- src[BATCH][OC][M][M]
  xmax_cpyin(1, i_inp, &IMX, in0, BATCH, IC, IM, M, K); //dst[IC][IMX][BATCH][IMX] <- src[BATCH][IC][IM][IM]
  xmax_bzero(i_ker, OC*IC*K*K); /* g_kernel */
#if 0
  for (oset=0; oset<((OC+OMAP-1)&~(OMAP-1)); oset+=OMAP) { /* set output channel */
    Uint *ip0[IMAP], *op0[OMAP], *kp0[OMAP][IMAP];
    for (rofs=0; rofs<M; rofs++) {
      for (iset=0; iset<((IC+IMAP-1)&~(IMAP-1)); iset+=IMAP) { /* set offset of input channel */
	kidx = 0;
	for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
	  for (x=-(K/2); x<K-(K/2); x++) {
	    for (ic=0; ic<IMAP; ic++)
	      ip0[ic] = (iset+ic)<IC ? &i_inp[(iset+ic)*IMX*BATCH*IMX+(rofs+y+K/2)*BATCH*IMX+(x+K/2)] : 0; /* input */
	    for (oc=0; oc<OMAP; oc++)
	      op0[oc] = (oset+oc)<OC ? &i_out[(oset+oc)*M*BATCH*M+rofs*BATCH*M] : 0; /* output */
	    for (ic=0; ic<IMAP; ic++) {
	      for (oc=0; oc<OMAP; oc++)
		kp0[oc][ic] = ((iset+ic)<IC && (oset+oc)<OC) ? &i_ker[((oset+oc)*IC+iset+ic)*K*K+kidx] : 0; /* NULL skip DMA */
	    }
	    for (ic=0; ic<IMAP&&(iset+ic)<IC; ic++) { /* set output channel */
	      for (oc=0; oc<OMAP&&(oset+oc)<OC; oc++) { /* set output channel */
		for (img=0; img<BATCH; img++) {
		  for (cofs=0; cofs<M; cofs++) { /* image loop (cofs) */
		    float in  = *(float*)&ip0[ic][img*IMX+cofs];
		    float out = *(float*)&op0[oc][img*M+cofs];
		    *(float*)kp0[oc][ic] += out * in;
		  }
		}
	      }
	    }
	    kidx++;
	  }
        }
      }
    }
  }
#else
  IMX4   = IMX*4;
  M4     = M*4;
  IMX4M4 = IMX4<<32|M4;
  IMXlen = IMX*BATCH;
  Mlen   = M*BATCH;

  if (IMXlen > 65536/4/2 || Mlen > 65536/4/4)
    printf("   BACK00  IMXlen=%dB Mlen=%dB\n", (Uint)IMXlen*4, (Uint)Mlen*4);

  /* +----------------------+-----------------------+                     */
  /* |     inp[ic][row]     |out[oc+0][row+yx*]再利用 K行                 */
  /* |                      |ker[oc+0][ic][yx*]     |                     */
  /* +----------------------+-----------------------+                     */
  /* |     inp[ic][row]     |out[oc+1][row+yx*]再利用 K行                 */
  /* |                      |ker[oc+1][ic][yx*]     |                     */
  /* +----------------------+-----------------------+                     */
  /* |     inp[ic][row]     |out[oc+2][row+yx*]再利用 K行                 */
  /* |                      |ker[oc+2][ic][yx*]     |                     */
  /* +----------------------+-----------------------+                     */
  /* |     inp[ic][row]     |out[oc+3][row+yx*]再利用 K行                 */
  /* |                      |ker[oc+3][ic][yx*]     |                     */
  /* +----------------------+-----------------------+                     */
  /*                             oc:stageに展開                           */
  /*                                   ic:最外ループ                      */
  /*                                       y:段数を埋めるほど多くない     */
  /*                                        x:隣接要素は複数LMMに分散不可 */
  for (oset=0; oset<((OC+OMAP-1)&~(OMAP-1)); oset+=OMAP) { /* set output channel */
    Ull  cc0[OMAP][IMAP], cc1[OMAP][IMAP];
    Uint inum[IMAP][NCHIP], *ip0[IMAP][NCHIP], *it0[IMAP][NCHIP];
    Uint onum[OMAP], *op0[OMAP], *ot0[OMAP];
    Uint *kp0[OMAP][IMAP][NCHIP];
    for (iset=0; iset<((IC+IMAP*NCHIP-1)&~(IMAP*NCHIP-1)); iset+=IMAP*NCHIP) { /* set offset of input channel */
      for (rofs=0; rofs<M; rofs++) {
	kidx = 0;
	for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
	  for (x=-(K/2); x<K-(K/2); x++) {
	    for (CHIP=0; CHIP<NCHIP; CHIP++) {
	      for (ic=0; ic<IMAP; ic++) {
		inum[ic][CHIP] = iset+IMAP*CHIP+ic;
		ip0[ic][CHIP]  = (iset+IMAP*CHIP+ic)<IC ? &i_inp[(iset+IMAP*CHIP+ic)*IMX*BATCH*IMX+(rofs+y+K/2)*BATCH*IMX+(x+K/2)] : 0; /* input */
		if (IMX*BATCH*IMX <= 32768/4) {
		  IMXlen = IMX*BATCH*IMX;
		  it0[ic][CHIP] = (iset+IMAP*CHIP+ic)<IC ? &i_inp[(iset+IMAP*CHIP+ic)*IMX*BATCH*IMX                       ] : 0;         /* input */
		}
		else if (IMX*BATCH*K <= 32768/4) {
		  IMXlen = IMX*BATCH*K;
		  it0[ic][CHIP] = (iset+IMAP*CHIP+ic)<IC ? &i_inp[(iset+IMAP*CHIP+ic)*IMX*BATCH*IMX+(rofs      )*BATCH*IMX] : 0;         /* input */
		}
		else
		  it0[ic][CHIP] = (iset+IMAP*CHIP+ic)<IC ? &i_inp[(iset+IMAP*CHIP+ic)*IMX*BATCH*IMX+(rofs+y+K/2)*BATCH*IMX] : 0;         /* input */
	      }
	    }
	    for (oc=0; oc<OMAP; oc++) {
	      onum[oc] = oset+oc;
	      op0[oc]  = (oset+oc)<OC ? &i_out[(oset+oc)*M*BATCH*M+rofs*BATCH*M] : 0; /* output */
	      if (M*BATCH*M <= 16384/4) {
		Mlen = M*BATCH*M;
		ot0[oc] = (oset+oc)<OC ? &i_out[(oset+oc)*M*BATCH*M] : 0; /* output */
	      }
	      else
		ot0[oc] = op0[oc];
	    }
	    for (oc=0; oc<OMAP; oc++) {
	      for (CHIP=0; CHIP<NCHIP; CHIP++) {
		for (ic=0; ic<IMAP; ic++)
		  kp0[oc][ic][CHIP] = ((iset+IMAP*CHIP+ic)<IC && (oset+oc)<OC) ? &i_ker[((oset+oc)*IC+iset+IMAP*CHIP+ic)*K*K+kidx] : 0; /* NULL skip DMA */
	      }
	    }

#define back_g_ker_core1(b, o, i) \
  exe(OP_CMP_LT,   &cc0[o][i],onum[o],       EXP_H3210,            OC,          EXP_H3210, 0LL,                  EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#1 */\
  exe(OP_CMP_LT,   &cc1[o][i],inum[i][CHIP], EXP_H3210,            IC,          EXP_H3210, 0LL,                  EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#1 */\
  mop(OP_LDWR,  1, &BR[b][1][1],             (Ull)op0[o],          oofs,        MSK_W0,    (Ull)ot0[o],          Mlen,      0,      0,   NULL,   Mlen);   /* stage#2 */\
  mop(OP_LDWR,  1, &BR[b][2][1],             (Ull)ip0[i][CHIP],    iofs,        MSK_W1,    (Ull)it0[i][CHIP],    IMXlen,    0,      0,   NULL,   IMXlen); /* stage#2 IMXlenが大きいのでLMM*2使用 */\
  exe(OP_NOP,      &AR[b][0], 0LL,           EXP_H3210,            0LL,         EXP_H3210, 0LL,                  EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#2 (dummy to set target location) */\
  mop(OP_LDWR,  1, &b00,                     (Ull)kp0[o][i][CHIP], 0LL,         MSK_W0,    (Ull)kp0[o][i][CHIP], 1LL,       0,      1,   NULL,   1LL);    /* stage#2 foldはunit[0]に要指定 */\
  exe(OP_FMA,      &b00,      b00,           EXP_H3210,            BR[b][2][1], EXP_H3210, BR[b][1][1],          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#2 */\
  cex(OP_CEXE,     &ex0, 0, 0, cc1[o][i], cc0[o][i], 0x8888);                                                                                             /* stage#2 */\
  mop(OP_STWR,ex0, &b00,                     (Ull)kp0[o][i][CHIP], 0LL,         MSK_D0,    (Ull)kp0[o][i][CHIP], 1LL,       0,      1,   NULL,   1LL)     /* stage#2 */

//EMAX5A begin back_g_ker mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
        /*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-IMX4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                           /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                            /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,      &img,  img,             EXP_H3210,  INIT0?IMX4M4:0, EXP_H3210,  0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
		  exe(OP_ADD,      &cofs, INIT0?cofs:cofs, EXP_H3210,  4LL<<32|4LL,    EXP_H3210,  0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
		  exe(OP_ADD,      &iofs, img,             EXP_H3210,  cofs,           EXP_H3210,  0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
		  exe(OP_ADD,      &oofs, img,             EXP_H3210,  cofs,           EXP_H3210,  0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */
#if 1
		  back_g_ker_core1( 2,  0,  0); /**** oc0 ic0*****/
		  back_g_ker_core1( 3,  0,  1); /**** oc0 ic1*****/
		  back_g_ker_core1( 4,  0,  2); /**** oc0 ic2*****/
		  back_g_ker_core1( 5,  0,  3); /**** oc0 ic3*****/
		  back_g_ker_core1( 6,  1,  0); /**** oc1 ic0*****/
		  back_g_ker_core1( 7,  1,  1); /**** oc1 ic1*****/
		  back_g_ker_core1( 8,  1,  2); /**** oc1 ic2*****/
		  back_g_ker_core1( 9,  1,  3); /**** oc1 ic3*****/
		  back_g_ker_core1(10,  2,  0); /**** oc2 ic0*****/
		  back_g_ker_core1(11,  2,  1); /**** oc2 ic1*****/
		  back_g_ker_core1(12,  2,  2); /**** oc2 ic2*****/
		  back_g_ker_core1(13,  2,  3); /**** oc2 ic3*****/
		  back_g_ker_core1(14,  3,  0); /**** oc3 ic0*****/
		  back_g_ker_core1(15,  3,  1); /**** oc3 ic1*****/
		  back_g_ker_core1(16,  3,  2); /**** oc3 ic2*****/
		  back_g_ker_core1(17,  3,  3); /**** oc3 ic3*****/
		  back_g_ker_core1(18,  4,  0); /**** oc4 ic0*****/
		  back_g_ker_core1(19,  4,  1); /**** oc4 ic1*****/
		  back_g_ker_core1(20,  4,  2); /**** oc4 ic2*****/
		  back_g_ker_core1(21,  4,  3); /**** oc4 ic3*****/
		  back_g_ker_core1(22,  5,  0); /**** oc5 ic0*****/
		  back_g_ker_core1(23,  5,  1); /**** oc5 ic1*****/
		  back_g_ker_core1(24,  5,  2); /**** oc5 ic2*****/
		  back_g_ker_core1(25,  5,  3); /**** oc5 ic3*****/
		  back_g_ker_core1(26,  6,  0); /**** oc6 ic0*****/
		  back_g_ker_core1(27,  6,  1); /**** oc6 ic1*****/
		  back_g_ker_core1(28,  6,  2); /**** oc6 ic2*****/
		  back_g_ker_core1(29,  6,  3); /**** oc6 ic3*****/
		  back_g_ker_core1(30,  7,  0); /**** oc7 ic0*****/
		  back_g_ker_core1(31,  7,  1); /**** oc7 ic1*****/
		  back_g_ker_core1(32,  7,  2); /**** oc7 ic2*****/
		  back_g_ker_core1(33,  7,  3); /**** oc7 ic3*****/
#endif
#if 0
		  back_g_ker_core1( 2,  0,  0); /**** oc0  ic0*****/
		  back_g_ker_core1( 3,  0,  1); /**** oc0  ic1*****/
		  back_g_ker_core1( 4,  1,  0); /**** oc1  ic0*****/
		  back_g_ker_core1( 5,  1,  1); /**** oc1  ic1*****/
		  back_g_ker_core1( 6,  2,  0); /**** oc2  ic0*****/
		  back_g_ker_core1( 7,  2,  1); /**** oc2  ic1*****/
		  back_g_ker_core1( 8,  3,  0); /**** oc3  ic0*****/
		  back_g_ker_core1( 9,  3,  1); /**** oc3  ic1*****/
		  back_g_ker_core1(10,  4,  0); /**** oc4  ic0*****/
		  back_g_ker_core1(11,  4,  1); /**** oc4  ic1*****/
		  back_g_ker_core1(12,  5,  0); /**** oc5  ic0*****/
		  back_g_ker_core1(13,  5,  1); /**** oc5  ic1*****/
		  back_g_ker_core1(14,  6,  0); /**** oc6  ic0*****/
		  back_g_ker_core1(15,  6,  1); /**** oc6  ic1*****/
		  back_g_ker_core1(16,  7,  0); /**** oc7  ic0*****/
		  back_g_ker_core1(17,  7,  1); /**** oc7  ic1*****/
		  back_g_ker_core1(18,  8,  0); /**** oc8  ic0*****/
		  back_g_ker_core1(19,  8,  1); /**** oc8  ic1*****/
		  back_g_ker_core1(20,  9,  0); /**** oc9  ic0*****/
		  back_g_ker_core1(21,  9,  1); /**** oc9  ic1*****/
		  back_g_ker_core1(22, 10,  0); /**** oc10 ic0*****/
		  back_g_ker_core1(23, 10,  1); /**** oc10 ic1*****/
		  back_g_ker_core1(24, 11,  0); /**** oc11 ic0*****/
		  back_g_ker_core1(25, 11,  1); /**** oc11 ic1*****/
		  back_g_ker_core1(26, 12,  0); /**** oc12 ic0*****/
		  back_g_ker_core1(27, 12,  1); /**** oc12 ic1*****/
		  back_g_ker_core1(28, 13,  0); /**** oc13 ic0*****/
		  back_g_ker_core1(29, 13,  1); /**** oc13 ic1*****/
		  back_g_ker_core1(30, 14,  0); /**** oc14 ic0*****/
		  back_g_ker_core1(31, 14,  1); /**** oc14 ic1*****/
		  back_g_ker_core1(32, 15,  0); /**** oc15 ic0*****/
		  back_g_ker_core1(33, 15,  1); /**** oc15 ic1*****/
#endif
#if 0
		  back_g_ker_core1( 2,  0,  0); /**** oc0  ic0 *****/
		  back_g_ker_core1( 3,  1,  0); /**** oc1  ic0 *****/
		  back_g_ker_core1( 4,  2,  0); /**** oc2  ic0 *****/
		  back_g_ker_core1( 5,  3,  0); /**** oc3  ic0 *****/
		  back_g_ker_core1( 6,  4,  0); /**** oc4  ic0 *****/
		  back_g_ker_core1( 7,  5,  0); /**** oc5  ic0 *****/
		  back_g_ker_core1( 8,  6,  0); /**** oc6  ic0 *****/
		  back_g_ker_core1( 9,  7,  0); /**** oc7  ic0 *****/
		  back_g_ker_core1(10,  8,  0); /**** oc8  ic0 *****/
		  back_g_ker_core1(11,  9,  0); /**** oc9  ic0 *****/
		  back_g_ker_core1(12, 10,  0); /**** oc10 ic0 *****/
		  back_g_ker_core1(13, 11,  0); /**** oc11 ic0 *****/
		  back_g_ker_core1(14, 12,  0); /**** oc12 ic0 *****/
		  back_g_ker_core1(15, 13,  0); /**** oc13 ic0 *****/
		  back_g_ker_core1(16, 14,  0); /**** oc14 ic0 *****/
		  back_g_ker_core1(17, 15,  0); /**** oc15 ic0 *****/
		  back_g_ker_core1(18, 16,  0); /**** oc16 ic0 *****/
		  back_g_ker_core1(19, 17,  0); /**** oc17 ic0 *****/
		  back_g_ker_core1(20, 18,  0); /**** oc18 ic0 *****/
		  back_g_ker_core1(21, 19,  0); /**** oc19 ic0 *****/
		  back_g_ker_core1(22, 20,  0); /**** oc20 ic0 *****/
		  back_g_ker_core1(23, 21,  0); /**** oc21 ic0 *****/
		  back_g_ker_core1(24, 22,  0); /**** oc22 ic0 *****/
		  back_g_ker_core1(25, 23,  0); /**** oc23 ic0 *****/
		  back_g_ker_core1(26, 24,  0); /**** oc24 ic0 *****/
		  back_g_ker_core1(27, 25,  0); /**** oc25 ic0 *****/
		  back_g_ker_core1(28, 26,  0); /**** oc26 ic0 *****/
		  back_g_ker_core1(29, 27,  0); /**** oc27 ic0 *****/
		  back_g_ker_core1(30, 28,  0); /**** oc28 ic0 *****/
		  back_g_ker_core1(31, 29,  0); /**** oc29 ic0 *****/
		  back_g_ker_core1(32, 30,  0); /**** oc30 ic0 *****/
		  back_g_ker_core1(33, 31,  0); /**** oc31 ic0 *****/
#endif
                }
              }
            }
//EMAX5A end
            kidx++;
          }
        }
      }
    }
  }
//EMAX5A drain_dirty_lmm
#endif
#ifdef XMAX_VALIDATE
  count = 0;
  for (ic=0; ic<IC; ic++) {
    for (oc=0; oc<OC; oc++) {
      for (y=0; y<K; y++) {
	for (x=0; x<K; x++) {
	  float host = *(float*)&g_ker[(oc*IC+ic)*K*K+y*K+x];
	  float xmax = *(float*)&i_ker[(oc*IC+ic)*K*K+y*K+x];
	  if (udiff(host,xmax)>ERRTH) {
	    count++;
	    printf("[%d][%d][%d][%d]: g_ker=%7.5e(%8.8x) i_ker=%7.5e(%8.8x)\n", oc, ic, y, x, host, *(Uint*)&host, xmax, *(Uint*)&xmax);
	  }
	}
      }
    }
  }
  if (count)
    printf("Num of diffs 1-4: %d\n", count);
#endif
  xmax_cpyout(2, g_ker, 1, 1, i_ker, IC*K*K, OC, OC); /* g_kernel */
#endif

  /*================================================================================================*/
  /*=== back_in ====================================================================================*/
  /*================================================================================================*/

#undef  IMAP
#undef  OMAP
#undef  W
#undef  NCHIP
#define IMAP  4
#define OMAP  8
#define W     1
#define NCHIP 2

#undef XMAX_VALIDATE
//#define XMAX_VALIDATE
#ifdef XMAX_VALIDATE
  /* in <- kernel, out */
  memset(in0, 0, sizeof(in0[0])*BATCH*IC*IM*IM);
  if (K == 1 || IM-K+1 == M) { y0 = 0;    x0 = 0;    }
  else if (IM == M)          { y0 = -K/2; x0 = -K/2; }
  for (img=0;img<BATCH;img++) { /*100, 100*/
    for (ch=0;ch<IC*K*K;ch++) { /*5x5, 8x3x3*/
      ic = ch/(K*K);
      y  = ch%(K*K)/K + y0;
      x  = ch%(K*K)%K + x0;
      for (oc=0; oc<OC; oc++) {
	for (rofs=0;rofs<M;rofs++) { /*24, 10*/
	  ip0 = &in0[((img*IC+ic)*IM+rofs+y)*IM+x];
	  for (cofs=0;cofs<M;cofs++) { /*24, 10*/
	    if (0<=rofs+y && rofs+y<IM && 0<=cofs+x && cofs+x<IM)
	      *(float*)ip0 += *(float*)&ker[oc*IC*K*K+ch] * *(float*)&out0[img*M*M*OC+oc*M*M+rofs*M+cofs];
	    ip0++;
	  }
	}
      }
    }
  }
#endif
//efine PBL1_5_VERSION0
#define PBL1_5_VERSION1
#ifdef  PBL1_5_VERSION0
//xmax_cpyin(0, i_out, &M, out0, BATCH, OC, M, M, 1); /* out *//*★★★削除可能★★★*/
  xmax_cpyin(0, i_ker, &K, ker,  IC,    OC, K, K, 1); //imemcpy(i_ker, ker,  OC*IC*K*K); K=K;
  xmax_bzero(i_inp, BATCH*IC*IM*IM); /* in */
#if 0
  if (K == 1 || IM-K+1 == M) { y0 = 0;    x0 = 0;    }
  else if (IM == M)          { y0 = -K/2; x0 = -K/2; }
  for (img=0;img<BATCH;img++) { /*100, 100*/
    for (ch=0;ch<IC*K*K;ch++) { /*5x5, 8x3x3*/
      ic = ch/(K*K);
      y  = ch%(K*K)/K + y0;
      x  = ch%(K*K)%K + x0;
      for (oc=0; oc<OC; oc++) {
	op0 = &i_out[img*M*M*OC+oc*M*M];
	ip0 = &i_inp[(img*IC+ic)*IM*IM+y*IM+x];
	float cker = *(float*)&i_ker[oc*IC*K*K+ch];
	for (rofs=0;rofs<M;rofs++) { /*24, 10*/
	  for (cofs=0;cofs<M;cofs++) { /*24, 10*/
	    if (0<=rofs+y && rofs+y<IM && 0<=cofs+x && cofs+x<IM)
	      *(float*)&ip0[rofs*IM+cofs] += cker * *(float*)&op0[rofs*M+cofs];
	  }
	}
      }
    }
  }
#else
  IM4    = IM*4;
  M4     = M*4;
  M4IM4  = M4<<32|IM4;
  IMlen  = IM*IM;
  Mlen   = M*M;
  if (K == 1 || IM-K+1 == M) { y0 = 0;    x0 = 0;    }
  else if (IM == M)          { y0 = -K/2; x0 = -K/2; }
#if 0
printf("   BACK10  IMlen=%dB Mlen=%dB\n", (Uint)IMlen*4, (Uint)Mlen*4);
#endif
  for (img=0;img<BATCH;img++) { /*100, 100*/
    for (ch=0;ch<IC*K*K;ch++) { /*5x5, 8x3x3*/
      ic = ch/(K*K);
      y  = ch%(K*K)/K + y0;
      x  = ch%(K*K)%K + x0;
      Ull  yIM4  = y*IM4;
      Ull  x4    = x*4;
      Ull  IMIM4 = IM*IM4;
      for (oc=0; oc<OC; oc++) {
	op0 = &i_out[img*M*M*OC+oc*M*M];
	ip0 = &i_inp[(img*IC+ic)*IM*IM+y*IM+x];
	c00 = (Ull)i_ker[oc*IC*K*K+ch];
	Uint *ot00 = op0;                      
	Uint *it00 = &i_inp[(img*IC+ic)*IM*IM];
//EMAX5A begin back_in mapdist=0
  /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
    /*2*/ for (INIT1=1,LOOP1=M,rofs=(0-M4)<<32|((0-IM4)&0xffffffff); LOOP1--; INIT1=0) {                             /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
      /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                          /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
              exe(OP_ADD,      &rofs, rofs,            EXP_H3210, INIT0?M4IM4:0,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
              exe(OP_ADD,      &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,    EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
              exe(OP_ADD,      &iofs, rofs,            EXP_H3210, cofs,           EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */
              exe(OP_ADD,      &oofs, rofs,            EXP_H3210, cofs,           EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
              /****in0*****/
              exe(OP_ADD,      &r10,  rofs,            EXP_H3210, yIM4,           EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */
              exe(OP_ADD,      &r11,  cofs,            EXP_H3210, x4,             EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */
	      exe(OP_CMP_LT,   &cc0,  r10,             EXP_H3210, IMIM4,          EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#2 */
	      exe(OP_CMP_LT,   &cc1,  r11,             EXP_H3210, IM4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#2 */
	      mop(OP_LDWR,  1, &BR[3][0][1],           (Ull)ip0,  iofs,  MSK_W0,  (Ull)it00, IMlen,  0, 1,   NULL,     IMlen);            /* stage#3 */
	      mop(OP_LDWR,  1, &BR[3][2][1],           (Ull)op0,  oofs,  MSK_W1,  (Ull)ot00, Mlen,   0, 0,   NULL,     Mlen);             /* stage#3 */
	      exe(OP_FMA,      &AR[3][0], BR[3][0][1], EXP_H3210, c00,   EXP_H3210, BR[3][2][1],  EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#3 */
	      cex(OP_CEXE,     &ex0,  0, 0, cc1, cc0, 0x8888);                                                                            /* stage#3 */
	      mop(OP_STWR,ex0, &AR[3][0],              iofs,   (Ull)ip0, MSK_D0,  (Ull)it00, IMlen,  0, 1,   NULL,     IMlen);            /* stage#3 */
	    }
	  }
	}
//EMAX5A end
      }
    }
  }
//EMAX5A drain_dirty_lmm
#endif
#ifdef XMAX_VALIDATE
  count = 0;
  for (img=0; img<BATCH; img++) {
    for (ic=0; ic<IC; ic++) {
      for (y=0; y<IM; y++) {
        for (x=0; x<IM; x++) {
          float host = *(float*)&in0[(img*IC+ic)*IM*IM+y*IM+x];
          float xmax = *(float*)&i_inp[(img*IC+ic)*IM*IM+y*IM+x];
          if (udiff(host,xmax)>ERRTH) {
            count++;
            printf("[%d][%d][%d][%d]: in0=%7.5e(%8.8x) i_inp=%7.5e(%8.8x)\n", (Uint)img, ic, y, x, host, *(Uint*)&host, xmax, *(Uint*)&xmax);
          }
        }
      }
    }
  }
  if (count)
    printf("Num of diffs 1-5: %d\n", count);
#endif
  xmax_cpyout(2, in0, 1, 1, i_inp, BATCH*IM*IM, IC, IC); /* in */
#endif
#ifdef  PBL1_5_VERSION1
  /***********************************/
  /* ★★★ PBL1-5 (in)       ★★★ */
  /***********************************/
//xmax_cpyin(1, i_out, &M, out0, BATCH, OC, M, M, 1); //dst[OC][M][BATCH][M] <- src[BATCH][OC][M][M]
  xmax_cpyin(0, i_ker, &K, ker,  IC,    OC, K, K, 1); //imemcpy(i_ker, ker,  OC*IC*K*K); K=K;
  xmax_bzero(i_inp, IC*IM*BATCH*IM); /* in */
#if 0
  if (K == 1 || IM-K+1 == M) { y0 = 0;    x0 = 0;    }
  else if (IM == M)          { y0 = -K/2; x0 = -K/2; }
  for (oset=0; oset<((OC+OMAP-1)&~(OMAP-1)); oset+=OMAP) { /* set output channel */
    Uint *op0[OMAP]; float kp0[OMAP];
    for (rofs=0;rofs<M;rofs++) { /*24, 10*/
      for (ch=0;ch<IC*K*K;ch++) { /*5x5, 8x3x3*/
	ic = ch/(K*K);
	y  = ch%(K*K)/K + y0;
	x  = ch%(K*K)%K + x0;
	if (0<=rofs+y && rofs+y<IM) {
	  ip0 = &i_inp[ic*IM*BATCH*IM+(rofs+y)*BATCH*IM+x];
	  for (oc=0; oc<OMAP&&(oset+oc)<OC; oc++) { /* set output channel */
	    op0[oc] = &i_out[(oset+oc)*M*BATCH*M+rofs*BATCH*M];
	    kp0[oc] = *(float*)&i_ker[(oset+oc)*IC*K*K+ch];
	    for (img=0;img<BATCH;img++) { /*100, 100*/
	      for (cofs=0;cofs<M;cofs++) { /*24, 10*/
		if (0<=cofs+x && cofs+x<IM) {
		  *(float*)&ip0[img*IM+cofs] += kp0[oc] * *(float*)&op0[oc][img*M+cofs];
		}
	      }
	    }
	  }
	}
      }
    }
  }
#else
  IM4    = IM*4;
  M4     = M*4;
  M4IM4  = M4<<32|IM4;
  IMlen  = IM*BATCH;
  Mlen   = M*BATCH;
  if (K == 1 || IM-K+1 == M) { y0 = 0;    x0 = 0;    }
  else if (IM == M)          { y0 = -K/2; x0 = -K/2; }

  if (IMlen > 65536/4 || Mlen > 65536/4)
    printf("   BACK10  IMlen=%dB Mlen=%dB\n", (Uint)IMlen*4, (Uint)Mlen*4);

  /* +----------------------+-----------------------+                     */
  /* |   ker[oc+0][ic][yx]  |out[oc+0][row+yx*]再利用 K行                 */
  /* +----------------------+-----------------------+                     */
  /* |   ker[oc+1][ic][yx]  |out[oc+1][row+yx*]再利用 K行                 */
  /* +----------------------+-----------------------+                     */
  /* |   ker[oc+2][ic][yx]  |out[oc+2][row+yx*]再利用 K行                 */
  /* +----------------------+-----------------------+                     */
  /* |   ker[oc+3][ic][yx]  |out[oc+3][row+yx*]再利用 K行                 */
  /* |                      |inp[ic][row]           |                     */
  /* +----------------------+-----------------------+                     */
  /*                             oc:stageに展開                           */
  /*                             ic:最外ループ                            */
  /*                                       y:段数を埋めるほど多くない     */
  /*                                        x:行方向                      */
  for (oset=0; oset<((OC+OMAP-1)&~(OMAP-1)); oset+=OMAP) { /* set output channel */
    Uint inum[IMAP][NCHIP], *ip0[IMAP][NCHIP], *it0[IMAP][NCHIP];
    Uint onum[OMAP], *op0[OMAP], *ot0[OMAP];
    Uint kp0[OMAP][IMAP][NCHIP];
    for (iset=0; iset<((IC+IMAP*NCHIP-1)&~(IMAP*NCHIP-1)); iset+=IMAP*NCHIP) { /* set offset of input channel */
      for (rofs=0;rofs<M;rofs++) { /*24, 10*/
	for (xy=0;xy<K*K;xy++) { /*5x5, 8x3x3*/
	  y  = xy/K + y0;
	  x  = xy%K + x0;
	  Ull  yIM4  = y*IM4;
	  Ull  x4    = x*4;
	  Ull  IMIM4 = IM*IM4;
	  if (0<=rofs+y && rofs+y<IM) {
	    for (CHIP=0; CHIP<NCHIP; CHIP++) {
	      for (ic=0; ic<IMAP; ic++) {
		inum[ic][CHIP] = iset+IMAP*CHIP+ic;
		ip0[ic][CHIP]  = (iset+IMAP*CHIP+ic)<IC ? &i_inp[(iset+IMAP*CHIP+ic)*IM*BATCH*IM+(rofs+y)*BATCH*IM+x] : 0;
		it0[ic][CHIP]  = (iset+IMAP*CHIP+ic)<IC ? &i_inp[(iset+IMAP*CHIP+ic)*IM*BATCH*IM+(rofs+y)*BATCH*IM] : 0; // xのマイナス成分を除去
	      }
	    }
            for (oc=0; oc<OMAP; oc++) {
	      onum[oc] = oset+oc;
	      op0[oc]  = (oset+oc)<OC ? &i_out[(oset+oc)*M*BATCH*M  +rofs*BATCH*M] : 0;
	      if (M*BATCH*M <= 65536/4) {
		Mlen = M*BATCH*M;
		ot0[oc] = (oset+oc)<OC ? &i_out[(oset+oc)*M*BATCH*M] : 0; /* output */
	      }
	      else
		ot0[oc] = op0[oc];
	    }
	    for (oc=0; oc<OMAP; oc++) {
	      for (CHIP=0; CHIP<NCHIP; CHIP++) {
		for (ic=0; ic<IMAP; ic++)
		  kp0[oc][ic][CHIP]  = ((oset+oc)<OC && (iset+IMAP*CHIP+ic)<IC) ? i_ker[(oset+oc)*IC*K*K+(iset+IMAP*CHIP+ic)*K*K+xy] : 0; /* 0.0 */
	      }
	    }

#define back_in_core1(b, bp1, o, i) \
  mop(OP_LDWR,  1, &BR[b][0][1],          (Ull)op0[o], oofs,            MSK_W1,    (Ull)ot0[o], Mlen,      0,      0,   NULL,   Mlen); /* stage#2 */\
  exe(OP_FMA,      &AR[bp1][0], AR[b][0], EXP_H3210,   kp0[o][i][CHIP], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)   /* stage#3 */

#define back_in_final(b, bp2, i) \
  exe(OP_ADD,      &r10,      cofs,            EXP_H3210,         x4,                  EXP_H3210, 0LL,               EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);   /* stage#5 */\
  exe(OP_CMP_LT,   &cc0,      r10,             EXP_H3210,         IM4,                 EXP_H3210, 0LL,               EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#6 */\
  mop(OP_LDWR,  1, &BR[bp2][0][1],             (Ull)ip0[i][CHIP], iofs,                MSK_W0,    (Ull)it0[i][CHIP], IMlen,     0,      1,                    NULL,   IMlen); /* stage#7 */\
  exe(OP_FAD,      &AR[bp2][0], AR[b][0],      EXP_H3210,         BR[bp2][0][1],       EXP_H3210, 0LL,               EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#7 */\
  cex(OP_CEXE,     &ex0, 0, 0, 0, cc0, 0xaaaa);                                                                                                                               /* stage#7 */\
  mop(OP_STWR,ex0, &AR[bp2][0],                iofs,              (Ull)ip0[i][CHIP],   MSK_D0,    (Ull)it0[i][CHIP], IMlen,     0,      1,                    NULL,   IMlen)  /* stage#7 */

//EMAX5A begin back_in mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
        /*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-M4)<<32|((0-IM4)&0xffffffff); LOOP1--; INIT1=0) {                                       /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,      &img,      img,             EXP_H3210,   INIT0?M4IM4:0, EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#0 */
		  exe(OP_ADD,      &cofs,     INIT0?cofs:cofs, EXP_H3210,   4LL<<32|4LL,   EXP_H3210, 0LL,         EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);   /* stage#0 */
		  exe(OP_ADD,      &iofs,     img,             EXP_H3210,   cofs,          EXP_H3210, 0LL,         EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);   /* stage#1 */
		  exe(OP_ADD,      &oofs,     img,             EXP_H3210,   cofs,          EXP_H3210, 0LL,         EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL);   /* stage#1 */
#if 1
		  /****ic0*****/
		  mop(OP_LDWR,  1, &BR[2][0][1],                   (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[3][0],    kp0[0][0][CHIP],  EXP_H3210,   BR[2][0][1],   EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1( 3,  4,  1,  0); /**** oc1  ic0*****/
		  back_in_core1( 4,  5,  2,  0); /**** oc2  ic0*****/
		  back_in_core1( 5,  6,  3,  0); /**** oc3  ic0*****/
		  back_in_core1( 6,  7,  4,  0); /**** oc4  ic0*****/
		  back_in_core1( 7,  8,  5,  0); /**** oc5  ic0*****/
		  back_in_core1( 8,  9,  6,  0); /**** oc6  ic0*****/
		  back_in_core1( 9, 10,  7,  0); /**** oc7  ic0*****/
		  back_in_final(10, 12,  0);     /****OMAP( 8)+2,OMAP( 8)+4****/
		  /****ic1*****/
		  mop(OP_LDWR,  1, &BR[13][0][1],                  (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[14][0],   kp0[0][1][CHIP],  EXP_H3210,   BR[13][0][1],  EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1(14, 15,  1,  1); /**** oc1  ic1*****/
		  back_in_core1(15, 16,  2,  1); /**** oc2  ic1*****/
		  back_in_core1(16, 17,  3,  1); /**** oc3  ic1*****/
		  back_in_core1(17, 18,  4,  1); /**** oc4  ic1*****/
		  back_in_core1(18, 19,  5,  1); /**** oc5  ic1*****/
		  back_in_core1(19, 20,  6,  1); /**** oc6  ic1*****/
		  back_in_core1(20, 21,  7,  1); /**** oc7  ic1*****/
		  back_in_final(21, 23,  1);     /****OMAP( 8)+2,OMAP( 8)+4****/
		  /****ic2*****/
		  mop(OP_LDWR,  1, &BR[24][0][1],                  (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[25][0],   kp0[0][2][CHIP],  EXP_H3210,   BR[24][0][1],  EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1(25, 26,  1,  2); /**** oc1  ic2*****/
		  back_in_core1(26, 27,  2,  2); /**** oc2  ic2*****/
		  back_in_core1(27, 28,  3,  2); /**** oc3  ic2*****/
		  back_in_core1(28, 29,  4,  2); /**** oc4  ic2*****/
		  back_in_core1(29, 30,  5,  2); /**** oc5  ic2*****/
		  back_in_core1(30, 31,  6,  2); /**** oc6  ic2*****/
		  back_in_core1(31, 32,  7,  2); /**** oc7  ic2*****/
		  back_in_final(32, 34,  2);     /****OMAP( 8)+2,OMAP( 8)+4****/
		  /****ic3*****/
		  mop(OP_LDWR,  1, &BR[35][0][1],                  (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[36][0],   kp0[0][3][CHIP],  EXP_H3210,   BR[35][0][1],  EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1(36, 37,  1,  3); /**** oc1  ic3*****/
		  back_in_core1(37, 38,  2,  3); /**** oc2  ic3*****/
		  back_in_core1(38, 39,  3,  3); /**** oc3  ic3*****/
		  back_in_core1(39, 40,  4,  3); /**** oc4  ic3*****/
		  back_in_core1(40, 41,  5,  3); /**** oc5  ic3*****/
		  back_in_core1(41, 42,  6,  3); /**** oc6  ic3*****/
		  back_in_core1(42, 43,  7,  3); /**** oc7  ic3*****/
		  back_in_final(43, 45,  3);     /****OMAP( 8)+2,OMAP( 8)+4****/
#endif
#if 0
		  /****ic0*****/
		  mop(OP_LDWR,  1, &BR[2][0][1],                   (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[3][0],    kp0[0][0][CHIP],  EXP_H3210,   BR[2][0][1],   EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1( 3,  4,  1,  0); /**** oc1  ic0*****/
		  back_in_core1( 4,  5,  2,  0); /**** oc2  ic0*****/
		  back_in_core1( 5,  6,  3,  0); /**** oc3  ic0*****/
		  back_in_core1( 6,  7,  4,  0); /**** oc4  ic0*****/
		  back_in_core1( 7,  8,  5,  0); /**** oc5  ic0*****/
		  back_in_core1( 8,  9,  6,  0); /**** oc6  ic0*****/
		  back_in_core1( 9, 10,  7,  0); /**** oc7  ic0*****/
		  back_in_core1(10, 11,  8,  0); /**** oc8  ic0*****/
		  back_in_core1(11, 12,  9,  0); /**** oc9  ic0*****/
		  back_in_core1(12, 13, 10,  0); /**** oc10 ic0*****/
		  back_in_core1(13, 14, 11,  0); /**** oc11 ic0*****/
		  back_in_core1(14, 15, 12,  0); /**** oc12 ic0*****/
		  back_in_core1(15, 16, 13,  0); /**** oc13 ic0*****/
		  back_in_core1(16, 17, 14,  0); /**** oc14 ic0*****/
		  back_in_core1(17, 18, 15,  0); /**** oc15 ic0*****/
		  back_in_final(18, 20,  0);     /****OMAP(16)+2,OMAP(16)+4****/
		  /****ic1*****/
		  mop(OP_LDWR,  1, &BR[21][0][1],                  (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[22][0],   kp0[0][1][CHIP],  EXP_H3210,  BR[21][0][1],   EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1(22, 23,  1,  1); /**** oc1  ic1*****/
		  back_in_core1(23, 24,  2,  1); /**** oc2  ic1*****/
		  back_in_core1(24, 25,  3,  1); /**** oc3  ic1*****/
		  back_in_core1(25, 26,  4,  1); /**** oc4  ic1*****/
		  back_in_core1(26, 27,  5,  1); /**** oc5  ic1*****/
		  back_in_core1(27, 28,  6,  1); /**** oc6  ic1*****/
		  back_in_core1(28, 29,  7,  1); /**** oc7  ic1*****/
		  back_in_core1(29, 30,  8,  1); /**** oc8  ic1*****/
		  back_in_core1(30, 31,  9,  1); /**** oc9  ic1*****/
		  back_in_core1(31, 32, 10,  1); /**** oc10 ic1*****/
		  back_in_core1(32, 33, 11,  1); /**** oc11 ic1*****/
		  back_in_core1(33, 34, 12,  1); /**** oc12 ic1*****/
		  back_in_core1(34, 35, 13,  1); /**** oc13 ic1*****/
		  back_in_core1(35, 36, 14,  1); /**** oc14 ic1*****/
		  back_in_core1(36, 37, 15,  1); /**** oc15 ic1*****/
		  back_in_final(37, 39,  1);     /****OMAP(35)+2,OMAP(35)+4****/
#endif
#if 0
		  /****ic0*****/
		  mop(OP_LDWR,  1, &BR[2][0][1],                   (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[3][0],    kp0[0][0][CHIP],  EXP_H3210,   BR[2][0][1],   EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1( 3,  4,  1,  0); /**** oc1  ic0*****/
		  back_in_core1( 4,  5,  2,  0); /**** oc2  ic0*****/
		  back_in_core1( 5,  6,  3,  0); /**** oc3  ic0*****/
		  back_in_core1( 6,  7,  4,  0); /**** oc4  ic0*****/
		  back_in_core1( 7,  8,  5,  0); /**** oc5  ic0*****/
		  back_in_core1( 8,  9,  6,  0); /**** oc6  ic0*****/
		  back_in_core1( 9, 10,  7,  0); /**** oc7  ic0*****/
		  back_in_core1(10, 11,  8,  0); /**** oc8  ic0*****/
		  back_in_core1(11, 12,  9,  0); /**** oc9  ic0*****/
		  back_in_core1(12, 13, 10,  0); /**** oc10 ic0*****/
		  back_in_core1(13, 14, 11,  0); /**** oc11 ic0*****/
		  back_in_core1(14, 15, 12,  0); /**** oc12 ic0*****/
		  back_in_core1(15, 16, 13,  0); /**** oc13 ic0*****/
		  back_in_core1(16, 17, 14,  0); /**** oc14 ic0*****/
		  back_in_core1(17, 18, 15,  0); /**** oc15 ic0*****/
		  back_in_core1(18, 19, 16,  0); /**** oc16 ic0*****/
		  back_in_core1(19, 20, 17,  0); /**** oc17 ic0*****/
		  back_in_core1(20, 21, 18,  0); /**** oc18 ic0*****/
		  back_in_core1(21, 22, 19,  0); /**** oc19 ic0*****/
		  back_in_core1(22, 23, 20,  0); /**** oc20 ic0*****/
		  back_in_core1(23, 24, 21,  0); /**** oc21 ic0*****/
		  back_in_core1(24, 25, 22,  0); /**** oc22 ic0*****/
		  back_in_core1(25, 26, 23,  0); /**** oc23 ic0*****/
		  back_in_core1(26, 27, 24,  0); /**** oc24 ic0*****/
		  back_in_core1(27, 28, 25,  0); /**** oc25 ic0*****/
		  back_in_core1(28, 29, 26,  0); /**** oc26 ic0*****/
		  back_in_core1(29, 30, 27,  0); /**** oc27 ic0*****/
		  back_in_core1(30, 31, 28,  0); /**** oc28 ic0*****/
		  back_in_core1(31, 32, 29,  0); /**** oc29 ic0*****/
		  back_in_core1(32, 33, 30,  0); /**** oc30 ic0*****/
		  back_in_core1(33, 34, 31,  0); /**** oc31 ic0*****/
		  back_in_final(34, 36,  0);     /****OMAP(32)+2,OMAP(32)+4****/
#endif
	        }
	      }
	    }
//EMAX5A end
          }
	}
      }
    }
  }
//EMAX5A drain_dirty_lmm
#endif
#ifdef XMAX_VALIDATE
  count = 0;
  for (img=0; img<BATCH; img++) {
    for (ic=0; ic<IC; ic++) {
      for (y=0; y<IM; y++) {
	for (x=0; x<IM; x++) {
	  float host = *(float*)&in0[(img*IC+ic)*IM*IM+y*IM+x];
	  float xmax = *(float*)&i_inp[ic*IM*BATCH*IM+y*BATCH*IM+img*IM+x];
	  if (udiff(host,xmax)>ERRTH) {
	    count++;
	    printf("[%d][%d][%d][%d]: in0=%7.5e(%8.8x) i_inp=%7.5e(%8.8x)\n", (Uint)img, (Uint)ic, (Uint)y, (Uint)x, host, *(Uint*)&host, xmax, *(Uint*)&xmax);
	  }
	}
      }
    }
  }
  if (count)
    printf("Num of diffs 1-5: %d\n", count);
#endif
  xmax_cpyout(1, in0, BATCH, IC, i_inp, IM, IM, IC); /* in */
#endif
}
