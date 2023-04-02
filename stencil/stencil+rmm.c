
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-emax/sample/filter/backend.c,v 1.1 2012/09/27 00:04:03 nakashim Exp nakashim $";

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

#define AWD 320
#define AHT 242

int WD=AWD, HT=AHT, BITMAP=320*240, SCRWD=5, SCRHT=5, VECWD=240, VECHT=240, VECSTEP=4;

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#else
Ull nanosec_sav;
Ull nanosec;
reset_nanosec()
{
  int i;
  nanosec = 0;
#if defined(ARMSIML)
  nanosec_sav = _getclk(0);
#else
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec_sav = 1000000000*ts.tv_sec + ts.tv_nsec;
#endif
}
get_nanosec(int class)
{
  Ull nanosec_now;
#if defined(ARMSIML)
  nanosec_now = _getclk(0);
  nanosec += nanosec_now - nanosec_sav;
  nanosec_sav = nanosec_now;
#else
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec_now = 1000000000*ts.tv_sec + ts.tv_nsec;
  nanosec += nanosec_now - nanosec_sav;
  nanosec_sav = nanosec_now;
#endif
}
show_nanosec()
{
#if defined(ARMSIML)
  printf("SIML_cycle: ARM:%llu\n", nanosec);
#else
  printf("nanosec: ARM:%llu\n", nanosec);
#endif
}
#endif
#if !defined(ARMSIML)
#include "./xdisp.c"
#endif

#define XC                  13
#define MID               ((XC-1)/2)
#define DP                  54/*122*/
#define WDHT               (AWD*AHT)
#define WDHTDP             (AWD*AHT*DP)

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

unsigned int   *L;
unsigned int   *R;
unsigned int   *W;
unsigned int   *D;
struct GrA   { float   GrA[XC][DP][AHT][AWD];} *GrA;
struct B3D   { float   B3D[DP][AHT][AWD];}     *B3D;
struct C3D   { float   C3D[DP][AHT][AWD];}     *C3D;
struct D3D   { float   D3D[DP][AHT][AWD];}     *D3D;
struct WZ0   { float   WZ0[AHT][AWD];}         *WZ0;
struct WZ1   { float   WZ1[AHT][AWD];}         *WZ1;
struct WZ2   { float   WZ2[AHT][AWD];}         *WZ2;

#define CSIMWD 320
#define CSIMHT 240
#define CSIMBM (CSIMWD*CSIMHT)
Uint Z[CSIMBM];

/****************/
/*** MAIN     ***/
/****************/

main(argc, argv)
  int argc;  char **argv;
{
  static wait_previous = 0;
  int i, j, k, l, dx, dy;

  int x, y, z;
  float c1, c2, c3, c4;

  sysinit(sizeof(struct GrA)
	 +sizeof(struct B3D)
	 +sizeof(struct C3D)
	 +sizeof(struct D3D)
	 +sizeof(int)*BITMAP*4,32);
  printf("membase: %08.8x\n", (Uint)membase);
  GrA = (struct GrA*)((Uchar*)membase);
  B3D = (struct B3D*)((Uchar*)GrA+sizeof(struct GrA));
  C3D = (struct C3D*)((Uchar*)B3D+sizeof(struct B3D));
  D3D = (struct D3D*)((Uchar*)C3D+sizeof(struct C3D));
  WZ0 = (struct WZ0*)         B3D;
  WZ1 = (struct WZ1*)         C3D;
  WZ2 = (struct WZ2*)         D3D;
  L   = (Uint*)      ((Uchar*)D3D+sizeof(struct D3D));
  R   = (Uint*)      ((Uchar*)L  +sizeof(int)*BITMAP);
  W   = (Uint*)      ((Uchar*)R  +sizeof(int)*BITMAP);
  D   = (Uint*)      ((Uchar*)W  +sizeof(int)*BITMAP);
  printf("GrA   : %08.8x(%08.8x)\n", GrA, sizeof(struct GrA));
  printf("B3D   : %08.8x(%08.8x)\n", B3D, sizeof(struct B3D));
  printf("C3D   : %08.8x(%08.8x)\n", C3D, sizeof(struct C3D));
  printf("D3D   : %08.8x(%08.8x)\n", D3D, sizeof(struct D3D));
  printf("WZ0   : %08.8x(%08.8x)\n", WZ0, sizeof(struct WZ0));
  printf("WZ1   : %08.8x(%08.8x)\n", WZ1, sizeof(struct WZ1));
  printf("WZ2   : %08.8x(%08.8x)\n", WZ2, sizeof(struct WZ2));
  printf("L     : %08.8x\n", L);
  printf("R     : %08.8x\n", R);
  printf("W     : %08.8x\n", W);
  printf("D     : %08.8x\n", D);

#if !defined(ARMSIML)
  x11_open(0);
#endif

#if 1
  /*****************************************************/
  /* grapes */
  for (i=0; i<XC; i++) {
    for (z=0; z<DP; z++) {
      for (y=0; y<HT; y++) {
	for (x=0; x<WD; x++)
	  GrA->GrA[i][z][y][x] = 2.0;
      }
    }
  }
  for (z=0; z<DP; z++) {
    for (y=0; y<HT; y++) {
      for (x=0; x<WD; x++) {
	B3D->B3D[z][y][x] = (float)pow(-1,(float)(z/8))*(float)(z*y+16*x);
	D3D->D3D[z][y][x] = 0.0;
      }
    }
  }

  puts("grapes-start");
  reset_nanosec();
  grapes(D3D->D3D, GrA->GrA, B3D->B3D);
  get_nanosec(0);
  show_nanosec();
  puts("grapes-end");

  for(y=0;y<HT;y++){
    for(x=0;x<WD;x++){
      W[WD*y+x] = (unsigned int)(D3D->D3D[DP/3][y][x]);       
      W[WD*y+x] =  W[WD*y+x] << 8;
      //printf("%d ", W[WD*y+x] );
    }
  }
#ifdef ARMSIML
  _copyX(0, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(0, W);
  x11_update();
#endif

#endif
#if 1
  /*****************************************************/
  /* jacobi */
  for (z=0; z<DP; z++) {
    for (y=0; y<HT; y++) {
      for (x=0; x<WD; x++) {
	B3D->B3D[z][y][x] = pow(-1,z/128)*(float)x*y+z*z;
	D3D->D3D[z][y][x] = 0.0;
      }
    }
  } /* c1 = 0.2; c2 = 0.3; */

  puts("jacobi-start");
  reset_nanosec();
  jacobi(D3D->D3D, B3D->B3D);
  get_nanosec(0);
  show_nanosec();
  puts("jacobi-end");

  for(y=0;y<HT;y++){
    for(x=0;x<WD;x++){
      W[WD*y+x] = (unsigned int)(D3D->D3D[DP/2][y][x]);	
      W[WD*y+x] = W[WD*y+x]<<8;
      //printf("%d ", W[WD*y+x] );
    }
  }
#ifdef ARMSIML
  _copyX(1, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(1, W);
  x11_update();
#endif

#endif
#if 1
  /*****************************************************/
  /* fd6 */
  for (z=0; z<DP; z++) {
    for (y=0; y<HT; y++) {
      for (x=0; x<WD; x++) {
	B3D->B3D[z][y][x] = pow(-1,z*z)*(float)x*y/32*(float)z*z;
	D3D->D3D[z][y][x] = 0.0;
      }
    }
  } /* c1 = 0.1; c2 = 0.2; c3 = 0.3; c4 = 0.4; */

  puts("fd6-start");
  reset_nanosec();
  fd6(D3D->D3D, B3D->B3D);
  get_nanosec(0);
  show_nanosec();
  puts("fd6-end");

  for(y=0;y<HT;y++){
    for(x=0;x<WD;x++){
      W[WD*y+x] = (unsigned int)(D3D->D3D[DP/3][y][x]);
      W[WD*y+x] = W[WD*y+x]<<8;
      //printf("%d ", W[WD*y+x] );
    }
  }
#ifdef ARMSIML
  _copyX(2, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(2, W);
  x11_update();
#endif

#endif
#if 1
 /*****************************************************/
  /* RESID */
  for (z=0; z<DP; z++) {
     for (y=0; y<HT; y++) {
        for (x=0; x<WD; x++) {
           B3D->B3D[z][y][x] = (float)z*z;
           C3D->C3D[z][y][x] = pow(-1,(float)x*y/32)*(float)x+(float)y;
           D3D->D3D[z][y][x] = 0.0;
        }
     }
  } /* a0 = -0.1; a1 = -0.2; a2 = -0.3; a3 = -0.4 */

  puts("resid-start");
  reset_nanosec();
  resid(D3D->D3D, B3D->B3D, C3D->C3D);
  get_nanosec(0);
  show_nanosec();
  puts("resid-end");

  for(y=0;y<HT;y++){
    for(x=0;x<WD;x++){
      W[WD*y+x] = isnan(D3D->D3D[DP/2][y][x])?255:(Uint)(D3D->D3D[DP/2][y][x]);
      W[WD*y+x] = W[WD*y+x]<<8;
      //printf("%d ", W[WD*y+x] );
    }
  }
#ifdef ARMSIML
  _copyX(3, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(3, W);
  x11_update();
#endif

#endif
#if 1
  /*****************************************************/
  /* wave2d */
  for(y=0; y<HT; y++) {
    for(x=0; x<WD; x++) {
      if( (y>30 && y<100) || (y>HT-100 && y<HT-30) ) {
	WZ0->WZ0[y][x]=200000000.0;
      }
      else {
	WZ0->WZ0[y][x]=0.0;
      }
    }
  }
  /* C = 1.0; DT = 0.1; DD = 2.0; */
  for(y=1;y<HT-1;y++) {
    for(x=1;x<WD-1;x++) {
      WZ1->WZ1[y][x] = WZ0->WZ0[y][x]
	/* + C * C / 2.0 * DT * DT / (DD * DD) */
	+ 0.00125
	* (WZ0->WZ0[y+1][x] + WZ0->WZ0[y-1][x] + WZ0->WZ0[y][x+1] + WZ0->WZ0[y][x-1] - 4.0 * WZ0->WZ0[y][x]);
    }
  }

  for(y=0;y<WD;y++) {
    WZ1->WZ1[y][0]=0.0;
    WZ1->WZ1[y][WD-1]=0.0;
    WZ1->WZ1[0][y]=0.0;
    WZ1->WZ1[HT-1][y]=0.0;
  } WZ1->WZ1[HT/2][WD/2]=429496729;

  puts("wave2d-start");
  k = 1;
  /* value = C * C * DT * DT / (DD * DD) = 0.0025 */
#if 0
  for (z=2; z<DP; z++) {
#endif
    reset_nanosec();
    wave2d(WZ2->WZ2, WZ0->WZ0, WZ1->WZ1);
    get_nanosec(0);
    show_nanosec();
    for(y=0;y<WD;y++) {
      WZ2->WZ2[y][0]=0.0;
      WZ2->WZ2[y][WD-1]=0.0;
      WZ2->WZ2[0][y]=0.0;
      WZ2->WZ2[y][1]=0.0;
      WZ2->WZ2[HT-1][y]=0.0;
    }
    for(y=0;y<HT;y++) {
      for(x=0;x<WD;x++) {
	WZ0->WZ0[y][x]=WZ1->WZ1[y][x];
	WZ1->WZ1[y][x]=WZ2->WZ2[y][x];
      }
    } WZ1->WZ1[HT/2][WD/2]=229872*z;
#if 0
    if(z == k*DP/4-1){
#endif
      for(y=0;y<HT;y++){
	for(x=0;x<WD;x++){
	  W[WD*y+x] = (unsigned int)WZ2->WZ2[y][x];	
	  W[WD*y+x] = W[WD*y+x]<<8;
	  //printf("%d ", (unsigned int)WZ2->WZ2[y][x] );
	}
      }
#ifdef ARMSIML
      _copyX(4+((k-1)%4), W);
      _updateX();
#endif
#if !defined(ARMSIML)
      BGR_to_X(4+((k-1)%4), W);
      x11_update();
#endif
      k++;
#if 0
    }
#endif
#if 0
  }
#endif
  puts("wave2d-end");
#endif

#ifndef ARMSIML
  printf("==== Program normal end. ==== Hit any key in X.\n");
  while (!x11_checkevent());
#endif

  exit(0);
}

/* ---------------------------------------------------------------- */

grapes( float *c, float *a, float *b )
     /*D3D[DP][HT][WD]*/
     /*GrA[XC][DP][HT][WD]*/
     /*B3D[DP][HT][WD]*/
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     4
#define RMGRP     12
#define OMAP      1
#define PAD       1
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  int  x, y, z;
  int  row, col, n;
  Ull  roofs, coofs, aofs, bofs, cofs;

#if !defined(EMAX5) && !defined(EMAX6)
  for (z=PAD; z<DP-PAD; z++) {
    for (y=PAD; y<HT-PAD; y++) {
      for (x=PAD; x<WD-PAD; x++) {
	*(c+z*WDHT+y*WD+x) = *(b+(z-1)*WDHT+(y-1)*WD+x  ) * *(a+(MID-6)*WDHTDP+(z-1)*WDHT+(y-1)*WD+x  ) /* braw00 */ /* araw00 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x-1) * *(a+(MID-5)*WDHTDP+(z-1)*WDHT+(y  )*WD+x-1) /* braw01 */ /* araw01 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x  ) * *(a+(MID-4)*WDHTDP+(z-1)*WDHT+(y  )*WD+x  ) /* braw01 */ /* araw02 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x+1) * *(a+(MID-5)*WDHTDP+(z-1)*WDHT+(y  )*WD+x+1) /* braw01 */ /* araw01 */
                           + *(b+(z-1)*WDHT+(y+1)*WD+x  ) * *(a+(MID-3)*WDHTDP+(z-1)*WDHT+(y+1)*WD+x  ) /* braw02 */ /* araw03 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x-1) * *(a+(MID-2)*WDHTDP+(z  )*WDHT+(y-1)*WD+x-1) /* braw03 */ /* araw04 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x  ) * *(a+(MID-1)*WDHTDP+(z  )*WDHT+(y-1)*WD+x  ) /* braw03 */ /* araw05 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x+1) * *(a+(MID-2)*WDHTDP+(z  )*WDHT+(y-1)*WD+x+1) /* braw03 */ /* araw04 */
                           + *(b+(z  )*WDHT+(y  )*WD+x-1) * *(a+(MID  )*WDHTDP+(z  )*WDHT+(y  )*WD+x-1) /* braw04 */ /* araw06 */
                           + *(b+(z  )*WDHT+(y  )*WD+x  )                                               /* braw04 */
                           + *(b+(z  )*WDHT+(y  )*WD+x+1) * *(a+(MID  )*WDHTDP+(z  )*WDHT+(y  )*WD+x+1) /* braw04 */ /* araw06 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x-1) * *(a+(MID+2)*WDHTDP+(z  )*WDHT+(y+1)*WD+x-1) /* braw05 */ /* araw08 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x  ) * *(a+(MID+1)*WDHTDP+(z  )*WDHT+(y+1)*WD+x  ) /* braw05 */ /* araw07 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x+1) * *(a+(MID+2)*WDHTDP+(z  )*WDHT+(y+1)*WD+x+1) /* braw05 */ /* araw08 */
                           + *(b+(z+1)*WDHT+(y-1)*WD+x  ) * *(a+(MID+3)*WDHTDP+(z+1)*WDHT+(y-1)*WD+x  ) /* braw06 */ /* araw09 */
                           + *(b+(z+1)*WDHT+(y  )*WD+x-1) * *(a+(MID+5)*WDHTDP+(z+1)*WDHT+(y  )*WD+x-1) /* braw07 */ /* araw0b */
                           + *(b+(z+1)*WDHT+(y  )*WD+x  ) * *(a+(MID+4)*WDHTDP+(z+1)*WDHT+(y  )*WD+x  ) /* braw07 */ /* araw0a */
                           + *(b+(z+1)*WDHT+(y  )*WD+x+1) * *(a+(MID+5)*WDHTDP+(z+1)*WDHT+(y  )*WD+x+1) /* braw07 */ /* araw0b */
                           + *(b+(z+1)*WDHT+(y+1)*WD+x  ) * *(a+(MID+6)*WDHTDP+(z+1)*WDHT+(y+1)*WD+x  );/* braw08 */ /* araw0c */
      }
    }
  }
#else
  for (z=PAD; z<DP-PAD; z++) {
    for (y=0; y<RRANGE; y+=RMGRP) {
      Ull  atop[NCHIP], btop[NCHIP], ctop[NCHIP];
      Ull  arow00[NCHIP], arow01[NCHIP], arow02[NCHIP], arow03[NCHIP], arow04[NCHIP], arow05[NCHIP], arow06[NCHIP], arow07[NCHIP], arow08[NCHIP], arow09[NCHIP], arow0a[NCHIP], arow0b[NCHIP], arow0c[NCHIP];
      Ull  arowp0[NCHIP], arowp1[NCHIP], arowp2[NCHIP], arowp3[NCHIP], arowp4[NCHIP], arowp5[NCHIP], arowp6[NCHIP], arowp7[NCHIP], arowp8[NCHIP], arowp9[NCHIP], arowpa[NCHIP], arowpb[NCHIP], arowpc[NCHIP];
      Ull  brow00[NCHIP], brow01[NCHIP], brow02[NCHIP], brow03[NCHIP], brow04[NCHIP], brow05[NCHIP], brow06[NCHIP], brow07[NCHIP], brow08[NCHIP];
      Ull  browp0[NCHIP], browp3[NCHIP], browp6[NCHIP];
      Ull  crow0[NCHIP], crowp[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	atop[CHIP]   = a               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	btop[CHIP]   = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	ctop[CHIP]   = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	arow00[CHIP] = a+(MID-6)*WDHTDP+(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD; 
	/*arowp0[CHIP] = a+(MID-6)*WDHTDP+(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;*/
	arow01[CHIP] = a+(MID-5)*WDHTDP+(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	/*arowp1[CHIP] = a+(MID-5)*WDHTDP+(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  +RMGRP)*WD;*/
	arow02[CHIP] = a+(MID-4)*WDHTDP+(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD; 
	/*arowp2[CHIP] = a+(MID-4)*WDHTDP+(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  +RMGRP)*WD;*/
	arow03[CHIP] = a+(MID-3)*WDHTDP+(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD; 
	/*arowp3[CHIP] = a+(MID-3)*WDHTDP+(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1+RMGRP)*WD;*/
	arow04[CHIP] = a+(MID-2)*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD;
	/*arowp4[CHIP] = a+(MID-2)*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;*/
	arow05[CHIP] = a+(MID-1)*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD; 
	/*arowp5[CHIP] = a+(MID-1)*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;*/
	arow06[CHIP] = a+(MID  )*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD; 
	/*arowp6[CHIP] = a+(MID  )*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  +RMGRP)*WD;*/
	arow07[CHIP] = a+(MID+1)*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD; 
	/*arowp7[CHIP] = a+(MID+1)*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1+RMGRP)*WD;*/
	arow08[CHIP] = a+(MID+2)*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD; 
	/*arowp8[CHIP] = a+(MID+2)*WDHTDP+(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1+RMGRP)*WD;*/
	arow09[CHIP] = a+(MID+3)*WDHTDP+(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD; 
	/*arowp9[CHIP] = a+(MID+3)*WDHTDP+(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;*/
	arow0a[CHIP] = a+(MID+4)*WDHTDP+(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD; 
	/*arowpa[CHIP] = a+(MID+4)*WDHTDP+(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  +RMGRP)*WD;*/
	arow0b[CHIP] = a+(MID+5)*WDHTDP+(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD; 
	/*arowpb[CHIP] = a+(MID+5)*WDHTDP+(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  +RMGRP)*WD;*/
	arow0c[CHIP] = a+(MID+6)*WDHTDP+(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD; 
	/*arowpc[CHIP] = a+(MID+6)*WDHTDP+(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1+RMGRP)*WD;*/
	brow00[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD;
	brow01[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	brow02[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;
	/*browp0[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;*/
	brow03[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD;
	brow04[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	brow05[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;
	/*browp3[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;*/
	brow06[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD;
	brow07[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	brow08[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;
	/*browp6[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;*/
	crow0[CHIP]  = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	/*crowp[CHIP]  = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  -RMGRP)*WD;*/
      }
//EMAX5A begin grapes mapdist=1
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*2*/for (INIT1=1,LOOP1=RMGRP,roofs=0-AWD*4; LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
     /*1*/for (INIT0=1,LOOP0=AWD-PAD*2,coofs=(PAD-1)*4; LOOP0--; INIT0=0) {          /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,  &coofs, INIT0?coofs:coofs,  EXP_H3210, 4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD,  &roofs, roofs,  EXP_H3210, INIT0?AWD*4:0, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD3, &aofs,  atop[CHIP],  EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    exe(OP_ADD3, &bofs,  btop[CHIP],  EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    exe(OP_ADD3, &cofs,  ctop[CHIP],  EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    /*map0*/
	    mop(OP_LDWR, 1, &BR[2][0][1], bofs, (0               -WDHT-AWD  )*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#2 */
	    mop(OP_LDWR, 1, &BR[2][2][1], aofs, (0+WDHTDP*(MID-6)-WDHT-AWD  )*4, MSK_D0, arow00[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#2 */
	    exe(OP_FML, &r0, BR[2][0][1], EXP_H3210,  BR[2][2][1], EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][0][1], bofs, (0               -WDHT   -1)*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][0][0], bofs, (0               -WDHT   +1)*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][1][1], bofs, (0               -WDHT     )*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][2][1], aofs, (0+WDHTDP*(MID-5)-WDHT   -1)*4, MSK_D0, arow01[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][2][0], aofs, (0+WDHTDP*(MID-5)-WDHT   +1)*4, MSK_D0, arow01[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][3][1], aofs, (0+WDHTDP*(MID-4)-WDHT     )*4, MSK_D0, arow02[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#3 */
	    exe(OP_FMA, &r1, r0,          EXP_H3210,  BR[3][0][1], EXP_H3210, BR[3][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#4 */
	    exe(OP_FML, &r2, BR[3][0][0], EXP_H3210,  BR[3][2][0], EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#4 */
	    exe(OP_FML, &r3, BR[3][1][1], EXP_H3210,  BR[3][3][1], EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][0][1], bofs, (0               -WDHT+AWD  )*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][2][1], aofs, (0+WDHTDP*(MID-3)-WDHT+AWD  )*4, MSK_D0, arow03[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#4 */
	    exe(OP_FMA, &r4, r1,          EXP_H3210,  BR[4][0][1], EXP_H3210, BR[4][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#5 */
	    exe(OP_FAD, &r5, r2,          EXP_H3210,  r3,          EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#5 */

	    mop(OP_LDWR, 1, &BR[6][0][1], bofs, (0                    -AWD-1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][0][0], bofs, (0                    -AWD+1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][1][1], bofs, (0                    -AWD  )*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][2][1], aofs, (0+WDHTDP*(MID-2)     -AWD-1)*4, MSK_D0, arow04[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][2][0], aofs, (0+WDHTDP*(MID-2)     -AWD+1)*4, MSK_D0, arow04[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][3][1], aofs, (0+WDHTDP*(MID-1)     -AWD  )*4, MSK_D0, arow05[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#6 */
	    exe(OP_FMA, &r0, r4,          EXP_H3210,  BR[6][0][1], EXP_H3210, BR[6][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#7 */
 	    exe(OP_FMA, &r1, r5,          EXP_H3210,  BR[6][0][0], EXP_H3210, BR[6][2][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#7 */
	    exe(OP_FML, &r2, BR[6][1][1], EXP_H3210,  BR[6][3][1], EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][0][1], bofs, (0                       -1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][0][0], bofs, (0                       +1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][1][1], bofs, (0                         )*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][2][1], aofs, (0+WDHTDP*(MID  )        -1)*4, MSK_D0, arow06[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][2][0], aofs, (0+WDHTDP*(MID  )        +1)*4, MSK_D0, arow06[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#7 */
	    exe(OP_FMA, &r3, r0,          EXP_H3210,  BR[7][0][1], EXP_H3210, BR[7][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#8 */
	    exe(OP_FMA, &r4, r1,          EXP_H3210,  BR[7][0][0], EXP_H3210, BR[7][2][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#8 */
 	    exe(OP_FAD, &r5, r2,          EXP_H3210,  BR[7][1][1], EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][0][1], bofs, (0                    +AWD-1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][0][0], bofs, (0                    +AWD+1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][1][1], bofs, (0                    +AWD  )*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][2][1], aofs, (0+WDHTDP*(MID+2)     +AWD-1)*4, MSK_D0, arow08[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][2][0], aofs, (0+WDHTDP*(MID+2)     +AWD+1)*4, MSK_D0, arow08[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][3][1], aofs, (0+WDHTDP*(MID+1)     +AWD  )*4, MSK_D0, arow07[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#8 */
	    exe(OP_FMA, &r6, r3,          EXP_H3210,  BR[8][0][1], EXP_H3210, BR[8][2][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#9 */
 	    exe(OP_FMA, &r7, r4,          EXP_H3210,  BR[8][0][0], EXP_H3210, BR[8][2][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#9 */
	    exe(OP_FMA, &r8, r5,          EXP_H3210,  BR[8][1][1], EXP_H3210, BR[8][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#9 */

	    mop(OP_LDWR, 1, &BR[10][0][1],bofs, (0               +WDHT-AWD  )*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#10*/
	    mop(OP_LDWR, 1, &BR[10][2][1],aofs, (0+WDHTDP*(MID+3)+WDHT-AWD  )*4, MSK_D0, arow09[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#10*/
 	    exe(OP_FMA, &r0, r6,          EXP_H3210,  BR[10][0][1],EXP_H3210, BR[10][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#11*/
	    exe(OP_FAD, &r1, r7,          EXP_H3210,  r8,          EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#11*/
	    mop(OP_LDWR, 1, &BR[11][0][1],bofs, (0               +WDHT   -1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#11*/
	    mop(OP_LDWR, 1, &BR[11][0][0],bofs, (0               +WDHT   +1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#11*/
	    mop(OP_LDWR, 1, &BR[11][1][1],bofs, (0               +WDHT     )*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#11*/
	    mop(OP_LDWR, 1, &BR[11][2][1],aofs, (0+WDHTDP*(MID+5)+WDHT   -1)*4, MSK_D0, arow0b[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#11*/
	    mop(OP_LDWR, 1, &BR[11][2][0],aofs, (0+WDHTDP*(MID+5)+WDHT   +1)*4, MSK_D0, arow0b[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#11*/
	    mop(OP_LDWR, 1, &BR[11][3][1],aofs, (0+WDHTDP*(MID+4)+WDHT     )*4, MSK_D0, arow0a[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#11*/
	    exe(OP_FMA, &r2, r0,          EXP_H3210,  BR[11][0][1],EXP_H3210, BR[11][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#12*/
 	    exe(OP_FMA, &r3, r1,          EXP_H3210,  BR[11][0][0],EXP_H3210, BR[11][2][0],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#12*/
	    exe(OP_FML, &r4, BR[11][1][1],EXP_H3210,  BR[11][3][1],EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#12*/
	    mop(OP_LDWR, 1, &BR[12][0][1],bofs, (0               +WDHT+AWD  )*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, (Ull)NULL, AWD*(RMGRP+PAD*2));/* stage#12*/
	    mop(OP_LDWR, 1, &BR[12][2][1],aofs, (0+WDHTDP*(MID+6)+WDHT+AWD  )*4, MSK_D0, arow0c[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#12*/
	    exe(OP_FMA, &r5, r2,          EXP_H3210,  BR[12][0][1],EXP_H3210, BR[12][2][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#13*/
	    exe(OP_FAD, &r6, r3,          EXP_H3210,  r4,          EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#13*/
	    exe(OP_FAD, &r7, r5,          EXP_H3210,  r6,          EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);              /* stage#14*/
	    mop(OP_STWR, 3, &r7,          cofs, (0                         )*4, MSK_D0, crow0[CHIP],  AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP); /* stage#14*/
          }
        }
      }
//EMAX5A end
    }
//EMAX5A drain_dirty_lmm
  }
#endif
}

jacobi( float *c, float *b )
     /*D3D[DP][HT][WD]*/
     /*B3D[DP][HT][WD]*/
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     4
#define RMGRP     12
#define OMAP      1
#define PAD       1
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  int  x, y, z;
  int  row, col, n;
  Ull  roofs, coofs, aofs, bofs, cofs;
  union {float f; int i;} C1, C2;
  C1.f = 0.2;
  C2.f = 0.3;
  Ull  I1 = C1.i;
  Ull  I2 = C2.i;

#if !defined(EMAX5) && !defined(EMAX6)
  for (z=PAD; z<DP-PAD; z++) {
    for (y=PAD; y<HT-PAD; y++) {
      for (x=PAD; x<WD-PAD; x++) {
	*(c+z*WDHT+y*WD+x) = C2.f *(*(b+(z-1)*WDHT+(y  )*WD+x  )
		                  + *(b+(z  )*WDHT+(y-1)*WD+x  )
		                  + *(b+(z  )*WDHT+(y  )*WD+x-1)
	                          + *(b+(z  )*WDHT+(y  )*WD+x+1)
	                          + *(b+(z  )*WDHT+(y+1)*WD+x  )
	                          + *(b+(z+1)*WDHT+(y  )*WD+x  ))
	                   + C1.f * *(b+(z  )*WDHT+(y  )*WD+x  );
      }
    }
  }
#else
  for (z=PAD; z<DP-PAD; z++) {
    for (y=0; y<RRANGE; y+=RMGRP) {
      Ull  btop[NCHIP], ctop[NCHIP];
      Ull  brow00[NCHIP], brow01[NCHIP], brow02[NCHIP], brow03[NCHIP], brow04[NCHIP];
      Ull  browp0[NCHIP], browp1[NCHIP], browp2[NCHIP];
      Ull  crow0[NCHIP], crowp[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	btop[CHIP]   = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	ctop[CHIP]   = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	brow00[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	browp0[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	brow01[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	browp1[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	brow02[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD;
	brow03[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;/* not used for RMGRP>1 */
	brow04[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;/* not used for RMGRP>1 */
	browp2[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;
	crow0[CHIP]  = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	crowp[CHIP]  = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-RMGRP)*WD;
      }
//EMAX5A begin jacobi mapdist=7 /* 7 PAD>0の場合,PLOADとLOAD領域が一部重複.load中のLMMにもPLOADを取り込むために渋滞が発生する */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*2*/for (INIT1=1,LOOP1=RMGRP,roofs=0-AWD*4; LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
     /*1*/for (INIT0=1,LOOP0=AWD-PAD*2,coofs=(PAD-1)*4; LOOP0--; INIT0=0) {          /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,  &coofs, INIT0?coofs:coofs,  EXP_H3210, 4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD,  &roofs, roofs,  EXP_H3210, INIT0?AWD*4:0, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD3, &bofs,  btop[CHIP],  EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    exe(OP_ADD3, &cofs,  ctop[CHIP],  EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    /*map0*/
	    mop(OP_LDWR, 1, &BR[2][0][1], bofs, (0        -WDHT      )*4, MSK_D0, brow00[CHIP], AWD*RMGRP, 0, 0, browp0[CHIP], AWD*RMGRP);/* stage#2 */
	    mop(OP_LDWR, 1, &BR[2][2][1], bofs, (0        +WDHT      )*4, MSK_D0, brow01[CHIP], AWD*RMGRP, 0, 0, browp1[CHIP], AWD*RMGRP);/* stage#2 */
	    exe(OP_FAD, &r0, BR[2][0][1],EXP_H3210,  BR[2][2][1],EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][0][1], bofs, (0             -AWD  )*4, MSK_D0, brow02[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp2[CHIP], AWD*(RMGRP+PAD*2)); /* stage#3 */
	    exe(OP_FAD, &r1, r0,         EXP_H3210,  BR[3][0][1],EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#3 */
	    mop(OP_LDWR, 1, &BR[4][0][1], bofs, (0                 -1)*4, MSK_D0, brow02[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp2[CHIP], AWD*(RMGRP+PAD*2)); /* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][1][1], bofs, (0                   )*4, MSK_D0, brow02[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp2[CHIP], AWD*(RMGRP+PAD*2)); /* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][2][1], bofs, (0                 +1)*4, MSK_D0, brow02[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp2[CHIP], AWD*(RMGRP+PAD*2)); /* stage#4 */
	    exe(OP_FAD, &r2, r1,         EXP_H3210,  BR[4][0][1],EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#5 */
	    exe(OP_FML, &r3, I1,         EXP_H3210,  BR[4][1][1],EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#5 */
	    mop(OP_LDWR, 1, &BR[5][0][1], bofs, (0             +AWD  )*4, MSK_D0, brow02[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp2[CHIP], AWD*(RMGRP+PAD*2)); /* stage#5 */
	    exe(OP_FAD, &r4, r2,         EXP_H3210,  BR[5][0][1],EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#6 */
	    exe(OP_FAD, &r5, r4,         EXP_H3210,  BR[4][2][1],EXP_H3210, 0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#7 */
	    exe(OP_FMA, &r6, r3,         EXP_H3210,  r5,         EXP_H3210, I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#8 */
	    mop(OP_STWR, 3, &r6,          cofs, (0                   )*4, MSK_D0, crow0[CHIP],  AWD*RMGRP, 0, 0, crowp[CHIP], AWD*RMGRP); /* stage#8 */
          }
        }
      }
//EMAX5A end
    }
//EMAX5A drain_dirty_lmm
  }
#endif
}

fd6( float *c, float *b )
     /*D3D[DP][HT][WD]*/
     /*B3D[DP][HT][WD]*/
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     4
#define RMGRP     12
#define OMAP      1
#define PAD       3
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  int  x, y, z;
  int  row, col, n;
  Ull  roofs, coofs, aofs, bofs, cofs;
  union {float f; int i;} C1, C2, C3, C4;
  C1.f = 0.1;
  C2.f = 0.2;
  C3.f = 0.4;
  C4.f = 0.8;
  Ull  I1 = C1.i;
  Ull  I2 = C2.i;
  Ull  I3 = C3.i;
  Ull  I4 = C4.i;

#if !defined(EMAX5) && !defined(EMAX6)
  for (z=PAD; z<DP-PAD; z++) {
    for (y=PAD; y<HT-PAD; y++) {
      for (x=PAD; x<WD-PAD; x++) {
	*(c+z*WDHT+y*WD+x) = C4.f *(*(b+((z-3)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-3)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-3)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+3)
		                  + *(b+((z  )*WDHT)+(y+3)*WD+x  )
		                  + *(b+((z+3)*WDHT)+(y  )*WD+x  ))
                           + C3.f *(*(b+((z-2)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-2)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-2)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+2)
		                  + *(b+((z  )*WDHT)+(y+2)*WD+x  )
		                  + *(b+((z+2)*WDHT)+(y  )*WD+x  ))
                           + C2.f *(*(b+((z-1)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-1)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-1)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+1)
		                  + *(b+((z  )*WDHT)+(y+1)*WD+x  )
		                  + *(b+((z+1)*WDHT)+(y  )*WD+x  ))
                           + C1.f * *(b+((z  )*WDHT)+(y  )*WD+x  );
      }
    }
  }
#else
  for (z=PAD; z<DP-PAD; z++) {
    for (y=0; y<RRANGE; y+=RMGRP) {
      Ull  btop[NCHIP], ctop[NCHIP];
      Ull  brow00[NCHIP], brow01[NCHIP], brow02[NCHIP], brow03[NCHIP], brow04[NCHIP], brow05[NCHIP], brow06[NCHIP];
      Ull  brow07[NCHIP], brow08[NCHIP], brow09[NCHIP], brow0a[NCHIP], brow0b[NCHIP], brow0c[NCHIP];
      Ull  browp0[NCHIP], browp1[NCHIP], browp2[NCHIP], browp3[NCHIP], browp4[NCHIP], browp5[NCHIP], browp6[NCHIP];
      Ull  crow0[NCHIP], crowp[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	btop[CHIP]   = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	ctop[CHIP]   = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	brow00[CHIP] = b               +(z-3)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	browp0[CHIP] = b               +(z-3)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	brow01[CHIP] = b               +(z-2)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	browp1[CHIP] = b               +(z-2)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	brow02[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	browp2[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	brow03[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	browp3[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	brow04[CHIP] = b               +(z+2)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	browp4[CHIP] = b               +(z+2)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	brow05[CHIP] = b               +(z+3)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	browp5[CHIP] = b               +(z+3)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	brow06[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-3)*WD;
	brow07[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-2)*WD;/* not used for RMGRP>1 */
	brow08[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD;/* not used for RMGRP>1 */
	brow09[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;/* not used for RMGRP>1 */
	brow0a[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;/* not used for RMGRP>1 */
	brow0b[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+2)*WD;/* not used for RMGRP>1 */
	brow0c[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+3)*WD;/* not used for RMGRP>1 */
	browp6[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-3+RMGRP)*WD;
	crow0[CHIP]  = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	crowp[CHIP]  = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-RMGRP)*WD;
      }
//EMAX5A begin fd6 mapdist=11 /* 11 */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*2*/for (INIT1=1,LOOP1=RMGRP,roofs=0-AWD*4; LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
     /*1*/for (INIT0=1,LOOP0=AWD-PAD*2,coofs=(PAD-1)*4; LOOP0--; INIT0=0) {          /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,  &coofs, INIT0?coofs:coofs, EXP_H3210, 4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD,  &roofs, roofs, EXP_H3210, INIT0?AWD*4:0, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD3, &bofs,  btop[CHIP], EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    exe(OP_ADD3, &cofs,  ctop[CHIP], EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    /*map0*/
	    mop(OP_LDWR, 1, &BR[2][0][1], bofs, (0        -WDHT*3   )*4, MSK_D0, brow00[CHIP], AWD*RMGRP, 0, 0, browp0[CHIP], AWD*RMGRP);/* stage#2 */
	    mop(OP_LDWR, 1, &BR[2][2][1], bofs, (0        +WDHT*3   )*4, MSK_D0, brow05[CHIP], AWD*RMGRP, 0, 0, browp5[CHIP], AWD*RMGRP);/* stage#2 */
	    exe(OP_FAD, &r3, BR[2][0][1],EXP_H3210,  BR[2][2][1], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][0][1], bofs, (0        -WDHT*2   )*4, MSK_D0, brow01[CHIP], AWD*RMGRP, 0, 0, browp1[CHIP], AWD*RMGRP);/* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][2][1], bofs, (0        +WDHT*2   )*4, MSK_D0, brow04[CHIP], AWD*RMGRP, 0, 0, browp4[CHIP], AWD*RMGRP);/* stage#3 */
	    exe(OP_FAD, &r2, BR[3][0][1],EXP_H3210,  BR[3][2][1], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][0][1], bofs, (0        -WDHT*1   )*4, MSK_D0, brow02[CHIP], AWD*RMGRP, 0, 0, browp2[CHIP], AWD*RMGRP);/* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][2][1], bofs, (0        +WDHT*1   )*4, MSK_D0, brow03[CHIP], AWD*RMGRP, 0, 0, browp3[CHIP], AWD*RMGRP);/* stage#4 */
	    exe(OP_FAD, &r1, BR[4][0][1],EXP_H3210,  BR[4][2][1], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#5 */
	    mop(OP_LDWR, 1, &BR[5][0][1], bofs, (0            -AWD*3)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#5 */
	    mop(OP_LDWR, 1, &BR[5][0][0], bofs, (0            +AWD*3)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#5 */
	    mop(OP_LDWR, 1, &BR[5][1][1], bofs, (0                -3)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#5 */
	    mop(OP_LDWR, 1, &BR[5][1][0], bofs, (0                +3)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#5 */
	    exe(OP_FAD, &r13,BR[5][0][1],EXP_H3210,  BR[5][0][0], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#6 */
	    exe(OP_FAD, &r23,BR[5][1][1],EXP_H3210,  BR[5][1][0], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][0][1], bofs, (0            +AWD*2)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][0][0], bofs, (0            -AWD*2)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][1][1], bofs, (0                -2)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][1][0], bofs, (0                +2)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#6 */
	    exe(OP_FAD, &r12,BR[6][0][1],EXP_H3210,  BR[6][0][0], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#7 */
	    exe(OP_FAD, &r22,BR[6][1][1],EXP_H3210,  BR[6][1][0], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#7 */
	    exe(OP_FAD, &r23,r13,        EXP_H3210,  r23,         EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][0][1], bofs, (0            +AWD*1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][0][0], bofs, (0            -AWD*1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][1][1], bofs, (0                -1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][1][0], bofs, (0                +1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#7 */
	    exe(OP_FAD, &r11,BR[7][0][1],EXP_H3210,  BR[7][0][0], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#8 */
	    exe(OP_FAD, &r21,BR[7][1][1],EXP_H3210,  BR[7][1][0], EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#8 */
	    exe(OP_FAD, &r22,r12,        EXP_H3210,  r22,         EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#8 */
	    exe(OP_FAD, &r3, r23,        EXP_H3210,  r3,          EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][0][1], bofs, (0                  )*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2)); /* stage#8 */
	    exe(OP_FML, &r10,BR[8][0][1],EXP_H3210,  I1,          EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#9  */
	    exe(OP_FAD, &r21,r11,        EXP_H3210,  r21,         EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#9 */
	    exe(OP_FAD, &r2, r22,        EXP_H3210,  r2,          EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#9 */
	    exe(OP_FMA, &r13,r10,        EXP_H3210,  r3,          EXP_H3210, I4,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#10 */
	    exe(OP_FAD, &r1, r21,        EXP_H3210,  r1,          EXP_H3210, 0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#10 */
	    exe(OP_FMA, &r12,r13,        EXP_H3210,  r2,          EXP_H3210, I3,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#11 */
	    exe(OP_FMA, &r11,r12,        EXP_H3210,  r1,          EXP_H3210, I2,         EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#12 */
	    mop(OP_STWR, 3, &r11,        cofs, (0                   )*4, MSK_D0, crow0[CHIP],  AWD*RMGRP, 0, 0, crowp[CHIP], AWD*RMGRP); /* stage#12 */
          }
        }
      }
//EMAX5A end
    }
//EMAX5A drain_dirty_lmm
  }
#endif
}

resid( float *d, float *b, float *c )
     /*D3D[DP][HT][WD]*/
     /*B3D[DP][HT][WD]*/
     /*C3D[DP][HT][WD]*/
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     4
#define RMGRP     24
#define OMAP      1
#define PAD       1
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  int  x, y, z;
  int  row, col, n;
  Ull  roofs, coofs, bofs, cofs, dofs;
  union {float f; int i;} A0, A1, A2, A3;
  A0.f = -0.1;
  A1.f = -0.2;
  A2.f = -0.3;
  A3.f = -0.4;
  Ull  I0 = A0.i;
  Ull  I1 = A1.i;
  Ull  I2 = A2.i;
  Ull  I3 = A3.i;

#if !defined(EMAX5) && !defined(EMAX6)
  for (z=PAD; z<DP-PAD; z++) {
    for (y=PAD; y<HT-PAD; y++) {
      for (x=PAD; x<WD-PAD; x++) {
        *(d+z*WDHT+y*WD+x) = *(c+z*WDHT+y*WD+x)
	            + A0.f * *(b+(z  )*WDHT+(y  )*WD+x  )
                    + A1.f *(*(b+(z-1)*WDHT+(y  )*WD+x  )
			   + *(b+(z  )*WDHT+(y-1)*WD+x  )
			   + *(b+(z  )*WDHT+(y  )*WD+x-1)
			   + *(b+(z  )*WDHT+(y  )*WD+x+1)
			   + *(b+(z  )*WDHT+(y+1)*WD+x  )
		           + *(b+(z+1)*WDHT+(y  )*WD+x  ))
	            + A2.f *(*(b+(z-1)*WDHT+(y-1)*WD+x  )
		           + *(b+(z-1)*WDHT+(y  )*WD+x-1)
		           + *(b+(z-1)*WDHT+(y  )*WD+x+1)
			   + *(b+(z-1)*WDHT+(y+1)*WD+x  )
		           + *(b+(z  )*WDHT+(y-1)*WD+x-1)
		           + *(b+(z  )*WDHT+(y-1)*WD+x+1)
		           + *(b+(z  )*WDHT+(y+1)*WD+x-1)
		           + *(b+(z  )*WDHT+(y+1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x  )
		           + *(b+(z+1)*WDHT+(y  )*WD+x-1)
		           + *(b+(z+1)*WDHT+(y  )*WD+x+1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x  ))
	            + A3.f *(*(b+(z-1)*WDHT+(y-1)*WD+x-1)
		           + *(b+(z-1)*WDHT+(y-1)*WD+x+1)
		           + *(b+(z-1)*WDHT+(y+1)*WD+x-1)
		           + *(b+(z-1)*WDHT+(y+1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x-1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x-1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x+1));
      }
    }
  }
#else
  for (z=PAD; z<DP-PAD; z++) {
    for (y=0; y<RRANGE; y+=RMGRP) {
      Ull  btop[NCHIP], ctop[NCHIP], dtop[NCHIP];
      Ull  brow00[NCHIP], brow01[NCHIP], brow02[NCHIP], brow03[NCHIP], brow04[NCHIP], brow05[NCHIP], brow06[NCHIP], brow07[NCHIP], brow08[NCHIP];
      Ull  browp0[NCHIP], browp3[NCHIP], browp6[NCHIP];
      Ull  crow0[NCHIP], crowp[NCHIP];
      Ull  drow0[NCHIP], drowp[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
	btop[CHIP]   = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	ctop[CHIP]   = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	dtop[CHIP]   = d               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	brow00[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD; 
	brow01[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;/* not used for RMGRP>1 */
	brow02[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;/* not used for RMGRP>1 */
	browp0[CHIP] = b               +(z-1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;
	brow03[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD;
	brow04[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;/* not used for RMGRP>1 */
	brow05[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;/* not used for RMGRP>1 */
	browp3[CHIP] = b               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;
	brow06[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD; 
	brow07[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;/* not used for RMGRP>1 */
	brow08[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;/* not used for RMGRP>1 */
	browp6[CHIP] = b               +(z+1)*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;
	crow0[CHIP]  = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	crowp[CHIP]  = c               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
	drow0[CHIP]  = d               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
	drowp[CHIP]  = d               +(z  )*WDHT+(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-RMGRP)*WD;
      }
//EMAX5A begin resid mapdist=12 /* 12 */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*2*/for (INIT1=1,LOOP1=RMGRP,roofs=0-AWD*4; LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
     /*1*/for (INIT0=1,LOOP0=AWD-PAD*2,coofs=(PAD-1)*4; LOOP0--; INIT0=0) {          /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,  &coofs, INIT0?coofs:coofs, EXP_H3210, 4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD,  &roofs, roofs, EXP_H3210, INIT0?AWD*4:0, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	    exe(OP_ADD3, &bofs,  btop[CHIP], EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    exe(OP_ADD3, &cofs,  ctop[CHIP], EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    exe(OP_ADD3, &dofs,  dtop[CHIP], EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	    /*map0*/
	    mop(OP_LDWR, 1, &BR[2][0][1], bofs, (0       -WDHT-AWD-1)*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#2 */
	    mop(OP_LDWR, 1, &BR[2][0][0], bofs, (0       -WDHT-AWD  )*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#2 */
	    mop(OP_LDWR, 1, &BR[2][1][1], bofs, (0       -WDHT-AWD+1)*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#2 */
	    exe(OP_FML, &r0, BR[2][0][1], EXP_H3210,  I3,          EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#3 */
	    exe(OP_FML, &r1, BR[2][0][0], EXP_H3210,  I2,          EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#3 */
	    exe(OP_FML, &r2, BR[2][1][1], EXP_H3210,  I3,          EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][0][1], bofs, (0       -WDHT   -1)*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][0][0], bofs, (0       -WDHT     )*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#3 */
	    mop(OP_LDWR, 1, &BR[3][1][1], bofs, (0       -WDHT   +1)*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#3 */
	    exe(OP_FMA, &r3, r0,          EXP_H3210,  BR[3][0][1], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#4 */
	    exe(OP_FMA, &r4, r1,          EXP_H3210,  BR[3][0][0], EXP_H3210,  I1,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#4 */
	    exe(OP_FMA, &r5, r2,          EXP_H3210,  BR[3][1][1], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][0][1], bofs, (0      -WDHT+AWD-1)*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][0][0], bofs, (0      -WDHT+AWD  )*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#4 */
	    mop(OP_LDWR, 1, &BR[4][1][1], bofs, (0      -WDHT+AWD+1)*4, MSK_D0, brow00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp0[CHIP], AWD*(RMGRP+PAD*2));/* stage#4 */
	    exe(OP_FMA, &r6, r3,          EXP_H3210,  BR[4][0][1], EXP_H3210,  I3,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#5 */
	    exe(OP_FMA, &r7, r4,          EXP_H3210,  BR[4][0][0], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#5 */
	    exe(OP_FMA, &r8, r5,          EXP_H3210,  BR[4][1][1], EXP_H3210,  I3,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#5 */

	    mop(OP_LDWR, 1, &BR[5][0][1], bofs, (0           -AWD-1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#5 */
	    mop(OP_LDWR, 1, &BR[5][0][0], bofs, (0           -AWD  )*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#5 */
	    mop(OP_LDWR, 1, &BR[5][1][1], bofs, (0           -AWD+1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#5 */
	    exe(OP_FMA, &r0, r6,          EXP_H3210,  BR[5][0][1], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#6 */
	    exe(OP_FMA, &r1, r7,          EXP_H3210,  BR[5][0][0], EXP_H3210,  I1,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#6 */
	    exe(OP_FMA, &r2, r8,          EXP_H3210,  BR[5][1][1], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][0][1], bofs, (0               -1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][0][0], bofs, (0                 )*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#6 */
	    mop(OP_LDWR, 1, &BR[6][1][1], bofs, (0               +1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#6 */
	    exe(OP_FMA, &r3, r0,          EXP_H3210,  BR[6][0][1], EXP_H3210,  I1,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#7 */
	    exe(OP_FMA, &r4, r1,          EXP_H3210,  BR[6][0][0], EXP_H3210,  I0,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#7 */
	    exe(OP_FMA, &r5, r2,          EXP_H3210,  BR[6][1][1], EXP_H3210,  I1,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][0][1], bofs, (0           +AWD-1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][0][0], bofs, (0           +AWD  )*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#7 */
	    mop(OP_LDWR, 1, &BR[7][1][1], bofs, (0           +AWD+1)*4, MSK_D0, brow03[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp3[CHIP], AWD*(RMGRP+PAD*2));/* stage#7 */
	    exe(OP_FMA, &r6, r3,          EXP_H3210,  BR[7][0][1], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#8 */
	    exe(OP_FMA, &r7, r4,          EXP_H3210,  BR[7][0][0], EXP_H3210,  I1,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#8 */
	    exe(OP_FMA, &r8, r5,          EXP_H3210,  BR[7][1][1], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#8 */

	    mop(OP_LDWR, 1, &BR[8][0][1], bofs, (0      +WDHT-AWD-1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][0][0], bofs, (0      +WDHT-AWD  )*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#8 */
	    mop(OP_LDWR, 1, &BR[8][1][1], bofs, (0      +WDHT-AWD+1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#8 */
	    exe(OP_FMA, &r0, r6,          EXP_H3210,  BR[8][0][1], EXP_H3210,  I3,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#9 */
	    exe(OP_FMA, &r1, r7,          EXP_H3210,  BR[8][0][0], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#9 */
	    exe(OP_FMA, &r2, r8,          EXP_H3210,  BR[8][1][1], EXP_H3210,  I3,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#9 */
	    mop(OP_LDWR, 1, &BR[9][0][1], bofs, (0       +WDHT   -1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#9 */
	    mop(OP_LDWR, 1, &BR[9][0][0], bofs, (0       +WDHT     )*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#9 */
	    mop(OP_LDWR, 1, &BR[9][1][1], bofs, (0       +WDHT   +1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#9 */
	    exe(OP_FMA, &r3, r0,          EXP_H3210,  BR[9][0][1], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#10*/
	    exe(OP_FMA, &r4, r1,          EXP_H3210,  BR[9][0][0], EXP_H3210,  I1,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#10*/
	    exe(OP_FMA, &r5, r2,          EXP_H3210,  BR[9][1][1], EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#10*/
	    mop(OP_LDWR, 1, &BR[10][0][1],bofs, (0      +WDHT+AWD-1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#10*/
	    mop(OP_LDWR, 1, &BR[10][0][0],bofs, (0      +WDHT+AWD  )*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#10*/
	    mop(OP_LDWR, 1, &BR[10][1][1],bofs, (0      +WDHT+AWD+1)*4, MSK_D0, brow06[CHIP], AWD*(RMGRP+PAD*2), 0, 0, browp6[CHIP], AWD*(RMGRP+PAD*2));/* stage#10*/
	    exe(OP_FMA, &r6, r3,          EXP_H3210,  BR[10][0][1],EXP_H3210,  I3,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#11*/
	    exe(OP_FMA, &r7, r4,          EXP_H3210,  BR[10][0][0],EXP_H3210,  I2,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#11*/
	    exe(OP_FMA, &r8, r5,          EXP_H3210,  BR[10][1][1],EXP_H3210,  I3,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#11*/
	    mop(OP_LDWR, 1, &BR[11][0][1],cofs, (0                 )*4, MSK_D0, crow0[CHIP],  AWD*RMGRP, 0, 0, crowp[CHIP], AWD*RMGRP);/* stage#11*/
	    exe(OP_FAD, &r1, r6,          EXP_H3210,  BR[11][0][1],EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#12*/
	    exe(OP_FAD, &r2, r7,          EXP_H3210,  r8,          EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#12*/
	    exe(OP_FAD, &r0, r1,          EXP_H3210,  r2,          EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#13*/
	    mop(OP_STWR, 3, &r0,          dofs, (0                 )*4, MSK_D0, drow0[CHIP],  AWD*RMGRP, 0, 0, drowp[CHIP], AWD*RMGRP);/* stage#13*/
          }
        }
      }
//EMAX5A end
    }
//EMAX5A drain_dirty_lmm
  }
#endif
}

wave2d( float *z2, float *z0, float *z1 )
     /*WZ2[HT][WD]*/
     /*WZ0[HT][WD]*/
     /*WZ1[HT][WD]*/
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     4
#define RMGRP     24
#define OMAP      1
#define PAD       1
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  int  x, y, z;
  int  row, col, n;
  Ull  roofs, coofs, z0ofs, z1ofs, z2ofs;
  union {float f; int i;} C1, C2, C3, C4;
  C1.f =  2.00;
  C2.f = -1.00;
  C3.f =  0.25;
  C4.f = -4.00;
  Ull  I1 = C1.i;
  Ull  I2 = C2.i;
  Ull  I3 = C3.i;
  Ull  I4 = C4.i;

#if !defined(EMAX5) && !defined(EMAX6)
  for (y=PAD; y<HT-PAD; y++) {
    for (x=PAD; x<WD-PAD; x++) {
      *(z2+y*WD+x) =  C1.f * *(z1+y*WD+x)
	           +  C2.f * *(z0+y*WD+x)
	           +  C3.f *(*(z1+(y+1)*WD+x  )
	                   + *(z1+(y-1)*WD+x  )
	                   + *(z1+(y  )*WD+x-1)
	                   + *(z1+(y  )*WD+x+1) + C4.f * *(z1+y*WD+x));
    }
  }
#else
  for (y=0; y<RRANGE; y+=RMGRP) {
    Ull  z0top[NCHIP], z1top[NCHIP], z2top[NCHIP];
    Ull  z0row0[NCHIP], z0rowp[NCHIP];
    Ull  z1row00[NCHIP], z1row01[NCHIP], z1row02[NCHIP], z1rowp0[NCHIP];
    Ull  z2row0[NCHIP], z2rowp[NCHIP];
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
      z0top[CHIP]   = z0              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
      z1top[CHIP]   = z1              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
      z2top[CHIP]   = z2              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
      z0row0[CHIP]  = z0              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
      z0rowp[CHIP]  = z0              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+RMGRP)*WD;
      z1row00[CHIP] = z1              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1)*WD;
      z1row01[CHIP] = z1              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;/* not used for RMGRP>1 */
      z1row02[CHIP] = z1              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y+1)*WD;/* not used for RMGRP>1 */
      z1rowp0[CHIP] = z1              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-1+RMGRP)*WD;
      z2row0[CHIP]  = z2              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y  )*WD;
      z2rowp[CHIP]  = z2              +(CHIP*RRANGE*OMAP+RRANGE*0+PAD+y-RMGRP)*WD;
    }
//EMAX5A begin wave2d mapdist=8 /* 8 */
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
 /*2*/for (INIT1=1,LOOP1=RMGRP,roofs=0-AWD*4; LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
   /*1*/for (INIT0=1,LOOP0=AWD-PAD*2,coofs=(PAD-1)*4; LOOP0--; INIT0=0) {          /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
          exe(OP_ADD,  &coofs, INIT0?coofs:coofs, EXP_H3210, 4, EXP_H3210,  0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	  exe(OP_ADD,  &roofs, roofs, EXP_H3210, INIT0?AWD*4:0, EXP_H3210,  0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
	  exe(OP_ADD3, &z0ofs, z0top[CHIP], EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	  exe(OP_ADD3, &z1ofs, z1top[CHIP], EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	  exe(OP_ADD3, &z2ofs, z2top[CHIP], EXP_H3210, roofs, EXP_H3210,  coofs, EXP_H3210, OP_AND, 0x000000ffffffffffLL, OP_NOP, 0LL); /* stage#1 */
	  /*map0*/
	  mop(OP_LDWR, 1, &BR[2][0][1], z0ofs, (0                  )*4, MSK_D0, z0row0[CHIP],  AWD*RMGRP, 0, 0, z0rowp[CHIP], AWD*RMGRP); /* stage#2 */
	  exe(OP_FML, &r0, BR[2][0][1], EXP_H3210,  I2,          EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#3 */
	  mop(OP_LDWR, 1, &BR[3][0][1], z1ofs, (0            -AWD  )*4, MSK_D0, z1row00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, z1rowp0[CHIP], AWD*(RMGRP+PAD*2)); /* stage#3 */
	  mop(OP_LDWR, 1, &BR[4][0][1], z1ofs, (0                -1)*4, MSK_D0, z1row00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, z1rowp0[CHIP], AWD*(RMGRP+PAD*2)); /* stage#4 */
	  mop(OP_LDWR, 1, &BR[4][0][0], z1ofs, (0                  )*4, MSK_D0, z1row00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, z1rowp0[CHIP], AWD*(RMGRP+PAD*2)); /* stage#4 */
	  mop(OP_LDWR, 1, &BR[4][1][1], z1ofs, (0                +1)*4, MSK_D0, z1row00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, z1rowp0[CHIP], AWD*(RMGRP+PAD*2)); /* stage#4 */
	  exe(OP_FAD, &r1, BR[3][0][1], EXP_H3210,  BR[4][0][1], EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#5 */
	  mop(OP_LDWR, 1, &BR[5][0][1], z1ofs, (0            +AWD  )*4, MSK_D0, z1row00[CHIP], AWD*(RMGRP+PAD*2), 0, 0, z1rowp0[CHIP], AWD*(RMGRP+PAD*2)); /* stage#5 */
	  exe(OP_FMA, &r2, r1,          EXP_H3210,  BR[4][0][0], EXP_H3210,  I4,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#6 */
	  exe(OP_FAD, &r3, BR[4][1][1], EXP_H3210,  BR[5][0][1], EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#6 */
	  exe(OP_FAD, &r4, r2,          EXP_H3210,  r3,          EXP_H3210,  0,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#7 */
	  exe(OP_FMA, &r5, r0,          EXP_H3210,  r4,          EXP_H3210,  I3,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#8 */
	  exe(OP_FMA, &r6, r5,          EXP_H3210,  BR[4][0][0], EXP_H3210,  I1,          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           /* stage#9 */
	  mop(OP_STWR, 3, &r6,          z2ofs, (0                  )*4, MSK_D0, z2row0[CHIP],  AWD*RMGRP, 0, 0, z2rowp[CHIP], AWD*RMGRP); /* stage#9 */
        }
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm
#endif
}
