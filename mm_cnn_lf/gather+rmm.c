
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/* Gather data from light-field-camera and display image */
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
#elif defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
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
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // ¡ú¡ú¡ú RESET
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

#define IM  7500
/* height=5433 */
#define OM  1600
#define OMAP  6
/*#define NCHIP 4*/
#define NCHIP 4
#define RRANGE ((OM-PAD*2)/NCHIP/OMAP)
#define CRANGE ((OM-PAD*2))
/* height=1200 */
#define R   75
#define PAD 32
#define MAXDELTA  4  /* -3,-2,-1,0,1,2,3 */
#define WBASE    (MAXDELTA*MAXDELTA*2)
#define ofs    14
#define delta ((R/2/(14+1)-1) ? (R/2/(14+1)-1) : 1)
int weight[MAXDELTA*MAXDELTA*2*2];
int total_weight;  /* 7 x+3   +60   (+3,+3) */
                     /* 6 x+2   +52           */
                     /* 5 x+1   +44           */
                     /* 4 x 0   +36*-center   */
                     /* 3 x-1   +28           */
                     /* 2 x-2   +20  MAXDELTA */
                     /* 1 x-3   +12 /         */
                     /* 0 x x x x x x x x     */
                     /*           0->         */
                     /*   0 1 2 3 4 5 6 7     */
Uchar *rgb; /*[IM*IM*3];*/
Uint *in;   /*[IM*IM];*/
Uint *out0; /*[OM*OM];*/
Uint *out1; /*[OM*OM];*/
Uint *ip, *wp, *op;
int c, s, i, j, p0, p1;
int row, col, oc, rx, ry, y, x;
int count0, count1, count2;

#define CSIMWD 320
#define CSIMHT 240
#define CSIMBM (CSIMWD*CSIMHT)
Uint Z[CSIMBM];

main(argc, argv)
     int argc;
     char **argv;
{
  FILE *fp;
  int fd;
  char dummy[16];

  sysinit(IM*IM*3
	 +IM*IM*sizeof(int)
	 +OM*OM*sizeof(int)
         +OM*OM*sizeof(int),32);
  printf("membase: %08.8x\n", (Uint)membase);
  rgb  = (Uchar*)membase;
  in   = (Uint*)((Uchar*)rgb  + IM*IM*3);
  out0 = (Uint*)((Uchar*)in   + IM*IM*sizeof(int));
  out1 = (Uint*)((Uchar*)out0 + OM*OM*sizeof(int));
  printf("irgb: %08.8x\n", rgb);
  printf("in  : %08.8x\n", in);
  printf("out0: %08.8x\n", out0);
  printf("out1: %08.8x\n", out1);

#if 0
  if ((fp = fopen("../4dimage/472.pnm", "r")) == NULL) {
    printf("can't open ../4dimage/472.pnm\n");
    exit(1);
  }
  fgets(dummy, 3, fp);
  fscanf(fp, "%d %d\n", &i, &j); /* width, height */
  fscanf(fp, "%d\n", &i);        /* grad */
  fread(rgb, 7240*5433*3, 1, fp);
  printf("reading pnm_file 1stRGB=%02x%02x%02x\n", rgb[0], rgb[1], rgb[2]);
  fclose(fp);
  for (i=0; i<5433; i++) {
    for (j=0; j<7240; j++) {
      in[(i+75-56)*IM+(j+75-30)] = *(rgb+2)<<24 | *(rgb+1)<<16 | *rgb<<8;
      rgb+=3;
    }
  }
#else
  if ((fd = open("472.ini", O_RDONLY)) < 0) {
    printf("can't open 472.ini\n");
    exit(1);
  }
  read(fd, in, IM*IM*4);
  printf("reading init_file 1stWORD=%08.8x\n", in[0]);
  close(fd);
#endif

#if !defined(ARMSIML)
  x11_open(0);
#endif

  total_weight=0;
  for (i=-delta; i<=delta; i++) {
    for (j=-delta; j<=delta; j++) {
      weight[WBASE+i*MAXDELTA*2+j] = delta*delta*4/(abs(i)+abs(j)+1);
      total_weight += (weight[WBASE+i*MAXDELTA*2+j] = delta*delta*4/(abs(i)+abs(j)+1));
    }
  }
  for (i=-delta; i<=delta; i++) {
    for (j=-delta; j<=delta; j++) {
      weight[WBASE+i*MAXDELTA*2+j] = weight[WBASE+i*MAXDELTA*2+j]*256/total_weight;
    }
  }

#if 1
  reset_nanosec();
  orig();
  get_nanosec(0);
  show_nanosec();
#endif

  reset_nanosec();
  imax();
  get_nanosec(0);
  show_nanosec();

#ifdef ARMSIML
  copy_Z( 0, out1); _copyX(0, Z);
  copy_Z( 1, out1); _copyX(1, Z);
  copy_Z( 2, out1); _copyX(2, Z);
  copy_Z( 3, out1); _copyX(3, Z);

  copy_Z(10, out1); _copyX(4, Z);
  copy_Z(11, out1); _copyX(5, Z);
  copy_Z(12, out1); _copyX(6, Z);
  copy_Z(13, out1); _copyX(7, Z);

  copy_Z(20, out1); _copyX(8, Z);
  copy_Z(21, out1); _copyX(9, Z);
  copy_Z(22, out1); _copyX(10,Z);
  copy_Z(23, out1); _copyX(11,Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, out1); BGR_to_X(0, Z);
  copy_Z(1, out1); BGR_to_X(1, Z);
  copy_Z(2, out1); BGR_to_X(2, Z);
  copy_Z(3, out1); BGR_to_X(3, Z);
  copy_Z(4, out1); BGR_to_X(4, Z);
  copy_Z(5, out1); BGR_to_X(5, Z);
  copy_Z(6, out1); BGR_to_X(6, Z);
  copy_Z(7, out1); BGR_to_X(7, Z);
  copy_Z(8, out1); BGR_to_X(8 ,Z);
  copy_Z(9, out1); BGR_to_X(9 ,Z);
  copy_Z(10,out1); BGR_to_X(10,Z);
  copy_Z(11,out1); BGR_to_X(11,Z);
  copy_Z(12,out1); BGR_to_X(12,Z);
  copy_Z(13,out1); BGR_to_X(13,Z);
  copy_Z(14,out1); BGR_to_X(14,Z);
  copy_Z(15,out1); BGR_to_X(15,Z);
  copy_Z(16,out1); BGR_to_X(16,Z);
  copy_Z(17,out1); BGR_to_X(17,Z);
  copy_Z(18,out1); BGR_to_X(18,Z);
  copy_Z(19,out1); BGR_to_X(19,Z);
  copy_Z(20,out1); BGR_to_X(20,Z);
  copy_Z(21,out1); BGR_to_X(21,Z);
  copy_Z(22,out1); BGR_to_X(22,Z);
  copy_Z(23,out1); BGR_to_X(23,Z);
  copy_Z(24,out1); BGR_to_X(24,Z);
  x11_update();
#endif

  printf("Num of MULT: orig=%d imax=%d\n", count0, count1);

  for (row=1; row<OM-1; row++) {
    for (col=1; col<OM-1; col++) {
      if (out0[row*OM+col] != out1[row*OM+col]) {
	count2++;
	printf("o0[%d]=%x o1[%d]=%x\n",
	       row*OM+col, out0[row*OM+col],
	       row*OM+col, out1[row*OM+col]);
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

  switch (id) {
  case 0:                         break;
  case 1:  from += WD;            break;
  case 2:  from += WD*2;          break;
  case 3:  from += WD*3;          break;
  case 4:  from += WD*4;          break;
  case 5:  from += BITMAP*5;      break;
  case 6:  from += BITMAP*5+WD;   break;
  case 7:  from += BITMAP*5+WD*2; break;
  case 8:  from += BITMAP*5+WD*3; break;
  case 9:  from += BITMAP*5+WD*4; break;
  case 10: from += BITMAP*10;     break;
  case 11: from += BITMAP*10+WD;  break;
  case 12: from += BITMAP*10+WD*2;break;
  case 13: from += BITMAP*10+WD*3;break;
  case 14: from += BITMAP*10+WD*4;break;
  case 15: from += BITMAP*15;     break;
  case 16: from += BITMAP*15+WD;  break;
  case 17: from += BITMAP*15+WD*2;break;
  case 18: from += BITMAP*15+WD*3;break;
  case 19: from += BITMAP*15+WD*4;break;
  case 20: from += BITMAP*20;     break;
  case 21: from += BITMAP*20+WD;  break;
  case 22: from += BITMAP*20+WD*2;break;
  case 23: from += BITMAP*20+WD*3;break;
  case 24: from += BITMAP*20+WD*4;break;
  }
  for (i=0; i<HT; i++, from+=WD*4) {
    for (j=0; j<WD; j++) {
      *to++ = *from++;
    }
  }
}

orig()
{
  printf("<<<ORIG>>>\n");
  ry = (R+ofs)*IM;
  rx = (R+ofs);
  int w, pix, cvalR, cvalG, cvalB;

  for (row=PAD; row<OM-PAD; row++) {
    for (col=PAD; col<OM-PAD; col++) {
       c = ((row>>4)*R + (((~row&15)*ofs)>>4))*IM
	  + (col>>4)*R + (((~col&15)*ofs)>>4);
      cvalR=0;
      cvalG=0;
      cvalB=0;
      for (i=-1; i<=1; i++) {
	for (j=-1; j<=1; j++) {
	  Uint pix = in[c+ry*i+rx*j];
	  w = weight[WBASE+i*MAXDELTA*2+j];
	  cvalB += ((pix>>24)&255)*w;
	  cvalG += ((pix>>16)&255)*w;
	  cvalR += ((pix>> 8)&255)*w;
	  count0++;
	}
      }
      out0[row*OM+col] = ((cvalB>>8)<<24) | ((cvalG>>8)<<16) | ((cvalR>>8)<<8);
    }
  }
}

#if 0
imax()
{
  Ull CHIP;
  printf("<<<IMAX>>>\n");
  ry = (R+ofs)*IM;
  rx = (R+ofs);
  int w, pix, cvalR, cvalG, cvalB;

  for (row=0; row<RRANGE; row++) { /* 0..381 */
    for (CHIP=0; CHIP<NCHIP; CHIP++) {
      for (col=0; col<CRANGE; col++) {
	for (oc=0; oc<OMAP; oc++) {
	  c =((((CHIP*OMAP+oc)*RRANGE+row+PAD)>>4)*R + (((~((CHIP*OMAP+oc)*RRANGE+row+PAD)&15)*ofs)>>4))*IM
	    + ((                      col+PAD)>>4)*R + (((~(                      col+PAD)&15)*ofs)>>4);
	  /* 256  512 256 */
	  pix = in[c+ry*(-1)+rx*(-1)]; w = 16; cvalB =((pix>>24)&255)*w; cvalG =((pix>>16)&255)*w; cvalR =((pix>> 8)&255)*w;
	  pix = in[c+ry*(-1)+rx*( 0)]; w = 32; cvalB+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalR+=((pix>> 8)&255)*w;
	  pix = in[c+ry*(-1)+rx*( 1)]; w = 16; cvalB+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalR+=((pix>> 8)&255)*w;
	  /* 512 1024 512 */
	  pix = in[c+ry*( 0)+rx*(-1)]; w = 32; cvalB+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalR+=((pix>> 8)&255)*w;
	  pix = in[c+ry*( 0)+rx*( 0)]; w = 64; cvalB+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalR+=((pix>> 8)&255)*w;
	  pix = in[c+ry*( 0)+rx*( 1)]; w = 32; cvalB+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalR+=((pix>> 8)&255)*w;
	  /* 256  512 256 */
	  pix = in[c+ry*( 1)+rx*(-1)]; w = 16; cvalB+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalR+=((pix>> 8)&255)*w;
	  pix = in[c+ry*( 1)+rx*( 0)]; w = 32; cvalB+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalR+=((pix>> 8)&255)*w;
	  pix = in[c+ry*( 1)+rx*( 1)]; w = 16; cvalB+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalR+=((pix>> 8)&255)*w;
	  count1+=9;
	  out1[((CHIP*OMAP+oc)*RRANGE+row+PAD)*OM+(col+PAD)] = ((cvalB>>8)<<24) | ((cvalG>>8)<<16) | ((cvalR>>8)<<8);
	}
      }
    }
  }
}

#else

imax()
{
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  c0, c1, c2, c3, ex0, ex1;
  Ull  x[NCHIP];

  ry = (R+ofs)*IM;
  rx = (R+ofs);
  /* ¥Þ¥ë¥Á¥Á¥Ã¥×¤Îcycle¿ô¤Ï¡¤yÊý¸þ¤Î¥µ¥¤¥º¤ò1/m¤Ë¤·¤¿cycle¿ô¤ÈÆ±¤¸ */
  /* ¤¿¤À¤·¥Ç¡¼¥¿¶¡µë¤Ï¥·¥ê¥¢¥é¥¤¥º */
  /* 1chipÆâ¤Ç¤Ï²èÌÌ¤òyÊý¸þ¤Ë4Ê¬³ä */
  /* +-----0-----------------------------------+ */
  /* <-in--------------------------------------> */  
  /* |0- 168    1958   3750    5540   7332-7499| */
  /* |   36                            1563    | */
  /* |   +-PAD-----------------36----------+   | */
  /* |   |                                 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢chip3 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢chip2 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢chip1 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢chip0 |   | */
  /* |   |                                 |   | */
  /* |   |                                 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢¶¦Í­  |   | */
  /* |   |                                 |   | */
  /* |   |                                 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢chip0 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢chip1 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢chip2 |   | */
  /* |   |     ¢¢¢¢¢¢  ¢¢¢¢¢¢  ¢¢¢¢¢¢chip3 |   | */
  /* |   |                                 |   | */
  /* |   +---------------------1563--------+   | */
  /* |     OM-PAD                              | */
  /* +-----------------------------------------+ */
  /*       OM                                    */
  printf("<<<IMAX>>>\n");
  Uint *ym_xm   = in         -ry-rx;
  Uint *ym_xz   = in         -ry;
  Uint *ym_xp   = in         -ry+rx;
  Uint *yz_xm   = in            -rx;
  Uint *yz_xz   = in;
  Uint *yz_xp   = in            +rx;
  Uint *yp_xm   = in         +ry-rx;
  Uint *yp_xz   = in         +ry;
  Uint *yp_xp   = in         +ry+rx;

  for (row=RRANGE-1; row>=0; row--) {
    int yin0[NCHIP];
    Uint *acci_ym0[NCHIP];
    Uint *acci_yz0[NCHIP];
    Uint *acci_yp0[NCHIP];
    Uint *acco_base0[NCHIP];  Uint *acco0[NCHIP];
    int yin1[NCHIP];
    Uint *acci_ym1[NCHIP];
    Uint *acci_yz1[NCHIP];
    Uint *acci_yp1[NCHIP];
    Uint *acco_base1[NCHIP];  Uint *acco1[NCHIP];
    int yin2[NCHIP];
    Uint *acci_ym2[NCHIP];
    Uint *acci_yz2[NCHIP];
    Uint *acci_yp2[NCHIP];
    Uint *acco_base2[NCHIP];  Uint *acco2[NCHIP];
    int yin3[NCHIP];
    Uint *acci_ym3[NCHIP];
    Uint *acci_yz3[NCHIP];
    Uint *acci_yp3[NCHIP];
    Uint *acco_base3[NCHIP];  Uint *acco3[NCHIP];
    int yin4[NCHIP];
    Uint *acci_ym4[NCHIP];
    Uint *acci_yz4[NCHIP];
    Uint *acci_yp4[NCHIP];
    Uint *acco_base4[NCHIP];  Uint *acco4[NCHIP];
    int yin5[NCHIP];
    Uint *acci_ym5[NCHIP];
    Uint *acci_yz5[NCHIP];
    Uint *acci_yp5[NCHIP];
    Uint *acco_base5[NCHIP];  Uint *acco5[NCHIP];

    for (CHIP=0; CHIP<NCHIP; CHIP++) {
      int row0 = (CHIP*OMAP+0)*RRANGE+row+PAD;
      int yout0 = row0*OM;
      yin0[CHIP] = ((row0>>4)*R + (((~row0&15)*ofs)>>4))*IM;
      acci_ym0[CHIP] = in+yin0[CHIP]     -ry;
      acci_yz0[CHIP] = in+yin0[CHIP];
      acci_yp0[CHIP] = in+yin0[CHIP]     +ry;
      acco_base0[CHIP] = out1+yout0+PAD;  acco0[CHIP] = out1+yout0+PAD;
      int row1 = (CHIP*OMAP+1)*RRANGE+row+PAD;
      int yout1 = row1*OM;
      yin1[CHIP] = ((row1>>4)*R + (((~row1&15)*ofs)>>4))*IM;
      acci_ym1[CHIP] = in+yin1[CHIP]     -ry;
      acci_yz1[CHIP] = in+yin1[CHIP];
      acci_yp1[CHIP] = in+yin1[CHIP]     +ry;
      acco_base1[CHIP] = out1+yout1+PAD;  acco1[CHIP] = out1+yout1+PAD;
      int row2 = (CHIP*OMAP+2)*RRANGE+row+PAD;
      int yout2 = row2*OM;
      yin2[CHIP] = ((row2>>4)*R + (((~row2&15)*ofs)>>4))*IM;
      acci_ym2[CHIP] = in+yin2[CHIP]     -ry;
      acci_yz2[CHIP] = in+yin2[CHIP];
      acci_yp2[CHIP] = in+yin2[CHIP]     +ry;
      acco_base2[CHIP] = out1+yout2+PAD;  acco2[CHIP] = out1+yout2+PAD;
      int row3 = (CHIP*OMAP+3)*RRANGE+row+PAD;
      int yout3 = row3*OM;
      yin3[CHIP] = ((row3>>4)*R + (((~row3&15)*ofs)>>4))*IM;
      acci_ym3[CHIP] = in+yin3[CHIP]     -ry;
      acci_yz3[CHIP] = in+yin3[CHIP];
      acci_yp3[CHIP] = in+yin3[CHIP]     +ry;
      acco_base3[CHIP] = out1+yout3+PAD;  acco3[CHIP] = out1+yout3+PAD;
      int row4 = (CHIP*OMAP+4)*RRANGE+row+PAD;
      int yout4 = row4*OM;
      yin4[CHIP] = ((row4>>4)*R + (((~row4&15)*ofs)>>4))*IM;
      acci_ym4[CHIP] = in+yin4[CHIP]     -ry;
      acci_yz4[CHIP] = in+yin4[CHIP];
      acci_yp4[CHIP] = in+yin4[CHIP]     +ry;
      acco_base4[CHIP] = out1+yout4+PAD;  acco4[CHIP] = out1+yout4+PAD;
      int row5 = (CHIP*OMAP+5)*RRANGE+row+PAD;
      int yout5 = row5*OM;
      yin5[CHIP] = ((row5>>4)*R + (((~row5&15)*ofs)>>4))*IM;
      acci_ym5[CHIP] = in+yin5[CHIP]     -ry;
      acci_yz5[CHIP] = in+yin5[CHIP];
      acci_yp5[CHIP] = in+yin5[CHIP]     +ry;
      acco_base5[CHIP] = out1+yout5+PAD;  acco5[CHIP] = out1+yout5+PAD;
    }
//EMAX5A begin gather mapdist=0
    for (CHIP=0; CHIP<NCHIP; CHIP++) {
      for (INIT0=1,LOOP0=CRANGE,x[CHIP]=PAD-1; LOOP0--; INIT0=0) {
	exe(OP_ADD,  &x[CHIP], x[CHIP],  EXP_H3210,         1LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);  /* stage#0 */
	exe(OP_SUB,  &r1,         -1LL,  EXP_H3210,     x[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_AND,  15LL, OP_NOP, 0LL);  /* stage#1 */
	exe(OP_NOP,  &r2,      x[CHIP],  EXP_H3210,         0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL);  /* stage#1 */
	exe(OP_MLUH, &r3,           r1,  EXP_H3210,    (Ull)ofs, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL);  /* stage#2 */
	exe(OP_MLUH, &r4,           r2,  EXP_H3210,        75LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);  /* stage#2 */
	exe(OP_ADD3, &r0,           r3,  EXP_H3210,          r4, EXP_H3210, (Ull)yin0[CHIP], EXP_H3210, OP_OR, 0LL, OP_SLL, 2LL);/* stage#3 */
	exe(OP_ADD,  &r1,           r3,  EXP_H3210,          r4, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_NOP, 0LL);  /* stage#3 */
	mop(OP_LDWR,    1, &BR[4][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym0[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#4 */
	mop(OP_LDWR,    1, &BR[4][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym0[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#4 */
	mop(OP_LDWR,    1, &BR[4][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym0[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#4 */
	exe(OP_MLUH,  &r10,     BR[4][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
	exe(OP_MLUH,  &r11,     BR[4][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
	exe(OP_MLUH,  &r12,     BR[4][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
	exe(OP_MLUH,  &r13,     BR[4][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
	exe(OP_MLUH,  &r14,     BR[4][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
	exe(OP_MLUH,  &r15,     BR[4][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
	exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#6 */
	mop(OP_LDWR,    1, &BR[6][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz0[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#6 */
	mop(OP_LDWR,    1, &BR[6][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz0[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#6 */
	mop(OP_LDWR,    1, &BR[6][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz0[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#6 */
	exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#7 */
	exe(OP_MLUH,  &r10,     BR[6][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
	exe(OP_MLUH,  &r11,     BR[6][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
	exe(OP_MLUH,  &r12,     BR[6][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
	exe(OP_MLUH,  &r13,     BR[6][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
	exe(OP_MLUH,  &r14,     BR[6][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
	exe(OP_MLUH,  &r15,     BR[6][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
	exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#8 */
	mop(OP_LDWR,    1, &BR[8][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp0[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#8 */
	mop(OP_LDWR,    1, &BR[8][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp0[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#8 */
	mop(OP_LDWR,    1, &BR[8][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp0[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#8 */
	exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#9 */
	exe(OP_MLUH,  &r10,     BR[8][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
	exe(OP_MLUH,  &r11,     BR[8][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
	exe(OP_MLUH,  &r12,     BR[8][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
	exe(OP_MLUH,  &r13,     BR[8][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
	exe(OP_MLUH,  &r14,     BR[8][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
	exe(OP_MLUH,  &r15,     BR[8][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
	exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#10 */
	exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#11 */
	exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#12 */
	exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#12 */
	exe(OP_MH2BW, &r29,  r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);  /* stage#13 */
	mop(OP_STWR,    3, &r29, (Ull)(acco0[CHIP]++), 0LL, MSK_D0, (Ull)acco_base0[CHIP], CRANGE, 0, 0, (Ull)NULL, CRANGE);/* stage#13 */
	/**********************/
	exe(OP_ADD,  &r0,           r1,  EXP_H3210,    (Ull)yin1[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#13 */
	/**********************/
	mop(OP_LDWR,    1, &BR[14][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym1[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#14 */
	mop(OP_LDWR,    1, &BR[14][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym1[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#14 */
	mop(OP_LDWR,    1, &BR[14][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym1[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#14 */
	exe(OP_MLUH,  &r10,     BR[14][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#15 */
	exe(OP_MLUH,  &r11,     BR[14][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#15 */
	exe(OP_MLUH,  &r12,     BR[14][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#15 */
	exe(OP_MLUH,  &r13,     BR[14][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#16 */
	exe(OP_MLUH,  &r14,     BR[14][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#16 */
	exe(OP_MLUH,  &r15,     BR[14][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#16 */
	exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#16 */
	mop(OP_LDWR,    1, &BR[16][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz1[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#16 */
	mop(OP_LDWR,    1, &BR[16][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz1[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#16 */
	mop(OP_LDWR,    1, &BR[16][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz1[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#16 */
	exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#17 */
	exe(OP_MLUH,  &r10,     BR[16][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#17 */
	exe(OP_MLUH,  &r11,     BR[16][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#17 */
	exe(OP_MLUH,  &r12,     BR[16][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#17 */
	exe(OP_MLUH,  &r13,     BR[16][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#18 */
	exe(OP_MLUH,  &r14,     BR[16][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#18 */
	exe(OP_MLUH,  &r15,     BR[16][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#18 */
	exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#18 */
	mop(OP_LDWR,    1, &BR[18][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp1[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#18 */
	mop(OP_LDWR,    1, &BR[18][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp1[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#18 */
	mop(OP_LDWR,    1, &BR[18][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp1[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#18 */
	exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#19 */
	exe(OP_MLUH,  &r10,     BR[18][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_MLUH,  &r11,     BR[18][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_MLUH,  &r12,     BR[18][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_MLUH,  &r13,     BR[18][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_MLUH,  &r14,     BR[18][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_MLUH,  &r15,     BR[18][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#20 */
	exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#21 */
	exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#22 */
	exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#22 */
	exe(OP_MH2BW, &r29,  r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);  /* stage#23 */
	mop(OP_STWR,    3, &r29, (Ull)(acco1[CHIP]++), 0LL, MSK_D0, (Ull)acco_base1[CHIP], CRANGE, 0, 0, (Ull)NULL, CRANGE);/* stage#23 */
	/**********************/
	exe(OP_ADD,  &r0,           r1,  EXP_H3210,    (Ull)yin2[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#23 */
	/**********************/
	mop(OP_LDWR,    1, &BR[24][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym2[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#24 */
	mop(OP_LDWR,    1, &BR[24][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym2[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#24 */
	mop(OP_LDWR,    1, &BR[24][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym2[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#24 */
	exe(OP_MLUH,  &r10,     BR[24][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#25 */
	exe(OP_MLUH,  &r11,     BR[24][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#25 */
	exe(OP_MLUH,  &r12,     BR[24][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#25 */
	exe(OP_MLUH,  &r13,     BR[24][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#26 */
	exe(OP_MLUH,  &r14,     BR[24][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#26 */
	exe(OP_MLUH,  &r15,     BR[24][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#26 */
	exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#26 */
	mop(OP_LDWR,    1, &BR[26][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz2[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#26 */
	mop(OP_LDWR,    1, &BR[26][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz2[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#26 */
	mop(OP_LDWR,    1, &BR[26][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz2[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#26 */
	exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#27 */
	exe(OP_MLUH,  &r10,     BR[26][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#27 */
	exe(OP_MLUH,  &r11,     BR[26][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#27 */
	exe(OP_MLUH,  &r12,     BR[26][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#27 */
	exe(OP_MLUH,  &r13,     BR[26][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#28 */
	exe(OP_MLUH,  &r14,     BR[26][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#28 */
	exe(OP_MLUH,  &r15,     BR[26][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#28 */
	exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#28 */
	mop(OP_LDWR,    1, &BR[28][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp2[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#28 */
	mop(OP_LDWR,    1, &BR[28][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp2[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#28 */
	mop(OP_LDWR,    1, &BR[28][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp2[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#28 */
	exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#29 */
	exe(OP_MLUH,  &r10,     BR[28][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#29 */
	exe(OP_MLUH,  &r11,     BR[28][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#29 */
	exe(OP_MLUH,  &r12,     BR[28][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#29 */
	exe(OP_MLUH,  &r13,     BR[28][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#30 */
	exe(OP_MLUH,  &r14,     BR[28][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#30 */
	exe(OP_MLUH,  &r15,     BR[28][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#30 */
	exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#30 */
	exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#31 */
	exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#32 */
	exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#32 */
	exe(OP_MH2BW, &r29,  r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);  /* stage#33 */
	mop(OP_STWR,    3, &r29, (Ull)(acco2[CHIP]++), 0LL, MSK_D0, (Ull)acco_base2[CHIP], CRANGE, 0, 0, (Ull)NULL, CRANGE);/* stage#33 */
	/**********************/
	exe(OP_ADD,  &r0,           r1,  EXP_H3210,    (Ull)yin3[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#33 */
	/**********************/
	mop(OP_LDWR,    1, &BR[34][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym3[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#34 */
	mop(OP_LDWR,    1, &BR[34][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym3[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#34 */
	mop(OP_LDWR,    1, &BR[34][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym3[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#34 */
	exe(OP_MLUH,  &r10,     BR[34][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#35 */
	exe(OP_MLUH,  &r11,     BR[34][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#35 */
	exe(OP_MLUH,  &r12,     BR[34][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#35 */
	exe(OP_MLUH,  &r13,     BR[34][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#36 */
	exe(OP_MLUH,  &r14,     BR[34][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#36 */
	exe(OP_MLUH,  &r15,     BR[34][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#36 */
	exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#36 */
	mop(OP_LDWR,    1, &BR[36][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz3[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#36 */
	mop(OP_LDWR,    1, &BR[36][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz3[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#36 */
	mop(OP_LDWR,    1, &BR[36][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz3[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#36 */
	exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#37 */
	exe(OP_MLUH,  &r10,     BR[36][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#37 */
	exe(OP_MLUH,  &r11,     BR[36][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#37 */
	exe(OP_MLUH,  &r12,     BR[36][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#37 */
	exe(OP_MLUH,  &r13,     BR[36][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#38 */
	exe(OP_MLUH,  &r14,     BR[36][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#38 */
	exe(OP_MLUH,  &r15,     BR[36][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#38 */
	exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#38 */
	mop(OP_LDWR,    1, &BR[38][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp3[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#38 */
	mop(OP_LDWR,    1, &BR[38][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp3[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#38 */
	mop(OP_LDWR,    1, &BR[38][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp3[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#38 */
	exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#39 */
	exe(OP_MLUH,  &r10,     BR[38][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#39 */
	exe(OP_MLUH,  &r11,     BR[38][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#39 */
	exe(OP_MLUH,  &r12,     BR[38][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#39 */
	exe(OP_MLUH,  &r13,     BR[38][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#40 */
	exe(OP_MLUH,  &r14,     BR[38][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#40 */
	exe(OP_MLUH,  &r15,     BR[38][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#40 */
	exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#40 */
	exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#41 */
	exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#42 */
	exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#42 */
	exe(OP_MH2BW, &r29,  r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);  /* stage#43 */
	mop(OP_STWR,    3, &r29, (Ull)(acco3[CHIP]++), 0LL, MSK_D0, (Ull)acco_base3[CHIP], CRANGE, 0, 0, (Ull)NULL, CRANGE);/* stage#43 */
	/**********************/
	exe(OP_ADD,  &r0,           r1,  EXP_H3210,    (Ull)yin4[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#43 */
	/**********************/
	mop(OP_LDWR,    1, &BR[44][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym4[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#44 */
	mop(OP_LDWR,    1, &BR[44][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym4[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#44 */
	mop(OP_LDWR,    1, &BR[44][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym4[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#44 */
	exe(OP_MLUH,  &r10,     BR[44][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#45 */
	exe(OP_MLUH,  &r11,     BR[44][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#45 */
	exe(OP_MLUH,  &r12,     BR[44][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#45 */
	exe(OP_MLUH,  &r13,     BR[44][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#46 */
	exe(OP_MLUH,  &r14,     BR[44][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#46 */
	exe(OP_MLUH,  &r15,     BR[44][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#46 */
	exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#46 */
	mop(OP_LDWR,    1, &BR[46][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz4[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#46 */
	mop(OP_LDWR,    1, &BR[46][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz4[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#46 */
	mop(OP_LDWR,    1, &BR[46][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz4[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#46 */
	exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#47 */
	exe(OP_MLUH,  &r10,     BR[46][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#47 */
	exe(OP_MLUH,  &r11,     BR[46][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#47 */
	exe(OP_MLUH,  &r12,     BR[46][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#47 */
	exe(OP_MLUH,  &r13,     BR[46][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#48 */
	exe(OP_MLUH,  &r14,     BR[46][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#48 */
	exe(OP_MLUH,  &r15,     BR[46][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#48 */
	exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#48 */
	mop(OP_LDWR,    1, &BR[48][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp4[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#48 */
	mop(OP_LDWR,    1, &BR[48][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp4[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#48 */
	mop(OP_LDWR,    1, &BR[48][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp4[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#48 */
	exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#49 */
	exe(OP_MLUH,  &r10,     BR[48][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#49 */
	exe(OP_MLUH,  &r11,     BR[48][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#49 */
	exe(OP_MLUH,  &r12,     BR[48][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#49 */
	exe(OP_MLUH,  &r13,     BR[48][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#50 */
	exe(OP_MLUH,  &r14,     BR[48][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#50 */
	exe(OP_MLUH,  &r15,     BR[48][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#50 */
	exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#50 */
	exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#51 */
	exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#52 */
	exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#52 */
	exe(OP_MH2BW, &r29,  r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);  /* stage#53 */
	mop(OP_STWR,    3, &r29, (Ull)(acco4[CHIP]++), 0LL, MSK_D0, (Ull)acco_base4[CHIP], CRANGE, 0, 0, (Ull)NULL, CRANGE);/* stage#53 */
	/**********************/
	exe(OP_ADD,  &r0,           r1,  EXP_H3210,    (Ull)yin5[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#53 */
	/**********************/
	mop(OP_LDWR,    1, &BR[54][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym5[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#54 */
	mop(OP_LDWR,    1, &BR[54][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym5[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#54 */
	mop(OP_LDWR,    1, &BR[54][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym5[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#54 */
	exe(OP_MLUH,  &r10,     BR[54][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#55 */
	exe(OP_MLUH,  &r11,     BR[54][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#55 */
	exe(OP_MLUH,  &r12,     BR[54][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#55 */
	exe(OP_MLUH,  &r13,     BR[54][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#56 */
	exe(OP_MLUH,  &r14,     BR[54][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#56 */
	exe(OP_MLUH,  &r15,     BR[54][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#56 */
	exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#56 */
	mop(OP_LDWR,    1, &BR[56][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz5[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#56 */
	mop(OP_LDWR,    1, &BR[56][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz5[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#56 */
	mop(OP_LDWR,    1, &BR[56][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz5[CHIP], IM, 0, 0, (Ull)NULL, IM);          /* stage#56 */
	exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#57 */
	exe(OP_MLUH,  &r10,     BR[56][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#57 */
	exe(OP_MLUH,  &r11,     BR[56][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#57 */
	exe(OP_MLUH,  &r12,     BR[56][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#57 */
	exe(OP_MLUH,  &r13,     BR[56][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#58 */
	exe(OP_MLUH,  &r14,     BR[56][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#58 */
	exe(OP_MLUH,  &r15,     BR[56][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#58 */
	exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#58 */
	mop(OP_LDWR,    1, &BR[58][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp5[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#58 */
	mop(OP_LDWR,    1, &BR[58][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp5[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#58 */
	mop(OP_LDWR,    1, &BR[58][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp5[CHIP], IM, 0, 0, (Ull)NULL, IM);         /* stage#58 */
	exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#59 */
	exe(OP_MLUH,  &r10,     BR[58][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#59 */
	exe(OP_MLUH,  &r11,     BR[58][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#59 */
	exe(OP_MLUH,  &r12,     BR[58][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#59 */
	exe(OP_MLUH,  &r13,     BR[58][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#60 */
	exe(OP_MLUH,  &r14,     BR[58][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#60 */
	exe(OP_MLUH,  &r15,     BR[58][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#60 */
	exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#60 */
	exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#61 */
	exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#62 */
	exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#62 */
	exe(OP_MH2BW, &r29,  r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);  /* stage#63 */
	mop(OP_STWR,    3, &r29, (Ull)(acco5[CHIP]++), 0LL, MSK_D0, (Ull)acco_base5[CHIP], CRANGE, 0, 0, (Ull)NULL, CRANGE);/* stage#63 */
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm
}
#endif
