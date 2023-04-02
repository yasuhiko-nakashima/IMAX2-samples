
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm64/sample/mm_cnn_lf/RCS/gdepth.c,v 1.4 2018/02/04 10:28:49 nakashim Exp nakashim $";

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

#define MINOFFSET 8
#define MAXOFFSET 14
#define IM    7500
/* height=5433 */
#define OM    1600
#define OMAP  4
/*#define NCHIP 4*/
#define NCHIP 4
#define RRANGE ((OM-PAD*2)/NCHIP/OMAP)
#define CRANGE ((OM-PAD*2))
/* height=1200 */
#define R     75
#define PAD   32
#define TH    137
Uchar *rgb; /*[IM*IM*3];*/
Uint *in;   /*[IM*IM];*/
Uint *sad0; /*[OM*OM];*/
Uint *sad1; /*[OM*OM];*/
Uint *out0; /*[OM*OM];*/
Uint *out1; /*[OM*OM];*/
Uint *ip, *wp, *op;
int ofs = 14;
int c, s, i, j, p0, p1;
int row, col, oc, rx, ry, y, x;
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
  FILE *fp;
  int fd;
  char dummy[16];

  sysinit(IM*IM*3
	 +IM*IM*sizeof(int)
	 +OM*OM*sizeof(int)
	 +OM*OM*sizeof(int)
	 +OM*OM*sizeof(int)
         +OM*OM*sizeof(int),32);
  printf("membase: %08.8x\n", (Uint)membase);
  rgb  = (Uchar*)membase;
  in   = (Uint*)((Uchar*)rgb  + IM*IM*3);
  out0 = (Uint*)((Uchar*)in   + IM*IM*sizeof(int));
  out1 = (Uint*)((Uchar*)out0 + OM*OM*sizeof(int));
  sad0 = (Uint*)((Uchar*)out1 + OM*OM*sizeof(int));
  sad1 = (Uint*)((Uchar*)sad0 + OM*OM*sizeof(int));
  printf("rgb : %08.8x\n", rgb);
  printf("in  : %08.8x\n", in);
  printf("out0: %08.8x\n", out0);
  printf("out1: %08.8x\n", out1);
  printf("sad0: %08.8x\n", sad0);
  printf("sad1: %08.8x\n", sad1);

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
  for (i=0; i<OM*OM; i++) {
    out0[i] = MINOFFSET;
    out1[i] = MINOFFSET;
    sad0[i] = MAXINT;
    sad1[i] = MAXINT;
  }
#else
  if ((fd = open("472.ini", O_RDONLY)) < 0) {
    printf("can't open 472.ini\n");
    exit(1);
  }
  read(fd, in, IM*IM*4);
  printf("reading init_file 1stWORD=%08.8x\n", in[0]);
  read(fd, out0, OM*OM*sizeof(int));
  read(fd, out1, OM*OM*sizeof(int));
  read(fd, sad0, OM*OM*sizeof(int));
  read(fd, sad1, OM*OM*sizeof(int));
  close(fd);
#endif

#if !defined(ARMSIML)
  x11_open(0);
#endif

#if 1
  reset_nanosec();
  for (ofs=MINOFFSET+1; ofs<=MAXOFFSET-1; ofs++)
    orig();
  get_nanosec(0);
  show_nanosec();
#endif

  reset_nanosec();
  for (ofs=MINOFFSET+1; ofs<=MAXOFFSET-1; ofs++)
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
      if (sad0[row*OM+col] != sad1[row*OM+col]) {
	count2++;
	printf("s0[%d]=%x s1[%d]=%x\n",
	       row*OM+col, sad0[row*OM+col],
	       row*OM+col, sad1[row*OM+col]);
      }
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
      Uint val = (Uchar)(((Uchar)(*from++)-MINOFFSET)*256/(MAXOFFSET-MINOFFSET));
      *to++ = (val<<24)|(val<<16)|(val<<8);
    }
  }
}

orig()
{
  printf("<<<ORIG>>>\n");
  ry = (R+ofs)*IM;
  rx = (R+ofs); /* ofs: from 8 to 14 */
  for (row=PAD; row<OM-PAD; row++) {
    for (col=PAD; col<OM-PAD; col++) {
      c =((row>>4)*R + (((~row&15)*ofs)>>4))*IM
	+ (col>>4)*R + (((~col&15)*ofs)>>4);
      s = 0;
      for (y=-1; y<=1; y++) {
	for (x=-1; x<=1; x++) {
	  if (x == 0) continue;
	  for (i=-1; i<=1; i++) {
	    for (j=-1; j<=1; j++) {
	      if (j == 0) continue;
	      p0 = in[c     +(i*IM)     +j]; /* center */
	      p1 = in[c+ry*y+(i*IM)+rx*x+j]; /* comparand */
	      s += dif(p0, p1);
	      if (s > 0xffff) s = 0xffff;
	      /*printf("row=%d col=%d y=%d x=%d i=%d j=%d s=%08.8x\n", row, col, y, x, i, j, s);*/
	      count0++;
	    }
	  }
	}
      }
      if (sad0[row*OM+col]>TH && s<sad0[row*OM+col]) {
	sad0[row*OM+col] = s;
	out0[row*OM+col] = ofs;
      }
    }
  }
}

#if 0
imax()
{
  Ull CHIP;
  printf("<<<IMAX>>>\n");
  ry = (R+ofs)*IM;
  rx = (R+ofs); /* ofs: from 8 to 14 */
  for (row=0; row<RRANGE; row++) { /* 0..381 */
    for (CHIP=0; CHIP<NCHIP; CHIP++) {
      for (col=0; col<CRANGE; col++) {
	for (oc=0; oc<OMAP; oc++) {
	  c =((((CHIP*OMAP+oc)*RRANGE+row+PAD)>>4)*R + (((~((CHIP*OMAP+oc)*RRANGE+row+PAD)&15)*ofs)>>4))*IM
	    + ((                      col+PAD)>>4)*R + (((~(                      col+PAD)&15)*ofs)>>4);
	  s = 0;
	  for (y=-1; y<=1; y++) {
	    for (x=-1; x<=1; x++) {
	      if (x == 0) continue;
	      for (i=-1; i<=1; i++) {
		for (j=-1; j<=1; j++) {
		  if (j == 0) continue;
		  p0 = in[c     +(i*IM)     +j]; /* center */
		  p1 = in[c+ry*y+(i*IM)+rx*x+j]; /* comparand */
		  s += dif(p0, p1);
		  /*printf("row=%d col=%d y=%d x=%d i=%d j=%d\n", row, col, y, x, i, j);*/
		  count1++;
		}
	      }
	    }
	  }
	  if (sad1[((CHIP*OMAP+oc)*RRANGE+row+PAD)*OM+(col)+PAD]>TH && s<sad1[((CHIP*OMAP+oc)*RRANGE+row+PAD)*OM+(col+PAD)]) {
	    sad1[((CHIP*OMAP+oc)*RRANGE+row+PAD)*OM+(col+PAD)] = s;
	    out1[((CHIP*OMAP+oc)*RRANGE+row+PAD)*OM+(col+PAD)] = ofs;
	  }
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
  Uint *yzm_xm_m4 = in-IM   -rx-1;  Uint *yzm_xm_p4 = in-IM   -rx+1;
  Uint *yzm_xz_m4 = in-IM      -1;  Uint *yzm_xz_p4 = in-IM      +1;
  Uint *yzm_xp_m4 = in-IM   +rx-1;  Uint *yzm_xp_p4 = in-IM   +rx+1;
  Uint *ymm_xm_m4 = in-IM-ry-rx-1;  Uint *ymm_xm_p4 = in-IM-ry-rx+1;
  Uint *ymm_xp_m4 = in-IM-ry+rx-1;  Uint *ymm_xp_p4 = in-IM-ry+rx+1;
  Uint *ypm_xm_m4 = in-IM+ry-rx-1;  Uint *ypm_xm_p4 = in-IM+ry-rx+1;
  Uint *ypm_xp_m4 = in-IM+ry+rx-1;  Uint *ypm_xp_p4 = in-IM+ry+rx+1;
  Uint *yzz_xm_m4 = in      -rx-1;  Uint *yzz_xm_p4 = in      -rx+1;
  Uint *yzz_xz_m4 = in         -1;  Uint *yzz_xz_p4 = in         +1;
  Uint *yzz_xp_m4 = in      +rx-1;  Uint *yzz_xp_p4 = in      +rx+1;
  Uint *ymz_xm_m4 = in   -ry-rx-1;  Uint *ymz_xm_p4 = in   -ry-rx+1;
  Uint *ymz_xp_m4 = in   -ry+rx-1;  Uint *ymz_xp_p4 = in   -ry+rx+1;
  Uint *ypz_xm_m4 = in   +ry-rx-1;  Uint *ypz_xm_p4 = in   +ry-rx+1;
  Uint *ypz_xp_m4 = in   +ry+rx-1;  Uint *ypz_xp_p4 = in   +ry+rx+1;
  Uint *yzp_xm_m4 = in+IM   -rx-1;  Uint *yzp_xm_p4 = in+IM   -rx+1;
  Uint *yzp_xz_m4 = in+IM      -1;  Uint *yzp_xz_p4 = in+IM      +1;
  Uint *yzp_xp_m4 = in+IM   +rx-1;  Uint *yzp_xp_p4 = in+IM   +rx+1;
  Uint *ymp_xm_m4 = in+IM-ry-rx-1;  Uint *ymp_xm_p4 = in+IM-ry-rx+1;
  Uint *ymp_xp_m4 = in+IM-ry+rx-1;  Uint *ymp_xp_p4 = in+IM-ry+rx+1;
  Uint *ypp_xm_m4 = in+IM+ry-rx-1;  Uint *ypp_xm_p4 = in+IM+ry-rx+1;
  Uint *ypp_xp_m4 = in+IM+ry+rx-1;  Uint *ypp_xp_p4 = in+IM+ry+rx+1;

  for (row=RRANGE-1; row>=0; row--) {
    int  yin0[NCHIP];
    Uint *acci_yzm0[NCHIP];  Uint *acci_ymm0[NCHIP];  Uint *acci_ypm0[NCHIP];
    Uint *acci_yzz0[NCHIP];  Uint *acci_ymz0[NCHIP];  Uint *acci_ypz0[NCHIP];
    Uint *acci_yzp0[NCHIP];  Uint *acci_ymp0[NCHIP];  Uint *acci_ypp0[NCHIP];
    Uint *sadx_base0[NCHIP]; Uint *sadi0[NCHIP]; Uint *sado0[NCHIP];
    Uint *acco_base0[NCHIP]; Uint *acco0[NCHIP];
    int  yin1[NCHIP];
    Uint *acci_yzm1[NCHIP];  Uint *acci_ymm1[NCHIP];  Uint *acci_ypm1[NCHIP];
    Uint *acci_yzz1[NCHIP];  Uint *acci_ymz1[NCHIP];  Uint *acci_ypz1[NCHIP];
    Uint *acci_yzp1[NCHIP];  Uint *acci_ymp1[NCHIP];  Uint *acci_ypp1[NCHIP];
    Uint *sadx_base1[NCHIP]; Uint *sadi1[NCHIP]; Uint *sado1[NCHIP];
    Uint *acco_base1[NCHIP]; Uint *acco1[NCHIP];
    int  yin2[NCHIP];
    Uint *acci_yzm2[NCHIP];  Uint *acci_ymm2[NCHIP];  Uint *acci_ypm2[NCHIP];
    Uint *acci_yzz2[NCHIP];  Uint *acci_ymz2[NCHIP];  Uint *acci_ypz2[NCHIP];
    Uint *acci_yzp2[NCHIP];  Uint *acci_ymp2[NCHIP];  Uint *acci_ypp2[NCHIP];
    Uint *sadx_base2[NCHIP]; Uint *sadi2[NCHIP]; Uint *sado2[NCHIP];
    Uint *acco_base2[NCHIP]; Uint *acco2[NCHIP];
    int  yin3[NCHIP];
    Uint *acci_yzm3[NCHIP];  Uint *acci_ymm3[NCHIP];  Uint *acci_ypm3[NCHIP];
    Uint *acci_yzz3[NCHIP];  Uint *acci_ymz3[NCHIP];  Uint *acci_ypz3[NCHIP];
    Uint *acci_yzp3[NCHIP];  Uint *acci_ymp3[NCHIP];  Uint *acci_ypp3[NCHIP];
    Uint *sadx_base3[NCHIP]; Uint *sadi3[NCHIP]; Uint *sado3[NCHIP];
    Uint *acco_base3[NCHIP]; Uint *acco3[NCHIP];

    for (CHIP=0; CHIP<NCHIP; CHIP++) {
      int  row0  = (CHIP*OMAP+0)*RRANGE+row+PAD;
      int  yout0 = row0*OM;
      yin0[CHIP] = ((row0>>4)*R + (((~row0&15)*ofs)>>4))*IM;
      acci_yzm0[CHIP] = in+yin0[CHIP]-IM;  acci_ymm0[CHIP] = in+yin0[CHIP]-IM-ry;  acci_ypm0[CHIP] = in+yin0[CHIP]-IM+ry;
      acci_yzz0[CHIP] = in+yin0[CHIP];     acci_ymz0[CHIP] = in+yin0[CHIP]   -ry;  acci_ypz0[CHIP] = in+yin0[CHIP]   +ry;
      acci_yzp0[CHIP] = in+yin0[CHIP]+IM;  acci_ymp0[CHIP] = in+yin0[CHIP]+IM-ry;  acci_ypp0[CHIP] = in+yin0[CHIP]+IM+ry;
      sadx_base0[CHIP] = sad1+yout0+PAD;  sadi0[CHIP] = sad1+yout0+PAD;  sado0[CHIP] = sad1+yout0+PAD;
      acco_base0[CHIP] = out1+yout0+PAD;  acco0[CHIP] = out1+yout0+PAD;
      int  row1  = (CHIP*OMAP+1)*RRANGE+row+PAD;
      int  yout1 = row1*OM;
      yin1[CHIP] = ((row1>>4)*R + (((~row1&15)*ofs)>>4))*IM;
      acci_yzm1[CHIP] = in+yin1[CHIP]-IM;  acci_ymm1[CHIP] = in+yin1[CHIP]-IM-ry;  acci_ypm1[CHIP] = in+yin1[CHIP]-IM+ry;
      acci_yzz1[CHIP] = in+yin1[CHIP];     acci_ymz1[CHIP] = in+yin1[CHIP]   -ry;  acci_ypz1[CHIP] = in+yin1[CHIP]   +ry;
      acci_yzp1[CHIP] = in+yin1[CHIP]+IM;  acci_ymp1[CHIP] = in+yin1[CHIP]+IM-ry;  acci_ypp1[CHIP] = in+yin1[CHIP]+IM+ry;
      sadx_base1[CHIP] = sad1+yout1+PAD;  sadi1[CHIP] = sad1+yout1+PAD;  sado1[CHIP] = sad1+yout1+PAD;
      acco_base1[CHIP] = out1+yout1+PAD;  acco1[CHIP] = out1+yout1+PAD;
      int  row2  = (CHIP*OMAP+2)*RRANGE+row+PAD;
      int  yout2 = row2*OM;
      yin2[CHIP] = ((row2>>4)*R + (((~row2&15)*ofs)>>4))*IM;
      acci_yzm2[CHIP] = in+yin2[CHIP]-IM;  acci_ymm2[CHIP] = in+yin2[CHIP]-IM-ry;  acci_ypm2[CHIP] = in+yin2[CHIP]-IM+ry;
      acci_yzz2[CHIP] = in+yin2[CHIP];     acci_ymz2[CHIP] = in+yin2[CHIP]   -ry;  acci_ypz2[CHIP] = in+yin2[CHIP]   +ry;
      acci_yzp2[CHIP] = in+yin2[CHIP]+IM;  acci_ymp2[CHIP] = in+yin2[CHIP]+IM-ry;  acci_ypp2[CHIP] = in+yin2[CHIP]+IM+ry;
      sadx_base2[CHIP] = sad1+yout2+PAD;  sadi2[CHIP] = sad1+yout2+PAD;  sado2[CHIP] = sad1+yout2+PAD;
      acco_base2[CHIP] = out1+yout2+PAD;  acco2[CHIP] = out1+yout2+PAD;
      int  row3  = (CHIP*OMAP+3)*RRANGE+row+PAD;
      int  yout3 = row3*OM;
      yin3[CHIP] = ((row3>>4)*R + (((~row3&15)*ofs)>>4))*IM;
      acci_yzm3[CHIP] = in+yin3[CHIP]-IM;  acci_ymm3[CHIP] = in+yin3[CHIP]-IM-ry;  acci_ypm3[CHIP] = in+yin3[CHIP]-IM+ry;
      acci_yzz3[CHIP] = in+yin3[CHIP];     acci_ymz3[CHIP] = in+yin3[CHIP]   -ry;  acci_ypz3[CHIP] = in+yin3[CHIP]   +ry;
      acci_yzp3[CHIP] = in+yin3[CHIP]+IM;  acci_ymp3[CHIP] = in+yin3[CHIP]+IM-ry;  acci_ypp3[CHIP] = in+yin3[CHIP]+IM+ry;
      sadx_base3[CHIP] = sad1+yout3+PAD;  sadi3[CHIP] = sad1+yout3+PAD;  sado3[CHIP] = sad1+yout3+PAD;
      acco_base3[CHIP] = out1+yout3+PAD;  acco3[CHIP] = out1+yout3+PAD;
    }
//EMAX5A begin gdepth mapdist=3
    for (CHIP=0; CHIP<NCHIP; CHIP++) {
      for (INIT0=1,LOOP0=CRANGE,x[CHIP]=PAD-1; LOOP0--; INIT0=0) {
	exe(OP_ADD,  &x[CHIP], x[CHIP],  EXP_H3210,         1LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);  /* stage#0 */
	exe(OP_SUB,  &r1,         -1LL,  EXP_H3210,     x[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_AND,  15LL, OP_NOP, 0LL);  /* stage#1 */
	exe(OP_NOP,  &r2,      x[CHIP],  EXP_H3210,         0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL);  /* stage#1 */
	exe(OP_MLUH, &r3,           r1,  EXP_H3210,    (Ull)ofs, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL);  /* stage#2 */
	exe(OP_MLUH, &r4,           r2,  EXP_H3210,        75LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);  /* stage#2 */
	exe(OP_ADD3, &r0,           r3,  EXP_H3210,          r4, EXP_H3210, (Ull)yin0[CHIP], EXP_H3210, OP_OR, 0LL, OP_SLL, 2LL);/* stage#3 */
	exe(OP_ADD,  &r1,           r3,  EXP_H3210,          r4, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_NOP, 0LL);  /* stage#3 */
	mop(OP_LDWR,   1, &BR[4][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#4 */
	exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[4][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
	exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[4][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
	exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[4][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
	exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[4][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
	mop(OP_LDWR,   1, &BR[5][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#5 */
	mop(OP_LDWR,   1, &BR[5][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#5 */
	mop(OP_LDWR,   1, &BR[5][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#5 */
	mop(OP_LDWR,   1, &BR[5][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#5 */
	exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[5][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
	exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[5][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
	exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[5][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
	exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[5][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
	mop(OP_LDWR,   1, &BR[6][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#6 */
	mop(OP_LDWR,   1, &BR[6][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#6 */
	mop(OP_LDWR,   1, &BR[6][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#6 */
	mop(OP_LDWR,   1, &BR[6][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#6 */
	exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[6][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
	exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[6][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
	exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[6][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
	exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[6][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#7 */
	exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[7][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
	exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[7][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
	exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[7][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
	exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[7][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
	mop(OP_LDWR,   1, &BR[8][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#8 */
	mop(OP_LDWR,   1, &BR[8][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#8 */
	mop(OP_LDWR,   1, &BR[8][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#8 */
	mop(OP_LDWR,   1, &BR[8][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#8 */
	exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[8][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
	exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[8][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
	exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[8][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
	exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[8][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
	mop(OP_LDWR,   1, &BR[9][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#9 */
	mop(OP_LDWR,   1, &BR[9][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#9 */
	mop(OP_LDWR,   1, &BR[9][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#9 */
	mop(OP_LDWR,   1, &BR[9][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz0[CHIP], IM, 0, 0, (Ull)NULL, IM);       /* stage#9 */
	exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[9][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
	exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[9][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
	exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[9][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
	exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[9][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#10 */
	exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[10][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[10][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[10][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[10][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	mop(OP_LDWR,   1, &BR[11][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#11 */
	mop(OP_LDWR,   1, &BR[11][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#11 */
	mop(OP_LDWR,   1, &BR[11][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#11 */
	mop(OP_LDWR,   1, &BR[11][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#11 */
	exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[11][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[11][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[11][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[11][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	mop(OP_LDWR,   1, &BR[12][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#12 */
	mop(OP_LDWR,   1, &BR[12][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#12 */
	mop(OP_LDWR,   1, &BR[12][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#12 */
	mop(OP_LDWR,   1, &BR[12][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp0[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#12 */
	exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[12][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[12][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[12][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[12][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#14 */
	exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#14 */
	exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#15 */
	mop(OP_LDWR,   1, &BR[15][1][1], (Ull)(sadi0[CHIP]++), 0LL, MSK_D0, (Ull)sadx_base0[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);    /* stage#15 */
	exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[15][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#16 */
	exe(OP_CMP_GT, &c1, BR[15][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#16 */
	exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    (Ull)ofs, OP_NOP, 0LL); /* stage#16 */
	cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                                            /* stage#17 */
	mop(OP_STWR, ex1, &r31, (Ull)(acco0[CHIP]++), 0LL, MSK_D0, (Ull)acco_base0[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);      /* stage#17 */
	cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                                            /* stage#17 */
	mop(OP_STWR, ex0, &r30, (Ull)(sado0[CHIP]++), 0LL, MSK_D0, (Ull)sadx_base0[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);      /* stage#17 */
	/********************/
	exe(OP_ADD,  &r0,           r1,  EXP_H3210,    (Ull)yin1[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL);  /* stage#17 */
	/********************/
	mop(OP_LDWR,   1, &BR[18][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#18 */
	exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[18][0][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[18][0][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[18][2][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[18][2][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	mop(OP_LDWR,   1, &BR[19][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#19 */
	mop(OP_LDWR,   1, &BR[19][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#19 */
	mop(OP_LDWR,   1, &BR[19][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#19 */
	mop(OP_LDWR,   1, &BR[19][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#19 */
	exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[19][0][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[19][0][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[19][2][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[19][2][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	mop(OP_LDWR,   1, &BR[20][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#20 */
	mop(OP_LDWR,   1, &BR[20][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#20 */
	mop(OP_LDWR,   1, &BR[20][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#20 */
	mop(OP_LDWR,   1, &BR[20][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#20 */
	exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[20][0][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[20][0][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[20][2][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[20][2][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#21 */
	exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[21][0][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[21][0][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[21][2][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[21][2][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	mop(OP_LDWR,   1, &BR[22][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#22 */
	mop(OP_LDWR,   1, &BR[22][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#22 */
	mop(OP_LDWR,   1, &BR[22][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#22 */
	mop(OP_LDWR,   1, &BR[22][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#22 */
	exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[22][0][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[22][0][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[22][2][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[22][2][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	mop(OP_LDWR,   1, &BR[23][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#23 */
	mop(OP_LDWR,   1, &BR[23][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#23 */
	mop(OP_LDWR,   1, &BR[23][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#23 */
	mop(OP_LDWR,   1, &BR[23][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#23 */
	exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[23][0][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[23][0][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[23][2][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[23][2][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	mop(OP_LDWR,   1, &BR[24][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#24 */
	mop(OP_LDWR,   1, &BR[24][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#24 */
        mop(OP_LDWR,   1, &BR[24][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#24 */
        mop(OP_LDWR,   1, &BR[24][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#24 */
        mop(OP_LDWR,   1, &BR[24][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#24 */
        mop(OP_LDWR,   1, &BR[24][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#24 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[24][0][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[24][0][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[24][2][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[24][2][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
        mop(OP_LDWR,   1, &BR[25][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#25 */
        mop(OP_LDWR,   1, &BR[25][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#25 */
        mop(OP_LDWR,   1, &BR[25][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#25 */
        mop(OP_LDWR,   1, &BR[25][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#25 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[25][0][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[25][0][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[25][2][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[25][2][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
        mop(OP_LDWR,   1, &BR[26][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#26 */
        mop(OP_LDWR,   1, &BR[26][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#26 */
        mop(OP_LDWR,   1, &BR[26][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#26 */
        mop(OP_LDWR,   1, &BR[26][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp1[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#26 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[26][0][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[26][0][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[26][2][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[26][2][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
        exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#28 */
        exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#28 */
        exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#29 */
        mop(OP_LDWR,   1, &BR[29][1][1], (Ull)(sadi1[CHIP]++), 0LL, MSK_D0, (Ull)sadx_base1[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);    /* stage#29 */
        exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[29][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#30 */
        exe(OP_CMP_GT, &c1, BR[29][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#30 */
        exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    (Ull)ofs, OP_NOP, 0LL); /* stage#30 */
        cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                                            /* stage#31 */
        mop(OP_STWR, ex1, &r31, (Ull)(acco1[CHIP]++), 0LL, MSK_D0, (Ull)acco_base1[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);      /* stage#31 */
        cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                                            /* stage#31 */
        mop(OP_STWR, ex0, &r30, (Ull)(sado1[CHIP]++), 0LL, MSK_D0, (Ull)sadx_base1[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);      /* stage#31 */
        /********************/
        exe(OP_ADD,  &r0,           r1,  EXP_H3210,    (Ull)yin2[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL);  /* stage#31 */
        /********************/
        mop(OP_LDWR,   1, &BR[32][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#32 */
        mop(OP_LDWR,   1, &BR[32][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#32 */
        mop(OP_LDWR,   1, &BR[32][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#32 */
        mop(OP_LDWR,   1, &BR[32][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#32 */
        mop(OP_LDWR,   1, &BR[32][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#32 */
        mop(OP_LDWR,   1, &BR[32][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#32 */
        exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[32][0][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
        exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[32][0][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
        exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[32][2][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
        exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[32][2][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
        mop(OP_LDWR,   1, &BR[33][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#33 */
        mop(OP_LDWR,   1, &BR[33][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#33 */
        mop(OP_LDWR,   1, &BR[33][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#33 */
        mop(OP_LDWR,   1, &BR[33][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#33 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[33][0][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[33][0][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[33][2][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[33][2][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
        mop(OP_LDWR,   1, &BR[34][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#34 */
        mop(OP_LDWR,   1, &BR[34][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#34 */
        mop(OP_LDWR,   1, &BR[34][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#34 */
        mop(OP_LDWR,   1, &BR[34][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#34 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[34][0][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[34][0][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[34][2][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[34][2][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
        mop(OP_LDWR,   1, &BR[35][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#35 */
        mop(OP_LDWR,   1, &BR[35][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#35 */
        mop(OP_LDWR,   1, &BR[35][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#35 */
        mop(OP_LDWR,   1, &BR[35][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#35 */
        mop(OP_LDWR,   1, &BR[35][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#35 */
        mop(OP_LDWR,   1, &BR[35][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#35 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[35][0][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[35][0][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[35][2][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[35][2][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
        mop(OP_LDWR,   1, &BR[36][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#36 */
        mop(OP_LDWR,   1, &BR[36][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#36 */
        mop(OP_LDWR,   1, &BR[36][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#36 */
        mop(OP_LDWR,   1, &BR[36][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#36 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[36][0][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[36][0][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[36][2][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[36][2][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
        mop(OP_LDWR,   1, &BR[37][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#37 */
        mop(OP_LDWR,   1, &BR[37][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#37 */
        mop(OP_LDWR,   1, &BR[37][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#37 */
        mop(OP_LDWR,   1, &BR[37][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#37 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[37][0][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[37][0][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[37][2][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[37][2][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
        mop(OP_LDWR,   1, &BR[38][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#38 */
        mop(OP_LDWR,   1, &BR[38][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#38 */
        mop(OP_LDWR,   1, &BR[38][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#38 */
        mop(OP_LDWR,   1, &BR[38][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#38 */
        mop(OP_LDWR,   1, &BR[38][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#38 */
        mop(OP_LDWR,   1, &BR[38][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#38 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[38][0][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[38][0][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[38][2][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[38][2][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
        mop(OP_LDWR,   1, &BR[39][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#39 */
        mop(OP_LDWR,   1, &BR[39][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#39 */
        mop(OP_LDWR,   1, &BR[39][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#39 */
        mop(OP_LDWR,   1, &BR[39][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#39 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[39][0][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[39][0][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[39][2][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[39][2][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
        mop(OP_LDWR,   1, &BR[40][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#40 */
        mop(OP_LDWR,   1, &BR[40][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#40 */
        mop(OP_LDWR,   1, &BR[40][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#40 */
        mop(OP_LDWR,   1, &BR[40][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp2[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#40 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[40][0][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[40][0][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[40][2][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[40][2][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
        exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#42 */
        exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#42 */
        exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#43 */
        mop(OP_LDWR,   1, &BR[43][1][1], (Ull)(sadi2[CHIP]++), 0LL, MSK_D0, (Ull)sadx_base2[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);    /* stage#43 */
        exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[43][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#44 */
        exe(OP_CMP_GT, &c1, BR[43][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#44 */
        exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    (Ull)ofs, OP_NOP, 0LL); /* stage#44 */
        cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                                            /* stage#45 */
        mop(OP_STWR, ex1, &r31, (Ull)(acco2[CHIP]++), 0LL, MSK_D0, (Ull)acco_base2[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);      /* stage#45 */
        cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                                            /* stage#45 */
        mop(OP_STWR, ex0, &r30, (Ull)(sado2[CHIP]++), 0LL, MSK_D0, (Ull)sadx_base2[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);      /* stage#45 */
        /********************/
        exe(OP_ADD,  &r0,           r1,  EXP_H3210,    (Ull)yin3[CHIP], EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL);  /* stage#45 */
        /********************/
        mop(OP_LDWR,   1, &BR[46][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#46 */
        mop(OP_LDWR,   1, &BR[46][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#46 */
        mop(OP_LDWR,   1, &BR[46][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#46 */
        mop(OP_LDWR,   1, &BR[46][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#46 */
        mop(OP_LDWR,   1, &BR[46][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#46 */
        mop(OP_LDWR,   1, &BR[46][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#46 */
        exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[46][0][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
        exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[46][0][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
        exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[46][2][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
        exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[46][2][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
        mop(OP_LDWR,   1, &BR[47][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#47 */
        mop(OP_LDWR,   1, &BR[47][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#47 */
        mop(OP_LDWR,   1, &BR[47][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#47 */
        mop(OP_LDWR,   1, &BR[47][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#47 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[47][0][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[47][0][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[47][2][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[47][2][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
        mop(OP_LDWR,   1, &BR[48][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#48 */
        mop(OP_LDWR,   1, &BR[48][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#48 */
        mop(OP_LDWR,   1, &BR[48][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#48 */
        mop(OP_LDWR,   1, &BR[48][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#48 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[48][0][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[48][0][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[48][2][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[48][2][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
        mop(OP_LDWR,   1, &BR[49][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#49 */
        mop(OP_LDWR,   1, &BR[49][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#49 */
        mop(OP_LDWR,   1, &BR[49][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#49 */
        mop(OP_LDWR,   1, &BR[49][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#49 */
        mop(OP_LDWR,   1, &BR[49][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#49 */
        mop(OP_LDWR,   1, &BR[49][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#49 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[49][0][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[49][0][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[49][2][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[49][2][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
        mop(OP_LDWR,   1, &BR[50][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#50 */
        mop(OP_LDWR,   1, &BR[50][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#50 */
        mop(OP_LDWR,   1, &BR[50][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#50 */
        mop(OP_LDWR,   1, &BR[50][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#50 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[50][0][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[50][0][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[50][2][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[50][2][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
        mop(OP_LDWR,   1, &BR[51][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#51 */
        mop(OP_LDWR,   1, &BR[51][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#51 */
        mop(OP_LDWR,   1, &BR[51][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#51 */
        mop(OP_LDWR,   1, &BR[51][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#51 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[51][0][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[51][0][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[51][2][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[51][2][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
        mop(OP_LDWR,   1, &BR[52][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#52 */
        mop(OP_LDWR,   1, &BR[52][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#52 */
        mop(OP_LDWR,   1, &BR[52][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#52 */
        mop(OP_LDWR,   1, &BR[52][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#52 */
        mop(OP_LDWR,   1, &BR[52][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#52 */
        mop(OP_LDWR,   1, &BR[52][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#52 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[52][0][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[52][0][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[52][2][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[52][2][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
        mop(OP_LDWR,   1, &BR[53][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#53 */
        mop(OP_LDWR,   1, &BR[53][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#53 */
        mop(OP_LDWR,   1, &BR[53][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#53 */
        mop(OP_LDWR,   1, &BR[53][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#53 */
        exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[53][0][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
        exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[53][0][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
        exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[53][2][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
        exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[53][2][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
        mop(OP_LDWR,   1, &BR[54][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#54 */
        mop(OP_LDWR,   1, &BR[54][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#54 */
        mop(OP_LDWR,   1, &BR[54][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#54 */
        mop(OP_LDWR,   1, &BR[54][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp3[CHIP], IM, 0, 0, (Ull)NULL, IM);      /* stage#54 */
        exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[54][0][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
        exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[54][0][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
        exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[54][2][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
        exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[54][2][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
        exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#56 */
        exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#56 */
        exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#57 */
        mop(OP_LDWR,   1, &BR[57][1][1], (Ull)(sadi3[CHIP]++), 0LL, MSK_D0, (Ull)sadx_base3[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);    /* stage#57 */
        exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[57][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#58 */
        exe(OP_CMP_GT, &c1, BR[57][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#58 */
        exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    (Ull)ofs, OP_NOP, 0LL); /* stage#58 */
        cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                                            /* stage#59 */
        mop(OP_STWR, ex1, &r31, (Ull)(acco3[CHIP]++), 0LL, MSK_D0, (Ull)acco_base3[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);      /* stage#59 */
        cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                                            /* stage#59 */
        mop(OP_STWR, ex0, &r30, (Ull)(sado3[CHIP]++), 0LL, MSK_D0, (Ull)sadx_base3[CHIP], CRANGE, 0, 1, (Ull)NULL, CRANGE);      /* stage#59 */
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm
}
#endif
