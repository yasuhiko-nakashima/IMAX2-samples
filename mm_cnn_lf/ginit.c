
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
#include <fcntl.h>
#include <math.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#endif

#if defined(EMAX5)
#include "../../src/conv-c2b/emax5.h"
#include "../../src/conv-c2b/emax5lib.c"
#endif
#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif

#define WD           320
#define HT           240
#define BITMAP       (WD*HT)
Uint    Z[BITMAP];
Uchar   X[BITMAP*3*25];

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
#define IM  7500
/* height=5433 */
#define OM  1600
/* height=1200 */
#define R   75
#define PAD 36
#define TH  137
Uchar *rgb; /*[IM*IM*3];*/
Uint *in;   /*[IM*IM];*/
Uint *sad0; /*[OM*OM];*/
Uint *sad1; /*[OM*OM];*/
Uint *out0; /*[OM*OM];*/
Uint *out1; /*[OM*OM];*/
Uint *ip, *wp, *op;
int ofs = 14;
int c, s, i, j, p0, p1;
int row, col, rx, ry, y, x;
int count0, count1, count2;

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))

main()
{
  FILE *fp;
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

  if ((fp = fopen("472.ini", "w")) == NULL) {
    printf("can't open 472.ini\n");
    exit(1);
  }
  fwrite(in, IM*IM*4, 1, fp);
  fwrite(out0, OM*OM*sizeof(int), 1, fp);
  fwrite(out1, OM*OM*sizeof(int), 1, fp);
  fwrite(sad0, OM*OM*sizeof(int), 1, fp);
  fwrite(sad1, OM*OM*sizeof(int), 1, fp);
  fclose(fp);
}

