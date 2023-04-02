
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/* test013 light-field-camera and display image */
/*                 Copyright (C) 2013- by NAIST */
/*                  Primary writer: Y.Nakashima */
/*                         nakashim@is.naist.jp */

#undef PRECISE_SCALE

#ifndef ARMSIML
#define _POSIX_THREADS
#endif

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
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/Xdbe.h>
#endif

#if defined(EMAX5)
#include "../../src/conv-c2b/emax5.h"
#include "../../src/conv-c2b/emax5lib.c"
#endif

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif

void *test_kernel();

#define MAXTHNUM 2048
#ifdef PTHREAD
#define THNUM 8
#else
#define THNUM 1
#endif

struct param_kernel {
  int th;
  int v; /* valid */
  int from;
  int to;
} param_kernel[MAXTHNUM];

/****************/
/*** IN/OUT   ***/
/****************/
Uint    image_size;
Uint    *ACCI; /* accelerator input */
Uchar   *ACCO; /* accelerator output */

#define WD           320
#define HT           240
#define BITMAP       (WD*HT)
Uint    W[BITMAP];

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

/****************/
/*** MAIN     ***/
/****************/
main(argc, argv) int argc; char **argv;
{
  int i, j;

  image_size = WD*HT;
  sysinit((sizeof(int)*image_size)*2, 32);

  printf("membase: %08.8x\n", (Uint)membase);
  ACCI = (Uchar*)((Uchar*)membase);
  ACCO = (Uint*) ((Uchar*)ACCI + (sizeof(int)*image_size));
  printf("ACCI: %08.8x\n", ACCI);
  printf("ACCO: %08.8x\n", ACCO);

  for (i=0; i<HT-200; i++) {
    for (j=0; j<WD; j++) {
      ACCI[i*WD+j] = i*65536 + j*256;
      if ((j % 8)==0)
	ACCI[i*WD+j] = 0;
      if ((j % 8)==1)
	ACCI[i*WD+j] = 0;
    }
  }

  while (1) {
    test_kernel(); /* search triangle in {frontier,next} */
    for (i=1; i<HT-200-1; i++) { /* scan-lines */
      int p0 = i*WD;
      for (j=1; j<WD-1; j++) {
	W[p0] = *(ACCO+i*WD+j)<<24 | *(ACCO+i*WD+j)<<16;
	p0++;
      }
    }
#ifdef ARMSIML
    _copyX(0, ACCI);
    _copyX(1, W);
    _copyX(2, ACCI);
    _updateX();
#endif
    break;
  }
  exit(0);
}

void *test_kernel()
{
  int y;
  for (y=1; y<HT-200-1; y++)
    test_x1(ACCI+y*WD, ACCO+y*WD);
//EMAX5A drain_dirty_lmm
//emax5_drain_dirty_lmm();
}

test_x1(Uint *yin, Uchar *yout)
{
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
  Ull  AR[16][4];                     /* output of EX     in each unit */
  Ull  BR[16][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  c0, c1, c2, c3, ex0, ex1;
  Sll  j=yin;
  int  p0=yin-320;
  int  p1=yin;
  int  p2=yin+320;
  int  loop=WD-2;
#if 1
//EMAX5A begin x1 mapdist=1
  while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
    /*@0,1*/ exe(OP_ADD,       &j,    j,    EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@1,0*/ mop(OP_LDWR, 1,   &r5,   j,     -1276, MSK_D0,    (Ull)p0,       320,    0, 0, (Ull)NULL,       320);
    /*@1,0*/ mop(OP_LDWR, 1,   &r3,   j,     -1280, MSK_D0,    (Ull)p0,       320,    0, 0, (Ull)NULL,       320);
    /*@1,1*/ mop(OP_LDWR, 1,   &r1,   j,     -1284, MSK_D0,    (Ull)p0,       320,    0, 0, (Ull)NULL,       320);

    /*@2,0*/ exe(OP_NOP,    &AR[2][0],0LL,  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,  0LL,          OP_NOP,  0LL);
    /*@2,0*/ mop(OP_LDWR, 1,   &r8,   j,      4,    MSK_D0,    (Ull)p1,       320,    0, 0, (Ull)NULL,       320);
    /*@2,0*/ mop(OP_LDWR, 1,   &r7,   j,     -4,    MSK_D0,    (Ull)p1,       320,    0, 0, (Ull)NULL,       320);

    /*@3,0*/ exe(OP_NOP,    &AR[3][0],0LL,  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,  0LL,          OP_NOP,  0LL);
    /*@3,0*/ mop(OP_LDWR, 1,   &r2,   j,      1284, MSK_D0,    (Ull)p2,       320,    0, 0, (Ull)NULL,       320);
    /*@3,0*/ mop(OP_LDWR, 1,   &r4,   j,      1280, MSK_D0,    (Ull)p2,       320,    0, 0, (Ull)NULL,       320);
    /*@3,1*/ mop(OP_LDWR, 1,   &r6,   j,      1276, MSK_D0,    (Ull)p2,       320,    0, 0, (Ull)NULL,       320);
    /*@3,1*/ exe(OP_MSSAD,     &r7,   0LL,  EXP_H3210, r7,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@4,0*/ exe(OP_MSSAD,     &r1,   0LL,  EXP_H3210, r1,  EXP_H3210, r2,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,1*/ exe(OP_MSSAD,     &r3,   0LL,  EXP_H3210, r3,  EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,2*/ exe(OP_MSSAD,     &r5,   0LL,  EXP_H3210, r5,  EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@5,0*/ exe(OP_MAUH,      &r1,   r3,   EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@5,1*/ exe(OP_MAUH,      &r5,   r7,   EXP_H3210, r5,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@6,0*/ exe(OP_MAUH,      &r1,   r5,   EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL,          OP_NOP,  0LL);

    /*@7,0*/ exe(OP_MCAS,      &r31,  r1,   EXP_H3210, 64,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@7,0*/ mop(OP_STBR, 3,   &r31,  yout++,    0LL,  MSK_D0,    (Ull)yout,       80,     0,  0, (Ull)NULL,       80);
  }
//EMAX5A end
#endif
//emax5_start((Ull*)emax5_conf_x1, (Ull*)emax5_lmmi_x1, (Ull*)emax5_regv_x1);

  return(0);
}
