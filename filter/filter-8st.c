
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */
/* filter.c 2002/4/18 */

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
#include <time.h>
#include <fcntl.h>
#include <errno.h>
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
#define AHT 240

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

unsigned int   *L;
unsigned int   *R;
unsigned int   *W;
unsigned int   *D;
unsigned char  *lut;
struct SADmin { unsigned int    SADmin[AHT][AWD];}         *SADmin;
struct SAD1   { unsigned short  SAD1[AHT/4][8][AWD/4][8];} *SAD1;
struct SAD2   { unsigned int    SAD2[AHT][AWD];}           *SAD2;
struct X      { unsigned int    X[AHT][1024];}             *Xl;
struct        { unsigned int    X[AHT][1024];}             *Xr;
unsigned int   *Bl;
unsigned int   *Br;
struct E      { unsigned char   E[AHT][AWD];}             *El;
struct        { unsigned char   E[AHT][AWD];}             *Er;
struct F      { unsigned char   F[AHT][AWD];}             *Fl;
struct        { unsigned char   F[AHT][AWD];}             *Fr;
struct D      { unsigned char   D[AHT][AWD];}             *Dl;
struct        { unsigned char   D[AHT][AWD];}             *Dr;

#define ad(a,b)   ((a)<(b)?(b)-(a):(a)-(b))
#define ss(a,b)   ((a)<(b)?   0   :(a)-(b))

/************/
/* for File */
/************/
#define PPMHEADER   15

char magic[PPMHEADER] = {
  0x50, 0x36, 0x0A,        /* P6  */
  0x33, 0x32, 0x30, 0x20,  /* 320 */
  0x32, 0x34, 0x30, 0x0A,  /* 240 */
  0x32, 0x35, 0x35, 0x0A,  /* 255 */
};

ppm_capt(fr0, fr1)
     unsigned char *fr0, *fr1;
{
  read_ppm(fr0, L);
  read_ppm(fr1, R);
}

read_ppm(file, ppm)
     unsigned char *file;
     unsigned int  *ppm;
{
  FILE  *fp;
  unsigned char buf[BITMAP*3];
  int i, j;

  if (!(fp = fopen(file, "r"))) {
    printf("Can't open %s\n", file);
    exit(1);
  }
  
  if (fread(buf, 1, PPMHEADER, fp) !=  PPMHEADER){
    printf("Can't read ppm header from %s\n", file);
    exit(1);
  }

  if (memcmp(buf, magic, PPMHEADER)) {
    printf("Can't match ppm header from %s\n", file);
    exit(1);
  }
  
  if (fread(buf, 1, BITMAP*3, fp) != BITMAP*3) {
    printf("Can't read ppm body from %s\n", file);
    exit(1);
  }

  for (i=0; i<HT; i++) {
    for (j=0; j<WD; j++) {
      ppm[i*WD+j] = buf[(i*WD+j)*3]<<8|buf[(i*WD+j)*3+1]<<16|buf[(i*WD+j)*3+2]<<24; /* RGB -> BGR */
    }
  }

  fclose(fp);
}

write_ppm(file, ppm)
     unsigned char *file;
     unsigned int  *ppm;
{
  FILE  *fp;
  unsigned char buf[BITMAP*3];
  int i, j;
  
  if (!(fp = fopen(file, "w"))) {
    printf("Can't open %s\n", file);
    exit(1);
  }
  
  if (fwrite(magic, 1, PPMHEADER, fp) !=  PPMHEADER){
    printf("Can't write ppm header to %s\n", file);
    exit(1);
  }

  for (i=0; i<HT; i++) {
    for (j=0; j<WD; j++) {
      buf[(i*WD+j)*3  ] = ppm[i*WD+j]>>24;
      buf[(i*WD+j)*3+1] = ppm[i*WD+j]>>16;
      buf[(i*WD+j)*3+2] = ppm[i*WD+j]>> 8;
    }
  }

  if (fwrite(buf, 1, BITMAP*3, fp) != BITMAP*3) {
    printf("Can't write ppm body to %s\n", file);
    exit(1);
  }

  fclose(fp);
}

#define FROMFILE     1
int            flag;
unsigned char  *fr0, *fr1;

main(argc, argv)
     int argc;
     char **argv;
{
  static wait_previous = 0;
  int i, j, k, l, dx, dy;

  for (argc--, argv++; argc; argc--, argv++) {
    if (**argv != '-') { /* regard as a name */
      if (!fr0)      strcpy(fr0  = (char*)malloc(strlen(*argv) + 1), *argv);
      else if (!fr1) strcpy(fr1  = (char*)malloc(strlen(*argv) + 1), *argv);
      continue;
    }
    switch (*(*argv+1)) {
    case 'f':
      flag |= FROMFILE;
      break;
    default:
      printf("usage: filter -f left right\n");
      exit(1);
    }
  }

  sysinit(
    +(sizeof(int)*BITMAP)    /* L */
    +(sizeof(int)*BITMAP)    /* R */
    +(sizeof(int)*BITMAP)    /* W */
    +(sizeof(int)*BITMAP)    /* D */
    +(256*3)                 /* lut */
    +(sizeof(struct SADmin)) /* SADmin */
    +(sizeof(struct SAD1))   /* SAD1 */
    +(sizeof(struct SAD2))   /* SAD2 */
    +(sizeof(struct X))      /* Xl */
    +(sizeof(struct X))      /* Xr */
    +(sizeof(int)*BITMAP)    /* Bl */
    +(sizeof(int)*BITMAP)    /* Br */
    +(sizeof(struct E))      /* El */
    +(sizeof(struct E))      /* Er */
    +(sizeof(struct F))      /* Fl */
    +(sizeof(struct F))      /* Fr */
    +(sizeof(struct D))      /* Dl */
    +(sizeof(struct D))      /* Dr */
    ,32);

  printf("membase: %08.8x\n", (Uint)membase);
  L      = (Uint*)         ((Uchar*)membase);
  R      = (Uint*)         ((Uchar*)L     + (sizeof(int)*BITMAP));
  W      = (Uint*)         ((Uchar*)R     + (sizeof(int)*BITMAP));
  D      = (Uint*)         ((Uchar*)W     + (sizeof(int)*BITMAP));
  lut    = (Uchar*)        ((Uchar*)D     + (sizeof(int)*BITMAP));
  SADmin = (struct SADmin*)((Uchar*)lut   + (256*3));
  SAD1   = (struct SAD1*)  ((Uchar*)SADmin+ (sizeof(struct SADmin)));
  SAD2   = (struct SAD2*)  ((Uchar*)SAD1  + (sizeof(struct SAD1)));
  Xl     = (struct X*)     ((Uchar*)SAD2  + (sizeof(struct SAD2)));
  Xr     = (struct X*)     ((Uchar*)Xl    + (sizeof(struct X)));
  Bl     = (Uint*)         ((Uchar*)Xr    + (sizeof(struct X)));
  Br     = (Uint*)         ((Uchar*)Bl    + (sizeof(int)*BITMAP));
  El     = (struct E*)     ((Uchar*)Br    + (sizeof(int)*BITMAP));
  Er     = (struct E*)     ((Uchar*)El    + (sizeof(struct E)));
  Fl     = (struct F*)     ((Uchar*)Er    + (sizeof(struct E)));
  Fr     = (struct F*)     ((Uchar*)Fl    + (sizeof(struct F)));
  Dl     = (struct D*)     ((Uchar*)Fr    + (sizeof(struct F)));
  Dr     = (struct D*)     ((Uchar*)Dl    + (sizeof(struct D)));
  printf("L     : %08.8x\n", L);
  printf("R     : %08.8x\n", R);
  printf("W     : %08.8x\n", W);
  printf("D     : %08.8x\n", D);
  printf("lut   : %08.8x\n", lut);
  printf("SADmin: %08.8x\n", SADmin->SADmin);
  printf("SAD1  : %08.8x\n", SAD1->SAD1);
  printf("SAD2  : %08.8x\n", SAD2->SAD2);
  printf("Xl    : %08.8x\n", Xl->X);
  printf("Xr    : %08.8x\n", Xr->X);
  printf("Bl    : %08.8x\n", Bl);
  printf("Br    : %08.8x\n", Br);
  printf("El    : %08.8x\n", El->E);
  printf("Er    : %08.8x\n", Er->E);
  printf("Fl    : %08.8x\n", Fl->F);
  printf("Fr    : %08.8x\n", Fr->F);
  printf("Dl    : %08.8x\n", Dl->D);
  printf("Dr    : %08.8x\n", Dr->D);

#if !defined(ARMSIML)
  x11_open(0);
#endif
  ppm_capt(fr0, fr1);

  for(i=0; i<256; i++) {
    lut[i+  0] = 0xff-i;
    lut[i+256] = 0xff-i;
    lut[i+512] = 0xff-i;
  }

  /*****************************************************/
  /* gamma */
  puts("tone-start");
#ifdef ARMSIML
  _getpa();
#endif

  for(i=0; i<HT; i++)
    tone_curve( &R[i*WD], &D[i*WD], lut );
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("tone-end");
#ifdef ARMSIML
  _copyX(0, L);
  _copyX(1, R);
  _copyX(2, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(0, L);
  BGR_to_X(1, R);
  BGR_to_X(2, D);
  x11_update();
#endif

#ifndef ARMSIML
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (!x11_checkevent());
#endif

  exit(0);
}

msll(s1, s2)
     unsigned int s1, s2;
{
  return ((s1&0xffff0000) << s2)|((s1 << s2)&0x0000ffff); /* immediate */
}
msrl(s1, s2)
     unsigned int s1, s2;
{
  return ((s1 >> s2)&0xffff0000)|((s1&0x0000ffff) >> s2); /* immediate */
}
msra(s1, s2)
     unsigned int s1, s2;
{
  return (((int)s1 >> s2)&0xffff0000)|(((int)(s1<<16)>>(16+s2))&0x0000ffff); /* immediate */
}
b2h(s1, s2) /* BYTE->HALF s2==0: 0x00001122 -> 0x00110022, s2==1: 0x11220000 -> 0x00110022 */
     unsigned int s1, s2;
{
  return s2==0 ? ((s1<<8) & 0x00ff0000) | ( s1      & 0x000000ff):
                 ((s1>>8) & 0x00ff0000) | ((s1>>16) & 0x000000ff);
}
h2b(s1, s2) /* HALF->BYTE s2==0: 0x11112222 -> 0x00001122, s2==1: 0x11112222 -> 0x11220000 */
     unsigned int s1, s2;
{
  return s2==0 ? (((s1>>16   )<0x100 ? (s1>>16   )&0xff : 255)<< 8)
                |(((s1&0xffff)<0x100 ? (s1&0xffff)&0xff : 255)    )
               : (((s1>>16   )<0x100 ? (s1>>16   )&0xff : 255)<<24)
                |(((s1&0xffff)<0x100 ? (s1&0xffff)&0xff : 255)<<16);
}
madd(s1, s2) /* ADD (16bit+16bit)[2] -> 16bit[2] */
     unsigned int s1, s2;
{
  unsigned int t1, t2;
  t1 = ( s1     >>16)+( s2     >>16);
  if (t1 > 0x0000ffff) t1 = 0xffff;
  t2 = ((s1<<16)>>16)+((s2<<16)>>16);
  if (t2 > 0x0000ffff) t2 = 0xffff;
  return (t1<<16)|t2;
}
msub(s1, s2) /* SUB (16bit-16bit)[2] -> 16bit[2] */
     unsigned int s1, s2;
{
  unsigned int t1, t2;
  t1 = ( s1     >>16)-( s2     >>16);
  if (t1 > 0x0000ffff) t1 = 0x0000;
  t2 = ((s1<<16)>>16)-((s2<<16)>>16);
  if (t2 > 0x0000ffff) t2 = 0x0000;
  return (t1<<16)|t2;
}
mmul(s1, s2) /* MUL (10bit*9bit)[2] -> 16bit[2] */
     unsigned int s1, s2;
{
  unsigned int t1, t2;
  t1 = (( s1     >>16)&0x3ff)*(s2&0x1ff);
  if (t1 > 0x0000ffff) t1 = 0xffff;
  t2 = (((s1<<16)>>16)&0x3ff)*(s2&0x1ff);
  if (t2 > 0x0000ffff) t2 = 0xffff;
  return (t1<<16)|t2;
}
pmin3(x, y, z)
     unsigned int x, y, z;
{
  unsigned char r[3], g[3], b[3];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;  r[2]=z>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;  g[2]=z>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;  b[2]=z>> 8&0xff;
  if (r[0] < r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] < g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] < b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  if (r[1] < r[2]) {t = r[2]; r[2]=r[1]; r[1]=t;}
  if (g[1] < g[2]) {t = g[2]; g[2]=g[1]; g[1]=t;}
  if (b[1] < b[2]) {t = b[2]; b[2]=b[1]; b[1]=t;}
  return (r[2]<<24)|(g[2]<<16)|(b[2]<<8);
}
pmid3(x, y, z)
     unsigned int x, y, z;
{
  unsigned char r[3], g[3], b[3];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;  r[2]=z>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;  g[2]=z>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;  b[2]=z>> 8&0xff;
  if (r[0] > r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] > g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] > b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  if (r[1] > r[2]) {t = r[2]; r[2]=r[1]; r[1]=t;}
  if (g[1] > g[2]) {t = g[2]; g[2]=g[1]; g[1]=t;}
  if (b[1] > b[2]) {t = b[2]; b[2]=b[1]; b[1]=t;}
  if (r[0] > r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] > g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] > b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  return (r[1]<<24)|(g[1]<<16)|(b[1]<<8);
}
pmax3(x, y, z)
     unsigned int x, y, z;
{
  unsigned char r[3], g[3], b[3];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;  r[2]=z>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;  g[2]=z>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;  b[2]=z>> 8&0xff;
  if (r[0] > r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] > g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] > b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  if (r[1] > r[2]) {t = r[2]; r[2]=r[1]; r[1]=t;}
  if (g[1] > g[2]) {t = g[2]; g[2]=g[1]; g[1]=t;}
  if (b[1] > b[2]) {t = b[2]; b[2]=b[1]; b[1]=t;}
  return (r[2]<<24)|(g[2]<<16)|(b[2]<<8);
}
pmin2(x, y)
     unsigned int x, y;
{
  unsigned char r[2], g[2], b[2];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;
  if (r[0] < r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] < g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] < b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  return (r[1]<<24)|(g[1]<<16)|(b[1]<<8);
}
pmax2(x, y)
     unsigned int x, y;
{
  unsigned char r[2], g[2], b[2];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;
  if (r[0] > r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] > g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] > b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  return (r[1]<<24)|(g[1]<<16)|(b[1]<<8);
}
df(l, r)
     unsigned int l, r;
{
  return(ad(l>>24&0xff,r>>24&0xff)+ad(l>>16&0xff,r>>16&0xff)+ad(l>>8&0xff,r>>8&0xff));
}

void tone_curve(r, d, t)
     unsigned int *r, *d;
     unsigned char *t;
{
#if !defined(EMAX5) && !defined(EMAX6)
  int j;
  for (j=0; j<WD; j++) {
    *d = ((t)[*r>>24])<<24 | (t[256+((*r>>16)&255)])<<16 | (t[512+((*r>>8)&255)])<<8;
    r++; d++;
  }
#else
//EMAX4A start .emax_start_tone_curve:
//EMAX4A ctl map_dist=0
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]  & ld   (ri+=,4),r9   rgi[.emax_rgi00_tone_curve:,] lmf[.emax_lmfla0_tone_curve:,0,0,0,0,.emax_lmfma0_tone_curve:,320] ! lmm_top mem_bank width block dist top len
//EMAX4A @1,0                            & ldub (ri,r9.3),r10 rgi[.emax_rgi01_tone_curve:,] lmr[.emax_lmrla1_tone_curve:,0,0,0,0,.emax_lmrma1_tone_curve:, 64]
//EMAX4A @1,1                            & ldub (ri,r9.2),r11 rgi[.emax_rgi02_tone_curve:,] lmr[.emax_lmrla2_tone_curve:,0,0,0,0,.emax_lmrma2_tone_curve:, 64]
//EMAX4A @1,2                            & ldub (ri,r9.1),r12 rgi[.emax_rgi03_tone_curve:,] lmr[.emax_lmrla3_tone_curve:,0,0,0,0,.emax_lmrma3_tone_curve:, 64]
//EMAX4A @2,0 mmrg3 (r10,r11,r12) rgi[,] & st   -,(ri+=,4)    rgi[.emax_rgi04_tone_curve:,] lmw[.emax_lmwla4_tone_curve:,0,0,0,0,.emax_lmwma4_tone_curve:,320]
//EMAX4A end .emax_end_tone_curve:
  Ull  t1 = t;
  Ull  t2 = t+256;
  Ull  t3 = t+512;
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define TONE_CURVE_32BIT
#ifdef TONE_CURVE_32BIT
  int loop=WD;
//EMAX5A begin tone_curve mapdist=0
  while (loop--) {
    mop(OP_LDWR,  1, &BR[0][1][1], (Ull)(r++), 0LL,         MSK_D0, (Ull)r, 320,   0, 0, (Ull)NULL, 320);   /* stage#0 */
    mop(OP_LDBR,  1, &BR[1][1][1], (Ull)t1,    BR[0][1][1], MSK_B3, (Ull)t1, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][2][1], (Ull)t2,    BR[0][1][1], MSK_B2, (Ull)t2, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][3][1], (Ull)t3,    BR[0][1][1], MSK_B1, (Ull)t3, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    exe(OP_MMRG, &r1, BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, BR[1][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    mop(OP_STWR,  3, &r1,          (Ull)(d++), 0LL,         MSK_D0, (Ull)d, 320,   0, 0, (Ull)NULL, 320);   /* stage#2 */
  }
//EMAX5A end
#endif
#ifdef TONE_CURVE_64BIT
  Ull *rr = r;
  Ull *dd = d;
  int loop=WD/2;
//EMAX5A begin tone_curve mapdist=0
  while (loop--) {
    mop(OP_LDR,   1, &BR[0][1][1], (Ull)(rr++), 0LL,        MSK_D0, (Ull)r, 320,   0, 0, (Ull)NULL, 320);   /* stage#0 */
    mop(OP_LDBR,  1, &BR[1][1][1], (Ull)t1,    BR[0][1][1], MSK_B3, (Ull)t1, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][1][0], (Ull)t1,    BR[0][1][1], MSK_B7, (Ull)t1, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][2][1], (Ull)t2,    BR[0][1][1], MSK_B2, (Ull)t2, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][2][0], (Ull)t2,    BR[0][1][1], MSK_B6, (Ull)t2, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][3][1], (Ull)t3,    BR[0][1][1], MSK_B1, (Ull)t3, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][3][0], (Ull)t3,    BR[0][1][1], MSK_B5, (Ull)t3, 64,  0,  0, (Ull)NULL, 64);   /* stage#1 */
    exe(OP_CCAT,  &r1, BR[1][1][0], EXP_H3210, BR[1][1][1], EXP_H3210,        0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    exe(OP_CCAT,  &r2, BR[1][2][0], EXP_H3210, BR[1][2][1], EXP_H3210,        0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    exe(OP_CCAT,  &r3, BR[1][3][0], EXP_H3210, BR[1][3][1], EXP_H3210,        0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    exe(OP_MMRG,  &r0,          r1, EXP_H3210, r2, EXP_H3210, r3, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    mop(OP_STR,   3, &r0,          (Ull)(dd++), 0LL,        MSK_D0, (Ull)d, 320,   0, 0, (Ull)NULL, 320);   /* stage#2 */
  }
//EMAX5A end
#endif
#endif
}
