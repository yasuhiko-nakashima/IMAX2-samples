
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
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <netinet/in.h>
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

/************/
/* for Time */
/************/
#ifndef ARMSIML
double        tmssave, tms;
long          ticksave, ticks;
struct rusage rusage;

void reset_time(void)
{
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  tmssave = tv.tv_sec+tv.tv_usec/1000000.0;

  times(&utms);
  ticksave = utms.tms_utime;
}

void show_time(void)
{
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  tms = tv.tv_sec+tv.tv_usec/1000000.0;
  printf("====TOTAL-EXEC-TIME(w/o IO) %g sec===\n", (double)(tms - tmssave));

  times(&utms);
  ticks = utms.tms_utime;
  printf("====TOTAL-CPUS-TIME(w/o IO) %g sec===\n", (double)(ticks-ticksave)/sysconf(_SC_CLK_TCK));
}
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
      ppm[i*WD+j] = buf[(i*WD+j)*3+2]<<24|buf[(i*WD+j)*3+1]<<16|buf[(i*WD+j)*3]<<8; /* RGB -> BGR */
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
#define FROMCAM      2
int            flag;
unsigned char  *fr0, *fr1;

#ifndef ARMSIML
unsigned char  *host;
int            port = 1518;
int            camfd;
struct sockaddr_in serv_addr;
#endif

main(argc, argv)
     int argc;
     char **argv;
{
  static wait_previous = 0;
  int i, j, k, l, dx, dy;

  for (argc--, argv++; argc; argc--, argv++) {
    if (**argv != '-') { /* regard as a name */
      if (flag & FROMFILE) {
	if      (!fr0) strcpy(fr0 = (char*)malloc(strlen(*argv) + 1), *argv);
	else if (!fr1) strcpy(fr1 = (char*)malloc(strlen(*argv) + 1), *argv);
      }
#ifndef ARMSIML
      if (flag & FROMCAM) {
	if     (!host) strcpy(host = (char*)malloc(strlen(*argv) + 1), *argv);
      }
#endif
      continue;
    }
    switch (*(*argv+1)) {
    case 'f':
      flag |= FROMFILE;
      break;
#ifndef ARMSIML
    case 'c':
      flag |= FROMCAM;
      break;
#endif
    default:
      printf("usage: filter -f left right\n");
#ifndef ARMSIML
      printf("usage: filter -c host(ip_addr)\n");
#endif
      exit(1);
    }
  }

#if !defined(ARMSIML)
  if (host) {
  }
#endif

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

  do {
    if (flag & FROMFILE)
      ppm_capt(fr0, fr1);
#ifndef ARMSIML
    if (flag & FROMCAM)
      cam_capt();
#endif

  /*****************************************************/
  /* gamma */
  puts("tone-start");
  reset_nanosec();
  tone_curve(R, D, lut);
  get_nanosec(0);
  show_nanosec();
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

  /*****************************************************/
  /* shift */
  for (i=0; i<HT; i++) { /* scan-lines */
    int pp = i*WD;
    for (j=0; j<WD; j++) {
      W[pp] = R[pp];
      pp++;
    }
  }
  for (i=120; i<200; i++) { /* scan-lines */
    for (j=120; j<200; j++) {
      W[i*WD+j+4] = R[i*WD+j];
    }
  }
  for (i=20; i<100; i++) { /* scan-lines */
    for (j=20; j<100; j++) {
      W[(i+4)*WD+j+4] = R[i*WD+j];
    }
  }
  for (i=20; i<100; i++) { /* scan-lines */
    for (j=220; j<300; j++) {
      W[(i+4)*WD+j] = R[i*WD+j];
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

  /*****************************************************/
  /* hokan1 SAD·×»» */
  puts("hokan1-start");
  reset_nanosec();
  hokan1(W, R, SAD1);
  get_nanosec(0);
  show_nanosec();
  puts("hokan1-end");

  for (i=0; i<HT; i++) { /* scan-lines */
    int pp = i*WD;
    for (j=0; j<WD; j++) {
      D[pp] = (int)(SAD1->SAD1[i/4][i%4*2][j/4][j%4*2]/32)<<16;
      pp++;
    }
  }

#ifdef ARMSIML
  _copyX(4, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(4, D);
  x11_update();
#endif

  /***************************************************/
  /* hokan2 SADºÇ¾®ÃÍ¤ËÂÐ±þ¤¹¤ëXY¤ò·×»»              */
  /* for (i=0; i<HT; i+=4) {                         */
  /*   for (j=0; j<WD; j+=4) {                       */
  /*     int sadmin = 0xffff;                        */
  /*     int x = 0, y = 0;                           */
  /*     for (k=-4; k<4; k++) {                      */
  /*       for (l=-4; l<4; l++) {                    */
  /*         if (sadmin > SAD1[i/4][k+4][j/4][l+4]) {*/
  /*           sadmin = SAD1[i/4][k+4][j/4][l+4];    */
  /*           x = l; y = k;                         */
  /*         }                                       */
  /*       }                                         */
  /*     }                                           */
  /*     W[i*WD+j] = (x/2<<16)|((y/2)&0xffff);       */
  /*   }                                             */
  /* }                                               */
  /***************************************************/
  /* 8²óÁöºº¤·¤ÆSADºÇ¾®°ÌÃÖ¤òµá¤á¤ë */
  puts("hokan2-start");
  reset_nanosec();
  hokan2(SAD1, W);
  get_nanosec(0);
  show_nanosec();
  puts("hokan2-end");

  for (i=0; i<HT; i++) { /* scan-lines */
    for (j=0; j<WD; j++) {
      int x = (int) W[(i/4*4)*WD+(j/4*4)]>>24;
      int y = (int)(W[(i/4*4)*WD+(j/4*4)]<<8)>>24;
      D[i*WD+j] = (ad(y,0)<<22)|(ad(x,0)<<14); /* BGR */
    }
  }

#ifdef ARMSIML
  _copyX(5, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(5, D);
  x11_update();
#endif

  /*****************************************************/
  /* hokan3 XY¤ò¸µ¤Ë, T1·×»»                           */
  /* for (i=0; i<HT; i+=4) {                           */
  /*   for (j=0; j<WD; j+=4) {                         */
  /*     int x = (int) W[i*WD+j]>>16;                  */
  /*     int y = (int)(W[i*WD+j]<<16)>>16;             */
  /*     for (k=0; k<4; k++) {                         */
  /*       for (l=0; l<4; l++) {                       */
  /*         D[(i+k)*WD+(j+l)] = R[(i+k+y)*WD+(j+l+x)];*/
  /*       }                                           */
  /*     }                                             */
  /*   }                                               */
  /* }                                                 */
  /*****************************************************/
  puts("hokan3-start");
  reset_nanosec();
  hokan3(W, R, D);
  get_nanosec(0);
  show_nanosec();
  puts("hokan3-end");

#ifdef ARMSIML
  _copyX(6, W);
  _copyX(7, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(6, W);
  BGR_to_X(7, D);
  x11_update();
#endif

  /*****************************************************/
  /* expand 3x3 */
  puts("expandR-start");
  reset_nanosec();
  expand4k(L, Xr);
  get_nanosec(0);
  show_nanosec();
  puts("expandR-end");

  for (i=0; i<HT; i++) { /* scan-lines */
    int pp = i*WD;
    for (j=0; j<WD; j++) {
      W[pp] = Xr->X[i][j];
      pp++;
    }
  }

#ifdef ARMSIML
  _copyX(8, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(8, W);
  x11_update();
#endif

  /*****************************************************/
  /* unsharp 3x3 */
  puts("unsharpR-start");
  reset_nanosec();
  unsharp(R, D);
  get_nanosec(0);
  show_nanosec();
  puts("unsharpR-end");

#ifdef ARMSIML
  _copyX(9, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(9, D);
  x11_update();
#endif

  /*****************************************************/
  /* blur 3x3 */
  puts("blur-start");
  reset_nanosec();
  blur(L, Bl);
  blur(R, Br);
  get_nanosec(0);
  show_nanosec();
  puts("blur-end");

#ifdef ARMSIML
  _copyX(10, Bl);
  _copyX(11, Br);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(10, Bl);
  BGR_to_X(11, Br);
  x11_update();
#endif

  /*****************************************************/
  /* edge detection */
  puts("edge-start");
  reset_nanosec();
  edge(Bl, El);
  edge(Br, Er);
  get_nanosec(0);
  show_nanosec();
  puts("edge-end");

  for (i=0; i<HT; i++) { /* scan-lines */
    int pp = i*WD;
    for (j=0; j<WD; j++) {
      W[pp] = (El->E[i][j])<<8 | (Er->E[i][j])<<16;
      pp++;
    }
  }

#ifdef ARMSIML
  _copyX(0, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(12, W);
  x11_update();
#endif

  /*****************************************************/
  /* dusts removal */
  puts("dust-start");
  bblur(El, Fl);
  bblur(Er, Fr);
  puts("dust-end");

  for (i=0; i<HT; i++) { /* scan-lines */
    int pp = i*WD;
    for (j=0; j<WD; j++) {
      W[pp] = (Fl->F[i][j])<<8 | (Fr->F[i][j])<<16;
      pp++;
    }
  }

#ifdef ARMSIML
  _copyX(1, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(13, W);
  x11_update();
#endif

  /*****************************************************/
  puts("depth-start");
  reset_nanosec();
  Depth_retrieval_L();
  Depth_retrieval_R();
  get_nanosec(0);
  show_nanosec();
  puts("depth-end");

#undef  PIXMAX
#undef  DMAX
#undef  DMIN
#define PIXMAX      255
#define DMAX         80
#define DWIN          8
  /* merge result */
  puts("merge");
  for (i=0; i<HT; i++) { /* scan-lines */
    int pp = i*WD;
    for (j=0; j<WD; j++) {
      D[pp] = (Dl->D[i][j]<Dr->D[i][j]?Dl->D[i][j]:Dr->D[i][j])*PIXMAX/DMAX;
      pp++;
    }
  }

  puts("filter1");
  for (i=1; i<HT-1; i++) { /* scan-lines */
    int p0 = (i  )*WD  ;
    int p1 = (i  )*WD  ;
    int p2 = (i  )*WD-1;
    int p3 = (i  )*WD+1;
    int p4 = (i-1)*WD  ;
    int p5 = (i+1)*WD  ;
    int p6 = (i-1)*WD-1;
    int p7 = (i-1)*WD+1;
    int p8 = (i+1)*WD-1;
    int p9 = (i+1)*WD+1;
    for (j=0; j<WD; j++) {
      unsigned char s[9], t;
      s[0]=D[p1];s[1]=D[p2];s[2]=D[p3];s[3]=D[p4];s[4]=D[p5];s[5]=D[p6];s[6]=D[p7];s[7]=D[p8];s[8]=D[p9];
      for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
      Bl[p0]  = s[5];
      p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; p9++;
    }
  }

  puts("filter2");
  for (i=1; i<HT-1; i++) { /* scan-lines */
    int p0 = (i  )*WD  ;
    int p1 = (i  )*WD  ;
    int p2 = (i  )*WD-1;
    int p3 = (i  )*WD+1;
    int p4 = (i-1)*WD  ;
    int p5 = (i+1)*WD  ;
    int p6 = (i-1)*WD-1;
    int p7 = (i-1)*WD+1;
    int p8 = (i+1)*WD-1;
    int p9 = (i+1)*WD+1;
    for (j=0; j<WD; j++) {
      unsigned char s[9], t;
      s[0]=Bl[p1];s[1]=Bl[p2];s[2]=Bl[p3];s[3]=Bl[p4];s[4]=Bl[p5];s[5]=Bl[p6];s[6]=Bl[p7];s[7]=Bl[p8];s[8]=Bl[p9];
      for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
      D[p0]  = s[5];
      p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; p9++;
    }
  }

  for (i=0; i<HT; i++) { /* scan-lines */
    int pp = i*WD;
    for (j=0; j<WD; j++) {
      W[pp] = (Dl->D[i][j] | Fl->F[i][j])<<8 | (Dr->D[i][j] | Fr->F[i][j])<<16;
      D[pp] = D[pp]<<24 | D[pp]<<16 | D[pp]<<8;
      pp++;
    }
  }

#ifdef ARMSIML
  _copyX(2, W);
  _copyX(3, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(14, W);
  BGR_to_X(15, D);
  x11_update();
#endif

  } while (flag & FROMCAM);

#ifndef ARMSIML
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (!x11_checkevent());
#endif

  exit(0);
}

Depth_retrieval_L()
{
#undef  PIXMAX
#undef  DMAX
#undef  DMIN
#undef  CORRDET
#define PIXMAX      255
#define DMAX         80
#define DWIN          8
#define CORRDET      ((DWIN*2)*(DWIN*2)*16*16)

  int i, j, k, l;
  for (i=DWIN; i<HT-DWIN; i++) { /* scan-lines */
    for (j=DWIN; j<WD-DWIN; j++) {
      SADmin->SADmin[i][j] = CORRDET;
      Dl->D[i][j] = 0;
    }
  }
  /* slide-WIN */
  for (k=0; k<DMAX*4; k+=2) {
    if (k < DMAX)
      wdifline(Bl, Br, SAD2, k);
    for (i=DWIN; i<HT-DWIN; i++) { /* scan-lines */
      for (j=WD-1; j>=1; j--) { /* one scan-line */
        if (!((j+k/2<=WD-1)?Fl->F[i][j+k/2]:0) && Dl->D[i][j] < Dl->D[i][j-1])
          Dl->D[i][j] = Dl->D[i][j-1];
      }
      if (k < DMAX) {
        for (j=WD-DWIN-k/2-1; j>=DWIN+k/2; j--) { /* one scan-line */
          if (SADmin->SADmin[i][j] > SAD2->SAD2[i][j]) {
            if (!Fl->F[i][j+k/2] || Fr->F[i][j-k/2]) /* ¥¨¥Ã¥¸¤Ç¤Ê¤±¤ì¤Ðº¸ÎÙ¤«¤é¥³¥Ô¡¼ */
              SADmin->SADmin[i][j] = SAD2->SAD2[i][j];
            if ( Fl->F[i][j+k/2] && Fr->F[i][j-k/2])
              Dl->D[i][j] = k;
          }
        }
      }
    }
  }
}

Depth_retrieval_R()
{
#undef  PIXMAX
#undef  DMAX
#undef  DMIN
#undef  CORRDET
#define PIXMAX      255
#define DMAX         80
#define DWIN          8
#define CORRDET      ((DWIN*2)*(DWIN*2)*16*16)

  int i, j, k, l;
  for (i=DWIN; i<HT-DWIN; i++) { /* scan-lines */
    for (j=DWIN; j<WD-DWIN; j++) {
      SADmin->SADmin[i][j] = CORRDET;
      Dr->D[i][j] = 0;
    }
  }
  /* slide-WIN */
  for (k=0; k<DMAX*4; k+=2) {
    if (k < DMAX)
      wdifline(Bl, Br, SAD2, k);
    for (i=DWIN; i<HT-DWIN; i++) { /* scan-lines */
      for (j=0; j<WD-1; j++) { /* one scan-line */
        if (!((j-k/2>=0)?Fr->F[i][j-k/2]:0) && Dr->D[i][j] < Dr->D[i][j+1])
          Dr->D[i][j] = Dr->D[i][j+1];
      }
      if (k < DMAX) {
        for (j=DWIN+k/2; j<WD-DWIN-k/2; j++) { /* one scan-line */
          if (SADmin->SADmin[i][j] > SAD2->SAD2[i][j]) {
            if (!Fr->F[i][j-k/2] || Fl->F[i][j+k/2])  /* ¥¨¥Ã¥¸¤Ç¤Ê¤±¤ì¤Ð±¦ÎÙ¤«¤é¥³¥Ô¡¼ */
              SADmin->SADmin[i][j] = SAD2->SAD2[i][j];
            if ( Fr->F[i][j-k/2] && Fl->F[i][j+k/2])
              Dr->D[i][j] = k;
          }
        }
      }
    }
  }
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

void tone_curve(Uint *r, Uint *d, Uchar *t) /* R, D, lut */
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     1
#define RMGRP     6
#define OMAP     10
#define PAD       0
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)

  int i;
  for(i=0; i<256; i++) {
    t[i+  0] = 0xff-i;
    t[i+256] = 0xff-i;
    t[i+512] = 0xff-i;
  }

  /* 240/NCHIP(4)/OMAP(10)=6  ... RRANGE=6  IMAXµ¯Æ°²ó¿ô(top)=1 */
  /* 240/NCHIP(2)/OMAP(10)=12 ... RRANGE=12 IMAXµ¯Æ°²ó¿ô(top)=2 */
  /* 240/NCHIP(1)/OMAP(10)=24 ... RRANGE=24 IMAXµ¯Æ°²ó¿ô(top)=4 */
  Ull  top, rofs, cofs, oc, pofs;
  Ull  t1 = t;
  Ull  t2 = t+256;
  Ull  t3 = t+512;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
#if 0
  for (top=PAD; top<HT-PAD; top++) { /* will be parallelized by multi-chip (M/#chip) */
    for (cofs=PAD; cofs<WD-PAD; cofs++) {
      Uint pix = *(r+top*WD+cofs);
      *(d+top*WD+cofs) = ((t)[pix>>24])<<24 | (t[256+((pix>>16)&255)])<<16 | (t[512+((pix>>8)&255)])<<8;
    }
  }
#else
  for (top=0; top<RRANGE; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
      for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
        int idx = (CHIP*RRANGE*OMAP+top+rofs)*WD;
        for (cofs=PAD; cofs<WD-PAD; cofs++) {
          for (oc=0; oc<OMAP; oc++) {
            Uint pix = *(r+idx+oc*RRANGE*WD+cofs);
            *(d+idx+oc*RRANGE*WD+cofs) = ((t)[pix>>24])<<24 | (t[256+((pix>>16)&255)])<<16 | (t[512+((pix>>8)&255)])<<8;
          }
        }
      }
    }
  }
#endif
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  for (top=0; top<RRANGE; top+=RMGRP) {
    Ull  rtop0[NCHIP], dtop0[NCHIP];
    Ull  rtop1[NCHIP], dtop1[NCHIP];
    Ull  rtop2[NCHIP], dtop2[NCHIP];
    Ull  rtop3[NCHIP], dtop3[NCHIP];
    Ull  rtop4[NCHIP], dtop4[NCHIP];
    Ull  rtop5[NCHIP], dtop5[NCHIP];
    Ull  rtop6[NCHIP], dtop6[NCHIP];
    Ull  rtop7[NCHIP], dtop7[NCHIP];
    Ull  rtop8[NCHIP], dtop8[NCHIP];
    Ull  rtop9[NCHIP], dtop9[NCHIP];
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
      rtop0[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*0+top)*WD; dtop0[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*0+top)*WD;
      rtop1[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*1+top)*WD; dtop1[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*1+top)*WD;
      rtop2[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*2+top)*WD; dtop2[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*2+top)*WD;
      rtop3[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*3+top)*WD; dtop3[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*3+top)*WD;
      rtop4[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*4+top)*WD; dtop4[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*4+top)*WD;
      rtop5[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*5+top)*WD; dtop5[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*5+top)*WD;
      rtop6[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*6+top)*WD; dtop6[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*6+top)*WD;
      rtop7[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*7+top)*WD; dtop7[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*7+top)*WD;
      rtop8[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*8+top)*WD; dtop8[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*8+top)*WD;
      rtop9[CHIP] = r+(CHIP*RRANGE*OMAP+RRANGE*9+top)*WD; dtop9[CHIP] = d+(CHIP*RRANGE*OMAP+RRANGE*9+top)*WD;
    }
//EMAX5A begin tone_curve mapdist=0
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
 /*2*/for (INIT1=1,LOOP1=RMGRP,rofs=0-AWD*4; LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
   /*1*/for (INIT0=1,LOOP0=AWD,cofs=0-4; LOOP0--; INIT0=0) {          /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
          exe(OP_ADD,  &cofs, INIT0?cofs:cofs, EXP_H3210, 4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);/* stage#0 */
          exe(OP_ADD,  &rofs, rofs, EXP_H3210, INIT0?AWD*4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                /* stage#0 */
          exe(OP_ADD,  &pofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);        /* stage#1 */
          /*map0*/
          mop(OP_LDWR,  1, &BR[2][1][1], (Ull)rtop0[CHIP], pofs,  MSK_D0, (Ull)rtop0[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#2 */
          mop(OP_LDBR,  1, &BR[3][1][1], (Ull)t1,    BR[2][1][1], MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#3 */
          mop(OP_LDBR,  1, &BR[3][2][1], (Ull)t2,    BR[2][1][1], MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#3 */
          mop(OP_LDBR,  1, &BR[3][3][1], (Ull)t3,    BR[2][1][1], MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#3 */
          exe(OP_MMRG, &r1, BR[3][1][1], EXP_H3210,  BR[3][2][1], EXP_H3210, BR[3][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#3 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop0[CHIP], pofs,  MSK_D0, (Ull)dtop0[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#3 */
          /*map1*/
          mop(OP_LDWR,  1, &BR[4][1][1], (Ull)rtop1[CHIP], pofs,  MSK_D0, (Ull)rtop1[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#4 */
          mop(OP_LDBR,  1, &BR[5][1][1], (Ull)t1,    BR[4][1][1], MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#5 */
          mop(OP_LDBR,  1, &BR[5][2][1], (Ull)t2,    BR[4][1][1], MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#5 */
          mop(OP_LDBR,  1, &BR[5][3][1], (Ull)t3,    BR[4][1][1], MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#5 */
          exe(OP_MMRG, &r1, BR[5][1][1], EXP_H3210,  BR[5][2][1], EXP_H3210, BR[5][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#5 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop1[CHIP], pofs,  MSK_D0, (Ull)dtop1[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#5 */
          /*map2*/
          mop(OP_LDWR,  1, &BR[6][1][1], (Ull)rtop2[CHIP], pofs,  MSK_D0, (Ull)rtop2[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#6 */
          mop(OP_LDBR,  1, &BR[7][1][1], (Ull)t1,    BR[6][1][1], MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#7 */
          mop(OP_LDBR,  1, &BR[7][2][1], (Ull)t2,    BR[6][1][1], MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#7 */
          mop(OP_LDBR,  1, &BR[7][3][1], (Ull)t3,    BR[6][1][1], MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#7 */
          exe(OP_MMRG, &r1, BR[7][1][1], EXP_H3210,  BR[7][2][1], EXP_H3210, BR[7][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#7 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop2[CHIP], pofs,  MSK_D0, (Ull)dtop2[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#7 */
          /*map3*/
          mop(OP_LDWR,  1, &BR[8][1][1], (Ull)rtop3[CHIP], pofs,  MSK_D0, (Ull)rtop3[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#8 */
          mop(OP_LDBR,  1, &BR[9][1][1], (Ull)t1,    BR[8][1][1], MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#9 */
          mop(OP_LDBR,  1, &BR[9][2][1], (Ull)t2,    BR[8][1][1], MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#9 */
          mop(OP_LDBR,  1, &BR[9][3][1], (Ull)t3,    BR[8][1][1], MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#9 */
          exe(OP_MMRG, &r1, BR[9][1][1], EXP_H3210,  BR[9][2][1], EXP_H3210, BR[9][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#9 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop3[CHIP], pofs,  MSK_D0, (Ull)dtop3[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#9 */
          /*map4*/
          mop(OP_LDWR,  1, &BR[10][1][1],(Ull)rtop4[CHIP], pofs,  MSK_D0, (Ull)rtop4[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#10 */
          mop(OP_LDBR,  1, &BR[11][1][1],(Ull)t1,    BR[10][1][1],MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#11 */
          mop(OP_LDBR,  1, &BR[11][2][1],(Ull)t2,    BR[10][1][1],MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#11 */
          mop(OP_LDBR,  1, &BR[11][3][1],(Ull)t3,    BR[10][1][1],MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#11 */
          exe(OP_MMRG, &r1, BR[11][1][1],EXP_H3210,  BR[11][2][1],EXP_H3210, BR[11][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#11 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop4[CHIP], pofs,  MSK_D0, (Ull)dtop4[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#11 */
          /*map5*/
          mop(OP_LDWR,  1, &BR[12][1][1],(Ull)rtop5[CHIP], pofs,  MSK_D0, (Ull)rtop5[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#12 */
          mop(OP_LDBR,  1, &BR[13][1][1],(Ull)t1,    BR[12][1][1],MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#13 */
          mop(OP_LDBR,  1, &BR[13][2][1],(Ull)t2,    BR[12][1][1],MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#13 */
          mop(OP_LDBR,  1, &BR[13][3][1],(Ull)t3,    BR[12][1][1],MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#13 */
          exe(OP_MMRG, &r1, BR[13][1][1],EXP_H3210,  BR[13][2][1],EXP_H3210, BR[13][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#13 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop5[CHIP], pofs,  MSK_D0, (Ull)dtop5[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#13 */
          /*map6*/
          mop(OP_LDWR,  1, &BR[14][1][1],(Ull)rtop6[CHIP], pofs,  MSK_D0, (Ull)rtop6[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#14 */
          mop(OP_LDBR,  1, &BR[15][1][1],(Ull)t1,    BR[14][1][1],MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#15 */
          mop(OP_LDBR,  1, &BR[15][2][1],(Ull)t2,    BR[14][1][1],MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#15 */
          mop(OP_LDBR,  1, &BR[15][3][1],(Ull)t3,    BR[14][1][1],MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#15 */
          exe(OP_MMRG, &r1, BR[15][1][1],EXP_H3210,  BR[15][2][1],EXP_H3210, BR[15][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#15 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop6[CHIP], pofs,  MSK_D0, (Ull)dtop6[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#15 */
          /*map7*/
          mop(OP_LDWR,  1, &BR[16][1][1],(Ull)rtop7[CHIP], pofs,  MSK_D0, (Ull)rtop7[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#16 */
          mop(OP_LDBR,  1, &BR[17][1][1],(Ull)t1,    BR[16][1][1],MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#17 */
          mop(OP_LDBR,  1, &BR[17][2][1],(Ull)t2,    BR[16][1][1],MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#17 */
          mop(OP_LDBR,  1, &BR[17][3][1],(Ull)t3,    BR[16][1][1],MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#17 */
          exe(OP_MMRG, &r1, BR[17][1][1],EXP_H3210,  BR[17][2][1],EXP_H3210, BR[17][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#17 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop7[CHIP], pofs,  MSK_D0, (Ull)dtop7[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#17 */
          /*map8*/
          mop(OP_LDWR,  1, &BR[18][1][1],(Ull)rtop8[CHIP], pofs,  MSK_D0, (Ull)rtop8[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#18 */
          mop(OP_LDBR,  1, &BR[19][1][1],(Ull)t1,    BR[18][1][1],MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#19 */
          mop(OP_LDBR,  1, &BR[19][2][1],(Ull)t2,    BR[18][1][1],MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#19 */
          mop(OP_LDBR,  1, &BR[19][3][1],(Ull)t3,    BR[18][1][1],MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#19 */
          exe(OP_MMRG, &r1, BR[19][1][1],EXP_H3210,  BR[19][2][1],EXP_H3210, BR[19][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#19 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop8[CHIP], pofs,  MSK_D0, (Ull)dtop8[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#19 */
          /*map9*/
          mop(OP_LDWR,  1, &BR[20][1][1],(Ull)rtop9[CHIP], pofs,  MSK_D0, (Ull)rtop9[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#20 */
          mop(OP_LDBR,  1, &BR[21][1][1],(Ull)t1,    BR[20][1][1],MSK_B3, (Ull)t1, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#21 */
          mop(OP_LDBR,  1, &BR[21][2][1],(Ull)t2,    BR[20][1][1],MSK_B2, (Ull)t2, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#21 */
          mop(OP_LDBR,  1, &BR[21][3][1],(Ull)t3,    BR[20][1][1],MSK_B1, (Ull)t3, 256/4,  0,  0, (Ull)NULL, 256/4);               /* stage#21 */
          exe(OP_MMRG, &r1, BR[21][1][1],EXP_H3210,  BR[21][2][1],EXP_H3210, BR[21][3][1],EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#21 */
          mop(OP_STWR,  3, &r1,          (Ull)dtop9[CHIP], pofs,  MSK_D0, (Ull)dtop9[CHIP], AWD*RMGRP, 0, 0, (Ull)NULL, AWD*RMGRP);/* stage#21 */
        }
      }
    }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm
#endif
}

void hokan1(Uint *c, Uint *p, struct SAD1 *s) /* W, R, SAD1 */
{
#undef  NCHIP
#undef  RMGRP
#undef  PAD
#undef  RRANGE
#define NCHIP     1
#define RMGRP     8
#define PAD       8
#define RRANGE   ((HT-PAD*2)/NCHIP)

  int i, j, k, l;
  for (i=0; i<PAD; i+=4) { /* scan-lines */
    for (j=0; j<WD; j+=4) {
      for (k=-4; k<4; k++) {
        for (l=-4; l<4; l++) {
          if (k != 0 || l != 0) s->SAD1[i/4][k+4][j/4][l+4] = 0xffff;
          else                  s->SAD1[i/4][k+4][j/4][l+4] = 0;
        }
      }
    }
  }
  for (i=HT-PAD; i<HT; i+=4) { /* scan-lines */
    for (j=0; j<WD; j+=4) {
      for (k=-4; k<4; k++) {
        for (l=-4; l<4; l++) {
          if (k != 0 || l != 0) s->SAD1[i/4][k+4][j/4][l+4] = 0xffff;
          else                  s->SAD1[i/4][k+4][j/4][l+4] = 0;
        }
      }
    }
  }

  /* ¸µ¤ÏPAD=4¤À¤¬,³ä¤êÀÚ¤ì¤ë¤è¤¦¤Ë8¤ËÊÑ¹¹ */
  /* 4x4¤Îwindow¤ËÂÐ¤·,Ushort8x8¤Î¾ðÊó¤ò³ÊÇ¼ */
  Ull  top, rofs, cofs, oc;
  int  pofs;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
#if 0
  for (top=PAD; top<HT-PAD; top++) { /* scan-lines */
    for (pofs=-4; pofs<4; pofs++) {
      Ushort *t = s->SAD1[top/4][pofs+4];
      for (cofs=0; cofs<WD; cofs++) {
        int j = cofs/4*4;
        int k = cofs%4*2;
        Uint *c2 = c+top*WD;
        Uint *p2 = p+(top+pofs)*WD;
        * t    += df(c2[j],p2[j+k-4]) + df(c2[j+1],p2[j+k-3]) + df(c2[j+2],p2[j+k-2]) + df(c2[j+3],p2[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
        *(t+1) += df(c2[j],p2[j+k-3]) + df(c2[j+1],p2[j+k-2]) + df(c2[j+2],p2[j+k-1]) + df(c2[j+3],p2[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
        t += 2;
      }
    }
  }
#else
  for (top=0; top<RRANGE; top+=RMGRP) {  /* will be parallelized by multi-chip (M/#chip) */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      for (CHIP=0; CHIP<NCHIP; CHIP++) {   /* will be parallelized by multi-chip (M/#chip) */
        int idx = CHIP*RRANGE+PAD+top+rofs;
        Uint *c0 = c+ idx   *WD;
        Uint *p0 = p+(idx-4)*WD;                                                                            /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        Uint *p1 = p+(idx-3)*WD;                                                                            /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        Uint *p2 = p+(idx-2)*WD;                                                                            /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        Uint *p3 = p+(idx-1)*WD;                                                                            /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        Uint *p4 = p+(idx+0)*WD;                                                                            /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        Uint *p5 = p+(idx+1)*WD;                                                                            /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        Uint *p6 = p+(idx+2)*WD;                                                                            /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        Uint *p7 = p+(idx+3)*WD;                                                                            /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        Ushort *t0 = s->SAD1[idx/4][0];
        Ushort *t1 = s->SAD1[idx/4][1];
        Ushort *t2 = s->SAD1[idx/4][2];
        Ushort *t3 = s->SAD1[idx/4][3];
        Ushort *t4 = s->SAD1[idx/4][4];
        Ushort *t5 = s->SAD1[idx/4][5];
        Ushort *t6 = s->SAD1[idx/4][6];
        Ushort *t7 = s->SAD1[idx/4][7];
        for (cofs=0; cofs<WD; cofs++) {
          int j = cofs/4*4;
          int k = cofs%4*2;
          * t0    += df(c0[j],p0[j+k-4]) + df(c0[j+1],p0[j+k-3]) + df(c0[j+2],p0[j+k-2]) + df(c0[j+3],p0[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
          *(t0+1) += df(c0[j],p0[j+k-3]) + df(c0[j+1],p0[j+k-2]) + df(c0[j+2],p0[j+k-1]) + df(c0[j+3],p0[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
          t0 += 2;
          * t1    += df(c0[j],p1[j+k-4]) + df(c0[j+1],p1[j+k-3]) + df(c0[j+2],p1[j+k-2]) + df(c0[j+3],p1[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
          *(t1+1) += df(c0[j],p1[j+k-3]) + df(c0[j+1],p1[j+k-2]) + df(c0[j+2],p1[j+k-1]) + df(c0[j+3],p1[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
          t1 += 2;
          * t2    += df(c0[j],p2[j+k-4]) + df(c0[j+1],p2[j+k-3]) + df(c0[j+2],p2[j+k-2]) + df(c0[j+3],p2[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
          *(t2+1) += df(c0[j],p2[j+k-3]) + df(c0[j+1],p2[j+k-2]) + df(c0[j+2],p2[j+k-1]) + df(c0[j+3],p2[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
          t2 += 2;
          * t3    += df(c0[j],p3[j+k-4]) + df(c0[j+1],p3[j+k-3]) + df(c0[j+2],p3[j+k-2]) + df(c0[j+3],p3[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
          *(t3+1) += df(c0[j],p3[j+k-3]) + df(c0[j+1],p3[j+k-2]) + df(c0[j+2],p3[j+k-1]) + df(c0[j+3],p3[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
          t3 += 2;
          * t4    += df(c0[j],p4[j+k-4]) + df(c0[j+1],p4[j+k-3]) + df(c0[j+2],p4[j+k-2]) + df(c0[j+3],p4[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
          *(t4+1) += df(c0[j],p4[j+k-3]) + df(c0[j+1],p4[j+k-2]) + df(c0[j+2],p4[j+k-1]) + df(c0[j+3],p4[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
          t4 += 2;
          * t5    += df(c0[j],p5[j+k-4]) + df(c0[j+1],p5[j+k-3]) + df(c0[j+2],p5[j+k-2]) + df(c0[j+3],p5[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
          *(t5+1) += df(c0[j],p5[j+k-3]) + df(c0[j+1],p5[j+k-2]) + df(c0[j+2],p5[j+k-1]) + df(c0[j+3],p5[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
          t5 += 2;
          * t6    += df(c0[j],p6[j+k-4]) + df(c0[j+1],p6[j+k-3]) + df(c0[j+2],p6[j+k-2]) + df(c0[j+3],p6[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
          *(t6+1) += df(c0[j],p6[j+k-3]) + df(c0[j+1],p6[j+k-2]) + df(c0[j+2],p6[j+k-1]) + df(c0[j+3],p6[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
          t6 += 2;
          * t7    += df(c0[j],p7[j+k-4]) + df(c0[j+1],p7[j+k-3]) + df(c0[j+2],p7[j+k-2]) + df(c0[j+3],p7[j+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
          *(t7+1) += df(c0[j],p7[j+k-3]) + df(c0[j+1],p7[j+k-2]) + df(c0[j+2],p7[j+k-1]) + df(c0[j+3],p7[j+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
          t7 += 2;
        }
      }
    }
  }
#endif
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  for (top=0; top<RRANGE; top+=RMGRP) {  /* will be parallelized by multi-chip (M/#chip) */
    for (rofs=0; rofs<RMGRP; rofs++) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
      Ull    jw, kw;
      Uint   *c0[NCHIP];
      Uint   *p0[NCHIP], *p1[NCHIP], *p2[NCHIP], *p3[NCHIP], *p4[NCHIP], *p5[NCHIP], *p6[NCHIP], *p7[NCHIP];
      Ushort *t0[NCHIP], *t1[NCHIP], *t2[NCHIP], *t3[NCHIP], *t4[NCHIP], *t5[NCHIP], *t6[NCHIP], *t7[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) {
        int idx = CHIP*RRANGE+PAD+top+rofs;
        c0[CHIP] = c+ idx   *WD;
        p0[CHIP] = p+(idx-4)*WD; /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        p1[CHIP] = p+(idx-3)*WD; /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        p2[CHIP] = p+(idx-2)*WD; /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        p3[CHIP] = p+(idx-1)*WD; /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        p4[CHIP] = p+(idx+0)*WD; /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        p5[CHIP] = p+(idx+1)*WD; /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        p6[CHIP] = p+(idx+2)*WD; /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        p7[CHIP] = p+(idx+3)*WD; /* j+k: 0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
        t0[CHIP] = s->SAD1[idx/4][0]; /* SAD1[HT/4][8][WD/4][8] ... [8][WD/4][8]=5120(2B) 0x1400 */
        t1[CHIP] = s->SAD1[idx/4][1];
        t2[CHIP] = s->SAD1[idx/4][2];
        t3[CHIP] = s->SAD1[idx/4][3];
        t4[CHIP] = s->SAD1[idx/4][4];
        t5[CHIP] = s->SAD1[idx/4][5];
        t6[CHIP] = s->SAD1[idx/4][6];
        t7[CHIP] = s->SAD1[idx/4][7];
      }
//EMAX5A begin hokan1 mapdist=7
 /*2*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*1*/for (INIT0=1,LOOP0=AWD,cofs=0-4; LOOP0--; INIT0=0) {       /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
 /*@0,1*/ exe(OP_ADD,  &cofs, INIT0?cofs:cofs, EXP_H3210, 4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
          /*int j = cofs/4 /4*4 *4;*/
          /*int k = cofs/4 %4*2 *4;*/
 /*@1,0*/ exe(OP_NOP,     &jw,          cofs,     EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND,~15LL, OP_SLL, 0LL);
 /*@1,1*/ exe(OP_NOP,     &kw,          cofs,     EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, 12LL, OP_SLL, 1LL);
          /*k=-4*/
 /*@2,0*/ exe(OP_ADD,     &r12,         c0[CHIP], EXP_H3210, jw,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@2,1*/ exe(OP_ADD3,    &r13,         p0[CHIP], EXP_H3210, jw,  EXP_H3210, kw,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@3,0*/ mop(OP_LDWR, 1, &r0,          r12,    0LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@3,1*/ mop(OP_LDWR, 1, &r1,          r12,    4LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@3,2*/ mop(OP_LDWR, 1, &r2,          r12,    8LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@3,3*/ mop(OP_LDWR, 1, &r3,          r12,   12LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@4,0*/ mop(OP_LDWR, 1, &BR[4][0][1], r13,  -16LL,  MSK_D0,  (Ull)p0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@4,1*/ mop(OP_LDWR, 1, &r25,         r13,  -12LL,  MSK_D0,  (Ull)p0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@4,2*/ mop(OP_LDWR, 1, &r26,         r13,   -8LL,  MSK_D0,  (Ull)p0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@4,3*/ mop(OP_LDWR, 1, &r27,         r13,   -4LL,  MSK_D0,  (Ull)p0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@4,3*/ mop(OP_LDWR, 1, &r28,         r13,    0LL,  MSK_D0,  (Ull)p0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@5,0*/ exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@5,1*/ exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@5,2*/ exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@5,3*/ exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@6,0*/ exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[4][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@6,1*/ exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@6,2*/ exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@6,3*/ exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@7,0*/ exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@7,1*/ exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@7,2*/ exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@7,3*/ exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@8,0*/ exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
 /*@8,1*/ exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
 /*@9,0*/ mop(OP_LDWR, 1, &BR[9][0][1], t0[CHIP],  cofs, MSK_D0, (Ull)t0[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
 /*@9,0*/ exe(OP_MAUH3,   &AR[9][0],    BR[9][0][1],     EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
 /*@9,0*/ mop(OP_STWR, 3, &AR[9][0],    cofs,  t0[CHIP], MSK_D0, (Ull)t0[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
          /*k=-3*/
 /*@9,2*/ exe(OP_ADD,     &r12,         c0[CHIP], EXP_H3210, jw,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@9,3*/ exe(OP_ADD3,    &r13,         p1[CHIP], EXP_H3210, jw,  EXP_H3210, kw,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@10,0*/mop(OP_LDWR, 1, &r0,          r12,    0LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@10,1*/mop(OP_LDWR, 1, &r1,          r12,    4LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@10,2*/mop(OP_LDWR, 1, &r2,          r12,    8LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@10,3*/mop(OP_LDWR, 1, &r3,          r12,   12LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@11,0*/mop(OP_LDWR, 1, &BR[11][0][1],r13,  -16LL,  MSK_D0,  (Ull)p1[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@11,1*/mop(OP_LDWR, 1, &r25,         r13,  -12LL,  MSK_D0,  (Ull)p1[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@11,2*/mop(OP_LDWR, 1, &r26,         r13,   -8LL,  MSK_D0,  (Ull)p1[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@11,3*/mop(OP_LDWR, 1, &r27,         r13,   -4LL,  MSK_D0,  (Ull)p1[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@11,3*/mop(OP_LDWR, 1, &r28,         r13,    0LL,  MSK_D0,  (Ull)p1[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@12,0*/exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@12,1*/exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@12,2*/exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@12,3*/exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@13,0*/exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[11][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@13,1*/exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@13,2*/exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@13,3*/exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@14,0*/exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@14,1*/exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@14,2*/exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@14,3*/exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@15,0*/exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
 /*@15,1*/exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
 /*@16,0*/mop(OP_LDWR, 1, &BR[16][0][1],t1[CHIP],  cofs, MSK_D0, (Ull)t1[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
 /*@16,0*/exe(OP_MAUH3,   &AR[16][0],   BR[16][0][1],    EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
 /*@16,0*/mop(OP_STWR, 3, &AR[16][0],   cofs,  t1[CHIP], MSK_D0, (Ull)t1[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
          /*k=-2*/
 /*@16,2*/exe(OP_ADD,     &r12,         c0[CHIP], EXP_H3210, jw,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@16,3*/exe(OP_ADD3,    &r13,         p2[CHIP], EXP_H3210, jw,  EXP_H3210, kw,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@17,0*/mop(OP_LDWR, 1, &r0,          r12,    0LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@17,1*/mop(OP_LDWR, 1, &r1,          r12,    4LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@17,2*/mop(OP_LDWR, 1, &r2,          r12,    8LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@17,3*/mop(OP_LDWR, 1, &r3,          r12,   12LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@18,0*/mop(OP_LDWR, 1, &BR[18][0][1],r13,  -16LL,  MSK_D0,  (Ull)p2[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@18,1*/mop(OP_LDWR, 1, &r25,         r13,  -12LL,  MSK_D0,  (Ull)p2[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@18,2*/mop(OP_LDWR, 1, &r26,         r13,   -8LL,  MSK_D0,  (Ull)p2[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@18,3*/mop(OP_LDWR, 1, &r27,         r13,   -4LL,  MSK_D0,  (Ull)p2[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@18,3*/mop(OP_LDWR, 1, &r28,         r13,    0LL,  MSK_D0,  (Ull)p2[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@19,0*/exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@19,1*/exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@19,2*/exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@19,3*/exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@20,0*/exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[18][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@20,1*/exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@20,2*/exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@20,3*/exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@21,0*/exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@21,1*/exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@21,2*/exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@21,3*/exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@22,0*/exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
 /*@22,1*/exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
 /*@23,0*/mop(OP_LDWR, 1, &BR[23][0][1],t2[CHIP],  cofs, MSK_D0, (Ull)t2[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
 /*@23,0*/exe(OP_MAUH3,   &AR[23][0],   BR[23][0][1],    EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
 /*@23,0*/mop(OP_STWR, 3, &AR[23][0],   cofs,  t2[CHIP], MSK_D0, (Ull)t2[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
          /*k=-1*/
 /*@23,2*/exe(OP_ADD,     &r12,         c0[CHIP], EXP_H3210, jw,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@23,3*/exe(OP_ADD3,    &r13,         p3[CHIP], EXP_H3210, jw,  EXP_H3210, kw,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@24,0*/mop(OP_LDWR, 1, &r0,          r12,    0LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@24,1*/mop(OP_LDWR, 1, &r1,          r12,    4LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@24,2*/mop(OP_LDWR, 1, &r2,          r12,    8LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@24,3*/mop(OP_LDWR, 1, &r3,          r12,   12LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@25,0*/mop(OP_LDWR, 1, &BR[25][0][1],r13,  -16LL,  MSK_D0,  (Ull)p3[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@25,1*/mop(OP_LDWR, 1, &r25,         r13,  -12LL,  MSK_D0,  (Ull)p3[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@25,2*/mop(OP_LDWR, 1, &r26,         r13,   -8LL,  MSK_D0,  (Ull)p3[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@25,3*/mop(OP_LDWR, 1, &r27,         r13,   -4LL,  MSK_D0,  (Ull)p3[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@25,3*/mop(OP_LDWR, 1, &r28,         r13,    0LL,  MSK_D0,  (Ull)p3[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@26,0*/exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@26,1*/exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@26,2*/exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@26,3*/exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@27,0*/exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[25][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@27,1*/exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@27,2*/exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@27,3*/exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@28,0*/exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@28,1*/exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@28,2*/exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@28,3*/exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@29,0*/exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
 /*@29,1*/exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
 /*@30,0*/mop(OP_LDWR, 1, &BR[30][0][1],t3[CHIP],  cofs, MSK_D0, (Ull)t3[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
 /*@30,0*/exe(OP_MAUH3,   &AR[30][0],   BR[30][0][1],    EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
 /*@30,0*/mop(OP_STWR, 3, &AR[30][0],   cofs,  t3[CHIP], MSK_D0, (Ull)t3[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
          /*k= 0*/
 /*@30,2*/exe(OP_ADD,     &r12,         c0[CHIP], EXP_H3210, jw,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@30,3*/exe(OP_ADD3,    &r13,         p4[CHIP], EXP_H3210, jw,  EXP_H3210, kw,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@31,0*/mop(OP_LDWR, 1, &r0,          r12,    0LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@31,1*/mop(OP_LDWR, 1, &r1,          r12,    4LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@31,2*/mop(OP_LDWR, 1, &r2,          r12,    8LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@31,3*/mop(OP_LDWR, 1, &r3,          r12,   12LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@32,0*/mop(OP_LDWR, 1, &BR[32][0][1],r13,  -16LL,  MSK_D0,  (Ull)p4[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@32,1*/mop(OP_LDWR, 1, &r25,         r13,  -12LL,  MSK_D0,  (Ull)p4[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@32,2*/mop(OP_LDWR, 1, &r26,         r13,   -8LL,  MSK_D0,  (Ull)p4[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@32,3*/mop(OP_LDWR, 1, &r27,         r13,   -4LL,  MSK_D0,  (Ull)p4[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@32,3*/mop(OP_LDWR, 1, &r28,         r13,    0LL,  MSK_D0,  (Ull)p4[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@33,0*/exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@33,1*/exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@33,2*/exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@33,3*/exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@34,0*/exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[32][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@34,1*/exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@34,2*/exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@34,3*/exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@35,0*/exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@35,1*/exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@35,2*/exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@35,3*/exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@36,0*/exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
 /*@36,1*/exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
 /*@37,0*/mop(OP_LDWR, 1, &BR[37][0][1],t4[CHIP],  cofs, MSK_D0, (Ull)t4[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
 /*@37,0*/exe(OP_MAUH3,   &AR[37][0],   BR[37][0][1],    EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
 /*@37,0*/mop(OP_STWR, 3, &AR[37][0],   cofs,  t4[CHIP], MSK_D0, (Ull)t4[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
          /*k=+1*/
 /*@37,2*/exe(OP_ADD,     &r12,         c0[CHIP], EXP_H3210, jw,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@37,3*/exe(OP_ADD3,    &r13,         p5[CHIP], EXP_H3210, jw,  EXP_H3210, kw,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@38,0*/mop(OP_LDWR, 1, &r0,          r12,    0LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@38,1*/mop(OP_LDWR, 1, &r1,          r12,    4LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@38,2*/mop(OP_LDWR, 1, &r2,          r12,    8LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@38,3*/mop(OP_LDWR, 1, &r3,          r12,   12LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@39,0*/mop(OP_LDWR, 1, &BR[39][0][1],r13,  -16LL,  MSK_D0,  (Ull)p5[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@39,1*/mop(OP_LDWR, 1, &r25,         r13,  -12LL,  MSK_D0,  (Ull)p5[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@39,2*/mop(OP_LDWR, 1, &r26,         r13,   -8LL,  MSK_D0,  (Ull)p5[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@39,3*/mop(OP_LDWR, 1, &r27,         r13,   -4LL,  MSK_D0,  (Ull)p5[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@39,3*/mop(OP_LDWR, 1, &r28,         r13,    0LL,  MSK_D0,  (Ull)p5[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@40,0*/exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@40,1*/exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@40,2*/exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@40,3*/exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@41,0*/exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[39][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@41,1*/exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@41,2*/exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@41,3*/exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@42,0*/exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@42,1*/exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@42,2*/exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@42,3*/exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@43,0*/exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
 /*@43,1*/exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
 /*@44,0*/mop(OP_LDWR, 1, &BR[44][0][1],t5[CHIP],  cofs, MSK_D0, (Ull)t5[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
 /*@44,0*/exe(OP_MAUH3,   &AR[44][0],   BR[44][0][1],    EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
 /*@44,0*/mop(OP_STWR, 3, &AR[44][0],   cofs,  t5[CHIP], MSK_D0, (Ull)t5[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
          /*k=+2*/
 /*@44,2*/exe(OP_ADD,     &r12,         c0[CHIP], EXP_H3210, jw,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@44,3*/exe(OP_ADD3,    &r13,         p6[CHIP], EXP_H3210, jw,  EXP_H3210, kw,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@45,0*/mop(OP_LDWR, 1, &r0,          r12,    0LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@45,1*/mop(OP_LDWR, 1, &r1,          r12,    4LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@45,2*/mop(OP_LDWR, 1, &r2,          r12,    8LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@45,3*/mop(OP_LDWR, 1, &r3,          r12,   12LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@46,0*/mop(OP_LDWR, 1, &BR[46][0][1],r13,  -16LL,  MSK_D0,  (Ull)p6[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@46,1*/mop(OP_LDWR, 1, &r25,         r13,  -12LL,  MSK_D0,  (Ull)p6[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@46,2*/mop(OP_LDWR, 1, &r26,         r13,   -8LL,  MSK_D0,  (Ull)p6[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@46,3*/mop(OP_LDWR, 1, &r27,         r13,   -4LL,  MSK_D0,  (Ull)p6[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@46,3*/mop(OP_LDWR, 1, &r28,         r13,    0LL,  MSK_D0,  (Ull)p6[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@47,0*/exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@47,1*/exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@47,2*/exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@47,3*/exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@48,0*/exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[46][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@48,1*/exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@48,2*/exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@48,3*/exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@49,0*/exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@49,1*/exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@49,2*/exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@49,3*/exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@50,0*/exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
 /*@50,1*/exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
 /*@51,0*/mop(OP_LDWR, 1, &BR[51][0][1],t6[CHIP],  cofs, MSK_D0, (Ull)t6[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
 /*@51,0*/exe(OP_MAUH3,   &AR[51][0],   BR[51][0][1],    EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
 /*@51,0*/mop(OP_STWR, 3, &AR[51][0],   cofs,  t6[CHIP], MSK_D0, (Ull)t6[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
          /*k=+3*/
 /*@51,2*/exe(OP_ADD,     &r12,         c0[CHIP], EXP_H3210, jw,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@51,3*/exe(OP_ADD3,    &r13,         p7[CHIP], EXP_H3210, jw,  EXP_H3210, kw,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@52,0*/mop(OP_LDWR, 1, &r0,          r12,    0LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@52,1*/mop(OP_LDWR, 1, &r1,          r12,    4LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@52,2*/mop(OP_LDWR, 1, &r2,          r12,    8LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@52,3*/mop(OP_LDWR, 1, &r3,          r12,   12LL,  MSK_D0,  (Ull)c0[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@53,0*/mop(OP_LDWR, 1, &BR[53][0][1],r13,  -16LL,  MSK_D0,  (Ull)p7[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@53,1*/mop(OP_LDWR, 1, &r25,         r13,  -12LL,  MSK_D0,  (Ull)p7[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@53,2*/mop(OP_LDWR, 1, &r26,         r13,   -8LL,  MSK_D0,  (Ull)p7[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@53,3*/mop(OP_LDWR, 1, &r27,         r13,   -4LL,  MSK_D0,  (Ull)p7[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@53,3*/mop(OP_LDWR, 1, &r28,         r13,    0LL,  MSK_D0,  (Ull)p7[CHIP], AWD, 0, 0, (Ull)NULL, AWD);
 /*@54,0*/exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@54,1*/exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@54,2*/exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@54,3*/exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@55,0*/exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[53][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@55,1*/exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@55,2*/exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@55,3*/exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@56,0*/exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@56,1*/exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@56,2*/exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@56,3*/exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@57,0*/exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
 /*@57,1*/exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
 /*@58,0*/mop(OP_LDWR, 1, &BR[58][0][1],t7[CHIP],  cofs, MSK_D0, (Ull)t7[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
 /*@58,0*/exe(OP_MAUH3,   &AR[58][0],   BR[58][0][1],    EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
 /*@58,0*/mop(OP_STWR, 3, &AR[58][0],   cofs,  t7[CHIP], MSK_D0, (Ull)t7[CHIP], AWD, 0, 1, (Ull)NULL, AWD);
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
#endif
}

void hokan2(struct SAD1 *s, Uint *minxy) /* [WD/4][8] */
{
#undef  NCHIP
#undef  RMGRP
#undef  PAD
#undef  RRANGE
#define NCHIP     1
#define RMGRP    12
#define PAD       0
#define RRANGE   ((HT-PAD*2)/NCHIP)
#define HM        0xffff

  int i, j;
  for (i=0; i<HT; i+=4) { /* scan-lines */
    for (j=0; j<WD; j++)
      minxy[i*WD+j] = HM; /* x,y,sadmin */
  }

  Ull  top, rofs, cofs, oc;
  int  pofs;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
#if 0
  for (top=PAD; top<HT-PAD; top+=4) { /* scan-lines */
    Uint *xy = minxy+top*WD;
    for (pofs=-4; pofs<4; pofs++) {
      Ushort *t  = s->SAD1[top/4][pofs+4];
      int    idx = ((pofs/2)&0xff)<<16;
      for (cofs=0; cofs<WD; cofs++) { /* j%4==0¤Î»þ¤Î¤ßminxy[j]¤ËÍ­¸úÃÍ¡¥Â¾¤Ï¥´¥ß */
        int l1 = ((-2)<<24)|idx|*(t  ); if ((xy[cofs]&HM) > *(t  )) xy[cofs] = l1;
        int l2 = ((-1)<<24)|idx|*(t+1); if ((xy[cofs]&HM) > *(t+1)) xy[cofs] = l2;
        int l3 = ((-1)<<24)|idx|*(t+2); if ((xy[cofs]&HM) > *(t+2)) xy[cofs] = l3;
        int l4 = (( 0)<<24)|idx|*(t+3); if ((xy[cofs]&HM) > *(t+3)) xy[cofs] = l4;
        int l5 = (( 0)<<24)|idx|*(t+4); if ((xy[cofs]&HM) > *(t+4)) xy[cofs] = l5;
        int l6 = (( 0)<<24)|idx|*(t+5); if ((xy[cofs]&HM) > *(t+5)) xy[cofs] = l6;
        int l7 = (( 1)<<24)|idx|*(t+6); if ((xy[cofs]&HM) > *(t+6)) xy[cofs] = l7;
        int l8 = (( 1)<<24)|idx|*(t+7); if ((xy[cofs]&HM) > *(t+7)) xy[cofs] = l8;
        t += 2;
      }
    }
  }
#else
  Uint ix0 = ((-4/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix1 = ((-3/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix2 = ((-2/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix3 = ((-1/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix4 = (( 0/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix5 = ((+1/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix6 = ((+2/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix7 = ((+3/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  for (top=0; top<RRANGE; top+=RMGRP) {  /* will be parallelized by multi-chip (M/#chip) */
    for (rofs=0; rofs<RMGRP; rofs+=4) { /* will be parallelized by multi-chip (M/#chip) */
      for (CHIP=0; CHIP<NCHIP; CHIP++) {   /* will be parallelized by multi-chip (M/#chip) */
        Ushort *t0 = s->SAD1[(CHIP*RRANGE+top+rofs)/4][0];
        Ushort *t1 = s->SAD1[(CHIP*RRANGE+top+rofs)/4][1];
        Ushort *t2 = s->SAD1[(CHIP*RRANGE+top+rofs)/4][2];
        Ushort *t3 = s->SAD1[(CHIP*RRANGE+top+rofs)/4][3];
        Ushort *t4 = s->SAD1[(CHIP*RRANGE+top+rofs)/4][4];
        Ushort *t5 = s->SAD1[(CHIP*RRANGE+top+rofs)/4][5];
        Ushort *t6 = s->SAD1[(CHIP*RRANGE+top+rofs)/4][6];
        Ushort *t7 = s->SAD1[(CHIP*RRANGE+top+rofs)/4][7];
        Uint   *xy = minxy+(CHIP*RRANGE+top+rofs)*WD;
        for (cofs=0; cofs<WD; cofs++) { /* j%4==0¤Î»þ¤Î¤ßminxy[j]¤ËÍ­¸úÃÍ¡¥Â¾¤Ï¥´¥ß */
          int l1, l2, l3, l4, l5, l6, l7, l8;
          l1=((-2)<<24)|ix0|*(t0  ); if((xy[cofs]&HM)>*(t0  ))xy[cofs]=l1;
          l2=((-1)<<24)|ix0|*(t0+1); if((xy[cofs]&HM)>*(t0+1))xy[cofs]=l2;
          l3=((-1)<<24)|ix0|*(t0+2); if((xy[cofs]&HM)>*(t0+2))xy[cofs]=l3;
          l4=           ix0|*(t0+3); if((xy[cofs]&HM)>*(t0+3))xy[cofs]=l4;
          l5=           ix0|*(t0+4); if((xy[cofs]&HM)>*(t0+4))xy[cofs]=l5;
          l6=           ix0|*(t0+5); if((xy[cofs]&HM)>*(t0+5))xy[cofs]=l6;
          l7=(( 1)<<24)|ix0|*(t0+6); if((xy[cofs]&HM)>*(t0+6))xy[cofs]=l7;
          l8=(( 1)<<24)|ix0|*(t0+7); if((xy[cofs]&HM)>*(t0+7))xy[cofs]=l8;
          t0 += 2;
          l1=((-2)<<24)|ix1|*(t1  ); if((xy[cofs]&HM)>*(t1  ))xy[cofs]=l1;
          l2=((-1)<<24)|ix1|*(t1+1); if((xy[cofs]&HM)>*(t1+1))xy[cofs]=l2;
          l3=((-1)<<24)|ix1|*(t1+2); if((xy[cofs]&HM)>*(t1+2))xy[cofs]=l3;
          l4=           ix1|*(t1+3); if((xy[cofs]&HM)>*(t1+3))xy[cofs]=l4;
          l5=           ix1|*(t1+4); if((xy[cofs]&HM)>*(t1+4))xy[cofs]=l5;
          l6=           ix1|*(t1+5); if((xy[cofs]&HM)>*(t1+5))xy[cofs]=l6;
          l7=(( 1)<<24)|ix1|*(t1+6); if((xy[cofs]&HM)>*(t1+6))xy[cofs]=l7;
          l8=(( 1)<<24)|ix1|*(t1+7); if((xy[cofs]&HM)>*(t1+7))xy[cofs]=l8;
          t1 += 2;
          l1=((-2)<<24)|ix2|*(t2  ); if((xy[cofs]&HM)>*(t2  ))xy[cofs]=l1;
          l2=((-1)<<24)|ix2|*(t2+1); if((xy[cofs]&HM)>*(t2+1))xy[cofs]=l2;
          l3=((-1)<<24)|ix2|*(t2+2); if((xy[cofs]&HM)>*(t2+2))xy[cofs]=l3;
          l4=           ix2|*(t2+3); if((xy[cofs]&HM)>*(t2+3))xy[cofs]=l4;
          l5=           ix2|*(t2+4); if((xy[cofs]&HM)>*(t2+4))xy[cofs]=l5;
          l6=           ix2|*(t2+5); if((xy[cofs]&HM)>*(t2+5))xy[cofs]=l6;
          l7=(( 1)<<24)|ix2|*(t2+6); if((xy[cofs]&HM)>*(t2+6))xy[cofs]=l7;
          l8=(( 1)<<24)|ix2|*(t2+7); if((xy[cofs]&HM)>*(t2+7))xy[cofs]=l8;
          t2 += 2;
          l1=((-2)<<24)|ix3|*(t3  ); if((xy[cofs]&HM)>*(t3  ))xy[cofs]=l1;
          l2=((-1)<<24)|ix3|*(t3+1); if((xy[cofs]&HM)>*(t3+1))xy[cofs]=l2;
          l3=((-1)<<24)|ix3|*(t3+2); if((xy[cofs]&HM)>*(t3+2))xy[cofs]=l3;
          l4=           ix3|*(t3+3); if((xy[cofs]&HM)>*(t3+3))xy[cofs]=l4;
          l5=           ix3|*(t3+4); if((xy[cofs]&HM)>*(t3+4))xy[cofs]=l5;
          l6=           ix3|*(t3+5); if((xy[cofs]&HM)>*(t3+5))xy[cofs]=l6;
          l7=(( 1)<<24)|ix3|*(t3+6); if((xy[cofs]&HM)>*(t3+6))xy[cofs]=l7;
          l8=(( 1)<<24)|ix3|*(t3+7); if((xy[cofs]&HM)>*(t3+7))xy[cofs]=l8;
          t3 += 2;
          l1=((-2)<<24)|ix4|*(t4  ); if((xy[cofs]&HM)>*(t4  ))xy[cofs]=l1;
          l2=((-1)<<24)|ix4|*(t4+1); if((xy[cofs]&HM)>*(t4+1))xy[cofs]=l2;
          l3=((-1)<<24)|ix4|*(t4+2); if((xy[cofs]&HM)>*(t4+2))xy[cofs]=l3;
          l4=           ix4|*(t4+3); if((xy[cofs]&HM)>*(t4+3))xy[cofs]=l4;
          l5=           ix4|*(t4+4); if((xy[cofs]&HM)>*(t4+4))xy[cofs]=l5;
          l6=           ix4|*(t4+5); if((xy[cofs]&HM)>*(t4+5))xy[cofs]=l6;
          l7=(( 1)<<24)|ix4|*(t4+6); if((xy[cofs]&HM)>*(t4+6))xy[cofs]=l7;
          l8=(( 1)<<24)|ix4|*(t4+7); if((xy[cofs]&HM)>*(t4+7))xy[cofs]=l8;
          t4 += 2;
          l1=((-2)<<24)|ix5|*(t5  ); if((xy[cofs]&HM)>*(t5  ))xy[cofs]=l1;
          l2=((-1)<<24)|ix5|*(t5+1); if((xy[cofs]&HM)>*(t5+1))xy[cofs]=l2;
          l3=((-1)<<24)|ix5|*(t5+2); if((xy[cofs]&HM)>*(t5+2))xy[cofs]=l3;
          l4=           ix5|*(t5+3); if((xy[cofs]&HM)>*(t5+3))xy[cofs]=l4;
          l5=           ix5|*(t5+4); if((xy[cofs]&HM)>*(t5+4))xy[cofs]=l5;
          l6=           ix5|*(t5+5); if((xy[cofs]&HM)>*(t5+5))xy[cofs]=l6;
          l7=(( 1)<<24)|ix5|*(t5+6); if((xy[cofs]&HM)>*(t5+6))xy[cofs]=l7;
          l8=(( 1)<<24)|ix5|*(t5+7); if((xy[cofs]&HM)>*(t5+7))xy[cofs]=l8;
          t5 += 2;
          l1=((-2)<<24)|ix6|*(t6  ); if((xy[cofs]&HM)>*(t6  ))xy[cofs]=l1;
          l2=((-1)<<24)|ix6|*(t6+1); if((xy[cofs]&HM)>*(t6+1))xy[cofs]=l2;
          l3=((-1)<<24)|ix6|*(t6+2); if((xy[cofs]&HM)>*(t6+2))xy[cofs]=l3;
          l4=           ix6|*(t6+3); if((xy[cofs]&HM)>*(t6+3))xy[cofs]=l4;
          l5=           ix6|*(t6+4); if((xy[cofs]&HM)>*(t6+4))xy[cofs]=l5;
          l6=           ix6|*(t6+5); if((xy[cofs]&HM)>*(t6+5))xy[cofs]=l6;
          l7=(( 1)<<24)|ix6|*(t6+6); if((xy[cofs]&HM)>*(t6+6))xy[cofs]=l7;
          l8=(( 1)<<24)|ix6|*(t6+7); if((xy[cofs]&HM)>*(t6+7))xy[cofs]=l8;
          t6 += 2;
          l1=((-2)<<24)|ix7|*(t7  ); if((xy[cofs]&HM)>*(t7  ))xy[cofs]=l1;
          l2=((-1)<<24)|ix7|*(t7+1); if((xy[cofs]&HM)>*(t7+1))xy[cofs]=l2;
          l3=((-1)<<24)|ix7|*(t7+2); if((xy[cofs]&HM)>*(t7+2))xy[cofs]=l3;
          l4=           ix7|*(t7+3); if((xy[cofs]&HM)>*(t7+3))xy[cofs]=l4;
          l5=           ix7|*(t7+4); if((xy[cofs]&HM)>*(t7+4))xy[cofs]=l5;
          l6=           ix7|*(t7+5); if((xy[cofs]&HM)>*(t7+5))xy[cofs]=l6;
          l7=(( 1)<<24)|ix7|*(t7+6); if((xy[cofs]&HM)>*(t7+6))xy[cofs]=l7;
          l8=(( 1)<<24)|ix7|*(t7+7); if((xy[cofs]&HM)>*(t7+7))xy[cofs]=l8;
          t7 += 2;
        }
      }
    }
  }
#endif
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Uint ix0 = ((-4/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix1 = ((-3/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix2 = ((-2/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix3 = ((-1/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix4 = (( 0/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix5 = ((+1/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix6 = ((+2/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  Uint ix7 = ((+3/2)&0xff)<<16; /* -2,-1,-1,0,0,0,1,1 */
  for (top=0; top<RRANGE; top+=RMGRP) {  /* will be parallelized by multi-chip (M/#chip) */
    for (rofs=0; rofs<RMGRP; rofs+=4) { /* will be parallelized by multi-chip (M/#chip) */
      Uint *xy[NCHIP];
      Uint *t00[NCHIP],*t10[NCHIP],*t20[NCHIP],*t30[NCHIP],*t40[NCHIP],*t50[NCHIP],*t60[NCHIP],*t70[NCHIP];
      Uint *t01[NCHIP],*t11[NCHIP],*t21[NCHIP],*t31[NCHIP],*t41[NCHIP],*t51[NCHIP],*t61[NCHIP],*t71[NCHIP];
      Uint *t02[NCHIP],*t12[NCHIP],*t22[NCHIP],*t32[NCHIP],*t42[NCHIP],*t52[NCHIP],*t62[NCHIP],*t72[NCHIP];
      Uint *t03[NCHIP],*t13[NCHIP],*t23[NCHIP],*t33[NCHIP],*t43[NCHIP],*t53[NCHIP],*t63[NCHIP],*t73[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) {   /* will be parallelized by multi-chip (M/#chip) */
        int idx = CHIP*RRANGE+PAD+top+rofs;
        t00[CHIP] = s->SAD1[idx/4][0]; t01[CHIP] = t00[CHIP]+1; t02[CHIP] = t00[CHIP]+2; t03[CHIP] = t00[CHIP]+3;
        t10[CHIP] = s->SAD1[idx/4][1]; t11[CHIP] = t10[CHIP]+1; t12[CHIP] = t10[CHIP]+2; t13[CHIP] = t10[CHIP]+3;
        t20[CHIP] = s->SAD1[idx/4][2]; t21[CHIP] = t20[CHIP]+1; t22[CHIP] = t20[CHIP]+2; t23[CHIP] = t20[CHIP]+3;
        t30[CHIP] = s->SAD1[idx/4][3]; t31[CHIP] = t30[CHIP]+1; t32[CHIP] = t30[CHIP]+2; t33[CHIP] = t30[CHIP]+3;
        t40[CHIP] = s->SAD1[idx/4][4]; t41[CHIP] = t40[CHIP]+1; t42[CHIP] = t40[CHIP]+2; t43[CHIP] = t40[CHIP]+3;
        t50[CHIP] = s->SAD1[idx/4][5]; t51[CHIP] = t50[CHIP]+1; t52[CHIP] = t50[CHIP]+2; t53[CHIP] = t50[CHIP]+3;
        t60[CHIP] = s->SAD1[idx/4][6]; t61[CHIP] = t60[CHIP]+1; t62[CHIP] = t60[CHIP]+2; t63[CHIP] = t60[CHIP]+3;
        t70[CHIP] = s->SAD1[idx/4][7]; t71[CHIP] = t70[CHIP]+1; t72[CHIP] = t70[CHIP]+2; t73[CHIP] = t70[CHIP]+3;
        xy[CHIP] = minxy+idx*WD;
      }
//EMAX5A begin hokan2 mapdist=0
 /*2*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*1*/for (INIT0=1,LOOP0=AWD,cofs=0-4; LOOP0--; INIT0=0) {       /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
 /*@0,1*/ exe(OP_ADD,     &cofs,  INIT0?cofs:cofs, EXP_H3210, 4,           EXP_H3210, 0LL,  EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
          /*k=-4*/
 /*@1,0*/ mop(OP_LDWR, 1, &r10,         t00[CHIP], cofs,      MSK_D0,      t00[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@1,0*/ exe(OP_NOP,     &r28,        (-2LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix0, OP_NOP, 0LL);
 /*@1,0*/ mop(OP_LDWR, 1, &r12,         t01[CHIP], cofs,      MSK_D0,      t00[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@1,1*/ exe(OP_NOP,     &r29,        (-1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix0, OP_NOP, 0LL);
 /*@1,1*/ mop(OP_LDWR, 1, &r14,         t02[CHIP], cofs,      MSK_D0,      t00[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@1,2*/ exe(OP_NOP,     &r31,        ( 1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix0, OP_NOP, 0LL);
 /*@1,2*/ mop(OP_LDWR, 1, &r16,         t03[CHIP], cofs,      MSK_D0,      t00[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@2,0*/ exe(OP_MINL3,   &r10,         r29,       EXP_H3210, r28,         EXP_H3210, r10,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@2,1*/ exe(OP_MINL3,   &r12,         ix0,       EXP_H3210, r29,         EXP_H3210, r12,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@2,2*/ exe(OP_MINL3,   &r14,         ix0,       EXP_H3210, ix0,         EXP_H3210, r14,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@2,3*/ exe(OP_MINL3,   &r16,         r31,       EXP_H3210, r31,         EXP_H3210, r16,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@3,0*/ exe(OP_MINL,    &r20,         r10,       EXP_H3210, r12,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@3,1*/ exe(OP_MINL,    &r24,         r14,       EXP_H3210, r16,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@4,0*/ exe(OP_MINL,    &r0,          r20,       EXP_H3210, r24,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
          /*k=-3*/
 /*@4,0*/ mop(OP_LDWR, 1, &r10,         t10[CHIP], cofs,      MSK_D0,      t10[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@4,1*/ exe(OP_NOP,     &r28,        (-2LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix1, OP_NOP, 0LL);
 /*@4,1*/ mop(OP_LDWR, 1, &r12,         t11[CHIP], cofs,      MSK_D0,      t10[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@4,2*/ exe(OP_NOP,     &r29,        (-1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix1, OP_NOP, 0LL);
 /*@4,2*/ mop(OP_LDWR, 1, &r14,         t12[CHIP], cofs,      MSK_D0,      t10[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@4,3*/ exe(OP_NOP,     &r31,        ( 1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix1, OP_NOP, 0LL);
 /*@4,3*/ mop(OP_LDWR, 1, &r16,         t13[CHIP], cofs,      MSK_D0,      t10[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@5,0*/ exe(OP_MINL3,   &r10,         r29,       EXP_H3210, r28,         EXP_H3210, r10,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@5,1*/ exe(OP_MINL3,   &r12,         ix1,       EXP_H3210, r29,         EXP_H3210, r12,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@5,2*/ exe(OP_MINL3,   &r14,         ix1,       EXP_H3210, ix1,         EXP_H3210, r14,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@5,3*/ exe(OP_MINL3,   &r16,         r31,       EXP_H3210, r31,         EXP_H3210, r16,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@6,0*/ exe(OP_MINL,    &r20,         r10,       EXP_H3210, r12,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@6,1*/ exe(OP_MINL,    &r24,         r14,       EXP_H3210, r16,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@7,0*/ exe(OP_MINL,    &r1,          r20,       EXP_H3210, r24,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@8,0*/ exe(OP_MINL,    &r0,          r1,        EXP_H3210, r0,          EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
          /*k=-2*/
 /*@8,0*/ mop(OP_LDWR, 1, &r10,         t20[CHIP], cofs,      MSK_D0,      t20[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@8,1*/ exe(OP_NOP,     &r28,        (-2LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix2, OP_NOP, 0LL);
 /*@8,1*/ mop(OP_LDWR, 1, &r12,         t21[CHIP], cofs,      MSK_D0,      t20[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@8,2*/ exe(OP_NOP,     &r29,        (-1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix2, OP_NOP, 0LL);
 /*@8,2*/ mop(OP_LDWR, 1, &r14,         t22[CHIP], cofs,      MSK_D0,      t20[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@8,3*/ exe(OP_NOP,     &r31,        ( 1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix2, OP_NOP, 0LL);
 /*@8,3*/ mop(OP_LDWR, 1, &r16,         t23[CHIP], cofs,      MSK_D0,      t20[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@9,0*/ exe(OP_MINL3,   &r10,         r29,       EXP_H3210, r28,         EXP_H3210, r10,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@9,1*/ exe(OP_MINL3,   &r12,         ix2,       EXP_H3210, r29,         EXP_H3210, r12,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@9,2*/ exe(OP_MINL3,   &r14,         ix2,       EXP_H3210, ix2,         EXP_H3210, r14,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@9,3*/ exe(OP_MINL3,   &r16,         r31,       EXP_H3210, r31,         EXP_H3210, r16,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@10,0*/exe(OP_MINL,    &r20,         r10,       EXP_H3210, r12,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@10,1*/exe(OP_MINL,    &r24,         r14,       EXP_H3210, r16,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@11,0*/exe(OP_MINL,    &r1,          r20,       EXP_H3210, r24,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@12,0*/exe(OP_MINL,    &r0,          r1,        EXP_H3210, r0,          EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
          /*k=-1*/
 /*@12,0*/mop(OP_LDWR, 1, &r10,         t30[CHIP], cofs,      MSK_D0,      t30[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@12,1*/exe(OP_NOP,     &r28,        (-2LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix3, OP_NOP, 0LL);
 /*@12,1*/mop(OP_LDWR, 1, &r12,         t31[CHIP], cofs,      MSK_D0,      t30[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@12,2*/exe(OP_NOP,     &r29,        (-1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix3, OP_NOP, 0LL);
 /*@12,2*/mop(OP_LDWR, 1, &r14,         t32[CHIP], cofs,      MSK_D0,      t30[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@12,3*/exe(OP_NOP,     &r31,        ( 1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix3, OP_NOP, 0LL);
 /*@12,3*/mop(OP_LDWR, 1, &r16,         t33[CHIP], cofs,      MSK_D0,      t30[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@13,0*/exe(OP_MINL3,   &r10,         r29,       EXP_H3210, r28,         EXP_H3210, r10,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@13,1*/exe(OP_MINL3,   &r12,         ix3,       EXP_H3210, r29,         EXP_H3210, r12,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@13,2*/exe(OP_MINL3,   &r14,         ix3,       EXP_H3210, ix3,         EXP_H3210, r14,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@13,3*/exe(OP_MINL3,   &r16,         r31,       EXP_H3210, r31,         EXP_H3210, r16,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@14,0*/exe(OP_MINL,    &r20,         r10,       EXP_H3210, r12,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@14,1*/exe(OP_MINL,    &r24,         r14,       EXP_H3210, r16,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@15,0*/exe(OP_MINL,    &r1,          r20,       EXP_H3210, r24,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@16,0*/exe(OP_MINL,    &r0,          r1,        EXP_H3210, r0,          EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
          /*k=0*/
 /*@16,0*/mop(OP_LDWR, 1, &r10,         t40[CHIP], cofs,      MSK_D0,      t40[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@16,1*/exe(OP_NOP,     &r28,        (-2LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix4, OP_NOP, 0LL);
 /*@16,1*/mop(OP_LDWR, 1, &r12,         t41[CHIP], cofs,      MSK_D0,      t40[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@16,2*/exe(OP_NOP,     &r29,        (-1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix4, OP_NOP, 0LL);
 /*@16,2*/mop(OP_LDWR, 1, &r14,         t42[CHIP], cofs,      MSK_D0,      t40[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@16,3*/exe(OP_NOP,     &r31,        ( 1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix4, OP_NOP, 0LL);
 /*@16,3*/mop(OP_LDWR, 1, &r16,         t43[CHIP], cofs,      MSK_D0,      t40[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@17,0*/exe(OP_MINL3,   &r10,         r29,       EXP_H3210, r28,         EXP_H3210, r10,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@17,1*/exe(OP_MINL3,   &r12,         ix4,       EXP_H3210, r29,         EXP_H3210, r12,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@17,2*/exe(OP_MINL3,   &r14,         ix4,       EXP_H3210, ix4,         EXP_H3210, r14,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@17,3*/exe(OP_MINL3,   &r16,         r31,       EXP_H3210, r31,         EXP_H3210, r16,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@18,0*/exe(OP_MINL,    &r20,         r10,       EXP_H3210, r12,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@18,1*/exe(OP_MINL,    &r24,         r14,       EXP_H3210, r16,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@19,0*/exe(OP_MINL,    &r1,          r20,       EXP_H3210, r24,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@20,0*/exe(OP_MINL,    &r0,          r1,        EXP_H3210, r0,          EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
          /*k=1*/
 /*@20,0*/mop(OP_LDWR, 1, &r10,         t50[CHIP], cofs,      MSK_D0,      t50[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@20,1*/exe(OP_NOP,     &r28,        (-2LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix5, OP_NOP, 0LL);
 /*@20,1*/mop(OP_LDWR, 1, &r12,         t51[CHIP], cofs,      MSK_D0,      t50[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@20,2*/exe(OP_NOP,     &r29,        (-1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix5, OP_NOP, 0LL);
 /*@20,2*/mop(OP_LDWR, 1, &r14,         t52[CHIP], cofs,      MSK_D0,      t50[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@20,3*/exe(OP_NOP,     &r31,        ( 1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix5, OP_NOP, 0LL);
 /*@20,3*/mop(OP_LDWR, 1, &r16,         t53[CHIP], cofs,      MSK_D0,      t50[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@21,0*/exe(OP_MINL3,   &r10,         r29,       EXP_H3210, r28,         EXP_H3210, r10,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@21,1*/exe(OP_MINL3,   &r12,         ix5,       EXP_H3210, r29,         EXP_H3210, r12,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@21,2*/exe(OP_MINL3,   &r14,         ix5,       EXP_H3210, ix5,         EXP_H3210, r14,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@21,3*/exe(OP_MINL3,   &r16,         r31,       EXP_H3210, r31,         EXP_H3210, r16,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@22,0*/exe(OP_MINL,    &r20,         r10,       EXP_H3210, r12,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@22,1*/exe(OP_MINL,    &r24,         r14,       EXP_H3210, r16,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@23,0*/exe(OP_MINL,    &r1,          r20,       EXP_H3210, r24,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@24,0*/exe(OP_MINL,    &r0,          r1,        EXP_H3210, r0,          EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
          /*k=2*/
 /*@24,0*/mop(OP_LDWR, 1, &r10,         t60[CHIP], cofs,      MSK_D0,      t60[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@24,1*/exe(OP_NOP,     &r28,        (-2LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix6, OP_NOP, 0LL);
 /*@24,1*/mop(OP_LDWR, 1, &r12,         t61[CHIP], cofs,      MSK_D0,      t60[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@24,2*/exe(OP_NOP,     &r29,        (-1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix6, OP_NOP, 0LL);
 /*@24,2*/mop(OP_LDWR, 1, &r14,         t62[CHIP], cofs,      MSK_D0,      t60[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@24,3*/exe(OP_NOP,     &r31,        ( 1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix6, OP_NOP, 0LL);
 /*@24,3*/mop(OP_LDWR, 1, &r16,         t63[CHIP], cofs,      MSK_D0,      t60[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@25,0*/exe(OP_MINL3,   &r10,         r29,       EXP_H3210, r28,         EXP_H3210, r10,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@25,1*/exe(OP_MINL3,   &r12,         ix6,       EXP_H3210, r29,         EXP_H3210, r12,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@25,2*/exe(OP_MINL3,   &r14,         ix6,       EXP_H3210, ix6,         EXP_H3210, r14,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@25,3*/exe(OP_MINL3,   &r16,         r31,       EXP_H3210, r31,         EXP_H3210, r16,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@26,0*/exe(OP_MINL,    &r20,         r10,       EXP_H3210, r12,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@26,1*/exe(OP_MINL,    &r24,         r14,       EXP_H3210, r16,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@27,0*/exe(OP_MINL,    &r1,          r20,       EXP_H3210, r24,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@28,0*/exe(OP_MINL,    &r0,          r1,        EXP_H3210, r0,          EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
          /*k=3*/
 /*@28,0*/mop(OP_LDWR, 1, &r10,         t70[CHIP], cofs,      MSK_D0,      t70[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@28,1*/exe(OP_NOP,     &r28,        (-2LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix7, OP_NOP, 0LL);
 /*@28,1*/mop(OP_LDWR, 1, &r12,         t71[CHIP], cofs,      MSK_D0,      t70[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@28,2*/exe(OP_NOP,     &r29,        (-1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix7, OP_NOP, 0LL);
 /*@28,2*/mop(OP_LDWR, 1, &r14,         t72[CHIP], cofs,      MSK_D0,      t70[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@28,3*/exe(OP_NOP,     &r31,        ( 1LL<<24), EXP_H3210, 0,           EXP_H3210, 0LL,  EXP_H3210, OP_OR,   ix7, OP_NOP, 0LL);
 /*@28,3*/mop(OP_LDWR, 1, &r16,         t73[CHIP], cofs,      MSK_D0,      t70[CHIP], AWD,   0, 0, (Ull)NULL, AWD);
 /*@29,0*/exe(OP_MINL3,   &r10,         r29,       EXP_H3210, r28,         EXP_H3210, r10,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@29,1*/exe(OP_MINL3,   &r12,         ix7,       EXP_H3210, r29,         EXP_H3210, r12,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@29,2*/exe(OP_MINL3,   &r14,         ix7,       EXP_H3210, ix7,         EXP_H3210, r14,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@29,3*/exe(OP_MINL3,   &r16,         r31,       EXP_H3210, r31,         EXP_H3210, r16,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@30,0*/exe(OP_MINL,    &r20,         r10,       EXP_H3210, r12,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@30,1*/exe(OP_MINL,    &r24,         r14,       EXP_H3210, r16,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@31,0*/exe(OP_MINL,    &r1,          r20,       EXP_H3210, r24,         EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@32,0*/exe(OP_MINL,    &r0,          r1,        EXP_H3210, r0,          EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@33,0*/mop(OP_LDWR, 1, &BR[33][0][1],xy[CHIP],  cofs,      MSK_D0,      xy[CHIP],  AWD,   0, 1, (Ull)NULL, AWD);
 /*@33,0*/exe(OP_MINL,    &AR[33][0],   r0,        EXP_H3210, BR[33][0][1],EXP_H3210, 0LL,  EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
 /*@33,0*/mop(OP_STWR, 3, &AR[33][0],   cofs,      xy[CHIP],  MSK_D0,      xy[CHIP],  AWD,   0, 1, (Ull)NULL, AWD);
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
#endif
}

void hokan3(Uint *minxy, Uint *r, Uint *d)
{
#undef  NCHIP
#undef  RMGRP
#undef  PAD
#undef  RRANGE
#define NCHIP     1
#define RMGRP    12
#define PAD       0
#define RRANGE   ((HT-PAD*2)/NCHIP)

  Ull  top, rofs, cofs, oc;
  int  pofs;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
#if 0
  for (top=PAD; top<HT-PAD; top++) { /* scan-lines */
    Uint *xy = minxy+(top/4*4)*WD;
    Uint *dp = d+top*WD;
    for (pofs=-2; pofs<2; pofs++) {
      Uint *rp = r+(top+pofs)*WD;
      for (cofs=0; cofs<WD; cofs++) {
	int x = (int) xy[cofs/4*4]>>24;
	int y = (int)(xy[cofs/4*4]<<8)>>24;
	if (y == pofs) dp[cofs] = rp[cofs+x];
      }
    }
  }
#else
  for (top=0; top<RRANGE; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        int idx = CHIP*RRANGE+top+rofs;
	Uint *xy = minxy+(idx/4*4)*WD;
	Uint *dp = d+idx*WD;
	Uint *rp0 = r+(idx-2)*WD;
	Uint *rp1 = r+(idx-1)*WD;
	Uint *rp2 = r+(idx+0)*WD;
	Uint *rp3 = r+(idx+1)*WD;
	for (cofs=0; cofs<WD; cofs++) {
	  int x = (int) xy[cofs/4*4]>>24;
	  int y = (int)(xy[cofs/4*4]<<8)>>24;
	  dp[cofs] = (y == -2)?rp0[cofs+x]:
	             (y == -1)?rp1[cofs+x]:
                     (y ==  0)?rp2[cofs+x]:
                     (y ==  1)?rp3[cofs+x]:0;
	}
      }
    }
  }
#endif
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  for (top=0; top<RRANGE; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      Ull  jw;
      Uint *xy[NCHIP], *dp[NCHIP], *rp0[NCHIP], *rp1[NCHIP], *rp2[NCHIP], *rp3[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	int idx   = CHIP*RRANGE+top+rofs;
	xy[CHIP]  = minxy+(idx/4*4)*WD; 
	dp[CHIP]  = d+idx*WD;
	rp0[CHIP] = r+(idx-2)*WD;
	rp1[CHIP] = r+(idx-1)*WD;
	rp2[CHIP] = r+(idx+0)*WD;
	rp3[CHIP] = r+(idx+1)*WD;
      }
//EMAX5A begin hokan3 mapdist=0
 /*2*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*1*/for (INIT0=1,LOOP0=AWD,cofs=0-4; LOOP0--; INIT0=0) {       /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
 /*@0,1*/ exe(OP_ADD,     &cofs, INIT0?cofs:cofs, EXP_H3210, 4, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
 /*@1,0*/ exe(OP_NOP,     &jw,   cofs, EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_AND, ~15LL,         OP_SLL,   0LL);

 /*@2,0*/ mop(OP_LDWR, 1, &r10,  xy[CHIP],    jw, MSK_D0,   xy[CHIP],   AWD,        0, 0, (Ull)NULL,       AWD);
 /*@3,0*/ exe(OP_NOP,     &r2,   r10,  EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_AND,  0xff000000LL, OP_SRAA, 22LL); /*x*/
 /*@3,1*/ exe(OP_NOP,     &r3,   r10,  EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_AND,  0x00ff0000LL, OP_SRAB, 16LL); /*y*/
 /*@4,0*/ exe(OP_ADD,     &r4,   r2,   EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL);

 /*@5,0*/ mop(OP_LDWR, 1, &r10,  rp0[CHIP],  r4,  MSK_D0,    rp0[CHIP], AWD,        0, 0, (Ull)NULL,       AWD);          /*rp0[cofs+x]*/
 /*@5,0*/ exe(OP_CMP_EQ,  &r5,   r3,   EXP_H3210, -2,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL); /*y==-2?*/
 /*@6,0*/ exe(OP_CMOV,    &r0,   r5,   EXP_H3210, r10,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL);

 /*@6,0*/ mop(OP_LDWR, 1, &r10,  rp1[CHIP],  r4,  MSK_D0,    rp1[CHIP], AWD,        0, 0, (Ull)NULL,       AWD);          /*rp1[cofs+x]*/
 /*@6,1*/ exe(OP_CMP_EQ,  &r5,   r3,   EXP_H3210, -1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL); /*y==-1?*/
 /*@7,0*/ exe(OP_CMOV,    &r0,   r5,   EXP_H3210, r10,  EXP_H3210,  r0, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL);

 /*@7,0*/ mop(OP_LDWR, 1, &r10,  rp2[CHIP],  r4,  MSK_D0,    rp2[CHIP], AWD,        0, 0, (Ull)NULL,       AWD);          /*rp2[cofs+x]*/
 /*@7,1*/ exe(OP_CMP_EQ,  &r5,   r3,   EXP_H3210,  0,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL); /*y== 0?*/
 /*@8,0*/ exe(OP_CMOV,    &r0,   r5,   EXP_H3210, r10,  EXP_H3210,  r0, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL);

 /*@8,0*/ mop(OP_LDWR, 1, &r10,  rp3[CHIP],  r4,  MSK_D0,    rp3[CHIP], AWD,        0, 0, (Ull)NULL,       AWD);          /*rp3[cofs+x]*/
 /*@8,1*/ exe(OP_CMP_EQ,  &r5,   r3,   EXP_H3210,  1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL); /*y== 1?*/
 /*@9,0*/ exe(OP_CMOV,    &r0,   r5,   EXP_H3210, r10,  EXP_H3210,  r0, EXP_H3210, OP_NOP,  0LL,          OP_NOP,   0LL);
 /*@9,0*/ mop(OP_STWR, 3, &r0,   dp[CHIP],  cofs, MSK_D0,     dp[CHIP], AWD,        0, 1, (Ull)NULL,       AWD);
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
#endif
}

void expand4k(Uint *p, struct X *r)
{
#undef  NCHIP
#undef  RMGRP
#undef  PAD
#undef  RRANGE
#define NCHIP     1
#define RMGRP    12
#define PAD       0
#define RRANGE   ((HT-PAD*2)/NCHIP)

  /*    ¨£¨¡¨¨¨¡¨¨¨¡¨¤              */
  /*    ¨¢  ¨¢k-1   ¨¢ p[k][l:320]  */
  /*    ¨§¨¡¨«¨¡¨«¨¡¨© r[i][j:1024] */
  /*     l-1¨¢kl¨¢l+1  i:1-767      */
  /*    ¨§¨¡¨«¨¡¨«¨¡¨©              */
  /*    ¨¢  ¨¢l+1   ¨¢              */
  /*    ¨¦¨¡¨ª¨¡¨ª¨¡¨¥              */

  /*  ¨£¨¡¨¡¨¨¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¤    */
  /*  ¨¢¨£¨¡¨¢¨¡¨¡¨¡¨¤¡¡¡¡¡¡¡¡          ¨¢    */
  /*  ¨¢¨¢  ¨¢¡¡¡¡¡¡¨¢¡¡¡¡¡¡¡¡          ¨¢k-1 */
  /*  ¨§¨¡¨¡¨«¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©    */
  /*  ¨¢¨¢  ¨¢¡ú¡¡¡¡¨¢¡¡¡¡¡¡¡¡         0¨¢    */
  /*  ¨¢¨¦¨¡¨¢¨¡¨¡¨¡¨¥¡¡¡¡¡¡¡¡          ¨¢    */
  /*  ¨¢    ¨¢(((i*HT)<<4)/768)&15 : 8  ¨¢k   */
  /*  ¨¢    ¨¢                          ¨¢    */
  /*  ¨¢    ¨¢                        15¨¢    */
  /*  ¨§¨¡¨¡¨«¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©    */
  /*  ¨¢    ¨¢¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡          ¨¢k+1 */
  /*  ¨¦¨¡¨¡¨ª¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¥    */
  /*  ¡ú¤òÃæ¿´¤È¤¹¤ëÀµÊý·Á¤¬£²¡ß£²ÎÎ°è¤Î¸Ä¡¹¤È½Å¤Ê¤ëÌÌÀÑÈæ¤òkfraq¤Èlfraq¤ÇÉ½¸½¤¹¤ë */

  Ull  top, rofs, cofs, oc;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
  for (top=0; top<RRANGE; top+=RMGRP) { /* scan-lines */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        int idx = CHIP*RRANGE+top+rofs;
	int k = idx*HT/768;
	int kfraq = (((idx*HT)<<4)/768)&15; /* 4bit */
	int kad = 16-ad(kfraq,8);
	int sk1 = ss(kfraq,8);
	int sk2 = ss(8,kfraq);
	Uint *pp = p+k*WD;
	Uint *rp = r->X[idx];
	for (cofs=0; cofs<1024; cofs++) { /* ËÜÅö¤Ï4095¤Þ¤Ç */
	  int p1 = cofs*WD/1024;
	  int lfraq = (((cofs*WD)<<4)/1024)&15; /* 4bit */
	  int lad = 16-ad(lfraq,8);
	  int sl1 = ss(lfraq,8);
	  int sl2 = ss(8,lfraq);
	  int r1 = kad*lad; /* 4bit*4bit */
	  int r3 = kad*sl1; /* 4bit*4bit */
	  int r2 = kad*sl2; /* 4bit*4bit */
	  int r5 = sk1*lad; /* 4bit*4bit */
	  int r9 = sk1*sl1; /* 4bit*4bit */
	  int r8 = sk1*sl2; /* 4bit*4bit */
	  int r4 = sk2*lad; /* 4bit*4bit */
	  int r7 = sk2*sl1; /* 4bit*4bit */
	  int r6 = sk2*sl2; /* 4bit*4bit */
#if 0
	  printf(" %d %d %d",   r6,r4,r7);
	  printf(" %d %d %d",   r2,r1,r3);
	  printf(" %d %d %d",   r8,r5,r9);
	  printf(" %d\n",       r1+r2+r3+r4+r5+r6+r7+r8+r9); /* ¹ç·×¤ÏÉ¬¤º256¤Ë¤Ê¤ë¤Ï¤º */
#endif
#if 0
	  *rp = (unsigned int)((pp[p1]>>24&0xff)*r1
              +  (pp[p1   -1]>>24&0xff)*r2 + (pp[p1   +1]>>24&0xff)*r3 + (pp[p1-WD  ]>>24&0xff)*r4 + (pp[p1+WD  ]>>24&0xff)*r5
              +  (pp[p1-WD-1]>>24&0xff)*r6 + (pp[p1-WD+1]>>24&0xff)*r7 + (pp[p1+WD-1]>>24&0xff)*r8 + (pp[p1+WD+1]>>24&0xff)*r9)/256<<24
              | (unsigned int)((pp[p1]>>16&0xff)*r1
              +  (pp[p1   -1]>>16&0xff)*r2 + (pp[p1   +1]>>16&0xff)*r3 + (pp[p1-WD  ]>>16&0xff)*r4 + (pp[p1+WD  ]>>16&0xff)*r5
              +  (pp[p1-WD-1]>>16&0xff)*r6 + (pp[p1-WD+1]>>16&0xff)*r7 + (pp[p1+WD-1]>>16&0xff)*r8 + (pp[p1+WD+1]>>16&0xff)*r9)/256<<16
              | (unsigned int)((pp[p1]>> 8&0xff)*r1
              +  (pp[p1   -1]>> 8&0xff)*r2 + (pp[p1   +1]>> 8&0xff)*r3 + (pp[p1-WD  ]>> 8&0xff)*r4 + (pp[p1+WD  ]>> 8&0xff)*r5
              +  (pp[p1-WD-1]>> 8&0xff)*r6 + (pp[p1-WD+1]>> 8&0xff)*r7 + (pp[p1+WD-1]>> 8&0xff)*r8 + (pp[p1+WD+1]>> 8&0xff)*r9)/256<<8;
	  rp++;
#else
	  Uint ph, pl, x;
	  ph = madd(mmul(b2h(pp[p1     ], 1), r1), mmul(b2h(pp[p1-1], 1), r2));
	  ph = madd(mmul(b2h(pp[p1   +1], 1), r3), ph);
	  ph = madd(mmul(b2h(pp[p1-WD  ], 1), r4), ph);
	  ph = madd(mmul(b2h(pp[p1+WD  ], 1), r5), ph);
	  ph = madd(mmul(b2h(pp[p1-WD-1], 1), r6), ph);
	  ph = madd(mmul(b2h(pp[p1-WD+1], 1), r7), ph);
	  ph = madd(mmul(b2h(pp[p1+WD-1], 1), r8), ph);
	  ph = madd(mmul(b2h(pp[p1+WD+1], 1), r9), ph);
	  pl = madd(mmul(b2h(pp[p1     ], 0), r1), mmul(b2h(pp[p1-1], 0), r2));
	  pl = madd(mmul(b2h(pp[p1   +1], 0), r3), pl);
	  pl = madd(mmul(b2h(pp[p1-WD  ], 0), r4), pl);
	  pl = madd(mmul(b2h(pp[p1+WD  ], 0), r5), pl);
	  pl = madd(mmul(b2h(pp[p1-WD-1], 0), r6), pl);
	  pl = madd(mmul(b2h(pp[p1-WD+1], 0), r7), pl);
	  pl = madd(mmul(b2h(pp[p1+WD-1], 0), r8), pl);
	  pl = madd(mmul(b2h(pp[p1+WD+1], 0), r9), pl);
	  *rp = h2b(msrl(ph, 8), 1) | h2b(msrl(pl, 8), 0);
	  rp++;
#endif
	}
      }
    }
  }
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  for (top=0; top<RRANGE; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      Sll  kad[NCHIP], sk1[NCHIP], sk2[NCHIP];
      Uint *pp[NCHIP], *p0[NCHIP], *p1[NCHIP], *p2[NCHIP], *rp[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        int  idx   = CHIP*RRANGE+top+rofs;
	int  k     = idx*HT/768;
	int  kfraq = (((idx*HT)<<4)/768)&15; /* 4bit */
	kad[CHIP]  = 16-ad(kfraq,8);
	sk1[CHIP]  = ss(kfraq,8);
	sk2[CHIP]  = ss(8,kfraq);
	pp[CHIP]   = p+k*WD;
	p0[CHIP]   = pp[CHIP]-WD;
	p1[CHIP]   = pp[CHIP];
	p2[CHIP]   = pp[CHIP]+WD;
	rp[CHIP]   = r->X[idx];
      }
//EMAX5A begin expand4k mapdist=0
 /*2*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*1*/for (INIT0=1,LOOP0=1024,cofs=0-AWD; LOOP0--; INIT0=0) {       /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
 /*@0,1*/ exe(OP_ADD,       &cofs, cofs,      EXP_H3210, AWD,    EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
 /*@1,0*/ exe(OP_NOP,       &r0,   cofs,      EXP_H3210, 0LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND, ~1023LL,  OP_SRL,  8LL);
 /*@1,1*/ exe(OP_NOP,       &r4,   cofs,      EXP_H3210, 0LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,  0x3c0LL, OP_SRL,  6LL);
 /*@2,0*/ exe(OP_ADD,       &r0,   pp[CHIP],  EXP_H3210, r0,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@2,1*/ exe(OP_MSUH,      &r1,   r4,        EXP_H3210, 8LL,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@2,2*/ exe(OP_MSUH,      &r2,   8LL,       EXP_H3210, r4,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@2,3*/ exe(OP_MSSAD,     &r3,   0LL,       EXP_H3210, r4,    EXP_H3210, 8LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@3,3*/ exe(OP_MSUH,      &r3,   16LL,      EXP_H3210, r3,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

 /*@4,1*/ exe(OP_MLUH,      &r21,  sk2[CHIP], EXP_H3210, r1,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@4,1*/ mop(OP_LDWR, 1,   &r10,  r0, -1276, MSK_D0,    (Ull)p0[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);
 /*@4,2*/ exe(OP_MLUH,      &r22,  sk2[CHIP], EXP_H3210, r2,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@4,2*/ mop(OP_LDWR, 1,   &r11,  r0, -1284, MSK_D0,    (Ull)p0[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);
 /*@4,3*/ exe(OP_MLUH,      &r23,  sk2[CHIP], EXP_H3210, r3,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@4,3*/ mop(OP_LDWR, 1,   &r12,  r0, -1280, MSK_D0,    (Ull)p0[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);

 /*@5,1*/ exe(OP_MLUH,      &r13,  r10,       EXP_B5410, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@5,2*/ exe(OP_MLUH,      &r14,  r11,       EXP_B5410, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@5,3*/ exe(OP_MLUH,      &r15,  r12,       EXP_B5410, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

 /*@6,0*/ exe(OP_MAUH3,     &r16,  r13,       EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@6,1*/ exe(OP_MLUH,      &r13,  r10,       EXP_B7632, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@6,2*/ exe(OP_MLUH,      &r14,  r11,       EXP_B7632, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@6,3*/ exe(OP_MLUH,      &r15,  r12,       EXP_B7632, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

 /*@7,0*/ exe(OP_MAUH3,     &r17,  r13,       EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@7,1*/ exe(OP_MLUH,      &r21,  kad[CHIP], EXP_H3210, r1,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@7,1*/ mop(OP_LDWR, 1,   &r10,  r0,     4, MSK_D0,    (Ull)p1[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);
 /*@7,2*/ exe(OP_MLUH,      &r22,  kad[CHIP], EXP_H3210, r2,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@7,2*/ mop(OP_LDWR, 1,   &r11,  r0,    -4, MSK_D0,    (Ull)p1[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);
 /*@7,3*/ exe(OP_MLUH,      &r23,  kad[CHIP], EXP_H3210, r3,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@7,3*/ mop(OP_LDWR, 1,   &r12,  r0,     0, MSK_D0,    (Ull)p1[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);

 /*@8,1*/ exe(OP_MLUH,      &r13,  r10,       EXP_B5410, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@8,2*/ exe(OP_MLUH,      &r14,  r11,       EXP_B5410, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@8,3*/ exe(OP_MLUH,      &r15,  r12,       EXP_B5410, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

 /*@9,0*/ exe(OP_MAUH3,     &r18,  r13,       EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@9,1*/ exe(OP_MLUH,      &r13,  r10,       EXP_B7632, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@9,2*/ exe(OP_MLUH,      &r14,  r11,       EXP_B7632, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@9,3*/ exe(OP_MLUH,      &r15,  r12,       EXP_B7632, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

 /*@10,0*/exe(OP_MAUH3,     &r19,  r13,       EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@10,1*/exe(OP_MLUH,      &r21,  sk1[CHIP], EXP_H3210, r1,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@10,1*/mop(OP_LDWR, 1,   &r10,  r0,  1284, MSK_D0,    (Ull)p2[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);
 /*@10,2*/exe(OP_MLUH,      &r22,  sk1[CHIP], EXP_H3210, r2,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@10,2*/mop(OP_LDWR, 1,   &r11,  r0,  1276, MSK_D0,    (Ull)p2[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);
 /*@10,3*/exe(OP_MLUH,      &r23,  sk1[CHIP], EXP_H3210, r3,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@10,3*/mop(OP_LDWR, 1,   &r12,  r0,  1280, MSK_D0,    (Ull)p2[CHIP],    AWD,    0, 0, (Ull)NULL,       AWD);

 /*@11,1*/exe(OP_MLUH,      &r13,  r10,       EXP_B5410, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@11,2*/exe(OP_MLUH,      &r14,  r11,       EXP_B5410, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@11,3*/exe(OP_MLUH,      &r15,  r12,       EXP_B5410, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

 /*@12,0*/exe(OP_MAUH3,     &r20,  r13,       EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@12,1*/exe(OP_MLUH,      &r13,  r10,       EXP_B7632, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@12,2*/exe(OP_MLUH,      &r14,  r11,       EXP_B7632, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@12,3*/exe(OP_MLUH,      &r15,  r12,       EXP_B7632, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 
 /*@13,0*/exe(OP_MAUH3,     &r21,  r13,       EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

 /*@14,0*/exe(OP_MAUH3,     &r21,  r17,       EXP_H3210, r19,   EXP_H3210, r21, EXP_H3210, OP_OR,   0LL,     OP_SRLM, 8LL);
 /*@14,1*/exe(OP_MAUH3,     &r20,  r16,       EXP_H3210, r18,   EXP_H3210, r20, EXP_H3210, OP_OR,   0LL,     OP_SRLM, 8LL);

 /*@15,0*/exe(OP_MH2BW,     &r31,  r21,       EXP_H3210, r20,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
 /*@15,0*/mop(OP_STWR, 3,   &r31,  (Ull)(rp[CHIP]++),    0LL,   MSK_D0, (Ull)rp[CHIP],  1024, 0, 0, (Ull)NULL, 1024);
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
#endif
}

inline unsigned char limitRGB(int c) {
  if (c<0x00) return 0x00;
  if (c>0xff) return 0xff;
  return c;
}

void unsharp(Uchar *p, Uchar *r)
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     1
#define RMGRP    10
#define OMAP      6
#define PAD       0
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)

  Ull  top, rofs, cofs, oc, pofs;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
#if 0
  for (top=PAD; top<HT-PAD; top++) { /* scan-lines */
    Uchar *p0 = p+((top  )*WD+(0  ))*4;  // p1 p5 p2
    Uchar *p1 = p+((top-1)*WD+(0-1))*4;  // p6 p0 p7
    Uchar *p2 = p+((top-1)*WD+(0+1))*4;  // p3 p8 p4
    Uchar *p3 = p+((top+1)*WD+(0-1))*4;
    Uchar *p4 = p+((top+1)*WD+(0+1))*4;
    Uchar *p5 = p+((top-1)*WD+(0  ))*4;
    Uchar *p6 = p+((top  )*WD+(0-1))*4;
    Uchar *p7 = p+((top  )*WD+(0+1))*4;
    Uchar *p8 = p+((top+1)*WD+(0  ))*4;
    Uchar *rp = r+((top  )*WD+(0  ))*4;
    for (cofs=0; cofs<WD; cofs++) {
      int t0,t1,t2;
      rp[0] = 0;
      t0 = p0[1]; t1 = p1[1]+p2[1]+p3[1]+p4[1]; t2 = p5[1]+p6[1]+p7[1]+p8[1];
      rp[1] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);
      t0 = p0[2]; t1 = p1[2]+p2[2]+p3[2]+p4[2]; t2 = p5[2]+p6[2]+p7[2]+p8[2];
      rp[2] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);
      t0 = p0[3]; t1 = p1[3]+p2[3]+p3[3]+p4[3]; t2 = p5[3]+p6[3]+p7[3]+p8[3];
      rp[3] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);
      p0+=4; p1+=4; p2+=4; p3+=4; p4+=4; p5+=4; p6+=4; p7+=4; p8+=4; rp+=4;
    }
  }
#else
  for (top=0; top<RRANGE; top+=RMGRP) { /* scan-lines */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
        int idx = (CHIP*RRANGE*OMAP+top+rofs)*WD;
	Uchar *p0 = p+(idx   +(0  ))*4;  // p1 p5 p2
	Uchar *p1 = p+(idx-WD+(0-1))*4;  // p6 p0 p7
	Uchar *p2 = p+(idx-WD+(0+1))*4;  // p3 p8 p4
        Uchar *p3 = p+(idx+WD+(0-1))*4;
	Uchar *p4 = p+(idx+WD+(0+1))*4;
	Uchar *p5 = p+(idx-WD+(0  ))*4;
	Uchar *p6 = p+(idx   +(0-1))*4;
	Uchar *p7 = p+(idx   +(0+1))*4;
	Uchar *p8 = p+(idx+WD+(0  ))*4;
	Uchar *rp = r+(idx   +(0  ))*4;
	for (cofs=0; cofs<WD; cofs++) {
          for (oc=0; oc<OMAP; oc++) {
            Uint pix0 = (oc*RRANGE*WD+cofs)*4+0;
            Uint pix1 = (oc*RRANGE*WD+cofs)*4+1;
            Uint pix2 = (oc*RRANGE*WD+cofs)*4+2;
            Uint pix3 = (oc*RRANGE*WD+cofs)*4+3;
	    int t0,t1,t2;
	    rp[pix0] = 0;
	    t0 = p0[pix1]; t1 = p1[pix1]+p2[pix1]+p3[pix1]+p4[pix1]; t2 = p5[pix1]+p6[pix1]+p7[pix1]+p8[pix1];
	    rp[pix1] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);
	    t0 = p0[pix2]; t1 = p1[pix2]+p2[pix2]+p3[pix2]+p4[pix2]; t2 = p5[pix2]+p6[pix2]+p7[pix2]+p8[pix2];
	    rp[pix2] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);
	    t0 = p0[pix3]; t1 = p1[pix3]+p2[pix3]+p3[pix3]+p4[pix3]; t2 = p5[pix3]+p6[pix3]+p7[pix3]+p8[pix3];
	    rp[pix3] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);
	  }
	}
      }
    }
  }
#endif
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  for (top=0; top<RRANGE; top+=RMGRP) { /* scan-lines */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      Uchar *pp0[NCHIP], *pc0[NCHIP], *pn0[NCHIP], *rc0[NCHIP];
      Uchar *pp1[NCHIP], *pc1[NCHIP], *pn1[NCHIP], *rc1[NCHIP];
      Uchar *pp2[NCHIP], *pc2[NCHIP], *pn2[NCHIP], *rc2[NCHIP];
      Uchar *pp3[NCHIP], *pc3[NCHIP], *pn3[NCHIP], *rc3[NCHIP];
      Uchar *pp4[NCHIP], *pc4[NCHIP], *pn4[NCHIP], *rc4[NCHIP];
      Uchar *pp5[NCHIP], *pc5[NCHIP], *pn5[NCHIP], *rc5[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	int idx = (CHIP*RRANGE*OMAP+top+rofs)*WD*4;
	pp0[CHIP] = p+idx+RRANGE*WD* 0-1280;  pc0[CHIP] = p+idx+RRANGE*WD* 0; pn0[CHIP] = p+idx+RRANGE*WD* 0+1280; rc0[CHIP] = r+idx+RRANGE*WD* 0;
	pp1[CHIP] = p+idx+RRANGE*WD* 4-1280;  pc1[CHIP] = p+idx+RRANGE*WD* 4; pn1[CHIP] = p+idx+RRANGE*WD* 4+1280; rc1[CHIP] = r+idx+RRANGE*WD* 4;
	pp2[CHIP] = p+idx+RRANGE*WD* 8-1280;  pc2[CHIP] = p+idx+RRANGE*WD* 8; pn2[CHIP] = p+idx+RRANGE*WD* 8+1280; rc2[CHIP] = r+idx+RRANGE*WD* 8;
	pp3[CHIP] = p+idx+RRANGE*WD*12-1280;  pc3[CHIP] = p+idx+RRANGE*WD*12; pn3[CHIP] = p+idx+RRANGE*WD*12+1280; rc3[CHIP] = r+idx+RRANGE*WD*12;
	pp4[CHIP] = p+idx+RRANGE*WD*16-1280;  pc4[CHIP] = p+idx+RRANGE*WD*16; pn4[CHIP] = p+idx+RRANGE*WD*16+1280; rc4[CHIP] = r+idx+RRANGE*WD*16;
	pp5[CHIP] = p+idx+RRANGE*WD*20-1280;  pc5[CHIP] = p+idx+RRANGE*WD*20; pn5[CHIP] = p+idx+RRANGE*WD*20+1280; rc5[CHIP] = r+idx+RRANGE*WD*20;
      }
//EMAX5A begin unsharp mapdist=1
 /*2*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*1*/for (INIT0=1,LOOP0=AWD,cofs=0-4; LOOP0--; INIT0=0) {       /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
 /*@0,1*/ exe(OP_ADD,       &cofs, cofs,      EXP_H3210, 4LL,  EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
          /*map0*/
 /*@1,0*/ exe(OP_ADD,       &pofs, pc0[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@2,0*/ mop(OP_LDWR, 1,   &r1,   pofs,     -1276, MSK_D0,    (Ull)pp0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@2,0*/ mop(OP_LDWR, 1,   &r2,   pofs,     -1284, MSK_D0,    (Ull)pp0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@2,1*/ mop(OP_LDWR, 1,   &r5,   pofs,     -1280, MSK_D0,    (Ull)pp0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@3,0*/ exe(OP_MAUH,      &r11,  r1,        EXP_B5410, r2,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@3,0*/ mop(OP_LDWR, 1,   &r6,   pofs,      4,    MSK_D0,    (Ull)pc0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@3,0*/ mop(OP_LDWR, 1,   &r7,   pofs,     -4,    MSK_D0,    (Ull)pc0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@3,1*/ exe(OP_MAUH,      &r12,  r1,        EXP_B7632, r2,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@3,1*/ mop(OP_LDWR, 1,   &r0,   pofs,      0,    MSK_D0,    (Ull)pc0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@4,0*/ exe(OP_MLUH,      &r20,  r0,        EXP_B5410, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@4,0*/ mop(OP_LDWR, 1,   &r3,   pofs,      1284, MSK_D0,    (Ull)pn0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@4,0*/ mop(OP_LDWR, 1,   &r4,   pofs,      1276, MSK_D0,    (Ull)pn0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@4,1*/ exe(OP_MLUH,      &r21,  r0,        EXP_B7632, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@4,1*/ mop(OP_LDWR, 1,   &r8,   pofs,      1280, MSK_D0,    (Ull)pn0[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@4,2*/ exe(OP_MAUH,      &r15,  r5,        EXP_B5410, r6,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@4,3*/ exe(OP_MAUH,      &r16,  r5,        EXP_B7632, r6,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@5,0*/ exe(OP_MAUH3,     &r11,  r3,        EXP_B5410, r4,   EXP_B5410, r11, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@5,1*/ exe(OP_MAUH3,     &r12,  r3,        EXP_B7632, r4,   EXP_B7632, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@6,0*/ exe(OP_MLUH,      &r13,  r11,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@6,1*/ exe(OP_MLUH,      &r14,  r12,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@6,2*/ exe(OP_MAUH3,     &r15,  r7,        EXP_B5410, r8,   EXP_B5410, r15, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@6,3*/ exe(OP_MAUH3,     &r16,  r7,        EXP_B7632, r8,   EXP_B7632, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@7,0*/ exe(OP_NOP,       &r7,   r15,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@7,1*/ exe(OP_MLUH,      &r17,  r15,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@7,2*/ exe(OP_NOP,       &r8,   r16,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@7,3*/ exe(OP_MLUH,      &r18,  r16,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@8,0*/ exe(OP_MSUH3,     &r10,  r20,       EXP_H3210, r7,   EXP_H3210, r17, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@8,1*/ exe(OP_MSUH3,     &r11,  r21,       EXP_H3210, r8,   EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@9,0*/ exe(OP_MSUH,      &r20,  r10,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@9,1*/ exe(OP_MSUH,      &r21,  r11,       EXP_H3210, r14,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@10,0*/exe(OP_MH2BW,     &r31,  r21,       EXP_H3210, r20,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@10,0*/mop(OP_STWR, 3,   &r31,  rc0[CHIP], cofs, MSK_D0,    rc0[CHIP],      AWD,      0, 0, (Ull)NULL,   AWD);
          /*map1*/
 /*@10,1*/exe(OP_ADD,       &pofs, pc1[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@11,0*/mop(OP_LDWR, 1,   &r1,   pofs,     -1276, MSK_D0,    (Ull)pp1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@11,0*/mop(OP_LDWR, 1,   &r2,   pofs,     -1284, MSK_D0,    (Ull)pp1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@11,1*/mop(OP_LDWR, 1,   &r5,   pofs,     -1280, MSK_D0,    (Ull)pp1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@12,0*/exe(OP_MAUH,      &r11,  r1,        EXP_B5410, r2,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@12,0*/mop(OP_LDWR, 1,   &r6,   pofs,      4,    MSK_D0,    (Ull)pc1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@12,0*/mop(OP_LDWR, 1,   &r7,   pofs,     -4,    MSK_D0,    (Ull)pc1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@12,1*/exe(OP_MAUH,      &r12,  r1,        EXP_B7632, r2,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@12,1*/mop(OP_LDWR, 1,   &r0,   pofs,      0,    MSK_D0,    (Ull)pc1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@13,0*/exe(OP_MLUH,      &r20,  r0,        EXP_B5410, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@13,0*/mop(OP_LDWR, 1,   &r3,   pofs,      1284, MSK_D0,    (Ull)pn1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@13,0*/mop(OP_LDWR, 1,   &r4,   pofs,      1276, MSK_D0,    (Ull)pn1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@13,1*/exe(OP_MLUH,      &r21,  r0,        EXP_B7632, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@13,1*/mop(OP_LDWR, 1,   &r8,   pofs,      1280, MSK_D0,    (Ull)pn1[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@13,2*/exe(OP_MAUH,      &r15,  r5,        EXP_B5410, r6,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@13,3*/exe(OP_MAUH,      &r16,  r5,        EXP_B7632, r6,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@14,0*/exe(OP_MAUH3,     &r11,  r3,        EXP_B5410, r4,   EXP_B5410, r11, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@14,1*/exe(OP_MAUH3,     &r12,  r3,        EXP_B7632, r4,   EXP_B7632, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@15,0*/exe(OP_MLUH,      &r13,  r11,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@15,1*/exe(OP_MLUH,      &r14,  r12,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@15,2*/exe(OP_MAUH3,     &r15,  r7,        EXP_B5410, r8,   EXP_B5410, r15, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@15,3*/exe(OP_MAUH3,     &r16,  r7,        EXP_B7632, r8,   EXP_B7632, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@16,0*/exe(OP_NOP,       &r7,   r15,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@16,1*/exe(OP_MLUH,      &r17,  r15,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@16,2*/exe(OP_NOP,       &r8,   r16,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@16,3*/exe(OP_MLUH,      &r18,  r16,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@17,0*/exe(OP_MSUH3,     &r10,  r20,       EXP_H3210, r7,   EXP_H3210, r17, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@17,1*/exe(OP_MSUH3,     &r11,  r21,       EXP_H3210, r8,   EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@18,0*/exe(OP_MSUH,      &r20,  r10,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@18,1*/exe(OP_MSUH,      &r21,  r11,       EXP_H3210, r14,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@19,0*/exe(OP_MH2BW,     &r31,  r21,       EXP_H3210, r20,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@19,0*/mop(OP_STWR, 3,   &r31,  rc1[CHIP], cofs, MSK_D0,    rc1[CHIP],      AWD,      0, 0, (Ull)NULL,   AWD);
          /*map2*/
 /*@19,1*/exe(OP_ADD,       &pofs, pc2[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@20,0*/mop(OP_LDWR, 1,   &r1,   pofs,     -1276, MSK_D0,    (Ull)pp2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@20,0*/mop(OP_LDWR, 1,   &r2,   pofs,     -1284, MSK_D0,    (Ull)pp2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@20,1*/mop(OP_LDWR, 1,   &r5,   pofs,     -1280, MSK_D0,    (Ull)pp2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@21,0*/exe(OP_MAUH,      &r11,  r1,        EXP_B5410, r2,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@21,0*/mop(OP_LDWR, 1,   &r6,   pofs,      4,    MSK_D0,    (Ull)pc2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@21,0*/mop(OP_LDWR, 1,   &r7,   pofs,     -4,    MSK_D0,    (Ull)pc2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@21,1*/exe(OP_MAUH,      &r12,  r1,        EXP_B7632, r2,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@21,1*/mop(OP_LDWR, 1,   &r0,   pofs,      0,    MSK_D0,    (Ull)pc2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@22,0*/exe(OP_MLUH,      &r20,  r0,        EXP_B5410, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@22,0*/mop(OP_LDWR, 1,   &r3,   pofs,      1284, MSK_D0,    (Ull)pn2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@22,0*/mop(OP_LDWR, 1,   &r4,   pofs,      1276, MSK_D0,    (Ull)pn2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@22,1*/exe(OP_MLUH,      &r21,  r0,        EXP_B7632, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@22,1*/mop(OP_LDWR, 1,   &r8,   pofs,      1280, MSK_D0,    (Ull)pn2[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@22,2*/exe(OP_MAUH,      &r15,  r5,        EXP_B5410, r6,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@22,3*/exe(OP_MAUH,      &r16,  r5,        EXP_B7632, r6,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@23,0*/exe(OP_MAUH3,     &r11,  r3,        EXP_B5410, r4,   EXP_B5410, r11, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@23,1*/exe(OP_MAUH3,     &r12,  r3,        EXP_B7632, r4,   EXP_B7632, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@24,0*/exe(OP_MLUH,      &r13,  r11,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@24,1*/exe(OP_MLUH,      &r14,  r12,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@24,2*/exe(OP_MAUH3,     &r15,  r7,        EXP_B5410, r8,   EXP_B5410, r15, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@24,3*/exe(OP_MAUH3,     &r16,  r7,        EXP_B7632, r8,   EXP_B7632, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@25,0*/exe(OP_NOP,       &r7,   r15,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@25,1*/exe(OP_MLUH,      &r17,  r15,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@25,2*/exe(OP_NOP,       &r8,   r16,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@25,3*/exe(OP_MLUH,      &r18,  r16,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@26,0*/exe(OP_MSUH3,     &r10,  r20,       EXP_H3210, r7,   EXP_H3210, r17, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@26,1*/exe(OP_MSUH3,     &r11,  r21,       EXP_H3210, r8,   EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@27,0*/exe(OP_MSUH,      &r20,  r10,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@27,1*/exe(OP_MSUH,      &r21,  r11,       EXP_H3210, r14,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@28,0*/exe(OP_MH2BW,     &r31,  r21,       EXP_H3210, r20,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@28,0*/mop(OP_STWR, 3,   &r31,  rc2[CHIP], cofs, MSK_D0,    rc2[CHIP],      AWD,      0, 0, (Ull)NULL,   AWD);
          /*map3*/
 /*@28,1*/exe(OP_ADD,       &pofs, pc3[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@29,0*/mop(OP_LDWR, 1,   &r1,   pofs,     -1276, MSK_D0,    (Ull)pp3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@29,0*/mop(OP_LDWR, 1,   &r2,   pofs,     -1284, MSK_D0,    (Ull)pp3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@29,1*/mop(OP_LDWR, 1,   &r5,   pofs,     -1280, MSK_D0,    (Ull)pp3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@30,0*/exe(OP_MAUH,      &r11,  r1,        EXP_B5410, r2,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@30,0*/mop(OP_LDWR, 1,   &r6,   pofs,      4,    MSK_D0,    (Ull)pc3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@30,0*/mop(OP_LDWR, 1,   &r7,   pofs,     -4,    MSK_D0,    (Ull)pc3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@30,1*/exe(OP_MAUH,      &r12,  r1,        EXP_B7632, r2,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@30,1*/mop(OP_LDWR, 1,   &r0,   pofs,      0,    MSK_D0,    (Ull)pc3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@31,0*/exe(OP_MLUH,      &r20,  r0,        EXP_B5410, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@31,0*/mop(OP_LDWR, 1,   &r3,   pofs,      1284, MSK_D0,    (Ull)pn3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@31,0*/mop(OP_LDWR, 1,   &r4,   pofs,      1276, MSK_D0,    (Ull)pn3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@31,1*/exe(OP_MLUH,      &r21,  r0,        EXP_B7632, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@31,1*/mop(OP_LDWR, 1,   &r8,   pofs,      1280, MSK_D0,    (Ull)pn3[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@31,2*/exe(OP_MAUH,      &r15,  r5,        EXP_B5410, r6,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@31,3*/exe(OP_MAUH,      &r16,  r5,        EXP_B7632, r6,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@32,0*/exe(OP_MAUH3,     &r11,  r3,        EXP_B5410, r4,   EXP_B5410, r11, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@32,1*/exe(OP_MAUH3,     &r12,  r3,        EXP_B7632, r4,   EXP_B7632, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@33,0*/exe(OP_MLUH,      &r13,  r11,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@33,1*/exe(OP_MLUH,      &r14,  r12,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@33,2*/exe(OP_MAUH3,     &r15,  r7,        EXP_B5410, r8,   EXP_B5410, r15, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@33,3*/exe(OP_MAUH3,     &r16,  r7,        EXP_B7632, r8,   EXP_B7632, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@34,0*/exe(OP_NOP,       &r7,   r15,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@34,1*/exe(OP_MLUH,      &r17,  r15,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@34,2*/exe(OP_NOP,       &r8,   r16,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@34,3*/exe(OP_MLUH,      &r18,  r16,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@35,0*/exe(OP_MSUH3,     &r10,  r20,       EXP_H3210, r7,   EXP_H3210, r17, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@35,1*/exe(OP_MSUH3,     &r11,  r21,       EXP_H3210, r8,   EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@36,0*/exe(OP_MSUH,      &r20,  r10,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@36,1*/exe(OP_MSUH,      &r21,  r11,       EXP_H3210, r14,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@37,0*/exe(OP_MH2BW,     &r31,  r21,       EXP_H3210, r20,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@37,0*/mop(OP_STWR, 3,   &r31,  rc3[CHIP], cofs, MSK_D0,    rc3[CHIP],      AWD,      0, 0, (Ull)NULL,   AWD);
          /*map4*/
 /*@37,1*/exe(OP_ADD,       &pofs, pc4[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@38,0*/mop(OP_LDWR, 1,   &r1,   pofs,     -1276, MSK_D0,    (Ull)pp4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@38,0*/mop(OP_LDWR, 1,   &r2,   pofs,     -1284, MSK_D0,    (Ull)pp4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@38,1*/mop(OP_LDWR, 1,   &r5,   pofs,     -1280, MSK_D0,    (Ull)pp4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@39,0*/exe(OP_MAUH,      &r11,  r1,        EXP_B5410, r2,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@39,0*/mop(OP_LDWR, 1,   &r6,   pofs,      4,    MSK_D0,    (Ull)pc4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@39,0*/mop(OP_LDWR, 1,   &r7,   pofs,     -4,    MSK_D0,    (Ull)pc4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@39,1*/exe(OP_MAUH,      &r12,  r1,        EXP_B7632, r2,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@39,1*/mop(OP_LDWR, 1,   &r0,   pofs,      0,    MSK_D0,    (Ull)pc4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@40,0*/exe(OP_MLUH,      &r20,  r0,        EXP_B5410, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@40,0*/mop(OP_LDWR, 1,   &r3,   pofs,      1284, MSK_D0,    (Ull)pn4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@40,0*/mop(OP_LDWR, 1,   &r4,   pofs,      1276, MSK_D0,    (Ull)pn4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@40,1*/exe(OP_MLUH,      &r21,  r0,        EXP_B7632, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@40,1*/mop(OP_LDWR, 1,   &r8,   pofs,      1280, MSK_D0,    (Ull)pn4[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@40,2*/exe(OP_MAUH,      &r15,  r5,        EXP_B5410, r6,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@40,3*/exe(OP_MAUH,      &r16,  r5,        EXP_B7632, r6,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@41,0*/exe(OP_MAUH3,     &r11,  r3,        EXP_B5410, r4,   EXP_B5410, r11, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@41,1*/exe(OP_MAUH3,     &r12,  r3,        EXP_B7632, r4,   EXP_B7632, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@42,0*/exe(OP_MLUH,      &r13,  r11,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@42,1*/exe(OP_MLUH,      &r14,  r12,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@42,2*/exe(OP_MAUH3,     &r15,  r7,        EXP_B5410, r8,   EXP_B5410, r15, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@42,3*/exe(OP_MAUH3,     &r16,  r7,        EXP_B7632, r8,   EXP_B7632, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@43,0*/exe(OP_NOP,       &r7,   r15,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@43,1*/exe(OP_MLUH,      &r17,  r15,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@43,2*/exe(OP_NOP,       &r8,   r16,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@43,3*/exe(OP_MLUH,      &r18,  r16,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@44,0*/exe(OP_MSUH3,     &r10,  r20,       EXP_H3210, r7,   EXP_H3210, r17, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@44,1*/exe(OP_MSUH3,     &r11,  r21,       EXP_H3210, r8,   EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@45,0*/exe(OP_MSUH,      &r20,  r10,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@45,1*/exe(OP_MSUH,      &r21,  r11,       EXP_H3210, r14,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@46,0*/exe(OP_MH2BW,     &r31,  r21,       EXP_H3210, r20,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@46,0*/mop(OP_STWR, 3,   &r31,  rc4[CHIP], cofs, MSK_D0,    rc4[CHIP],      AWD,      0, 0, (Ull)NULL,   AWD);
          /*map5*/
 /*@46,1*/exe(OP_ADD,       &pofs, pc5[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@47,0*/mop(OP_LDWR, 1,   &r1,   pofs,     -1276, MSK_D0,    (Ull)pp5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@47,0*/mop(OP_LDWR, 1,   &r2,   pofs,     -1284, MSK_D0,    (Ull)pp5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@47,1*/mop(OP_LDWR, 1,   &r5,   pofs,     -1280, MSK_D0,    (Ull)pp5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@48,0*/exe(OP_MAUH,      &r11,  r1,        EXP_B5410, r2,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@48,0*/mop(OP_LDWR, 1,   &r6,   pofs,      4,    MSK_D0,    (Ull)pc5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@48,0*/mop(OP_LDWR, 1,   &r7,   pofs,     -4,    MSK_D0,    (Ull)pc5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@48,1*/exe(OP_MAUH,      &r12,  r1,        EXP_B7632, r2,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@48,1*/mop(OP_LDWR, 1,   &r0,   pofs,      0,    MSK_D0,    (Ull)pc5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@49,0*/exe(OP_MLUH,      &r20,  r0,        EXP_B5410, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@49,0*/mop(OP_LDWR, 1,   &r3,   pofs,      1284, MSK_D0,    (Ull)pn5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@49,0*/mop(OP_LDWR, 1,   &r4,   pofs,      1276, MSK_D0,    (Ull)pn5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@49,1*/exe(OP_MLUH,      &r21,  r0,        EXP_B7632, 239,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@49,1*/mop(OP_LDWR, 1,   &r8,   pofs,      1280, MSK_D0,    (Ull)pn5[CHIP], AWD,      0, 0, (Ull)NULL,   AWD);
 /*@49,2*/exe(OP_MAUH,      &r15,  r5,        EXP_B5410, r6,   EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@49,3*/exe(OP_MAUH,      &r16,  r5,        EXP_B7632, r6,   EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@50,0*/exe(OP_MAUH3,     &r11,  r3,        EXP_B5410, r4,   EXP_B5410, r11, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@50,1*/exe(OP_MAUH3,     &r12,  r3,        EXP_B7632, r4,   EXP_B7632, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@51,0*/exe(OP_MLUH,      &r13,  r11,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@51,1*/exe(OP_MLUH,      &r14,  r12,       EXP_H3210, 13,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@51,2*/exe(OP_MAUH3,     &r15,  r7,        EXP_B5410, r8,   EXP_B5410, r15, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@51,3*/exe(OP_MAUH3,     &r16,  r7,        EXP_B7632, r8,   EXP_B7632, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@52,0*/exe(OP_NOP,       &r7,   r15,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@52,1*/exe(OP_MLUH,      &r17,  r15,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@52,2*/exe(OP_NOP,       &r8,   r16,       EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 2LL);
 /*@52,3*/exe(OP_MLUH,      &r18,  r16,       EXP_H3210, 15,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@53,0*/exe(OP_MSUH3,     &r10,  r20,       EXP_H3210, r7,   EXP_H3210, r17, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@53,1*/exe(OP_MSUH3,     &r11,  r21,       EXP_H3210, r8,   EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@54,0*/exe(OP_MSUH,      &r20,  r10,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@54,1*/exe(OP_MSUH,      &r21,  r11,       EXP_H3210, r14,  EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,    OP_SRLM, 7LL);
 /*@55,0*/exe(OP_MH2BW,     &r31,  r21,       EXP_H3210, r20,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@55,0*/mop(OP_STWR, 3,   &r31,  rc5[CHIP], cofs, MSK_D0,    rc5[CHIP],      AWD,      0, 0, (Ull)NULL,   AWD);
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
#endif
}

void blur(Uint *p, Uint *r)
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     1
#define RMGRP    15
#define OMAP      4
#define PAD       0
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)

  Ull  top, rofs, cofs, oc, pofs;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
  for (top=PAD; top<HT-PAD; top++) { /* scan-lines */
    Uint *p0 = p+(top  )*WD  ;
    Uint *p1 = p+(top  )*WD  ;
    Uint *p2 = p+(top  )*WD-1;
    Uint *p3 = p+(top  )*WD+1;
    Uint *p4 = p+(top-1)*WD  ;
    Uint *p5 = p+(top+1)*WD  ;
    Uint *p6 = p+(top-1)*WD-1;
    Uint *p7 = p+(top-1)*WD+1;
    Uint *p8 = p+(top+1)*WD-1;
    Uint *p9 = p+(top+1)*WD+1;
    Uint *rp = r+(top  )*WD  ;
    for (cofs=0; cofs<WD; cofs++) {
#if 0
      *rp = (Uint)((*p1>>24&0xff)*20
          +  (*p2>>24&0xff)*12 + (*p3>>24&0xff)*12 + (*p4>>24&0xff)*12 + (*p5>>24&0xff)*12
          +  (*p6>>24&0xff)* 8 + (*p7>>24&0xff)* 8 + (*p8>>24&0xff)* 8 + (*p9>>24&0xff)* 8)/100<<24
          | (Uint)((*p1>>16&0xff)*20
          +  (*p2>>16&0xff)*12 + (*p3>>16&0xff)*12 + (*p4>>16&0xff)*12 + (*p5>>16&0xff)*12
          +  (*p6>>16&0xff)* 8 + (*p7>>16&0xff)* 8 + (*p8>>16&0xff)* 8 + (*p9>>16&0xff)* 8)/100<<16
          | (Uint)((*p1>> 8&0xff)*20
          +  (*p2>> 8&0xff)*12 + (*p3>> 8&0xff)*12 + (*p4>> 8&0xff)*12 + (*p5>> 8&0xff)*12
          +  (*p6>> 8&0xff)* 8 + (*p7>> 8&0xff)* 8 + (*p8>> 8&0xff)* 8 + (*p9>> 8&0xff)* 8)/100<<8;
      p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; p9++; rp++;
#elif 0
      Uchar s[9], t; int k, l;
      s[0]=*p1>>24;s[1]=*p2>>24;s[2]=*p3>>24;s[3]=*p4>>24;s[4]=*p5>>24;s[5]=*p6>>24;s[6]=*p7>>24;s[7]=*p8>>24;s[8]=*p9>>24;
      for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
      *rp  = s[4]<<24;
      s[0]=*p1>>16;s[1]=*p2>>16;s[2]=*p3>>16;s[3]=*p4>>16;s[4]=*p5>>16;s[5]=*p6>>16;s[6]=*p7>>16;s[7]=*p8>>16;s[8]=*p9>>16;
      for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
      *rp |= s[4]<<16;
      s[0]=*p1>> 8;s[1]=*p2>> 8;s[2]=*p3>> 8;s[3]=*p4>> 8;s[4]=*p5>> 8;s[5]=*p6>> 8;s[6]=*p7>> 8;s[7]=*p8>> 8;s[8]=*p9>> 8;
      for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
      *rp |= s[4]<< 8;
      p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; p9++; rp++;
#else
      Uint s0,s1,s2,s3,s4,s5,s6,s7,s8;
      Uint t0,t1,t2;
      s0=*p1;s1=*p2;s2=*p3;s3=*p4;s4=*p5;s5=*p6;s6=*p7;s7=*p8;s8=*p9;
      /*¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¤
        ¨¢£µ¨¢£³¨¢£¶¨¢  ¨¢£µ¡ã£³¡ã¡ú¨¢  ¨¢£µ¨¢£³¨¢£²¨¢  ¨¢£µ¡ã£³¡ã¡ú¨¢  ¨¢£µ¨¢  ¨¢£³¨¢  ¨¢£µ¡ã  ¡ã¡ú¨¢  ¨¢£µ¨¢
        ¨§¢Ë¨«¢Ë¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨«¢Ë¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨«¨¡¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨©
        ¨¢£±¨¢£°¨¢£²¨¢  ¨¢£±¡ã£°¡ã£²¨¢  ¨¢¡Ý¨¢£°¨¢¡Ý¨¢  ¨¢    £°    ¨¢  ¨¢  ¨¢  ¨¢£°¨¢  ¨¢  ¨¢  ¨¢£°¨¢  ¨¢£°¨¢Ãæ´ÖÃÍ³ÎÄê
        ¨§¢Ë¨«¢Ë¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨«¢Ë¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨«¨¡¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨©
        ¨¢£·¨¢£´¨¢£¸¨¢  ¨¢¡ú¡ã£´¡ã£¸¨¢  ¨¢£±¨¢£´¨¢£¸¨¢  ¨¢¡ú¡ã£´¡ã£¸¨¢  ¨¢£´¨¢  ¨¢£¸¨¢  ¨¢¡ú¡ã  ¡ã£¸¨¢  ¨¢£¸¨¢
        ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨¥*/
      t0 = pmax3(s5,s1,s7); t1 = pmid3(s5,s1,s7); t2 = pmin3(s5,s1,s7);      s5 = t0; s1 = t1; s7 = t2;
      t0 = pmax3(s3,s0,s4); t1 = pmid3(s3,s0,s4); t2 = pmin3(s3,s0,s4);      s3 = t0; s0 = t1; s4 = t2;
      t0 = pmax3(s6,s2,s8); t1 = pmid3(s6,s2,s8); t2 = pmin3(s6,s2,s8);      s6 = t0; s2 = t1; s8 = t2;

      t0 = pmin3(s5,s3,s6); t1 = pmid3(s5,s3,s6);                            s5 = t0; s3 = t1;
      t0 = pmin3(s1,s0,s2); t1 = pmid3(s1,s0,s2); t2 = pmax3(s1,s0,s2);      s1 = t0; s0 = t1; s2 = t2;
      t0 = pmid3(s7,s4,s8); t1 = pmax3(s7,s4,s8);                            s4 = t0; s8 = t1;

      t0 = pmax2(s5,s1);    t1 = pmin2(s5,s1);                               s5 = t0; s1 = t1;
      t0 = pmax3(s3,s0,s4); t1 = pmid3(s3,s0,s4); t2 = pmin3(s3,s0,s4);      s3 = t0; s0 = t1; s4 = t2;
      t0 = pmax2(s2,s8);    t1 = pmin2(s2,s8);                               s2 = t0; s8 = t1;

      t0 = pmin3(s5,s3,s2); t1 = pmid3(s5,s3,s2);                            s5 = t0; s3 = t1;
      t0 = pmid3(s1,s4,s8); t1 = pmax3(s1,s4,s8);                            s4 = t0; s8 = t1;

      t0 = pmax2(s5,s4);    t1 = pmin2(s5,s4);                               s5 = t0; s4 = t1;
      t0 = pmax3(s3,s0,s8); t1 = pmid3(s3,s0,s8); t2 = pmin3(s3,s0,s8);      s3 = t0; s0 = t1; s8 = t2;

      s5 = pmin2(s5,s3);    s8 = pmax2(s4,s8);

      *rp = pmid3(s5,s0,s8);
      p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; p9++; rp++;
#endif
    }
  }
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  for (top=0; top<RRANGE; top+=RMGRP) { /* scan-lines */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      Uint *pp0[NCHIP], *pc0[NCHIP], *pn0[NCHIP], *rc0[NCHIP];
      Uint *pp1[NCHIP], *pc1[NCHIP], *pn1[NCHIP], *rc1[NCHIP];
      Uint *pp2[NCHIP], *pc2[NCHIP], *pn2[NCHIP], *rc2[NCHIP];
      Uint *pp3[NCHIP], *pc3[NCHIP], *pn3[NCHIP], *rc3[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	int idx = (CHIP*RRANGE*OMAP+top+rofs)*WD;
	pp0[CHIP] = p+idx+RRANGE*WD*0-WD;  pc0[CHIP] = p+idx+RRANGE*WD*0; pn0[CHIP] = p+idx+RRANGE*WD*0+WD; rc0[CHIP] = r+idx+RRANGE*WD*0;
	pp1[CHIP] = p+idx+RRANGE*WD*1-WD;  pc1[CHIP] = p+idx+RRANGE*WD*1; pn1[CHIP] = p+idx+RRANGE*WD*1+WD; rc1[CHIP] = r+idx+RRANGE*WD*1;
	pp2[CHIP] = p+idx+RRANGE*WD*2-WD;  pc2[CHIP] = p+idx+RRANGE*WD*2; pn2[CHIP] = p+idx+RRANGE*WD*2+WD; rc2[CHIP] = r+idx+RRANGE*WD*2;
	pp3[CHIP] = p+idx+RRANGE*WD*3-WD;  pc3[CHIP] = p+idx+RRANGE*WD*3; pn3[CHIP] = p+idx+RRANGE*WD*3+WD; rc3[CHIP] = r+idx+RRANGE*WD*3;
      }
//EMAX5A begin blur mapdist=1
 /*2*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*1*/for (INIT0=1,LOOP0=AWD,cofs=0-4; LOOP0--; INIT0=0) {       /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
 /*@0,1*/ exe(OP_ADD,       &cofs, cofs,      EXP_H3210, 4LL,  EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
          /*map0*/
 /*@1,0*/ exe(OP_ADD,       &pofs, pc0[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@2,0*/ mop(OP_LDWR, 1,   &r7,   pofs,     -1276, MSK_D0,    pp0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@2,0*/ mop(OP_LDWR, 1,   &r1,   pofs,     -1280, MSK_D0,    pp0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@2,1*/ mop(OP_LDWR, 1,   &r5,   pofs,     -1284, MSK_D0,    pp0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@3,0*/ exe(OP_MMIN3,     &r17,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@3,0*/ mop(OP_LDWR, 1,   &r4,   pofs,         4, MSK_D0,    pc0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@3,0*/ mop(OP_LDWR, 1,   &r0,   pofs,         0, MSK_D0,    pc0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@3,1*/ exe(OP_MMID3,     &r11,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@3,1*/ mop(OP_LDWR, 1,   &r3,   pofs,        -4, MSK_D0,    pc0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@3,2*/ exe(OP_MMAX3,     &r15,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@4,0*/ exe(OP_MMIN3,     &r14,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@4,0*/ mop(OP_LDWR, 1,   &r8,   pofs,      1284, MSK_D0,    pn0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@4,0*/ mop(OP_LDWR, 1,   &r2,   pofs,      1280, MSK_D0,    pn0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@4,1*/ exe(OP_MMID3,     &r10,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@4,1*/ mop(OP_LDWR, 1,   &r6,   pofs,      1276, MSK_D0,    pn0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@4,2*/ exe(OP_MMAX3,     &r13,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@5,0*/ exe(OP_MMIN3,     &r18,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@5,1*/ exe(OP_MMID3,     &r12,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@5,2*/ exe(OP_MMAX3,     &r16,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-2*/
 /*@6,0*/ exe(OP_MMAX3,     &r2,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@6,1*/ exe(OP_MMID3,     &r0,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@6,2*/ exe(OP_MMIN3,     &r1,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@6,3*/ exe(OP_MMAX3,     &r8,   r17,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@7,0*/ exe(OP_MMID3,     &r4,   r17,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@7,1*/ exe(OP_MMID3,     &r3,   r15,       EXP_H3210, r13,  EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@7,2*/ exe(OP_MMIN3,     &r5,   r15,       EXP_H3210, r13,  EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-3*/
 /*@8,0*/ exe(OP_MMIN3,     &r14,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@8,1*/ exe(OP_MMID3,     &r10,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@8,2*/ exe(OP_MMAX3,     &r13,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@8,3*/ exe(OP_MMIN,      &r18,  r2,        EXP_H3210, r8,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@9,0*/ exe(OP_MMAX,      &r12,  r2,        EXP_H3210, r8,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@9,1*/ exe(OP_MMIN,      &r11,  r5,        EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@9,2*/ exe(OP_MMAX,      &r15,  r5,        EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-4*/
 /*@10,0*/exe(OP_MMID3,     &r4,   r11,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@10,1*/exe(OP_MMIN3,     &r5,   r15,       EXP_H3210, r13,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-5*/
 /*@10,2*/exe(OP_MMAX3,     &r8,   r11,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@10,3*/exe(OP_MMID3,     &r3,   r15,       EXP_H3210, r13,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@11,0*/exe(OP_MMIN,      &r14,  r5,        EXP_H3210, r4,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@11,1*/exe(OP_MMAX,      &r15,  r5,        EXP_H3210, r4,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@11,2*/exe(OP_MMIN3,     &r18,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@11,3*/exe(OP_MMID3,     &r10,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@12,0*/exe(OP_MMAX3,     &r13,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-6*/
 /*@12,1*/exe(OP_MMAX,      &r8,   r14,       EXP_H3210, r18,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@13,0*/exe(OP_MMIN,      &r5,   r15,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@14,0*/exe(OP_MMID3,     &r31,  r5,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@14,0*/mop(OP_STWR, 3,   &r31,  rc0[CHIP], cofs, MSK_D0,    rc0[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
          /*map1*/
 /*@14,1*/exe(OP_ADD,       &pofs, pc1[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@15,0*/mop(OP_LDWR, 1,   &r7,   pofs,     -1276, MSK_D0,    pp1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@15,0*/mop(OP_LDWR, 1,   &r1,   pofs,     -1280, MSK_D0,    pp1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@15,1*/mop(OP_LDWR, 1,   &r5,   pofs,     -1284, MSK_D0,    pp1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@16,0*/exe(OP_MMIN3,     &r17,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@16,0*/mop(OP_LDWR, 1,   &r4,   pofs,         4, MSK_D0,    pc1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@16,0*/mop(OP_LDWR, 1,   &r0,   pofs,         0, MSK_D0,    pc1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@16,1*/exe(OP_MMID3,     &r11,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@16,1*/mop(OP_LDWR, 1,   &r3,   pofs,        -4, MSK_D0,    pc1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@16,2*/exe(OP_MMAX3,     &r15,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@17,0*/exe(OP_MMIN3,     &r14,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@17,0*/mop(OP_LDWR, 1,   &r8,   pofs,      1284, MSK_D0,    pn1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@17,0*/mop(OP_LDWR, 1,   &r2,   pofs,      1280, MSK_D0,    pn1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@17,1*/exe(OP_MMID3,     &r10,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@17,1*/mop(OP_LDWR, 1,   &r6,   pofs,      1276, MSK_D0,    pn1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@17,2*/exe(OP_MMAX3,     &r13,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@18,0*/exe(OP_MMIN3,     &r18,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@18,1*/exe(OP_MMID3,     &r12,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@18,2*/exe(OP_MMAX3,     &r16,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-2*/
 /*@19,0*/exe(OP_MMAX3,     &r2,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@19,1*/exe(OP_MMID3,     &r0,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@19,2*/exe(OP_MMIN3,     &r1,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@19,3*/exe(OP_MMAX3,     &r8,   r17,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@20,0*/exe(OP_MMID3,     &r4,   r17,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@20,1*/exe(OP_MMID3,     &r3,   r15,       EXP_H3210, r13,  EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@20,2*/exe(OP_MMIN3,     &r5,   r15,       EXP_H3210, r13,  EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-3*/
 /*@21,0*/exe(OP_MMIN3,     &r14,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@21,1*/exe(OP_MMID3,     &r10,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@21,2*/exe(OP_MMAX3,     &r13,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@21,3*/exe(OP_MMIN,      &r18,  r2,        EXP_H3210, r8,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@22,0*/exe(OP_MMAX,      &r12,  r2,        EXP_H3210, r8,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@22,1*/exe(OP_MMIN,      &r11,  r5,        EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@22,2*/exe(OP_MMAX,      &r15,  r5,        EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-4*/
 /*@23,0*/exe(OP_MMID3,     &r4,   r11,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@23,1*/exe(OP_MMIN3,     &r5,   r15,       EXP_H3210, r13,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-5*/
 /*@23,2*/exe(OP_MMAX3,     &r8,   r11,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@23,3*/exe(OP_MMID3,     &r3,   r15,       EXP_H3210, r13,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@24,0*/exe(OP_MMIN,      &r14,  r5,        EXP_H3210, r4,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@24,1*/exe(OP_MMAX,      &r15,  r5,        EXP_H3210, r4,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@24,2*/exe(OP_MMIN3,     &r18,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@24,3*/exe(OP_MMID3,     &r10,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@25,0*/exe(OP_MMAX3,     &r13,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-6*/
 /*@25,1*/exe(OP_MMAX,      &r8,   r14,       EXP_H3210, r18,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@26,0*/exe(OP_MMIN,      &r5,   r15,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@27,0*/exe(OP_MMID3,     &r31,  r5,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@27,0*/mop(OP_STWR, 3,   &r31,  rc1[CHIP], cofs, MSK_D0,    rc1[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
          /*map2*/
 /*@27,1*/exe(OP_ADD,       &pofs, pc2[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@28,0*/mop(OP_LDWR, 1,   &r7,   pofs,     -1276, MSK_D0,    pp2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@28,0*/mop(OP_LDWR, 1,   &r1,   pofs,     -1280, MSK_D0,    pp2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@28,1*/mop(OP_LDWR, 1,   &r5,   pofs,     -1284, MSK_D0,    pp2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@29,0*/exe(OP_MMIN3,     &r17,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@29,0*/mop(OP_LDWR, 1,   &r4,   pofs,         4, MSK_D0,    pc2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@29,0*/mop(OP_LDWR, 1,   &r0,   pofs,         0, MSK_D0,    pc2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@29,1*/exe(OP_MMID3,     &r11,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@29,1*/mop(OP_LDWR, 1,   &r3,   pofs,        -4, MSK_D0,    pc2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@29,2*/exe(OP_MMAX3,     &r15,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@30,0*/exe(OP_MMIN3,     &r14,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@30,0*/mop(OP_LDWR, 1,   &r8,   pofs,      1284, MSK_D0,    pn2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@30,0*/mop(OP_LDWR, 1,   &r2,   pofs,      1280, MSK_D0,    pn2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@30,1*/exe(OP_MMID3,     &r10,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@30,1*/mop(OP_LDWR, 1,   &r6,   pofs,      1276, MSK_D0,    pn2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@30,2*/exe(OP_MMAX3,     &r13,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@31,0*/exe(OP_MMIN3,     &r18,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@31,1*/exe(OP_MMID3,     &r12,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@31,2*/exe(OP_MMAX3,     &r16,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-2*/
 /*@32,0*/exe(OP_MMAX3,     &r2,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@32,1*/exe(OP_MMID3,     &r0,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@32,2*/exe(OP_MMIN3,     &r1,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@32,3*/exe(OP_MMAX3,     &r8,   r17,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@33,0*/exe(OP_MMID3,     &r4,   r17,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@33,1*/exe(OP_MMID3,     &r3,   r15,       EXP_H3210, r13,  EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@33,2*/exe(OP_MMIN3,     &r5,   r15,       EXP_H3210, r13,  EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-3*/
 /*@34,0*/exe(OP_MMIN3,     &r14,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@34,1*/exe(OP_MMID3,     &r10,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@34,2*/exe(OP_MMAX3,     &r13,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@34,3*/exe(OP_MMIN,      &r18,  r2,        EXP_H3210, r8,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@35,0*/exe(OP_MMAX,      &r12,  r2,        EXP_H3210, r8,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@35,1*/exe(OP_MMIN,      &r11,  r5,        EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@35,2*/exe(OP_MMAX,      &r15,  r5,        EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-4*/
 /*@36,0*/exe(OP_MMID3,     &r4,   r11,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@36,1*/exe(OP_MMIN3,     &r5,   r15,       EXP_H3210, r13,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-5*/
 /*@36,2*/exe(OP_MMAX3,     &r8,   r11,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@36,3*/exe(OP_MMID3,     &r3,   r15,       EXP_H3210, r13,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@37,0*/exe(OP_MMIN,      &r14,  r5,        EXP_H3210, r4,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@37,1*/exe(OP_MMAX,      &r15,  r5,        EXP_H3210, r4,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@37,2*/exe(OP_MMIN3,     &r18,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@37,3*/exe(OP_MMID3,     &r10,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@38,0*/exe(OP_MMAX3,     &r13,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-6*/
 /*@38,1*/exe(OP_MMAX,      &r8,   r14,       EXP_H3210, r18,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@39,0*/exe(OP_MMIN,      &r5,   r15,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@40,0*/exe(OP_MMID3,     &r31,  r5,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@40,0*/mop(OP_STWR, 3,   &r31,  rc2[CHIP], cofs, MSK_D0,    rc2[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
          /*map3*/
 /*@40,1*/exe(OP_ADD,       &pofs, pc3[CHIP], EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@41,0*/mop(OP_LDWR, 1,   &r7,   pofs,     -1276, MSK_D0,    pp3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@41,0*/mop(OP_LDWR, 1,   &r1,   pofs,     -1280, MSK_D0,    pp3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@41,1*/mop(OP_LDWR, 1,   &r5,   pofs,     -1284, MSK_D0,    pp3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@42,0*/exe(OP_MMIN3,     &r17,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@42,0*/mop(OP_LDWR, 1,   &r4,   pofs,         4, MSK_D0,    pc3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@42,0*/mop(OP_LDWR, 1,   &r0,   pofs,         0, MSK_D0,    pc3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@42,1*/exe(OP_MMID3,     &r11,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@42,1*/mop(OP_LDWR, 1,   &r3,   pofs,        -4, MSK_D0,    pc3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@42,2*/exe(OP_MMAX3,     &r15,  r7,        EXP_H3210, r1,   EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@43,0*/exe(OP_MMIN3,     &r14,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@43,0*/mop(OP_LDWR, 1,   &r8,   pofs,      1284, MSK_D0,    pn3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@43,0*/mop(OP_LDWR, 1,   &r2,   pofs,      1280, MSK_D0,    pn3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@43,1*/exe(OP_MMID3,     &r10,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@43,1*/mop(OP_LDWR, 1,   &r6,   pofs,      1276, MSK_D0,    pn3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
 /*@43,2*/exe(OP_MMAX3,     &r13,  r4,        EXP_H3210, r0,   EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@44,0*/exe(OP_MMIN3,     &r18,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@44,1*/exe(OP_MMID3,     &r12,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@44,2*/exe(OP_MMAX3,     &r16,  r8,        EXP_H3210, r2,   EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-2*/
 /*@45,0*/exe(OP_MMAX3,     &r2,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@45,1*/exe(OP_MMID3,     &r0,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@45,2*/exe(OP_MMIN3,     &r1,   r11,       EXP_H3210, r10,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@45,3*/exe(OP_MMAX3,     &r8,   r17,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@46,0*/exe(OP_MMID3,     &r4,   r17,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@46,1*/exe(OP_MMID3,     &r3,   r15,       EXP_H3210, r13,  EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@46,2*/exe(OP_MMIN3,     &r5,   r15,       EXP_H3210, r13,  EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-3*/
 /*@47,0*/exe(OP_MMIN3,     &r14,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@47,1*/exe(OP_MMID3,     &r10,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@47,2*/exe(OP_MMAX3,     &r13,  r3,        EXP_H3210, r0,   EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@47,3*/exe(OP_MMIN,      &r18,  r2,        EXP_H3210, r8,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@48,0*/exe(OP_MMAX,      &r12,  r2,        EXP_H3210, r8,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@48,1*/exe(OP_MMIN,      &r11,  r5,        EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@48,2*/exe(OP_MMAX,      &r15,  r5,        EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-4*/
 /*@49,0*/exe(OP_MMID3,     &r4,   r11,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@49,1*/exe(OP_MMIN3,     &r5,   r15,       EXP_H3210, r13,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-5*/
 /*@49,2*/exe(OP_MMAX3,     &r8,   r11,       EXP_H3210, r14,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@49,3*/exe(OP_MMID3,     &r3,   r15,       EXP_H3210, r13,  EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@50,0*/exe(OP_MMIN,      &r14,  r5,        EXP_H3210, r4,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@50,1*/exe(OP_MMAX,      &r15,  r5,        EXP_H3210, r4,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@50,2*/exe(OP_MMIN3,     &r18,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@50,3*/exe(OP_MMID3,     &r10,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@51,0*/exe(OP_MMAX3,     &r13,  r3,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*step-6*/
 /*@51,1*/exe(OP_MMAX,      &r8,   r14,       EXP_H3210, r18,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@52,0*/exe(OP_MMIN,      &r5,   r15,       EXP_H3210, r13,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@53,0*/exe(OP_MMID3,     &r31,  r5,        EXP_H3210, r10,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,    OP_NOP,  0LL);
 /*@53,0*/mop(OP_STWR, 3,   &r31,  rc3[CHIP], cofs, MSK_D0,    rc3[CHIP],      AWD,   0, 0, (Ull)NULL,    AWD);
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
#endif
}

void edge(Uint *p, struct E *r)
{
#undef  NCHIP
#undef  RMGRP
#undef  OMAP
#undef  PAD
#undef  RRANGE
#define NCHIP     1
#define RMGRP    10
#define OMAP      6
#define PAD       0
#define RRANGE   ((HT-PAD*2)/NCHIP/OMAP)

#undef  MASK
#undef  EDGEDET
#define MASK      0xffffff00
#define EDGEDET   64

  Ull  top, rofs, cofs, oc, pofs;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
  for (top=PAD; top<HT-PAD; top++) { /* scan-lines */
    Uint  *p0 = p+(top  )*WD  ;
    Uint  *p1 = p+(top-1)*WD-1;
    Uint  *p2 = p+(top+1)*WD+1;
    Uint  *p3 = p+(top-1)*WD  ;
    Uint  *p4 = p+(top+1)*WD  ;
    Uint  *p5 = p+(top-1)*WD+1;
    Uint  *p6 = p+(top+1)*WD-1;
    Uint  *p7 = p+(top  )*WD-1;
    Uint  *p8 = p+(top  )*WD+1;
    Uchar *rp = r->E[top];
    for (cofs=0; cofs<WD; cofs++) {
      int d1 = df(*p1&MASK,*p2&MASK)+df(*p3&MASK,*p4&MASK)+df(*p5&MASK,*p6&MASK)+df(*p7&MASK,*p8&MASK);
      /* 0 < d1(42) < 256*2*4 */
      *rp = d1 < EDGEDET ? 0 : PIXMAX;
      p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; rp++;
    }
  }
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  for (top=0; top<RRANGE; top+=RMGRP) { /* scan-lines */
    for (rofs=0; rofs<RMGRP; rofs++) { /* will be parallelized by multi-chip (M/#chip) */
      Uint *pp0[NCHIP], *pc0[NCHIP], *pn0[NCHIP]; Uchar *rc0[NCHIP];
      Uint *pp1[NCHIP], *pc1[NCHIP], *pn1[NCHIP]; Uchar *rc1[NCHIP];
      Uint *pp2[NCHIP], *pc2[NCHIP], *pn2[NCHIP]; Uchar *rc2[NCHIP];
      Uint *pp3[NCHIP], *pc3[NCHIP], *pn3[NCHIP]; Uchar *rc3[NCHIP];
      Uint *pp4[NCHIP], *pc4[NCHIP], *pn4[NCHIP]; Uchar *rc4[NCHIP];
      Uint *pp5[NCHIP], *pc5[NCHIP], *pn5[NCHIP]; Uchar *rc5[NCHIP];
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	int idx = (CHIP*RRANGE*OMAP+top+rofs)*WD;
	pp0[CHIP] = p+idx+RRANGE*WD*0-WD;  pc0[CHIP] = p+idx+RRANGE*WD*0; pn0[CHIP] = p+idx+RRANGE*WD*0+WD; rc0[CHIP] = (Uchar*)(r->E)+idx+RRANGE*WD*0;
	pp1[CHIP] = p+idx+RRANGE*WD*1-WD;  pc1[CHIP] = p+idx+RRANGE*WD*1; pn1[CHIP] = p+idx+RRANGE*WD*1+WD; rc1[CHIP] = (Uchar*)(r->E)+idx+RRANGE*WD*1;
	pp2[CHIP] = p+idx+RRANGE*WD*2-WD;  pc2[CHIP] = p+idx+RRANGE*WD*2; pn2[CHIP] = p+idx+RRANGE*WD*2+WD; rc2[CHIP] = (Uchar*)(r->E)+idx+RRANGE*WD*2;
	pp3[CHIP] = p+idx+RRANGE*WD*3-WD;  pc3[CHIP] = p+idx+RRANGE*WD*3; pn3[CHIP] = p+idx+RRANGE*WD*3+WD; rc3[CHIP] = (Uchar*)(r->E)+idx+RRANGE*WD*3;
	pp4[CHIP] = p+idx+RRANGE*WD*4-WD;  pc4[CHIP] = p+idx+RRANGE*WD*4; pn4[CHIP] = p+idx+RRANGE*WD*4+WD; rc4[CHIP] = (Uchar*)(r->E)+idx+RRANGE*WD*4;
	pp5[CHIP] = p+idx+RRANGE*WD*5-WD;  pc5[CHIP] = p+idx+RRANGE*WD*5; pn5[CHIP] = p+idx+RRANGE*WD*5+WD; rc5[CHIP] = (Uchar*)(r->E)+idx+RRANGE*WD*5;
      }
//EMAX5A begin edge mapdist=1
 /*2*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*1*/for (INIT0=1,LOOP0=AWD,cofs=0-4; LOOP0--; INIT0=0) {       /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
 /*@0,1*/ exe(OP_ADD,       &cofs, cofs,        EXP_H3210, 4LL,  EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
          /*map0*/
 /*@1,0*/ exe(OP_ADD,       &pofs, pc0[CHIP],   EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@2,0*/ mop(OP_LDWR, 1,   &r5,   pofs,       -1276, MSK_D0,    (Ull)pp0[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@2,0*/ mop(OP_LDWR, 1,   &r3,   pofs,       -1280, MSK_D0,    (Ull)pp0[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@2,1*/ mop(OP_LDWR, 1,   &r1,   pofs,       -1284, MSK_D0,    (Ull)pp0[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@3,0*/ exe(OP_NOP,    &AR[3][0],0LL,         EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@3,0*/ mop(OP_LDWR, 1,   &r8,   pofs,           4, MSK_D0,    (Ull)pc0[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@3,0*/ mop(OP_LDWR, 1,   &r7,   pofs,          -4, MSK_D0,    (Ull)pc0[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@4,0*/ exe(OP_NOP,    &AR[4][0],0LL,         EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@4,0*/ mop(OP_LDWR, 1,   &r2,   pofs,        1284, MSK_D0,    (Ull)pn0[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@4,0*/ mop(OP_LDWR, 1,   &r4,   pofs,        1280, MSK_D0,    (Ull)pn0[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@4,1*/ mop(OP_LDWR, 1,   &r6,   pofs,        1276, MSK_D0,    (Ull)pn0[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@4,1*/ exe(OP_MSSAD,     &r7,   0LL,         EXP_H3210, r7,   EXP_H3210, r8,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@5,0*/ exe(OP_MSSAD,     &r1,   0LL,         EXP_H3210, r1,   EXP_H3210, r2,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@5,1*/ exe(OP_MSSAD,     &r3,   0LL,         EXP_H3210, r3,   EXP_H3210, r4,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@5,2*/ exe(OP_MSSAD,     &r5,   0LL,         EXP_H3210, r5,   EXP_H3210, r6,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@6,0*/ exe(OP_MAUH,      &r1,   r3,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@6,1*/ exe(OP_MAUH,      &r5,   r7,          EXP_H3210, r5,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@7,0*/ exe(OP_MAUH,      &r1,   r5,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL,    OP_NOP,  0LL);
 /*@8,0*/ exe(OP_MCAS,      &r31,  r1,          EXP_H3210, 64,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@8,0*/ mop(OP_STBR, 3,   &r31,  rc0[CHIP]++,    0, MSK_D0,    (Ull)rc0[CHIP],     AWD/4,     0,   0,   (Ull)NULL, AWD/4);
          /*map1*/
 /*@8,1*/ exe(OP_ADD,       &pofs, pc1[CHIP],   EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@9,0*/ mop(OP_LDWR, 1,   &r5,   pofs,       -1276, MSK_D0,    (Ull)pp1[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@9,0*/ mop(OP_LDWR, 1,   &r3,   pofs,       -1280, MSK_D0,    (Ull)pp1[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@9,1*/ mop(OP_LDWR, 1,   &r1,   pofs,       -1284, MSK_D0,    (Ull)pp1[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@10,0*/exe(OP_NOP,    &AR[10][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@10,0*/mop(OP_LDWR, 1,   &r8,   pofs,           4, MSK_D0,    (Ull)pc1[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@10,0*/mop(OP_LDWR, 1,   &r7,   pofs,          -4, MSK_D0,    (Ull)pc1[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@11,0*/exe(OP_NOP,    &AR[11][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@11,0*/mop(OP_LDWR, 1,   &r2,   pofs,        1284, MSK_D0,    (Ull)pn1[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@11,0*/mop(OP_LDWR, 1,   &r4,   pofs,        1280, MSK_D0,    (Ull)pn1[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@11,1*/mop(OP_LDWR, 1,   &r6,   pofs,        1276, MSK_D0,    (Ull)pn1[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@11,1*/exe(OP_MSSAD,     &r7,   0LL,         EXP_H3210, r7,   EXP_H3210, r8,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@12,0*/exe(OP_MSSAD,     &r1,   0LL,         EXP_H3210, r1,   EXP_H3210, r2,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@12,1*/exe(OP_MSSAD,     &r3,   0LL,         EXP_H3210, r3,   EXP_H3210, r4,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@12,2*/exe(OP_MSSAD,     &r5,   0LL,         EXP_H3210, r5,   EXP_H3210, r6,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@13,0*/exe(OP_MAUH,      &r1,   r3,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@13,1*/exe(OP_MAUH,      &r5,   r7,          EXP_H3210, r5,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@14,0*/exe(OP_MAUH,      &r1,   r5,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL,    OP_NOP,  0LL);
 /*@15,0*/exe(OP_MCAS,      &r31,  r1,          EXP_H3210, 64,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@15,0*/mop(OP_STBR, 3,   &r31,  rc1[CHIP]++,    0, MSK_D0,    (Ull)rc1[CHIP],     AWD/4,     0,   0,   (Ull)NULL,  AWD/4);
          /*map2*/
 /*@15,1*/exe(OP_ADD,       &pofs, pc2[CHIP],   EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@16,0*/mop(OP_LDWR, 1,   &r5,   pofs,       -1276, MSK_D0,    (Ull)pp2[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@16,0*/mop(OP_LDWR, 1,   &r3,   pofs,       -1280, MSK_D0,    (Ull)pp2[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@16,1*/mop(OP_LDWR, 1,   &r1,   pofs,       -1284, MSK_D0,    (Ull)pp2[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@17,0*/exe(OP_NOP,    &AR[17][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@17,0*/mop(OP_LDWR, 1,   &r8,   pofs,           4, MSK_D0,    (Ull)pc2[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@17,0*/mop(OP_LDWR, 1,   &r7,   pofs,          -4, MSK_D0,    (Ull)pc2[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@18,0*/exe(OP_NOP,    &AR[18][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@18,0*/mop(OP_LDWR, 1,   &r2,   pofs,        1284, MSK_D0,    (Ull)pn2[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@18,0*/mop(OP_LDWR, 1,   &r4,   pofs,        1280, MSK_D0,    (Ull)pn2[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@18,1*/mop(OP_LDWR, 1,   &r6,   pofs,        1276, MSK_D0,    (Ull)pn2[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@18,1*/exe(OP_MSSAD,     &r7,   0LL,         EXP_H3210, r7,   EXP_H3210, r8,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@19,0*/exe(OP_MSSAD,     &r1,   0LL,         EXP_H3210, r1,   EXP_H3210, r2,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@19,1*/exe(OP_MSSAD,     &r3,   0LL,         EXP_H3210, r3,   EXP_H3210, r4,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@19,2*/exe(OP_MSSAD,     &r5,   0LL,         EXP_H3210, r5,   EXP_H3210, r6,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@20,0*/exe(OP_MAUH,      &r1,   r3,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@20,1*/exe(OP_MAUH,      &r5,   r7,          EXP_H3210, r5,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@21,0*/exe(OP_MAUH,      &r1,   r5,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL,    OP_NOP,  0LL);
 /*@22,0*/exe(OP_MCAS,      &r31,  r1,          EXP_H3210, 64,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@22,0*/mop(OP_STBR, 3,   &r31,  rc2[CHIP]++,    0, MSK_D0,    (Ull)rc2[CHIP],     AWD/4,     0,   0,   (Ull)NULL,  AWD/4);
          /*map3*/
 /*@22,1*/exe(OP_ADD,       &pofs, pc3[CHIP],   EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@23,0*/mop(OP_LDWR, 1,   &r5,   pofs,       -1276, MSK_D0,    (Ull)pp3[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@23,0*/mop(OP_LDWR, 1,   &r3,   pofs,       -1280, MSK_D0,    (Ull)pp3[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@23,1*/mop(OP_LDWR, 1,   &r1,   pofs,       -1284, MSK_D0,    (Ull)pp3[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@24,0*/exe(OP_NOP,    &AR[24][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@24,0*/mop(OP_LDWR, 1,   &r8,   pofs,           4, MSK_D0,    (Ull)pc3[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@24,0*/mop(OP_LDWR, 1,   &r7,   pofs,          -4, MSK_D0,    (Ull)pc3[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@25,0*/exe(OP_NOP,    &AR[25][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@25,0*/mop(OP_LDWR, 1,   &r2,   pofs,        1284, MSK_D0,    (Ull)pn3[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@25,0*/mop(OP_LDWR, 1,   &r4,   pofs,        1280, MSK_D0,    (Ull)pn3[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@25,1*/mop(OP_LDWR, 1,   &r6,   pofs,        1276, MSK_D0,    (Ull)pn3[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@25,1*/exe(OP_MSSAD,     &r7,   0LL,         EXP_H3210, r7,   EXP_H3210, r8,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@26,0*/exe(OP_MSSAD,     &r1,   0LL,         EXP_H3210, r1,   EXP_H3210, r2,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@26,1*/exe(OP_MSSAD,     &r3,   0LL,         EXP_H3210, r3,   EXP_H3210, r4,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@26,2*/exe(OP_MSSAD,     &r5,   0LL,         EXP_H3210, r5,   EXP_H3210, r6,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@27,0*/exe(OP_MAUH,      &r1,   r3,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@27,1*/exe(OP_MAUH,      &r5,   r7,          EXP_H3210, r5,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@28,0*/exe(OP_MAUH,      &r1,   r5,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL,    OP_NOP,  0LL);
 /*@29,0*/exe(OP_MCAS,      &r31,  r1,          EXP_H3210, 64,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@29,0*/mop(OP_STBR, 3,   &r31,  rc3[CHIP]++,    0, MSK_D0,    (Ull)rc3[CHIP],     AWD/4,     0,   0,   (Ull)NULL,  AWD/4);
          /*map4*/
 /*@29,1*/exe(OP_ADD,       &pofs, pc4[CHIP],   EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@30,0*/mop(OP_LDWR, 1,   &r5,   pofs,       -1276, MSK_D0,    (Ull)pp4[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@30,0*/mop(OP_LDWR, 1,   &r3,   pofs,       -1280, MSK_D0,    (Ull)pp4[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@30,1*/mop(OP_LDWR, 1,   &r1,   pofs,       -1284, MSK_D0,    (Ull)pp4[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@31,0*/exe(OP_NOP,    &AR[31][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@31,0*/mop(OP_LDWR, 1,   &r8,   pofs,           4, MSK_D0,    (Ull)pc4[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@31,0*/mop(OP_LDWR, 1,   &r7,   pofs,          -4, MSK_D0,    (Ull)pc4[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@32,0*/exe(OP_NOP,    &AR[32][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@32,0*/mop(OP_LDWR, 1,   &r2,   pofs,        1284, MSK_D0,    (Ull)pn4[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@32,0*/mop(OP_LDWR, 1,   &r4,   pofs,        1280, MSK_D0,    (Ull)pn4[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@32,1*/mop(OP_LDWR, 1,   &r6,   pofs,        1276, MSK_D0,    (Ull)pn4[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@32,1*/exe(OP_MSSAD,     &r7,   0LL,         EXP_H3210, r7,   EXP_H3210, r8,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@33,0*/exe(OP_MSSAD,     &r1,   0LL,         EXP_H3210, r1,   EXP_H3210, r2,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@33,1*/exe(OP_MSSAD,     &r3,   0LL,         EXP_H3210, r3,   EXP_H3210, r4,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@33,2*/exe(OP_MSSAD,     &r5,   0LL,         EXP_H3210, r5,   EXP_H3210, r6,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@34,0*/exe(OP_MAUH,      &r1,   r3,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@34,1*/exe(OP_MAUH,      &r5,   r7,          EXP_H3210, r5,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@35,0*/exe(OP_MAUH,      &r1,   r5,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL,    OP_NOP,  0LL);
 /*@36,0*/exe(OP_MCAS,      &r31,  r1,          EXP_H3210, 64,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@36,0*/mop(OP_STBR, 3,   &r31,  rc4[CHIP]++,    0, MSK_D0,    (Ull)rc4[CHIP],     AWD/4,     0,   0,   (Ull)NULL,  AWD/4);
          /*map5*/
 /*@36,1*/exe(OP_ADD,       &pofs, pc5[CHIP],   EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@37,0*/mop(OP_LDWR, 1,   &r5,   pofs,       -1276, MSK_D0,    (Ull)pp5[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@37,0*/mop(OP_LDWR, 1,   &r3,   pofs,       -1280, MSK_D0,    (Ull)pp5[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@37,1*/mop(OP_LDWR, 1,   &r1,   pofs,       -1284, MSK_D0,    (Ull)pp5[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@38,0*/exe(OP_NOP,    &AR[38][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@38,0*/mop(OP_LDWR, 1,   &r8,   pofs,           4, MSK_D0,    (Ull)pc5[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@38,0*/mop(OP_LDWR, 1,   &r7,   pofs,          -4, MSK_D0,    (Ull)pc5[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@39,0*/exe(OP_NOP,    &AR[39][0],0LL,        EXP_H3210, 0LL,  EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL,    OP_NOP,  0LL);
 /*@39,0*/mop(OP_LDWR, 1,   &r2,   pofs,        1284, MSK_D0,    (Ull)pn5[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@39,0*/mop(OP_LDWR, 1,   &r4,   pofs,        1280, MSK_D0,    (Ull)pn5[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@39,1*/mop(OP_LDWR, 1,   &r6,   pofs,        1276, MSK_D0,    (Ull)pn5[CHIP],       AWD,     0,   0,   (Ull)NULL,  AWD);
 /*@39,1*/exe(OP_MSSAD,     &r7,   0LL,         EXP_H3210, r7,   EXP_H3210, r8,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@40,0*/exe(OP_MSSAD,     &r1,   0LL,         EXP_H3210, r1,   EXP_H3210, r2,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@40,1*/exe(OP_MSSAD,     &r3,   0LL,         EXP_H3210, r3,   EXP_H3210, r4,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@40,2*/exe(OP_MSSAD,     &r5,   0LL,         EXP_H3210, r5,   EXP_H3210, r6,  EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@41,0*/exe(OP_MAUH,      &r1,   r3,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@41,1*/exe(OP_MAUH,      &r5,   r7,          EXP_H3210, r5,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@42,0*/exe(OP_MAUH,      &r1,   r5,          EXP_H3210, r1,   EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL,    OP_NOP,  0LL);
 /*@43,0*/exe(OP_MCAS,      &r31,  r1,          EXP_H3210, 64,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,    OP_NOP,  0LL);
 /*@43,0*/mop(OP_STBR, 3,   &r31,  rc5[CHIP]++,    0, MSK_D0,    (Ull)rc5[CHIP],     AWD/4,     0,   0,   (Ull)NULL,  AWD/4);
        }
      }
//EMAX5A end
    }
  }
//EMAX5A drain_dirty_lmm
#endif
}

void bblur(struct E *p, struct F *r)
{
  Ull  top, rofs, cofs, oc, pofs;
  for (top=0; top<HT; top++) {
    Uchar *p1 = p->E[top]-WD-1;
    Uchar *p2 = p->E[top]   -1;
    Uchar *p3 = p->E[top]+WD-1;
    Uchar *p4 = p->E[top]-WD  ;
    Uchar *p5 = p->E[top]     ; Uchar *r5 = r->F[top]+0;
    Uchar *p6 = p->E[top]+WD  ;
    Uchar *p7 = p->E[top]-WD+1;
    Uchar *p8 = p->E[top]   +1; Uchar *r8 = r->F[top]+1;
    Uchar *p9 = p->E[top]+WD+1;
    Uchar *pa = p->E[top]-WD+2;
    Uchar *pb = p->E[top]   +2; Uchar *rb = r->F[top]+2;
    Uchar *pc = p->E[top]+WD+2;
    Uchar *pd = p->E[top]-WD+3;
    Uchar *pe = p->E[top]   +3; Uchar *re = r->F[top]+3;
    Uchar *pf = p->E[top]+WD+3;
    Uchar *pg = p->E[top]-WD+4;
    Uchar *ph = p->E[top]   +4;
    Uchar *pi = p->E[top]+WD+4;
    for (cofs=0; cofs<WD; cofs+=4) {
      int d1, d2, d3, d4;
      d1 = *p1 + *p2 + *p3 + *p4 + *p5 + *p6 + *p7 + *p8 + *p9;
      if (d1 < 255*5) *r5 = 0; else *r5 = 255;
      d2 = *p4 + *p5 + *p6 + *p7 + *p8 + *p9 + *pa + *pb + *pc;
      if (d2 < 255*5) *r8 = 0; else *r8 = 255;
      d3 = *p7 + *p8 + *p9 + *pa + *pb + *pc + *pd + *pe + *pf;
      if (d3 < 255*5) *rb = 0; else *rb = 255;
      d4 = *pa + *pb + *pc + *pd + *pe + *pf + *pg + *ph + *pi;
      if (d4 < 255*5) *re = 0; else *re = 255;
      p1+=4; p2+=4; p3+=4; p4+=4; p5+=4; p6+=4; p7+=4; p8+=4; p9+=4;
      pa+=4; pb+=4; pc+=4; pd+=4; pe+=4; pf+=4; pg+=4; ph+=4; pi+=4;
      r5+=4; r8+=4; rb+=4; re+=4;
    }
  }
}

void wdifline(Uint *l, Uint *r, struct SAD2 *d, int k)
{
#undef  NCHIP
#undef  RRANGE
#define NCHIP     1
#define RRANGE   ((HT-DWIN*2)/NCHIP)

  /*printf("wdiflineL-start k=%d\n", k);*/

  Ull  top, rofs1, rofs2, cofs, oc;
  int  pofs1;
  Ull  CHIP;
#if !defined(EMAX5) && !defined(EMAX6)
#if 0
  for (top=DWIN; top<HT-DWIN; top++) { /* scan-lines */
    for (cofs=DWIN+k/2; cofs<WD-DWIN-k/2; cofs++) /* one scan-line */
      d->SAD2[top][cofs] = 0;
  }

  for (top=DWIN; top<HT-DWIN; top++) { /* scan-lines */
    Uint *lp = l + top*WD+k; /* L */
    Uint *rp = r + top*WD; /* R */
    for (pofs1=-DWIN; pofs1<DWIN; pofs1++) {
      Uint *dp = &d->SAD2[top+pofs1][DWIN+k/2];
      for (cofs=0; cofs<WD-DWIN*2; cofs++) { /* one scan-line */
	int x, retval = 0;
	for (x=0; x<DWIN*2; x++)
	  retval += df((*(lp+cofs+x))&MASK, (*(rp+cofs+x))&MASK);
	*(dp+cofs) += retval;
      }
    }
  }
#else
  for (top=0; top<RRANGE; top++) { /* will be parallelized by multi-chip (M/#chip) */
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
      int idx = CHIP*RRANGE+DWIN+top;
      for (cofs=DWIN+k/2; cofs<WD-DWIN-k/2; cofs++) /* one scan-line */
	d->SAD2[idx][cofs] = 0;
    }
  }

  for (top=0; top<RRANGE; top++) { /* will be parallelized by multi-chip (M/#chip) */
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
      int  idx = CHIP*RRANGE+DWIN+top;
      Uint *lp = l + idx*WD+k; /* L */
      Uint *rp = r + idx*WD; /* R */
      for (pofs1=-DWIN; pofs1<DWIN; pofs1++) {
	Uint *dp = &d->SAD2[idx+pofs1][DWIN+k/2];
	for (cofs=0; cofs<WD-DWIN*2; cofs++) { /* one scan-line */
	  int x, retval = 0;
	  for (x=0; x<DWIN*2; x++)
	    retval += df((*(lp+cofs+x))&MASK, (*(rp+cofs+x))&MASK);
	  *(dp+cofs) += retval;
	}
      }
    }
  }
#endif
#else
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  for (top=0; top<RRANGE; top++) { /* will be parallelized by multi-chip (M/#chip) */
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
      int idx = CHIP*RRANGE+DWIN+top;
      for (cofs=DWIN+k/2; cofs<WD-DWIN-k/2; cofs++) /* one scan-line */
	d->SAD2[idx][cofs] = 0;
    }
  }
  for (top=0; top<RRANGE; top++) { /* scan-lines */
    Uint *lp[NCHIP],  *rp[NCHIP];
    Uint *dp0[NCHIP], *dp1[NCHIP], *dp2[NCHIP], *dp3[NCHIP], *dp4[NCHIP], *dp5[NCHIP], *dp6[NCHIP], *dp7[NCHIP];
    Uint *dp8[NCHIP], *dp9[NCHIP], *dpa[NCHIP], *dpb[NCHIP], *dpc[NCHIP], *dpd[NCHIP], *dpe[NCHIP], *dpf[NCHIP];
    for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
      int idx = CHIP*RRANGE+DWIN+top;
      lp[CHIP] = l+idx*WD+k; rp[CHIP] = r+idx*WD;
      dp0[CHIP] = &d->SAD2[idx-8][DWIN+k/2];
      dp1[CHIP] = &d->SAD2[idx-7][DWIN+k/2];
      dp2[CHIP] = &d->SAD2[idx-6][DWIN+k/2];
      dp3[CHIP] = &d->SAD2[idx-5][DWIN+k/2];
      dp4[CHIP] = &d->SAD2[idx-4][DWIN+k/2];
      dp5[CHIP] = &d->SAD2[idx-3][DWIN+k/2];
      dp6[CHIP] = &d->SAD2[idx-2][DWIN+k/2];
      dp7[CHIP] = &d->SAD2[idx-1][DWIN+k/2];
      dp8[CHIP] = &d->SAD2[idx  ][DWIN+k/2];
      dp9[CHIP] = &d->SAD2[idx+1][DWIN+k/2];
      dpa[CHIP] = &d->SAD2[idx+2][DWIN+k/2];
      dpb[CHIP] = &d->SAD2[idx+3][DWIN+k/2];
      dpc[CHIP] = &d->SAD2[idx+4][DWIN+k/2];
      dpd[CHIP] = &d->SAD2[idx+5][DWIN+k/2];
      dpe[CHIP] = &d->SAD2[idx+6][DWIN+k/2];
      dpf[CHIP] = &d->SAD2[idx+7][DWIN+k/2];
    }
//EMAX5A begin wdifline mapdist=0
 /*2*/for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC/#chip) */
   /*1*/for (INIT0=1,LOOP0=AWD,cofs=0-4; LOOP0--; INIT0=0) {       /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
 /*@0,1*/ exe(OP_ADD,      &cofs,        cofs,        EXP_H3210, 4LL,  EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
          /*map0*/
 /*@1,0*/ exe(OP_ADD,      &rofs1,       lp[CHIP],   EXP_H3210, cofs, EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@1,1*/ exe(OP_ADD,      &rofs2,       rp[CHIP],   EXP_H3210, cofs, EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@2,0*/ mop(OP_LDWR, 1,  &r2,          rofs1,   0,  MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@2,0*/ mop(OP_LDWR, 1,  &r3,          rofs1,   4,  MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@2,1*/ mop(OP_LDWR, 1,  &r4,          rofs1,   8,  MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@2,1*/ mop(OP_LDWR, 1,  &r5,          rofs1,   12, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@2,2*/ mop(OP_LDWR, 1,  &r6,          rofs2,   0,  MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@2,2*/ mop(OP_LDWR, 1,  &r7,          rofs2,   4,  MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@2,3*/ mop(OP_LDWR, 1,  &r8,          rofs2,   8,  MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@2,3*/ mop(OP_LDWR, 1,  &r9,          rofs2,   12, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@3,0*/ exe(OP_MSAD,     &r22,         r2,          EXP_H3210, r6,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@3,0*/ mop(OP_LDWR, 1,  &r12,         rofs1,   16, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@3,0*/ mop(OP_LDWR, 1,  &r13,         rofs1,   20, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@3,1*/ exe(OP_MSAD,     &r23,         r3,          EXP_H3210, r7,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@3,1*/ mop(OP_LDWR, 1,  &r14,         rofs1,   24, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@3,1*/ mop(OP_LDWR, 1,  &r15,         rofs1,   28, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@3,2*/ exe(OP_MSAD,     &r24,         r4,          EXP_H3210, r8,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@3,2*/ mop(OP_LDWR, 1,  &r16,         rofs2,   16, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@3,2*/ mop(OP_LDWR, 1,  &r17,         rofs2,   20, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@3,3*/ exe(OP_MSAD,     &r25,         r5,          EXP_H3210, r9,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@3,3*/ mop(OP_LDWR, 1,  &r18,         rofs2,   24, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@3,3*/ mop(OP_LDWR, 1,  &r19,         rofs2,   28, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@4,0*/ exe(OP_MSSAD,    &r12,         r22,         EXP_H3210, r12,  EXP_H3210,  r16, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@4,0*/ mop(OP_LDWR, 1,  &r2,          rofs1,   32, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@4,0*/ mop(OP_LDWR, 1,  &r3,          rofs1,   36, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@4,1*/ exe(OP_MSSAD,    &r13,         r23,         EXP_H3210, r13,  EXP_H3210,  r17, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@4,1*/ mop(OP_LDWR, 1,  &r4,          rofs1,   40, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@4,1*/ mop(OP_LDWR, 1,  &r5,          rofs1,   44, MSK_D0,    lp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@4,2*/ exe(OP_MSSAD,    &r14,         r24,         EXP_H3210, r14,  EXP_H3210,  r18, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@4,2*/ mop(OP_LDWR, 1,  &r6,          rofs2,   32, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@4,2*/ mop(OP_LDWR, 1,  &r7,          rofs2,   36, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@4,3*/ exe(OP_MSSAD,    &r15,         r25,         EXP_H3210, r15,  EXP_H3210,  r19, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@4,3*/ mop(OP_LDWR, 1,  &r8,          rofs2,   40, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@4,3*/ mop(OP_LDWR, 1,  &r9,          rofs2,   44, MSK_D0,    rp[CHIP],       AWD, 0, 0, (Ull)NULL,    AWD);
 /*@5,0*/ exe(OP_MSSAD,    &r22,         r12,         EXP_H3210, r2,   EXP_H3210,  r6, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@5,1*/ exe(OP_MSSAD,    &r23,         r13,         EXP_H3210, r3,   EXP_H3210,  r7, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@5,2*/ exe(OP_MSSAD,    &r24,         r14,         EXP_H3210, r4,   EXP_H3210,  r8, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@5,3*/ exe(OP_MSSAD,    &r25,         r15,         EXP_H3210, r5,   EXP_H3210,  r9, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@6,0*/ exe(OP_MAUH3,    &r31,         r22,         EXP_H3210, r23,  EXP_H3210, r24, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@7,0*/ exe(OP_MAUH3,    &r1,          r31,         EXP_H3210, r25,  EXP_H3210,   0, EXP_H3210, OP_SUMHL,0LL,  OP_NOP,  0LL);
 /*@8,0*/ mop(OP_LDWR, 1,  &BR[8][0][1], dp0[CHIP], cofs, MSK_D0, (Ull)dp0[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@8,0*/ exe(OP_ADD,      &AR[8][0],    BR[8][0][1], EXP_H3210, r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@8,0*/ mop(OP_STWR, 3,  &AR[8][0],    cofs, dp0[CHIP], MSK_D0, (Ull)dp0[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map1*/
 /*@9,0*/ mop(OP_LDWR, 1,  &BR[9][0][1], dp1[CHIP], cofs, MSK_D0, (Ull)dp1[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@9,0*/ exe(OP_ADD,      &AR[9][0],    BR[9][0][1], EXP_H3210, r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@9,0*/ mop(OP_STWR, 3,  &AR[9][0],    cofs, dp1[CHIP], MSK_D0, (Ull)dp1[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map2*/
 /*@10,0*/mop(OP_LDWR, 1,  &BR[10][0][1],dp2[CHIP], cofs, MSK_D0, (Ull)dp2[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@10,0*/exe(OP_ADD,      &AR[10][0],   BR[10][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@10,0*/mop(OP_STWR, 3,  &AR[10][0],   cofs, dp2[CHIP], MSK_D0, (Ull)dp2[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map3*/
 /*@11,0*/mop(OP_LDWR, 1,  &BR[11][0][1],dp3[CHIP], cofs, MSK_D0, (Ull)dp3[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@11,0*/exe(OP_ADD,      &AR[11][0],   BR[11][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@11,0*/mop(OP_STWR, 3,  &AR[11][0],   cofs, dp3[CHIP], MSK_D0, (Ull)dp3[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map4*/
 /*@12,0*/mop(OP_LDWR, 1,  &BR[12][0][1],dp4[CHIP], cofs, MSK_D0, (Ull)dp4[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@12,0*/exe(OP_ADD,      &AR[12][0],   BR[12][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@12,0*/mop(OP_STWR, 3,  &AR[12][0],   cofs, dp4[CHIP], MSK_D0, (Ull)dp4[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map5*/
 /*@13,0*/mop(OP_LDWR, 1,  &BR[13][0][1],dp5[CHIP], cofs, MSK_D0, (Ull)dp5[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@13,0*/exe(OP_ADD,      &AR[13][0],   BR[13][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@13,0*/mop(OP_STWR, 3,  &AR[13][0],   cofs, dp5[CHIP], MSK_D0, (Ull)dp5[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map6*/
 /*@14,0*/mop(OP_LDWR, 1,  &BR[14][0][1],dp6[CHIP], cofs, MSK_D0, (Ull)dp6[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@14,0*/exe(OP_ADD,      &AR[14][0],   BR[14][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@14,0*/mop(OP_STWR, 3,  &AR[14][0],   cofs, dp6[CHIP], MSK_D0, (Ull)dp6[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map7*/
 /*@15,0*/mop(OP_LDWR, 1,  &BR[15][0][1],dp7[CHIP], cofs, MSK_D0, (Ull)dp7[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@15,0*/exe(OP_ADD,      &AR[15][0],   BR[15][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@15,0*/mop(OP_STWR, 3,  &AR[15][0],   cofs, dp7[CHIP], MSK_D0, (Ull)dp7[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map8*/
 /*@16,0*/mop(OP_LDWR, 1,  &BR[16][0][1],dp8[CHIP], cofs, MSK_D0, (Ull)dp8[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@16,0*/exe(OP_ADD,      &AR[16][0],   BR[16][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@16,0*/mop(OP_STWR, 3,  &AR[16][0],   cofs, dp8[CHIP], MSK_D0, (Ull)dp8[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map9*/
 /*@17,0*/mop(OP_LDWR, 1,  &BR[17][0][1],dp9[CHIP], cofs, MSK_D0, (Ull)dp9[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@17,0*/exe(OP_ADD,      &AR[17][0],   BR[17][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@17,0*/mop(OP_STWR, 3,  &AR[17][0],   cofs, dp9[CHIP], MSK_D0, (Ull)dp9[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map10*/
 /*@18,0*/mop(OP_LDWR, 1,  &BR[18][0][1],dpa[CHIP], cofs, MSK_D0, (Ull)dpa[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@18,0*/exe(OP_ADD,      &AR[18][0],   BR[18][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@18,0*/mop(OP_STWR, 3,  &AR[18][0],   cofs, dpa[CHIP], MSK_D0, (Ull)dpa[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map11*/
 /*@19,0*/mop(OP_LDWR, 1,  &BR[19][0][1],dpb[CHIP], cofs, MSK_D0, (Ull)dpb[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@19,0*/exe(OP_ADD,      &AR[19][0],   BR[19][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@19,0*/mop(OP_STWR, 3,  &AR[19][0],   cofs, dpb[CHIP], MSK_D0, (Ull)dpb[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map12*/
 /*@20,0*/mop(OP_LDWR, 1,  &BR[20][0][1],dpc[CHIP], cofs, MSK_D0, (Ull)dpc[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@20,0*/exe(OP_ADD,      &AR[20][0],   BR[20][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@20,0*/mop(OP_STWR, 3,  &AR[20][0],   cofs, dpc[CHIP], MSK_D0, (Ull)dpc[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map13*/
 /*@21,0*/mop(OP_LDWR, 1,  &BR[21][0][1],dpd[CHIP], cofs, MSK_D0, (Ull)dpd[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@21,0*/exe(OP_ADD,      &AR[21][0],   BR[21][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@21,0*/mop(OP_STWR, 3,  &AR[21][0],   cofs, dpd[CHIP], MSK_D0, (Ull)dpd[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map14*/
 /*@22,0*/mop(OP_LDWR, 1,  &BR[22][0][1],dpe[CHIP], cofs, MSK_D0, (Ull)dpe[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@22,0*/exe(OP_ADD,      &AR[22][0],   BR[22][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@22,0*/mop(OP_STWR, 3,  &AR[22][0],   cofs, dpe[CHIP], MSK_D0, (Ull)dpe[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
          /*map15*/
 /*@23,0*/mop(OP_LDWR, 1,  &BR[23][0][1],dpf[CHIP], cofs, MSK_D0, (Ull)dpf[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
 /*@23,0*/exe(OP_ADD,      &AR[23][0],   BR[23][0][1], EXP_H3210,r1,   EXP_H3210,    0, EXP_H3210, OP_NOP,  0LL,  OP_NOP,  0LL);
 /*@23,0*/mop(OP_STWR, 3,  &AR[23][0],   cofs, dpf[CHIP], MSK_D0, (Ull)dpf[CHIP], AWD, 0, 1, (Ull)NULL,    AWD);
        }
      }
//EMAX5A end
  }
//EMAX5A drain_dirty_lmm
#endif
  /*printf("wdiflineL-end k=%d\n", k);*/
}

#ifndef ARMSIML
int readn(fd, p, len)
     int fd;
     char *p;
     int len;
{
  int n, val;
  char c;
  
  for (n=len; n>0;) {
    if ((val = read(fd, p, n)) < 0)
      return(val);
    p += val;
    n -= val;
  }
  return(len);
}

int writen(fd, p, len)
     int fd;
     char *p;
     int len;
{
  int n, val;
  
  for (n=len; n>0;) {
    if ((val = write(fd, p, n)) < 0)
      return(val);
    p += val;
    n -= val;
  }
  return(len);
}

cam_capt()
{
  read_cam(L);
  read_cam(R);
}

read_cam(Uint *fb)
{
  unsigned int buf[BITMAP];
  int i, j, retval;

  if ((camfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    fprintf(stderr, "filter: can't open stream socket\n");
    exit(1);
  }
  memset((char*)&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family      = AF_INET;
  serv_addr.sin_addr.s_addr = inet_addr(host);
  serv_addr.sin_port        = htons(port);

  if (connect(camfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    fprintf(stderr, "filter: can't connect to server\n");
    exit(1);
  }
  if ((retval = readn(camfd, buf, BITMAP*4)) != BITMAP*4) {
    fprintf(stderr, "filter: read error errno=%d\n", errno);
    exit(1);
  }
  for (i=0; i<HT; i++) {
    for (j=0; j<WD; j++)
      fb[i*WD+j] = buf[i*WD+j];
  }

  close(camfd);
}
#endif
