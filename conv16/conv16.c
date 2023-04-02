
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/* 16x16¾ö¤ß¹þ¤ß±é»»              */
/*   Copyright (C) 2013- by NAIST */
/*    Primary writer: Y.Nakashima */
/*           nakashim@is.naist.jp */

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

int WD=320, HT=240, BITMAP=320*240, SCRWD=5, SCRHT=5, VECWD=240, VECHT=240, VECSTEP=4;

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif
#if !defined(ARMSIML)
#include "./xdisp.c"
#endif

void *gather_kernel();
#define abs(a) (((a)<0)?-(a):(a))

#define MAXTHNUM 2048
#ifdef PTHREAD
#define THNUM 8
#ifndef ARMSIML
pthread_t th[MAXTHNUM];
#endif
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
Uint image_WD, image_HT, image_GRAD;
Uint image_size;
float    *ACCI; /* accelerator input */
float    *ACCO; /* accelerator output */
float    *SCON;
Ull      *DCON;

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
main(argc, argv)
     int argc;
     char **argv;
{
  FILE *fp;
  int i, j, k, fc;
#ifndef ARMSIML
  fd_set rfds;
  struct timeval tv;
#endif

  image_WD = 320;
  image_HT = 240;
  image_size=image_WD*image_HT;
  sysinit((sizeof(float)*image_size)*2+sizeof(float)*256*2,
	  32);
    
  printf("membase: %08.8x\n", (Uint)membase);
  ACCI = (float*)membase;
  ACCO = ACCI + image_size;
  SCON = ACCO + image_size;
  DCON = SCON + 256;
  printf("ACCI: %08.8x\n", ACCI);
  printf("ACCO: %08.8x\n", ACCO);
  printf("SCON: %08.8x\n", SCON);
  printf("DCON: %08.8x\n", DCON);

#if !defined(ARMSIML)
  x11_open(0);
#endif

  for (i=0; i<image_HT; i++) {
    for (j=0; j<image_WD; j++) {
      ACCI[i*image_WD+j] = i*65536 + j*256;
    }
  }
  for (i=-8; i<8; i++) {
    for (j=-8; j<8; j++) {
      SCON[(i+8)*16+(j+8)] = abs(i)*abs(j);
    }
  }
  for (i=-8; i<8; i++) {
    for (j=-8; j<8; j+=2) {
      DCON[(i+8)*16+(j+8)] = ((Ull)(*(Uint*)&SCON[(i+8)*16+(j+9)])<<32)|((Ull)(*(Uint*)&SCON[(i+8)*16+(j+8)]));
    }
  }

#ifdef ARMSIML
  _getpa();
#endif
  gather_kernel(); /* search triangle in {frontier,next} */
#ifdef ARMSIML
  _getpa();
  _copyX(0, ACCI);
  _copyX(1, ACCO);
  _copyX(2, ACCI);
  _updateX();
#endif
#if !defined(ARMSIML)
  BGR_to_X(0, ACCI);
  BGR_to_X(1, ACCO);
  BGR_to_X(2, ACCI);
  x11_update();
#endif

#if 0
  for (i=0; i<image_HT; i++) {
    for (j=0; j<image_WD; j++) {
      printf("%d %d: %f\n", i, j, ACCO[i*image_WD+j]);
    }
  }
#endif

#if !defined(ARMSIML)
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (!x11_checkevent());
#endif

  exit(0);
}

void *gather_kernel()
{
  int y;
  for (y=8; y<image_HT-8; y++)
    gather_x1(ACCI+y*image_WD, ACCO+y*image_WD);
//EMAX5A drain_dirty_lmm
//emax5_drain_dirty_lmm();
}

gather_x1(float *yi, float *yo)
{
  float *yim8 = yi-image_WD*8;
  float *yim7 = yi-image_WD*7;
  float *yim6 = yi-image_WD*6;
  float *yim5 = yi-image_WD*5;
  float *yim4 = yi-image_WD*4;
  float *yim3 = yi-image_WD*3;
  float *yim2 = yi-image_WD*2;
  float *yim1 = yi-image_WD*1;
  float *yizz = yi;
  float *yip1 = yi+image_WD*1;
  float *yip2 = yi+image_WD*2;
  float *yip3 = yi+image_WD*3;
  float *yip4 = yi+image_WD*4;
  float *yip5 = yi+image_WD*5;
  float *yip6 = yi+image_WD*6;
  float *yip7 = yi+image_WD*7;
  float *yip8 = yi+image_WD*8;
  
  double *yim80=yim8-8,*yim81=yim8-6,*yim82=yim8-4,*yim83=yim8-2,*yim84=yim8-0,*yim85=yim8+2,*yim86=yim8+4,*yim87=yim8+6;
  double *yim70=yim7-8,*yim71=yim7-6,*yim72=yim7-4,*yim73=yim7-2,*yim74=yim7-0,*yim75=yim7+2,*yim76=yim7+4,*yim77=yim7+6;
  double *yim60=yim6-8,*yim61=yim6-6,*yim62=yim6-4,*yim63=yim6-2,*yim64=yim6-0,*yim65=yim6+2,*yim66=yim6+4,*yim67=yim6+6;
  double *yim50=yim5-8,*yim51=yim5-6,*yim52=yim5-4,*yim53=yim5-2,*yim54=yim5-0,*yim55=yim5+2,*yim56=yim5+4,*yim57=yim5+6;
  double *yim40=yim4-8,*yim41=yim4-6,*yim42=yim4-4,*yim43=yim4-2,*yim44=yim4-0,*yim45=yim4+2,*yim46=yim4+4,*yim47=yim4+6;
  double *yim30=yim3-8,*yim31=yim3-6,*yim32=yim3-4,*yim33=yim3-2,*yim34=yim3-0,*yim35=yim3+2,*yim36=yim3+4,*yim37=yim3+6;
  double *yim20=yim2-8,*yim21=yim2-6,*yim22=yim2-4,*yim23=yim2-2,*yim24=yim2-0,*yim25=yim2+2,*yim26=yim2+4,*yim27=yim2+6;
  double *yim10=yim1-8,*yim11=yim1-6,*yim12=yim1-4,*yim13=yim1-2,*yim14=yim1-0,*yim15=yim1+2,*yim16=yim1+4,*yim17=yim1+6;
  double *yizz0=yizz-8,*yizz1=yizz-6,*yizz2=yizz-4,*yizz3=yizz-2,*yizz4=yizz-0,*yizz5=yizz+2,*yizz6=yizz+4,*yizz7=yizz+6;
  double *yip10=yip1-8,*yip11=yip1-6,*yip12=yip1-4,*yip13=yip1-2,*yip14=yip1-0,*yip15=yip1+2,*yip16=yip1+4,*yip17=yip1+6;
  double *yip20=yip2-8,*yip21=yip2-6,*yip22=yip2-4,*yip23=yip2-2,*yip24=yip2-0,*yip25=yip2+2,*yip26=yip2+4,*yip27=yip2+6;
  double *yip30=yip3-8,*yip31=yip3-6,*yip32=yip3-4,*yip33=yip3-2,*yip34=yip3-0,*yip35=yip3+2,*yip36=yip3+4,*yip37=yip3+6;
  double *yip40=yip4-8,*yip41=yip4-6,*yip42=yip4-4,*yip43=yip4-2,*yip44=yip4-0,*yip45=yip4+2,*yip46=yip4+4,*yip47=yip4+6;
  double *yip50=yip5-8,*yip51=yip5-6,*yip52=yip5-4,*yip53=yip5-2,*yip54=yip5-0,*yip55=yip5+2,*yip56=yip5+4,*yip57=yip5+6;
  double *yip60=yip6-8,*yip61=yip6-6,*yip62=yip6-4,*yip63=yip6-2,*yip64=yip6-0,*yip65=yip6+2,*yip66=yip6+4,*yip67=yip6+6;
  double *yip70=yip7-8,*yip71=yip7-6,*yip72=yip7-4,*yip73=yip7-2,*yip74=yip7-0,*yip75=yip7+2,*yip76=yip7+4,*yip77=yip7+6;
  double *yip80=yip8-8;
  double *yop=yo-image_WD+8;
  double *yoo=yo+8;

  Ull  loop = image_WD/2-8;
  Ull  x = 8;
#if !defined(EMAX5) && !defined(EMAX6)
  /***********************************************/
  /* non EMAX5                                   */
  /***********************************************/
  while (loop--) {
    yo[x  ] = yim8[x-8]*SCON[  0]+yim8[x-7]*SCON[  1]+yim8[x-6]*SCON[  2]+yim8[x-5]*SCON[  3]+yim8[x-4]*SCON[  4]+yim8[x-3]*SCON[  5]+yim8[x-2]*SCON[  6]+yim8[x-1]*SCON[  7]
            + yim8[x+0]*SCON[  8]+yim8[x+1]*SCON[  9]+yim8[x+2]*SCON[ 10]+yim8[x+3]*SCON[ 11]+yim8[x+4]*SCON[ 12]+yim8[x+5]*SCON[ 13]+yim8[x+6]*SCON[ 14]+yim8[x+7]*SCON[ 15]
            + yim7[x-8]*SCON[ 16]+yim7[x-7]*SCON[ 17]+yim7[x-6]*SCON[ 18]+yim7[x-5]*SCON[ 19]+yim7[x-4]*SCON[ 20]+yim7[x-3]*SCON[ 21]+yim7[x-2]*SCON[ 22]+yim7[x-1]*SCON[ 23]
            + yim7[x+0]*SCON[ 24]+yim7[x+1]*SCON[ 25]+yim7[x+2]*SCON[ 26]+yim7[x+3]*SCON[ 27]+yim7[x+4]*SCON[ 28]+yim7[x+5]*SCON[ 29]+yim7[x+6]*SCON[ 30]+yim7[x+7]*SCON[ 31]
            + yim6[x-8]*SCON[ 32]+yim6[x-7]*SCON[ 33]+yim6[x-6]*SCON[ 34]+yim6[x-5]*SCON[ 35]+yim6[x-4]*SCON[ 36]+yim6[x-3]*SCON[ 37]+yim6[x-2]*SCON[ 38]+yim6[x-1]*SCON[ 39]
            + yim6[x+0]*SCON[ 40]+yim6[x+1]*SCON[ 41]+yim6[x+2]*SCON[ 42]+yim6[x+3]*SCON[ 43]+yim6[x+4]*SCON[ 44]+yim6[x+5]*SCON[ 45]+yim6[x+6]*SCON[ 46]+yim6[x+7]*SCON[ 47]
            + yim5[x-8]*SCON[ 48]+yim5[x-7]*SCON[ 49]+yim5[x-6]*SCON[ 50]+yim5[x-5]*SCON[ 51]+yim5[x-4]*SCON[ 52]+yim5[x-3]*SCON[ 53]+yim5[x-2]*SCON[ 54]+yim5[x-1]*SCON[ 55]
            + yim5[x+0]*SCON[ 56]+yim5[x+1]*SCON[ 57]+yim5[x+2]*SCON[ 58]+yim5[x+3]*SCON[ 59]+yim5[x+4]*SCON[ 60]+yim5[x+5]*SCON[ 61]+yim5[x+6]*SCON[ 62]+yim5[x+7]*SCON[ 63]
            + yim4[x-8]*SCON[ 64]+yim4[x-7]*SCON[ 65]+yim4[x-6]*SCON[ 66]+yim4[x-5]*SCON[ 67]+yim4[x-4]*SCON[ 68]+yim4[x-3]*SCON[ 69]+yim4[x-2]*SCON[ 70]+yim4[x-1]*SCON[ 71]
            + yim4[x+0]*SCON[ 72]+yim4[x+1]*SCON[ 73]+yim4[x+2]*SCON[ 74]+yim4[x+3]*SCON[ 75]+yim4[x+4]*SCON[ 76]+yim4[x+5]*SCON[ 77]+yim4[x+6]*SCON[ 78]+yim4[x+7]*SCON[ 79]
            + yim3[x-8]*SCON[ 80]+yim3[x-7]*SCON[ 81]+yim3[x-6]*SCON[ 82]+yim3[x-5]*SCON[ 83]+yim3[x-4]*SCON[ 84]+yim3[x-3]*SCON[ 85]+yim3[x-2]*SCON[ 86]+yim3[x-1]*SCON[ 87]
            + yim3[x+0]*SCON[ 88]+yim3[x+1]*SCON[ 89]+yim3[x+2]*SCON[ 90]+yim3[x+3]*SCON[ 91]+yim3[x+4]*SCON[ 92]+yim3[x+5]*SCON[ 93]+yim3[x+6]*SCON[ 94]+yim3[x+7]*SCON[ 95]
            + yim2[x-8]*SCON[ 96]+yim2[x-7]*SCON[ 97]+yim2[x-6]*SCON[ 98]+yim2[x-5]*SCON[ 99]+yim2[x-4]*SCON[100]+yim2[x-3]*SCON[101]+yim2[x-2]*SCON[102]+yim2[x-1]*SCON[103]
            + yim2[x+0]*SCON[104]+yim2[x+1]*SCON[105]+yim2[x+2]*SCON[106]+yim2[x+3]*SCON[107]+yim2[x+4]*SCON[108]+yim2[x+5]*SCON[109]+yim2[x+6]*SCON[110]+yim2[x+7]*SCON[111]
            + yim1[x-8]*SCON[112]+yim1[x-7]*SCON[113]+yim1[x-6]*SCON[114]+yim1[x-5]*SCON[115]+yim1[x-4]*SCON[116]+yim1[x-3]*SCON[117]+yim1[x-2]*SCON[118]+yim1[x-1]*SCON[119]
            + yim1[x+0]*SCON[120]+yim1[x+1]*SCON[121]+yim1[x+2]*SCON[122]+yim1[x+3]*SCON[123]+yim1[x+4]*SCON[124]+yim1[x+5]*SCON[125]+yim1[x+6]*SCON[126]+yim1[x+7]*SCON[127]
            + yizz[x-8]*SCON[128]+yizz[x-7]*SCON[129]+yizz[x-6]*SCON[130]+yizz[x-5]*SCON[131]+yizz[x-4]*SCON[132]+yizz[x-3]*SCON[133]+yizz[x-2]*SCON[134]+yizz[x-1]*SCON[135]
            + yizz[x+0]*SCON[136]+yizz[x+1]*SCON[137]+yizz[x+2]*SCON[138]+yizz[x+3]*SCON[139]+yizz[x+4]*SCON[140]+yizz[x+5]*SCON[141]+yizz[x+6]*SCON[142]+yizz[x+7]*SCON[143]
            + yip1[x-8]*SCON[144]+yip1[x-7]*SCON[145]+yip1[x-6]*SCON[146]+yip1[x-5]*SCON[147]+yip1[x-4]*SCON[148]+yip1[x-3]*SCON[149]+yip1[x-2]*SCON[150]+yip1[x-1]*SCON[151]
            + yip1[x+0]*SCON[152]+yip1[x+1]*SCON[153]+yip1[x+2]*SCON[154]+yip1[x+3]*SCON[155]+yip1[x+4]*SCON[156]+yip1[x+5]*SCON[157]+yip1[x+6]*SCON[158]+yip1[x+7]*SCON[159]
            + yip2[x-8]*SCON[160]+yip2[x-7]*SCON[161]+yip2[x-6]*SCON[162]+yip2[x-5]*SCON[163]+yip2[x-4]*SCON[164]+yip2[x-3]*SCON[165]+yip2[x-2]*SCON[166]+yip2[x-1]*SCON[167]
            + yip2[x+0]*SCON[168]+yip2[x+1]*SCON[169]+yip2[x+2]*SCON[170]+yip2[x+3]*SCON[171]+yip2[x+4]*SCON[172]+yip2[x+5]*SCON[173]+yip2[x+6]*SCON[174]+yip2[x+7]*SCON[175]
            + yip3[x-8]*SCON[176]+yip3[x-7]*SCON[177]+yip3[x-6]*SCON[178]+yip3[x-5]*SCON[179]+yip3[x-4]*SCON[180]+yip3[x-3]*SCON[181]+yip3[x-2]*SCON[182]+yip3[x-1]*SCON[183]
            + yip3[x+0]*SCON[184]+yip3[x+1]*SCON[185]+yip3[x+2]*SCON[186]+yip3[x+3]*SCON[187]+yip3[x+4]*SCON[188]+yip3[x+5]*SCON[189]+yip3[x+6]*SCON[190]+yip3[x+7]*SCON[191]
            + yip4[x-8]*SCON[192]+yip4[x-7]*SCON[193]+yip4[x-6]*SCON[194]+yip4[x-5]*SCON[195]+yip4[x-4]*SCON[196]+yip4[x-3]*SCON[197]+yip4[x-2]*SCON[198]+yip4[x-1]*SCON[199]
            + yip4[x+0]*SCON[200]+yip4[x+1]*SCON[201]+yip4[x+2]*SCON[202]+yip4[x+3]*SCON[203]+yip4[x+4]*SCON[204]+yip4[x+5]*SCON[205]+yip4[x+6]*SCON[206]+yip4[x+7]*SCON[207]
            + yip5[x-8]*SCON[208]+yip5[x-7]*SCON[209]+yip5[x-6]*SCON[210]+yip5[x-5]*SCON[211]+yip5[x-4]*SCON[212]+yip5[x-3]*SCON[213]+yip5[x-2]*SCON[214]+yip5[x-1]*SCON[215]
            + yip5[x+0]*SCON[216]+yip5[x+1]*SCON[217]+yip5[x+2]*SCON[218]+yip5[x+3]*SCON[219]+yip5[x+4]*SCON[220]+yip5[x+5]*SCON[221]+yip5[x+6]*SCON[222]+yip5[x+7]*SCON[223]
            + yip6[x-8]*SCON[224]+yip6[x-7]*SCON[225]+yip6[x-6]*SCON[226]+yip6[x-5]*SCON[227]+yip6[x-4]*SCON[228]+yip6[x-3]*SCON[229]+yip6[x-2]*SCON[230]+yip6[x-1]*SCON[231]
            + yip6[x+0]*SCON[232]+yip6[x+1]*SCON[233]+yip6[x+2]*SCON[234]+yip6[x+3]*SCON[235]+yip6[x+4]*SCON[236]+yip6[x+5]*SCON[237]+yip6[x+6]*SCON[238]+yip6[x+7]*SCON[239]
            + yip7[x-8]*SCON[240]+yip7[x-7]*SCON[241]+yip7[x-6]*SCON[242]+yip7[x-5]*SCON[243]+yip7[x-4]*SCON[244]+yip7[x-3]*SCON[245]+yip7[x-2]*SCON[246]+yip7[x-1]*SCON[247]
            + yip7[x+0]*SCON[248]+yip7[x+1]*SCON[249]+yip7[x+2]*SCON[250]+yip7[x+3]*SCON[251]+yip7[x+4]*SCON[252]+yip7[x+5]*SCON[253]+yip7[x+6]*SCON[254]+yip7[x+7]*SCON[255];
    x += 2;
  }
#else
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull c000=DCON[  0], c002=DCON[  1], c004=DCON[  2], c006=DCON[  3], c008=DCON[  4], c010=DCON[  5], c012=DCON[  6], c014=DCON[  7];
  Ull c016=DCON[  8], c018=DCON[  9], c020=DCON[ 10], c022=DCON[ 11], c024=DCON[ 12], c026=DCON[ 13], c028=DCON[ 14], c030=DCON[ 15];
  Ull c032=DCON[ 16], c034=DCON[ 17], c036=DCON[ 18], c038=DCON[ 19], c040=DCON[ 20], c042=DCON[ 21], c044=DCON[ 22], c046=DCON[ 23];
  Ull c048=DCON[ 24], c050=DCON[ 25], c052=DCON[ 26], c054=DCON[ 27], c056=DCON[ 28], c058=DCON[ 29], c060=DCON[ 30], c062=DCON[ 31];
  Ull c064=DCON[ 32], c066=DCON[ 33], c068=DCON[ 34], c070=DCON[ 35], c072=DCON[ 36], c074=DCON[ 37], c076=DCON[ 38], c078=DCON[ 39];
  Ull c080=DCON[ 40], c082=DCON[ 41], c084=DCON[ 42], c086=DCON[ 43], c088=DCON[ 44], c090=DCON[ 45], c092=DCON[ 46], c094=DCON[ 47];
  Ull c096=DCON[ 48], c098=DCON[ 49], c100=DCON[ 50], c102=DCON[ 51], c104=DCON[ 52], c106=DCON[ 53], c108=DCON[ 54], c110=DCON[ 55];
  Ull c112=DCON[ 56], c114=DCON[ 57], c116=DCON[ 58], c118=DCON[ 59], c120=DCON[ 60], c122=DCON[ 61], c124=DCON[ 62], c126=DCON[ 63];
  Ull c128=DCON[ 64], c130=DCON[ 65], c132=DCON[ 66], c134=DCON[ 67], c136=DCON[ 68], c138=DCON[ 69], c140=DCON[ 70], c142=DCON[ 71];
  Ull c144=DCON[ 72], c146=DCON[ 73], c148=DCON[ 74], c150=DCON[ 75], c152=DCON[ 76], c154=DCON[ 77], c156=DCON[ 78], c158=DCON[ 79];
  Ull c160=DCON[ 80], c162=DCON[ 81], c164=DCON[ 82], c166=DCON[ 83], c168=DCON[ 84], c170=DCON[ 85], c172=DCON[ 86], c174=DCON[ 87];
  Ull c176=DCON[ 88], c178=DCON[ 89], c180=DCON[ 90], c182=DCON[ 91], c184=DCON[ 92], c186=DCON[ 93], c188=DCON[ 94], c190=DCON[ 95];
  Ull c192=DCON[ 96], c194=DCON[ 97], c196=DCON[ 98], c198=DCON[ 99], c200=DCON[100], c202=DCON[101], c204=DCON[102], c206=DCON[103];
  Ull c208=DCON[104], c210=DCON[105], c212=DCON[106], c214=DCON[107], c216=DCON[108], c218=DCON[109], c220=DCON[110], c222=DCON[111];
  Ull c224=DCON[112], c226=DCON[113], c228=DCON[114], c230=DCON[115], c232=DCON[116], c234=DCON[117], c236=DCON[118], c238=DCON[119];
  Ull c240=DCON[120], c242=DCON[121], c244=DCON[122], c246=DCON[123], c248=DCON[124], c250=DCON[125], c252=DCON[126], c254=DCON[127];
//EMAX5A begin x1 mapdist=2
  while (loop--) {                                  /* mapped to WHILE() on BR[15][0][0] stage#0 */
    mop(OP_LDR,  3, &BR[0][0][1], yim80++, 0, MSK_D0, yim80, 320, 0, 0, (Ull)NULL, 320); /* stage#0 */
    mop(OP_LDR,  3, &r1,          yim81++, 0, MSK_D0, yim80, 320, 0, 0, (Ull)NULL, 320); /* stage#0 */
    mop(OP_LDR,  3, &r2,          yim82++, 0, MSK_D0, yim80, 320, 0, 0, (Ull)NULL, 320); /* stage#0 */
    mop(OP_LDR,  3, &r3,          yim83++, 0, MSK_D0, yim80, 320, 0, 0, (Ull)NULL, 320); /* stage#0 */
    mop(OP_LDR,  3, &r4,          yim84++, 0, MSK_D0, yim80, 320, 0, 0, (Ull)NULL, 320); /* stage#0 */
    mop(OP_LDR,  3, &r5,          yim85++, 0, MSK_D0, yim80, 320, 0, 0, (Ull)NULL, 320); /* stage#0 */
    mop(OP_LDR,  3, &r6,          yim86++, 0, MSK_D0, yim80, 320, 0, 0, (Ull)NULL, 320); /* stage#0 */
    mop(OP_LDR,  3, &r7,          yim87++, 0, MSK_D0, yim80, 320, 0, 0, (Ull)NULL, 320); /* stage#0 */
    exe(OP_FMA, &r10, 0LL, EXP_H3210, BR[0][0][1], EXP_H3210, c000, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#1 */
    exe(OP_FMA, &r11, 0LL, EXP_H3210, r1,          EXP_H3210, c002, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#1 */
    exe(OP_FMA, &r12, 0LL, EXP_H3210, r2,          EXP_H3210, c004, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#1 */
    exe(OP_FMA, &r13, 0LL, EXP_H3210, r3,          EXP_H3210, c006, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#1 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c008, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c010, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c012, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c014, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
    mop(OP_LDR,  3, &BR[2][0][1], yim70++, 0, MSK_D0, yim70, 320, 0, 0, (Ull)NULL, 320); /* stage#2 */
    mop(OP_LDR,  3, &r1,          yim71++, 0, MSK_D0, yim70, 320, 0, 0, (Ull)NULL, 320); /* stage#2 */
    mop(OP_LDR,  3, &r2,          yim72++, 0, MSK_D0, yim70, 320, 0, 0, (Ull)NULL, 320); /* stage#2 */
    mop(OP_LDR,  3, &r3,          yim73++, 0, MSK_D0, yim70, 320, 0, 0, (Ull)NULL, 320); /* stage#2 */
    mop(OP_LDR,  3, &r4,          yim74++, 0, MSK_D0, yim70, 320, 0, 0, (Ull)NULL, 320); /* stage#2 */
    mop(OP_LDR,  3, &r5,          yim75++, 0, MSK_D0, yim70, 320, 0, 0, (Ull)NULL, 320); /* stage#2 */
    mop(OP_LDR,  3, &r6,          yim76++, 0, MSK_D0, yim70, 320, 0, 0, (Ull)NULL, 320); /* stage#2 */
    mop(OP_LDR,  3, &r7,          yim77++, 0, MSK_D0, yim70, 320, 0, 0, (Ull)NULL, 320); /* stage#2 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[2][0][1], EXP_H3210, c016, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c018, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c020, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c022, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c024, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c026, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c028, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c030, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
    mop(OP_LDR,  3, &BR[4][0][1], yim60++, 0, MSK_D0, yim60, 320, 0, 0, (Ull)NULL, 320); /* stage#4 */
    mop(OP_LDR,  3, &r1,          yim61++, 0, MSK_D0, yim60, 320, 0, 0, (Ull)NULL, 320); /* stage#4 */
    mop(OP_LDR,  3, &r2,          yim62++, 0, MSK_D0, yim60, 320, 0, 0, (Ull)NULL, 320); /* stage#4 */
    mop(OP_LDR,  3, &r3,          yim63++, 0, MSK_D0, yim60, 320, 0, 0, (Ull)NULL, 320); /* stage#4 */
    mop(OP_LDR,  3, &r4,          yim64++, 0, MSK_D0, yim60, 320, 0, 0, (Ull)NULL, 320); /* stage#4 */
    mop(OP_LDR,  3, &r5,          yim65++, 0, MSK_D0, yim60, 320, 0, 0, (Ull)NULL, 320); /* stage#4 */
    mop(OP_LDR,  3, &r6,          yim66++, 0, MSK_D0, yim60, 320, 0, 0, (Ull)NULL, 320); /* stage#4 */
    mop(OP_LDR,  3, &r7,          yim67++, 0, MSK_D0, yim60, 320, 0, 0, (Ull)NULL, 320); /* stage#4 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[4][0][1], EXP_H3210, c032, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c034, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c036, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c038, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c040, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c042, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c044, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c046, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
    mop(OP_LDR,  3, &BR[6][0][1], yim50++, 0, MSK_D0, yim50, 320, 0, 0, (Ull)NULL, 320); /* stage#6 */
    mop(OP_LDR,  3, &r1,          yim51++, 0, MSK_D0, yim50, 320, 0, 0, (Ull)NULL, 320); /* stage#6 */
    mop(OP_LDR,  3, &r2,          yim52++, 0, MSK_D0, yim50, 320, 0, 0, (Ull)NULL, 320); /* stage#6 */
    mop(OP_LDR,  3, &r3,          yim53++, 0, MSK_D0, yim50, 320, 0, 0, (Ull)NULL, 320); /* stage#6 */
    mop(OP_LDR,  3, &r4,          yim54++, 0, MSK_D0, yim50, 320, 0, 0, (Ull)NULL, 320); /* stage#6 */
    mop(OP_LDR,  3, &r5,          yim55++, 0, MSK_D0, yim50, 320, 0, 0, (Ull)NULL, 320); /* stage#6 */
    mop(OP_LDR,  3, &r6,          yim56++, 0, MSK_D0, yim50, 320, 0, 0, (Ull)NULL, 320); /* stage#6 */
    mop(OP_LDR,  3, &r7,          yim57++, 0, MSK_D0, yim50, 320, 0, 0, (Ull)NULL, 320); /* stage#6 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[6][0][1], EXP_H3210, c048, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c050, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c052, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c054, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c056, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c058, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c060, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c062, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
    mop(OP_LDR,  3, &BR[8][0][1], yim40++, 0, MSK_D0, yim40, 320, 0, 0, (Ull)NULL, 320); /* stage#8 */
    mop(OP_LDR,  3, &r1,          yim41++, 0, MSK_D0, yim40, 320, 0, 0, (Ull)NULL, 320); /* stage#8 */
    mop(OP_LDR,  3, &r2,          yim42++, 0, MSK_D0, yim40, 320, 0, 0, (Ull)NULL, 320); /* stage#8 */
    mop(OP_LDR,  3, &r3,          yim43++, 0, MSK_D0, yim40, 320, 0, 0, (Ull)NULL, 320); /* stage#8 */
    mop(OP_LDR,  3, &r4,          yim44++, 0, MSK_D0, yim40, 320, 0, 0, (Ull)NULL, 320); /* stage#8 */
    mop(OP_LDR,  3, &r5,          yim45++, 0, MSK_D0, yim40, 320, 0, 0, (Ull)NULL, 320); /* stage#8 */
    mop(OP_LDR,  3, &r6,          yim46++, 0, MSK_D0, yim40, 320, 0, 0, (Ull)NULL, 320); /* stage#8 */
    mop(OP_LDR,  3, &r7,          yim47++, 0, MSK_D0, yim40, 320, 0, 0, (Ull)NULL, 320); /* stage#8 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[8][0][1], EXP_H3210, c064, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c066, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c068, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c070, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c072, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c074, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c076, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c078, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
    mop(OP_LDR,  3, &BR[10][0][1],yim30++, 0, MSK_D0, yim30, 320, 0, 0, (Ull)NULL, 320); /* stage#10 */
    mop(OP_LDR,  3, &r1,          yim31++, 0, MSK_D0, yim30, 320, 0, 0, (Ull)NULL, 320); /* stage#10 */
    mop(OP_LDR,  3, &r2,          yim32++, 0, MSK_D0, yim30, 320, 0, 0, (Ull)NULL, 320); /* stage#10 */
    mop(OP_LDR,  3, &r3,          yim33++, 0, MSK_D0, yim30, 320, 0, 0, (Ull)NULL, 320); /* stage#10 */
    mop(OP_LDR,  3, &r4,          yim34++, 0, MSK_D0, yim30, 320, 0, 0, (Ull)NULL, 320); /* stage#10 */
    mop(OP_LDR,  3, &r5,          yim35++, 0, MSK_D0, yim30, 320, 0, 0, (Ull)NULL, 320); /* stage#10 */
    mop(OP_LDR,  3, &r6,          yim36++, 0, MSK_D0, yim30, 320, 0, 0, (Ull)NULL, 320); /* stage#10 */
    mop(OP_LDR,  3, &r7,          yim37++, 0, MSK_D0, yim30, 320, 0, 0, (Ull)NULL, 320); /* stage#10 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[10][0][1],EXP_H3210, c080, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c082, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c084, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c086, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c088, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c090, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c092, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c094, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    mop(OP_LDR,  3, &BR[12][0][1],yim20++, 0, MSK_D0, yim20, 320, 0, 0, (Ull)NULL, 320); /* stage#12 */
    mop(OP_LDR,  3, &r1,          yim21++, 0, MSK_D0, yim20, 320, 0, 0, (Ull)NULL, 320); /* stage#12 */
    mop(OP_LDR,  3, &r2,          yim22++, 0, MSK_D0, yim20, 320, 0, 0, (Ull)NULL, 320); /* stage#12 */
    mop(OP_LDR,  3, &r3,          yim23++, 0, MSK_D0, yim20, 320, 0, 0, (Ull)NULL, 320); /* stage#12 */
    mop(OP_LDR,  3, &r4,          yim24++, 0, MSK_D0, yim20, 320, 0, 0, (Ull)NULL, 320); /* stage#12 */
    mop(OP_LDR,  3, &r5,          yim25++, 0, MSK_D0, yim20, 320, 0, 0, (Ull)NULL, 320); /* stage#12 */
    mop(OP_LDR,  3, &r6,          yim26++, 0, MSK_D0, yim20, 320, 0, 0, (Ull)NULL, 320); /* stage#12 */
    mop(OP_LDR,  3, &r7,          yim27++, 0, MSK_D0, yim20, 320, 0, 0, (Ull)NULL, 320); /* stage#12 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[12][0][1],EXP_H3210, c096, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c098, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c100, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c102, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c104, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c106, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c108, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c110, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
    mop(OP_LDR,  3, &BR[14][0][1],yim10++, 0, MSK_D0, yim10, 320, 0, 0, (Ull)NULL, 320); /* stage#14 */
    mop(OP_LDR,  3, &r1,          yim11++, 0, MSK_D0, yim10, 320, 0, 0, (Ull)NULL, 320); /* stage#14 */
    mop(OP_LDR,  3, &r2,          yim12++, 0, MSK_D0, yim10, 320, 0, 0, (Ull)NULL, 320); /* stage#14 */
    mop(OP_LDR,  3, &r3,          yim13++, 0, MSK_D0, yim10, 320, 0, 0, (Ull)NULL, 320); /* stage#14 */
    mop(OP_LDR,  3, &r4,          yim14++, 0, MSK_D0, yim10, 320, 0, 0, (Ull)NULL, 320); /* stage#14 */
    mop(OP_LDR,  3, &r5,          yim15++, 0, MSK_D0, yim10, 320, 0, 0, (Ull)NULL, 320); /* stage#14 */
    mop(OP_LDR,  3, &r6,          yim16++, 0, MSK_D0, yim10, 320, 0, 0, (Ull)NULL, 320); /* stage#14 */
    mop(OP_LDR,  3, &r7,          yim17++, 0, MSK_D0, yim10, 320, 0, 0, (Ull)NULL, 320); /* stage#14 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[14][0][1],EXP_H3210, c112, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c114, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c116, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c118, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c120, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c122, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c124, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c126, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
    mop(OP_LDR,  3, &BR[16][0][1],yizz0++, 0, MSK_D0, yizz0, 320, 0, 0, (Ull)NULL, 320); /* stage#16 */
    mop(OP_LDR,  3, &r1,          yizz1++, 0, MSK_D0, yizz0, 320, 0, 0, (Ull)NULL, 320); /* stage#16 */
    mop(OP_LDR,  3, &r2,          yizz2++, 0, MSK_D0, yizz0, 320, 0, 0, (Ull)NULL, 320); /* stage#16 */
    mop(OP_LDR,  3, &r3,          yizz3++, 0, MSK_D0, yizz0, 320, 0, 0, (Ull)NULL, 320); /* stage#16 */
    mop(OP_LDR,  3, &r4,          yizz4++, 0, MSK_D0, yizz0, 320, 0, 0, (Ull)NULL, 320); /* stage#16 */
    mop(OP_LDR,  3, &r5,          yizz5++, 0, MSK_D0, yizz0, 320, 0, 0, (Ull)NULL, 320); /* stage#16 */
    mop(OP_LDR,  3, &r6,          yizz6++, 0, MSK_D0, yizz0, 320, 0, 0, (Ull)NULL, 320); /* stage#16 */
    mop(OP_LDR,  3, &r7,          yizz7++, 0, MSK_D0, yizz0, 320, 0, 0, (Ull)NULL, 320); /* stage#16 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[16][0][1],EXP_H3210, c128, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c130, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c132, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c134, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c136, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c138, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c140, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c142, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
    mop(OP_LDR,  3, &BR[18][0][1],yip10++, 0, MSK_D0, yip10, 320, 0, 0, (Ull)NULL, 320); /* stage#18 */
    mop(OP_LDR,  3, &r1,          yip11++, 0, MSK_D0, yip10, 320, 0, 0, (Ull)NULL, 320); /* stage#18 */
    mop(OP_LDR,  3, &r2,          yip12++, 0, MSK_D0, yip10, 320, 0, 0, (Ull)NULL, 320); /* stage#18 */
    mop(OP_LDR,  3, &r3,          yip13++, 0, MSK_D0, yip10, 320, 0, 0, (Ull)NULL, 320); /* stage#18 */
    mop(OP_LDR,  3, &r4,          yip14++, 0, MSK_D0, yip10, 320, 0, 0, (Ull)NULL, 320); /* stage#18 */
    mop(OP_LDR,  3, &r5,          yip15++, 0, MSK_D0, yip10, 320, 0, 0, (Ull)NULL, 320); /* stage#18 */
    mop(OP_LDR,  3, &r6,          yip16++, 0, MSK_D0, yip10, 320, 0, 0, (Ull)NULL, 320); /* stage#18 */
    mop(OP_LDR,  3, &r7,          yip17++, 0, MSK_D0, yip10, 320, 0, 0, (Ull)NULL, 320); /* stage#18 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[18][0][1],EXP_H3210, c144, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c146, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c148, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c150, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c152, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c154, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c156, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c158, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
    mop(OP_LDR,  3, &BR[20][0][1],yip20++, 0, MSK_D0, yip20, 320, 0, 0, (Ull)NULL, 320); /* stage#20 */
    mop(OP_LDR,  3, &r1,          yip21++, 0, MSK_D0, yip20, 320, 0, 0, (Ull)NULL, 320); /* stage#20 */
    mop(OP_LDR,  3, &r2,          yip22++, 0, MSK_D0, yip20, 320, 0, 0, (Ull)NULL, 320); /* stage#20 */
    mop(OP_LDR,  3, &r3,          yip23++, 0, MSK_D0, yip20, 320, 0, 0, (Ull)NULL, 320); /* stage#20 */
    mop(OP_LDR,  3, &r4,          yip24++, 0, MSK_D0, yip20, 320, 0, 0, (Ull)NULL, 320); /* stage#20 */
    mop(OP_LDR,  3, &r5,          yip25++, 0, MSK_D0, yip20, 320, 0, 0, (Ull)NULL, 320); /* stage#20 */
    mop(OP_LDR,  3, &r6,          yip26++, 0, MSK_D0, yip20, 320, 0, 0, (Ull)NULL, 320); /* stage#20 */
    mop(OP_LDR,  3, &r7,          yip27++, 0, MSK_D0, yip20, 320, 0, 0, (Ull)NULL, 320); /* stage#20 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[20][0][1],EXP_H3210, c160, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c162, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c164, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c166, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c168, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c170, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c172, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c174, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
    mop(OP_LDR,  3, &BR[22][0][1],yip30++, 0, MSK_D0, yip30, 320, 0, 0, (Ull)NULL, 320); /* stage#22 */
    mop(OP_LDR,  3, &r1,          yip31++, 0, MSK_D0, yip30, 320, 0, 0, (Ull)NULL, 320); /* stage#22 */
    mop(OP_LDR,  3, &r2,          yip32++, 0, MSK_D0, yip30, 320, 0, 0, (Ull)NULL, 320); /* stage#22 */
    mop(OP_LDR,  3, &r3,          yip33++, 0, MSK_D0, yip30, 320, 0, 0, (Ull)NULL, 320); /* stage#22 */
    mop(OP_LDR,  3, &r4,          yip34++, 0, MSK_D0, yip30, 320, 0, 0, (Ull)NULL, 320); /* stage#22 */
    mop(OP_LDR,  3, &r5,          yip35++, 0, MSK_D0, yip30, 320, 0, 0, (Ull)NULL, 320); /* stage#22 */
    mop(OP_LDR,  3, &r6,          yip36++, 0, MSK_D0, yip30, 320, 0, 0, (Ull)NULL, 320); /* stage#22 */
    mop(OP_LDR,  3, &r7,          yip37++, 0, MSK_D0, yip30, 320, 0, 0, (Ull)NULL, 320); /* stage#22 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[22][0][1],EXP_H3210, c176, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c178, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c180, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c182, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c184, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c186, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c188, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c190, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
    mop(OP_LDR,  3, &BR[24][0][1],yip40++, 0, MSK_D0, yip40, 320, 0, 0, (Ull)NULL, 320); /* stage#24 */
    mop(OP_LDR,  3, &r1,          yip41++, 0, MSK_D0, yip40, 320, 0, 0, (Ull)NULL, 320); /* stage#24 */
    mop(OP_LDR,  3, &r2,          yip42++, 0, MSK_D0, yip40, 320, 0, 0, (Ull)NULL, 320); /* stage#24 */
    mop(OP_LDR,  3, &r3,          yip43++, 0, MSK_D0, yip40, 320, 0, 0, (Ull)NULL, 320); /* stage#24 */
    mop(OP_LDR,  3, &r4,          yip44++, 0, MSK_D0, yip40, 320, 0, 0, (Ull)NULL, 320); /* stage#24 */
    mop(OP_LDR,  3, &r5,          yip45++, 0, MSK_D0, yip40, 320, 0, 0, (Ull)NULL, 320); /* stage#24 */
    mop(OP_LDR,  3, &r6,          yip46++, 0, MSK_D0, yip40, 320, 0, 0, (Ull)NULL, 320); /* stage#24 */
    mop(OP_LDR,  3, &r7,          yip47++, 0, MSK_D0, yip40, 320, 0, 0, (Ull)NULL, 320); /* stage#24 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[24][0][1],EXP_H3210, c192, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c194, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c196, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c198, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c200, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c202, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c204, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c206, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
    mop(OP_LDR,  3, &BR[26][0][1],yip50++, 0, MSK_D0, yip50, 320, 0, 0, (Ull)NULL, 320); /* stage#26 */
    mop(OP_LDR,  3, &r1,          yip51++, 0, MSK_D0, yip50, 320, 0, 0, (Ull)NULL, 320); /* stage#26 */
    mop(OP_LDR,  3, &r2,          yip52++, 0, MSK_D0, yip50, 320, 0, 0, (Ull)NULL, 320); /* stage#26 */
    mop(OP_LDR,  3, &r3,          yip53++, 0, MSK_D0, yip50, 320, 0, 0, (Ull)NULL, 320); /* stage#26 */
    mop(OP_LDR,  3, &r4,          yip54++, 0, MSK_D0, yip50, 320, 0, 0, (Ull)NULL, 320); /* stage#26 */
    mop(OP_LDR,  3, &r5,          yip55++, 0, MSK_D0, yip50, 320, 0, 0, (Ull)NULL, 320); /* stage#26 */
    mop(OP_LDR,  3, &r6,          yip56++, 0, MSK_D0, yip50, 320, 0, 0, (Ull)NULL, 320); /* stage#26 */
    mop(OP_LDR,  3, &r7,          yip57++, 0, MSK_D0, yip50, 320, 0, 0, (Ull)NULL, 320); /* stage#26 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[26][0][1],EXP_H3210, c208, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c210, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c212, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c214, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c216, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c218, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c220, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c222, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
    mop(OP_LDR,  3, &BR[28][0][1],yip60++, 0, MSK_D0, yip60, 320, 0, 0, (Ull)NULL, 320); /* stage#28 */
    mop(OP_LDR,  3, &r1,          yip61++, 0, MSK_D0, yip60, 320, 0, 0, (Ull)NULL, 320); /* stage#28 */
    mop(OP_LDR,  3, &r2,          yip62++, 0, MSK_D0, yip60, 320, 0, 0, (Ull)NULL, 320); /* stage#28 */
    mop(OP_LDR,  3, &r3,          yip63++, 0, MSK_D0, yip60, 320, 0, 0, (Ull)NULL, 320); /* stage#28 */
    mop(OP_LDR,  3, &r4,          yip64++, 0, MSK_D0, yip60, 320, 0, 0, (Ull)NULL, 320); /* stage#28 */
    mop(OP_LDR,  3, &r5,          yip65++, 0, MSK_D0, yip60, 320, 0, 0, (Ull)NULL, 320); /* stage#28 */
    mop(OP_LDR,  3, &r6,          yip66++, 0, MSK_D0, yip60, 320, 0, 0, (Ull)NULL, 320); /* stage#28 */
    mop(OP_LDR,  3, &r7,          yip67++, 0, MSK_D0, yip60, 320, 0, 0, (Ull)NULL, 320); /* stage#28 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[28][0][1],EXP_H3210, c224, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c226, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c228, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c230, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c232, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c234, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c236, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c238, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
    mop(OP_LDR,  3, &BR[30][0][1],yip70++, 0, MSK_D0, yip70, 320, 0, 0, yip80, 320); /* stage#30 */
    mop(OP_LDR,  3, &r1,          yip71++, 0, MSK_D0, yip70, 320, 0, 0, yip80, 320); /* stage#30 */
    mop(OP_LDR,  3, &r2,          yip72++, 0, MSK_D0, yip70, 320, 0, 0, yip80, 320); /* stage#30 */
    mop(OP_LDR,  3, &r3,          yip73++, 0, MSK_D0, yip70, 320, 0, 0, yip80, 320); /* stage#30 */
    mop(OP_LDR,  3, &r4,          yip74++, 0, MSK_D0, yip70, 320, 0, 0, yip80, 320); /* stage#30 */
    mop(OP_LDR,  3, &r5,          yip75++, 0, MSK_D0, yip70, 320, 0, 0, yip80, 320); /* stage#30 */
    mop(OP_LDR,  3, &r6,          yip76++, 0, MSK_D0, yip70, 320, 0, 0, yip80, 320); /* stage#30 */
    mop(OP_LDR,  3, &r7,          yip77++, 0, MSK_D0, yip70, 320, 0, 0, yip80, 320); /* stage#30 */
    exe(OP_FMA, &r10, r20, EXP_H3210, BR[30][0][1],EXP_H3210, c240, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
    exe(OP_FMA, &r11, r21, EXP_H3210, r1,          EXP_H3210, c242, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
    exe(OP_FMA, &r12, r22, EXP_H3210, r2,          EXP_H3210, c244, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
    exe(OP_FMA, &r13, r23, EXP_H3210, r3,          EXP_H3210, c246, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
    exe(OP_FMA, &r20, r10, EXP_H3210, r4,          EXP_H3210, c248, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
    exe(OP_FMA, &r21, r11, EXP_H3210, r5,          EXP_H3210, c250, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
    exe(OP_FMA, &r22, r12, EXP_H3210, r6,          EXP_H3210, c252, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
    exe(OP_FMA, &r23, r13, EXP_H3210, r7,          EXP_H3210, c254, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */

    exe(OP_FAD, &r10,  r20, EXP_H3210,  r21, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
    exe(OP_FAD, &r12,  r22, EXP_H3210,  r23, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
#if 1
    exe(OP_FAD, &AR[35][0],  r10, EXP_H3210,  r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
    mop(OP_STR, 3, &AR[35][0], yoo++, 0LL, MSK_D0, (Ull)yoo, 304, 0, 0, yop, 304);                            /* stage#35 */
#else
    exe(OP_FAD, &r20,        r10, EXP_H3210,  r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
    mop(OP_STR, 3, &r20,       yoo++, 0LL, MSK_D0, (Ull)yoo, 304, 0, 0, yop, 304);                            /* stage#34 */
#endif
  }
//EMAX5A end
//emax5_start((Ull*)emax5_conf_x1, (Ull*)emax5_lmmi_x1, (Ull*)emax5_regv_x1);
#endif
  return(0);
}
